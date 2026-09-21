"""Runner: the synchronous execution engine for workflows."""

from __future__ import annotations

import signal
import warnings
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from taskmaestro.context import ExecutionContext
from taskmaestro.dependencies import CollectionRef, OutputRef
from taskmaestro.exceptions import (
    JobStateError,
    MappedTaskExecutionError,
    TaskOutputTypeError,
    TaskTimeoutError,
)
from taskmaestro.hooks.base import BaseHook, Event
from taskmaestro.job import Job, JobStatus, TaskResult, TaskStatus
from taskmaestro.mapping import MappedOutput, TaskMap
from taskmaestro.task import Task, get_input_type, get_output_type


class _JobTimeoutError(TaskTimeoutError):
    """A job deadline must abort even when mapped items collect failures."""


class Runner:
    """Synchronous execution engine for workflows.

    Iterates tasks in topological order, assembling each task's input
    from the outputs of its upstream dependencies.
    """

    def __init__(self, hooks: list[BaseHook] | None = None) -> None:
        self.hooks: list[BaseHook] = hooks or []

    def run(
        self,
        job: Job[Any],
        ctx: ExecutionContext | None = None,
        timeout_seconds: float | None = None,
    ) -> Job[Any]:
        """Execute all tasks in topological order. Stops on first failure."""
        if job.status != JobStatus.PENDING:
            raise JobStateError(f"Cannot run job with status '{job.status}'; expected 'pending'")

        ctx = ctx or ExecutionContext()
        workflow = job.workflow

        self._emit(Event.JOB_START, job)
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()

        # Set up job-level timeout
        job_alarm_set = False
        if timeout_seconds is not None:
            job_alarm_set = self._set_alarm(timeout_seconds, "Job", job_timeout=True)

        outputs: dict[str, BaseModel] = {}
        job_config = job.job_configuration

        try:
            for task_name, task_cls in workflow.topological_order():
                task = task_cls()
                task.name = task_name  # instance-level override for named instances
                deps = workflow.get_dependencies(task_name)
                config_fields = workflow.get_config_fields(task_name)
                task_map = workflow.get_task_map(task_name)
                all_config_values = (
                    job_config.get_config_for_task(task_name)
                    if job_config and (config_fields or task_map is not None)
                    else {}
                )
                # Mapped tasks consume the map source themselves; every other
                # configured value is passed through to the input model as before.
                config_values = {
                    key: value
                    for key, value in all_config_values.items()
                    if task_map is None or key != task_map.over
                }

                # Assemble input based on dependency type. Mapped tasks build
                # one validated input per configured item below.
                if task_map is not None:
                    task_input: Any = None
                elif deps is None:
                    if config_values:
                        # Root task with config: build input from config values
                        input_type = get_input_type(task_cls)
                        task_input = input_type.model_validate(config_values)
                    else:
                        task_input = job.config
                elif isinstance(deps, str):
                    if config_values:
                        # Single dep with config: decompose upstream, merge with config
                        input_type = get_input_type(task_cls)
                        upstream_output = outputs[deps]
                        assert isinstance(upstream_output, BaseModel)
                        upstream_data = upstream_output.model_dump()
                        down_fields = input_type.model_fields
                        merged: dict[str, object] = {
                            k: v for k, v in upstream_data.items() if k in down_fields
                        }
                        merged.update(config_values)
                        task_input = input_type.model_validate(merged)
                    else:
                        task_input = outputs[deps]
                elif isinstance(deps, tuple):
                    upstream_name, field_name = deps
                    task_input = getattr(outputs[upstream_name], field_name)
                elif isinstance(deps, dict):
                    input_type = get_input_type(task_cls)
                    field_values: dict[str, object] = {}
                    for fname, upstream_ref in deps.items():
                        if isinstance(upstream_ref, CollectionRef):
                            field_values[fname] = self._resolve_collection(upstream_ref, outputs)
                        elif isinstance(upstream_ref, tuple):
                            up_name, up_field = upstream_ref
                            field_values[fname] = getattr(outputs[up_name], up_field)
                        else:
                            field_values[fname] = outputs[upstream_ref]
                    if config_values:
                        field_values.update(config_values)
                    task_input = input_type.model_validate(field_values)
                else:
                    task_input = job.config  # pragma: no cover

                task_started = datetime.now()
                self._emit(Event.TASK_START, job, task)

                # Set up per-task timeout. Mapped tasks apply it per item.
                task_alarm_set = False
                if task_map is None and task.timeout_seconds is not None:
                    task_alarm_set = self._set_alarm(task.timeout_seconds, task.name)

                try:
                    if task_map is not None:
                        output = self._run_mapped_task(
                            job,
                            task_cls,
                            task,
                            task_map,
                            deps,
                            config_values,
                            all_config_values,
                            outputs,
                            ctx,
                        )
                    else:
                        output = task.run(task_input, ctx)

                        # Validate output matches declared type
                        expected_output_type = get_output_type(task_cls)
                        if not isinstance(output, expected_output_type):
                            raise TaskOutputTypeError(
                                f"Task '{task.name}' returned {type(output).__name__}, "
                                f"expected {expected_output_type.__name__}"
                            )

                    duration = (datetime.now() - task_started).total_seconds()
                    outputs[task.name] = output
                    job.task_results.append(
                        TaskResult(
                            task_name=task.name,
                            status=TaskStatus.COMPLETED,
                            output=output,
                            started_at=task_started,
                            duration_seconds=duration,
                        )
                    )
                    self._emit(Event.TASK_COMPLETE, job, task, output)
                except Exception as exc:
                    duration = (datetime.now() - task_started).total_seconds()
                    job.status = JobStatus.FAILED
                    job.error = str(exc)
                    job.failed_task = task.name
                    job.completed_at = datetime.now()
                    job.task_results.append(
                        TaskResult(
                            task_name=task.name,
                            status=TaskStatus.FAILED,
                            output=None,
                            started_at=task_started,
                            duration_seconds=duration,
                            error=str(exc),
                        )
                    )
                    self._emit(Event.TASK_FAIL, job, task, exc)
                    self._emit(Event.JOB_FAIL, job)
                    return job
                finally:
                    if task_alarm_set:
                        signal.alarm(0)
        finally:
            if job_alarm_set:
                signal.alarm(0)

        job.status = JobStatus.COMPLETED
        job.result = outputs[workflow.result_task_name]
        job.completed_at = datetime.now()
        self._emit(Event.JOB_COMPLETE, job)
        return job

    def _run_mapped_task(
        self,
        job: Job[Any],
        task_cls: type[Task[Any, Any]],
        parent_task: Task[Any, Any],
        task_map: TaskMap,
        deps: Any,
        config_values: dict[str, Any],
        all_config_values: dict[str, Any],
        outputs: dict[str, BaseModel],
        ctx: ExecutionContext,
    ) -> BaseModel:
        """Run all configured items for one mapped workflow node."""
        source = all_config_values[task_map.over]
        assert isinstance(source, Mapping)  # validated when the Job was created
        shared_values = self._mapped_shared_values(deps, config_values, outputs)
        expected_output_type = get_output_type(task_cls)
        collected: dict[str, BaseModel] = {}
        errors: dict[str, Exception] = {}
        item_results = job.mapped_item_results.setdefault(parent_task.name, [])

        for key, value in source.items():
            assert isinstance(key, str)  # validated when the Job was created
            item_task = task_cls()
            item_task.name = parent_task.name
            item_input_values = dict(shared_values)
            item_input_values[task_map.key_as] = key
            item_input_values[task_map.value_as] = value
            item_ctx = ctx.child(task_name=parent_task.name, item_key=key)
            item_started = datetime.now()
            self._emit(Event.MAP_ITEM_START, job, item_task, key)
            alarm_set = False
            if item_task.timeout_seconds is not None:
                alarm_set = self._set_alarm(
                    item_task.timeout_seconds, f"{parent_task.name}[{key}]"
                )
            try:
                input_type = get_input_type(task_cls)
                item_input = input_type.model_validate(item_input_values)
                output = item_task.run(item_input, item_ctx)
                if not isinstance(output, expected_output_type):
                    raise TaskOutputTypeError(
                        f"Task '{parent_task.name}[{key}]' returned "
                        f"{type(output).__name__}, expected {expected_output_type.__name__}"
                    )
                collected[key] = output
                item_results.append(
                    TaskResult(
                        task_name=f"{parent_task.name}[{key}]",
                        status=TaskStatus.COMPLETED,
                        output=output,
                        started_at=item_started,
                        duration_seconds=(datetime.now() - item_started).total_seconds(),
                    )
                )
                self._emit(Event.MAP_ITEM_COMPLETE, job, item_task, key, output)
            except Exception as exc:
                errors[key] = exc
                item_results.append(
                    TaskResult(
                        task_name=f"{parent_task.name}[{key}]",
                        status=TaskStatus.FAILED,
                        output=None,
                        started_at=item_started,
                        duration_seconds=(datetime.now() - item_started).total_seconds(),
                        error=str(exc),
                    )
                )
                self._emit(Event.MAP_ITEM_FAIL, job, item_task, key, exc)
                if isinstance(exc, _JobTimeoutError):
                    raise
                if task_map.error_mode == "fail_fast":
                    raise MappedTaskExecutionError(parent_task.name, errors) from exc
            finally:
                if alarm_set:
                    signal.alarm(0)

        if errors:
            raise MappedTaskExecutionError(parent_task.name, errors)
        mapped_output_type = MappedOutput[expected_output_type]  # type: ignore[valid-type]
        return mapped_output_type(root=collected)

    def _mapped_shared_values(
        self,
        deps: Any,
        config_values: dict[str, Any],
        outputs: dict[str, BaseModel],
    ) -> dict[str, object]:
        """Resolve fields shared by every invocation of a mapped task."""
        values: dict[str, object] = {}
        if isinstance(deps, dict):
            for field_name, ref in deps.items():
                if isinstance(ref, CollectionRef):
                    values[field_name] = self._resolve_collection(ref, outputs)
                elif isinstance(ref, tuple):
                    upstream_name, output_field = ref
                    values[field_name] = getattr(outputs[upstream_name], output_field)
                else:
                    values[field_name] = outputs[ref]
        values.update(config_values)
        return values

    @staticmethod
    def _resolve_output_ref(
        ref: OutputRef,
        outputs: dict[str, BaseModel],
    ) -> object:
        """Resolve one task output or output field from completed outputs."""
        output = outputs[ref.task_name]
        if ref.output_field is None:
            return output
        return getattr(output, ref.output_field)

    def _resolve_collection(
        self,
        collection: CollectionRef,
        outputs: dict[str, BaseModel],
    ) -> object:
        """Resolve a collection while preserving its declaration order."""
        if collection.kind == "positional":
            return [
                self._resolve_output_ref(ref, outputs) for ref in collection.positional_members
            ]
        return {
            key: self._resolve_output_ref(ref, outputs) for key, ref in collection.keyed_members
        }

    def _set_alarm(self, seconds: float, label: str, *, job_timeout: bool = False) -> bool:
        """Set a signal.alarm for timeout. Returns True if alarm was set."""
        try:

            def _handler(signum: int, frame: Any) -> None:
                error_type = _JobTimeoutError if job_timeout else TaskTimeoutError
                raise error_type(f"{label} timed out after {seconds}s")

            signal.signal(signal.SIGALRM, _handler)
            signal.alarm(int(seconds) if seconds >= 1 else 1)
            return True
        except (AttributeError, OSError):
            warnings.warn(
                f"signal.alarm not available on this platform; "
                f"timeout for {label} will not be enforced",
                stacklevel=2,
            )
            return False

    def _emit(self, event: Event, *args: object) -> None:
        """Dispatch event to all hooks, swallowing any hook errors."""
        for hook in self.hooks:
            handler = getattr(hook, f"on_{event}", None)
            if handler is not None:
                try:
                    handler(*args)
                except Exception:
                    warnings.warn(
                        f"Hook {type(hook).__name__} raised during {event}",
                        stacklevel=2,
                    )
