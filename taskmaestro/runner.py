"""Runner: the synchronous execution engine for workflows."""

from __future__ import annotations

import signal
import time
import warnings
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
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


class HookError(UserWarning):
    """Warning category used when a lifecycle hook raises.

    Subclasses :class:`UserWarning` so existing ``pytest.warns(UserWarning)``
    and ``-W error::UserWarning`` configurations keep working, while allowing
    callers to filter hook failures specifically.
    """


@dataclass
class _Deadline:
    """Per-run timer state shared by the job and its tasks.

    There is only one ``SIGALRM`` per process, so the job deadline is kept as an
    absolute ``time.monotonic()`` timestamp and folded into every task or item
    alarm.  Whichever deadline is nearer wins, and the job deadline is
    re-checked before each unit of work so an inner alarm can never cancel it.
    """

    job_timeout: float | None = None
    job_deadline: float | None = None
    warned: bool = False
    previous_handler: Any = field(default=None, repr=False)
    handler_installed: bool = False

    def remaining(self) -> float | None:
        """Seconds left until the job deadline, or ``None`` if there is none."""
        if self.job_deadline is None:
            return None
        return self.job_deadline - time.monotonic()

    def check(self) -> None:
        """Raise :class:`_JobTimeoutError` if the job deadline has passed."""
        remaining = self.remaining()
        if remaining is not None and remaining <= 0:
            raise _JobTimeoutError(f"Job timed out after {self.job_timeout}s")


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

        # Job-level timeout is tracked as an absolute deadline and folded into
        # every task/item alarm; see _Deadline.
        deadline = _Deadline(job_timeout=timeout_seconds)
        if timeout_seconds is not None:
            deadline.job_deadline = time.monotonic() + timeout_seconds

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

                try:
                    # Arming happens inside the guarded block so that an expired
                    # job deadline or an unusable timer is recorded as a task
                    # failure rather than escaping with the job left RUNNING.
                    deadline.check()
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
                            deadline,
                        )
                    else:
                        self._arm(task.timeout_seconds, task.name, deadline)
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
                    job.exception = exc
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
                    self._disarm(deadline)
        finally:
            self._disarm(deadline)
            self._restore_handler(deadline)

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
        deadline: _Deadline,
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
            try:
                deadline.check()
                self._arm(item_task.timeout_seconds, f"{parent_task.name}[{key}]", deadline)
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
                self._disarm(deadline)

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

    def _arm(self, task_timeout: float | None, label: str, deadline: _Deadline) -> None:
        """Arm the timer for one unit of work.

        The nearer of the task's own timeout and the remaining job time wins.
        Raises :class:`_JobTimeoutError` immediately if the job deadline has
        already passed.
        """
        remaining = deadline.remaining()
        if remaining is not None and remaining <= 0:
            raise _JobTimeoutError(f"Job timed out after {deadline.job_timeout}s")

        if task_timeout is not None and (remaining is None or task_timeout <= remaining):
            self._set_alarm(task_timeout, label, deadline=deadline)
        elif remaining is not None:
            self._set_alarm(remaining, "Job", deadline=deadline, job_timeout=True)

    def _set_alarm(
        self,
        seconds: float,
        label: str,
        *,
        deadline: _Deadline | None = None,
        job_timeout: bool = False,
    ) -> bool:
        """Install a SIGALRM handler and start a one-shot timer.

        Uses ``signal.setitimer`` for sub-second precision, falling back to
        ``signal.alarm`` where unavailable.  Returns True if the timer was set.
        On platforms or threads where signals cannot be used, a single warning
        is issued per run and the timeout is not enforced.
        """
        if job_timeout and deadline is not None:
            message = f"Job timed out after {deadline.job_timeout}s"
        else:
            message = f"{label} timed out after {seconds}s"
        error_type: type[TaskTimeoutError] = _JobTimeoutError if job_timeout else TaskTimeoutError

        def _handler(signum: int, frame: Any) -> None:
            raise error_type(message)

        try:
            previous = signal.signal(signal.SIGALRM, _handler)
            if deadline is not None and not deadline.handler_installed:
                deadline.previous_handler = previous
                deadline.handler_installed = True
            setitimer = getattr(signal, "setitimer", None)
            if setitimer is not None:
                setitimer(signal.ITIMER_REAL, max(seconds, 1e-6))
            else:  # pragma: no cover - every SIGALRM platform has setitimer
                signal.alarm(max(1, int(seconds + 0.999999)))
            return True
        except (AttributeError, OSError, ValueError):
            # ValueError: signal.signal() called outside the main thread.
            if deadline is None or not deadline.warned:
                if deadline is not None:
                    deadline.warned = True
                warnings.warn(
                    f"signal.alarm not available on this platform or thread; "
                    f"timeout for {label} will not be enforced",
                    stacklevel=2,
                )
            return False

    @staticmethod
    def _disarm(deadline: _Deadline) -> None:
        """Cancel any pending timer without touching the handler."""
        if not deadline.handler_installed:
            return
        setitimer = getattr(signal, "setitimer", None)
        if setitimer is not None:
            setitimer(signal.ITIMER_REAL, 0)
        else:  # pragma: no cover
            signal.alarm(0)

    @staticmethod
    def _restore_handler(deadline: _Deadline) -> None:
        """Put back the SIGALRM handler that was installed before this run."""
        if not deadline.handler_installed:
            return
        with suppress(AttributeError, OSError, ValueError, TypeError):  # pragma: no cover
            signal.signal(signal.SIGALRM, deadline.previous_handler)
        deadline.handler_installed = False

    def _emit(self, event: Event, *args: object) -> None:
        """Dispatch event to all hooks, swallowing any hook errors.

        A failing hook must not abort the workflow, but its error should not
        vanish either: the warning carries the exception and the original
        traceback is attached via ``source`` for ``-W error`` / logging capture.
        """
        for hook in self.hooks:
            handler = getattr(hook, f"on_{event}", None)
            if handler is not None:
                try:
                    handler(*args)
                except Exception as exc:
                    warnings.warn(
                        f"Hook {type(hook).__name__} raised during {event}: {exc!r}",
                        HookError,
                        stacklevel=2,
                        source=exc,
                    )
