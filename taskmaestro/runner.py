"""Runner: the synchronous execution engine for workflows."""

from __future__ import annotations

import signal
import warnings
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from taskmaestro.context import ExecutionContext
from taskmaestro.exceptions import (
    JobStateError,
    TaskOutputTypeError,
    TaskTimeoutError,
)
from taskmaestro.hooks.base import BaseHook, Event
from taskmaestro.job import Job, JobStatus, TaskResult, TaskStatus
from taskmaestro.task import get_input_type, get_output_type


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
            job_alarm_set = self._set_alarm(timeout_seconds, "Job")

        outputs: dict[str, BaseModel] = {}
        job_config = job.job_configuration

        try:
            for task_name, task_cls in workflow.topological_order():
                task = task_cls()
                task.name = task_name  # instance-level override for named instances
                deps = workflow.get_dependencies(task_name)
                config_fields = workflow.get_config_fields(task_name)
                config_values = (
                    job_config.get_config_for_task(task_name)
                    if job_config and config_fields
                    else {}
                )

                # Assemble input based on dependency type
                if deps is None:
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
                        upstream_data = outputs[deps].model_dump()
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
                        if isinstance(upstream_ref, tuple):
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

                # Set up per-task timeout
                task_alarm_set = False
                if task.timeout_seconds is not None:
                    task_alarm_set = self._set_alarm(task.timeout_seconds, task.name)

                try:
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

    def _set_alarm(self, seconds: float, label: str) -> bool:
        """Set a signal.alarm for timeout. Returns True if alarm was set."""
        try:

            def _handler(signum: int, frame: Any) -> None:
                raise TaskTimeoutError(f"{label} timed out after {seconds}s")

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
