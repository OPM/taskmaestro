"""Runner: the synchronous execution engine for workflows."""

from __future__ import annotations

import signal
import threading
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


@dataclass(eq=False)
class _Timer:
    """One armed timeout: an absolute expiry and the error to raise when it fires."""

    expiry: float
    error_type: type[TaskTimeoutError]
    message: str
    fired: bool = False


class _AlarmScheduler:
    """Multiplex the process's single ``ITIMER_REAL``/``SIGALRM`` across timers.

    Nested runs (e.g. a ``workflow_task`` whose inner workflow has its own
    timeouts) share one process timer.  Every armed timeout is registered here
    and the OS timer always tracks the nearest pending expiry, so an inner run
    arming or cancelling its own timers can neither extend nor cancel the
    timeouts of an enclosing run.  The SIGALRM handler that was installed before
    the first timer was added is restored when the last timer is removed.
    """

    def __init__(self) -> None:
        self._timers: list[_Timer] = []
        self._previous_handler: Any = None
        self._installed = False

    def add(self, timer: _Timer) -> None:
        """Register *timer* and re-arm the OS timer.

        Raises ``ValueError`` off the main thread (signals are delivered to the
        main thread only) and ``AttributeError``/``OSError`` where ``SIGALRM``
        is unavailable.
        """
        if threading.current_thread() is not threading.main_thread():
            raise ValueError("signal timers only work in the main thread")
        if not self._installed:
            self._previous_handler = signal.signal(signal.SIGALRM, self._handle)
            self._installed = True
        self._timers.append(timer)
        self._reschedule()

    def remove(self, timer: _Timer) -> None:
        """Unregister *timer*; restore the previous handler once none remain."""
        self._timers.remove(timer)
        if self._timers:
            self._reschedule()
            return
        _start_timer(0)
        with suppress(AttributeError, OSError, ValueError, TypeError):  # pragma: no cover
            signal.signal(signal.SIGALRM, self._previous_handler)
        self._previous_handler = None
        self._installed = False

    def _nearest_pending(self) -> _Timer | None:
        pending = [timer for timer in self._timers if not timer.fired]
        return min(pending, key=lambda timer: timer.expiry) if pending else None

    def _reschedule(self) -> None:
        nearest = self._nearest_pending()
        if nearest is None:
            _start_timer(0)
        else:
            _start_timer(max(nearest.expiry - time.monotonic(), 1e-6))

    def _handle(self, signum: int, frame: Any) -> None:
        nearest = self._nearest_pending()
        if nearest is None:
            return  # stale signal: every registered timer has already fired
        nearest.fired = True
        # The OS timer is one-shot; keep the remaining timers armed.
        self._reschedule()
        raise nearest.error_type(nearest.message)


def _start_timer(seconds: float) -> None:
    """Start (or with ``0`` cancel) the process's one-shot real-time timer."""
    setitimer = getattr(signal, "setitimer", None)
    if setitimer is not None:
        setitimer(signal.ITIMER_REAL, seconds)
    else:  # pragma: no cover - every SIGALRM platform has setitimer
        signal.alarm(max(1, int(seconds + 0.999999)) if seconds else 0)


_ALARMS = _AlarmScheduler()


@dataclass
class _Deadline:
    """Per-run timer state shared by the job and its tasks.

    The job deadline is kept as an absolute ``time.monotonic()`` timestamp and
    folded into every task or item alarm.  Whichever deadline is nearer wins,
    and the job deadline is re-checked before each unit of work.  ``timer`` is
    the unit of work's currently registered :class:`_Timer`, if any.
    """

    job_timeout: float | None = None
    job_deadline: float | None = None
    warned: bool = False
    timer: _Timer | None = field(default=None, repr=False)

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

                task_started = datetime.now()
                task: Task[Any, Any] | None = None
                try:
                    # Instantiation, input assembly and arming all happen inside
                    # the guarded block so that any failure is recorded as a task
                    # failure rather than escaping with the job left RUNNING.
                    task = task_cls()
                    task.name = task_name  # instance-level override for named instances
                    self._emit(Event.TASK_START, job, task)
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
                        task_input = self._build_task_input(
                            job, task_cls, deps, config_values, outputs
                        )
                        self._arm(task.timeout_seconds, task.name, deadline)
                        try:
                            output = task.run(task_input, ctx)
                        finally:
                            # Stop the clock as soon as the task returns: output
                            # checks and hooks must not be interrupted by (or
                            # swallow) this task's timeout.
                            self._disarm(deadline)

                        # Validate output matches declared type
                        expected_output_type = get_output_type(task_cls)
                        if not isinstance(output, expected_output_type):
                            raise TaskOutputTypeError(
                                f"Task '{task.name}' returned {type(output).__name__}, "
                                f"expected {expected_output_type.__name__}"
                            )

                    duration = (datetime.now() - task_started).total_seconds()
                    outputs[task_name] = output
                    job.task_results.append(
                        TaskResult(
                            task_name=task_name,
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
                    job.failed_task = task_name
                    job.completed_at = datetime.now()
                    job.task_results.append(
                        TaskResult(
                            task_name=task_name,
                            status=TaskStatus.FAILED,
                            output=None,
                            started_at=task_started,
                            duration_seconds=duration,
                            error=str(exc),
                        )
                    )
                    # Without an instance (the constructor raised) there is no
                    # task to report; the job-level failure is still emitted.
                    if task is not None:
                        self._emit(Event.TASK_FAIL, job, task, exc)
                    self._emit(Event.JOB_FAIL, job)
                    return job
        finally:
            # Safety net; each unit of work already disarms right after running.
            self._disarm(deadline)

        job.status = JobStatus.COMPLETED
        job.result = outputs[workflow.result_task_name]
        job.completed_at = datetime.now()
        self._emit(Event.JOB_COMPLETE, job)
        return job

    @staticmethod
    def _build_task_input(
        job: Job[Any],
        task_cls: type[Task[Any, Any]],
        deps: Any,
        config_values: dict[str, Any],
        outputs: dict[str, BaseModel],
    ) -> Any:
        """Assemble one (unmapped) task's input from upstream outputs and config."""
        if deps is None:
            if config_values:
                # Root task with config: build input from config values
                return get_input_type(task_cls).model_validate(config_values)
            return job.config
        if isinstance(deps, str):
            if not config_values:
                return outputs[deps]
            # Single dep with config: decompose upstream, merge with config
            input_type = get_input_type(task_cls)
            upstream_data = outputs[deps].model_dump()
            down_fields = input_type.model_fields
            merged: dict[str, object] = {
                k: v for k, v in upstream_data.items() if k in down_fields
            }
            merged.update(config_values)
            return input_type.model_validate(merged)
        if isinstance(deps, tuple):
            upstream_name, field_name = deps
            return getattr(outputs[upstream_name], field_name)
        # Fan-in: one named input field per upstream reference.
        field_values = Runner._resolve_named_dependencies(deps, outputs)
        field_values.update(config_values)
        return get_input_type(task_cls).model_validate(field_values)

    @staticmethod
    def _resolve_named_dependencies(
        deps: Mapping[str, Any],
        outputs: dict[str, BaseModel],
    ) -> dict[str, object]:
        """Resolve a fan-in dependency mapping to input field values."""
        values: dict[str, object] = {}
        for field_name, ref in deps.items():
            if isinstance(ref, CollectionRef):
                values[field_name] = Runner._resolve_collection(ref, outputs)
            elif isinstance(ref, tuple):
                upstream_name, output_field = ref
                values[field_name] = getattr(outputs[upstream_name], output_field)
            else:
                values[field_name] = outputs[ref]
        return values

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
                input_type = get_input_type(task_cls)
                item_input = input_type.model_validate(item_input_values)
                self._arm(item_task.timeout_seconds, f"{parent_task.name}[{key}]", deadline)
                try:
                    output = item_task.run(item_input, item_ctx)
                finally:
                    self._disarm(deadline)
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
        values: dict[str, object] = (
            self._resolve_named_dependencies(deps, outputs) if isinstance(deps, dict) else {}
        )
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

    @staticmethod
    def _resolve_collection(
        collection: CollectionRef,
        outputs: dict[str, BaseModel],
    ) -> object:
        """Resolve a collection while preserving its declaration order."""
        if collection.kind == "positional":
            return [
                Runner._resolve_output_ref(ref, outputs) for ref in collection.positional_members
            ]
        return {
            key: Runner._resolve_output_ref(ref, outputs) for key, ref in collection.keyed_members
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
        """Register a one-shot timeout with the process-wide alarm scheduler.

        Uses ``signal.setitimer`` for sub-second precision.  Returns True if the
        timer was set.  On platforms or threads where signals cannot be used, a
        single warning is issued per run and the timeout is not enforced.
        """
        if job_timeout and deadline is not None:
            message = f"Job timed out after {deadline.job_timeout}s"
        else:
            message = f"{label} timed out after {seconds}s"
        error_type: type[TaskTimeoutError] = _JobTimeoutError if job_timeout else TaskTimeoutError
        timer = _Timer(time.monotonic() + seconds, error_type, message)

        try:
            _ALARMS.add(timer)
        except (AttributeError, OSError, ValueError):
            # ValueError: signals can only be used from the main thread.
            if deadline is None or not deadline.warned:
                if deadline is not None:
                    deadline.warned = True
                warnings.warn(
                    f"signal.alarm not available on this platform or thread; "
                    f"timeout for {label} will not be enforced",
                    stacklevel=2,
                )
            return False
        if deadline is not None:
            deadline.timer = timer
        return True

    @staticmethod
    def _disarm(deadline: _Deadline) -> None:
        """Cancel this run's pending timer, leaving enclosing runs' timers armed."""
        if deadline.timer is None:
            return
        _ALARMS.remove(deadline.timer)
        deadline.timer = None

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
