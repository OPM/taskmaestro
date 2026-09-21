"""Hook protocol and base implementation for lifecycle events."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pydantic import BaseModel

    from taskmaestro.job import Job
    from taskmaestro.task import Task


class Event(StrEnum):
    """Lifecycle events emitted by the runner."""

    JOB_START = "job_start"
    JOB_COMPLETE = "job_complete"
    JOB_FAIL = "job_fail"
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    TASK_FAIL = "task_fail"
    MAP_ITEM_START = "map_item_start"
    MAP_ITEM_COMPLETE = "map_item_complete"
    MAP_ITEM_FAIL = "map_item_fail"


@runtime_checkable
class Hook(Protocol):
    """Protocol for lifecycle event hooks."""

    def on_job_start(self, job: Job[Any]) -> None: ...
    def on_job_complete(self, job: Job[Any]) -> None: ...
    def on_job_fail(self, job: Job[Any]) -> None: ...
    def on_task_start(self, job: Job[Any], task: Task[Any, Any]) -> None: ...
    def on_task_complete(self, job: Job[Any], task: Task[Any, Any], output: BaseModel) -> None: ...
    def on_task_fail(self, job: Job[Any], task: Task[Any, Any], error: Exception) -> None: ...
    def on_map_item_start(self, job: Job[Any], task: Task[Any, Any], key: str) -> None: ...
    def on_map_item_complete(
        self, job: Job[Any], task: Task[Any, Any], key: str, output: BaseModel
    ) -> None: ...
    def on_map_item_fail(
        self, job: Job[Any], task: Task[Any, Any], key: str, error: Exception
    ) -> None: ...


class BaseHook:
    """Base hook with no-op defaults for all events."""

    def on_job_start(self, job: Job[Any]) -> None:
        pass

    def on_job_complete(self, job: Job[Any]) -> None:
        pass

    def on_job_fail(self, job: Job[Any]) -> None:
        pass

    def on_task_start(self, job: Job[Any], task: Task[Any, Any]) -> None:
        pass

    def on_task_complete(self, job: Job[Any], task: Task[Any, Any], output: BaseModel) -> None:
        pass

    def on_task_fail(self, job: Job[Any], task: Task[Any, Any], error: Exception) -> None:
        pass

    def on_map_item_start(self, job: Job[Any], task: Task[Any, Any], key: str) -> None:
        pass

    def on_map_item_complete(
        self, job: Job[Any], task: Task[Any, Any], key: str, output: BaseModel
    ) -> None:
        pass

    def on_map_item_fail(
        self, job: Job[Any], task: Task[Any, Any], key: str, error: Exception
    ) -> None:
        pass
