"""Timing hook that records wall-clock durations for jobs and tasks."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel

from taskmaestro.hooks.base import BaseHook
from taskmaestro.job import Job
from taskmaestro.task import Task


class TimingHook(BaseHook):
    """Records wall-clock duration per task and total job time via time.monotonic()."""

    def __init__(self) -> None:
        self.job_duration: float | None = None
        self.task_timings: dict[str, float] = {}
        self._job_start: float | None = None
        self.mapped_item_timings: dict[str, dict[str, float]] = {}
        self._task_starts: dict[str, float] = {}
        self._map_item_starts: dict[tuple[str, str], float] = {}

    def on_job_start(self, job: Job[Any]) -> None:
        self._job_start = time.monotonic()

    def on_job_complete(self, job: Job[Any]) -> None:
        if self._job_start is not None:
            self.job_duration = time.monotonic() - self._job_start

    def on_job_fail(self, job: Job[Any]) -> None:
        if self._job_start is not None:
            self.job_duration = time.monotonic() - self._job_start

    def on_task_start(self, job: Job[Any], task: Task[Any, Any]) -> None:
        self._task_starts[task.name] = time.monotonic()

    def on_task_complete(self, job: Job[Any], task: Task[Any, Any], output: BaseModel) -> None:
        start = self._task_starts.get(task.name)
        if start is not None:
            self.task_timings[task.name] = time.monotonic() - start

    def on_task_fail(self, job: Job[Any], task: Task[Any, Any], error: Exception) -> None:
        start = self._task_starts.get(task.name)
        if start is not None:
            self.task_timings[task.name] = time.monotonic() - start

    def on_map_item_start(self, job: Job[Any], task: Task[Any, Any], key: str) -> None:
        self._map_item_starts[(task.name, key)] = time.monotonic()

    def on_map_item_complete(
        self, job: Job[Any], task: Task[Any, Any], key: str, output: BaseModel
    ) -> None:
        self._record_map_item(task.name, key)

    def on_map_item_fail(
        self, job: Job[Any], task: Task[Any, Any], key: str, error: Exception
    ) -> None:
        self._record_map_item(task.name, key)

    def _record_map_item(self, task_name: str, key: str) -> None:
        start = self._map_item_starts.get((task_name, key))
        if start is not None:
            self.mapped_item_timings.setdefault(task_name, {})[key] = time.monotonic() - start
