"""Timing hook that records wall-clock durations for jobs and tasks."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel

from taskekrabbe.hooks.base import BaseHook
from taskekrabbe.job import Job
from taskekrabbe.task import Task


class TimingHook(BaseHook):
    """Records wall-clock duration per task and total job time via time.monotonic()."""

    def __init__(self) -> None:
        self.job_duration: float | None = None
        self.task_timings: dict[str, float] = {}
        self._job_start: float | None = None
        self._task_starts: dict[str, float] = {}

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
