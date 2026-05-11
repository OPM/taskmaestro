"""Logging hook that logs lifecycle events via Python logging."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from taskmaestro.hooks.base import BaseHook
from taskmaestro.job import Job
from taskmaestro.task import Task


class LoggingHook(BaseHook):
    """Logs all lifecycle events via Python's logging module."""

    def __init__(self, level: int = logging.INFO) -> None:
        self._level = level
        self._logger = logging.getLogger("taskmaestro.hooks.logging")

    def on_job_start(self, job: Job[Any]) -> None:
        self._logger.log(self._level, "Job started: workflow=%s", job.workflow.name)

    def on_job_complete(self, job: Job[Any]) -> None:
        self._logger.log(self._level, "Job completed: workflow=%s", job.workflow.name)

    def on_job_fail(self, job: Job[Any]) -> None:
        self._logger.log(
            self._level,
            "Job failed: workflow=%s, failed_task=%s, error=%s",
            job.workflow.name,
            job.failed_task,
            job.error,
        )

    def on_task_start(self, job: Job[Any], task: Task[Any, Any]) -> None:
        self._logger.log(self._level, "Task started: %s", task.name)

    def on_task_complete(self, job: Job[Any], task: Task[Any, Any], output: BaseModel) -> None:
        self._logger.log(self._level, "Task completed: %s", task.name)

    def on_task_fail(self, job: Job[Any], task: Task[Any, Any], error: Exception) -> None:
        self._logger.log(self._level, "Task failed: %s, error=%s", task.name, error)
