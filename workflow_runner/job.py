"""Job: a workflow bound to a specific config, ready to execute."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel

from workflow_runner.exceptions import WorkflowDefinitionError
from workflow_runner.task import get_input_type
from workflow_runner.workflow import Workflow


class JobStatus(StrEnum):
    """Status of a job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(StrEnum):
    """Status of an individual task execution."""

    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskResult:
    """Record of a single task's execution within a job."""

    task_name: str
    status: TaskStatus
    output: BaseModel | None
    started_at: datetime
    duration_seconds: float
    error: str | None = None


C = TypeVar("C", bound=BaseModel)


class Job(Generic[C]):
    """A workflow bound to a specific config, ready to execute.

    Generic over C so that job.config retains its concrete type.
    Validates that the config type matches the input type of all root tasks.
    """

    def __init__(self, workflow: Workflow, config: C) -> None:
        self.workflow = workflow
        self.config: C = config
        self.status: JobStatus = JobStatus.PENDING
        self.result: BaseModel | None = None
        self.error: str | None = None
        self.failed_task: str | None = None
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.task_results: list[TaskResult] = []

        self._validate_root_task_inputs(config)

    def _validate_root_task_inputs(self, config: C) -> None:
        """Validate that config type matches the input type of all root tasks."""
        for task_name, deps in self.workflow._dependencies.items():
            if deps is None:
                task_cls = self.workflow._tasks[task_name]
                expected_input = get_input_type(task_cls)
                if not isinstance(config, expected_input):
                    raise WorkflowDefinitionError(
                        f"Root task '{task_name}' expects input type "
                        f"{expected_input.__name__} but got {type(config).__name__}"
                    )
