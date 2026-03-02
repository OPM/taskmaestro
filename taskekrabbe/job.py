"""Job: a workflow bound to a specific config, ready to execute."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from taskekrabbe.exceptions import WorkflowDefinitionError
from taskekrabbe.task import get_input_type
from taskekrabbe.workflow import Workflow


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


class EmptyConfig(BaseModel):
    """Sentinel config for workflows where all root tasks use JobConfiguration."""


class JobConfiguration:
    """Per-task configuration values, mapping task names to config field dicts.

    Used to provide static configuration values (from YAML or code) that get
    merged with upstream outputs when constructing task inputs.
    """

    def __init__(self, config: dict[str, dict[str, Any]]) -> None:
        self._config = config

    def get_config_for_task(self, name: str) -> dict[str, Any]:
        """Return config values for a task, or empty dict if none."""
        return dict(self._config.get(name, {}))

    def configured_tasks(self) -> set[str]:
        """Return set of task names that have configuration."""
        return set(self._config.keys())

    def config_fields_for_task(self, name: str) -> set[str]:
        """Return set of field names configured for a task."""
        return set(self._config.get(name, {}).keys())


C = TypeVar("C", bound=BaseModel)


class Job(Generic[C]):
    """A workflow bound to a specific config, ready to execute.

    Generic over C so that job.config retains its concrete type.
    Validates that the config type matches the input type of all root tasks.
    """

    def __init__(
        self,
        workflow: Workflow,
        config: C,
        *,
        job_configuration: JobConfiguration | None = None,
    ) -> None:
        self.workflow = workflow
        self.config: C = config
        self.job_configuration = job_configuration
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
                # Skip validation for root tasks that have config_fields
                config_fields = self.workflow.get_config_fields(task_name)
                if config_fields:
                    continue
                task_cls = self.workflow._tasks[task_name]
                expected_input = get_input_type(task_cls)
                if not isinstance(config, expected_input):
                    raise WorkflowDefinitionError(
                        f"Root task '{task_name}' expects input type "
                        f"{expected_input.__name__} but got {type(config).__name__}"
                    )
