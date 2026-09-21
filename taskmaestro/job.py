"""Job: a workflow bound to a specific config, ready to execute."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from taskmaestro.exceptions import WorkflowDefinitionError
from taskmaestro.task import get_input_type
from taskmaestro.workflow import Workflow


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
        self.mapped_item_results: dict[str, list[TaskResult]] = {}

        self._validate_root_task_inputs(config)
        self._validate_task_maps()

    def _validate_root_task_inputs(self, config: C) -> None:
        """Validate that config type matches the input type of all root tasks."""
        for task_name, deps in self.workflow._dependencies.items():
            if deps is None:
                # Configured and mapped roots do not consume job.config directly.
                config_fields = self.workflow.get_config_fields(task_name)
                if config_fields or self.workflow.is_mapped_task(task_name):
                    continue
                task_cls = self.workflow._tasks[task_name]
                expected_input = get_input_type(task_cls)
                if not isinstance(config, expected_input):
                    raise WorkflowDefinitionError(
                        f"Root task '{task_name}' expects input type "
                        f"{expected_input.__name__} but got {type(config).__name__}"
                    )

    def _validate_task_maps(self) -> None:
        """Validate configured map sources and their key/value types."""
        for task_name, task_cls in self.workflow._tasks.items():
            task_map = self.workflow.get_task_map(task_name)
            if task_map is None:
                continue
            if self.job_configuration is None:
                raise WorkflowDefinitionError(
                    f"Mapped task '{task_name}' requires JobConfiguration"
                )
            task_config = self.job_configuration.get_config_for_task(task_name)
            if task_map.over not in task_config:
                raise WorkflowDefinitionError(
                    f"Mapped task '{task_name}' requires configuration field '{task_map.over}'"
                )
            source = task_config[task_map.over]
            if not isinstance(source, Mapping):
                raise WorkflowDefinitionError(
                    f"Configuration field '{task_name}.{task_map.over}' must be a mapping"
                )

            input_type = get_input_type(task_cls)
            input_fields = input_type.model_fields
            # A tuple adapter carries the input model's config while allowing
            # nested BaseModels to retain their own config. Keep field metadata too.
            key_type = input_fields[task_map.key_as].rebuild_annotation()
            value_type = input_fields[task_map.value_as].rebuild_annotation()
            item_adapter: TypeAdapter[Any] = TypeAdapter(
                tuple[key_type, value_type],  # type: ignore[valid-type]
                config=input_type.model_config,
            )
            for key, value in source.items():
                if not isinstance(key, str):
                    raise WorkflowDefinitionError(
                        f"Mapping keys for task '{task_name}' must be strings"
                    )
                try:
                    item_adapter.validate_python((key, value))
                except ValidationError as exc:
                    raise WorkflowDefinitionError(
                        f"Invalid mapping item '{key}' for task '{task_name}': {exc}"
                    ) from exc
