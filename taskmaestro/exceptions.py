"""Exception hierarchy for the workflow runner library."""

from typing import Any


class WorkflowRunnerError(Exception):
    """Base exception for all workflow runner errors."""


class WorkflowDefinitionError(WorkflowRunnerError):
    """Raised at workflow construction time for invalid definitions."""


class CycleDetectedError(WorkflowDefinitionError):
    """Dependency graph contains a cycle."""


class IncompleteInputError(WorkflowDefinitionError):
    """Fan-in input has required fields not mapped to any upstream task."""


class JobStateError(WorkflowRunnerError):
    """Job not in expected state (e.g., attempting to re-run a completed job)."""


class TaskExecutionError(WorkflowRunnerError):
    """Raised during task execution."""


class MappedTaskExecutionError(TaskExecutionError):
    """One or more invocations of a mapped task failed."""

    def __init__(self, task_name: str, errors: dict[str, Exception]) -> None:
        self.task_name = task_name
        self.errors = errors
        details = "; ".join(f"{key}: {error}" for key, error in errors.items())
        super().__init__(f"Mapped task '{task_name}' failed: {details}")


class TaskOutputTypeError(TaskExecutionError):
    """Task returned an output whose type doesn't match the declared output type."""


class TaskTimeoutError(TaskExecutionError):
    """Raised when a task exceeds its timeout_seconds."""


class WorkflowTaskError(TaskExecutionError):
    """An inner workflow wrapped by ``workflow_task`` failed.

    Carries the completed inner :class:`~taskmaestro.job.Job` so callers can
    inspect ``inner_job.task_results``, ``inner_job.failed_task`` and the
    per-item results of mapped tasks.  The original exception raised by the
    failing inner task is attached as ``__cause__`` when it is available.
    """

    def __init__(self, workflow_name: str, inner_job: Any) -> None:
        self.workflow_name = workflow_name
        self.inner_job = inner_job
        super().__init__(
            f"Inner workflow '{workflow_name}' failed at task "
            f"'{inner_job.failed_task}': {inner_job.error}"
        )


class ConfigLoadError(WorkflowRunnerError):
    """Raised when YAML config loading fails (parse errors, import failures, validation)."""


class PluginLoadError(WorkflowRunnerError):
    """Raised when an installed task or workflow entry point is invalid."""
