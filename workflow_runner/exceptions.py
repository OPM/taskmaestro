"""Exception hierarchy for the workflow runner library."""


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


class TaskOutputTypeError(TaskExecutionError):
    """Task returned an output whose type doesn't match the declared output type."""


class TaskTimeoutError(TaskExecutionError):
    """Raised when a task exceeds its timeout_seconds."""


class ConfigLoadError(WorkflowRunnerError):
    """Raised when YAML config loading fails (parse errors, import failures, validation)."""
