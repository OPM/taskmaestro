"""Exception hierarchy for the workflow runner library."""

from typing import Any


class WorkflowRunnerError(Exception):
    """Base exception for all workflow runner errors."""


class WorkflowDefinitionError(WorkflowRunnerError):
    """Raised at workflow construction time for invalid definitions."""

    def __init__(
        self,
        message: str,
        *,
        task_name: str | None = None,
        fields: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.task_name = task_name
        self.fields = fields or []


class CycleDetectedError(WorkflowDefinitionError):
    """Dependency graph contains a cycle."""


class IncompleteInputError(WorkflowDefinitionError):
    """Fan-in input has required fields not mapped to any upstream task."""


class ContextValueMissingError(WorkflowDefinitionError):
    """A ``ContextFlag`` key is not registered on the execution context.

    Raised by ``Runner.run`` before any task starts. The message only names
    context keys and task names.
    """


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


class ResultTaskSkippedError(TaskExecutionError):
    """The workflow's result task was skipped by conditional execution."""


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


class ScriptError(TaskExecutionError):
    """A script or command run by ``taskmaestro.python``/``taskmaestro.command`` failed.

    ``returncode`` is ``None`` when the process was not started or did not finish
    (missing script, unserializable inputs, timeout). ``stdout``/``stderr`` hold the
    captured output, if any.
    """

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class ConfigLoadError(WorkflowRunnerError):
    """Raised when YAML config loading fails (parse errors, import failures, validation)."""


class WorkflowLookupError(ConfigLoadError):
    """A registered workflow name is unknown or its entry point cannot be loaded.

    The message names the registered workflows and contains no input values.
    """


class PluginLoadError(WorkflowRunnerError):
    """Raised when an installed task or workflow entry point is invalid."""


class DefinitionError(WorkflowRunnerError):
    """Raised when a workflow definition cannot be loaded from or saved to YAML.

    ``task`` and ``field`` name the offending task and input field, if known.
    """

    def __init__(self, message: str, *, task: str | None = None, field: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.task = task
        self.field = field

    def to_json(self) -> dict[str, str | None]:
        return {"message": self.message, "task": self.task, "field": self.field}
