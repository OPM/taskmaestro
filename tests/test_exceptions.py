"""Tests for the exception hierarchy."""

from workflow_runner.exceptions import (
    CycleDetectedError,
    IncompleteInputError,
    JobStateError,
    TaskExecutionError,
    TaskOutputTypeError,
    TaskTimeoutError,
    WorkflowDefinitionError,
    WorkflowRunnerError,
)


class TestExceptionHierarchy:
    def test_base_exception(self) -> None:
        assert issubclass(WorkflowRunnerError, Exception)

    def test_workflow_definition_error(self) -> None:
        assert issubclass(WorkflowDefinitionError, WorkflowRunnerError)

    def test_cycle_detected_error(self) -> None:
        assert issubclass(CycleDetectedError, WorkflowDefinitionError)

    def test_incomplete_input_error(self) -> None:
        assert issubclass(IncompleteInputError, WorkflowDefinitionError)

    def test_job_state_error(self) -> None:
        assert issubclass(JobStateError, WorkflowRunnerError)

    def test_task_execution_error(self) -> None:
        assert issubclass(TaskExecutionError, WorkflowRunnerError)

    def test_task_output_type_error(self) -> None:
        assert issubclass(TaskOutputTypeError, TaskExecutionError)

    def test_task_timeout_error(self) -> None:
        assert issubclass(TaskTimeoutError, TaskExecutionError)

    def test_exception_messages(self) -> None:
        exc = CycleDetectedError("cycle found")
        assert str(exc) == "cycle found"
        assert isinstance(exc, WorkflowRunnerError)
