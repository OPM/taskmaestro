"""Tests for the Runner execution engine."""

from __future__ import annotations

import pytest

from tests.conftest import (
    AddOne,
    AddOneB,
    Double,
    FailingTask,
    FanInTask,
    NumberInput,
    NumberOutput,
    SlowTask,
    Stringify,
    WrongOutputTask,
)
from workflow_runner import (
    ExecutionContext,
    Job,
    JobStatus,
    Runner,
    Task,
    Workflow,
)
from workflow_runner.exceptions import JobStateError
from workflow_runner.job import TaskStatus


class TestLinearExecution:
    def test_happy_path(self, ctx: ExecutionContext, number_input: NumberInput) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double])
        job = Job(workflow=wf, config=number_input)
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.COMPLETED
        assert result.result is not None
        assert result.result.value == 12  # type: ignore[attr-defined]

    def test_three_task_chain(self, ctx: ExecutionContext) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double, Stringify])
        job = Job(workflow=wf, config=NumberInput(value=3))
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.COMPLETED
        assert result.result is not None
        assert result.result.text == "8"  # type: ignore[attr-defined]

    def test_task_results_populated(self, ctx: ExecutionContext) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double])
        job = Job(workflow=wf, config=NumberInput(value=1))
        result = Runner().run(job, ctx=ctx)
        assert len(result.task_results) == 2
        assert result.task_results[0].task_name == "add_one"
        assert result.task_results[0].status == TaskStatus.COMPLETED
        assert result.task_results[1].task_name == "double"
        assert result.task_results[1].status == TaskStatus.COMPLETED

    def test_timestamps_populated(self, ctx: ExecutionContext) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double])
        job = Job(workflow=wf, config=NumberInput(value=1))
        result = Runner().run(job, ctx=ctx)
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.completed_at >= result.started_at
        for tr in result.task_results:
            assert tr.started_at is not None
            assert tr.duration_seconds >= 0


class TestFailurePaths:
    def test_task_failure(self, ctx: ExecutionContext) -> None:
        wf = Workflow(name="test", tasks=[FailingTask])
        job = Job(workflow=wf, config=NumberInput(value=1))
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.FAILED
        assert result.failed_task == "failing_task"
        assert result.error is not None
        assert "intentionally" in result.error

    def test_output_type_mismatch(self, ctx: ExecutionContext) -> None:
        wf = Workflow(name="test", tasks=[WrongOutputTask])
        job = Job(workflow=wf, config=NumberInput(value=1))
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.FAILED
        assert result.failed_task == "wrong_output"
        assert "StringOutput" in (result.error or "")

    def test_rerun_guard(self, ctx: ExecutionContext) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double])
        job = Job(workflow=wf, config=NumberInput(value=1))
        runner = Runner()
        runner.run(job, ctx=ctx)
        with pytest.raises(JobStateError, match="Cannot run job"):
            runner.run(job, ctx=ctx)

    def test_failed_task_results(self, ctx: ExecutionContext) -> None:
        wf = Workflow(name="test", tasks=[FailingTask])
        job = Job(workflow=wf, config=NumberInput(value=1))
        Runner().run(job, ctx=ctx)
        assert len(job.task_results) == 1
        assert job.task_results[0].status == TaskStatus.FAILED
        assert job.task_results[0].error is not None


class TestDAGExecution:
    def test_fan_in_execution(self, ctx: ExecutionContext) -> None:
        wf = (
            Workflow.builder(name="fan_in")
            .add_task(AddOne)
            .add_task(AddOneB)
            .add_task(FanInTask, depends_on={"a": AddOne, "b": AddOneB})
            .build()
        )
        job = Job(workflow=wf, config=NumberInput(value=5))
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.COMPLETED
        assert result.result is not None
        assert result.result.total == 12  # type: ignore[attr-defined]

    def test_fan_out_execution(self, ctx: ExecutionContext) -> None:
        wf = (
            Workflow.builder(name="fan_out", result_task=Stringify)
            .add_task(AddOne)
            .add_task(Double, depends_on=AddOne)
            .add_task(Stringify, depends_on=AddOne)
            .build()
        )
        job = Job(workflow=wf, config=NumberInput(value=3))
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.COMPLETED
        assert result.result is not None
        assert result.result.text == "4"  # type: ignore[attr-defined]


class TestTimeouts:
    @pytest.mark.skipif(
        not hasattr(__import__("signal"), "SIGALRM"),
        reason="signal.SIGALRM not available on this platform",
    )
    def test_per_task_timeout(self, ctx: ExecutionContext) -> None:
        wf = Workflow(name="test", tasks=[SlowTask])
        job = Job(workflow=wf, config=NumberInput(value=1))
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.FAILED
        assert "timed out" in (result.error or "")

    @pytest.mark.skipif(
        not hasattr(__import__("signal"), "SIGALRM"),
        reason="signal.SIGALRM not available on this platform",
    )
    def test_per_job_timeout(self, ctx: ExecutionContext) -> None:
        class VerySlowTask(Task[NumberInput, NumberOutput]):
            name = "very_slow"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
                import time

                time.sleep(10)
                return NumberOutput(value=input.value)

        wf = Workflow(name="test", tasks=[VerySlowTask])
        job = Job(workflow=wf, config=NumberInput(value=1))
        result = Runner().run(job, ctx=ctx, timeout_seconds=1)
        assert result.status == JobStatus.FAILED
        assert "timed out" in (result.error or "")


class TestContextIntegration:
    def test_context_auto_created(self) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double])
        job = Job(workflow=wf, config=NumberInput(value=1))
        result = Runner().run(job)
        assert result.status == JobStatus.COMPLETED

    def test_service_registry_accessible(self) -> None:
        class ServiceTask(Task[NumberInput, NumberOutput]):
            name = "service_task"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
                multiplier = ctx.resolve("multiplier")
                return NumberOutput(value=input.value * multiplier)

        wf = Workflow(name="test", tasks=[ServiceTask])
        job = Job(workflow=wf, config=NumberInput(value=5))
        ctx = ExecutionContext()
        ctx.register("multiplier", 3)
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.COMPLETED
        assert result.result is not None
        assert result.result.value == 15  # type: ignore[attr-defined]
