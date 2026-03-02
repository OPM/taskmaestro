"""Tests for the Runner execution engine."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from taskekrabbe import (
    ExecutionContext,
    Job,
    JobStatus,
    Runner,
    Task,
    Workflow,
)
from taskekrabbe.exceptions import JobStateError
from taskekrabbe.job import TaskStatus
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
    StringOutput,
    WrongOutputTask,
)


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


class TestFieldRoutingExecution:
    """Tests for output field routing in Runner."""

    def test_single_field_ref_execution(self, ctx: ExecutionContext) -> None:
        class MultiOut(BaseModel):
            stats: NumberOutput
            other: NumberOutput

        class Producer(Task[NumberInput, MultiOut]):
            name = "producer"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> MultiOut:
                return MultiOut(
                    stats=NumberOutput(value=input.value * 10),
                    other=NumberOutput(value=input.value * 100),
                )

        wf = (
            Workflow.builder("field_exec")
            .add_task(Producer)
            .add_task(Double, depends_on=(Producer, "stats"))
            .build()
        )
        job = Job(workflow=wf, config=NumberInput(value=3))
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.COMPLETED
        assert result.result is not None
        # Producer.stats = 30, Double doubles it = 60
        assert result.result.value == 60  # type: ignore[attr-defined]

    def test_dual_fan_out_field_routing(self, ctx: ExecutionContext) -> None:
        class DualOut(BaseModel):
            num: NumberOutput
            text: StringOutput

        class DualProducer(Task[NumberInput, DualOut]):
            name = "dual_producer"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> DualOut:
                return DualOut(
                    num=NumberOutput(value=input.value),
                    text=StringOutput(text=str(input.value)),
                )

        wf = (
            Workflow.builder("dual_fan", result_task=Stringify)
            .add_task(DualProducer)
            .add_task(Double, depends_on=(DualProducer, "num"))
            .add_task(Stringify, depends_on=Double)
            .build()
        )
        job = Job(workflow=wf, config=NumberInput(value=5))
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.COMPLETED
        assert result.result.text == "10"  # type: ignore[union-attr]

    def test_mixed_fan_in_field_routing(self, ctx: ExecutionContext) -> None:
        class MixedOut(BaseModel):
            a: NumberOutput
            b: NumberOutput

        class MixedProducer(Task[NumberInput, MixedOut]):
            name = "mixed_producer"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> MixedOut:
                return MixedOut(
                    a=NumberOutput(value=input.value),
                    b=NumberOutput(value=input.value + 100),
                )

        wf = (
            Workflow.builder("mixed")
            .add_task(MixedProducer)
            .add_task(AddOneB)
            .add_task(
                FanInTask,
                depends_on={
                    "a": (MixedProducer, "a"),
                    "b": AddOneB,
                },
            )
            .build()
        )
        job = Job(workflow=wf, config=NumberInput(value=5))
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.COMPLETED
        # Producer.a = 5, AddOneB = 6, total = 11
        assert result.result.total == 11  # type: ignore[union-attr]


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


class TestAlarmUnavailable:
    def test_alarm_unavailable_warns(self, ctx: ExecutionContext) -> None:
        """When signal.alarm is unavailable, a warning is issued and execution proceeds."""
        from unittest.mock import patch

        wf = Workflow(name="test", tasks=[AddOne])
        job = Job(workflow=wf, config=NumberInput(value=1))
        with (
            patch("taskekrabbe.runner.signal.signal", side_effect=AttributeError),
            pytest.warns(UserWarning, match="signal.alarm not available"),
        ):
            result = Runner().run(job, ctx=ctx, timeout_seconds=60)
        assert result.status == JobStatus.COMPLETED


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


class TestNamedTaskInstanceExecution:
    """Tests for executing workflows with named task instances."""

    def test_same_class_two_names_executes(self, ctx: ExecutionContext) -> None:
        """Workflow with same class under two names executes correctly."""
        wf = (
            Workflow.builder(name="named_exec")
            .add_task(AddOne, name="first_add")
            .add_task(AddOne, name="second_add")
            .add_task(
                FanInTask,
                depends_on={"a": "first_add", "b": "second_add"},
            )
            .build()
        )
        job = Job(workflow=wf, config=NumberInput(value=5))
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.COMPLETED
        assert result.result is not None
        assert result.result.total == 12  # type: ignore[attr-defined]

    def test_task_results_use_instance_names(self, ctx: ExecutionContext) -> None:
        """task_results contain the correct instance names, not class defaults."""
        wf = (
            Workflow.builder(name="named_results")
            .add_task(AddOne, name="step_a")
            .add_task(Double, depends_on="step_a")
            .build()
        )
        job = Job(workflow=wf, config=NumberInput(value=3))
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.COMPLETED
        names = [tr.task_name for tr in result.task_results]
        assert names == ["step_a", "double"]

    def test_string_dep_fan_in_execution(self, ctx: ExecutionContext) -> None:
        """Fan-in with string dependencies executes correctly."""
        wf = (
            Workflow.builder(name="str_fan_in")
            .add_task(AddOne, name="branch_1")
            .add_task(AddOne, name="branch_2")
            .add_task(
                FanInTask,
                depends_on={"a": "branch_1", "b": "branch_2"},
            )
            .build()
        )
        job = Job(workflow=wf, config=NumberInput(value=10))
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.COMPLETED
        # Both branches add 1 to 10 = 11, total = 22
        assert result.result.total == 22  # type: ignore[union-attr]
