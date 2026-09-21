"""Tests for the Runner execution engine."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from taskmaestro import (
    EmptyConfig,
    ExecutionContext,
    Job,
    JobConfiguration,
    JobStatus,
    Runner,
    Task,
    Workflow,
)
from taskmaestro.exceptions import JobStateError
from taskmaestro.job import TaskStatus
from tests.conftest import (
    AddOne,
    AddOneB,
    ConfigOnlyTask,
    Double,
    FailingTask,
    FanInTask,
    FanInWithConfigTask,
    MergeTask,
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

    @pytest.mark.skipif(
        not hasattr(__import__("signal"), "SIGALRM"),
        reason="signal.SIGALRM not available on this platform",
    )
    def test_job_timeout_survives_task_with_own_timeout(self, ctx: ExecutionContext) -> None:
        """A task's own alarm must not cancel the job deadline for later tasks."""
        import time

        class QuickWithTimeout(Task[NumberInput, NumberOutput]):
            name = "quick"
            timeout_seconds = 30

            def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
                return NumberOutput(value=input.value)

        class SlowNoTimeout(Task[NumberOutput, NumberOutput]):
            name = "slow_no_timeout"

            def run(self, input: NumberOutput, ctx: ExecutionContext) -> NumberOutput:
                time.sleep(5)
                return input

        wf = Workflow(name="test", tasks=[QuickWithTimeout, SlowNoTimeout])
        job = Job(workflow=wf, config=NumberInput(value=1))
        start = time.monotonic()
        result = Runner().run(job, ctx=ctx, timeout_seconds=0.5)
        assert time.monotonic() - start < 3
        assert result.status == JobStatus.FAILED
        assert result.failed_task == "slow_no_timeout"
        assert "Job timed out after 0.5s" in (result.error or "")

    @pytest.mark.skipif(
        not hasattr(__import__("signal"), "SIGALRM"),
        reason="signal.SIGALRM not available on this platform",
    )
    def test_job_timeout_survives_nested_workflow_task(self, ctx: ExecutionContext) -> None:
        """An inner workflow's runner must not cancel the outer job deadline."""
        import time

        class InnerWithTimeout(Task[NumberInput, NumberOutput]):
            name = "inner_with_timeout"
            timeout_seconds = 30

            def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
                return NumberOutput(value=input.value)

        class SlowNoTimeout(Task[NumberOutput, NumberOutput]):
            name = "slow_no_timeout"

            def run(self, input: NumberOutput, ctx: ExecutionContext) -> NumberOutput:
                time.sleep(5)
                return input

        inner = Workflow(name="inner", tasks=[InnerWithTimeout]).as_task(name="inner")
        wf = Workflow(name="outer", tasks=[inner, SlowNoTimeout])
        job = Job(workflow=wf, config=NumberInput(value=1))
        start = time.monotonic()
        result = Runner().run(job, ctx=ctx, timeout_seconds=0.5)
        assert time.monotonic() - start < 3
        assert result.status == JobStatus.FAILED
        assert result.failed_task == "slow_no_timeout"

    @pytest.mark.skipif(
        not hasattr(__import__("signal"), "SIGALRM"),
        reason="signal.SIGALRM not available on this platform",
    )
    def test_expired_job_deadline_fails_next_task_immediately(self, ctx: ExecutionContext) -> None:
        """If the deadline passes during a task, the following task is not started."""
        import time

        ran: list[str] = []

        class Sleeper(Task[NumberInput, NumberOutput]):
            name = "sleeper"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
                ran.append(self.name)
                time.sleep(0.3)
                return NumberOutput(value=input.value)

        class Never(Task[NumberOutput, NumberOutput]):
            name = "never"

            def run(self, input: NumberOutput, ctx: ExecutionContext) -> NumberOutput:
                ran.append(self.name)
                return input

        wf = Workflow(name="test", tasks=[Sleeper, Never])
        job = Job(workflow=wf, config=NumberInput(value=1))
        # Deadline expires while Sleeper is running; Sleeper itself is only
        # interrupted by the alarm, but Never must not run at all.
        result = Runner().run(job, ctx=ctx, timeout_seconds=0.2)
        assert result.status == JobStatus.FAILED
        assert ran == ["sleeper"]

    @pytest.mark.skipif(
        not hasattr(__import__("signal"), "SIGALRM"),
        reason="signal.SIGALRM not available on this platform",
    )
    def test_deadline_already_expired_before_next_task(self, ctx: ExecutionContext) -> None:
        """A task that swallows the alarm and overruns the deadline still stops the job."""
        import time
        from contextlib import suppress

        from taskmaestro.exceptions import TaskTimeoutError

        ran: list[str] = []

        class SwallowsAlarm(Task[NumberInput, NumberOutput]):
            name = "swallows_alarm"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
                ran.append(self.name)
                # A misbehaving task that ignores the job deadline.
                with suppress(TaskTimeoutError):
                    time.sleep(0.6)
                return NumberOutput(value=input.value)

        class Never(Task[NumberOutput, NumberOutput]):
            name = "never"

            def run(self, input: NumberOutput, ctx: ExecutionContext) -> NumberOutput:
                ran.append(self.name)
                return input

        wf = Workflow(name="test", tasks=[SwallowsAlarm, Never])
        job = Job(workflow=wf, config=NumberInput(value=1))
        result = Runner().run(job, ctx=ctx, timeout_seconds=0.2)

        assert ran == ["swallows_alarm"]
        assert result.status == JobStatus.FAILED
        assert result.failed_task == "never"
        assert result.error == "Job timed out after 0.2s"
        assert [r.status for r in result.task_results] == [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
        ]

    def test_arm_raises_when_deadline_already_passed(self) -> None:
        import time

        from taskmaestro.exceptions import TaskTimeoutError
        from taskmaestro.runner import _Deadline

        deadline = _Deadline(job_timeout=1.0, job_deadline=time.monotonic() - 1)
        with pytest.raises(TaskTimeoutError, match=r"Job timed out after 1\.0s"):
            Runner()._arm(None, "task", deadline)

    @pytest.mark.skipif(
        not hasattr(__import__("signal"), "SIGALRM"),
        reason="signal.SIGALRM not available on this platform",
    )
    def test_sub_second_timeout_is_not_truncated(self, ctx: ExecutionContext) -> None:
        """timeout_seconds=1.9 must allow a 1.4s task to finish (was truncated to 1s)."""
        import time

        class MidTask(Task[NumberInput, NumberOutput]):
            name = "mid"
            timeout_seconds = 1.9

            def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
                time.sleep(1.4)
                return NumberOutput(value=input.value)

        wf = Workflow(name="test", tasks=[MidTask])
        job = Job(workflow=wf, config=NumberInput(value=1))
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.COMPLETED

    @pytest.mark.skipif(
        not hasattr(__import__("signal"), "SIGALRM"),
        reason="signal.SIGALRM not available on this platform",
    )
    def test_previous_sigalrm_handler_restored(self, ctx: ExecutionContext) -> None:
        import signal

        def sentinel(signum: int, frame: object) -> None:  # pragma: no cover
            pass

        previous = signal.signal(signal.SIGALRM, sentinel)
        try:
            wf = Workflow(name="test", tasks=[AddOne])
            job = Job(workflow=wf, config=NumberInput(value=1))
            Runner().run(job, ctx=ctx, timeout_seconds=60)
            assert signal.getsignal(signal.SIGALRM) is sentinel
        finally:
            signal.signal(signal.SIGALRM, previous)


class TestAlarmUnavailable:
    def test_alarm_unavailable_warns(self, ctx: ExecutionContext) -> None:
        """When signal.alarm is unavailable, a warning is issued and execution proceeds."""
        from unittest.mock import patch

        wf = Workflow(name="test", tasks=[AddOne])
        job = Job(workflow=wf, config=NumberInput(value=1))
        with (
            patch("taskmaestro.runner.signal.signal", side_effect=AttributeError),
            pytest.warns(UserWarning, match="signal.alarm not available"),
        ):
            result = Runner().run(job, ctx=ctx, timeout_seconds=60)
        assert result.status == JobStatus.COMPLETED

    @pytest.mark.skipif(
        not hasattr(__import__("signal"), "SIGALRM"),
        reason="signal.SIGALRM not available on this platform",
    )
    def test_timeouts_in_non_main_thread_warn_and_continue(self) -> None:
        """signal.signal() raises ValueError off the main thread; the job must still finish."""
        import threading
        import warnings

        outcome: dict[str, object] = {}

        def worker() -> None:
            wf = Workflow(name="test", tasks=[AddOne, Double])
            job = Job(workflow=wf, config=NumberInput(value=1))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    result = Runner().run(job, timeout_seconds=60)
                except Exception as exc:  # pragma: no cover - the bug under test
                    outcome["exc"] = exc
                    return
            outcome["status"] = result.status
            outcome["warnings"] = [str(w.message) for w in caught]

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert "exc" not in outcome, outcome.get("exc")
        assert outcome["status"] == JobStatus.COMPLETED
        messages = outcome["warnings"]
        assert isinstance(messages, list)
        assert len(messages) == 1  # warned once per run, not per task
        assert "signal.alarm not available" in messages[0]

    def test_arming_failure_marks_task_failed_not_running(self, ctx: ExecutionContext) -> None:
        """An unexpected error while arming the timer is recorded as a task failure."""
        from unittest.mock import patch

        wf = Workflow(name="test", tasks=[SlowTask])
        job = Job(workflow=wf, config=NumberInput(value=1))
        with patch.object(Runner, "_arm", side_effect=RuntimeError("boom")):
            result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.FAILED
        assert result.failed_task == "slow_task"
        assert result.error == "boom"


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


class TestConfigFieldsExecution:
    """Tests for config_fields merging during execution."""

    def test_root_task_from_config(self, ctx: ExecutionContext) -> None:
        """Root task input is built entirely from JobConfiguration."""
        wf = (
            Workflow.builder("root_cfg")
            .add_task(ConfigOnlyTask, config_fields=["path", "count"])
            .build()
        )
        jc = JobConfiguration({"config_only_task": {"path": "/data", "count": 7}})
        job = Job(wf, EmptyConfig(), job_configuration=jc)
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.COMPLETED
        assert result.result.summary == "/datax7"  # type: ignore[union-attr]

    def test_single_dep_merge(self, ctx: ExecutionContext) -> None:
        """Single dep + config: upstream output fields merged with config values."""
        wf = (
            Workflow.builder("merge")
            .add_task(AddOne)
            .add_task(
                MergeTask,
                depends_on=AddOne,
                config_fields=["label"],
            )
            .build()
        )
        jc = JobConfiguration({"merge_task": {"label": "result"}})
        job = Job(wf, NumberInput(value=5), job_configuration=jc)
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.COMPLETED
        # AddOne: 5+1=6, MergeTask: "result:6"
        assert result.result.result == "result:6"  # type: ignore[union-attr]

    def test_fan_in_merge(self, ctx: ExecutionContext) -> None:
        """Fan-in + config: mapped fields merged with config values."""
        wf = (
            Workflow.builder("fan_cfg")
            .add_task(AddOne)
            .add_task(
                FanInWithConfigTask,
                depends_on={"a": AddOne},
                config_fields=["extra"],
            )
            .build()
        )
        jc = JobConfiguration({"fan_in_with_config": {"extra": "hello"}})
        job = Job(wf, NumberInput(value=3), job_configuration=jc)
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.COMPLETED
        # AddOne: 3+1=4, FanInWithConfig: "hello:4"
        assert result.result.combined == "hello:4"  # type: ignore[union-attr]

    def test_extra_config_values_reach_input_model(self, ctx: ExecutionContext) -> None:
        """Configured values are passed through to the input model even when they
        are not listed in config_fields, so the model decides how to treat them."""

        class StrictInput(BaseModel):
            model_config = ConfigDict(extra="forbid")
            path: str

        class StrictTask(Task[StrictInput, NumberOutput]):
            name = "strict_task"

            def run(self, input: StrictInput, ctx: ExecutionContext) -> NumberOutput:
                return NumberOutput(value=len(input.path))

        wf = Workflow.builder("strict").add_task(StrictTask, config_fields=["path"]).build()
        jc = JobConfiguration({"strict_task": {"path": "/data", "unexpected": 1}})
        job = Job(wf, EmptyConfig(), job_configuration=jc)
        with pytest.raises(ValidationError, match="unexpected"):
            Runner().run(job, ctx=ctx)

    def test_backward_compat_no_config(self, ctx: ExecutionContext) -> None:
        """Workflow without config_fields runs normally."""
        wf = Workflow(name="compat", tasks=[AddOne, Double])
        job = Job(wf, NumberInput(value=5))
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.COMPLETED
        assert result.result.value == 12  # type: ignore[union-attr]

    def test_config_only_root_chain(self, ctx: ExecutionContext) -> None:
        """Root from config feeds into a downstream task."""
        wf = (
            Workflow.builder("chain")
            .add_task(ConfigOnlyTask, config_fields=["path", "count"])
            .build()
        )
        jc = JobConfiguration({"config_only_task": {"path": "/test", "count": 2}})
        job = Job(wf, EmptyConfig(), job_configuration=jc)
        result = Runner().run(job, ctx=ctx)
        assert result.status == JobStatus.COMPLETED
        assert result.result.summary == "/testx2"  # type: ignore[union-attr]
