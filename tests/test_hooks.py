"""Tests for the hook system."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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
from taskekrabbe.hooks import LoggingHook, ResultPersistenceHook, TimingHook
from taskekrabbe.hooks.base import BaseHook
from tests.conftest import (
    AddOne,
    Double,
    FailingTask,
    FanInTask,
    NumberInput,
)


class RecordingHook(BaseHook):
    """Records all lifecycle events for testing."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_job_start(self, job: Job[Any]) -> None:
        self.events.append("job_start")

    def on_job_complete(self, job: Job[Any]) -> None:
        self.events.append("job_complete")

    def on_job_fail(self, job: Job[Any]) -> None:
        self.events.append("job_fail")

    def on_task_start(self, job: Job[Any], task: Task[Any, Any]) -> None:
        self.events.append(f"task_start:{task.name}")

    def on_task_complete(self, job: Job[Any], task: Task[Any, Any], output: BaseModel) -> None:
        self.events.append(f"task_complete:{task.name}")

    def on_task_fail(self, job: Job[Any], task: Task[Any, Any], error: Exception) -> None:
        self.events.append(f"task_fail:{task.name}")


class TestRecordingHook:
    def test_success_event_sequence(self, ctx: ExecutionContext) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double])
        job = Job(workflow=wf, config=NumberInput(value=1))
        hook = RecordingHook()
        Runner(hooks=[hook]).run(job, ctx=ctx)
        assert hook.events == [
            "job_start",
            "task_start:add_one",
            "task_complete:add_one",
            "task_start:double",
            "task_complete:double",
            "job_complete",
        ]

    def test_failure_event_sequence(self, ctx: ExecutionContext) -> None:
        wf = Workflow(name="test", tasks=[FailingTask])
        job = Job(workflow=wf, config=NumberInput(value=1))
        hook = RecordingHook()
        Runner(hooks=[hook]).run(job, ctx=ctx)
        assert hook.events == [
            "job_start",
            "task_start:failing_task",
            "task_fail:failing_task",
            "job_fail",
        ]


class TestLoggingHook:
    def test_logs_events(self, ctx: ExecutionContext, caplog: pytest.LogCaptureFixture) -> None:
        wf = Workflow(name="test", tasks=[AddOne])
        job = Job(workflow=wf, config=NumberInput(value=1))
        hook = LoggingHook(level=logging.INFO)
        with caplog.at_level(logging.INFO, logger="taskekrabbe.hooks.logging"):
            Runner(hooks=[hook]).run(job, ctx=ctx)
        assert any("Job started" in msg for msg in caplog.messages)
        assert any("Task started: add_one" in msg for msg in caplog.messages)
        assert any("Task completed: add_one" in msg for msg in caplog.messages)
        assert any("Job completed" in msg for msg in caplog.messages)

    def test_logs_failure(self, ctx: ExecutionContext, caplog: pytest.LogCaptureFixture) -> None:
        wf = Workflow(name="test", tasks=[FailingTask])
        job = Job(workflow=wf, config=NumberInput(value=1))
        hook = LoggingHook(level=logging.INFO)
        with caplog.at_level(logging.INFO, logger="taskekrabbe.hooks.logging"):
            Runner(hooks=[hook]).run(job, ctx=ctx)
        assert any("Task failed" in msg for msg in caplog.messages)
        assert any("Job failed" in msg for msg in caplog.messages)


class TestTimingHook:
    def test_records_durations(self, ctx: ExecutionContext) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double])
        job = Job(workflow=wf, config=NumberInput(value=1))
        hook = TimingHook()
        Runner(hooks=[hook]).run(job, ctx=ctx)
        assert hook.job_duration is not None
        assert hook.job_duration >= 0
        assert "add_one" in hook.task_timings
        assert "double" in hook.task_timings
        assert all(t >= 0 for t in hook.task_timings.values())

    def test_records_failure_duration(self, ctx: ExecutionContext) -> None:
        wf = Workflow(name="test", tasks=[FailingTask])
        job = Job(workflow=wf, config=NumberInput(value=1))
        hook = TimingHook()
        Runner(hooks=[hook]).run(job, ctx=ctx)
        assert hook.job_duration is not None
        assert "failing_task" in hook.task_timings


class TestResultPersistenceHook:
    def test_writes_json_files(self, ctx: ExecutionContext, tmp_path: Path) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double])
        job = Job(workflow=wf, config=NumberInput(value=5))
        hook = ResultPersistenceHook(output_dir=tmp_path / "results")
        Runner(hooks=[hook]).run(job, ctx=ctx)

        add_one_path = tmp_path / "results" / "add_one.json"
        double_path = tmp_path / "results" / "double.json"
        assert add_one_path.exists()
        assert double_path.exists()

        add_one_data = json.loads(add_one_path.read_text())
        assert add_one_data["value"] == 6

        double_data = json.loads(double_path.read_text())
        assert double_data["value"] == 12


class TestHookErrorHandling:
    def test_hook_error_swallowed(self, ctx: ExecutionContext) -> None:
        class BrokenHook(BaseHook):
            def on_task_start(self, job: Job[Any], task: Task[Any, Any]) -> None:
                raise RuntimeError("Hook is broken")

        wf = Workflow(name="test", tasks=[AddOne])
        job = Job(workflow=wf, config=NumberInput(value=1))
        with pytest.warns(UserWarning, match="BrokenHook raised during task_start"):
            result = Runner(hooks=[BrokenHook()]).run(job, ctx=ctx)
        assert result.status == JobStatus.COMPLETED

    def test_multiple_hooks(self, ctx: ExecutionContext) -> None:
        wf = Workflow(name="test", tasks=[AddOne])
        job = Job(workflow=wf, config=NumberInput(value=1))
        hook1 = RecordingHook()
        hook2 = RecordingHook()
        Runner(hooks=[hook1, hook2]).run(job, ctx=ctx)
        assert hook1.events == hook2.events
        assert len(hook1.events) == 4  # job_start, task_start, task_complete, job_complete


class TestBaseHookNoOps:
    def test_base_hook_handles_failure_events(self, ctx: ExecutionContext) -> None:
        """BaseHook's on_job_fail and on_task_fail no-ops execute without error."""
        wf = Workflow(name="test", tasks=[FailingTask])
        job = Job(workflow=wf, config=NumberInput(value=1))
        result = Runner(hooks=[BaseHook()]).run(job, ctx=ctx)
        assert result.status == JobStatus.FAILED


class TestNamedInstanceHooks:
    """Tests that hooks see the registered instance name, not the class name."""

    def test_hooks_see_instance_names(self, ctx: ExecutionContext) -> None:
        wf = (
            Workflow.builder(name="named_hooks")
            .add_task(AddOne, name="step_alpha")
            .add_task(Double, depends_on="step_alpha")
            .build()
        )
        job = Job(workflow=wf, config=NumberInput(value=1))
        hook = RecordingHook()
        Runner(hooks=[hook]).run(job, ctx=ctx)
        assert hook.events == [
            "job_start",
            "task_start:step_alpha",
            "task_complete:step_alpha",
            "task_start:double",
            "task_complete:double",
            "job_complete",
        ]

    def test_hooks_see_both_named_instances(self, ctx: ExecutionContext) -> None:
        """When same class is used twice, hooks see both registered names."""
        wf = (
            Workflow.builder(name="dual_named")
            .add_task(AddOne, name="first")
            .add_task(AddOne, name="second")
            .add_task(
                FanInTask,
                depends_on={"a": "first", "b": "second"},
            )
            .build()
        )
        job = Job(workflow=wf, config=NumberInput(value=5))
        hook = RecordingHook()
        Runner(hooks=[hook]).run(job, ctx=ctx)
        assert "task_start:first" in hook.events
        assert "task_complete:first" in hook.events
        assert "task_start:second" in hook.events
        assert "task_complete:second" in hook.events
