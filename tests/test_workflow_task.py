"""Tests for the workflow_task factory function."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from taskmaestro import (
    ExecutionContext,
    Job,
    JobConfiguration,
    JobStatus,
    Runner,
    Task,
    Workflow,
    WorkflowDefinitionError,
    workflow_task,
)
from taskmaestro.hooks.base import BaseHook
from taskmaestro.task import get_input_type, get_output_type

# --- Models ---


class InnerInput(BaseModel):
    value: int


class InnerMid(BaseModel):
    value: int


class InnerOutput(BaseModel):
    result: int


class StringResult(BaseModel):
    text: str


# --- Tasks ---


class InnerAdd(Task[InnerInput, InnerMid]):
    name = "inner_add"

    def run(self, input: InnerInput, ctx: ExecutionContext) -> InnerMid:
        return InnerMid(value=input.value + 1)


class InnerDouble(Task[InnerMid, InnerOutput]):
    name = "inner_double"

    def run(self, input: InnerMid, ctx: ExecutionContext) -> InnerOutput:
        return InnerOutput(result=input.value * 2)


class InnerFailing(Task[InnerInput, InnerOutput]):
    name = "inner_failing"

    def run(self, input: InnerInput, ctx: ExecutionContext) -> InnerOutput:
        raise ValueError("inner task broke")


class UpstreamTask(Task[InnerInput, InnerInput]):
    name = "upstream"

    def run(self, input: InnerInput, ctx: ExecutionContext) -> InnerInput:
        return InnerInput(value=input.value + 10)


class DownstreamTask(Task[InnerOutput, StringResult]):
    name = "downstream"

    def run(self, input: InnerOutput, ctx: ExecutionContext) -> StringResult:
        return StringResult(text=str(input.result))


class ServiceReader(Task[InnerInput, InnerOutput]):
    name = "service_reader"

    def run(self, input: InnerInput, ctx: ExecutionContext) -> InnerOutput:
        multiplier = ctx.resolve("multiplier")
        return InnerOutput(result=input.value * multiplier)


# --- DAG inner workflow tasks ---


class BranchA(Task[InnerInput, InnerMid]):
    name = "branch_a"

    def run(self, input: InnerInput, ctx: ExecutionContext) -> InnerMid:
        return InnerMid(value=input.value + 1)


class BranchB(Task[InnerInput, InnerMid]):
    name = "branch_b"

    def run(self, input: InnerInput, ctx: ExecutionContext) -> InnerMid:
        return InnerMid(value=input.value * 2)


class FanInInput(BaseModel):
    a: InnerMid
    b: InnerMid


class MergeTask(Task[FanInInput, InnerOutput]):
    name = "merge"

    def run(self, input: FanInInput, ctx: ExecutionContext) -> InnerOutput:
        return InnerOutput(result=input.a.value + input.b.value)


# --- Config-fields tasks ---


class ConfigInput(BaseModel):
    value: int
    label: str


class ConfigTask(Task[ConfigInput, InnerOutput]):
    name = "config_task"

    def run(self, input: ConfigInput, ctx: ExecutionContext) -> InnerOutput:
        return InnerOutput(result=input.value + len(input.label))


# --- Recording hook ---


class RecordingHook(BaseHook):
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


# ============================================================
# TestBasicWorkflowTask
# ============================================================


class TestBasicWorkflowTask:
    def test_type_derivation(self) -> None:
        inner_wf = Workflow("inner", tasks=[InnerAdd, InnerDouble])
        WrappedTask = workflow_task(inner_wf)

        assert get_input_type(WrappedTask) is InnerInput
        assert get_output_type(WrappedTask) is InnerOutput

    def test_default_name(self) -> None:
        inner_wf = Workflow("my_pipeline", tasks=[InnerAdd, InnerDouble])
        WrappedTask = workflow_task(inner_wf)

        assert WrappedTask.name == "my_pipeline"

    def test_custom_name(self) -> None:
        inner_wf = Workflow("inner", tasks=[InnerAdd, InnerDouble])
        WrappedTask = workflow_task(inner_wf, name="custom_sub")

        assert WrappedTask.name == "custom_sub"

    def test_class_name(self) -> None:
        inner_wf = Workflow("inner", tasks=[InnerAdd, InnerDouble])
        WrappedTask = workflow_task(inner_wf, name="sub_pipe")

        assert WrappedTask.__name__ == "WorkflowTask_sub_pipe"
        assert WrappedTask.__qualname__ == "WorkflowTask_sub_pipe"

    def test_execution(self) -> None:
        inner_wf = Workflow("inner", tasks=[InnerAdd, InnerDouble])
        WrappedTask = workflow_task(inner_wf)

        ctx = ExecutionContext()

        # Run the wrapper task directly
        task_instance = WrappedTask()
        result = task_instance.run(InnerInput(value=5), ctx)
        # (5 + 1) * 2 = 12
        assert result.result == 12


# ============================================================
# TestWorkflowTaskInOuterWorkflow
# ============================================================


class TestWorkflowTaskInOuterWorkflow:
    def test_linear_chain(self) -> None:
        """upstream -> workflow_task -> downstream in a linear outer workflow."""
        inner_wf = Workflow("inner", tasks=[InnerAdd, InnerDouble])
        SubTask = workflow_task(inner_wf, name="sub_pipeline")

        outer_wf = (
            Workflow.builder("outer")
            .add_task(UpstreamTask)
            .add_task(SubTask, depends_on=UpstreamTask)
            .add_task(DownstreamTask, depends_on=SubTask)
            .build()
        )

        job = Job(outer_wf, InnerInput(value=3))
        result = Runner().run(job, ctx=ExecutionContext())

        assert result.status == JobStatus.COMPLETED
        # upstream: 3+10=13, inner_add: 13+1=14, inner_double: 14*2=28
        assert result.result.text == "28"  # type: ignore[union-attr]

    def test_dag_inner_workflow_multiple_roots_rejected(self) -> None:
        """Inner DAG with multiple roots is rejected by workflow_task."""
        inner_wf = (
            Workflow.builder("inner_dag")
            .add_task(BranchA)
            .add_task(BranchB)
            .add_task(
                MergeTask,
                depends_on={"a": BranchA, "b": BranchB},
            )
            .build()
        )
        with pytest.raises(WorkflowDefinitionError, match="multiple root tasks"):
            workflow_task(inner_wf)

    def test_single_root_dag(self) -> None:
        """Inner DAG with a single root (linear start, then fan-out/in)."""
        inner_wf = (
            Workflow.builder("inner_single_root")
            .add_task(InnerAdd)
            .add_task(InnerDouble, depends_on=InnerAdd)
            .build()
        )
        SubTask = workflow_task(inner_wf, name="dag_sub")

        outer_wf = (
            Workflow.builder("outer")
            .add_task(UpstreamTask)
            .add_task(SubTask, depends_on=UpstreamTask)
            .add_task(DownstreamTask, depends_on=SubTask)
            .build()
        )
        job = Job(outer_wf, InnerInput(value=2))
        result = Runner().run(job, ctx=ExecutionContext())
        assert result.status == JobStatus.COMPLETED
        # upstream: 2+10=12, inner_add: 12+1=13, inner_double: 13*2=26
        assert result.result.text == "26"  # type: ignore[union-attr]

    def test_type_validation_in_outer(self) -> None:
        """Outer workflow type validation catches mismatched types."""
        inner_wf = Workflow("inner", tasks=[InnerAdd, InnerDouble])
        SubTask = workflow_task(inner_wf, name="sub")

        # SubTask expects InnerInput, but DownstreamTask outputs StringResult
        with pytest.raises(WorkflowDefinitionError, match="Type mismatch"):
            (
                Workflow.builder("bad_outer")
                .add_task(DownstreamTask)
                .add_task(SubTask, depends_on=DownstreamTask)
                .build()
            )


# ============================================================
# TestErrorPropagation
# ============================================================


class TestErrorPropagation:
    def test_inner_failure_surfaces(self) -> None:
        """Inner workflow failure becomes outer task failure with inner details."""
        inner_wf = Workflow("failing_inner", tasks=[InnerFailing])
        SubTask = workflow_task(inner_wf, name="fail_sub")

        outer_wf = Workflow.builder("outer").add_task(SubTask).build()
        job = Job(outer_wf, InnerInput(value=1))
        result = Runner().run(job, ctx=ExecutionContext())

        assert result.status == JobStatus.FAILED
        assert result.failed_task == "fail_sub"
        assert "inner_failing" in result.error  # type: ignore[operator]
        assert "inner task broke" in result.error  # type: ignore[operator]


# ============================================================
# TestContextSharing
# ============================================================


class TestContextSharing:
    def test_shared_context(self) -> None:
        """Inner tasks can access services from the shared ExecutionContext."""
        inner_wf = Workflow("ctx_inner", tasks=[ServiceReader])
        SubTask = workflow_task(inner_wf, name="ctx_sub")

        outer_wf = Workflow.builder("outer").add_task(SubTask).build()
        ctx = ExecutionContext()
        ctx.register("multiplier", 7)

        job = Job(outer_wf, InnerInput(value=3))
        result = Runner().run(job, ctx=ctx)

        assert result.status == JobStatus.COMPLETED
        assert result.result.result == 21  # type: ignore[union-attr]


# ============================================================
# TestOpaqueBoundary
# ============================================================


class TestOpaqueBoundary:
    def test_outer_sees_only_wrapper_events(self) -> None:
        """Outer RecordingHook sees only wrapper task events, not inner tasks."""
        inner_wf = Workflow("inner", tasks=[InnerAdd, InnerDouble])
        SubTask = workflow_task(inner_wf, name="sub")

        outer_wf = (
            Workflow.builder("outer")
            .add_task(UpstreamTask)
            .add_task(SubTask, depends_on=UpstreamTask)
            .build()
        )

        hook = RecordingHook()
        job = Job(outer_wf, InnerInput(value=1))
        Runner(hooks=[hook]).run(job, ctx=ExecutionContext())

        # Outer runner should see: job_start, task_start:upstream,
        # task_complete:upstream, task_start:sub, task_complete:sub,
        # job_complete
        task_events = [e for e in hook.events if e.startswith("task_")]
        assert task_events == [
            "task_start:upstream",
            "task_complete:upstream",
            "task_start:sub",
            "task_complete:sub",
        ]
        # No inner task names visible
        assert not any("inner_add" in e for e in hook.events)
        assert not any("inner_double" in e for e in hook.events)

    def test_single_task_result(self) -> None:
        """Outer job has a single TaskResult for the wrapper task."""
        inner_wf = Workflow("inner", tasks=[InnerAdd, InnerDouble])
        SubTask = workflow_task(inner_wf, name="sub")

        outer_wf = Workflow.builder("outer").add_task(SubTask).build()
        job = Job(outer_wf, InnerInput(value=5))
        result = Runner().run(job, ctx=ExecutionContext())

        assert result.status == JobStatus.COMPLETED
        assert len(result.task_results) == 1
        assert result.task_results[0].task_name == "sub"


# ============================================================
# TestValidation
# ============================================================


class TestValidation:
    def test_multiple_roots_rejected(self) -> None:
        """Inner workflow with multiple root tasks (no config_fields) is rejected."""
        inner_wf = (
            Workflow.builder("multi_root")
            .add_task(BranchA)
            .add_task(BranchB)
            .add_task(MergeTask, depends_on={"a": BranchA, "b": BranchB})
            .build()
        )
        with pytest.raises(WorkflowDefinitionError, match="multiple root tasks"):
            workflow_task(inner_wf)

    def test_no_valid_roots_rejected(self) -> None:
        """Inner workflow where all roots have config_fields is rejected."""
        inner_wf = (
            Workflow.builder("no_roots")
            .add_task(ConfigTask, config_fields=["value", "label"])
            .build()
        )
        with pytest.raises(WorkflowDefinitionError, match="no root tasks"):
            workflow_task(inner_wf)

    def test_root_with_config_fields_excluded(self) -> None:
        """Root tasks with config_fields are excluded from root detection."""
        # ConfigTask has config_fields (excluded), InnerAdd is the only valid root
        inner_wf = (
            Workflow.builder("mixed_roots", result_task=InnerDouble)
            .add_task(ConfigTask, config_fields=["value", "label"])
            .add_task(InnerAdd)
            .add_task(InnerDouble, depends_on=InnerAdd)
            .build()
        )
        # Should succeed: only InnerAdd is a valid root
        SubTask = workflow_task(
            inner_wf,
            name="mixed_sub",
            job_configuration=JobConfiguration({"config_task": {"value": 1, "label": "x"}}),
        )
        assert get_input_type(SubTask) is InnerInput


# ============================================================
# TestJobConfiguration
# ============================================================


class TestJobConfiguration:
    def test_inner_config_fields(self) -> None:
        """Inner tasks with config_fields work when job_configuration is provided."""
        inner_wf = (
            Workflow.builder("cfg_inner")
            .add_task(InnerAdd)
            .add_task(
                ConfigTask,
                depends_on=InnerAdd,
                config_fields=["label"],
            )
            .build()
        )

        jc = JobConfiguration({"config_task": {"label": "hello"}})
        SubTask = workflow_task(inner_wf, name="cfg_sub", job_configuration=jc)

        outer_wf = Workflow.builder("outer").add_task(SubTask).build()
        job = Job(outer_wf, InnerInput(value=3))
        result = Runner().run(job, ctx=ExecutionContext())

        assert result.status == JobStatus.COMPLETED
        # InnerAdd: 3+1=4 -> value=4, label="hello" -> 4 + len("hello") = 9
        assert result.result.result == 9  # type: ignore[union-attr]
