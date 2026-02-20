"""Tests for Workflow and WorkflowBuilder."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from tests.conftest import (
    AddOne,
    AddOneB,
    Double,
    FanInOutput,
    FanInTask,
    NumberInput,
    NumberOutput,
    Stringify,
)
from workflow_runner import (
    ExecutionContext,
    Task,
    Workflow,
    WorkflowDefinitionError,
)
from workflow_runner.exceptions import CycleDetectedError, IncompleteInputError


class TestLinearWorkflow:
    def test_valid_two_task_chain(self) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double])
        assert wf.name == "test"
        assert wf.result_task is Double

    def test_valid_three_task_chain(self) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double, Stringify])
        order = wf.topological_order()
        assert [t.name for t in order] == ["add_one", "double", "stringify"]

    def test_type_mismatch_raises(self) -> None:
        """Stringify outputs StringOutput, but Double expects NumberOutput."""
        with pytest.raises(WorkflowDefinitionError, match="Type mismatch"):
            Workflow(name="bad", tasks=[AddOne, Stringify, Double])

    def test_single_task_workflow(self) -> None:
        wf = Workflow(name="single", tasks=[AddOne])
        assert wf.result_task is AddOne
        assert wf.topological_order() == [AddOne]

    def test_empty_workflow(self) -> None:
        wf = Workflow(name="empty")
        assert wf._tasks == {}


class TestDAGWorkflow:
    def test_fan_in_workflow(self) -> None:
        wf = (
            Workflow.builder(name="fan_in")
            .add_task(AddOne)
            .add_task(AddOneB)
            .add_task(FanInTask, depends_on={"a": AddOne, "b": AddOneB})
            .build()
        )
        assert wf.result_task is FanInTask

    def test_fan_out_workflow(self) -> None:
        """Two tasks depend on the same upstream."""
        wf = (
            Workflow.builder(name="fan_out", result_task=Stringify)
            .add_task(AddOne)
            .add_task(Double, depends_on=AddOne)
            .add_task(Stringify, depends_on=AddOne)
            .build()
        )
        order = wf.topological_order()
        assert order[0] is AddOne
        assert set(order[1:]) == {Double, Stringify}

    def test_duplicate_task_name_raises(self) -> None:
        with pytest.raises(WorkflowDefinitionError, match="Duplicate"):
            Workflow.builder(name="bad").add_task(AddOne).add_task(AddOne).build()

    def test_cycle_detection(self) -> None:
        """Mutual dependency creates a cycle."""

        class TaskA(Task[NumberInput, NumberOutput]):
            name = "task_a"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
                return NumberOutput(value=input.value)

        class TaskB(Task[NumberOutput, NumberInput]):
            name = "task_b"

            def run(self, input: NumberOutput, ctx: ExecutionContext) -> NumberInput:
                return NumberInput(value=input.value)

        with pytest.raises(CycleDetectedError):
            (
                Workflow.builder(name="bad")
                .add_task(TaskA, depends_on=TaskB)
                .add_task(TaskB, depends_on=TaskA)
                .build()
            )

    def test_incomplete_fan_in_raises(self) -> None:
        """Fan-in task with missing required field."""

        class PartialInput(BaseModel):
            a: NumberOutput
            b: NumberOutput
            c: NumberOutput

        class PartialTask(Task[PartialInput, FanInOutput]):
            name = "partial_task"

            def run(self, input: PartialInput, ctx: ExecutionContext) -> FanInOutput:
                return FanInOutput(total=0)

        with pytest.raises(IncompleteInputError, match="Required field 'c'"):
            (
                Workflow.builder(name="bad")
                .add_task(AddOne)
                .add_task(AddOneB)
                .add_task(
                    PartialTask,
                    depends_on={"a": AddOne, "b": AddOneB},
                )
                .build()
            )

    def test_fan_in_type_mismatch(self) -> None:
        """Fan-in field type doesn't match upstream output."""

        class MismatchInput(BaseModel):
            a: NumberOutput
            b: str  # type: ignore[assignment]

        class MismatchTask(Task[MismatchInput, FanInOutput]):
            name = "mismatch_task"

            def run(self, input: MismatchInput, ctx: ExecutionContext) -> FanInOutput:
                return FanInOutput(total=0)

        with pytest.raises(WorkflowDefinitionError, match="Fan-in type mismatch"):
            (
                Workflow.builder(name="bad")
                .add_task(AddOne)
                .add_task(AddOneB)
                .add_task(
                    MismatchTask,
                    depends_on={"a": AddOne, "b": AddOneB},
                )
                .build()
            )

    def test_fan_in_unknown_field_raises(self) -> None:
        with pytest.raises(WorkflowDefinitionError, match="not found"):
            (
                Workflow.builder(name="bad")
                .add_task(AddOne)
                .add_task(
                    FanInTask,
                    depends_on={"a": AddOne, "nonexistent": AddOne},
                )
                .build()
            )


class TestTopologicalOrder:
    def test_linear_order(self) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double])
        order = wf.topological_order()
        assert order == [AddOne, Double]

    def test_dag_order_respects_dependencies(self) -> None:
        wf = (
            Workflow.builder(name="dag")
            .add_task(AddOne)
            .add_task(AddOneB)
            .add_task(FanInTask, depends_on={"a": AddOne, "b": AddOneB})
            .build()
        )
        order = wf.topological_order()
        # FanInTask must come after both AddOne and AddOneB
        assert order[-1] is FanInTask
        assert set(order[:2]) == {AddOne, AddOneB}


class TestResultTask:
    def test_auto_detect_sole_sink(self) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double])
        assert wf.result_task is Double

    def test_explicit_result_task(self) -> None:
        wf = (
            Workflow.builder(name="test", result_task=Double)
            .add_task(AddOne)
            .add_task(Double, depends_on=AddOne)
            .add_task(Stringify, depends_on=AddOne)
            .build()
        )
        assert wf.result_task is Double

    def test_ambiguous_sinks_raises(self) -> None:
        with pytest.raises(WorkflowDefinitionError, match="sink tasks"):
            (
                Workflow.builder(name="bad")
                .add_task(AddOne)
                .add_task(Double, depends_on=AddOne)
                .add_task(Stringify, depends_on=AddOne)
                .build()
            )


class TestGetDependencies:
    def test_root_task_deps_none(self) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double])
        assert wf.get_dependencies("add_one") is None

    def test_single_dep_string(self) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double])
        assert wf.get_dependencies("double") == "add_one"

    def test_fan_in_deps_dict(self) -> None:
        wf = (
            Workflow.builder(name="fan_in")
            .add_task(AddOne)
            .add_task(AddOneB)
            .add_task(FanInTask, depends_on={"a": AddOne, "b": AddOneB})
            .build()
        )
        deps = wf.get_dependencies("fan_in_task")
        assert deps == {"a": "add_one", "b": "add_one_b"}
