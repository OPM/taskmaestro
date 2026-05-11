"""Tests for Workflow and WorkflowBuilder."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from taskmaestro import (
    ExecutionContext,
    Task,
    Workflow,
    WorkflowDefinitionError,
)
from taskmaestro.exceptions import CycleDetectedError, IncompleteInputError
from tests.conftest import (
    AddOne,
    AddOneB,
    ConfigOnlyTask,
    Double,
    FanInOutput,
    FanInTask,
    FanInWithConfigTask,
    MergedOutput,
    MergeTask,
    NumberInput,
    NumberOutput,
    Stringify,
)


class TestLinearWorkflow:
    def test_valid_two_task_chain(self) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double])
        assert wf.name == "test"
        assert wf.result_task is Double

    def test_valid_three_task_chain(self) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double, Stringify])
        order = wf.topological_order()
        assert [name for name, _ in order] == ["add_one", "double", "stringify"]

    def test_type_mismatch_raises(self) -> None:
        """Stringify outputs StringOutput, but Double expects NumberOutput."""
        with pytest.raises(WorkflowDefinitionError, match="Type mismatch"):
            Workflow(name="bad", tasks=[AddOne, Stringify, Double])

    def test_single_task_workflow(self) -> None:
        wf = Workflow(name="single", tasks=[AddOne])
        assert wf.result_task is AddOne
        assert wf.topological_order() == [("add_one", AddOne)]

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
        assert order[0] == ("add_one", AddOne)
        assert set(order[1:]) == {("double", Double), ("stringify", Stringify)}

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

        # Use string forward references to create a cycle
        with pytest.raises(CycleDetectedError):
            (
                Workflow.builder(name="bad")
                .add_task(TaskA, depends_on="task_b")
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
        assert order == [("add_one", AddOne), ("double", Double)]

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
        assert order[-1] == ("fan_in_task", FanInTask)
        assert set(order[:2]) == {("add_one", AddOne), ("add_one_b", AddOneB)}


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


class TestOutputFieldRouting:
    """Tests for Feature 2: output field routing via tuple deps."""

    def test_valid_field_ref(self) -> None:
        """Single field ref validates and builds."""

        class MultiOut(BaseModel):
            stats: NumberOutput
            keywords: NumberOutput

        class Producer(Task[NumberInput, MultiOut]):
            name = "producer"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> MultiOut:
                return MultiOut(
                    stats=NumberOutput(value=input.value),
                    keywords=NumberOutput(value=input.value * 2),
                )

        wf = (
            Workflow.builder("field_ref", result_task=Double)
            .add_task(Producer)
            .add_task(Double, depends_on=(Producer, "stats"))
            .build()
        )
        assert wf.result_task is Double

    def test_field_ref_nonexistent_field_raises(self) -> None:
        class MultiOut(BaseModel):
            stats: NumberOutput

        class Producer(Task[NumberInput, MultiOut]):
            name = "producer"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> MultiOut:
                return MultiOut(stats=NumberOutput(value=0))

        with pytest.raises(WorkflowDefinitionError, match="Field 'bad_field' not found"):
            (
                Workflow.builder("bad")
                .add_task(Producer)
                .add_task(Double, depends_on=(Producer, "bad_field"))
                .build()
            )

    def test_field_ref_type_mismatch_raises(self) -> None:
        class MultiOut(BaseModel):
            stats: NumberOutput
            text: str

        class Producer(Task[NumberInput, MultiOut]):
            name = "producer"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> MultiOut:
                return MultiOut(stats=NumberOutput(value=0), text="hello")

        # Double expects NumberOutput as input, but "text" field is str
        with pytest.raises(WorkflowDefinitionError, match="Type mismatch"):
            (
                Workflow.builder("bad")
                .add_task(Producer)
                .add_task(Double, depends_on=(Producer, "text"))
                .build()
            )

    def test_fan_in_with_field_refs(self) -> None:
        """Mixed fan-in: some values are field refs, some are whole outputs."""

        class MultiOut(BaseModel):
            a: NumberOutput
            b: NumberOutput

        class Producer(Task[NumberInput, MultiOut]):
            name = "producer"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> MultiOut:
                return MultiOut(
                    a=NumberOutput(value=1),
                    b=NumberOutput(value=2),
                )

        wf = (
            Workflow.builder("mixed")
            .add_task(Producer)
            .add_task(AddOneB)
            .add_task(
                FanInTask,
                depends_on={
                    "a": (Producer, "a"),
                    "b": AddOneB,
                },
            )
            .build()
        )
        assert wf.result_task is FanInTask

    def test_fan_in_field_ref_nonexistent_raises(self) -> None:
        class MultiOut(BaseModel):
            a: NumberOutput

        class Producer(Task[NumberInput, MultiOut]):
            name = "producer"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> MultiOut:
                return MultiOut(a=NumberOutput(value=0))

        with pytest.raises(WorkflowDefinitionError, match="Field 'nope' not found"):
            (
                Workflow.builder("bad")
                .add_task(Producer)
                .add_task(AddOneB)
                .add_task(
                    FanInTask,
                    depends_on={
                        "a": (Producer, "nope"),
                        "b": AddOneB,
                    },
                )
                .build()
            )

    def test_topological_order_with_tuple_deps(self) -> None:
        class MultiOut(BaseModel):
            stats: NumberOutput
            keywords: NumberOutput

        class Producer(Task[NumberInput, MultiOut]):
            name = "producer"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> MultiOut:
                return MultiOut(
                    stats=NumberOutput(value=0),
                    keywords=NumberOutput(value=0),
                )

        wf = (
            Workflow.builder("topo", result_task=Stringify)
            .add_task(Producer)
            .add_task(Double, depends_on=(Producer, "stats"))
            .add_task(Stringify, depends_on=Double)
            .build()
        )
        order = wf.topological_order()
        names = [name for name, _ in order]
        assert names.index("producer") < names.index("double")
        assert names.index("double") < names.index("stringify")

    def test_sink_detection_with_tuple_deps(self) -> None:
        """Tuple deps correctly mark upstream as having dependents."""

        class MultiOut(BaseModel):
            stats: NumberOutput
            other: NumberOutput

        class Producer(Task[NumberInput, MultiOut]):
            name = "producer"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> MultiOut:
                return MultiOut(stats=NumberOutput(value=0), other=NumberOutput(value=0))

        wf = (
            Workflow.builder("sink_test")
            .add_task(Producer)
            .add_task(Double, depends_on=(Producer, "stats"))
            .build()
        )
        # Double is the only sink; Producer is not (it has a dependent)
        assert wf.result_task is Double

    def test_get_dependencies_tuple(self) -> None:
        class MultiOut(BaseModel):
            stats: NumberOutput

        class Producer(Task[NumberInput, MultiOut]):
            name = "producer"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> MultiOut:
                return MultiOut(stats=NumberOutput(value=0))

        wf = (
            Workflow.builder("deps_test")
            .add_task(Producer)
            .add_task(Double, depends_on=(Producer, "stats"))
            .build()
        )
        deps = wf.get_dependencies("double")
        assert deps == ("producer", "stats")


class TestResultTaskNotSet:
    def test_result_task_not_set_raises(self) -> None:
        """Accessing result_task on a workflow with no result task raises."""
        wf = Workflow(name="empty")
        with pytest.raises(WorkflowDefinitionError, match="No result task set"):
            _ = wf.result_task


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


class TestNamedTaskInstances:
    """Tests for using the same Task class with different names."""

    def test_same_class_different_names(self) -> None:
        """Same class added twice with different name= parameters builds successfully."""
        wf = (
            Workflow.builder(name="named")
            .add_task(AddOne, name="first")
            .add_task(AddOne, name="second")
            .add_task(
                FanInTask,
                depends_on={"a": "first", "b": "second"},
            )
            .build()
        )
        assert wf.result_task is FanInTask
        order = wf.topological_order()
        names = [name for name, _ in order]
        assert "first" in names
        assert "second" in names

    def test_string_dep_resolves(self) -> None:
        """depends_on with string name reference resolves correctly."""
        wf = (
            Workflow.builder(name="str_dep")
            .add_task(AddOne, name="step_one")
            .add_task(Double, depends_on="step_one")
            .build()
        )
        assert wf.get_dependencies("double") == "step_one"

    def test_class_ref_still_works_unambiguous(self) -> None:
        """Class reference in depends_on works when unambiguous (backwards compat)."""
        wf = (
            Workflow.builder(name="compat")
            .add_task(AddOne)
            .add_task(Double, depends_on=AddOne)
            .build()
        )
        assert wf.get_dependencies("double") == "add_one"

    def test_ambiguous_class_ref_raises(self) -> None:
        """Ambiguous class reference raises WorkflowDefinitionError."""
        with pytest.raises(WorkflowDefinitionError, match="Ambiguous reference"):
            (
                Workflow.builder(name="bad")
                .add_task(AddOne, name="first")
                .add_task(AddOne, name="second")
                .add_task(Double, depends_on=AddOne)
                .build()
            )

    def test_fan_in_mixed_string_and_class(self) -> None:
        """Fan-in dict with mixed string/class references."""
        wf = (
            Workflow.builder(name="mixed")
            .add_task(AddOne, name="branch_a")
            .add_task(AddOneB)
            .add_task(
                FanInTask,
                depends_on={"a": "branch_a", "b": AddOneB},
            )
            .build()
        )
        deps = wf.get_dependencies("fan_in_task")
        assert deps == {"a": "branch_a", "b": "add_one_b"}

    def test_type_validation_named_instances(self) -> None:
        """Type validation works for named instances (same I/O types)."""
        wf = (
            Workflow.builder(name="typed")
            .add_task(AddOne, name="first")
            .add_task(Double, depends_on="first")
            .build()
        )
        assert wf.result_task is Double

    def test_result_task_name_property(self) -> None:
        """result_task_name property returns the registered name."""
        wf = (
            Workflow.builder(name="rtn", result_task="custom_name")
            .add_task(AddOne, name="custom_name")
            .build()
        )
        assert wf.result_task_name == "custom_name"

    def test_result_task_name_not_set_raises(self) -> None:
        """result_task_name raises when no result task set."""
        wf = Workflow(name="empty")
        with pytest.raises(WorkflowDefinitionError, match="No result task set"):
            _ = wf.result_task_name

    def test_result_task_as_string(self) -> None:
        """result_task can be passed as string to builder."""
        wf = (
            Workflow.builder(name="str_result", result_task="my_double")
            .add_task(AddOne)
            .add_task(Double, name="my_double", depends_on=AddOne)
            .add_task(Stringify, depends_on=AddOne)
            .build()
        )
        assert wf.result_task is Double
        assert wf.result_task_name == "my_double"

    def test_dep_not_found_string_raises(self) -> None:
        """String dependency that doesn't exist raises."""
        with pytest.raises(WorkflowDefinitionError, match="not registered"):
            (
                Workflow.builder(name="bad")
                .add_task(AddOne)
                .add_task(Double, depends_on="nonexistent")
                .build()
            )

    def test_dep_not_found_class_raises(self) -> None:
        """Class dependency that hasn't been added raises."""
        with pytest.raises(WorkflowDefinitionError, match="not found"):
            (Workflow.builder(name="bad").add_task(Double, depends_on=AddOne).build())

    def test_tuple_dep_with_string_name(self) -> None:
        """Tuple dependency with string name reference."""

        class MultiOut(BaseModel):
            stats: NumberOutput

        class Producer(Task[NumberInput, MultiOut]):
            name = "producer"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> MultiOut:
                return MultiOut(stats=NumberOutput(value=0))

        wf = (
            Workflow.builder("tuple_str")
            .add_task(Producer, name="my_producer")
            .add_task(Double, depends_on=("my_producer", "stats"))
            .build()
        )
        assert wf.get_dependencies("double") == ("my_producer", "stats")

    def test_fan_in_tuple_with_string_name(self) -> None:
        """Fan-in with tuple field ref using string name."""

        class MultiOut(BaseModel):
            a: NumberOutput
            b: NumberOutput

        class Producer(Task[NumberInput, MultiOut]):
            name = "producer"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> MultiOut:
                return MultiOut(
                    a=NumberOutput(value=1),
                    b=NumberOutput(value=2),
                )

        wf = (
            Workflow.builder("fan_in_str")
            .add_task(Producer, name="my_producer")
            .add_task(AddOneB)
            .add_task(
                FanInTask,
                depends_on={
                    "a": ("my_producer", "a"),
                    "b": AddOneB,
                },
            )
            .build()
        )
        assert wf.result_task is FanInTask


class TestConfigFields:
    """Tests for config_fields support in workflow validation."""

    def test_root_task_with_config_fields(self) -> None:
        """Root task with config_fields covering all required fields validates."""
        wf = (
            Workflow.builder("root_cfg")
            .add_task(ConfigOnlyTask, config_fields=["path", "count"])
            .build()
        )
        assert wf.get_config_fields("config_only_task") == {"path", "count"}

    def test_root_task_config_fields_incomplete_raises(self) -> None:
        """Root task with config_fields not covering all required fields raises."""
        with pytest.raises(IncompleteInputError, match="Required field 'count'"):
            (Workflow.builder("bad").add_task(ConfigOnlyTask, config_fields=["path"]).build())

    def test_root_task_config_field_not_on_model_raises(self) -> None:
        """Config field that doesn't exist on input model raises."""
        with pytest.raises(WorkflowDefinitionError, match="Config field 'nonexistent'"):
            (
                Workflow.builder("bad")
                .add_task(
                    ConfigOnlyTask,
                    config_fields=["path", "count", "nonexistent"],
                )
                .build()
            )

    def test_single_dep_with_config_fields(self) -> None:
        """Single dep + config_fields: upstream fields + config cover all required."""
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
        assert wf.get_config_fields("merge_task") == {"label"}

    def test_single_dep_with_config_fields_incomplete_raises(self) -> None:
        """Single dep + config_fields not covering required field raises."""

        class NeedsMoreInput(BaseModel):
            value: int
            label: str
            extra: str

        class NeedsMoreTask(Task[NeedsMoreInput, MergedOutput]):
            name = "needs_more"

            def run(self, input: NeedsMoreInput, ctx: ExecutionContext) -> MergedOutput:
                return MergedOutput(result="")

        with pytest.raises(IncompleteInputError, match="Required field 'extra'"):
            (
                Workflow.builder("bad")
                .add_task(AddOne)
                .add_task(
                    NeedsMoreTask,
                    depends_on=AddOne,
                    config_fields=["label"],
                )
                .build()
            )

    def test_single_dep_config_field_not_on_model_raises(self) -> None:
        """Config field not on downstream input model raises."""
        with pytest.raises(WorkflowDefinitionError, match="Config field 'nonexistent'"):
            (
                Workflow.builder("bad")
                .add_task(AddOne)
                .add_task(
                    MergeTask,
                    depends_on=AddOne,
                    config_fields=["label", "nonexistent"],
                )
                .build()
            )

    def test_single_dep_with_config_type_mismatch_raises(self) -> None:
        """Upstream output field type incompatible with downstream input field raises."""

        class BadDownInput(BaseModel):
            value: str  # upstream provides int, this expects str
            label: str

        class BadDownTask(Task[BadDownInput, MergedOutput]):
            name = "bad_down"

            def run(self, input: BadDownInput, ctx: ExecutionContext) -> MergedOutput:
                return MergedOutput(result="")

        with pytest.raises(WorkflowDefinitionError, match="Type mismatch"):
            (
                Workflow.builder("bad")
                .add_task(AddOne)
                .add_task(
                    BadDownTask,
                    depends_on=AddOne,
                    config_fields=["label"],
                )
                .build()
            )

    def test_fan_in_with_config_fields(self) -> None:
        """Fan-in + config_fields: mapped fields + config cover all required."""
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
        assert wf.get_config_fields("fan_in_with_config") == {"extra"}

    def test_fan_in_with_config_fields_incomplete_raises(self) -> None:
        """Fan-in + config_fields not covering a required field raises."""
        with pytest.raises(IncompleteInputError, match="Required field 'extra'"):
            (
                Workflow.builder("bad")
                .add_task(AddOne)
                .add_task(
                    FanInWithConfigTask,
                    depends_on={"a": AddOne},
                    # missing config_fields=["extra"]
                )
                .build()
            )

    def test_fan_in_config_field_not_on_model_raises(self) -> None:
        """Config field not on fan-in input model raises."""
        with pytest.raises(WorkflowDefinitionError, match="Config field 'bogus'"):
            (
                Workflow.builder("bad")
                .add_task(AddOne)
                .add_task(
                    FanInWithConfigTask,
                    depends_on={"a": AddOne},
                    config_fields=["extra", "bogus"],
                )
                .build()
            )

    def test_backward_compat_no_config_fields(self) -> None:
        """Workflows without config_fields work as before."""
        wf = Workflow(name="compat", tasks=[AddOne, Double])
        assert wf.get_config_fields("add_one") == set()
        assert wf.get_config_fields("double") == set()

    def test_get_config_fields_default(self) -> None:
        """get_config_fields returns empty set for unknown task."""
        wf = Workflow(name="empty")
        assert wf.get_config_fields("nonexistent") == set()
