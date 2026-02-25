"""Tests for Task base class and type introspection."""

from __future__ import annotations

from pydantic import BaseModel

from taskekrabbe import ExecutionContext, Task
from taskekrabbe.task import get_input_type, get_output_type
from tests.conftest import (
    AddOne,
    Double,
    FanInTask,
    NumberInput,
    NumberOutput,
    StringOutput,
)


class TestTaskName:
    def test_default_name_from_class(self) -> None:
        class MyTask(Task[NumberInput, NumberOutput]):
            def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
                return NumberOutput(value=input.value)

        assert MyTask.name == "MyTask"

    def test_explicit_name(self) -> None:
        assert AddOne.name == "add_one"

    def test_name_not_inherited(self) -> None:
        """Each subclass gets its own name if not explicitly set."""

        class Parent(Task[NumberInput, NumberOutput]):
            name = "parent"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
                return NumberOutput(value=input.value)

        class Child(Parent):
            def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
                return NumberOutput(value=input.value)

        assert Child.name == "Child"


class TestTaskTimeout:
    def test_default_timeout(self) -> None:
        assert AddOne.timeout_seconds is None

    def test_explicit_timeout(self) -> None:
        class TimedTask(Task[NumberInput, NumberOutput]):
            timeout_seconds = 30.0

            def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
                return NumberOutput(value=input.value)

        assert TimedTask.timeout_seconds == 30.0


class TestTypeIntrospection:
    def test_get_input_type(self) -> None:
        assert get_input_type(AddOne) is NumberInput

    def test_get_output_type(self) -> None:
        assert get_output_type(AddOne) is NumberOutput

    def test_get_input_type_double(self) -> None:
        assert get_input_type(Double) is NumberOutput

    def test_get_output_type_double(self) -> None:
        assert get_output_type(Double) is NumberOutput

    def test_fan_in_input_type(self) -> None:
        from tests.conftest import FanInInput

        assert get_input_type(FanInTask) is FanInInput

    def test_inherited_task_introspection(self) -> None:
        """Type introspection works through inheritance."""

        class SpecialInput(BaseModel):
            x: int

        class SpecialOutput(BaseModel):
            y: int

        class Base(Task[SpecialInput, SpecialOutput]):
            def run(self, input: SpecialInput, ctx: ExecutionContext) -> SpecialOutput:
                return SpecialOutput(y=input.x)

        class Derived(Base):
            pass

        assert get_input_type(Derived) is SpecialInput
        assert get_output_type(Derived) is SpecialOutput

    def test_stringify_types(self) -> None:
        from tests.conftest import Stringify

        assert get_input_type(Stringify) is NumberOutput
        assert get_output_type(Stringify) is StringOutput


class TestTypeVarSkipping:
    def test_unresolved_typevar_skipped(self) -> None:
        """Unresolved TypeVar in a generic Task subclass is skipped, then raises TypeError."""
        from typing import TypeVar as TV

        T = TV("T", bound=BaseModel)

        class GenericTask(Task[T, NumberOutput]):  # type: ignore[type-var]
            name = "generic_task"

            def run(self, input: BaseModel, ctx: ExecutionContext) -> NumberOutput:
                return NumberOutput(value=0)

        import pytest

        with pytest.raises(TypeError, match="Cannot resolve type argument"):
            get_input_type(GenericTask)


class TestInlinePorts:
    """Tests for Feature 1: inner Inputs/Outputs class detection."""

    def test_inner_inputs_resolved(self) -> None:
        class MyTask(Task):  # type: ignore[type-arg]
            name = "my_task"

            class Inputs(BaseModel):
                x: int

            class Outputs(BaseModel):
                y: int

            def run(self, input: MyTask.Inputs, ctx: ExecutionContext) -> MyTask.Outputs:
                return self.Outputs(y=input.x + 1)

        assert get_input_type(MyTask) is MyTask.Inputs

    def test_inner_outputs_resolved(self) -> None:
        class MyTask(Task):  # type: ignore[type-arg]
            name = "my_task"

            class Inputs(BaseModel):
                x: int

            class Outputs(BaseModel):
                y: int

            def run(self, input: MyTask.Inputs, ctx: ExecutionContext) -> MyTask.Outputs:
                return self.Outputs(y=input.x + 1)

        assert get_output_type(MyTask) is MyTask.Outputs

    def test_inner_classes_take_precedence_over_generics(self) -> None:
        """When both generics and inner classes are present, inner classes win."""

        class InnerIn(BaseModel):
            a: str

        class InnerOut(BaseModel):
            b: str

        class BothTask(Task[NumberInput, NumberOutput]):
            name = "both_task"

            Inputs = InnerIn
            Outputs = InnerOut

            def run(self, input: InnerIn, ctx: ExecutionContext) -> InnerOut:
                return InnerOut(b=input.a)

        assert get_input_type(BothTask) is InnerIn
        assert get_output_type(BothTask) is InnerOut

    def test_no_generics_no_inner_raises(self) -> None:
        """Task with neither generics nor inner classes raises TypeError."""

        class EmptyTask(Task):  # type: ignore[type-arg]
            name = "empty"

            def run(self, input: BaseModel, ctx: ExecutionContext) -> BaseModel:
                return BaseModel()

        with __import__("pytest").raises(TypeError, match="Cannot resolve type argument"):
            get_input_type(EmptyTask)

    def test_inline_task_in_workflow_end_to_end(self) -> None:
        """An inline-ports task validates and runs in a workflow."""

        class Intermediate(BaseModel):
            val: int

        class Source(Task):  # type: ignore[type-arg]
            name = "inline_source"

            class Inputs(BaseModel):
                val: int

            Outputs = Intermediate

            def run(self, input: Source.Inputs, ctx: ExecutionContext) -> Intermediate:
                return Intermediate(val=input.val + 1)

        class Sink(Task):  # type: ignore[type-arg]
            name = "inline_sink"

            Inputs = Intermediate

            class Outputs(BaseModel):
                text: str

            def run(self, input: Intermediate, ctx: ExecutionContext) -> Sink.Outputs:
                return self.Outputs(text=str(input.val))

        from taskekrabbe import Job, Runner, Workflow

        wf = Workflow("inline_test", tasks=[Source, Sink])
        job = Job(workflow=wf, config=Source.Inputs(val=10))
        result = Runner().run(job)
        assert result.result is not None
        assert result.result.text == "11"  # type: ignore[attr-defined]
