"""Tests for Task base class and type introspection."""

from __future__ import annotations

from pydantic import BaseModel

from tests.conftest import (
    AddOne,
    Double,
    FanInTask,
    NumberInput,
    NumberOutput,
    StringOutput,
)
from workflow_runner import ExecutionContext, Task
from workflow_runner.task import get_input_type, get_output_type


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
