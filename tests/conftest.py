"""Shared fixtures and reusable test components."""

from __future__ import annotations

import time

import pytest
from pydantic import BaseModel

from workflow_runner import ExecutionContext, Task

# --- Reusable Pydantic models ---


class NumberInput(BaseModel):
    value: int


class NumberOutput(BaseModel):
    value: int


class StringOutput(BaseModel):
    text: str


class FanInInput(BaseModel):
    a: NumberOutput
    b: NumberOutput


class FanInOutput(BaseModel):
    total: int


# --- Reusable tasks ---


class AddOne(Task[NumberInput, NumberOutput]):
    name = "add_one"

    def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
        return NumberOutput(value=input.value + 1)


class AddOneB(Task[NumberInput, NumberOutput]):
    name = "add_one_b"

    def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
        return NumberOutput(value=input.value + 1)


class Double(Task[NumberOutput, NumberOutput]):
    name = "double"

    def run(self, input: NumberOutput, ctx: ExecutionContext) -> NumberOutput:
        return NumberOutput(value=input.value * 2)


class Stringify(Task[NumberOutput, StringOutput]):
    name = "stringify"

    def run(self, input: NumberOutput, ctx: ExecutionContext) -> StringOutput:
        return StringOutput(text=str(input.value))


class FailingTask(Task[NumberInput, NumberOutput]):
    name = "failing_task"

    def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
        raise ValueError("Task failed intentionally")


class SlowTask(Task[NumberInput, NumberOutput]):
    name = "slow_task"
    timeout_seconds = 1.0

    def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
        time.sleep(5)
        return NumberOutput(value=input.value)


class WrongOutputTask(Task[NumberInput, NumberOutput]):
    name = "wrong_output"

    def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
        return StringOutput(text="wrong")  # type: ignore[return-value]


class FanInTask(Task[FanInInput, FanInOutput]):
    name = "fan_in_task"

    def run(self, input: FanInInput, ctx: ExecutionContext) -> FanInOutput:
        return FanInOutput(total=input.a.value + input.b.value)


# --- Fixtures ---


@pytest.fixture
def ctx() -> ExecutionContext:
    return ExecutionContext()


@pytest.fixture
def number_input() -> NumberInput:
    return NumberInput(value=5)
