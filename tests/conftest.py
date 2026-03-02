"""Shared fixtures and reusable test components."""

from __future__ import annotations

import time

import pytest
from pydantic import BaseModel

from taskekrabbe import ExecutionContext, Task

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


# --- Models and tasks for config_fields testing ---


class MergedInput(BaseModel):
    """Input that takes some fields from upstream and some from config."""

    value: int  # from upstream
    label: str  # from config


class MergedOutput(BaseModel):
    result: str


class MergeTask(Task[MergedInput, MergedOutput]):
    name = "merge_task"

    def run(self, input: MergedInput, ctx: ExecutionContext) -> MergedOutput:
        return MergedOutput(result=f"{input.label}:{input.value}")


class ConfigOnlyInput(BaseModel):
    """Input that comes entirely from config."""

    path: str
    count: int


class ConfigOnlyOutput(BaseModel):
    summary: str


class ConfigOnlyTask(Task[ConfigOnlyInput, ConfigOnlyOutput]):
    name = "config_only_task"

    def run(self, input: ConfigOnlyInput, ctx: ExecutionContext) -> ConfigOnlyOutput:
        return ConfigOnlyOutput(summary=f"{input.path}x{input.count}")


class FanInWithConfigInput(BaseModel):
    """Fan-in input with one field from upstream and one from config."""

    a: NumberOutput  # from upstream
    extra: str  # from config


class FanInWithConfigOutput(BaseModel):
    combined: str


class FanInWithConfigTask(Task[FanInWithConfigInput, FanInWithConfigOutput]):
    name = "fan_in_with_config"

    def run(self, input: FanInWithConfigInput, ctx: ExecutionContext) -> FanInWithConfigOutput:
        return FanInWithConfigOutput(combined=f"{input.extra}:{input.a.value}")


# --- Fixtures ---


@pytest.fixture
def ctx() -> ExecutionContext:
    return ExecutionContext()


@pytest.fixture
def number_input() -> NumberInput:
    return NumberInput(value=5)
