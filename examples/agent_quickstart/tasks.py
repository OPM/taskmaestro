"""Small, dependency-free tasks for the agent quickstart."""

from pydantic import BaseModel

from taskmaestro import ExecutionContext, Task


class Number(BaseModel):
    value: int


class AddOne(Task[Number, Number]):
    def run(self, input: Number, ctx: ExecutionContext) -> Number:
        return Number(value=input.value + 1)


class Double(Task[Number, Number]):
    def run(self, input: Number, ctx: ExecutionContext) -> Number:
        return Number(value=input.value * 2)
