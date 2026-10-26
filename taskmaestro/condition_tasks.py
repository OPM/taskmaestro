"""Ready-made condition tasks, registered as ``condition.*`` task plugins.

They make simple ``if`` branches possible without writing a task: wire an
upstream output into the condition and guard tasks with ``if_true()`` /
``if_false()`` (``case: true|false`` in YAML).
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from taskmaestro.conditions import ConditionTask
from taskmaestro.context import ExecutionContext

type CompareOp = Literal["<", "<=", "==", "!=", ">=", ">"]

_OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}


class CompareInput(BaseModel):
    """Input of :class:`Compare`."""

    value: int | float = Field(description="Value to compare")
    op: CompareOp = Field(default=">=", description="Comparison operator")
    threshold: float = Field(description="Value to compare against")


class Compare(ConditionTask[CompareInput]):
    """``True`` when ``value <op> threshold`` holds, e.g. ``value >= 100``."""

    name = "compare"

    def decide(self, input: CompareInput, ctx: ExecutionContext) -> bool:
        return _OPERATORS[input.op](input.value, input.threshold)


class IsTrueInput(BaseModel):
    """Input of :class:`IsTrue`."""

    value: Any = Field(description="Value to test")


class IsTrue(ConditionTask[IsTrueInput]):
    """``True`` when ``value`` is truthy (not ``False``, ``None``, zero or empty)."""

    name = "is_true"

    def decide(self, input: IsTrueInput, ctx: ExecutionContext) -> bool:
        return bool(input.value)


class PathExistsInput(BaseModel):
    """Input of :class:`PathExists`."""

    path: Path = Field(description="File or directory to look for")
    kind: Literal["any", "file", "directory"] = Field(
        default="any", description="Kind of path that must exist"
    )


class PathExists(ConditionTask[PathExistsInput]):
    """``True`` when ``path`` exists (and is a file or directory if ``kind`` says so)."""

    name = "path_exists"

    def decide(self, input: PathExistsInput, ctx: ExecutionContext) -> bool:
        path = input.path.expanduser()
        if input.kind == "file":
            return path.is_file()
        if input.kind == "directory":
            return path.is_dir()
        return path.exists()
