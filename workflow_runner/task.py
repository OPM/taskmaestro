"""Task base class with typed inputs/outputs and type introspection."""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, get_args

from pydantic import BaseModel

from workflow_runner.context import ExecutionContext

I = TypeVar("I", bound=BaseModel)
O = TypeVar("O", bound=BaseModel)


class Task(ABC, Generic[I, O]):
    """Base class for all tasks.

    Parameterized by input type I and output type O, both Pydantic models.
    """

    name: str
    timeout_seconds: float | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "name" not in cls.__dict__:
            cls.name = cls.__name__

    @abstractmethod
    def run(self, input: I, ctx: ExecutionContext) -> O:
        """Execute the task. Receives validated input and context, returns validated output."""
        ...


def get_input_type(task_cls: type[Task[Any, Any]]) -> type[BaseModel]:
    """Extract the concrete input type (I) from a Task subclass.

    Checks for an inner ``Inputs`` class first, then falls back to
    generic type argument introspection.
    """
    inner: type[Any] | None = getattr(task_cls, "Inputs", None)
    if inner is not None and isinstance(inner, type) and issubclass(inner, BaseModel):
        return inner
    return _get_type_arg(task_cls, 0)


def get_output_type(task_cls: type[Task[Any, Any]]) -> type[BaseModel]:
    """Extract the concrete output type (O) from a Task subclass.

    Checks for an inner ``Outputs`` class first, then falls back to
    generic type argument introspection.
    """
    inner: type[Any] | None = getattr(task_cls, "Outputs", None)
    if inner is not None and isinstance(inner, type) and issubclass(inner, BaseModel):
        return inner
    return _get_type_arg(task_cls, 1)


def _get_type_arg(task_cls: type[Task[Any, Any]], index: int) -> type[BaseModel]:
    """Walk the MRO to find concrete type arguments for Task[I, O]."""
    for base in task_cls.__mro__:
        for orig_base in getattr(base, "__orig_bases__", ()):
            origin = typing.get_origin(orig_base)
            if origin is Task or (origin is not None and issubclass(origin, Task)):
                args = get_args(orig_base)
                if args and len(args) > index:
                    arg = args[index]
                    if isinstance(arg, type) and issubclass(arg, BaseModel):
                        return arg
                    if isinstance(arg, TypeVar):
                        continue
    raise TypeError(
        f"Cannot resolve type argument {index} for {task_cls.__name__}. "
        f"Ensure the class directly specifies concrete types in Task[I, O]."
    )
