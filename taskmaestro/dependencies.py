"""Dependency references used to collect multiple task outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar, overload

from pydantic import BaseModel

from taskmaestro.exceptions import WorkflowDefinitionError
from taskmaestro.task import Task

O = TypeVar("O", bound=BaseModel)
T = TypeVar("T")


@dataclass(frozen=True)
class OutputHandle(Generic[T]):
    """Reference to one field of a task instance's output."""

    task_name: str
    field_name: str
    annotation: Any
    _owner: object = field(repr=False)


@dataclass(frozen=True)
class TaskHandle(Generic[O]):
    """Unambiguous reference to one registered task instance."""

    name: str
    output_type: type[BaseModel]
    _owner: object = field(repr=False)

    def field(self, name: str) -> OutputHandle[Any]:
        """Return a validated reference to a named output field."""
        fields = self.output_type.model_fields
        if name not in fields:
            raise WorkflowDefinitionError(
                f"Field '{name}' not found on {self.output_type.__name__} (output of {self.name})"
            )
        return OutputHandle(
            task_name=self.name,
            field_name=name,
            annotation=fields[name].annotation,
            _owner=self._owner,
        )


type TaskReference = type[Task[Any, Any]] | str | TaskHandle[Any]
type OutputReference = TaskReference | tuple[TaskReference, str] | OutputHandle[Any]


@dataclass(frozen=True)
class CollectionDependency:
    """Unresolved collection declared through :func:`collect`."""

    kind: Literal["positional", "keyed"]
    positional_members: tuple[OutputReference, ...] = ()
    keyed_members: tuple[tuple[str, OutputReference], ...] = ()


@dataclass(frozen=True)
class OutputRef:
    """A resolved reference to a task output or one of its fields."""

    task_name: str
    output_field: str | None = None


@dataclass(frozen=True)
class CollectionRef:
    """A collection dependency whose task references have been resolved."""

    kind: Literal["positional", "keyed"]
    positional_members: tuple[OutputRef, ...] = ()
    keyed_members: tuple[tuple[str, OutputRef], ...] = ()

    def output_refs(self) -> tuple[OutputRef, ...]:
        """Return all output references in declaration order."""
        if self.kind == "positional":
            return self.positional_members
        return tuple(ref for _key, ref in self.keyed_members)


@overload
def collect() -> CollectionDependency: ...


@overload
def collect(*members: OutputReference) -> CollectionDependency: ...


@overload
def collect(members: Mapping[str, OutputReference], /) -> CollectionDependency: ...


@overload
def collect(**members: OutputReference) -> CollectionDependency: ...


def collect(
    *members: OutputReference | Mapping[str, OutputReference],
    **keyed_members: OutputReference,
) -> CollectionDependency:
    """Collect several upstream outputs into one list or dictionary input field.

    Positional members target ``list[T]`` fields. A single mapping argument or
    keyword arguments target ``dict[str, T]`` fields. Members may be task
    classes, task handles, registered names, output handles, or
    ``(task, output_field)`` references.
    """
    if keyed_members:
        if members:
            raise TypeError("collect() accepts either positional members or keyword members")
        return CollectionDependency("keyed", keyed_members=tuple(keyed_members.items()))

    if len(members) == 1 and isinstance(members[0], Mapping):
        mapping = members[0]
        if not all(isinstance(key, str) for key in mapping):
            raise TypeError("collect() dictionary keys must be strings")
        return CollectionDependency("keyed", keyed_members=tuple(mapping.items()))

    if any(isinstance(member, Mapping) for member in members):
        raise TypeError("collect() accepts either positional members or one mapping")

    return CollectionDependency(
        "positional",
        positional_members=tuple(members),  # type: ignore[arg-type]
    )
