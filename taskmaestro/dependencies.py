"""Dependency references used to collect multiple task outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, overload

from taskmaestro.task import Task

type TaskReference = type[Task[Any, Any]] | str
type OutputReference = TaskReference | tuple[TaskReference, str]


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


def collect(
    *members: OutputReference | Mapping[str, OutputReference],
) -> CollectionDependency:
    """Collect several upstream outputs into one list or dictionary input field.

    Positional members target ``list[T]`` fields. A single mapping argument
    targets ``dict[str, T]`` fields. Members may be task classes, registered
    task names, or ``(task, output_field)`` references.
    """
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
