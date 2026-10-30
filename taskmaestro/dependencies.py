"""Dependency references used to collect multiple task outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, overload

from pydantic import BaseModel

from taskmaestro.exceptions import WorkflowDefinitionError
from taskmaestro.task import Task

if TYPE_CHECKING:
    from taskmaestro.conditions import Condition

O = TypeVar("O", bound=BaseModel)
T = TypeVar("T")


@dataclass(frozen=True)
class OutputHandle(Generic[T]):
    """Reference to one field of a task instance's output."""

    task_name: str
    field_name: str
    annotation: Any
    _owner: object = field(repr=False)

    def equals(self, value: Any) -> Condition:
        """Guard: run the task when this field equals *value*."""
        from taskmaestro.conditions import when

        return when(self, equals=value)

    def is_in(self, *values: Any) -> Condition:
        """Guard: run the task when this field equals one of *values*."""
        from taskmaestro.conditions import when

        return when(self, in_=values)

    def truthy(self) -> Condition:
        """Guard: run the task when this field is ``True`` (or, if not a bool, not ``None``)."""
        from taskmaestro.conditions import when

        return when(self, truthy=True)

    def falsy(self) -> Condition:
        """Guard: run the task when this field is ``False`` or ``None``."""
        from taskmaestro.conditions import when

        return when(self, truthy=False)


@dataclass(frozen=True)
class TaskHandle(Generic[O]):
    """Unambiguous reference to one registered task instance."""

    name: str
    output_type: type[BaseModel]
    _owner: object = field(repr=False)

    def on(self, *labels: Any) -> Condition:
        """Guard: run the task when this switch selects one of *labels*."""
        from taskmaestro.conditions import Condition

        if not labels:
            raise TypeError("on() needs at least one branch label")
        return Condition(self, "in", labels)

    def otherwise(self) -> Condition:
        """Guard: run the task when this switch selects a label no other guard claims."""
        from taskmaestro.conditions import Condition

        return Condition(self, "otherwise")

    def if_true(self) -> Condition:
        """Guard: run the task when this condition task returns ``True``."""
        return self.on(True)

    def if_false(self) -> Condition:
        """Guard: run the task when this condition task returns ``False``."""
        return self.on(False)

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


type MemberMode = Literal["item", "list", "dict"]


@dataclass(frozen=True)
class CollectionRef:
    """A collection dependency whose task references have been resolved.

    ``member_modes`` is filled in by workflow validation, aligned with
    :meth:`output_refs`: ``"item"`` adds a member's value as one element,
    ``"list"`` and ``"dict"`` flatten a ``list[T]`` or ``dict[str, T]`` value
    into the collection. An empty tuple means every member is an item.
    """

    kind: Literal["positional", "keyed"]
    positional_members: tuple[OutputRef, ...] = ()
    keyed_members: tuple[tuple[str, OutputRef], ...] = ()
    member_modes: tuple[MemberMode, ...] = ()

    def output_refs(self) -> tuple[OutputRef, ...]:
        """Return all output references in declaration order."""
        if self.kind == "positional":
            return self.positional_members
        return tuple(ref for _key, ref in self.keyed_members)

    def labeled_members(self) -> tuple[tuple[str, OutputRef, MemberMode], ...]:
        """Return ``(label, ref, mode)`` per member; labels are list indexes or keys."""
        if self.kind == "positional":
            labeled = [(str(index), ref) for index, ref in enumerate(self.positional_members)]
        else:
            labeled = list(self.keyed_members)
        item: MemberMode = "item"
        modes = self.member_modes or (item,) * len(labeled)
        return tuple((label, ref, mode) for (label, ref), mode in zip(labeled, modes, strict=True))


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


@dataclass(frozen=True)
class FirstOfDependency:
    """Unresolved alternative dependency declared through :func:`first_of`."""

    members: tuple[OutputReference, ...]


@dataclass(frozen=True)
class FirstOfRef:
    """An alternative dependency whose task references have been resolved."""

    members: tuple[OutputRef, ...]

    def output_refs(self) -> tuple[OutputRef, ...]:
        """Return all output references in declaration order."""
        return self.members


def first_of(*members: OutputReference) -> FirstOfDependency:
    """Take one input field from whichever of several conditional branches ran.

    At runtime the field receives the output of the first member, in
    declaration order, that was not skipped. If every member was skipped the
    downstream task is skipped too. Members may be task classes, task handles,
    registered names, output handles, or ``(task, output_field)`` references.
    """
    if len(members) < 2:
        raise TypeError("first_of() needs at least two members")
    return FirstOfDependency(tuple(members))
