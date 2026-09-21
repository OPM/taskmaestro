"""Workflow (DAG) definition with linear shorthand and builder pattern."""

from __future__ import annotations

import types
import typing
from typing import TYPE_CHECKING, Any, get_args, get_origin

from pydantic import BaseModel

if TYPE_CHECKING:
    from taskmaestro.job import JobConfiguration

from taskmaestro.dependencies import (
    CollectionDependency,
    CollectionRef,
    OutputRef,
    OutputReference,
)
from taskmaestro.exceptions import (
    CycleDetectedError,
    IncompleteInputError,
    WorkflowDefinitionError,
)
from taskmaestro.mapping import MappedOutput, TaskMap
from taskmaestro.task import Task, get_input_type, get_output_type

# Stored dependency types after name resolution:
#   None              — root task
#   str               — single upstream (whole output)
#   tuple[str, str]   — single upstream, specific field
#   dict[str, str | tuple[str, str]]  — fan-in (values may be field refs)
type DepValue = str | tuple[str, str]
type FanInValue = DepValue | CollectionRef
type StoredDeps = dict[str, FanInValue] | str | tuple[str, str] | None


def _extract_upstream_names(deps: StoredDeps) -> set[str]:
    """Return the set of upstream task names referenced by *deps*."""
    if deps is None:
        return set()
    if isinstance(deps, str):
        return {deps}
    if isinstance(deps, tuple):
        return {deps[0]}
    names: set[str] = set()
    for value in deps.values():
        if isinstance(value, CollectionRef):
            names.update(ref.task_name for ref in value.output_refs())
        elif isinstance(value, tuple):
            names.add(value[0])
        else:
            names.add(value)
    return names


def _is_union(annotation: Any) -> bool:
    """Return whether *annotation* is either spelling of a union."""
    return get_origin(annotation) in (typing.Union, types.UnionType)


def _is_type_compatible(produced: Any, expected: Any) -> bool:
    """Return whether a produced annotation can be assigned to an expected one.

    Parameterized annotations are compared recursively.  This deliberately
    treats their arguments covariantly: task outputs are validated by Pydantic
    before they cross an edge, so the question here is whether every produced
    value is accepted by the downstream annotation rather than whether a
    mutable container may safely be shared between arbitrary Python callers.
    """
    if expected is Any or produced is Any or produced == expected:
        return True

    # Every possible produced value must be accepted.  Conversely, an expected
    # union only needs one arm which accepts the produced annotation.
    if _is_union(produced):
        return all(_is_type_compatible(option, expected) for option in get_args(produced))
    if _is_union(expected):
        return any(_is_type_compatible(produced, option) for option in get_args(expected))

    produced_origin = get_origin(produced)
    expected_origin = get_origin(expected)
    if produced_origin is not None or expected_origin is not None:
        produced_base = produced_origin or produced
        expected_base = expected_origin or expected
        if not isinstance(produced_base, type) or not isinstance(expected_base, type):
            return False
        if not issubclass(produced_base, expected_base):
            return False

        produced_args = get_args(produced)
        expected_args = get_args(expected)
        if not expected_args:
            return True
        if not produced_args:
            return False

        # A fixed-length tuple can be assigned to tuple[T, ...] when each of
        # its elements can be assigned to T.
        if expected_base is tuple and len(expected_args) == 2 and expected_args[1] is Ellipsis:
            if len(produced_args) == 2 and produced_args[1] is Ellipsis:
                return _is_type_compatible(produced_args[0], expected_args[0])
            return all(_is_type_compatible(arg, expected_args[0]) for arg in produced_args)

        if len(produced_args) != len(expected_args):
            return False
        return all(
            _is_type_compatible(produced_arg, expected_arg)
            for produced_arg, expected_arg in zip(produced_args, expected_args, strict=True)
        )

    if isinstance(produced, type) and isinstance(expected, type):
        return issubclass(produced, expected)
    return False


def _type_name(annotation: Any) -> str:
    """Return a readable, complete name for a runtime or typing annotation."""
    if annotation is Any:
        return "Any"
    if annotation is None or annotation is type(None):
        return "None"
    if annotation is Ellipsis:
        return "..."
    if _is_union(annotation):
        return " | ".join(_type_name(arg) for arg in get_args(annotation))
    origin = get_origin(annotation)
    if origin is not None:
        origin_name = getattr(origin, "__name__", str(origin).removeprefix("typing."))
        return f"{origin_name}[{', '.join(_type_name(arg) for arg in get_args(annotation))}]"
    return getattr(annotation, "__name__", str(annotation))


class Workflow:
    """A DAG of tasks. Linear pipelines are a special case."""

    def __init__(
        self,
        name: str,
        tasks: list[type[Task[Any, Any]]] | None = None,
        *,
        result_task: type[Task[Any, Any]] | None = None,
    ) -> None:
        """Linear shorthand: auto-chain tasks[0] -> tasks[1] -> ... -> tasks[N].

        For DAG construction, use Workflow.builder() instead.
        """
        self.name = name
        self._tasks: dict[str, type[Task[Any, Any]]] = {}
        self._dependencies: dict[str, StoredDeps] = {}
        self._config_fields: dict[str, set[str]] = {}
        self._task_maps: dict[str, TaskMap] = {}
        self._result_task_name: str | None = None

        if tasks is not None:
            if not tasks:
                raise WorkflowDefinitionError(
                    f"Workflow '{name}' was given an empty task list; "
                    "pass at least one task or use Workflow.builder()"
                )
            for i, task_cls in enumerate(tasks):
                if task_cls.name in self._tasks:
                    raise WorkflowDefinitionError(f"Duplicate task name '{task_cls.name}'")
                self._tasks[task_cls.name] = task_cls
                if i == 0:
                    self._dependencies[task_cls.name] = None
                else:
                    prev = tasks[i - 1]
                    self._dependencies[task_cls.name] = prev.name
            self._result_task_name = result_task.name if result_task is not None else None
            self._validate()
        elif result_task is not None:
            self._result_task_name = result_task.name

    @classmethod
    def builder(
        cls,
        name: str,
        *,
        result_task: type[Task[Any, Any]] | str | None = None,
    ) -> WorkflowBuilder:
        """Return a builder for DAG construction."""
        return WorkflowBuilder(name, result_task=result_task)

    def topological_order(self) -> list[tuple[str, type[Task[Any, Any]]]]:
        """Return (name, task_class) pairs in a valid execution order (Kahn's algorithm)."""
        in_degree: dict[str, int] = {name: 0 for name in self._tasks}
        adjacency: dict[str, list[str]] = {name: [] for name in self._tasks}

        for name, deps in self._dependencies.items():
            for upstream in _extract_upstream_names(deps):
                in_degree[name] += 1
                adjacency[upstream].append(name)

        queue = [name for name, deg in in_degree.items() if deg == 0]
        result: list[tuple[str, type[Task[Any, Any]]]] = []

        while queue:
            queue.sort()
            current = queue.pop(0)
            result.append((current, self._tasks[current]))
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    @property
    def result_task(self) -> type[Task[Any, Any]]:
        """The task whose output becomes job.result."""
        if self._result_task_name is None:
            raise WorkflowDefinitionError("No result task set")
        return self._tasks[self._result_task_name]

    @property
    def result_task_name(self) -> str:
        """The registered name of the result task."""
        if self._result_task_name is None:
            raise WorkflowDefinitionError("No result task set")
        return self._result_task_name

    def get_dependencies(self, task_name: str) -> StoredDeps:
        """Return the dependency spec for a task."""
        return self._dependencies[task_name]

    def get_config_fields(self, task_name: str) -> set[str]:
        """Return the set of config field names for a task, or empty set."""
        return self._config_fields.get(task_name, set())

    def get_task_map(self, task_name: str) -> TaskMap | None:
        """Return the mapping declaration for a task, if it is mapped."""
        return self._task_maps.get(task_name)

    def is_mapped_task(self, task_name: str) -> bool:
        """Return whether a registered task expands over configured items."""
        return task_name in self._task_maps

    def get_output_annotation(self, task_name: str) -> Any:
        """Return a task instance's effective output annotation."""
        output_type = get_output_type(self._tasks[task_name])
        if self.is_mapped_task(task_name):
            return MappedOutput[output_type]  # type: ignore[valid-type]
        return output_type

    def _validate(self) -> None:
        self._validate_unique_names()
        self._validate_references()
        self._validate_acyclic()
        self._validate_task_maps()
        self._validate_types()
        self._validate_result_task()

    def _validate_unique_names(self) -> None:
        """Duplicate task names are rejected at registration time.

        Both ``Workflow(tasks=[...])`` and ``WorkflowBuilder.add_task`` check
        before inserting into ``_tasks``, so by the time validation runs the
        mapping is guaranteed to be unique.  Kept as an explicit step so the
        validation order documented in CLAUDE.md remains visible here.
        """

    def _validate_references(self) -> None:
        """Ensure all dependency references point to registered task names."""
        for name, deps in self._dependencies.items():
            for upstream in _extract_upstream_names(deps):
                if upstream not in self._tasks:
                    raise WorkflowDefinitionError(
                        f"Task '{name}' depends on '{upstream}', "
                        f"which is not registered in the workflow"
                    )

    def _validate_acyclic(self) -> None:
        """Detect cycles via DFS. Raise CycleDetectedError."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {name: WHITE for name in self._tasks}

        def _build_adjacency() -> dict[str, list[str]]:
            adj: dict[str, list[str]] = {name: [] for name in self._tasks}
            for name, deps in self._dependencies.items():
                for upstream in _extract_upstream_names(deps):
                    adj[upstream].append(name)
            return adj

        adjacency = _build_adjacency()

        def dfs(node: str) -> None:
            color[node] = GRAY
            for neighbor in adjacency[node]:
                if color[neighbor] == GRAY:
                    raise CycleDetectedError(f"Cycle detected involving task '{neighbor}'")
                if color[neighbor] == WHITE:
                    dfs(neighbor)
            color[node] = BLACK

        for node in self._tasks:
            if color[node] == WHITE:
                dfs(node)

    def _validate_task_maps(self) -> None:
        """Validate mapped input fields and their sources."""
        for task_name, task_map in self._task_maps.items():
            input_type = get_input_type(self._tasks[task_name])
            fields = input_type.model_fields
            for map_field in (task_map.key_as, task_map.value_as):
                if map_field not in fields:
                    raise WorkflowDefinitionError(
                        f"Map field '{map_field}' not found on {input_type.__name__} "
                        f"(input of '{task_name}')"
                    )
            if not _is_type_compatible(str, fields[task_map.key_as].annotation):
                raise WorkflowDefinitionError(
                    f"Map key field '{task_name}.{task_map.key_as}' must accept strings"
                )

            config_fields = self.get_config_fields(task_name)
            reserved = {task_map.key_as, task_map.value_as, task_map.over}
            overlap = reserved & config_fields
            if overlap:
                raise WorkflowDefinitionError(
                    f"Mapped task '{task_name}' fields {sorted(overlap)} cannot also be "
                    "config_fields"
                )

            deps = self._dependencies[task_name]
            if deps is None:
                dependency_fields: set[str] = set()
            elif isinstance(deps, dict):
                dependency_fields = set(deps)
            else:
                raise WorkflowDefinitionError(
                    f"Mapped task '{task_name}' requires named field dependencies"
                )
            injected = {task_map.key_as, task_map.value_as}
            overlap = injected & dependency_fields
            if overlap:
                raise WorkflowDefinitionError(
                    f"Mapped task '{task_name}' fields {sorted(overlap)} cannot also be "
                    "dependencies"
                )
            covered = dependency_fields | config_fields | injected
            for field_name, field_info in fields.items():
                if field_name not in covered and field_info.is_required():
                    raise IncompleteInputError(
                        f"Required field '{field_name}' on {input_type.__name__} is not "
                        f"covered for mapped task '{task_name}'"
                    )

    def _validate_types(self) -> None:
        """Validate type compatibility for all edges."""
        for name, deps in self._dependencies.items():
            task_cls = self._tasks[name]
            cf = self._config_fields.get(name, set())
            if deps is None:
                # Root task — validated at Job creation time, or via config_fields
                if cf:
                    downstream_input = get_input_type(task_cls)
                    model_fields = downstream_input.model_fields
                    # Validate config_fields cover all required input fields
                    task_map = self.get_task_map(name)
                    map_fields = (
                        {task_map.key_as, task_map.value_as} if task_map is not None else set()
                    )
                    for field_name, field_info in model_fields.items():
                        if field_name not in cf | map_fields and field_info.is_required():
                            raise IncompleteInputError(
                                f"Required field '{field_name}' on "
                                f"{downstream_input.__name__} is not covered by "
                                f"config_fields for root task '{name}'"
                            )
                    # Validate config field names exist on the model
                    for field_name in cf:
                        if field_name not in model_fields:
                            raise WorkflowDefinitionError(
                                f"Config field '{field_name}' not found on "
                                f"{downstream_input.__name__} (input of '{name}')"
                            )
                continue
            elif isinstance(deps, str):
                # Single dependency (whole output)
                upstream_output = self.get_output_annotation(deps)
                downstream_input = get_input_type(task_cls)
                if cf:
                    if self.is_mapped_task(deps):
                        raise WorkflowDefinitionError(
                            f"Mapped upstream task '{deps}' must be connected through "
                            "a named input field"
                        )
                    # With config_fields: check upstream output fields exist in
                    # downstream input with compatible types, and that upstream
                    # fields + config_fields cover all required fields
                    up_fields = upstream_output.model_fields
                    down_fields = downstream_input.model_fields
                    # Validate config field names exist on the model
                    for field_name in cf:
                        if field_name not in down_fields:
                            raise WorkflowDefinitionError(
                                f"Config field '{field_name}' not found on "
                                f"{downstream_input.__name__} (input of '{name}')"
                            )
                    # Check upstream output fields that match downstream input
                    for field_name, field_info in up_fields.items():
                        if field_name in down_fields:
                            up_annotation = field_info.annotation
                            down_annotation = down_fields[field_name].annotation
                            if (
                                up_annotation is not None
                                and down_annotation is not None
                                and not _is_type_compatible(up_annotation, down_annotation)
                            ):
                                raise WorkflowDefinitionError(
                                    f"Type mismatch: {deps}.{field_name} is "
                                    f"{_type_name(up_annotation)} but {name}.{field_name} "
                                    f"expects {_type_name(down_annotation)}"
                                )
                    # Check all required fields are covered by upstream or config
                    covered = set(up_fields.keys()) | cf
                    for field_name, field_info in down_fields.items():
                        if field_name not in covered and field_info.is_required():
                            raise IncompleteInputError(
                                f"Required field '{field_name}' on "
                                f"{downstream_input.__name__} is not covered by "
                                f"upstream output or config_fields"
                            )
                else:
                    if not _is_type_compatible(upstream_output, downstream_input):
                        raise WorkflowDefinitionError(
                            f"Type mismatch: {deps} outputs "
                            f"{_type_name(upstream_output)} but {name} expects "
                            f"{_type_name(downstream_input)}"
                        )
            elif isinstance(deps, tuple):
                # Single dependency, specific output field
                upstream_name, field_name = deps
                upstream_output = self.get_output_annotation(upstream_name)
                upstream_fields = upstream_output.model_fields
                if field_name not in upstream_fields:
                    raise WorkflowDefinitionError(
                        f"Field '{field_name}' not found on "
                        f"{upstream_output.__name__} (output of {upstream_name})"
                    )
                field_annotation = upstream_fields[field_name].annotation
                downstream_input = get_input_type(task_cls)
                if field_annotation is not None and not _is_type_compatible(
                    field_annotation, downstream_input
                ):
                    raise WorkflowDefinitionError(
                        f"Type mismatch: {upstream_name}.{field_name} is "
                        f"{_type_name(field_annotation)} but {name} expects "
                        f"{_type_name(downstream_input)}"
                    )
            elif isinstance(deps, dict):
                # Fan-in: validate each field
                downstream_input = get_input_type(task_cls)
                assert issubclass(downstream_input, BaseModel)  # guaranteed by Task[I, O] bound
                model_fields = downstream_input.model_fields
                # Check that every mapped field exists and types match
                for field_name, upstream_ref in deps.items():
                    if field_name not in model_fields:
                        raise WorkflowDefinitionError(
                            f"Fan-in field '{field_name}' not found on {downstream_input.__name__}"
                        )
                    field_annotation = model_fields[field_name].annotation
                    if isinstance(upstream_ref, CollectionRef):
                        if field_name in cf:
                            raise WorkflowDefinitionError(
                                f"Field '{field_name}' on task '{name}' is supplied by both "
                                "a collection dependency and config_fields"
                            )
                        self._validate_collection(
                            name,
                            field_name,
                            field_annotation,
                            upstream_ref,
                        )
                        continue
                    if isinstance(upstream_ref, tuple):
                        up_name, up_field = upstream_ref
                        up_output = self.get_output_annotation(up_name)
                        up_fields = up_output.model_fields
                        if up_field not in up_fields:
                            raise WorkflowDefinitionError(
                                f"Field '{up_field}' not found on "
                                f"{up_output.__name__} (output of {up_name})"
                            )
                        resolved_type = up_fields[up_field].annotation
                    else:
                        resolved_type = self.get_output_annotation(upstream_ref)
                    if (
                        field_annotation is not None
                        and resolved_type is not None
                        and not _is_type_compatible(resolved_type, field_annotation)
                    ):
                        raise WorkflowDefinitionError(
                            f"Fan-in type mismatch: {upstream_ref} outputs "
                            f"{_type_name(resolved_type)} but field '{field_name}' "
                            f"on {downstream_input.__name__} expects "
                            f"{_type_name(field_annotation)}"
                        )
                # Validate config field names exist on the model
                for field_name in cf:
                    if field_name not in model_fields:
                        raise WorkflowDefinitionError(
                            f"Config field '{field_name}' not found on "
                            f"{downstream_input.__name__} (input of '{name}')"
                        )
                # Check all required fields are covered by deps or config_fields
                task_map = self.get_task_map(name)
                map_fields = (
                    {task_map.key_as, task_map.value_as} if task_map is not None else set()
                )
                covered = set(deps.keys()) | cf | map_fields
                for field_name, field_info in model_fields.items():
                    if field_name not in covered and field_info.is_required():
                        raise IncompleteInputError(
                            f"Required field '{field_name}' on "
                            f"{downstream_input.__name__} is not mapped to any "
                            f"upstream task"
                        )

    def _resolve_output_ref_type(self, ref: OutputRef) -> Any:
        """Resolve the type produced by an output reference."""
        output_type = self.get_output_annotation(ref.task_name)
        if ref.output_field is None:
            return output_type
        if ref.output_field not in output_type.model_fields:
            raise WorkflowDefinitionError(
                f"Field '{ref.output_field}' not found on {output_type.__name__} "
                f"(output of {ref.task_name})"
            )
        return output_type.model_fields[ref.output_field].annotation

    def _validate_collection(
        self,
        task_name: str,
        field_name: str,
        field_annotation: Any,
        collection: CollectionRef,
    ) -> None:
        """Validate a collection dependency against its destination field."""
        origin = get_origin(field_annotation)
        args = get_args(field_annotation)
        if collection.kind == "positional":
            if origin is not list or len(args) != 1:
                raise WorkflowDefinitionError(
                    f"Positional collection for '{task_name}.{field_name}' requires "
                    f"a list[T] field, got {_type_name(field_annotation)}"
                )
            expected_type = args[0]
            members = [
                (str(index), ref) for index, ref in enumerate(collection.positional_members)
            ]
        else:
            if origin is not dict or len(args) != 2 or args[0] is not str:
                raise WorkflowDefinitionError(
                    f"Keyed collection for '{task_name}.{field_name}' requires "
                    f"a dict[str, T] field, got {_type_name(field_annotation)}"
                )
            expected_type = args[1]
            members = list(collection.keyed_members)

        for member_label, ref in members:
            produced_type = self._resolve_output_ref_type(ref)
            if produced_type is not None and not _is_type_compatible(produced_type, expected_type):
                source = ref.task_name
                if ref.output_field is not None:
                    source += f".{ref.output_field}"
                raise WorkflowDefinitionError(
                    f"Collection type mismatch for "
                    f"'{task_name}.{field_name}[{member_label}]': '{source}' produces "
                    f"{_type_name(produced_type)}, but collection element type is "
                    f"{_type_name(expected_type)}"
                )

    def _validate_result_task(self) -> None:
        """Ensure result_task is set and registered. Default to sole sink; raise if ambiguous."""
        if self._result_task_name is not None:
            if self._result_task_name not in self._tasks:
                raise WorkflowDefinitionError(
                    f"result_task '{self._result_task_name}' is not registered in "
                    f"workflow '{self.name}' (known tasks: {sorted(self._tasks)})"
                )
            return
        sinks = self._find_sinks()
        if len(sinks) == 1:
            self._result_task_name = sinks[0]
        else:
            raise WorkflowDefinitionError(
                f"Workflow '{self.name}' has {len(sinks)} sink tasks "
                f"({sinks}); specify result_task explicitly"
            )

    def as_task(
        self,
        *,
        name: str | None = None,
        job_configuration: JobConfiguration | None = None,
    ) -> type[Task[Any, Any]]:
        """Wrap this workflow as a task for composition in another workflow.

        The task input is inferred from the sole unconfigured root task and its
        output from this workflow's result task. If every root is configured,
        pass ``job_configuration`` and the generated task accepts ``EmptyConfig``.
        """
        from taskmaestro.workflow_task import workflow_task

        return workflow_task(self, name=name, job_configuration=job_configuration)

    def to_mermaid(self, *, job_configuration: JobConfiguration | None = None) -> str:
        """Return a Mermaid diagram string for this workflow."""
        from taskmaestro.visualization import to_mermaid

        return to_mermaid(self, job_configuration=job_configuration)

    def _find_sinks(self) -> list[str]:
        """Return task names with no downstream dependents."""
        has_dependents: set[str] = set()
        for deps in self._dependencies.values():
            has_dependents.update(_extract_upstream_names(deps))
        return [name for name in self._tasks if name not in has_dependents]


class WorkflowBuilder:
    """Fluent builder for DAG workflows."""

    def __init__(
        self,
        name: str,
        *,
        result_task: type[Task[Any, Any]] | str | None = None,
    ) -> None:
        self._workflow = Workflow.__new__(Workflow)
        self._workflow.name = name
        self._workflow._tasks = {}
        self._workflow._dependencies = {}
        self._workflow._config_fields = {}
        self._workflow._task_maps = {}
        self._workflow._result_task_name = None
        # Store the raw result_task ref for resolution at build() time
        self._result_task_ref: type[Task[Any, Any]] | str | None = result_task

    def _resolve_dep_name(self, dep_cls: type[Task[Any, Any]]) -> str:
        """Resolve a class reference to its registered name.

        Raises WorkflowDefinitionError if the class is not registered or
        is registered under multiple names (ambiguous).
        """
        wf = self._workflow
        matches = [name for name, cls in wf._tasks.items() if cls is dep_cls]
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            raise WorkflowDefinitionError(
                f"Task class '{dep_cls.__name__}' not found in workflow. "
                f"Add it with add_task() before referencing it as a dependency."
            )
        raise WorkflowDefinitionError(
            f"Ambiguous reference to task class '{dep_cls.__name__}' — "
            f"it is registered under multiple names: {matches}. "
            f"Use a string name to disambiguate."
        )

    def _resolve_dep_ref(
        self,
        dep: type[Task[Any, Any]] | str,
    ) -> str:
        """Resolve a dependency reference (class or string) to a registered name.

        String references are accepted as-is (validated at build time),
        allowing forward references to tasks not yet added.
        """
        if isinstance(dep, str):
            return dep
        return self._resolve_dep_name(dep)

    def _resolve_output_reference(self, ref: OutputReference) -> OutputRef:
        """Resolve a public task/output-field reference."""
        if isinstance(ref, tuple):
            task_ref, output_field = ref
            return OutputRef(self._resolve_dep_ref(task_ref), output_field)
        return OutputRef(self._resolve_dep_ref(ref))

    def _resolve_collection(self, collection: CollectionDependency) -> CollectionRef:
        """Resolve every task reference in a collection dependency."""
        if collection.kind == "positional":
            return CollectionRef(
                "positional",
                positional_members=tuple(
                    self._resolve_output_reference(ref) for ref in collection.positional_members
                ),
            )
        return CollectionRef(
            "keyed",
            keyed_members=tuple(
                (key, self._resolve_output_reference(ref)) for key, ref in collection.keyed_members
            ),
        )

    def add_task(
        self,
        task_cls: type[Task[Any, Any]],
        *,
        name: str | None = None,
        depends_on: (
            type[Task[Any, Any]]
            | str
            | tuple[type[Task[Any, Any]] | str, str]
            | dict[
                str,
                type[Task[Any, Any]]
                | str
                | tuple[type[Task[Any, Any]] | str, str]
                | CollectionDependency,
            ]
            | None
        ) = None,
        config_fields: list[str] | None = None,
        mapped_over: TaskMap | None = None,
    ) -> WorkflowBuilder:
        """Add a task to the DAG. Returns self for chaining.

        ``name`` optionally overrides the task class's default name,
        allowing the same class to appear multiple times with different names.

        ``depends_on`` accepts:
        - ``None`` — root task (no upstream)
        - ``TaskClass`` — single upstream, whole output
        - ``"task_name"`` — single upstream by registered name
        - ``(TaskClass | "name", "field")`` — single upstream, specific output field
        - ``{"field": TaskClass | "name", ...}`` — fan-in, whole outputs
        - ``{"field": (TaskClass | "name", "f"), ...}`` — fan-in with field routing
        - ``{"field": collect(...), ...}`` — collect outputs into a list or dictionary

        ``mapped_over`` expands this logical task over a configured mapping.
        """
        wf = self._workflow
        task_name = name if name is not None else task_cls.name
        if task_name in wf._tasks:
            raise WorkflowDefinitionError(f"Duplicate task name '{task_name}'")
        wf._tasks[task_name] = task_cls

        if depends_on is None:
            wf._dependencies[task_name] = None
        elif isinstance(depends_on, tuple):
            dep_ref, field = depends_on
            resolved_name = self._resolve_dep_ref(dep_ref)
            wf._dependencies[task_name] = (resolved_name, field)
        elif isinstance(depends_on, dict):
            resolved: dict[str, FanInValue] = {}
            for field, dep in depends_on.items():
                if isinstance(dep, CollectionDependency):
                    resolved[field] = self._resolve_collection(dep)
                elif isinstance(dep, tuple):
                    dep_ref, dep_field = dep
                    resolved_name = self._resolve_dep_ref(dep_ref)
                    resolved[field] = (resolved_name, dep_field)
                else:
                    resolved[field] = self._resolve_dep_ref(dep)
            wf._dependencies[task_name] = resolved
        elif isinstance(depends_on, str):
            resolved_name = self._resolve_dep_ref(depends_on)
            wf._dependencies[task_name] = resolved_name
        else:
            # Class reference
            resolved_name = self._resolve_dep_name(depends_on)
            wf._dependencies[task_name] = resolved_name

        if config_fields is not None:
            wf._config_fields[task_name] = set(config_fields)
        if mapped_over is not None:
            wf._task_maps[task_name] = mapped_over

        return self

    def build(self) -> Workflow:
        """Finalize and validate the workflow. Returns an immutable Workflow."""
        # Resolve result_task ref
        ref = self._result_task_ref
        if ref is not None:
            if isinstance(ref, str):
                self._workflow._result_task_name = ref
            else:
                self._workflow._result_task_name = self._resolve_dep_name(ref)
        self._workflow._validate()
        return self._workflow
