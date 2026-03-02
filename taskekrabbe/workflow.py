"""Workflow (DAG) definition with linear shorthand and builder pattern."""

from __future__ import annotations

from typing import Any, Union

from pydantic import BaseModel

from taskekrabbe.exceptions import (
    CycleDetectedError,
    IncompleteInputError,
    WorkflowDefinitionError,
)
from taskekrabbe.task import Task, get_input_type, get_output_type

# Stored dependency types after name resolution:
#   None              — root task
#   str               — single upstream (whole output)
#   tuple[str, str]   — single upstream, specific field
#   dict[str, str | tuple[str, str]]  — fan-in (values may be field refs)
DepValue = Union[str, "tuple[str, str]"]
StoredDeps = Union[dict[str, DepValue], str, "tuple[str, str]", None]


def _extract_upstream_names(deps: StoredDeps) -> set[str]:
    """Return the set of upstream task names referenced by *deps*."""
    if deps is None:
        return set()
    if isinstance(deps, str):
        return {deps}
    if isinstance(deps, tuple):
        return {deps[0]}
    # dict
    names: set[str] = set()
    for v in deps.values():
        if isinstance(v, tuple):
            names.add(v[0])
        else:
            names.add(v)
    return names


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
        self._result_task_name: str | None = None

        if tasks:
            for i, task_cls in enumerate(tasks):
                self._tasks[task_cls.name] = task_cls
                if i == 0:
                    self._dependencies[task_cls.name] = None
                else:
                    prev = tasks[i - 1]
                    self._dependencies[task_cls.name] = prev.name
            self._result_task_name = tasks[-1].name
            self._validate()

        if result_task is not None:
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

    def _validate(self) -> None:
        self._validate_unique_names()
        self._validate_references()
        self._validate_acyclic()
        self._validate_types()
        self._validate_result_task()

    def _validate_unique_names(self) -> None:
        """Raise WorkflowDefinitionError on duplicate task names."""
        # Already handled by dict keys in _tasks; duplicates would overwrite.
        # For linear shorthand, check the input list explicitly.
        pass

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

    def _validate_types(self) -> None:
        """Validate type compatibility for all edges."""
        for name, deps in self._dependencies.items():
            task_cls = self._tasks[name]
            if deps is None:
                # Root task — validated at Job creation time
                continue
            elif isinstance(deps, str):
                # Single dependency (whole output)
                upstream_cls = self._tasks[deps]
                upstream_output = get_output_type(upstream_cls)
                downstream_input = get_input_type(task_cls)
                if upstream_output is not downstream_input:
                    raise WorkflowDefinitionError(
                        f"Type mismatch: {deps} outputs "
                        f"{upstream_output.__name__} but {name} expects "
                        f"{downstream_input.__name__}"
                    )
            elif isinstance(deps, tuple):
                # Single dependency, specific output field
                upstream_name, field_name = deps
                upstream_cls = self._tasks[upstream_name]
                upstream_output = get_output_type(upstream_cls)
                upstream_fields = upstream_output.model_fields
                if field_name not in upstream_fields:
                    raise WorkflowDefinitionError(
                        f"Field '{field_name}' not found on "
                        f"{upstream_output.__name__} (output of {upstream_name})"
                    )
                field_annotation = upstream_fields[field_name].annotation
                downstream_input = get_input_type(task_cls)
                if field_annotation is not None and downstream_input is not field_annotation:
                    raise WorkflowDefinitionError(
                        f"Type mismatch: {upstream_name}.{field_name} is "
                        f"{field_annotation.__name__} but {name} expects "
                        f"{downstream_input.__name__}"
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
                    if isinstance(upstream_ref, tuple):
                        up_name, up_field = upstream_ref
                        up_cls = self._tasks[up_name]
                        up_output = get_output_type(up_cls)
                        up_fields = up_output.model_fields
                        if up_field not in up_fields:
                            raise WorkflowDefinitionError(
                                f"Field '{up_field}' not found on "
                                f"{up_output.__name__} (output of {up_name})"
                            )
                        resolved_type = up_fields[up_field].annotation
                    else:
                        up_cls = self._tasks[upstream_ref]
                        resolved_type = get_output_type(up_cls)
                    field_annotation = model_fields[field_name].annotation
                    if (
                        field_annotation is not None
                        and resolved_type is not None
                        and not issubclass(resolved_type, field_annotation)
                    ):
                        raise WorkflowDefinitionError(
                            f"Fan-in type mismatch: {upstream_ref} outputs "
                            f"{resolved_type.__name__} but field '{field_name}' "
                            f"on {downstream_input.__name__} expects "
                            f"{field_annotation.__name__}"
                        )
                # Check all required fields are covered
                for field_name, field_info in model_fields.items():
                    if field_name not in deps and field_info.is_required():
                        raise IncompleteInputError(
                            f"Required field '{field_name}' on "
                            f"{downstream_input.__name__} is not mapped to any "
                            f"upstream task"
                        )

    def _validate_result_task(self) -> None:
        """Ensure result_task is set. Default to sole sink; raise if ambiguous."""
        sinks = self._find_sinks()
        if self._result_task_name is None:
            if len(sinks) == 1:
                self._result_task_name = sinks[0]
            else:
                raise WorkflowDefinitionError(
                    f"Workflow '{self.name}' has {len(sinks)} sink tasks "
                    f"({sinks}); specify result_task explicitly"
                )

    def to_mermaid(self) -> str:
        """Return a Mermaid diagram string for this workflow."""
        from taskekrabbe.visualization import to_mermaid

        return to_mermaid(self)

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

    def add_task(
        self,
        task_cls: type[Task[Any, Any]],
        *,
        name: str | None = None,
        depends_on: (
            type[Task[Any, Any]]
            | str
            | tuple[type[Task[Any, Any]] | str, str]
            | dict[str, type[Task[Any, Any]] | str | tuple[type[Task[Any, Any]] | str, str]]
            | None
        ) = None,
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
            resolved: dict[str, DepValue] = {}
            for field, dep in depends_on.items():
                if isinstance(dep, tuple):
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
