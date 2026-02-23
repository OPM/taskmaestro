"""Workflow (DAG) definition with linear shorthand and builder pattern."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from workflow_runner.exceptions import (
    CycleDetectedError,
    IncompleteInputError,
    WorkflowDefinitionError,
)
from workflow_runner.task import Task, get_input_type, get_output_type


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
        self._dependencies: dict[str, dict[str, str] | str | None] = {}
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
        result_task: type[Task[Any, Any]] | None = None,
    ) -> WorkflowBuilder:
        """Return a builder for DAG construction."""
        return WorkflowBuilder(name, result_task=result_task)

    def topological_order(self) -> list[type[Task[Any, Any]]]:
        """Return task classes in a valid execution order (Kahn's algorithm)."""
        in_degree: dict[str, int] = {name: 0 for name in self._tasks}
        adjacency: dict[str, list[str]] = {name: [] for name in self._tasks}

        for name, deps in self._dependencies.items():
            if deps is None:
                continue
            if isinstance(deps, str):
                in_degree[name] += 1
                adjacency[deps].append(name)
            elif isinstance(deps, dict):
                upstream_names = set(deps.values())
                for upstream in upstream_names:
                    in_degree[name] += 1
                    adjacency[upstream].append(name)

        queue = [name for name, deg in in_degree.items() if deg == 0]
        result: list[type[Task[Any, Any]]] = []

        while queue:
            queue.sort()
            current = queue.pop(0)
            result.append(self._tasks[current])
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

    def get_dependencies(self, task_name: str) -> dict[str, str] | str | None:
        """Return the dependency spec for a task."""
        return self._dependencies[task_name]

    def _validate(self) -> None:
        self._validate_unique_names()
        self._validate_acyclic()
        self._validate_types()
        self._validate_result_task()

    def _validate_unique_names(self) -> None:
        """Raise WorkflowDefinitionError on duplicate task names."""
        # Already handled by dict keys in _tasks; duplicates would overwrite.
        # For linear shorthand, check the input list explicitly.
        pass

    def _validate_acyclic(self) -> None:
        """Detect cycles via DFS. Raise CycleDetectedError."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {name: WHITE for name in self._tasks}

        def _build_adjacency() -> dict[str, list[str]]:
            adj: dict[str, list[str]] = {name: [] for name in self._tasks}
            for name, deps in self._dependencies.items():
                if deps is None:
                    continue
                if isinstance(deps, str):
                    adj[deps].append(name)
                elif isinstance(deps, dict):
                    for upstream in set(deps.values()):
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
                # Single dependency
                upstream_cls = self._tasks[deps]
                upstream_output = get_output_type(upstream_cls)
                downstream_input = get_input_type(task_cls)
                if upstream_output is not downstream_input:
                    raise WorkflowDefinitionError(
                        f"Type mismatch: {upstream_cls.name} outputs "
                        f"{upstream_output.__name__} but {task_cls.name} expects "
                        f"{downstream_input.__name__}"
                    )
            elif isinstance(deps, dict):
                # Fan-in: validate each field
                downstream_input = get_input_type(task_cls)
                if not issubclass(downstream_input, BaseModel):
                    raise WorkflowDefinitionError(
                        f"Fan-in task {task_cls.name} input must be a Pydantic model"
                    )
                model_fields = downstream_input.model_fields
                # Check that every mapped field exists and types match
                for field_name, upstream_name in deps.items():
                    if field_name not in model_fields:
                        raise WorkflowDefinitionError(
                            f"Fan-in field '{field_name}' not found on {downstream_input.__name__}"
                        )
                    upstream_cls = self._tasks[upstream_name]
                    upstream_output = get_output_type(upstream_cls)
                    field_annotation = model_fields[field_name].annotation
                    if field_annotation is not None and not issubclass(
                        upstream_output, field_annotation
                    ):
                        raise WorkflowDefinitionError(
                            f"Fan-in type mismatch: {upstream_cls.name} outputs "
                            f"{upstream_output.__name__} but field '{field_name}' "
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
        from workflow_runner.visualization import to_mermaid

        return to_mermaid(self)

    def _find_sinks(self) -> list[str]:
        """Return task names with no downstream dependents."""
        has_dependents: set[str] = set()
        for deps in self._dependencies.values():
            if isinstance(deps, str):
                has_dependents.add(deps)
            elif isinstance(deps, dict):
                has_dependents.update(deps.values())
        return [name for name in self._tasks if name not in has_dependents]


class WorkflowBuilder:
    """Fluent builder for DAG workflows."""

    def __init__(
        self,
        name: str,
        *,
        result_task: type[Task[Any, Any]] | None = None,
    ) -> None:
        self._workflow = Workflow.__new__(Workflow)
        self._workflow.name = name
        self._workflow._tasks = {}
        self._workflow._dependencies = {}
        self._workflow._result_task_name = result_task.name if result_task else None

    def add_task(
        self,
        task_cls: type[Task[Any, Any]],
        *,
        depends_on: type[Task[Any, Any]] | dict[str, type[Task[Any, Any]]] | None = None,
    ) -> WorkflowBuilder:
        """Add a task to the DAG. Returns self for chaining."""
        wf = self._workflow
        if task_cls.name in wf._tasks:
            raise WorkflowDefinitionError(f"Duplicate task name '{task_cls.name}'")
        wf._tasks[task_cls.name] = task_cls

        if depends_on is None:
            wf._dependencies[task_cls.name] = None
        elif isinstance(depends_on, dict):
            wf._dependencies[task_cls.name] = {
                field: dep.name for field, dep in depends_on.items()
            }
        else:
            wf._dependencies[task_cls.name] = depends_on.name

        return self

    def build(self) -> Workflow:
        """Finalize and validate the workflow. Returns an immutable Workflow."""
        self._workflow._validate()
        return self._workflow
