"""Discovery of tasks and workflows published by installed distributions."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from taskmaestro.exceptions import PluginLoadError
from taskmaestro.task import Task
from taskmaestro.workflow import Workflow

TASK_ENTRY_POINT_GROUP = "taskmaestro.tasks"
WORKFLOW_ENTRY_POINT_GROUP = "taskmaestro.workflows"


def _entry_points_by_name(group: str) -> dict[str, EntryPoint]:
    """Return entry points in *group*, rejecting ambiguous registrations."""
    result: dict[str, EntryPoint] = {}
    for entry_point in entry_points(group=group):
        if entry_point.name in result:
            raise PluginLoadError(
                f"Multiple entry points named '{entry_point.name}' are registered in '{group}'"
            )
        result[entry_point.name] = entry_point
    return result


def _load[T](entry_point: EntryPoint, expected_type: type[T], kind: str) -> T:
    try:
        value: Any = entry_point.load()
    except Exception as exc:
        raise PluginLoadError(
            f"Cannot load {kind} entry point '{entry_point.name}' ({entry_point.value}): {exc}"
        ) from exc
    if not isinstance(value, expected_type):
        raise PluginLoadError(
            f"{kind.capitalize()} entry point '{entry_point.name}' ({entry_point.value}) "
            f"must resolve to a {expected_type.__name__}"
        )
    return value


def registered_task_names() -> set[str]:
    """Return registered task identifiers without importing their packages."""
    return set(_entry_points_by_name(TASK_ENTRY_POINT_GROUP))


def registered_workflow_names() -> set[str]:
    """Return registered workflow identifiers without importing their packages."""
    return set(_entry_points_by_name(WORKFLOW_ENTRY_POINT_GROUP))


def registered_tasks() -> dict[str, type[Task[Any, Any]]]:
    """Load all tasks registered in the ``taskmaestro.tasks`` entry-point group.

    The returned mapping is keyed by the entry-point name, which is the stable
    identifier that configuration files and discovery clients should use.
    """
    tasks: dict[str, type[Task[Any, Any]]] = {}
    for name, entry_point in _entry_points_by_name(TASK_ENTRY_POINT_GROUP).items():
        task = _load(entry_point, type, "task")
        if not issubclass(task, Task):
            raise PluginLoadError(
                f"Task entry point '{name}' ({entry_point.value}) must resolve to a Task subclass"
            )
        tasks[name] = task
    return tasks


def registered_workflows() -> dict[str, Workflow]:
    """Load all workflows registered in the ``taskmaestro.workflows`` group."""
    return {
        name: _load(entry_point, Workflow, "workflow")
        for name, entry_point in _entry_points_by_name(WORKFLOW_ENTRY_POINT_GROUP).items()
    }


def get_registered_task(name: str) -> type[Task[Any, Any]]:
    """Load one registered task by its entry-point name."""
    entry_point = _entry_points_by_name(TASK_ENTRY_POINT_GROUP).get(name)
    if entry_point is None:
        raise PluginLoadError(f"No task entry point named '{name}' is registered")
    task = _load(entry_point, type, "task")
    if not issubclass(task, Task):
        raise PluginLoadError(
            f"Task entry point '{name}' ({entry_point.value}) must resolve to a Task subclass"
        )
    return task


def get_registered_workflow(name: str) -> Workflow:
    """Load one registered workflow by its entry-point name."""
    entry_point = _entry_points_by_name(WORKFLOW_ENTRY_POINT_GROUP).get(name)
    if entry_point is None:
        raise PluginLoadError(f"No workflow entry point named '{name}' is registered")
    return _load(entry_point, Workflow, "workflow")


@dataclass(frozen=True)
class RegistryProblem:
    """A registered entry point that could not be loaded."""

    kind: str  # "task" or "workflow"
    name: str
    entry_point: str
    message: str


@dataclass(frozen=True)
class Registry:
    """Everything that loaded from the entry-point groups, plus what did not."""

    tasks: dict[str, type[Task[Any, Any]]]
    workflows: dict[str, Workflow]
    problems: list[RegistryProblem]


def load_registry(*, include_workflows: bool = True) -> Registry:
    """Load all registered tasks and workflows, collecting failures instead of raising.

    Unlike ``registered_tasks``/``registered_workflows``, one broken or duplicate
    entry point does not hide the others. Entry points are visited in name
    order and the first of a duplicated name wins.
    """
    problems: list[RegistryProblem] = []

    def load_group[T](group: str, kind: str, expected: type[T]) -> dict[str, T]:
        loaded: dict[str, T] = {}
        for entry_point in sorted(entry_points(group=group), key=lambda ep: ep.name):
            name = entry_point.name
            if name in loaded:
                problems.append(
                    RegistryProblem(kind, name, entry_point.value, f"Duplicate {kind} id")
                )
                continue
            try:
                value: Any = entry_point.load()
                if kind == "task":
                    if not (isinstance(value, type) and issubclass(value, Task)):
                        raise TypeError("The entry point is not a Task subclass")
                elif not isinstance(value, expected):
                    raise TypeError("The entry point is not a Workflow")
            except Exception as exc:
                problems.append(RegistryProblem(kind, name, entry_point.value, str(exc)))
                continue
            loaded[name] = value
        return loaded

    tasks = load_group(TASK_ENTRY_POINT_GROUP, "task", type)
    workflows = (
        load_group(WORKFLOW_ENTRY_POINT_GROUP, "workflow", Workflow) if include_workflows else {}
    )
    return Registry(tasks=tasks, workflows=workflows, problems=problems)
