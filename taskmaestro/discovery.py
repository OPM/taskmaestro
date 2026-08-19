"""Discovery of tasks and workflows published by installed distributions."""

from __future__ import annotations

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
