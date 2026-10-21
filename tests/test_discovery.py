"""Tests for installed task and workflow discovery."""

from __future__ import annotations

from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from taskmaestro import (
    ConfigLoadError,
    ExecutionContext,
    PluginLoadError,
    Task,
    Workflow,
    get_registered_task,
    get_registered_workflow,
    load_workflow_from_yaml,
    registered_task_names,
    registered_tasks,
    registered_workflow_names,
    registered_workflows,
)
from taskmaestro.discovery import load_registry


class Input(BaseModel):
    value: int


class Output(BaseModel):
    value: int


class ExampleTask(Task[Input, Output]):
    def run(self, input: Input, ctx: ExecutionContext) -> Output:
        return Output(value=input.value + 1)


example_workflow = Workflow("example", [ExampleTask])


def _entry_point(name: str, value: str, group: str) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=group)


@pytest.fixture
def plugin_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = {
        "taskmaestro.tasks": [
            _entry_point(
                "example.increment", "tests.test_discovery:ExampleTask", "taskmaestro.tasks"
            )
        ],
        "taskmaestro.workflows": [
            _entry_point(
                "example.workflow",
                "tests.test_discovery:example_workflow",
                "taskmaestro.workflows",
            )
        ],
    }
    monkeypatch.setattr(
        "taskmaestro.discovery.entry_points", lambda *, group: entries.get(group, [])
    )


def test_discovers_registered_plugins(plugin_entry_points: None) -> None:
    assert registered_task_names() == {"example.increment"}
    assert registered_workflow_names() == {"example.workflow"}
    assert registered_tasks() == {"example.increment": ExampleTask}
    assert registered_workflows() == {"example.workflow": example_workflow}
    assert get_registered_task("example.increment") is ExampleTask
    assert get_registered_workflow("example.workflow") is example_workflow


def test_registered_task_can_be_used_in_yaml(plugin_entry_points: None, tmp_path: Path) -> None:
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        "workflow:\n  name: entry_point_workflow\n  tasks:\n    - task: example.increment\n"
    )
    input_path = tmp_path / "input.yaml"
    input_path.write_text("ExampleTask:\n  value: 4\n")

    result = load_workflow_from_yaml(workflow_path, input_path).run()

    assert result.result == Output(value=5)


def test_rejects_wrong_plugin_type(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _entry_point("invalid", "tests.test_discovery:Input", "taskmaestro.tasks")
    monkeypatch.setattr("taskmaestro.discovery.entry_points", lambda *, group: [entry])

    with pytest.raises(PluginLoadError, match="Task subclass"):
        registered_tasks()


def test_rejects_duplicate_names(monkeypatch: pytest.MonkeyPatch) -> None:
    entries: list[Any] = [
        _entry_point("duplicate", "tests.test_discovery:ExampleTask", "taskmaestro.tasks"),
        _entry_point("duplicate", "tests.test_discovery:ExampleTask", "taskmaestro.tasks"),
    ]
    monkeypatch.setattr("taskmaestro.discovery.entry_points", lambda *, group: entries)

    with pytest.raises(PluginLoadError, match="Multiple entry points"):
        registered_tasks()


def test_load_registry_collects_problems_and_keeps_good_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries: list[Any] = [
        _entry_point("good", "tests.test_discovery:ExampleTask", "taskmaestro.tasks"),
        _entry_point("good", "tests.test_discovery:ExampleTask", "taskmaestro.tasks"),
        _entry_point("broken", "tests.test_discovery:missing", "taskmaestro.tasks"),
    ]
    monkeypatch.setattr(
        "taskmaestro.discovery.entry_points",
        lambda *, group: entries if group == "taskmaestro.tasks" else [],
    )

    registry = load_registry()

    assert set(registry.tasks) == {"good"}
    assert registry.workflows == {}
    assert [(p.kind, p.name) for p in registry.problems] == [("task", "broken"), ("task", "good")]
    assert registry.problems[1].message == "Duplicate task id"


def test_load_registry_checks_entry_point_types(monkeypatch: pytest.MonkeyPatch) -> None:
    entries: dict[str, list[Any]] = {
        "taskmaestro.tasks": [
            _entry_point("not_a_task", "tests.test_discovery:Input", "taskmaestro.tasks")
        ],
        "taskmaestro.workflows": [
            _entry_point("good", "tests.test_discovery:example_workflow", "taskmaestro.workflows"),
            _entry_point("wrong", "tests.test_discovery:ExampleTask", "taskmaestro.workflows"),
        ],
    }
    monkeypatch.setattr(
        "taskmaestro.discovery.entry_points", lambda *, group: entries.get(group, [])
    )

    registry = load_registry()

    assert registry.tasks == {}
    assert registry.workflows == {"good": example_workflow}
    assert [(p.kind, p.name, p.message) for p in registry.problems] == [
        ("task", "not_a_task", "The entry point is not a Task subclass"),
        ("workflow", "wrong", "The entry point is not a Workflow"),
    ]
    assert load_registry(include_workflows=False).workflows == {}


def test_reports_entry_point_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _entry_point("broken", "tests.test_discovery:missing", "taskmaestro.tasks")
    monkeypatch.setattr("taskmaestro.discovery.entry_points", lambda *, group: [entry])

    with pytest.raises(PluginLoadError, match="Cannot load task entry point 'broken'") as exc_info:
        registered_tasks()

    assert isinstance(exc_info.value.__cause__, AttributeError)


def test_rejects_wrong_workflow_plugin_type(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = _entry_point("invalid", "tests.test_discovery:ExampleTask", "taskmaestro.workflows")
    monkeypatch.setattr("taskmaestro.discovery.entry_points", lambda *, group: [entry])

    with pytest.raises(PluginLoadError, match="must resolve to a Workflow"):
        registered_workflows()


def test_reports_missing_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("taskmaestro.discovery.entry_points", lambda *, group: [])

    with pytest.raises(PluginLoadError, match="No task entry point named 'missing'"):
        get_registered_task("missing")
    with pytest.raises(PluginLoadError, match="No workflow entry point named 'missing'"):
        get_registered_workflow("missing")


def test_yaml_wraps_invalid_task_plugin_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entry = _entry_point("invalid", "tests.test_discovery:Input", "taskmaestro.tasks")
    monkeypatch.setattr("taskmaestro.discovery.entry_points", lambda *, group: [entry])
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text("workflow:\n  name: invalid\n  tasks:\n    - task: invalid\n")
    input_path = tmp_path / "input.yaml"
    input_path.write_text("{}\n")

    with pytest.raises(ConfigLoadError, match="must resolve to a Task subclass"):
        load_workflow_from_yaml(workflow_path, input_path)
