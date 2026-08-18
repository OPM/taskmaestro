"""Tests for installed task and workflow discovery."""

from __future__ import annotations

from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from taskmaestro import (
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


def test_registered_task_can_be_used_in_yaml(
    plugin_entry_points: None, tmp_path: Path
) -> None:
    workflow_path = tmp_path / "workflow.yaml"
    workflow_path.write_text(
        "workflow:\n"
        "  name: entry_point_workflow\n"
        "  tasks:\n"
        "    - task: example.increment\n"
    )
    input_path = tmp_path / "input.yaml"
    input_path.write_text("value: 4\n")

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
