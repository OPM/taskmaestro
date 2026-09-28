"""Tests for the Taskmaestro command-line interface."""

from __future__ import annotations

import json
import sys
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest
from pydantic import BaseModel

from taskmaestro import ExecutionContext, ObjectModel, Task
from taskmaestro.cli import main


class ExternalClient:
    """Represents a Python-only value such as a live gRPC client."""


class ClientHandle(ObjectModel[ExternalClient]):
    pass


class OpaqueInput(BaseModel):
    handle: ClientHandle
    amount: int


class OpaqueTask(Task[OpaqueInput, ClientHandle]):
    def run(self, input: OpaqueInput, ctx: ExecutionContext) -> ClientHandle:
        return input.handle


def _files(tmp_path: Path, task: str = "Increment") -> tuple[Path, Path]:
    (tmp_path / "pipeline.py").write_text(
        """\
from pydantic import BaseModel
from taskmaestro import ExecutionContext, Task

class NumberInput(BaseModel):
    value: int

class NumberOutput(BaseModel):
    value: int

class Increment(Task[NumberInput, NumberOutput]):
    name = "increment"

    def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
        return NumberOutput(value=input.value + 1)

class Fail(Task[NumberInput, NumberOutput]):
    name = "fail"

    def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
        raise ValueError("intentional failure")
""",
        encoding="utf-8",
    )
    sys.modules.pop("pipeline", None)
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        f"""\
workflow:
  name: cli_test
  tasks:
    - task: pipeline.{task}
""",
        encoding="utf-8",
    )
    input_path = tmp_path / "input.yaml"
    input_path.write_text(f"{task.lower()}:\n  value: 4\n", encoding="utf-8")
    return workflow, input_path


def test_run_prints_json_result(tmp_path: Path, capsys: object) -> None:
    workflow, input_path = _files(tmp_path)

    status = main(
        [
            "run",
            str(workflow),
            "--input",
            str(input_path),
            "--log-level",
            "DEBUG",
        ]
    )

    assert status == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert '"value": 5' in captured.out


def test_run_reports_failed_job(tmp_path: Path, capsys: object) -> None:
    workflow, input_path = _files(tmp_path, "Fail")

    status = main(["run", str(workflow), "--input", str(input_path)])

    assert status == 1
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "Workflow failed at fail: intentional failure" in captured.err


def test_validate_reports_success(tmp_path: Path, capsys: object) -> None:
    workflow, input_path = _files(tmp_path)
    original_path = sys.path.copy()

    status = main(["validate", str(workflow), "--input", str(input_path)])

    assert status == 0
    assert sys.path == original_path
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "Workflow 'cli_test' is valid" in captured.out


def test_graph_prints_mermaid(tmp_path: Path, capsys: object) -> None:
    workflow, input_path = _files(tmp_path)

    status = main(["graph", str(workflow), "--input", str(input_path)])

    assert status == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "graph TD" in captured.out
    assert 'increment["increment"]' in captured.out


def test_configuration_errors_return_two(tmp_path: Path, capsys: object) -> None:
    missing = tmp_path / "missing.yaml"

    status = main(["validate", str(missing), "--input", str(missing)])

    assert status == 2
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "Configuration error: Cannot read file" in captured.err


def test_error_inside_task_module_is_configuration_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workflow, input_path = _files(tmp_path)
    (tmp_path / "pipeline.py").write_text("import missing_dependency_xyz\n", encoding="utf-8")
    sys.modules.pop("pipeline", None)

    status = main(["validate", str(workflow), "--input", str(input_path)])

    assert status == 2
    err = capsys.readouterr().err
    assert "Configuration error: Error while importing module 'pipeline'" in err
    assert "missing_dependency_xyz" in err


@pytest.mark.parametrize("module", ["taskmaestro", "taskmaestro.cli"])
def test_module_entry_points_run_the_cli(
    module: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``python -m taskmaestro`` and ``python -m taskmaestro.cli`` invoke main()."""
    import runpy
    import warnings

    workflow, input_path = _files(tmp_path)
    monkeypatch.setattr(
        sys, "argv", [module, "validate", str(workflow), "--input", str(input_path)]
    )
    with warnings.catch_warnings():
        # Re-executing an already imported module as __main__ warns; that is expected here.
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
    assert excinfo.value.code == 0
    assert "Workflow 'cli_test' is valid" in capsys.readouterr().out


def test_tasks_list_does_not_import_plugins(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entries = [
        EntryPoint(name="z.broken", value="missing_module:Task", group="taskmaestro.tasks"),
        EntryPoint(
            name="a.valid", value="tests.test_discovery:ExampleTask", group="taskmaestro.tasks"
        ),
    ]
    monkeypatch.setattr("taskmaestro.discovery.entry_points", lambda *, group: entries)

    assert main(["tasks", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"tasks": ["a.valid", "z.broken"]}
    assert main(["tasks", "list"]) == 0
    assert capsys.readouterr().out == "a.valid\nz.broken\n"


def test_tasks_list_rejects_duplicate_identifiers(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entry = EntryPoint(
        name="duplicate", value="tests.test_discovery:ExampleTask", group="taskmaestro.tasks"
    )
    monkeypatch.setattr("taskmaestro.discovery.entry_points", lambda *, group: [entry, entry])

    assert main(["tasks", "list", "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Plugin error: Multiple entry points named 'duplicate'" in captured.err


def test_tasks_describe_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entries = [
        EntryPoint(
            name="example.increment",
            value="tests.test_discovery:ExampleTask",
            group="taskmaestro.tasks",
        )
    ]
    monkeypatch.setattr("taskmaestro.discovery.entry_points", lambda *, group: entries)

    assert main(["tasks", "describe", "example.increment", "--json"]) == 0
    description = json.loads(capsys.readouterr().out)
    assert description["identifier"] == "example.increment"
    assert description["name"] == "ExampleTask"
    assert description["timeout_seconds"] is None
    assert description["input_schema"]["properties"]["value"]["type"] == "integer"
    assert description["input_schema"]["required"] == ["value"]
    assert description["output_schema"]["properties"]["value"]["type"] == "integer"
    assert main(["tasks", "describe", "example.increment"]) == 0
    assert json.loads(capsys.readouterr().out) == description


def test_tasks_describe_runtime_only_objects(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entries = [
        EntryPoint(
            name="example.opaque", value="tests.test_cli:OpaqueTask", group="taskmaestro.tasks"
        )
    ]
    monkeypatch.setattr("taskmaestro.discovery.entry_points", lambda *, group: entries)

    assert main(["tasks", "describe", "example.opaque", "--json"]) == 0
    description = json.loads(capsys.readouterr().out)
    input_schema = description["input_schema"]
    handle_schema = input_schema["$defs"]["ClientHandle"]["properties"]["value"]
    assert handle_schema["not"] == {}
    assert handle_schema["x-taskmaestro-opaque"] is True
    assert handle_schema["x-taskmaestro-python-type"] == "tests.test_cli.ExternalClient"
    assert input_schema["properties"]["amount"]["type"] == "integer"
    output_schema = description["output_schema"]
    assert output_schema["properties"]["value"]["x-taskmaestro-opaque"] is True


def test_tasks_describe_unknown_plugin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("taskmaestro.discovery.entry_points", lambda *, group: [])

    assert main(["tasks", "describe", "missing", "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Plugin error: No task entry point named 'missing'" in captured.err


def test_tasks_describe_broken_plugin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entries = [
        EntryPoint(
            name="broken", value="missing_taskmaestro_plugin:Task", group="taskmaestro.tasks"
        )
    ]
    monkeypatch.setattr("taskmaestro.discovery.entry_points", lambda *, group: entries)

    assert main(["tasks", "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"tasks": ["broken"]}
    assert main(["tasks", "describe", "broken", "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Plugin error: Cannot load task entry point 'broken'" in captured.err


def test_python_dash_m_exit_code(tmp_path: Path) -> None:
    """The real interpreter invocation propagates main()'s exit status."""
    import subprocess

    workflow, input_path = _files(tmp_path, "Fail")
    completed = subprocess.run(
        [sys.executable, "-m", "taskmaestro", "run", str(workflow), "--input", str(input_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "Workflow failed at fail: intentional failure" in completed.stderr
