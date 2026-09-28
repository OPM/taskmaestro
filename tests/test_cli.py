"""Tests for the Taskmaestro command-line interface."""

from __future__ import annotations

import json
import sys
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic_core import core_schema

from taskmaestro import ExecutionContext, ObjectModel, Task
from taskmaestro.cli import _dependency_spec, main
from taskmaestro.dependencies import CollectionRef, OutputRef


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


class UnsupportedValue:
    @classmethod
    def __get_pydantic_core_schema__(cls, source: object, handler: object) -> object:
        return core_schema.no_info_plain_validator_function(lambda value: value)


class UnsupportedInput(BaseModel):
    value: UnsupportedValue


class UnsupportedTask(Task[UnsupportedInput, UnsupportedInput]):
    def run(self, input: UnsupportedInput, ctx: ExecutionContext) -> UnsupportedInput:
        return input


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

class Noisy(Task[NumberInput, NumberOutput]):
    name = "noisy"

    def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
        print("task diagnostic")
        return NumberOutput(value=input.value)
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


def test_validate_json_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workflow, input_path = _files(tmp_path)

    assert main(["validate", str(workflow), "--input", str(input_path), "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "valid", "workflow": "cli_test"}
    assert captured.err == ""


def test_run_json_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workflow, input_path = _files(tmp_path)

    assert main(["run", str(workflow), "--input", str(input_path), "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "status": "completed",
        "workflow": "cli_test",
        "result": {"value": 5},
    }
    assert captured.err == ""


def test_run_json_redirects_task_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workflow, input_path = _files(tmp_path, "Noisy")

    assert main(["run", str(workflow), "--input", str(input_path), "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["result"] == {"value": 4}
    assert "task diagnostic" in captured.err


def test_run_json_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workflow, input_path = _files(tmp_path, "Fail")

    assert main(["run", str(workflow), "--input", str(input_path), "--json"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "status": "failed",
        "workflow": "cli_test",
        "failed_task": "fail",
        "error": {
            "code": "task_failed",
            "type": "ValueError",
            "message": "Task failed",
            "task": "fail",
            "field": None,
            "issues": [],
        },
    }
    assert captured.err == ""


def test_run_json_validation_failure_omits_input_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workflow, input_path = _files(tmp_path)
    input_path.write_text("increment:\n  value: secret-value\n", encoding="utf-8")

    assert main(["run", str(workflow), "--input", str(input_path), "--json"]) == 1
    captured = capsys.readouterr()
    assert "secret-value" not in captured.out + captured.err
    result = json.loads(captured.out)
    assert result["failed_task"] == "increment"
    assert result["error"]["type"] == "ValidationError"
    assert result["error"]["field"] == "value"
    assert result["error"]["issues"] == [{"field": "value", "code": "int_parsing"}]


def test_validate_json_redirects_import_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workflow, input_path = _files(tmp_path)
    (tmp_path / "pipeline.py").write_text(
        (tmp_path / "pipeline.py").read_text(encoding="utf-8") + "\nprint('import diagnostic')\n",
        encoding="utf-8",
    )
    sys.modules.pop("pipeline", None)

    assert main(["validate", str(workflow), "--input", str(input_path), "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["status"] == "valid"
    assert "import diagnostic" in captured.err


def test_validate_json_configuration_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workflow, input_path = _files(tmp_path)
    input_path.write_text("{}\n", encoding="utf-8")

    assert main(["validate", str(workflow), "--input", str(input_path), "--json"]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {
        "status": "invalid",
        "error": {
            "code": "configuration_error",
            "type": "ConfigLoadError",
            "message": "Workflow configuration is invalid",
            "task": None,
            "field": None,
            "issues": [],
        },
    }
    assert captured.err == ""


def test_validate_json_missing_config_identifies_task_and_field(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workflow, input_path = _files(tmp_path)
    workflow.write_text(
        "workflow:\n  name: cli_test\n  tasks:\n"
        "    - task: pipeline.Increment\n      config_fields: [value]\n",
        encoding="utf-8",
    )
    input_path.write_text("{}\n", encoding="utf-8")

    assert main(["validate", str(workflow), "--input", str(input_path), "--json"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error"]["task"] == "increment"
    assert result["error"]["field"] == "value"
    assert result["error"]["issues"] == [{"field": "value", "code": "missing"}]


def test_validate_json_schema_failure_omits_input_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workflow, input_path = _files(tmp_path)
    workflow.write_text(
        workflow.read_text(encoding="utf-8") + "runner:\n  timeout_seconds: secret-value\n",
        encoding="utf-8",
    )

    assert main(["validate", str(workflow), "--input", str(input_path), "--json"]) == 2
    captured = capsys.readouterr()
    assert "secret-value" not in captured.out + captured.err
    result = json.loads(captured.out)
    assert result["error"]["field"] == "runner.timeout_seconds"
    assert result["error"]["issues"][0]["code"] == "float_parsing"


def test_run_json_unserializable_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "opaque_pipeline.py").write_text(
        """\
from taskmaestro import EmptyConfig, ExecutionContext, ObjectModel, Task

class Resource:
    pass

class Handle(ObjectModel[Resource]):
    pass

class GetHandle(Task[EmptyConfig, Handle]):
    def run(self, input: EmptyConfig, ctx: ExecutionContext) -> Handle:
        return Handle(value=Resource())
""",
        encoding="utf-8",
    )
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        "workflow:\n  name: opaque\n  tasks:\n    - task: opaque_pipeline.GetHandle\n",
        encoding="utf-8",
    )
    input_path = tmp_path / "input.yaml"
    input_path.write_text("{}\n", encoding="utf-8")

    assert main(["run", str(workflow), "--input", str(input_path), "--json"]) == 1
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == "failed"
    assert result["error"]["code"] == "serialization_error"
    assert result["failed_task"] is None
    assert captured.err == ""


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


def test_workflow_describe_without_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workflow, _input_path = _files(tmp_path)
    original_path = sys.path.copy()

    assert main(["workflow", "describe", str(workflow), "--json"]) == 0
    assert sys.path == original_path
    captured = capsys.readouterr()
    assert captured.err == ""
    description = json.loads(captured.out)
    assert description["workflow"] == "cli_test"
    assert description["result_task"] == "increment"
    assert len(description["tasks"]) == 1
    task = description["tasks"][0]
    assert task["name"] == "increment"
    assert task["python_type"] == "pipeline.Increment"
    assert task["depends_on"] is None
    assert task["required_input_fields"] == ["value"]
    assert task["config_fields"] == []
    assert task["provided_config_fields"] is None
    assert task["input_schema"]["properties"]["value"]["type"] == "integer"
    assert task["output_schema"]["properties"]["value"]["type"] == "integer"
    assert main(["workflow", "describe", str(workflow)]) == 0
    assert json.loads(capsys.readouterr().out) == description


def test_workflow_describe_redirects_import_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workflow, _input_path = _files(tmp_path)
    (tmp_path / "pipeline.py").write_text(
        (tmp_path / "pipeline.py").read_text(encoding="utf-8") + "\nprint('import diagnostic')\n",
        encoding="utf-8",
    )
    sys.modules.pop("pipeline", None)

    assert main(["workflow", "describe", str(workflow), "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["workflow"] == "cli_test"
    assert "import diagnostic" in captured.err


def test_workflow_describe_with_input_does_not_execute(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workflow, input_path = _files(tmp_path, "Fail")
    # Inspections should not instantiate hooks or execute the task's failing run().
    input_path.write_text("fail:\n  value: private-token\n", encoding="utf-8")
    workflow.write_text(
        workflow.read_text(encoding="utf-8") + "runner:\n  hooks:\n    - hook: nonexistent.hook\n",
        encoding="utf-8",
    )

    assert main(["workflow", "describe", str(workflow), "--input", str(input_path), "--json"]) == 0
    description = json.loads(capsys.readouterr().out)
    assert description["tasks"][0]["provided_config_fields"] == ["value"]
    assert "private-token" not in json.dumps(description)


def test_workflow_describe_missing_input_reports_json_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workflow, input_path = _files(tmp_path)
    input_path.write_text("{}\n", encoding="utf-8")

    assert main(["workflow", "describe", str(workflow), "--input", str(input_path), "--json"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "invalid"
    assert result["error"]["code"] == "configuration_error"


def test_workflow_describe_reports_missing_config_fields(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workflow, input_path = _files(tmp_path)
    workflow.write_text(
        "workflow:\n  name: cli_test\n  tasks:\n"
        "    - task: pipeline.Increment\n      config_fields: [value]\n",
        encoding="utf-8",
    )
    input_path.write_text("{}\n", encoding="utf-8")

    assert main(["workflow", "describe", str(workflow), "--input", str(input_path), "--json"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["error"]["task"] == "increment"
    assert result["error"]["issues"] == [{"field": "value", "code": "missing"}]


def test_workflow_dependency_routing_and_positional_collection() -> None:
    """The inspector renders validated tuple and ordered collection references."""
    assert _dependency_spec(("producer", "content")) == {
        "task": "producer",
        "field": "content",
    }
    assert _dependency_spec(
        {
            "items": CollectionRef(
                "positional",
                positional_members=(OutputRef("first"), OutputRef("second", "content")),
            )
        }
    ) == {
        "items": {
            "collect": {
                "kind": "positional",
                "members": [
                    {"task": "first", "field": None},
                    {"task": "second", "field": "content"},
                ],
            }
        }
    }


def test_workflow_describe_fan_in_collections_and_mapping(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workflow = Path(__file__).resolve().parents[1] / "examples/release_pipeline/workflow.yaml"

    # Earlier CLI tests import a different local module also named "pipeline".
    with monkeypatch.context() as patch:
        patch.delitem(sys.modules, "pipeline", raising=False)
        assert main(["workflow", "describe", str(workflow), "--json"]) == 0
        patch.delitem(sys.modules, "pipeline", raising=False)
    tasks = {task["name"]: task for task in json.loads(capsys.readouterr().out)["tasks"]}
    checks = tasks["validate_release"]["depends_on"]["checks"]["collect"]
    assert checks == {
        "kind": "keyed",
        "members": {
            "tests": {"task": "run_tests", "field": None},
            "lint": {"task": "run_lint", "field": None},
            "types": {"task": "check_types", "field": None},
        },
    }
    assert tasks["build_targets"]["map"] == {
        "over": "targets",
        "key_as": "target_name",
        "value_as": "settings",
        "error_mode": "collect_all",
    }
    assert tasks["create_release_manifest"]["depends_on"]["artifacts"] == {
        "task": "build_targets",
        "field": "root",
    }
    assert tasks["build_targets"]["output_schema"]["additionalProperties"] == {
        "$ref": "#/$defs/Artifact"
    }


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


def test_tasks_describe_unsupported_schema_reports_plugin_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    entry = EntryPoint(
        name="unsupported", value="tests.test_cli:UnsupportedTask", group="taskmaestro.tasks"
    )
    monkeypatch.setattr("taskmaestro.discovery.entry_points", lambda *, group: [entry])

    assert main(["tasks", "describe", "unsupported", "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Plugin error: Cannot describe task 'unsupported'" in captured.err
    assert "PlainValidatorFunctionSchema" in captured.err


def test_workflow_describe_unsupported_schema_reports_configuration_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        "workflow:\n  name: unsupported\n  tasks:\n    - task: tests.test_cli.UnsupportedTask\n",
        encoding="utf-8",
    )

    assert main(["workflow", "describe", str(workflow), "--json"]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "configuration_error"
    assert captured.err == ""


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
