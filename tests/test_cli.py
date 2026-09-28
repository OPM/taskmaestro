"""Tests for the Taskmaestro command-line interface."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from taskmaestro import ExecutionContext, Task
from taskmaestro.cli import main

THIS_MODULE = "tests.test_cli"


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


def _files(tmp_path: Path, task: str = "Increment") -> tuple[Path, Path]:
    workflow = tmp_path / "workflow.yaml"
    workflow.write_text(
        f"""\
workflow:
  name: cli_test
  input_mode: flat
  tasks:
    - task: {THIS_MODULE}.{task}
""",
        encoding="utf-8",
    )
    input_path = tmp_path / "input.yaml"
    input_path.write_text("value: 4\n", encoding="utf-8")
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

    status = main(["validate", str(workflow), "--input", str(input_path)])

    assert status == 0
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
