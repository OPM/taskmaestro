"""Tests for the built-in ``taskmaestro.python`` and ``taskmaestro.command`` tasks."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from taskmaestro import (
    Command,
    CommandInput,
    ConfigLoadError,
    EmptyConfig,
    ExecutionContext,
    JobStatus,
    ObjectModel,
    PythonScript,
    PythonScriptInput,
    ScriptError,
    WorkflowBuilder,
    load_workflow_from_yaml,
    registered_task_names,
)
from taskmaestro.scripts import _kill_process_tree

ECHO_SCRIPT = """\
import json, os, sys
data = None
if "TASKMAESTRO_INPUT" in os.environ:
    with open(os.environ["TASKMAESTRO_INPUT"]) as f:
        data = json.load(f)
with open(os.environ["TASKMAESTRO_OUTPUT"], "w") as f:
    json.dump({"args": sys.argv[1:], "inputs": data, "cwd": os.getcwd(),
               "flag": os.environ.get("FLAG")}, f)
print("done")
"""


@pytest.fixture
def ctx(tmp_path: Path) -> ExecutionContext:
    return ExecutionContext(scratch_dir=tmp_path / "scratch")


def _script(directory: Path, text: str, name: str = "step.py") -> Path:
    path = directory / name
    path.write_text(textwrap.dedent(text))
    return path


class Point(BaseModel):
    x: int


def test_python_script_exchanges_json_and_captures_output(
    tmp_path: Path, ctx: ExecutionContext
) -> None:
    script = _script(tmp_path, ECHO_SCRIPT)
    workdir = tmp_path / "work"
    workdir.mkdir()

    output = PythonScript().run(
        PythonScriptInput(
            script=script,
            args=["a", "b"],
            inputs={"point": Point(x=1), "n": 2},
            env={"FLAG": "on"},
            cwd=workdir,
        ),
        ctx,
    )

    assert output.result == {
        "args": ["a", "b"],
        "inputs": {"point": {"x": 1}, "n": 2},
        "cwd": str(workdir),
        "flag": "on",
    }
    assert (output.returncode, output.stdout, output.stderr) == (0, "done\n", "")
    assert list(ctx.scratch_dir.iterdir()) == []  # Exchange directory removed.


def test_relative_script_resolves_against_the_process_cwd(
    tmp_path: Path, ctx: ExecutionContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    _script(tmp_path, ECHO_SCRIPT)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(tmp_path)

    output = PythonScript().run(PythonScriptInput(script=Path("step.py"), cwd=elsewhere), ctx)

    assert output.result["inputs"] is None
    assert output.result["cwd"] == str(elsewhere)


def test_missing_output_file_gives_null_result(tmp_path: Path, ctx: ExecutionContext) -> None:
    script = _script(tmp_path, "print('no result')\n")

    output = PythonScript().run(PythonScriptInput(script=script), ctx)

    assert output.result is None
    assert output.stdout == "no result\n"


def test_empty_output_file_gives_null_result(tmp_path: Path, ctx: ExecutionContext) -> None:
    script = _script(
        tmp_path, "import os\nopen(os.environ['TASKMAESTRO_OUTPUT'], 'w').write(' \\n')\n"
    )

    output = PythonScript().run(PythonScriptInput(script=script), ctx)

    assert output.result is None


def test_output_file_is_read_as_utf8(tmp_path: Path, ctx: ExecutionContext) -> None:
    script = _script(
        tmp_path,
        "import os\n"
        "open(os.environ['TASKMAESTRO_OUTPUT'], 'wb').write('\"Ærøskøbing\"'.encode('utf-8'))\n",
    )

    output = PythonScript().run(PythonScriptInput(script=script), ctx)

    assert output.result == "Ærøskøbing"


@pytest.mark.parametrize("executable", ["", "host"])
def test_python_script_fails_without_a_python_interpreter(
    tmp_path: Path, ctx: ExecutionContext, monkeypatch: pytest.MonkeyPatch, executable: str
) -> None:
    # Embedded Python: sys.executable is empty or the host application.
    if executable:
        host = tmp_path / "ResInsight"
        host.write_text("")
        executable = str(host)
    monkeypatch.setattr(sys, "executable", executable)
    script = _script(tmp_path, "print('never runs')\n")

    with pytest.raises(ScriptError, match="No Python interpreter") as info:
        PythonScript().run(PythonScriptInput(script=script), ctx)
    assert info.value.returncode is None


def test_nonzero_exit_fails_with_exit_code_and_stderr(
    tmp_path: Path, ctx: ExecutionContext
) -> None:
    script = _script(tmp_path, "import sys\nprint('bad things', file=sys.stderr)\nsys.exit(3)\n")

    with pytest.raises(ScriptError, match="failed with exit code 3: bad things") as info:
        PythonScript().run(PythonScriptInput(script=script), ctx)

    assert info.value.returncode == 3
    assert info.value.stderr == "bad things\n"


def test_nonzero_exit_without_check_is_returned(tmp_path: Path, ctx: ExecutionContext) -> None:
    script = _script(tmp_path, "import sys\nsys.exit(3)\n")

    output = PythonScript().run(PythonScriptInput(script=script, check=False), ctx)

    assert output.returncode == 3


def test_missing_script_fails_before_running(tmp_path: Path, ctx: ExecutionContext) -> None:
    with pytest.raises(ScriptError, match=r"Script not found: .*missing\.py"):
        PythonScript().run(PythonScriptInput(script=tmp_path / "missing.py"), ctx)


def test_invalid_output_json_fails(tmp_path: Path, ctx: ExecutionContext) -> None:
    script = _script(
        tmp_path, "import os\nopen(os.environ['TASKMAESTRO_OUTPUT'], 'w').write('{oops')\n"
    )

    with pytest.raises(ScriptError, match=r"wrote invalid JSON to \$TASKMAESTRO_OUTPUT"):
        PythonScript().run(PythonScriptInput(script=script), ctx)


def test_runtime_only_inputs_cannot_be_sent_to_a_process(
    tmp_path: Path, ctx: ExecutionContext
) -> None:
    script = _script(tmp_path, ECHO_SCRIPT)
    opaque = ObjectModel[object](value=object())

    with pytest.raises(ScriptError, match="cannot be converted to JSON"):
        PythonScript().run(PythonScriptInput(script=script, inputs={"case": opaque}), ctx)


def test_timeout_kills_the_process_and_keeps_partial_output(
    tmp_path: Path, ctx: ExecutionContext
) -> None:
    script = _script(tmp_path, "import time\nprint('working', flush=True)\ntime.sleep(30)\n")

    started = time.monotonic()
    with pytest.raises(ScriptError, match=r"timed out after 1\.0s") as info:
        PythonScript().run(PythonScriptInput(script=script, timeout=1.0), ctx)
    assert time.monotonic() - started < 10
    assert info.value.stdout == "working\n"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
def test_command_string_runs_through_the_shell(ctx: ExecutionContext) -> None:
    output = Command().run(
        CommandInput(
            command=(
                'printf "%s" "$GREETING" | tr a-z A-Z; printf \'[1, 2]\' > "$TASKMAESTRO_OUTPUT"'
            ),
            env={"GREETING": "hello"},
        ),
        ctx,
    )

    assert output.stdout == "HELLO"
    assert output.result == [1, 2]


def test_command_list_runs_without_a_shell(ctx: ExecutionContext) -> None:
    output = Command().run(
        CommandInput(command=[sys.executable, "-c", "import sys; print(sys.argv[1])", "$HOME"]),
        ctx,
    )

    assert output.stdout == "$HOME\n"


@pytest.mark.parametrize(
    ("body", "settings"),
    [
        ("import sys; sys.exit(3)", {}),
        ("import os; open(os.environ['TASKMAESTRO_OUTPUT'], 'w').write('{')", {}),
        ("import time; time.sleep(30)", {"timeout": 0.5}),
        ("pass", {"inputs": {"case": ObjectModel[object](value=object())}}),
    ],
    ids=["nonzero-exit", "invalid-json", "timeout", "unserializable-inputs"],
)
def test_exchange_directory_is_removed_on_failure(
    tmp_path: Path, ctx: ExecutionContext, body: str, settings: dict[str, Any]
) -> None:
    script = _script(tmp_path, body + "\n")

    with pytest.raises(ScriptError):
        PythonScript().run(PythonScriptInput(script=script, **settings), ctx)

    assert list(ctx.scratch_dir.iterdir()) == []


class _FakeProcess:
    """Stands in for ``Popen`` in ``_kill_process_tree`` tests."""

    pid = 4242

    def __init__(self) -> None:
        self.calls: list[str] = []

    def kill(self) -> None:
        self.calls.append("kill")

    def wait(self) -> int:
        self.calls.append("wait")
        return -9


def _patch_killpg(monkeypatch: pytest.MonkeyPatch, error: type[OSError] | None) -> list[int]:
    groups: list[int] = []

    def killpg(pgid: int, sig: int) -> None:
        groups.append(pgid)
        if error is not None:
            raise error()

    monkeypatch.setattr(os, "killpg", killpg, raising=False)
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    return groups


@pytest.mark.parametrize(
    ("error", "expected_calls"),
    [
        (None, ["wait"]),
        (ProcessLookupError, ["wait"]),  # Group already gone: nothing left to kill.
        (PermissionError, ["kill", "wait"]),  # Fall back to the direct child.
    ],
)
def test_kill_process_tree_on_posix(
    monkeypatch: pytest.MonkeyPatch, error: type[OSError] | None, expected_calls: list[str]
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    groups = _patch_killpg(monkeypatch, error)
    process = _FakeProcess()

    _kill_process_tree(process)  # type: ignore[arg-type]

    assert groups == [process.pid]
    assert process.calls == expected_calls


def test_kill_process_tree_on_windows_kills_the_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    groups = _patch_killpg(monkeypatch, None)
    process = _FakeProcess()

    _kill_process_tree(process)  # type: ignore[arg-type]

    assert groups == []
    assert process.calls == ["kill", "wait"]


def _assert_process_gone(pid_file: Path) -> None:
    pid = int(pid_file.read_text())
    for _ in range(50):  # The killed process may need a moment to be reaped.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    pytest.fail(f"Process {pid} survived the timeout")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_timeout_kills_processes_started_by_the_shell(
    tmp_path: Path, ctx: ExecutionContext
) -> None:
    pid_file = tmp_path / "pid"
    # No ``exec``: sleep is a child of the shell, not the shell itself.
    command = f'sleep 30 & echo $! > "{pid_file}"; wait'

    started = time.monotonic()
    with pytest.raises(ScriptError, match="timed out"):
        Command().run(CommandInput(command=command, timeout=1.0), ctx)

    assert time.monotonic() - started < 10
    _assert_process_gone(pid_file)


def test_command_that_cannot_start_fails(ctx: ExecutionContext) -> None:
    with pytest.raises(ScriptError, match="could not be started"):
        Command().run(CommandInput(command=["/nonexistent/taskmaestro-tool"]), ctx)


def test_empty_command_fails(ctx: ExecutionContext) -> None:
    with pytest.raises(ScriptError, match="Command is empty"):
        Command().run(CommandInput(command=[]), ctx)


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="Runner timeouts need SIGALRM")
def test_runner_timeout_kills_a_running_process(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    builder = WorkflowBuilder("slow")
    builder.task(Command, config_fields=["command"])
    command = ["sh", "-c", f'echo $$ > "{pid_file}"; exec sleep 30']

    started = time.monotonic()
    job = builder.build().run(
        EmptyConfig(),
        task_config={"command": {"command": command}},
        ctx=ExecutionContext(scratch_dir=tmp_path),
        timeout_seconds=1,
    )

    assert job.status == JobStatus.FAILED
    assert time.monotonic() - started < 10
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="Runner timeouts need SIGALRM")
def test_runner_timeout_kills_processes_started_by_the_shell(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    builder = WorkflowBuilder("slow")
    builder.task(Command, config_fields=["command"])
    command = f'sleep 30 & echo $! > "{pid_file}"; wait'

    started = time.monotonic()
    job = builder.build().run(
        EmptyConfig(),
        task_config={"command": {"command": command}},
        ctx=ExecutionContext(scratch_dir=tmp_path),
        timeout_seconds=1,
    )

    assert job.status == JobStatus.FAILED
    assert time.monotonic() - started < 10
    _assert_process_gone(pid_file)


def test_script_tasks_are_registered() -> None:
    assert {"taskmaestro.python", "taskmaestro.command"} <= registered_task_names()


def _yaml_files(tmp_path: Path, workflow: str, inputs: str = "{}\n") -> tuple[Path, Path]:
    workflow_path = tmp_path / "workflow.yaml"
    input_path = tmp_path / "input.yaml"
    workflow_path.write_text(textwrap.dedent(workflow))
    input_path.write_text(inputs)
    return workflow_path, input_path


def test_yaml_workflow_connects_script_results_with_collect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _script(tmp_path, ECHO_SCRIPT)
    monkeypatch.chdir(tmp_path)
    workflow_path, input_path = _yaml_files(
        tmp_path,
        """\
        workflow:
          name: scripts
          tasks:
            - task: taskmaestro.python
              name: first
            - task: taskmaestro.command
              name: second
              depends_on:
                inputs:
                  collect: {args: [first, result], code: [first, returncode]}
        """,
        f"first: {{script: step.py, args: [two]}}\n"
        f"second: {{command: [{json.dumps(sys.executable)}, step.py]}}\n",
    )

    job = load_workflow_from_yaml(workflow_path, input_path).run()

    assert job.status == JobStatus.COMPLETED, job.error
    assert job.result is not None
    assert job.result.result["inputs"] == {
        "args": {"args": ["two"], "inputs": None, "cwd": str(tmp_path), "flag": None},
        "code": 0,
    }


def test_whole_output_cannot_be_connected_to_inputs(tmp_path: Path) -> None:
    workflow_path, input_path = _yaml_files(
        tmp_path,
        """\
        workflow:
          name: scripts
          tasks:
            - task: taskmaestro.command
              name: first
            - task: taskmaestro.command
              name: second
              depends_on: {inputs: first}
        """,
        'first: {command: "true"}\nsecond: {command: "true"}\n',
    )

    with pytest.raises(ConfigLoadError, match=r"expects dict\[str, Any\]"):
        load_workflow_from_yaml(workflow_path, input_path)


@pytest.mark.skipif(sys.platform == "win32", reason="Example uses sh and wc")
def test_scripts_example_runs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    input_path = tmp_path / "input.yaml"
    example_input = (root / "examples/scripts/input.yaml").read_text()
    input_path.write_text(
        example_input.replace("/tmp/taskmaestro-script-tasks", json.dumps(str(tmp_path)))
    )

    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "taskmaestro",
            "run",
            "examples/scripts/workflow.yaml",
            "--input",
            str(input_path),
            "--json",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode == 0, run.stderr
    report = tmp_path / "word_count.json"
    assert json.loads(run.stdout)["result"]["result"] == {"report": str(report)}
    saved = json.loads(report.read_text())
    assert saved["counts"]["words"] == 32
    assert saved["wc_output"].split()[0] == "172"
