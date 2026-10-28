"""Built-in tasks that run a Python script or a command in a subprocess.

Registered as the ``taskmaestro.python`` and ``taskmaestro.command`` task plugins.
They make one-off workflow steps possible without writing a Task class.

Data exchange with the process uses JSON files in a fresh directory under the
run's scratch directory:

- ``TASKMAESTRO_INPUT``: path to a JSON object holding ``inputs`` (set only when
  ``inputs`` is not empty).
- ``TASKMAESTRO_OUTPUT``: path where the process may write a JSON value, which
  becomes ``ScriptOutput.result``.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_core import PydanticSerializationError, to_json

from taskmaestro.context import ExecutionContext
from taskmaestro.exceptions import ScriptError
from taskmaestro.task import Task

INPUT_ENV = "TASKMAESTRO_INPUT"
OUTPUT_ENV = "TASKMAESTRO_OUTPUT"
_STDERR_TAIL_LINES = 20
# Executable name prefixes accepted as a Python interpreter (python3, python3.14t,
# pythonw.exe, pypy3, graalpy, ...).
_PYTHON_NAMES = ("python", "pypy", "graalpy")


class ScriptOutput(BaseModel):
    """Output of :class:`PythonScript` and :class:`Command`."""

    result: Any = Field(default=None, description=f"JSON value written to ${OUTPUT_ENV}, or null")
    returncode: int = Field(default=0, description="Exit code of the process")
    stdout: str = Field(default="", description="Captured standard output")
    stderr: str = Field(default="", description="Captured standard error")


class _ProcessInput(BaseModel):
    """Settings shared by script and command tasks."""

    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            f"Upstream data, connected with a keyed collect; written as JSON to ${INPUT_ENV}"
        ),
    )
    env: dict[str, str] = Field(
        default_factory=dict, description="Extra environment variables for the process"
    )
    cwd: Path | None = Field(
        default=None, description="Working directory of the process (default: current directory)"
    )
    timeout: float | None = Field(
        default=None, gt=0, description="Kill the process after this many seconds"
    )
    check: bool = Field(default=True, description="Fail the task on a nonzero exit code")


class PythonScriptInput(_ProcessInput):
    """Input of :class:`PythonScript`."""

    script: Path = Field(description="Python file to run")
    args: list[str] = Field(default_factory=list, description="Command-line arguments")


class CommandInput(_ProcessInput):
    """Input of :class:`Command`."""

    command: str | list[str] = Field(
        description=(
            "A string runs through the shell; a list runs the executable directly, without a shell"
        )
    )


class PythonScript(Task[PythonScriptInput, ScriptOutput]):
    """Run a Python file with the current interpreter in a subprocess."""

    name = "python"

    def run(self, input: PythonScriptInput, ctx: ExecutionContext) -> ScriptOutput:
        # Resolve before the process changes directory to ``cwd``.
        script = input.script.expanduser().absolute()
        if not script.is_file():
            raise ScriptError(f"Script not found: {script}")
        return run_process(
            [_python_interpreter(), str(script), *input.args],
            settings=input,
            ctx=ctx,
            label=f"Script '{script.name}'",
        )


class Command(Task[CommandInput, ScriptOutput]):
    """Run a shell command, shell script or executable in a subprocess."""

    name = "command"

    def run(self, input: CommandInput, ctx: ExecutionContext) -> ScriptOutput:
        if not input.command:
            raise ScriptError("Command is empty")
        shown = input.command if isinstance(input.command, str) else " ".join(input.command)
        return run_process(input.command, settings=input, ctx=ctx, label=f"Command '{shown}'")


def run_process(
    command: str | list[str],
    *,
    settings: _ProcessInput,
    ctx: ExecutionContext,
    label: str,
) -> ScriptOutput:
    """Run ``command`` with the data-exchange protocol and return its output.

    A string runs through the shell. Raises :class:`ScriptError` when the
    process cannot be run, times out, exits nonzero with ``check`` set, or
    writes invalid JSON to ``TASKMAESTRO_OUTPUT``. The exchange directory is
    deleted before returning, also on failure: everything it held is returned in
    :class:`ScriptOutput` or :class:`ScriptError`, and inputs may be sensitive.
    """
    ctx.scratch_dir.mkdir(parents=True, exist_ok=True)
    exchange_dir = Path(tempfile.mkdtemp(prefix="script-", dir=ctx.scratch_dir))
    try:
        return _run_in_exchange_dir(command, exchange_dir, settings=settings, ctx=ctx, label=label)
    finally:
        shutil.rmtree(exchange_dir, ignore_errors=True)


def _run_in_exchange_dir(
    command: str | list[str],
    exchange_dir: Path,
    *,
    settings: _ProcessInput,
    ctx: ExecutionContext,
    label: str,
) -> ScriptOutput:
    output_path = exchange_dir / "output.json"
    env = {**os.environ, **settings.env, OUTPUT_ENV: str(output_path)}
    env.pop(INPUT_ENV, None)
    if settings.inputs:
        input_path = exchange_dir / "input.json"
        try:
            input_path.write_bytes(to_json(settings.inputs))
        except PydanticSerializationError as exc:
            raise ScriptError(
                f"{label}: inputs cannot be converted to JSON ({exc}). Runtime-only "
                "objects cannot be passed to a subprocess; use a regular Task instead."
            ) from exc
        env[INPUT_ENV] = str(input_path)

    ctx.logger.debug("%s: running in %s", label, exchange_dir)
    completed = _run(command, settings=settings, env=env, label=label)

    if settings.check and completed.returncode != 0:
        tail = "\n".join(completed.stderr.splitlines()[-_STDERR_TAIL_LINES:])
        message = f"{label} failed with exit code {completed.returncode}"
        raise ScriptError(
            f"{message}: {tail}" if tail else message,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    return ScriptOutput(
        result=_read_result(output_path, label, completed),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _read_result(
    output_path: Path, label: str, completed: subprocess.CompletedProcess[str]
) -> Any:
    """Return the JSON value written to the output file, or ``None`` if none was written."""
    try:
        text = output_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ScriptError(
            f"{label} wrote invalid JSON to ${OUTPUT_ENV}: {exc}",
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        ) from exc


def _run(
    command: str | list[str],
    *,
    settings: _ProcessInput,
    env: dict[str, str],
    label: str,
) -> subprocess.CompletedProcess[str]:
    """Run ``command`` to completion, killing its whole process tree on timeout or error.

    On POSIX the process leads a new session, so a timeout, a runner timeout
    (``SIGALRM``) or ``KeyboardInterrupt`` kills every process it started, not
    just the shell. ``subprocess.run`` would kill only the direct child.
    """
    try:
        process = subprocess.Popen(
            command,
            shell=isinstance(command, str),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=settings.cwd,
            start_new_session=sys.platform != "win32",
        )
    except OSError as exc:
        raise ScriptError(f"{label} could not be started: {exc}") from exc

    with process:  # Closes the pipes and reaps the process on exit.
        try:
            stdout, stderr = process.communicate(timeout=settings.timeout)
        except subprocess.TimeoutExpired as exc:
            _kill_process_tree(process)
            # ``communicate`` stored the output read so far on the exception
            # (POSIX). Not calling it again avoids blocking on pipes held open
            # by a process that escaped the process group.
            raise ScriptError(
                f"{label} timed out after {settings.timeout}s",
                stdout=_text(exc.stdout),
                stderr=_text(exc.stderr),
            ) from exc
        except BaseException:
            _kill_process_tree(process)
            raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    """Kill ``process`` and, on POSIX, every process in its process group."""
    if sys.platform != "win32":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:  # Group already gone.
            pass
        except PermissionError:  # A group member changed user; kill what we can.
            process.kill()
    else:
        process.kill()
    process.wait()


def _python_interpreter() -> str:
    """Return the interpreter for script tasks, or raise if there is no usable one.

    When Python is embedded in another application, ``sys.executable`` can be
    empty or point to the host application rather than a Python interpreter.
    """
    executable = sys.executable
    name = Path(executable).name.lower()
    if not executable or not Path(executable).is_file() or not name.startswith(_PYTHON_NAMES):
        raise ScriptError(
            f"No Python interpreter available to run the script (sys.executable is "
            f"{executable!r}). This happens when Python is embedded in another "
            "application. Use taskmaestro.command with an explicit interpreter instead, "
            "e.g. command: [/path/to/python, script.py]."
        )
    return executable


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value
