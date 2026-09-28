"""Keep the agent quickstart's CLI commands and expected output working."""

import json
import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "agent_quickstart"
WORKFLOW = EXAMPLE / "workflow.yaml"
INPUT = EXAMPLE / "input.yaml"


def _cli(command: str, input_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "taskmaestro", command, str(WORKFLOW), "--input", str(input_path)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_agent_quickstart(tmp_path: Path) -> None:
    validated = _cli("validate", INPUT)
    assert validated.returncode == 0, validated.stderr
    assert validated.stdout.strip() == "Workflow 'agent_quickstart' is valid"

    graph = _cli("graph", INPUT)
    assert graph.returncode == 0, graph.stderr
    assert "add_one -->|Number| double" in graph.stdout

    run = _cli("run", INPUT)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == {"value": 12}

    bad_input = tmp_path / "bad-input.yaml"
    bad_input.write_text("{}\n", encoding="utf-8")
    invalid = _cli("validate", bad_input)
    assert invalid.returncode == 2
    assert "Task 'add_one' is missing configuration fields ['value']" in invalid.stderr
