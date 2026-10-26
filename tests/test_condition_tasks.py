"""Tests for the ready-made condition tasks in ``taskmaestro.condition_tasks``."""

from __future__ import annotations

import tomllib
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from taskmaestro import ExecutionContext, Job, JobConfiguration, Runner, Workflow, first_of
from taskmaestro.condition_tasks import (
    Compare,
    CompareInput,
    IsTrue,
    IsTrueInput,
    PathExists,
    PathExistsInput,
)
from taskmaestro.conditions import is_switch_task
from tests.conftest import AddOne, Double, NumberInput, NumberOutput
from tests.test_conditions import Join, Negate


@pytest.mark.parametrize(
    ("value", "op", "threshold", "expected"),
    [
        (3, ">=", 3, True),
        (2.5, ">=", 3, False),
        (2, "<", 3, True),
        (3, "<=", 3, True),
        (3, "==", 3.0, True),
        (3, "!=", 3, False),
        (4, ">", 3, True),
    ],
)
def test_compare(value: float, op: Any, threshold: float, expected: bool) -> None:
    input = CompareInput(value=value, op=op, threshold=threshold)
    assert Compare().decide(input, ExecutionContext()) is expected


def test_compare_defaults_to_greater_or_equal() -> None:
    assert CompareInput(value=1, threshold=0).op == ">="


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (1, True), ("x", True), (0, False), ("", False), (None, False), ([], False)],
)
def test_is_true(value: Any, expected: bool) -> None:
    assert IsTrue().decide(IsTrueInput(value=value), ExecutionContext()) is expected


def test_path_exists(tmp_path: Path) -> None:
    file = tmp_path / "file.txt"
    file.write_text("x")
    ctx = ExecutionContext()

    def check(path: Path, kind: Any = "any") -> bool:
        return PathExists().decide(PathExistsInput(path=path, kind=kind), ctx)

    assert check(file) and check(file, "file") and not check(file, "directory")
    assert check(tmp_path) and check(tmp_path, "directory") and not check(tmp_path, "file")
    assert not check(tmp_path / "missing")


def test_condition_tasks_are_switch_tasks() -> None:
    for task in (Compare, IsTrue, PathExists):
        assert is_switch_task(task)
        assert task.branches == (True, False)


def test_condition_tasks_are_registered() -> None:
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    registered = pyproject["project"]["entry-points"]["taskmaestro.tasks"]
    loaded = {}
    for name, target in registered.items():
        module, _, attribute = target.partition(":")
        loaded[name] = getattr(import_module(module), attribute)
    assert loaded == {
        "condition.compare": Compare,
        "condition.is_true": IsTrue,
        "condition.path_exists": PathExists,
    }


@pytest.mark.parametrize(("start", "result"), [(1, 4), (5, -6)])
def test_compare_routes_on_upstream_field(start: int, result: int) -> None:
    builder = Workflow.builder("compare")
    added = builder.task(AddOne)
    large = builder.task(Compare, value=added.field("value"), config_fields=["op", "threshold"])
    yes = builder.task(Negate, depends_on=added, when=large.if_true())
    no = builder.task(Double, depends_on=added, when=large.if_false())
    builder.task(Join, picked=first_of(yes, no))
    job = Job(
        workflow=builder.build(),
        config=NumberInput(value=start),
        job_configuration=JobConfiguration({"compare": {"op": ">", "threshold": 3}}),
    )

    assert Runner().run(job).result == NumberOutput(value=result)
