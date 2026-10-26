"""Tests for conditional branches: switch/condition tasks, guards and first_of."""

from __future__ import annotations

import enum
import json
import logging
import re
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import pytest
from pydantic import BaseModel

from taskmaestro import (
    Branch,
    ConditionTask,
    ContextFlag,
    ExecutionContext,
    Job,
    JobConfiguration,
    JobStatus,
    ResultTaskSkippedError,
    Runner,
    SwitchTask,
    Task,
    TaskStatus,
    Workflow,
    WorkflowDefinitionError,
    collect,
    first_of,
    load_workflow_from_yaml,
    to_mermaid,
    when,
    workflow_task,
)
from taskmaestro.cli import main as cli_main
from taskmaestro.conditions import Condition
from taskmaestro.definition import definition_from_workflow, load_definition, write_definition
from taskmaestro.exceptions import ConfigLoadError, ContextValueMissingError, DefinitionError
from taskmaestro.hooks import LoggingHook
from taskmaestro.hooks.base import BaseHook
from taskmaestro.introspection import describe_workflow
from tests.conftest import AddOne, Double, NumberInput, NumberOutput

L = TypeVar("L")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

# --- Tasks ---


class Route(SwitchTask[NumberOutput, Literal["small", "medium", "large"]]):
    name = "route"

    def decide(
        self, input: NumberOutput, ctx: ExecutionContext
    ) -> Literal["small", "medium", "large"]:
        if input.value < 10:
            return "small"
        if input.value < 100:
            return "medium"
        return "large"


class IsEven(ConditionTask[NumberOutput]):
    name = "is_even"

    def decide(self, input: NumberOutput, ctx: ExecutionContext) -> bool:
        return input.value % 2 == 0


class Color(enum.StrEnum):
    RED = "red"
    BLUE = "blue"


class PickColor(SwitchTask[NumberOutput, Color]):
    name = "pick_color"

    def decide(self, input: NumberOutput, ctx: ExecutionContext) -> Color:
        return Color.RED if input.value % 2 else Color.BLUE


class Triple(Task[NumberOutput, NumberOutput]):
    name = "triple"

    def run(self, input: NumberOutput, ctx: ExecutionContext) -> NumberOutput:
        return NumberOutput(value=input.value * 3)


class Negate(Task[NumberOutput, NumberOutput]):
    name = "negate"

    def run(self, input: NumberOutput, ctx: ExecutionContext) -> NumberOutput:
        return NumberOutput(value=-input.value)


class Flagged(BaseModel):
    value: int
    flag: bool


class MakeFlag(Task[NumberOutput, Flagged]):
    name = "make_flag"

    def run(self, input: NumberOutput, ctx: ExecutionContext) -> Flagged:
        return Flagged(value=input.value, flag=input.value > 5)


class MaybeFlag(BaseModel):
    note: str | None


class MaybeCount(BaseModel):
    count: int | None


class MakeMaybeCount(Task[NumberOutput, MaybeCount]):
    name = "make_maybe_count"

    def run(self, input: NumberOutput, ctx: ExecutionContext) -> MaybeCount:
        return MaybeCount(count=None if input.value > 5 else input.value - 2)


class MakeMaybeFlag(Task[NumberOutput, MaybeFlag]):
    name = "make_maybe_flag"

    def run(self, input: NumberOutput, ctx: ExecutionContext) -> MaybeFlag:
        return MaybeFlag(note="big" if input.value > 5 else None)


class Item(BaseModel):
    key: str
    value: int


class ItemFlag(Task[Item, Flagged]):
    name = "item_flag"

    def run(self, input: Item, ctx: ExecutionContext) -> Flagged:
        return Flagged(value=input.value, flag=True)


class JoinInput(BaseModel):
    picked: NumberOutput


class Join(Task[JoinInput, NumberOutput]):
    name = "join"

    def run(self, input: JoinInput, ctx: ExecutionContext) -> NumberOutput:
        return input.picked


class CollectInput(BaseModel):
    values: list[NumberOutput]


class CountValues(Task[CollectInput, NumberOutput]):
    name = "count_values"

    def run(self, input: CollectInput, ctx: ExecutionContext) -> NumberOutput:
        return NumberOutput(value=len(input.values))


class DatedValues(BaseModel):
    day: date
    timestamp: datetime
    path: Path


class EchoDatedValues(Task[DatedValues, DatedValues]):
    name = "dated_values"

    def run(self, input: DatedValues, ctx: ExecutionContext) -> DatedValues:
        return input


class WhenInput(BaseModel):
    when: NumberOutput


class UsesWhenField(Task[WhenInput, NumberOutput]):
    name = "uses_when_field"

    def run(self, input: WhenInput, ctx: ExecutionContext) -> NumberOutput:
        return input.when


class SkipRecorder(BaseHook):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_task_start(self, job: Job[Any], task: Task[Any, Any]) -> None:
        self.events.append(f"start:{task.name}")

    def on_task_skip(self, job: Job[Any], task_name: str, reason: str) -> None:
        self.events.append(f"skip:{task_name}")

    def on_job_fail(self, job: Job[Any]) -> None:
        self.events.append("job_fail")


def _run(workflow: Workflow, value: int, hooks: list[Any] | None = None) -> Job[Any]:
    return Runner(hooks=hooks or []).run(Job(workflow=workflow, config=NumberInput(value=value)))


def _status(job: Job[Any]) -> dict[str, TaskStatus]:
    return {result.task_name: result.status for result in job.task_results}


def _switch_workflow() -> Workflow:
    builder = Workflow.builder("routing")
    start = builder.task(AddOne)
    route = builder.task(Route, depends_on=start)
    small = builder.task(Double, depends_on=start, when=route.on("small"))
    other = builder.task(Triple, depends_on=start, when=route.otherwise())
    builder.task(Join, picked=first_of(small, other))
    return builder.build()


# --- Routing ---


@pytest.mark.parametrize(
    ("value", "ran", "skipped", "result"),
    [(1, "double", "triple", 4), (20, "triple", "double", 63), (500, "triple", "double", 1503)],
)
def test_switch_routes_and_first_of_joins(value: int, ran: str, skipped: str, result: int) -> None:
    job = _run(_switch_workflow(), value)

    assert job.status == JobStatus.COMPLETED
    assert job.result == NumberOutput(value=result)
    status = _status(job)
    assert status[ran] == TaskStatus.COMPLETED
    assert status[skipped] == TaskStatus.SKIPPED
    assert job.skipped_tasks == [skipped]


def test_condition_task_if_true_and_if_false() -> None:
    builder = Workflow.builder("if")
    start = builder.task(AddOne)
    even = builder.task(IsEven, depends_on=start)
    yes = builder.task(Double, depends_on=start, when=even.if_true())
    no = builder.task(Negate, depends_on=start, when=even.if_false())
    builder.task(Join, picked=first_of(yes, no))
    workflow = builder.build()

    assert _run(workflow, 1).result == NumberOutput(value=4)
    assert _run(workflow, 2).result == NumberOutput(value=-3)


@pytest.mark.parametrize(
    ("value", "truthy_runs"),
    [(1, True), (2, True), (9, False)],  # counts 0, 1 and None
)
def test_truthy_guard_on_optional_non_bool_tests_presence(value: int, truthy_runs: bool) -> None:
    builder = Workflow.builder("presence")
    start = builder.task(AddOne)
    maybe = builder.task(MakeMaybeCount, depends_on=start)
    builder.task(Double, depends_on=start, when=maybe.field("count").truthy())
    builder.task(Negate, depends_on=start, when=maybe.field("count").falsy())
    builder.set_result_task(MakeMaybeCount)
    status = _status(_run(builder.build(), value))

    assert status["double"] == (TaskStatus.COMPLETED if truthy_runs else TaskStatus.SKIPPED)
    assert status["negate"] == (TaskStatus.SKIPPED if truthy_runs else TaskStatus.COMPLETED)


def test_enum_switch_accepts_members_and_values() -> None:
    builder = Workflow.builder("enum")
    start = builder.task(AddOne)
    color = builder.task(PickColor, depends_on=start)
    red = builder.task(Double, depends_on=start, when=color.on(Color.RED))
    blue = builder.task(Negate, depends_on=start, when=color.on("blue"))
    builder.task(Join, picked=first_of(red, blue))
    workflow = builder.build()

    assert _run(workflow, 2).result == NumberOutput(value=6)
    assert _run(workflow, 1).result == NumberOutput(value=-2)


@pytest.mark.parametrize(
    ("condition", "value", "runs"),
    [
        (lambda h: when(h.field("value"), equals=5), 4, True),
        (lambda h: h.field("value").equals(5), 5, False),
        (lambda h: when((h, "value"), in_=[1, 2, 3]), 2, True),
        (lambda h: h.field("value").is_in(7, 8), 2, False),
        (lambda h: h.field("flag").truthy(), 9, True),
        (lambda h: h.field("flag").falsy(), 9, False),
    ],
)
def test_field_guards(condition: Any, value: int, runs: bool) -> None:
    builder = Workflow.builder("field")
    start = builder.task(AddOne)
    flagged = builder.task(MakeFlag, depends_on=start)
    builder.task(Double, depends_on=start, when=condition(flagged))
    builder.set_result_task(MakeFlag)
    status = _status(_run(builder.build(), value))

    assert status["double"] == (TaskStatus.COMPLETED if runs else TaskStatus.SKIPPED)


@pytest.mark.parametrize(("value", "runs"), [(9, True), (1, False)])
def test_truthy_guard_on_optional_field(value: int, runs: bool) -> None:
    builder = Workflow.builder("optional")
    start = builder.task(AddOne)
    maybe = builder.task(MakeMaybeFlag, depends_on=start)
    builder.task(Double, depends_on=start, when=maybe.field("note").truthy())
    builder.set_result_task(MakeMaybeFlag)
    status = _status(_run(builder.build(), value))

    assert status["double"] == (TaskStatus.COMPLETED if runs else TaskStatus.SKIPPED)


def test_truthy_guard_on_condition_task_claims_its_label() -> None:
    builder = Workflow.builder("truthy_switch")
    start = builder.task(AddOne)
    even = builder.task(IsEven, depends_on=start)
    yes = builder.task(Double, depends_on=start, when=when(even, truthy=True))
    no = builder.task(Negate, depends_on=start, when=even.otherwise())
    builder.task(Join, picked=first_of(yes, no))
    workflow = builder.build()

    assert workflow.get_guard("negate").values == (False,)  # type: ignore[union-attr]
    assert _run(workflow, 1).result == NumberOutput(value=4)


# --- Skip semantics ---


def test_skip_propagates_and_collect_omits_skipped_members() -> None:
    builder = Workflow.builder("propagate")
    start = builder.task(AddOne)
    flagged = builder.task(MakeFlag, depends_on=start)
    yes = builder.task(Double, depends_on=start, when=flagged.field("flag").truthy())
    downstream = builder.task(Negate, depends_on=yes)
    builder.task(CountValues, values=collect(start, yes, downstream))
    job = _run(builder.build(), 2)

    assert job.status == JobStatus.COMPLETED
    assert job.result == NumberOutput(value=1)
    skipped = {r.task_name: r for r in job.task_results if r.status == TaskStatus.SKIPPED}
    assert set(skipped) == {"double", "negate"}
    assert "make_flag.flag" in (skipped["double"].skip_reason or "")
    assert "double" in (skipped["negate"].skip_reason or "")


def test_task_is_skipped_when_its_condition_source_was_skipped() -> None:
    builder = Workflow.builder("source_skipped")
    start = builder.task(AddOne)
    even = builder.task(IsEven, depends_on=start)
    flag = builder.task(MakeFlag, depends_on=start, when=even.if_true())
    builder.task(Negate, depends_on=start, when=even.if_false())
    builder.task(Double, depends_on=start, when=flag.field("flag").truthy())
    builder.set_result_task(AddOne)
    job = _run(builder.build(), 2)

    skipped = {r.task_name: r.skip_reason for r in job.task_results if r.skip_reason}
    assert skipped["double"] == "condition source 'make_flag' was skipped"


def test_first_of_with_every_member_skipped_skips_the_task() -> None:
    builder = Workflow.builder("all_skipped")
    start = builder.task(AddOne)
    even = builder.task(IsEven, depends_on=start)
    yes = builder.task(Double, depends_on=start, when=even.if_true())
    no = builder.task(Negate, depends_on=start, when=even.if_false())
    gated = builder.task(Triple, depends_on=no, when=even.if_true())
    builder.task(Join, picked=first_of(yes, gated))
    builder.set_result_task(AddOne)

    assert _status(_run(builder.build(), 2))["join"] == TaskStatus.SKIPPED


def test_skipped_result_task_fails_the_job_and_emits_skip_events() -> None:
    builder = Workflow.builder("result_skipped")
    start = builder.task(AddOne)
    even = builder.task(IsEven, depends_on=start)
    builder.task(Double, depends_on=start, when=even.if_true())
    builder.task(Negate, depends_on=start, when=even.if_false())
    builder.set_result_task(Double)
    recorder = SkipRecorder()
    job = _run(builder.build(), 2, hooks=[recorder])

    assert job.status == JobStatus.FAILED
    assert isinstance(job.exception, ResultTaskSkippedError)
    assert job.failed_task == "double"
    assert recorder.events == [
        "start:add_one",
        "start:is_even",
        "skip:double",
        "start:negate",
        "job_fail",
    ]


def test_guarded_map_task_is_skipped_without_running_items() -> None:
    class Item(BaseModel):
        key: str
        amount: int

    class Bump(Task[Item, NumberOutput]):
        name = "bump"

        def run(self, input: Item, ctx: ExecutionContext) -> NumberOutput:
            raise AssertionError("mapped items must not run")

    builder = Workflow.builder("mapped")
    start = builder.task(AddOne)
    flagged = builder.task(MakeFlag, depends_on=start)
    builder.map_task(
        Bump, over="items", key_as="key", value_as="amount", when=flagged.field("flag").truthy()
    )
    builder.set_result_task(MakeFlag)
    job = Runner().run(
        Job(
            workflow=builder.build(),
            config=NumberInput(value=1),
            job_configuration=JobConfiguration({"bump": {"items": {"a": 1, "b": 2}}}),
        )
    )

    assert job.status == JobStatus.COMPLETED
    assert _status(job)["bump"] == TaskStatus.SKIPPED


# --- Validation ---


def test_uncovered_switch_label_is_an_error() -> None:
    builder = Workflow.builder("uncovered")
    start = builder.task(AddOne)
    route = builder.task(Route, depends_on=start)
    builder.task(Double, depends_on=start, when=route.on("small"))
    builder.set_result_task(Route)

    with pytest.raises(WorkflowDefinitionError, match="no task for branch"):
        builder.build()


def test_unreachable_otherwise_is_an_error() -> None:
    builder = Workflow.builder("unreachable")
    start = builder.task(AddOne)
    even = builder.task(IsEven, depends_on=start)
    builder.task(Double, depends_on=start, when=even.if_true())
    builder.task(Negate, depends_on=start, when=even.if_false())
    builder.task(Triple, depends_on=start, when=even.otherwise())
    builder.set_result_task(AddOne)

    with pytest.raises(WorkflowDefinitionError, match="can never match"):
        builder.build()


@pytest.mark.parametrize(
    ("make_condition", "message"),
    [
        (lambda route, flag: route.on("smal"), "not a valid value"),
        (lambda route, flag: flag.field("flag").equals("no"), "not a valid value"),
        (lambda route, flag: flag.field("flag").equals(1), "not a valid value"),
        (lambda route, flag: flag.field("value").equals("5"), "not a valid value"),
        (lambda route, flag: flag.field("value").truthy(), "requires a bool"),
        (lambda route, flag: flag.otherwise(), "not a switch task"),
        (lambda route, flag: when(flag, equals=1), "not a switch task"),
        (lambda route, flag: when((flag, "missing"), equals=1), "not found on Flagged"),
    ],
)
def test_invalid_guards_are_rejected(make_condition: Any, message: str) -> None:
    builder = Workflow.builder("invalid")
    start = builder.task(AddOne)
    route = builder.task(Route, depends_on=start)
    builder.task(Triple, depends_on=start, when=route.otherwise())
    flag = builder.task(MakeFlag, depends_on=start)
    builder.set_result_task(AddOne)

    with pytest.raises(WorkflowDefinitionError, match=message):
        builder.task(Double, depends_on=start, when=make_condition(route, flag))
        builder.build()


def test_condition_on_mapped_task_is_rejected() -> None:
    builder = Workflow.builder("mapped_source")
    start = builder.task(AddOne)
    mapped = builder.map_task(ItemFlag, over="items", key_as="key", value_as="value")
    builder.task(Double, depends_on=start, when=when((mapped, "flag"), truthy=True))
    builder.set_result_task(AddOne)

    with pytest.raises(WorkflowDefinitionError, match="cannot refer to mapped task"):
        builder.build()


def test_first_of_must_target_a_named_field_without_config() -> None:
    builder = Workflow.builder("first_of_misuse")
    start = builder.task(AddOne)
    other = builder.task(Double, depends_on=start)
    with pytest.raises(WorkflowDefinitionError, match="named input field"):
        builder.task(Join, depends_on=first_of(start, other))  # type: ignore[arg-type]

    builder.task(Join, picked=first_of(start, other), config_fields=["picked"])
    with pytest.raises(WorkflowDefinitionError, match="both a first_of dependency"):
        builder.build()


def test_first_of_needs_two_members_and_is_type_checked() -> None:
    builder = Workflow.builder("first_of")
    start = builder.task(AddOne)
    flag = builder.task(MakeFlag, depends_on=start)
    with pytest.raises(TypeError, match="at least two"):
        first_of(start)

    builder.task(Join, picked=first_of(start, flag))
    with pytest.raises(WorkflowDefinitionError):
        builder.build()


def test_task_input_field_named_when_still_works() -> None:
    builder = Workflow.builder("back_compat")
    start = builder.task(AddOne)
    builder.task(UsesWhenField, when=start)

    assert _run(builder.build(), 1).result == NumberOutput(value=2)


def test_switch_without_guards_is_an_ordinary_task() -> None:
    builder = Workflow.builder("plain_switch")
    start = builder.task(AddOne)
    builder.task(Route, depends_on=start)

    assert _run(builder.build(), 1).result.branch == "small"  # type: ignore[union-attr]


# --- Surfaces: Mermaid, describe, definition, YAML ---


def test_mermaid_draws_switches_as_diamonds_with_dashed_guard_edges() -> None:
    diagram = to_mermaid(_switch_workflow())

    assert 'route{"route"}' in diagram
    assert "route -.->|small| double" in diagram
    assert "route -.->|otherwise| triple" in diagram
    assert '_first_of_join_picked_(["first_of picked"])' in diagram
    assert "_first_of_join_picked_ -->|picked: NumberOutput| join" in diagram


def test_mermaid_labels_field_guards() -> None:
    builder = Workflow.builder("field_labels")
    start = builder.task(AddOne)
    flag = builder.task(MakeFlag, depends_on=start)
    builder.task(Double, depends_on=start, when=flag.field("value").equals(3))
    builder.task(Triple, depends_on=start, when=flag.field("value").is_in(1, 2))
    builder.task(Negate, depends_on=start, when=flag.field("flag").truthy())
    builder.task(Join, picked=start, when=flag.field("flag").falsy())
    builder.set_result_task(AddOne)
    diagram = to_mermaid(builder.build())

    assert "make_flag -.->|.value = 3| double" in diagram
    assert "make_flag -.->|.value in 1 / 2| triple" in diagram
    assert "make_flag -.->|.flag| negate" in diagram
    assert "make_flag -.->|not .flag| join" in diagram


def test_describe_workflow_reports_guards_and_branches() -> None:
    tasks = {task["name"]: task for task in describe_workflow(_switch_workflow())["tasks"]}

    assert tasks["route"]["branches"] == ["small", "medium", "large"]
    assert tasks["double"]["when"] == {
        "task": "route",
        "field": "branch",
        "op": "in",
        "values": ["small"],
    }
    assert tasks["triple"]["when"]["values"] == ["medium", "large"]
    assert tasks["join"]["depends_on"]["picked"] == {
        "first_of": [{"task": "double", "field": None}, {"task": "triple", "field": None}]
    }


@pytest.mark.parametrize(
    ("field", "value", "serialized"),
    [
        ("day", date(2026, 1, 1), "2026-01-01"),
        ("timestamp", datetime(2026, 1, 1, tzinfo=UTC), "2026-01-01T00:00:00+00:00"),
        ("path", Path("data/input.txt"), "data/input.txt"),
    ],
)
def test_describe_workflow_serializes_guard_values(
    field: str, value: Any, serialized: str
) -> None:
    builder = Workflow.builder("serialized_guards")
    source = builder.task(EchoDatedValues)
    builder.task(
        EchoDatedValues, name="selected", depends_on=source, when=source.field(field).equals(value)
    )

    description = json.loads(json.dumps(describe_workflow(builder.build())))
    assert description["tasks"][1]["when"]["values"] == [serialized]


@pytest.mark.parametrize("field", ["day", "timestamp", "path"])
@pytest.mark.parametrize("op", ["equals", "in"])
@pytest.mark.parametrize("from_python", [False, True])
def test_definition_round_trip_keeps_serialized_guards(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
    op: str,
    from_python: bool,
) -> None:
    values = DatedValues(
        day=date(2026, 1, 1),
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        path=Path("data/input.txt"),
    )
    source = tmp_path / "source"
    source.mkdir()
    if from_python:
        builder = Workflow.builder("serialized_guards")
        start = builder.task(EchoDatedValues, config_fields=["day", "timestamp", "path"])
        value = getattr(values, field)
        condition = (
            start.field(field).equals(value) if op == "equals" else start.field(field).is_in(value)
        )
        builder.task(EchoDatedValues, name="selected", depends_on=start, when=condition)
        definition = definition_from_workflow(builder.build(), None, {})
        definition["inputs"]["dated_values"] = values.model_dump(mode="json")
    else:
        literal = {
            "day": "2026-01-01",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "path": "data/input.txt",
        }[field]
        operand = literal if op == "equals" else f"[{literal}]"
        (source / "workflow.yaml").write_text(
            "workflow:\n"
            "  name: serialized_guards\n"
            "  tasks:\n"
            "    - task: tests.test_conditions.EchoDatedValues\n"
            "      name: dated_values\n"
            "      config_fields: [day, timestamp, path]\n"
            "    - task: tests.test_conditions.EchoDatedValues\n"
            "      name: selected\n"
            "      depends_on: dated_values\n"
            f"      when: {{ref: [dated_values, {field}], {op}: {operand}}}\n"
        )
        (source / "input.yaml").write_text(
            json.dumps({"dated_values": values.model_dump(mode="json")})
        )
        original = load_workflow_from_yaml(source / "workflow.yaml", source / "input.yaml").run()
        assert original.status == JobStatus.COMPLETED
        definition = load_definition(source)
        assert cli_main(["workflow", "describe", str(source / "workflow.yaml"), "--json"]) == 0
        description = json.loads(capsys.readouterr().out)
        assert description["tasks"][1]["when"]["values"] == [literal]

    # Definitions must survive a JSON editor/service round trip as well as YAML saving.
    definition = json.loads(json.dumps(definition))
    saved = tmp_path / "saved"
    write_definition(definition, saved)
    reloaded = load_workflow_from_yaml(saved / "workflow.yaml", saved / "input.yaml").run()
    assert reloaded.status == JobStatus.COMPLETED
    assert reloaded.result == values
    assert not reloaded.skipped_tasks


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("day", 1767225600),
        ("timestamp", 1767225600),
        ("day", True),
        ("timestamp", False),
        ("path", 123),
        ("day", "not-a-date"),
    ],
)
def test_serialized_guards_still_reject_lossy_coercions(field: str, value: Any) -> None:
    builder = Workflow.builder("invalid_serialized_guard")
    source = builder.task(EchoDatedValues)
    builder.task(
        EchoDatedValues, name="selected", depends_on=source, when=source.field(field).equals(value)
    )
    with pytest.raises(WorkflowDefinitionError, match="not a valid value"):
        builder.build()


def test_definition_round_trip_keeps_guards_and_first_of(tmp_path: Path) -> None:
    builder = Workflow.builder("roundtrip")
    start = builder.task(AddOne, config_fields=["value"])
    route = builder.task(Route, depends_on=start)
    small = builder.task(Double, depends_on=start, when=route.on("small", "medium"))
    large = builder.task(Triple, depends_on=start, when=route.otherwise())
    flag = builder.task(MakeFlag, depends_on=start)
    builder.task(Negate, depends_on=large, when=flag.field("flag").truthy())
    builder.task(Join, picked=first_of(small, large))
    builder.set_result_task(Join)

    class_ids = {
        cls: f"{cls.__module__}.{cls.__name__}"
        for cls in (AddOne, Route, Double, Triple, MakeFlag, Negate, Join)
    }
    write_definition(definition_from_workflow(builder.build(), None, class_ids), tmp_path)
    text = (tmp_path / "workflow.yaml").read_text()
    assert "case: [small, medium]" in text
    assert "otherwise: true" in text
    assert "ref: [make_flag, flag]" in text

    reloaded = load_definition(tmp_path)
    nodes = {node["name"]: node for node in reloaded["nodes"]}
    assert nodes["triple"]["when"] == {"switch": "route", "otherwise": True}
    assert [e["from"] for e in reloaded["edges"] if e.get("collect") == "first_of"] == [
        "double",
        "triple",
    ]
    (tmp_path / "input.yaml").write_text("add_one: {value: 200}\n")
    job = load_workflow_from_yaml(tmp_path / "workflow.yaml", tmp_path / "input.yaml").run()
    assert job.result == NumberOutput(value=603)


YAML_WORKFLOW = """\
workflow:
  name: yaml_routing
  result_task: join
  tasks:
    - task: tests.conftest.AddOne
      name: add_one
      config_fields: [value]
    - task: tests.test_conditions.Route
      name: route
      depends_on: add_one
    - task: tests.conftest.Double
      name: double
      depends_on: add_one
      when: {switch: route, case: [small, medium]}
    - task: tests.test_conditions.Triple
      name: triple
      depends_on: add_one
      when: {switch: route, otherwise: true}
    - task: tests.test_conditions.MakeFlag
      name: make_flag
      depends_on: add_one
    - task: tests.test_conditions.Negate
      name: negate
      depends_on: add_one
      when: {ref: [make_flag, flag], truthy: false}
    - task: tests.test_conditions.Join
      name: join
      depends_on:
        picked:
          first_of: [double, triple]
"""


@pytest.mark.parametrize(
    ("value", "result", "negate"), [(1, 4, TaskStatus.COMPLETED), (499, 1500, TaskStatus.SKIPPED)]
)
def test_yaml_when_and_first_of(
    tmp_path: Path, value: int, result: int, negate: TaskStatus
) -> None:
    (tmp_path / "workflow.yaml").write_text(YAML_WORKFLOW)
    (tmp_path / "input.yaml").write_text(f"add_one: {{value: {value}}}\n")

    job = load_workflow_from_yaml(tmp_path / "workflow.yaml", tmp_path / "input.yaml").run()

    assert job.result == NumberOutput(value=result)
    assert _status(job)["negate"] == negate


FIELD_GUARD_YAML = """\
workflow:
  name: field_guards
  result_task: add_one
  tasks:
    - task: tests.conftest.AddOne
      name: add_one
      config_fields: [value]
    - task: tests.test_conditions.Route
      name: route
      depends_on: add_one
    - task: tests.conftest.Double
      name: double
      depends_on: add_one
      when: {ref: route, in: [small, medium]}
    - task: tests.test_conditions.Triple
      name: triple
      depends_on: add_one
      when: {switch: route, case: large}
    - task: tests.test_conditions.Negate
      name: negate
      depends_on: add_one
      when: {ref: [add_one, value], equals: 3}
    - task: tests.test_conditions.Join
      name: join
      depends_on:
        picked: add_one
      when: {ref: [add_one, value], in: [1, 2]}
"""


def test_yaml_field_guards_run_and_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "workflow.yaml").write_text(FIELD_GUARD_YAML)
    (source / "input.yaml").write_text("add_one: {value: 2}\n")
    job = load_workflow_from_yaml(source / "workflow.yaml", source / "input.yaml").run()
    assert _status(job) == {
        "add_one": TaskStatus.COMPLETED,
        "route": TaskStatus.COMPLETED,
        "double": TaskStatus.COMPLETED,
        "triple": TaskStatus.SKIPPED,
        "negate": TaskStatus.COMPLETED,
        "join": TaskStatus.SKIPPED,
    }

    definition = load_definition(source)
    nodes = {node["name"]: node for node in definition["nodes"]}
    assert nodes["double"]["when"] == {"ref": "route", "in": ["small", "medium"]}
    saved = tmp_path / "saved"
    write_definition(definition, saved)
    text = (saved / "workflow.yaml").read_text()
    assert "in: [small, medium]" in text
    assert "equals: 3" in text
    assert "in: [1, 2]" in text

    builder = Workflow.builder("field_guards")
    start = builder.task(AddOne, config_fields=["value"])
    builder.task(Negate, depends_on=start, when=start.field("value").equals(3))
    builder.task(Join, picked=start, when=start.field("value").is_in(1, 2))
    builder.set_result_task(AddOne)
    generated = definition_from_workflow(builder.build(), None, {})
    whens = {node["name"]: node.get("when") for node in generated["nodes"]}
    assert whens["negate"] == {"ref": ["add_one", "value"], "equals": 3}
    assert whens["join"] == {"ref": ["add_one", "value"], "in": [1, 2]}


def test_saving_a_condition_on_an_unknown_task_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "workflow.yaml").write_text(FIELD_GUARD_YAML)
    definition = load_definition(tmp_path)
    triple = next(node for node in definition["nodes"] if node["name"] == "triple")
    triple["when"] = {"switch": "missing", "case": "large"}

    with pytest.raises(DefinitionError, match="unknown task 'missing'"):
        write_definition(definition, tmp_path / "out")


@pytest.mark.parametrize(
    "when_yaml",
    [
        "{switch: route}",
        "{switch: route, case: small, otherwise: true}",
        "{ref: [make_flag, flag]}",
        "{ref: [make_flag, flag], equals: 1, truthy: true}",
        "{switch: route, ref: add_one, case: small}",
        "{case: small}",
        "{switch: route, case: []}",
        "{ref: [make_flag, flag], otherwise: true, truthy: true}",
        "{ref: [make_flag], truthy: true}",
        "{ref: [make_flag, value], in: []}",
        "{ref: [make_flag, value], in: null}",
        "{ref: [make_flag, flag], truthy: null}",
    ],
)
def test_yaml_rejects_malformed_when(tmp_path: Path, when_yaml: str) -> None:
    workflow = YAML_WORKFLOW.replace("{switch: route, otherwise: true}", when_yaml)
    (tmp_path / "workflow.yaml").write_text(workflow)
    (tmp_path / "input.yaml").write_text("add_one: {value: 1}\n")

    with pytest.raises(ConfigLoadError):
        load_workflow_from_yaml(tmp_path / "workflow.yaml", tmp_path / "input.yaml")


@pytest.mark.parametrize("example", ["conditional_if", "conditional_switch"])
@pytest.mark.parametrize("mode", [[], ["--yaml"]])
def test_conditional_examples_run(example: str, mode: list[str]) -> None:
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / example / "pipeline.py"), *mode],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Status: completed" in completed.stdout
    assert "skipped" in completed.stdout


# --- ContextFlag ---


def _flag_workflow() -> Workflow:
    builder = Workflow.builder("flag")
    start = builder.task(AddOne)
    dry = builder.task(ContextFlag, name="dry_run", config_fields=["key"])
    yes = builder.task(Negate, depends_on=start, when=dry.if_true())
    no = builder.task(Double, depends_on=start, when=dry.if_false())
    builder.task(Join, picked=first_of(yes, no))
    return builder.build()


def _flag_job() -> Job[Any]:
    return Job(
        workflow=_flag_workflow(),
        config=NumberInput(value=1),
        job_configuration=JobConfiguration({"dry_run": {"key": "dry"}}),
    )


@pytest.mark.parametrize(("value", "result"), [(True, -2), ("", 4)])
def test_context_flag_routes_on_context_value(value: Any, result: int) -> None:
    ctx = ExecutionContext()
    ctx.register("dry", value)

    assert Runner().run(_flag_job(), ctx=ctx).result == NumberOutput(value=result)


def test_context_flag_with_unregistered_key_fails_before_any_task() -> None:
    recorder = SkipRecorder()
    job = _flag_job()

    with pytest.raises(ContextValueMissingError, match="'dry' \\(task 'dry_run'\\)"):
        Runner(hooks=[recorder]).run(job, ctx=ExecutionContext())
    assert job.status == JobStatus.PENDING
    assert recorder.events == []


@pytest.mark.parametrize("as_json", [False, True])
def test_cli_reports_unregistered_context_flag_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], as_json: bool
) -> None:
    (tmp_path / "workflow.yaml").write_text(
        """\
workflow:
  name: cli_flag
  tasks:
    - task: tests.conftest.AddOne
      name: add_one
      config_fields: [value]
    - task: taskmaestro.ContextFlag
      name: dry_run
      config_fields: [key]
    - task: tests.conftest.Double
      name: double
      depends_on: add_one
      when: {switch: dry_run, case: false}
    - task: tests.test_conditions.Negate
      name: negate
      depends_on: add_one
      when: {switch: dry_run, case: true}
    - task: tests.test_conditions.Join
      name: join
      depends_on:
        picked:
          first_of: [double, negate]
"""
    )
    (tmp_path / "input.yaml").write_text("add_one: {value: 1}\ndry_run: {key: dry}\n")
    args = ["run", str(tmp_path / "workflow.yaml"), "--input", str(tmp_path / "input.yaml")]

    assert cli_main([*args, "--json"] if as_json else args) == 2
    captured = capsys.readouterr()
    if as_json:
        payload = json.loads(captured.out)
        assert payload["status"] == "invalid"
        assert "'dry' (task 'dry_run')" in payload["error"]["message"]
    else:
        assert "'dry' (task 'dry_run')" in captured.err


# --- Edge cases and error branches ---


class Verdict(BaseModel):
    ok: Literal[True, False]
    data: Any


class MakeVerdict(Task[NumberOutput, Verdict]):
    name = "make_verdict"

    def run(self, input: NumberOutput, ctx: ExecutionContext) -> Verdict:
        return Verdict(ok=input.value > 5, data=input.value)


class GenericSwitch(SwitchTask[NumberOutput, L], Generic[L]):
    """Intermediate generic switch; concrete labels come from subclasses."""

    def decide(self, input: NumberOutput, ctx: ExecutionContext) -> Any:
        return self.branches[-1]


class LabelMixin:
    """A non-switch base listed before the switch base."""


class ConcreteSwitch(LabelMixin, GenericSwitch[Literal["first", "last"]]):
    name = "concrete_switch"


class RenamedConcreteSwitch(ConcreteSwitch):
    name = "renamed_switch"


class UnlabelledSwitch(SwitchTask):  # type: ignore[type-arg]
    def decide(self, input: Any, ctx: ExecutionContext) -> Any:
        return None


def test_generic_switch_base_resolves_labels_in_concrete_subclasses() -> None:
    assert GenericSwitch.branches == ()
    assert UnlabelledSwitch.branches == ()
    assert ConcreteSwitch.branches == ("first", "last")
    assert RenamedConcreteSwitch.branches == ("first", "last")

    builder = Workflow.builder("generic_switch")
    start = builder.task(AddOne)
    switch = builder.task(ConcreteSwitch, depends_on=start)
    builder.task(Double, depends_on=start, when=switch.on("first"))
    builder.task(Triple, depends_on=start, when=switch.on("last"))
    builder.set_result_task(AddOne)
    status = _status(_run(builder.build(), 1))

    assert status["double"] == TaskStatus.SKIPPED
    assert status["triple"] == TaskStatus.COMPLETED


def test_switch_label_type_must_be_bool_literal_or_enum() -> None:
    with pytest.raises(TypeError, match="Switch label type must be"):

        class IntSwitch(SwitchTask[NumberOutput, int]):
            def decide(self, input: NumberOutput, ctx: ExecutionContext) -> int:
                return 0


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda h: when(h), "exactly one of"),
        (lambda h: when(h, equals=1, truthy=True), "exactly one of"),
        (lambda h: when(h, in_=[]), "at least one value"),
        (lambda h: when(h, in_="abc"), "not a string"),
        (lambda h: h.on(), "at least one branch label"),
    ],
)
def test_condition_helpers_reject_bad_arguments(make: Any, message: str) -> None:
    builder = Workflow.builder("helpers")
    route = builder.task(Route, depends_on=builder.task(AddOne))

    with pytest.raises(TypeError, match=message):
        make(route)


def test_guards_on_literal_bool_and_any_fields() -> None:
    builder = Workflow.builder("verdict")
    start = builder.task(AddOne)
    verdict = builder.task(MakeVerdict, depends_on=start)
    builder.task(Double, depends_on=start, when=verdict.field("ok").truthy())
    builder.task(Negate, depends_on=start, when=verdict.field("data").equals(2))
    builder.task(Triple, depends_on=start, when=verdict.field("ok").equals(True))
    builder.set_result_task(AddOne)
    workflow = builder.build()

    assert workflow.is_conditional("double")
    assert not workflow.is_conditional("add_one")
    assert "make_verdict -.->|.ok = true| triple" in to_mermaid(workflow)
    status = _status(_run(workflow, 1))
    assert status["double"] == TaskStatus.SKIPPED
    assert status["negate"] == TaskStatus.COMPLETED


@pytest.mark.parametrize(
    ("add", "message"),
    [
        (
            lambda b, flag: b.add_task(Double, depends_on="add_one", when=flag),
            "must be a condition",
        ),
        (
            lambda b, flag: b.add_task(
                Double, depends_on="add_one", when=Condition((flag, "flag"), "otherwise")
            ),
            "requires a switch task",
        ),
        (
            lambda b, flag: b.add_task(
                Double, depends_on="add_one", when=Condition("missing", "in", (1,))
            ),
            "missing",
        ),
        (
            lambda b, flag: b.task(Join, picked=first_of((flag, "flag"), "add_one")),
            "'make_flag.flag' produces bool",
        ),
    ],
)
def test_builder_rejects_invalid_conditions(add: Any, message: str) -> None:
    builder = Workflow.builder("builder_errors")
    builder.task(AddOne)
    flag = builder.task(MakeFlag, depends_on="add_one")
    builder.set_result_task(AddOne)

    with pytest.raises(WorkflowDefinitionError, match=message):
        add(builder, flag)
        builder.build()


def test_yaml_first_of_needs_two_members(tmp_path: Path) -> None:
    (tmp_path / "workflow.yaml").write_text(YAML_WORKFLOW.replace("[double, triple]", "[double]"))
    (tmp_path / "input.yaml").write_text("add_one: {value: 1}\n")

    with pytest.raises(ConfigLoadError, match="at least two members"):
        load_workflow_from_yaml(tmp_path / "workflow.yaml", tmp_path / "input.yaml")


def test_definition_rejects_malformed_first_of_and_conditions(tmp_path: Path) -> None:
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "workflow.yaml").write_text(
        YAML_WORKFLOW.replace("first_of: [double, triple]", "first_of: double")
    )
    with pytest.raises(DefinitionError, match="must be a list"):
        load_definition(broken)

    (tmp_path / "workflow.yaml").write_text(YAML_WORKFLOW)
    definition = load_definition(tmp_path)
    join = next(node for node in definition["nodes"] if node["name"] == "join")
    join["when"] = "always"
    with pytest.raises(DefinitionError, match="Invalid condition for task 'join'"):
        write_definition(definition, tmp_path / "out")

    join.pop("when")
    plain = {"from": "add_one", "output": None, "to": "join", "input": "picked"}
    definition["edges"].insert(0, plain)
    with pytest.raises(DefinitionError, match="mixes first_of alternatives"):
        write_definition(definition, tmp_path / "out")


def test_logging_hook_logs_skipped_tasks(caplog: pytest.LogCaptureFixture) -> None:
    builder = Workflow.builder("logged")
    start = builder.task(AddOne)
    even = builder.task(IsEven, depends_on=start)
    builder.task(Double, depends_on=start, when=even.if_true())
    builder.task(Negate, depends_on=start, when=even.if_false())
    builder.set_result_task(AddOne)

    with caplog.at_level(logging.INFO, logger="taskmaestro.hooks.logging"):
        _run(builder.build(), 2, hooks=[LoggingHook()])

    assert "Task skipped: double, reason=" in caplog.text


# --- Review fixes ---


class ScoredBranch(Branch):
    branch: Literal["a", "b"]
    score: int = 0


class ScoredSwitch(SwitchTask[NumberOutput, Literal["a", "b"]]):
    name = "scored"
    Outputs = ScoredBranch

    def decide(self, input: NumberOutput, ctx: ExecutionContext) -> Literal["a", "b"]:
        return "a"

    def run(self, input: NumberOutput, ctx: ExecutionContext) -> Branch:
        return ScoredBranch(branch=self.decide(input, ctx), score=input.value)


class ScoredSwitchChild(ScoredSwitch):
    name = "scored_child"


class ScoreInput(BaseModel):
    score: int


class UseScore(Task[ScoreInput, NumberOutput]):
    name = "use_score"

    def run(self, input: ScoreInput, ctx: ExecutionContext) -> NumberOutput:
        return NumberOutput(value=input.score)


def test_switch_subclass_keeps_inherited_custom_outputs() -> None:
    assert ScoredSwitchChild.Outputs is ScoredBranch  # type: ignore[attr-defined]
    # Generated outputs are still regenerated per subclass.
    generated = ConcreteSwitch.Outputs  # type: ignore[attr-defined]
    assert RenamedConcreteSwitch.Outputs is not generated  # type: ignore[attr-defined]

    builder = Workflow.builder("scored")
    start = builder.task(AddOne)
    switch = builder.task(ScoredSwitchChild, depends_on=start)
    builder.task(UseScore, name="on_a", score=switch.field("score"), when=switch.on("a"))
    builder.task(UseScore, name="on_b", score=switch.field("score"), when=switch.on("b"))
    builder.set_result_task("on_a")
    job = _run(builder.build(), 4)

    assert job.status == JobStatus.COMPLETED
    assert job.result == NumberOutput(value=5)


def test_skip_reason_names_the_condition_that_did_not_hold() -> None:
    builder = Workflow.builder("falsy_reason")
    start = builder.task(AddOne)
    flagged = builder.task(MakeFlag, depends_on=start)
    builder.task(Double, depends_on=start, when=flagged.field("flag").falsy())
    builder.set_result_task(AddOne)
    job = _run(builder.build(), 9)

    reasons = {r.task_name: r.skip_reason for r in job.task_results if r.skip_reason}
    assert reasons["double"] == "condition 'not make_flag.flag' did not hold (value True)"


def test_context_flag_in_nested_workflow_is_checked_before_any_task() -> None:
    inner = _flag_workflow()
    nested = workflow_task(
        inner, name="inner", job_configuration=JobConfiguration({"dry_run": {"key": "dry"}})
    )
    builder = Workflow.builder("outer")
    inner_task = builder.task(nested)
    builder.task(Double, depends_on=inner_task)
    recorder = SkipRecorder()
    job = Job(workflow=builder.build(), config=NumberInput(value=1))

    with pytest.raises(ContextValueMissingError, match=re.escape("'dry' (task 'inner.dry_run')")):
        Runner(hooks=[recorder]).run(job, ctx=ExecutionContext())
    assert job.status == JobStatus.PENDING
    assert recorder.events == []


@pytest.mark.parametrize("when_value", [{}, {"ref": []}, {"switch": None, "case": "a"}])
def test_write_definition_rejects_condition_without_source(
    tmp_path: Path, when_value: dict[str, Any]
) -> None:
    builder = Workflow.builder("no_source")
    start = builder.task(AddOne)
    builder.task(Double, depends_on=start)
    definition = definition_from_workflow(builder.build(), None, {})
    definition["nodes"][1]["when"] = when_value

    with pytest.raises(DefinitionError, match="needs 'switch' or 'ref'"):
        write_definition(definition, tmp_path)
