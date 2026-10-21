"""Tests for editable workflow definitions and the public introspection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from taskmaestro import (
    ConfigLoadError,
    ExecutionContext,
    JobStatus,
    Task,
    TaskMap,
    Workflow,
    collect,
    load_workflow_from_yaml,
    workflow_task,
)
from taskmaestro.definition import (
    definition_from_workflow,
    describe_yaml,
    load_definition,
    write_definition,
)
from taskmaestro.exceptions import DefinitionError
from taskmaestro.introspection import (
    TaskSchemaGenerator,
    describe_yaml_workflow,
    error_payload,
    jsonable,
    model_json_schema,
)
from taskmaestro.yaml_config import load_workflow_only, resolve_task
from tests.conftest import AddOne, Double, FanInTask, NumberInput, NumberOutput

WORKFLOW = """\
# A pipeline.
workflow:
  name: pipeline
  tasks:
    - task: tests.conftest.AddOne
      name: first
      config_fields: [value]
    - task: tests.conftest.Double
      name: second
      depends_on: first
"""


def _write(directory: Path, workflow: str = WORKFLOW, inputs: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "workflow.yaml").write_text(workflow)
    if inputs is not None:
        (directory / "input.yaml").write_text(inputs)
    return directory


class OpaqueAnnotator(TaskSchemaGenerator):
    @classmethod
    def type_annotations(cls, python_type: type[Any]) -> dict[str, Any]:
        return {"x-test-type": python_type.__name__}


def test_load_definition_builds_nodes_edges_and_inputs(tmp_path: Path) -> None:
    folder = _write(tmp_path / "wf", inputs="first: {value: 3}\n")

    definition = load_definition(folder)

    assert [node["name"] for node in definition["nodes"]] == ["first", "second"]
    assert definition["edges"] == [
        {"from": "first", "output": None, "to": "second", "input": None}
    ]
    assert definition["inputs"] == {"first": {"value": 3}, "second": {}}
    assert definition["header_comment"] == "# A pipeline."
    assert definition["editable"] is True


def test_written_definition_loads_and_runs_like_the_original(tmp_path: Path) -> None:
    source = _write(tmp_path / "src", inputs="first: {value: 3}\n")
    definition = load_definition(source)

    result = write_definition(definition, tmp_path / "out")

    assert result["backups"] == []
    reloaded = load_definition(tmp_path / "out")
    for key in ("nodes", "edges", "inputs", "result_task"):
        assert reloaded[key] == definition[key]
    loaded = load_workflow_from_yaml(result["workflow_yaml"], result["input_yaml"])
    assert loaded.run().status == JobStatus.COMPLETED


def test_value_filter_is_applied_to_saved_inputs(tmp_path: Path) -> None:
    definition = load_definition(_write(tmp_path / "src", inputs="first: {value: 3}\n"))

    write_definition(
        definition,
        tmp_path / "out",
        value_filter=lambda values: {k: v for k, v in values.items() if k != "value"},
    )

    assert load_definition(tmp_path / "out")["inputs"]["first"] == {}


def test_backup_keeps_the_first_version_of_existing_files(tmp_path: Path) -> None:
    folder = _write(tmp_path / "wf", inputs="first: {value: 3}\n")
    definition = load_definition(folder)

    result = write_definition(definition, folder, backup=True)

    assert sorted(Path(path).name for path in result["backups"]) == [
        "input.yaml.bak",
        "workflow.yaml.bak",
    ]


def test_schema_generator_annotates_task_types_and_description(tmp_path: Path) -> None:
    folder = _write(tmp_path / "wf", inputs="first: {value: 3}\n")

    definition = load_definition(folder, describe=True, schema_generator=OpaqueAnnotator)

    descriptor = definition["task_types"]["tests.conftest.AddOne"]
    assert descriptor["input_schema"]["x-test-type"] == "NumberInput"
    first = definition["describe"]["tasks"][0]
    assert first["output_schema"]["x-test-type"] == "NumberOutput"


def test_unresolvable_task_makes_definition_read_only(tmp_path: Path) -> None:
    workflow = WORKFLOW.replace("tests.conftest.Double", "tests.conftest.Missing")
    definition = load_definition(_write(tmp_path / "wf", workflow))

    assert definition["editable"] is False
    with pytest.raises(DefinitionError, match="read-only"):
        write_definition(definition, tmp_path / "out")


def test_invalid_yaml_raises_definition_error(tmp_path: Path) -> None:
    with pytest.raises(DefinitionError, match="parse error"):
        load_definition(_write(tmp_path / "wf", "workflow: [\n"))


def test_definition_from_registered_style_workflow(tmp_path: Path) -> None:
    folder = _write(tmp_path / "wf", inputs="first: {value: 3}\n")
    workflow, _config = load_workflow_only(folder / "workflow.yaml")

    definition = definition_from_workflow(workflow, None, {AddOne: "add_one"})

    first, second = definition["nodes"]
    assert first["task"] == "add_one"
    assert second["task"] == "tests.conftest.Double"
    assert definition["edges"] == [
        {"from": "first", "output": None, "to": "second", "input": None}
    ]


def test_describe_yaml_reports_errors_instead_of_raising(tmp_path: Path) -> None:
    folder = _write(tmp_path / "wf", inputs="first: {value: 3}\n")

    description, error = describe_yaml(folder / "workflow.yaml", folder / "input.yaml")
    assert error is None and description is not None
    assert description["tasks"][0]["provided_config_fields"] == ["value"]

    description, error = describe_yaml(tmp_path / "missing.yaml", None)
    assert description is None
    assert error is not None and error["code"] == "configuration_error"


def test_describe_yaml_workflow_raises_config_load_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigLoadError):
        describe_yaml_workflow(tmp_path / "missing.yaml")


def test_resolve_task_accepts_class_paths_and_rejects_non_tasks() -> None:
    assert resolve_task("tests.conftest.Double") is Double
    with pytest.raises(ConfigLoadError, match="not a Task subclass"):
        resolve_task("tests.conftest.NumberInput")


def test_public_json_helpers() -> None:
    assert jsonable({"path": Path("/tmp/x")}, "task") == {"path": "/tmp/x"}
    payload = error_payload("code", ValueError("boom"), "message", task="t")
    assert payload["task"] == "t" and payload["issues"] == []


def test_model_json_schema_adapts_non_model_types() -> None:
    assert model_json_schema(list[int]) == {"items": {"type": "integer"}, "type": "array"}


def test_definition_error_to_json() -> None:
    error = DefinitionError("bad", task="t", field="f")
    assert error.to_json() == {"message": "bad", "task": "t", "field": "f"}


# --- Loading YAML: graph features ---

COLLECTIONS = "tests.test_collections"

COLLECT_WORKFLOW = f"""\
# Collections.

workflow:
  name: collections
  result_task: listed
  tasks:
    - task: {COLLECTIONS}.ProduceSurface
      name: first
    - task: {COLLECTIONS}.ProduceEnvelope
      name: second
    - task: {COLLECTIONS}.CollectSurfaceList
      name: listed
      depends_on:
        surfaces:
          collect: [first, [second, surface]]
    - task: {COLLECTIONS}.CollectSurfaceDict
      name: keyed
      depends_on:
        surfaces:
          collect: {{top: first, base: [second, surface]}}
extra:
  note: kept
"""


def test_collections_result_task_and_passthrough_round_trip(tmp_path: Path) -> None:
    inputs = "first: {value: 1}\nsecond: {value: 2}\nunknown: {value: 3}\n"
    definition = load_definition(_write(tmp_path / "src", COLLECT_WORKFLOW, inputs))

    assert definition["result_task"] == "listed"
    assert definition["passthrough"] == {"extra": {"note": "kept"}}
    assert "Dropped input values for unknown task 'unknown'" in definition["warnings"]
    assert definition["edges"] == [
        {"from": "first", "output": None, "to": "listed", "input": "surfaces",
         "collect": "list", "key": None},
        {"from": "second", "output": "surface", "to": "listed", "input": "surfaces",
         "collect": "list", "key": None},
        {"from": "first", "output": None, "to": "keyed", "input": "surfaces",
         "collect": "dict", "key": "top"},
        {"from": "second", "output": "surface", "to": "keyed", "input": "surfaces",
         "collect": "dict", "key": "base"},
    ]  # fmt: skip

    result = write_definition(definition, tmp_path / "out")

    reloaded = load_definition(tmp_path / "out")
    for key in ("nodes", "edges", "inputs", "result_task", "passthrough", "header_comment"):
        assert reloaded[key] == definition[key]
    loaded = load_workflow_from_yaml(result["workflow_yaml"], result["input_yaml"])
    assert loaded.run().status == JobStatus.COMPLETED


MAP_WORKFLOW = """\
workflow:
  name: mapped
  tasks:
    - task: tests.conftest.AddOne
    - task: tests.test_mapping.MappedNumber
      depends_on: {base: tests.conftest.AddOne}
      config_fields: [multiplier]
      map: {over: items, key_as: item_name, value_as: amount, error_mode: collect_all}
    - task: tests.test_mapping.SumAggregate
      depends_on: {values: [tests.test_mapping.MappedNumber, root]}
"""


def test_mapped_task_round_trip(tmp_path: Path) -> None:
    inputs = (
        "add_one: {value: 1}\n"
        "mapped_number: {multiplier: 2, items: {a: 1, b: 2}, stray: 5}\n"
        "sum_aggregate:\n"
    )
    definition = load_definition(_write(tmp_path / "src", MAP_WORKFLOW, inputs))

    mapped = definition["nodes"][1]
    assert mapped["map"] == {
        "over": "items",
        "key_as": "item_name",
        "value_as": "amount",
        "error_mode": "collect_all",
    }
    assert definition["inputs"]["mapped_number"] == {"multiplier": 2, "items": {"a": 1, "b": 2}}
    assert definition["inputs"]["sum_aggregate"] == {}
    assert any("mapped_number.stray" in warning for warning in definition["warnings"])

    result = write_definition(definition, tmp_path / "out")

    assert "error_mode: collect_all" in Path(result["workflow_yaml"]).read_text()
    assert load_definition(tmp_path / "out")["inputs"] == definition["inputs"]
    loaded = load_workflow_from_yaml(result["workflow_yaml"], result["input_yaml"])
    assert loaded.run().status == JobStatus.COMPLETED


LINEAR_WORKFLOW = """\
workflow:
  name: linear
  tasks:
    - task: tests.conftest.AddOne
    - task: tests.conftest.Double
      name: twice
    - task: tests.conftest.Double
      name: again
"""


def test_linear_workflow_is_chained_and_saved_explicitly(tmp_path: Path) -> None:
    definition = load_definition(_write(tmp_path / "src", LINEAR_WORKFLOW))

    assert [(edge["from"], edge["to"]) for edge in definition["edges"]] == [
        ("add_one", "twice"),
        ("twice", "again"),
    ]
    assert any("chained in file order" in warning for warning in definition["warnings"])
    assert list(definition["task_types"]) == ["tests.conftest.AddOne", "tests.conftest.Double"]

    definition["edges"] = []
    result = write_definition(definition, tmp_path / "out", workflow_only=True)

    assert result["input_yaml"] is None
    assert "depends_on: {}" in Path(result["workflow_yaml"]).read_text()


def test_wired_config_fields_are_dropped(tmp_path: Path) -> None:
    workflow = """\
workflow:
  name: wired
  tasks:
    - task: tests.conftest.AddOne
      name: first
    - task: tests.conftest.Double
      name: second
      config_fields: [value]
      depends_on: {value: [first, value]}
"""
    definition = load_definition(_write(tmp_path / "wf", workflow, "second: {value: 5}\n"))

    assert definition["nodes"][1]["config_fields"] == []
    assert definition["inputs"]["second"] == {}
    assert any("connected to an upstream task" in w for w in definition["warnings"])


def test_nested_workflows_make_definition_read_only(tmp_path: Path) -> None:
    folder = tmp_path / "wf"
    _write(folder, "workflow:\n  name: outer\n  tasks:\n"
                   "    - workflow: inner.yaml\n    - workflow: missing.yaml\n")  # fmt: skip
    (folder / "inner.yaml").write_text(WORKFLOW)

    definition = load_definition(folder)

    assert [node["name"] for node in definition["nodes"]] == ["pipeline", "missing"]
    assert definition["nodes"][0]["workflow"] == "inner.yaml"
    assert definition["editable"] is False
    # Values of read-only definitions are kept even when they are not config fields.
    assert definition["inputs"] == {"pipeline": {}, "missing": {}}


def test_undescribable_task_makes_definition_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any) -> dict[str, Any]:
        raise ValueError("no schema")

    monkeypatch.setattr("taskmaestro.definition.task_descriptor", fail)

    definition = load_definition(_write(tmp_path / "wf"))

    assert (
        "Cannot describe task 'tests.conftest.AddOne': no schema"
        in (definition["readonly_reasons"])
    )


def _two_add_ones(depends_on: str) -> str:
    return f"""\
workflow:
  name: refs
  tasks:
    - task: tests.conftest.AddOne
      name: a
    - task: tests.conftest.AddOne
      name: b
    - task: tests.conftest.Double
      depends_on: {depends_on}
"""


@pytest.mark.parametrize(
    ("workflow", "inputs", "match"),
    [
        ("workflow: [\n", None, "YAML parse error"),
        ("- a\n", None, "mapping at top level"),
        ("workflow: {}\n", None, "schema validation error"),
        (WORKFLOW + "extra: .nan\n", None, "not a finite number"),
        (WORKFLOW, "- a\n", "Input YAML file must contain a mapping"),
        (WORKFLOW, "first: 3\n", "Input value for task 'first' must be a mapping"),
        (WORKFLOW, "first: [\n", "Input YAML parse error"),
        (_two_add_ones("missing"), None, "Dependency 'missing' for task 'double' not found"),
        (_two_add_ones("tests.conftest.AddOne"), None, "is ambiguous"),
        (_two_add_ones("[a, b, c]"), None, "Invalid dependency \\['a', 'b', 'c'\\] on"),
        (_two_add_ones("{value: [a, b, c]}"), None, "for field 'value'"),
        (_two_add_ones("{value: {collect: a}}"), None, "must be a list or mapping"),
        (
            LINEAR_WORKFLOW.replace("      name: again\n", "").replace("      name: twice\n", ""),
            None,
            "Duplicate task name",
        ),
        (WORKFLOW.replace("tasks:", "result_task: missing\n  tasks:"), None, "not found"),
    ],
)
def test_load_definition_errors(
    tmp_path: Path, workflow: str, inputs: str | None, match: str
) -> None:
    with pytest.raises(DefinitionError, match=match):
        load_definition(_write(tmp_path / "wf", workflow, inputs))


def test_load_definition_reports_unreadable_folder(tmp_path: Path) -> None:
    with pytest.raises(DefinitionError, match="Cannot read"):
        load_definition(tmp_path / "missing")


# --- Registered workflows ---


def test_definition_from_builder_workflow_with_collections_and_map() -> None:
    from tests import test_collections as coll
    from tests import test_mapping as mapping

    workflow = (
        Workflow.builder("built", result_task="first")
        .add_task(coll.ProduceSurface, name="first")
        .add_task(coll.ProduceEnvelope, name="second")
        .add_task(
            coll.CollectSurfaceList,
            name="listed",
            depends_on={"surfaces": collect("first", ("second", "surface"))},
        )
        .add_task(
            coll.CollectSurfaceDict,
            name="keyed",
            depends_on={"surfaces": collect(top="first", base=("second", "surface"))},
        )
        .add_task(AddOne, name="seed")
        .add_task(mapping.ProduceEnvelope, name="envelope")
        .add_task(Double, name="whole", depends_on=("envelope", "number"))
        .add_task(
            mapping.MappedNumber,
            depends_on={"base": "seed"},
            config_fields=["multiplier"],
            mapped_over=TaskMap(over="items", key_as="item_name", value_as="amount"),
        )
        .add_task(mapping.SumAggregate, depends_on={"values": ("mapped_number", "root")})
        .add_task(FanInTask, name="fan_in", depends_on={"a": ("envelope", "number"), "b": "whole"})
        .build()
    )

    definition = definition_from_workflow(
        workflow, "registered", {}, describe=True, schema_generator=OpaqueAnnotator
    )

    assert definition["editable"] is True
    assert definition["source"]["registered_id"] == "registered"
    assert definition["result_task"] == "first"
    edges = {(e["from"], e["to"], e["input"], e.get("key")) for e in definition["edges"]}
    assert ("second", "keyed", "surfaces", "base") in edges
    assert ("first", "listed", "surfaces", None) in edges
    assert ("envelope", "whole", None, None) in edges
    assert ("envelope", "fan_in", "a", None) in edges
    assert ("whole", "fan_in", "b", None) in edges
    mapped = next(n for n in definition["nodes"] if n["name"] == "mapped_number")
    assert mapped["map"]["over"] == "items"
    assert definition["describe"]["tasks"][0]["input_schema"]["x-test-type"] == "NumberInput"


def test_definition_from_workflow_with_unreferenceable_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LocalTask(Task[NumberInput, NumberOutput]):
        name = "local"

        def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
            return NumberOutput(value=input.value)

    Renamed = type("Renamed", (LocalTask,), {"name": "renamed", "__qualname__": "Renamed"})
    Renamed.__module__ = "tests.no_such_module"
    Shadowed = type("AddOne", (LocalTask,), {"name": "shadowed", "__qualname__": "AddOne"})
    Shadowed.__module__ = "tests.conftest"
    inner = Workflow.builder("inner").add_task(AddOne).build()

    workflow = (
        Workflow.builder("outer", result_task="nested")
        .add_task(LocalTask)
        .add_task(Renamed)
        .add_task(Shadowed)
        .add_task(workflow_task(inner, name="nested"))
        .build()
    )

    def fail(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("cannot describe")

    monkeypatch.setattr("taskmaestro.definition.describe_workflow", fail)

    definition = definition_from_workflow(workflow, None, {}, describe=True)

    assert definition["editable"] is False
    reasons = definition["readonly_reasons"]
    assert sum("cannot be referenced from YAML" in reason for reason in reasons) == 3
    assert "Task 'nested' is a nested workflow" in reasons
    assert definition["describe"] is None
    assert definition["describe_error"] == {"code": "describe_error", "message": "cannot describe"}


# --- Saving: validation ---


def _saveable(tmp_path: Path) -> dict[str, Any]:
    return load_definition(_write(tmp_path / "src", COLLECT_WORKFLOW))


def _edit(definition: dict[str, Any], change: str) -> None:
    nodes = definition["nodes"]
    edges = definition["edges"]
    if change == "name":
        definition["name"] = "not valid"
    elif change == "no_nodes":
        definition["nodes"] = []
    elif change == "node_name":
        nodes[0]["name"] = "1st"
    elif change == "duplicate_node":
        nodes[1]["name"] = "first"
    elif change == "no_task":
        nodes[0]["task"] = None
    elif change == "unknown_edge":
        edges[0]["from"] = "ghost"
    elif change == "result_task":
        definition["result_task"] = "ghost"
    elif change == "mixed_whole_and_fields":
        edges.append({"from": "second", "output": None, "to": "listed", "input": None})
    elif change == "two_whole":
        edges[:] = [
            {"from": "first", "output": None, "to": "second", "input": None},
            {"from": "listed", "output": None, "to": "second", "input": None},
        ]
    elif change == "double_field":
        edges[:] = [
            {"from": "first", "output": None, "to": "second", "input": "value"},
            {"from": "listed", "output": None, "to": "second", "input": "value"},
        ]
    elif change == "collect_kind":
        edges[0]["collect"] = "set"
    elif change == "collect_mixed":
        edges[1]["collect"] = "dict"
        edges[1]["key"] = "k"
    elif change == "collect_and_direct":
        edges.insert(0, {"from": "second", "output": None, "to": "listed", "input": "surfaces"})
    elif change == "collect_no_key":
        edges[2]["key"] = ""
    elif change == "collect_duplicate_key":
        edges[3]["key"] = "top"
    else:
        nodes[0]["map"] = {
            "map_type": "oops",
            "map_name": {"over": "1", "key_as": "k", "value_as": "v"},
            "map_same": {"over": "i", "key_as": "k", "value_as": "k"},
            "map_mode": {"over": "i", "key_as": "k", "value_as": "v", "error_mode": "x"},
        }[change]


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ("name", "Invalid workflow name"),
        ("no_nodes", "no tasks"),
        ("node_name", "Invalid task name '1st'"),
        ("duplicate_node", "Duplicate task name 'first'"),
        ("no_task", "has no task type"),
        ("unknown_edge", "unknown task 'ghost'"),
        ("result_task", "Unknown result task 'ghost'"),
        ("mixed_whole_and_fields", "whole input"),
        ("two_whole", "more than one connection to its whole input"),
        ("double_field", "'second.value' has more than one connection"),
        ("collect_kind", "Unknown collect kind 'set'"),
        ("collect_mixed", "mixes collected and direct"),
        ("collect_and_direct", "mixes collected and direct"),
        ("collect_no_key", "has no key"),
        ("collect_duplicate_key", "Duplicate key 'top'"),
        ("map_type", "Invalid map for task"),
        ("map_name", "Invalid map 'over'"),
        ("map_same", "for both the key and the value"),
        ("map_mode", "Invalid map error mode 'x'"),
    ],
)
def test_write_definition_rejects_invalid_definitions(
    tmp_path: Path, change: str, match: str
) -> None:
    definition = _saveable(tmp_path)
    _edit(definition, change)

    with pytest.raises(DefinitionError, match=match):
        write_definition(definition, tmp_path / "out")


def test_failed_write_keeps_the_original_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = _write(tmp_path / "wf")
    definition = load_definition(folder)

    def fail(*args: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("taskmaestro.definition.os.replace", fail)

    with pytest.raises(OSError, match="disk full"):
        write_definition(definition, folder)
    assert (folder / "workflow.yaml").read_text() == WORKFLOW
    assert sorted(path.name for path in folder.iterdir()) == ["workflow.yaml"]
