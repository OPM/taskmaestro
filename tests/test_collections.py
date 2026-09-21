"""Tests for collecting multiple task outputs into one input field."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from taskmaestro import (
    ConfigLoadError,
    EmptyConfig,
    ExecutionContext,
    Job,
    JobStatus,
    Runner,
    Task,
    Workflow,
    WorkflowDefinitionError,
    collect,
    load_workflow_from_yaml,
)
from taskmaestro.workflow import _is_type_compatible
from tests.conftest import NumberInput


class Surface(BaseModel):
    name: str


class RegularSurface(Surface):
    source: str = "generated"


class ProduceSurface(Task[NumberInput, RegularSurface]):
    name = "produce_surface"

    def run(self, input: NumberInput, ctx: ExecutionContext) -> RegularSurface:
        return RegularSurface(name=f"{self.name}-{input.value}")


class SurfaceEnvelope(BaseModel):
    surface: RegularSurface
    ignored: str


class ProduceEnvelope(Task[NumberInput, SurfaceEnvelope]):
    name = "produce_envelope"

    def run(self, input: NumberInput, ctx: ExecutionContext) -> SurfaceEnvelope:
        return SurfaceEnvelope(
            surface=RegularSurface(name=f"{self.name}-{input.value}"),
            ignored="ignored",
        )


class SurfaceListInput(BaseModel):
    surfaces: list[Surface]


class SurfaceNames(BaseModel):
    names: list[str]


class CollectSurfaceList(Task[SurfaceListInput, SurfaceNames]):
    name = "collect_surface_list"

    def run(self, input: SurfaceListInput, ctx: ExecutionContext) -> SurfaceNames:
        return SurfaceNames(names=[surface.name for surface in input.surfaces])


class SurfaceDictInput(BaseModel):
    surfaces: dict[str, Surface]


class CollectSurfaceDict(Task[SurfaceDictInput, SurfaceNames]):
    name = "collect_surface_dict"

    def run(self, input: SurfaceDictInput, ctx: ExecutionContext) -> SurfaceNames:
        return SurfaceNames(
            names=[f"{key}:{surface.name}" for key, surface in input.surfaces.items()]
        )


class TextOutput(BaseModel):
    text: str


class ProduceText(Task[NumberInput, TextOutput]):
    name = "produce_text"

    def run(self, input: NumberInput, ctx: ExecutionContext) -> TextOutput:
        return TextOutput(text=str(input.value))


class TestCollectDeclaration:
    def test_dictionary_keys_must_be_strings(self) -> None:
        with pytest.raises(TypeError, match="keys must be strings"):
            collect({1: ProduceSurface})  # type: ignore[dict-item]

    def test_mapping_cannot_be_mixed_with_positional_members(self) -> None:
        with pytest.raises(TypeError, match="either positional members or one mapping"):
            collect(ProduceSurface, {"other": ProduceSurface})  # type: ignore[call-overload]


class TestCollectionWorkflow:
    def test_list_collects_outputs_in_declaration_order(self) -> None:
        workflow = (
            Workflow.builder("surface_list")
            .add_task(ProduceSurface, name="second")
            .add_task(ProduceSurface, name="first")
            .add_task(
                CollectSurfaceList,
                depends_on={"surfaces": collect("first", "second")},
            )
            .build()
        )

        result = Runner().run(Job(workflow, NumberInput(value=7)))

        assert result.status == JobStatus.COMPLETED
        assert result.result == SurfaceNames(names=["first-7", "second-7"])
        collection = workflow.get_dependencies("collect_surface_list")
        assert collection is not None

    def test_collects_whole_outputs_and_routed_fields(self) -> None:
        workflow = (
            Workflow.builder("routed_collection")
            .add_task(ProduceSurface)
            .add_task(ProduceEnvelope)
            .add_task(
                CollectSurfaceList,
                depends_on={
                    "surfaces": collect(
                        ProduceSurface,
                        (ProduceEnvelope, "surface"),
                    )
                },
            )
            .build()
        )

        result = Runner().run(Job(workflow, NumberInput(value=3)))

        assert result.status == JobStatus.COMPLETED
        assert result.result == SurfaceNames(names=["produce_surface-3", "produce_envelope-3"])

    def test_collects_keyed_outputs_in_declaration_order(self) -> None:
        workflow = (
            Workflow.builder("surface_dict")
            .add_task(ProduceSurface, name="top_task")
            .add_task(ProduceEnvelope, name="base_task")
            .add_task(
                CollectSurfaceDict,
                depends_on={
                    "surfaces": collect(
                        {
                            "top": "top_task",
                            "base": ("base_task", "surface"),
                        }
                    )
                },
            )
            .build()
        )

        result = Runner().run(Job(workflow, NumberInput(value=4)))

        assert result.status == JobStatus.COMPLETED
        assert result.result == SurfaceNames(names=["top:top_task-4", "base:base_task-4"])

    def test_empty_list_collection(self) -> None:
        workflow = (
            Workflow.builder("empty_collection")
            .add_task(
                CollectSurfaceList,
                depends_on={"surfaces": collect()},
            )
            .build()
        )

        result = Runner().run(Job(workflow, EmptyConfig()))

        assert result.status == JobStatus.COMPLETED
        assert result.result == SurfaceNames(names=[])

    def test_empty_dictionary_collection(self) -> None:
        workflow = (
            Workflow.builder("empty_dictionary")
            .add_task(
                CollectSurfaceDict,
                depends_on={"surfaces": collect({})},
            )
            .build()
        )

        result = Runner().run(Job(workflow, EmptyConfig()))

        assert result.status == JobStatus.COMPLETED
        assert result.result == SurfaceNames(names=[])

    def test_incompatible_member_is_rejected(self) -> None:
        with pytest.raises(WorkflowDefinitionError, match="Collection type mismatch"):
            (
                Workflow.builder("bad_collection")
                .add_task(ProduceText)
                .add_task(
                    CollectSurfaceList,
                    depends_on={"surfaces": collect(ProduceText)},
                )
                .build()
            )

    def test_missing_routed_output_field_is_rejected(self) -> None:
        with pytest.raises(WorkflowDefinitionError, match="Field 'missing' not found"):
            (
                Workflow.builder("missing_field")
                .add_task(ProduceEnvelope)
                .add_task(
                    CollectSurfaceList,
                    depends_on={"surfaces": collect((ProduceEnvelope, "missing"))},
                )
                .build()
            )

    def test_routed_output_field_type_mismatch_names_source_field(self) -> None:
        with pytest.raises(
            WorkflowDefinitionError,
            match=r"produce_envelope\.ignored.*collection element type is Surface",
        ):
            (
                Workflow.builder("bad_field_type")
                .add_task(ProduceEnvelope)
                .add_task(
                    CollectSurfaceList,
                    depends_on={"surfaces": collect((ProduceEnvelope, "ignored"))},
                )
                .build()
            )

    def test_collection_shape_must_match_field(self) -> None:
        with pytest.raises(WorkflowDefinitionError, match=r"requires a list\[T\] field"):
            (
                Workflow.builder("bad_shape")
                .add_task(ProduceSurface)
                .add_task(
                    CollectSurfaceDict,
                    depends_on={"surfaces": collect(ProduceSurface)},
                )
                .build()
            )

    def test_keyed_collection_requires_dictionary_field(self) -> None:
        with pytest.raises(WorkflowDefinitionError, match=r"requires a dict\[str, T\] field"):
            (
                Workflow.builder("bad_keyed_shape")
                .add_task(ProduceSurface)
                .add_task(
                    CollectSurfaceList,
                    depends_on={"surfaces": collect({"surface": ProduceSurface})},
                )
                .build()
            )

    def test_type_compatibility_handles_unions_and_parameterized_types(self) -> None:
        assert _is_type_compatible(RegularSurface, Surface | TextOutput)
        assert not _is_type_compatible(list[int], list[str])

    def test_collection_and_config_cannot_supply_same_field(self) -> None:
        with pytest.raises(WorkflowDefinitionError, match="supplied by both"):
            (
                Workflow.builder("conflicting_sources")
                .add_task(ProduceSurface)
                .add_task(
                    CollectSurfaceList,
                    depends_on={"surfaces": collect(ProduceSurface)},
                    config_fields=["surfaces"],
                )
                .build()
            )


class TestCollectionVisualization:
    def test_collection_uses_explicit_junction_node(self) -> None:
        workflow = (
            Workflow.builder("collection_viz")
            .add_task(ProduceSurface, name="top")
            .add_task(ProduceEnvelope, name="base")
            .add_task(
                CollectSurfaceList,
                depends_on={
                    "surfaces": collect("top", ("base", "surface")),
                },
            )
            .build()
        )

        diagram = workflow.to_mermaid()

        assert '_collect_collect_surface_list_surfaces_{{"collect surfaces"}}' in diagram
        assert "top -->|0: RegularSurface| _collect_collect_surface_list_surfaces_" in diagram
        assert "base -->|1: .surface: RegularSurface|" in diagram
        assert "-->|surfaces: list&lsaquo;Surface&rsaquo;|" in diagram

    def test_keyed_collection_edges_use_aliases(self) -> None:
        workflow = (
            Workflow.builder("keyed_collection_viz")
            .add_task(ProduceSurface, name="top")
            .add_task(
                CollectSurfaceDict,
                depends_on={"surfaces": collect({"top_alias": "top"})},
            )
            .build()
        )

        diagram = workflow.to_mermaid()

        assert "top -->|top_alias: RegularSurface|" in diagram
        assert "-->|surfaces: dict&lsaquo;str, Surface&rsaquo;|" in diagram


class TestCollectionYaml:
    def test_yaml_collection_end_to_end(self, tmp_path: Path) -> None:
        module = "tests.test_collections"
        workflow_path = tmp_path / "workflow.yaml"
        workflow_path.write_text(
            f"""\
workflow:
  name: yaml_collection
  tasks:
    - task: {module}.ProduceSurface
      name: first
    - task: {module}.ProduceEnvelope
      name: second
    - task: {module}.CollectSurfaceList
      depends_on:
        surfaces:
          collect:
            - first
            - [second, surface]
"""
        )
        input_path = tmp_path / "input.yaml"
        input_path.write_text("value: 9\n")

        result = load_workflow_from_yaml(workflow_path, input_path).run()

        assert result.status == JobStatus.COMPLETED
        assert result.result == SurfaceNames(names=["first-9", "second-9"])

    def test_yaml_keyed_collection_end_to_end(self, tmp_path: Path) -> None:
        module = "tests.test_collections"
        workflow_path = tmp_path / "workflow.yaml"
        workflow_path.write_text(
            f"""\
workflow:
  name: yaml_keyed_collection
  tasks:
    - task: {module}.ProduceSurface
      name: top_task
    - task: {module}.ProduceEnvelope
      name: base_task
    - task: {module}.CollectSurfaceDict
      depends_on:
        surfaces:
          collect:
            top: top_task
            base: [base_task, surface]
"""
        )
        input_path = tmp_path / "input.yaml"
        input_path.write_text("value: 5\n")

        result = load_workflow_from_yaml(workflow_path, input_path).run()

        assert result.status == JobStatus.COMPLETED
        assert result.result == SurfaceNames(names=["top:top_task-5", "base:base_task-5"])

    def test_yaml_empty_collection(self, tmp_path: Path) -> None:
        module = "tests.test_collections"
        workflow_path = tmp_path / "workflow.yaml"
        workflow_path.write_text(
            f"""\
workflow:
  name: yaml_empty_collection
  tasks:
    - task: {module}.CollectSurfaceList
      depends_on:
        surfaces:
          collect: []
"""
        )
        input_path = tmp_path / "input.yaml"
        input_path.write_text("{}\n")

        result = load_workflow_from_yaml(workflow_path, input_path).run()

        assert result.status == JobStatus.COMPLETED
        assert result.result == SurfaceNames(names=[])

    @pytest.mark.parametrize(
        ("dependency_yaml", "message"),
        [
            ("collect: [[producer]]", "Collection member must be"),
            ("collect: [123]", "Collection member must be"),
            ("collect: producer", "must contain a list or mapping"),
            ("collect: {1: producer}", "Collection keys must be strings"),
            ("unexpected: producer", "Invalid dependency"),
        ],
    )
    def test_invalid_yaml_collection_forms(
        self,
        tmp_path: Path,
        dependency_yaml: str,
        message: str,
    ) -> None:
        module = "tests.test_collections"
        workflow_path = tmp_path / "workflow.yaml"
        workflow_path.write_text(
            f"""\
workflow:
  name: invalid_collection
  tasks:
    - task: {module}.ProduceSurface
      name: producer
    - task: {module}.CollectSurfaceList
      depends_on:
        surfaces:
          {dependency_yaml}
"""
        )
        input_path = tmp_path / "input.yaml"
        input_path.write_text("value: 1\n")

        with pytest.raises(ConfigLoadError, match=message):
            load_workflow_from_yaml(workflow_path, input_path)

    def test_configured_root_without_per_task_input_is_rejected(self, tmp_path: Path) -> None:
        workflow_path = tmp_path / "workflow.yaml"
        workflow_path.write_text(
            """\
workflow:
  name: missing_task_configuration
  tasks:
    - task: tests.conftest.ConfigOnlyTask
      config_fields: [path, count]
"""
        )
        input_path = tmp_path / "input.yaml"
        input_path.write_text("{}\n")

        with pytest.raises(ConfigLoadError, match="no per-task input configuration"):
            load_workflow_from_yaml(workflow_path, input_path)

    def test_duplicate_yaml_collection_key_is_rejected(self, tmp_path: Path) -> None:
        module = "tests.test_collections"
        workflow_path = tmp_path / "workflow.yaml"
        workflow_path.write_text(
            f"""\
workflow:
  name: duplicate_key
  tasks:
    - task: {module}.ProduceSurface
      name: producer
    - task: {module}.CollectSurfaceDict
      depends_on:
        surfaces:
          collect:
            top: producer
            top: producer
"""
        )
        input_path = tmp_path / "input.yaml"
        input_path.write_text("value: 1\n")

        with pytest.raises(ConfigLoadError, match="duplicate key"):
            load_workflow_from_yaml(workflow_path, input_path)
