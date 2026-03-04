"""Tests for Mermaid diagram visualization."""

from __future__ import annotations

from pydantic import BaseModel

from taskekrabbe import ExecutionContext, ObjectModel, Task, Workflow, to_mermaid

# --- Models for fan-in DAG test ---


class TextInput(BaseModel):
    text: str


class WordStatsOutput(BaseModel):
    word_count: int


class KeywordsOutput(BaseModel):
    keywords: list[str]


class ReportInput(BaseModel):
    stats: WordStatsOutput
    keywords: KeywordsOutput


class Report(BaseModel):
    summary: str


# --- Tasks ---


class WordStats(Task[TextInput, WordStatsOutput]):
    name = "word_stats"

    def run(self, input: TextInput, ctx: ExecutionContext) -> WordStatsOutput:
        return WordStatsOutput(word_count=len(input.text.split()))


class Keywords(Task[TextInput, KeywordsOutput]):
    name = "keywords"

    def run(self, input: TextInput, ctx: ExecutionContext) -> KeywordsOutput:
        return KeywordsOutput(keywords=input.text.split()[:3])


class BuildReport(Task[ReportInput, Report]):
    name = "build_report"

    def run(self, input: ReportInput, ctx: ExecutionContext) -> Report:
        return Report(summary="done")


# --- Tests ---


class TestLinearWorkflow:
    def test_linear_three_tasks(self) -> None:
        from tests.conftest import AddOne, Double, Stringify

        wf = Workflow("linear", [AddOne, Double, Stringify])
        result = to_mermaid(wf)

        # Check title and header
        assert result.startswith("---\ntitle: linear\n---\ngraph TD\n")
        assert '(("start"))' in result

        # Check plain node definitions
        assert '"add_one"' in result
        assert '"double"' in result
        assert '"stringify"' in result

        # Check start edge to root task with input type
        assert "_start_ -->|NumberInput| add_one" in result

        # Check edges labeled with output types
        assert "add_one -->|NumberOutput| double" in result
        assert "double -->|NumberOutput| stringify" in result

        # Check end node and edge from sink task
        assert '(("end"))' in result
        assert "stringify -->|StringOutput| _end_" in result


class TestFanInDAG:
    def test_fan_in_edges_have_labels(self) -> None:
        wf = (
            Workflow.builder("text_analysis")
            .add_task(WordStats)
            .add_task(Keywords)
            .add_task(
                BuildReport,
                depends_on={"stats": WordStats, "keywords": Keywords},
            )
            .build()
        )
        result = wf.to_mermaid()

        # Check title and header
        assert result.startswith("---\ntitle: text_analysis\n---\ngraph TD\n")
        assert '(("start"))' in result

        # Check plain node definitions
        assert '"word_stats"' in result
        assert '"keywords"' in result
        assert '"build_report"' in result

        # Check start edges to root tasks with input type
        assert "_start_ -->|TextInput| word_stats" in result
        assert "_start_ -->|TextInput| keywords" in result

        # Check fan-in edges with field: type labels
        assert "keywords -->|keywords: KeywordsOutput| build_report" in result
        assert "word_stats -->|stats: WordStatsOutput| build_report" in result

        # Check end node and edge from sink task
        assert '(("end"))' in result
        assert "build_report -->|Report| _end_" in result

    def test_all_edges_labeled(self) -> None:
        wf = (
            Workflow.builder("text_analysis")
            .add_task(WordStats)
            .add_task(Keywords)
            .add_task(
                BuildReport,
                depends_on={"stats": WordStats, "keywords": Keywords},
            )
            .build()
        )
        result = wf.to_mermaid()
        lines = result.strip().splitlines()
        edge_lines = [line for line in lines if "-->" in line]
        # All edges should be labeled
        for line in edge_lines:
            assert "-->|" in line


class TestFieldRouting:
    """Tests for Mermaid edges with output field routing."""

    def test_tuple_dep_edge_label(self) -> None:
        from pydantic import BaseModel

        from tests.conftest import Double, NumberInput, NumberOutput

        class MultiOut(BaseModel):
            stats: NumberOutput
            other: NumberOutput

        class Producer(Task[NumberInput, MultiOut]):
            name = "producer"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> MultiOut:
                return MultiOut(stats=NumberOutput(value=0), other=NumberOutput(value=0))

        wf = (
            Workflow.builder("field_viz")
            .add_task(Producer)
            .add_task(Double, depends_on=(Producer, "stats"))
            .build()
        )
        result = to_mermaid(wf)

        # Edge should show .stats: NumberOutput
        assert "producer -->|.stats: NumberOutput| double" in result

    def test_fan_in_with_field_ref_edge_label(self) -> None:
        from pydantic import BaseModel

        from tests.conftest import (
            AddOneB,
            FanInTask,
            NumberInput,
            NumberOutput,
        )

        class MultiOut(BaseModel):
            a: NumberOutput
            b: NumberOutput

        class Producer(Task[NumberInput, MultiOut]):
            name = "producer"

            def run(self, input: NumberInput, ctx: ExecutionContext) -> MultiOut:
                return MultiOut(a=NumberOutput(value=0), b=NumberOutput(value=0))

        wf = (
            Workflow.builder("mixed_viz")
            .add_task(Producer)
            .add_task(AddOneB)
            .add_task(
                FanInTask,
                depends_on={
                    "a": (Producer, "a"),
                    "b": AddOneB,
                },
            )
            .build()
        )
        result = to_mermaid(wf)

        # Field-ref edge: "a: .a: NumberOutput"
        assert "producer -->|a: .a: NumberOutput| fan_in_task" in result
        # Whole-output edge: "b: NumberOutput"
        assert "add_one_b -->|b: NumberOutput| fan_in_task" in result


class TestSingleTask:
    def test_single_task_start_and_end_edges(self) -> None:
        wf = Workflow("single", [WordStats])
        result = to_mermaid(wf)

        assert result.startswith("---\ntitle: single\n---\ngraph TD\n")
        assert '(("start"))' in result
        assert '(("end"))' in result
        assert '"word_stats"' in result

        # Start -> task and task -> end
        assert "_start_ -->|TextInput| word_stats" in result
        assert "word_stats -->|WordStatsOutput| _end_" in result

        # Exactly two edges
        lines = result.strip().splitlines()
        edge_lines = [line for line in lines if "-->" in line]
        assert len(edge_lines) == 2


# --- Module-level alias for generic type alias resolution test ---

WrappedStr = ObjectModel[str]


class ProduceWrapped(Task[TextInput, WrappedStr]):
    name = "produce_wrapped"

    def run(self, input: TextInput, ctx: ExecutionContext) -> WrappedStr:
        return WrappedStr(value=input.text)


class ConsumeWrapped(Task[WrappedStr, Report]):
    name = "consume_wrapped"

    def run(self, input: WrappedStr, ctx: ExecutionContext) -> Report:
        return Report(summary=input.value)


class TestGenericAliasResolution:
    """Tests that module-level type aliases resolve to their alias name."""

    def test_alias_resolved_in_edge_labels(self) -> None:
        wf = (
            Workflow.builder("alias_viz")
            .add_task(ProduceWrapped)
            .add_task(ConsumeWrapped, depends_on=ProduceWrapped)
            .build()
        )
        result = to_mermaid(wf)

        # Edge should use alias name "WrappedStr", not "ObjectModel[str]"
        assert "produce_wrapped -->|WrappedStr| consume_wrapped" in result
        assert "ObjectModel" not in result

    def test_alias_fallback_without_context(self) -> None:
        """Without context_cls, generic types fall back to escaped brackets."""
        from taskekrabbe.visualization import _safe_type_name

        # No context class — can't scan any module, uses HTML escaping
        result = _safe_type_name(WrappedStr)
        assert "ObjectModel" in result
        assert "&lsaquo;" in result

    def test_alias_fallback_no_match_in_module(self) -> None:
        """Generic type not aliased in the context module uses escaped brackets."""
        from taskekrabbe.visualization import _safe_type_name

        # ObjectModel[int] is not assigned to any name in this module
        DynamicType = ObjectModel[int]
        result = _safe_type_name(DynamicType, ProduceWrapped)
        assert "&lsaquo;" in result


class TestConfigFieldsVisualization:
    """Tests for JobConfiguration node and dashed edges in Mermaid output."""

    def test_config_fields_add_job_config_node(self) -> None:
        """Workflow with config_fields shows _job_config_ node."""
        from tests.conftest import AddOne, MergeTask

        wf = (
            Workflow.builder("viz_cfg")
            .add_task(AddOne)
            .add_task(
                MergeTask,
                depends_on=AddOne,
                config_fields=["label"],
            )
            .build()
        )
        result = to_mermaid(wf)

        # Job config node present
        assert '_job_config_[("JobConfiguration")]' in result
        # Dashed edge to configured task
        assert "_job_config_ -.->|label| merge_task" in result

    def test_config_fields_root_task_no_start_edge(self) -> None:
        """Root task with config_fields should NOT have a _start_ edge."""
        from tests.conftest import ConfigOnlyTask

        wf = (
            Workflow.builder("viz_root_cfg")
            .add_task(ConfigOnlyTask, config_fields=["path", "count"])
            .build()
        )
        result = to_mermaid(wf)

        # No start edge for configured root task
        assert "_start_ -->|" not in result
        # But job config node and dashed edge are present
        assert '_job_config_[("JobConfiguration")]' in result
        assert "_job_config_ -.->|" in result

    def test_no_config_fields_no_extra_node(self) -> None:
        """Workflow without config_fields has no _job_config_ node."""
        wf = Workflow("no_cfg", [WordStats])
        result = to_mermaid(wf)

        assert "_job_config_" not in result
        assert "-.->|" not in result

    def test_multiple_configured_tasks(self) -> None:
        """Multiple tasks with config_fields all get dashed edges."""
        from tests.conftest import (
            AddOne,
            ConfigOnlyTask,
            MergeTask,
        )

        wf = (
            Workflow.builder("multi_cfg", result_task=MergeTask)
            .add_task(ConfigOnlyTask, name="root_cfg", config_fields=["path", "count"])
            .add_task(AddOne)
            .add_task(
                MergeTask,
                depends_on=AddOne,
                config_fields=["label"],
            )
            .build()
        )
        result = to_mermaid(wf)

        assert '_job_config_[("JobConfiguration")]' in result
        # Dashed edges to both configured tasks
        assert "_job_config_ -.->|label| merge_task" in result
        assert "_job_config_ -.->|count, path| root_cfg" in result
