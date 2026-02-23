"""Tests for Mermaid diagram visualization."""

from __future__ import annotations

from pydantic import BaseModel

from workflow_runner import ExecutionContext, Task, Workflow, to_mermaid

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
