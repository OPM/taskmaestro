"""Example: Text Analysis Pipeline (DAG with fan-in)

This pipeline takes raw text, fans out to PrepareText and GenerateStopWords,
then fans in to ComputeWordStats and ExtractKeywords, while ScoreReadability
depends only on PrepareText, and all merge into a final BuildReport.

ExtractKeywords uses inline Inputs/Outputs port declarations and produces
two named outputs (.keywords, .num_words_removed) that are routed
independently to BuildReport via output field routing.

                  ┌── PrepareText ───────── ScoreReadability ──┐
    TextInput ──►│                 ↘              ↘            │
                  │           ComputeWordStats  ExtractKeywords ──► BuildReport
                  │                 ↗              ↗
                  └── GenerateStopWords ──────────┘

Run:
    source .venv/bin/activate
    python example.py                                              # Python API
    python example.py --yaml                                       # YAML config (defaults)
    python example.py --yaml example.yaml --input example_input.yaml  # custom files
"""

from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel

from taskekrabbe import (
    ExecutionContext,
    Job,
    Runner,
    Task,
    Workflow,
)
from taskekrabbe.hooks import LoggingHook, TimingHook

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TextInput(BaseModel):
    """Initial job config: raw text to analyze."""

    text: str
    title: str = "Untitled"


class TextContent(BaseModel):
    """Output of PrepareText: passes through text and title."""

    text: str
    title: str


class StopWordsOutput(BaseModel):
    """Output of GenerateStopWords."""

    stop_words: list[str]


class AnalysisInput(BaseModel):
    """Fan-in input for word_stats and keywords tasks."""

    content: TextContent
    stop_words: StopWordsOutput


class WordStatsOutput(BaseModel):
    word_count: int
    sentence_count: int
    avg_word_length: float
    most_common: list[tuple[str, int]]


class KeywordsOutput(BaseModel):
    keywords: list[str]
    bigrams: list[str]


class ReadabilityOutput(BaseModel):
    flesch_score: float
    grade_level: str


class ReportInput(BaseModel):
    """Fan-in: each field sourced from a different upstream task.

    ``keywords`` and ``num_words_removed`` are routed from individual
    fields of ExtractKeywords.Outputs via output field routing.
    """

    stats: WordStatsOutput
    keywords: KeywordsOutput
    readability: ReadabilityOutput
    num_words_removed: int


class AnalysisReport(BaseModel):
    title: str
    summary: str
    word_count: int
    sentence_count: int
    avg_word_length: float
    num_words_removed: int
    top_words: list[str]
    keywords: list[str]
    bigrams: list[str]
    flesch_score: float
    grade_level: str


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

class PrepareText(Task[TextInput, TextContent]):
    """Passthrough task: provides text content to downstream tasks."""

    name = "prepare_text"

    def run(self, input: TextInput, ctx: ExecutionContext) -> TextContent:
        ctx.logger.info("Preparing text for '%s'", input.title)
        return TextContent(text=input.text, title=input.title)


class GenerateStopWords(Task[TextInput, StopWordsOutput]):
    """Generate the set of stop words to filter out during analysis."""

    name = "generate_stop_words"

    def run(self, input: TextInput, ctx: ExecutionContext) -> StopWordsOutput:
        ctx.logger.info("Generating stop words list")
        stop_words = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "ought",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "and", "but", "or",
            "nor", "not", "so", "yet", "both", "either", "neither", "each",
            "every", "all", "any", "few", "more", "most", "other", "some",
            "such", "no", "only", "own", "same", "than", "too", "very",
            "just", "because", "it", "its", "this", "that", "these", "those",
            "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
            "she", "her", "they", "them", "their", "what", "which", "who",
        ]
        return StopWordsOutput(stop_words=stop_words)


class ComputeWordStats(Task[AnalysisInput, WordStatsOutput]):
    """Count words, sentences, and find the most common words."""

    name = "compute_word_stats"

    def run(self, input: AnalysisInput, ctx: ExecutionContext) -> WordStatsOutput:
        text = input.content.text
        stop_words = set(input.stop_words.stop_words)
        ctx.logger.info("Computing word statistics for '%s'", input.content.title)
        words = re.findall(r"[a-zA-Z']+", text.lower())
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        meaningful = [w for w in words if w not in stop_words and len(w) > 2]
        counter = Counter(meaningful)

        return WordStatsOutput(
            word_count=len(words),
            sentence_count=len(sentences),
            avg_word_length=sum(len(w) for w in words) / max(len(words), 1),
            most_common=counter.most_common(5),
        )


class ExtractKeywords(Task):  # type: ignore[type-arg]
    """Extract keywords (top frequent non-stop words) and bigrams.

    Uses inline Inputs/Outputs inner classes instead of Task[I, O] generics
    to demonstrate the named-ports API.  The Outputs model carries both the
    KeywordsOutput and a ``num_words_removed`` counter that is routed
    independently to downstream tasks via output field routing.
    """

    name = "extract_keywords"

    class Inputs(BaseModel):
        content: TextContent
        stop_words: StopWordsOutput

    class Outputs(BaseModel):
        keywords: KeywordsOutput
        num_words_removed: int

    def run(self, input: Inputs, ctx: ExecutionContext) -> Outputs:
        text = input.content.text
        stop_words = set(input.stop_words.stop_words)
        ctx.logger.info("Extracting keywords for '%s'", input.content.title)
        words = re.findall(r"[a-zA-Z']+", text.lower())
        meaningful = [w for w in words if w not in stop_words and len(w) > 2]
        num_words_removed = len(words) - len(meaningful)

        # Top keywords by frequency
        counter = Counter(meaningful)
        keywords = [word for word, _ in counter.most_common(8)]

        # Bigrams from meaningful words
        bigram_counter = Counter(
            f"{meaningful[i]} {meaningful[i + 1]}" for i in range(len(meaningful) - 1)
        )
        bigrams = [bg for bg, _ in bigram_counter.most_common(5)]

        return self.Outputs(
            keywords=KeywordsOutput(keywords=keywords, bigrams=bigrams),
            num_words_removed=num_words_removed,
        )


class ScoreReadability(Task[TextContent, ReadabilityOutput]):
    """Compute a simplified Flesch reading ease score."""

    name = "score_readability"

    def run(self, input: TextContent, ctx: ExecutionContext) -> ReadabilityOutput:
        ctx.logger.info("Scoring readability for '%s'", input.title)
        words = re.findall(r"[a-zA-Z']+", input.text.lower())
        sentences = [s.strip() for s in re.split(r"[.!?]+", input.text) if s.strip()]

        num_words = max(len(words), 1)
        num_sentences = max(len(sentences), 1)
        num_syllables = sum(self._count_syllables(w) for w in words)

        # Flesch Reading Ease formula
        score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
        score = max(0.0, min(100.0, score))

        return ReadabilityOutput(
            flesch_score=round(score, 1),
            grade_level=self._score_to_grade(score),
        )

    @staticmethod
    def _count_syllables(word: str) -> int:
        word = word.lower().rstrip("e")
        vowels = re.findall(r"[aeiouy]+", word)
        return max(len(vowels), 1)

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 90:
            return "5th grade (very easy)"
        if score >= 80:
            return "6th grade (easy)"
        if score >= 70:
            return "7th grade (fairly easy)"
        if score >= 60:
            return "8th-9th grade (standard)"
        if score >= 50:
            return "10th-12th grade (fairly difficult)"
        if score >= 30:
            return "College (difficult)"
        return "College graduate (very difficult)"


class BuildReport(Task[ReportInput, AnalysisReport]):
    """Fan-in task: merge outputs from all analysis branches into a report."""

    name = "build_report"

    def run(self, input: ReportInput, ctx: ExecutionContext) -> AnalysisReport:
        title = ctx.resolve("title")
        ctx.logger.info("Building final report for '%s'", title)

        return AnalysisReport(
            title=title,
            summary=(
                f"Analyzed {input.stats.word_count} words across "
                f"{input.stats.sentence_count} sentences. "
                f"Readability: {input.readability.grade_level}."
            ),
            word_count=input.stats.word_count,
            sentence_count=input.stats.sentence_count,
            avg_word_length=round(input.stats.avg_word_length, 2),
            num_words_removed=input.num_words_removed,
            top_words=[w for w, _ in input.stats.most_common],
            keywords=input.keywords.keywords,
            bigrams=input.keywords.bigrams,
            flesch_score=input.readability.flesch_score,
            grade_level=input.readability.grade_level,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """\
Python is a high-level programming language known for its clear syntax and \
readability. Python supports multiple programming paradigms, including \
structured, object-oriented, and functional programming. Its comprehensive \
standard library and active community make Python an excellent choice for \
web development, data science, artificial intelligence, and automation. \
Python's design philosophy emphasizes code readability and simplicity, \
allowing developers to express concepts in fewer lines of code compared to \
languages like C++ or Java. The language continues to grow in popularity, \
consistently ranking among the top programming languages worldwide.\
"""


def print_report(result: Job[TextInput], timing: TimingHook, workflow: Workflow) -> None:
    """Print the analysis report, timings, and Mermaid diagram."""
    report: AnalysisReport = result.result  # type: ignore[assignment]

    print("=" * 60)
    print(f"  {report.title} — Analysis Report")
    print("=" * 60)
    print(f"\n  {report.summary}\n")
    print(f"  Words:            {report.word_count}")
    print(f"  Stop words removed: {report.num_words_removed}")
    print(f"  Sentences:        {report.sentence_count}")
    print(f"  Avg word length:  {report.avg_word_length} chars")
    print(f"  Flesch score:     {report.flesch_score}")
    print(f"  Grade level:      {report.grade_level}")
    print(f"\n  Top words:        {', '.join(report.top_words)}")
    print(f"  Keywords:         {', '.join(report.keywords)}")
    print(f"  Top bigrams:      {', '.join(report.bigrams)}")

    print(f"\n  Job status:       {result.status}")
    print(f"  Total duration:   {timing.job_duration:.4f}s")
    print("  Task timings:")
    for name, duration in timing.task_timings.items():
        print(f"    {name:20s} {duration:.4f}s")
    print()

    # Print Mermaid diagram
    print("Mermaid diagram:")
    print("```mermaid")
    print(workflow.to_mermaid(), end="")
    print("```")


def run_python_mode() -> None:
    """Run the pipeline using the Python API."""
    # Build the DAG workflow
    workflow = (
        Workflow.builder(name="text_analysis")
        .add_task(PrepareText)                               # root (receives TextInput)
        .add_task(GenerateStopWords)                         # root (receives TextInput)
        .add_task(ComputeWordStats, depends_on={             # fan-in
            "content": PrepareText,
            "stop_words": GenerateStopWords,
        })
        .add_task(ExtractKeywords, depends_on={              # fan-in
            "content": PrepareText,
            "stop_words": GenerateStopWords,
        })
        .add_task(ScoreReadability, depends_on=PrepareText)  # single dep
        .add_task(BuildReport, depends_on={                  # fan-in from all three
            "stats": ComputeWordStats,
            "keywords": (ExtractKeywords, "keywords"),           # output field routing
            "readability": ScoreReadability,
            "num_words_removed": (ExtractKeywords, "num_words_removed"),  # output field routing
        })
        .build()
    )

    # Create the job
    config = TextInput(text=SAMPLE_TEXT, title="Python Overview")
    job = Job(workflow=workflow, config=config)

    # Set up context with a service
    ctx = ExecutionContext()
    ctx.register("title", config.title)

    # Run with hooks
    timing = TimingHook()
    runner = Runner(hooks=[LoggingHook(), timing])
    result = runner.run(job, ctx=ctx)

    print_report(result, timing, workflow)


def run_yaml_mode(workflow_path: str, input_path: str) -> None:
    """Run the pipeline from the YAML workflow and input files."""
    from taskekrabbe.yaml_config import load_workflow_from_yaml

    loaded = load_workflow_from_yaml(workflow_path, input_path)
    result = loaded.run()

    # Find TimingHook from the runner's hook list
    timing = next(
        (h for h in loaded.runner.hooks if isinstance(h, TimingHook)),
        None,
    )
    assert timing is not None, "TimingHook not found in loaded runner hooks"

    print_report(result, timing, loaded.workflow)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Text Analysis Pipeline example")
    parser.add_argument(
        "--yaml", metavar="FILE", nargs="?", const="example.yaml",
        help="Load workflow from a YAML config file (default: example.yaml)",
    )
    parser.add_argument(
        "--input", metavar="FILE", default="example_input.yaml",
        help="Input YAML file (default: example_input.yaml)",
    )
    args = parser.parse_args()

    if args.yaml:
        run_yaml_mode(args.yaml, args.input)
    else:
        run_python_mode()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="  %(name)s — %(message)s")
    main()
