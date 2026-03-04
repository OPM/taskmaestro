"""Example: Image Processing Pipeline (workflow_task composition)

This pipeline demonstrates wrapping an entire inner workflow as a single
opaque task using ``workflow_task``.  The outer runner sees only three
tasks, while the inner DAG with fan-out and fan-in runs transparently.

Inner workflow (image_analysis) — DAG with fan-out and fan-in:

                              +--> ReadImageMeta ---+
    ImagePath --> ValidateImage --+                     +--> BuildAnalysis
                              +--> ComputeHash -----+

Outer workflow (image_processing) — linear chain:

    ImageInput --> LoadImage --> analyze_image --> GenerateReport
                                 (workflow_task)

The outer runner sees ``analyze_image`` as a single TASK_START / TASK_COMPLETE.
The 4 inner tasks are invisible to the outer workflow.

Run:
    source .venv/bin/activate
    python examples/image_processing/pipeline.py                      # Python API
    python examples/image_processing/pipeline.py --yaml               # YAML config (defaults)
    python examples/image_processing/pipeline.py --yaml workflow.yaml --input input.yaml
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

from pydantic import BaseModel

from taskekrabbe import (
    ExecutionContext,
    Job,
    Runner,
    Task,
    Workflow,
    workflow_task,
)
from taskekrabbe.hooks import LoggingHook, TimingHook

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ImageInput(BaseModel):
    """Initial job config: path to image file."""

    image_path: str


class ImagePath(BaseModel):
    """Resolved absolute path to image file."""

    path: str


class ValidatedImage(BaseModel):
    """Output of ValidateImage: confirmed PNG with file size."""

    path: str
    file_size: int
    is_valid_png: bool


class ImageMeta(BaseModel):
    """PNG header metadata."""

    width: int
    height: int
    bit_depth: int
    color_type: int
    color_type_name: str


class FileHash(BaseModel):
    """SHA-256 hash of the file."""

    sha256: str


class AnalysisInput(BaseModel):
    """Fan-in input for BuildAnalysis: validated image, metadata, and hash."""

    validated: ValidatedImage
    meta: ImageMeta
    hash: FileHash


class ImageAnalysis(BaseModel):
    """Complete image analysis result."""

    path: str
    file_size: int
    width: int
    height: int
    bit_depth: int
    color_type: int
    color_type_name: str
    sha256: str
    megapixels: float
    file_size_kb: float


class ReportOutput(BaseModel):
    """Final report output."""

    title: str
    report: str
    image_path: str
    summary_lines: list[str]


# ---------------------------------------------------------------------------
# Inner workflow tasks
# ---------------------------------------------------------------------------


class ValidateImage(Task[ImagePath, ValidatedImage]):
    """Validate that the file exists and has a valid PNG header."""

    name = "validate_image"

    def run(self, input: ImagePath, ctx: ExecutionContext) -> ValidatedImage:
        ctx.logger.info("Validating image: %s", input.path)
        p = Path(input.path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {input.path}")
        file_size = p.stat().st_size
        with open(p, "rb") as f:
            header = f.read(8)
        is_valid_png = header == b"\x89PNG\r\n\x1a\n"
        if not is_valid_png:
            raise ValueError(f"Not a valid PNG file: {input.path}")
        return ValidatedImage(path=input.path, file_size=file_size, is_valid_png=True)


class ReadImageMeta(Task[ValidatedImage, ImageMeta]):
    """Read PNG IHDR chunk for image dimensions and color info."""

    name = "read_image_meta"

    def run(self, input: ValidatedImage, ctx: ExecutionContext) -> ImageMeta:
        ctx.logger.info("Reading PNG metadata from: %s", input.path)
        with open(input.path, "rb") as f:
            data = f.read(26)
        width, height = struct.unpack(">II", data[16:24])
        bit_depth = data[24]
        color_type = data[25]
        color_type_names = {
            0: "Grayscale",
            2: "Truecolor",
            3: "Indexed",
            4: "Grayscale+Alpha",
            6: "Truecolor+Alpha",
        }
        return ImageMeta(
            width=width,
            height=height,
            bit_depth=bit_depth,
            color_type=color_type,
            color_type_name=color_type_names.get(color_type, "Unknown"),
        )


class ComputeHash(Task[ValidatedImage, FileHash]):
    """Compute SHA-256 hash of the image file."""

    name = "compute_hash"

    def run(self, input: ValidatedImage, ctx: ExecutionContext) -> FileHash:
        ctx.logger.info("Computing SHA-256 hash for: %s", input.path)
        h = hashlib.sha256()
        with open(input.path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return FileHash(sha256=h.hexdigest())


class BuildAnalysis(Task[AnalysisInput, ImageAnalysis]):
    """Fan-in task: merge validation, metadata, and hash into analysis."""

    name = "build_analysis"

    def run(self, input: AnalysisInput, ctx: ExecutionContext) -> ImageAnalysis:
        ctx.logger.info("Building image analysis for: %s", input.validated.path)
        megapixels = round(input.meta.width * input.meta.height / 1_000_000, 2)
        file_size_kb = round(input.validated.file_size / 1024, 2)
        return ImageAnalysis(
            path=input.validated.path,
            file_size=input.validated.file_size,
            width=input.meta.width,
            height=input.meta.height,
            bit_depth=input.meta.bit_depth,
            color_type=input.meta.color_type,
            color_type_name=input.meta.color_type_name,
            sha256=input.hash.sha256,
            megapixels=megapixels,
            file_size_kb=file_size_kb,
        )


# ---------------------------------------------------------------------------
# Outer workflow tasks
# ---------------------------------------------------------------------------


class LoadImage(Task[ImageInput, ImagePath]):
    """Resolve the image path to an absolute path."""

    name = "load_image"

    def run(self, input: ImageInput, ctx: ExecutionContext) -> ImagePath:
        ctx.logger.info("Loading image path: %s", input.image_path)
        resolved = str(Path(input.image_path).resolve())
        return ImagePath(path=resolved)


class GenerateReport(Task[ImageAnalysis, ReportOutput]):
    """Generate a human-readable report from the image analysis."""

    name = "generate_report"

    def run(self, input: ImageAnalysis, ctx: ExecutionContext) -> ReportOutput:
        ctx.logger.info("Generating report for: %s", input.path)
        filename = Path(input.path).name
        summary_lines = [
            f"Image: {filename}",
            f"Dimensions: {input.width}x{input.height} ({input.megapixels} MP)",
            f"Color: {input.color_type_name}, {input.bit_depth}-bit",
            f"File size: {input.file_size_kb} KB",
            f"SHA-256: {input.sha256[:16]}...",
        ]
        report = "\n".join(f"  {line}" for line in summary_lines)
        return ReportOutput(
            title=f"Image Analysis: {filename}",
            report=report,
            image_path=input.path,
            summary_lines=summary_lines,
        )


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------

# Inner workflow defined at module level so both modes can print its Mermaid diagram
inner_workflow = (
    Workflow.builder(name="image_analysis")
    .add_task(ValidateImage)
    .add_task(ReadImageMeta, depends_on=ValidateImage)
    .add_task(ComputeHash, depends_on=ValidateImage)
    .add_task(
        BuildAnalysis,
        depends_on={
            "validated": ValidateImage,
            "meta": ReadImageMeta,
            "hash": ComputeHash,
        },
    )
    .build()
)

# Wrap the inner workflow as a single opaque task
AnalyzeImage = workflow_task(inner_workflow, name="analyze_image")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def print_report(
    result: Job[ImageInput],
    timing: TimingHook,
    outer_workflow: Workflow,
) -> None:
    """Print the analysis report, timings, and Mermaid diagrams."""
    report: ReportOutput = result.result  # type: ignore[assignment]

    print("=" * 60)
    print(f"  {report.title}")
    print("=" * 60)
    print()
    print(report.report)

    print(f"\n  Job status:       {result.status}")
    print(f"  Total duration:   {timing.job_duration:.4f}s")
    print("  Task timings:")
    for name, duration in timing.task_timings.items():
        print(f"    {name:20s} {duration:.4f}s")

    print()
    print("Workflow (with inner subgraph):")
    print("```mermaid")
    print(outer_workflow.to_mermaid(), end="")
    print("```")


def run_python_mode() -> None:
    """Run the pipeline using the Python API."""
    _dir = Path(__file__).resolve().parent
    image_path = str(_dir / ".." / ".." / "taskekrabbe.png")

    # Build the outer workflow
    outer_workflow = (
        Workflow.builder(name="image_processing")
        .add_task(LoadImage)
        .add_task(AnalyzeImage, depends_on=LoadImage)
        .add_task(GenerateReport, depends_on=AnalyzeImage)
        .build()
    )

    config = ImageInput(image_path=image_path)
    job = Job(workflow=outer_workflow, config=config)

    ctx = ExecutionContext()
    timing = TimingHook()
    runner = Runner(hooks=[LoggingHook(), timing])
    result = runner.run(job, ctx=ctx)

    print_report(result, timing, outer_workflow)


def run_yaml_mode(workflow_path: str, input_path: str) -> None:
    """Run the pipeline from the YAML workflow and input files."""
    from taskekrabbe.yaml_config import load_workflow_from_yaml

    loaded = load_workflow_from_yaml(workflow_path, input_path)
    result = loaded.run()

    timing = next(
        (h for h in loaded.runner.hooks if isinstance(h, TimingHook)),
        None,
    )
    assert timing is not None, "TimingHook not found in loaded runner hooks"

    print_report(result, timing, loaded.workflow)


def main() -> None:
    import argparse

    _dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Image Processing Pipeline example")
    parser.add_argument(
        "--yaml",
        metavar="FILE",
        nargs="?",
        const=str(_dir / "workflow.yaml"),
        help="Load workflow from a YAML config file (default: workflow.yaml)",
    )
    parser.add_argument(
        "--input",
        metavar="FILE",
        default=str(_dir / "input.yaml"),
        help="Input YAML file (default: input.yaml)",
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
