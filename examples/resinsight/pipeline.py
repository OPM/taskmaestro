"""Example: ResInsight Completions Pipeline (DAG with fan-out and fan-in)

This pipeline connects to a running ResInsight instance, loads a reservoir
model, imports two well paths in parallel, adds perforation events to each,
and merges everything into a single completions export.

                                    ┌──► LoadWellPath ──► AddPerforation ──┐
ConnectToResInsight ──► LoadModel ──┤     (well_1)        (perf_1)        ├──► ExportCompletions
                                    └──► LoadWellPath ──► AddPerforation ──┘
                                          (well_2)        (perf_2)
                                    │                                      ▲
                                    └──────────────────────────────────────┘

- Linear chain:  ConnectToResInsight → LoadModel (sequential setup)
- Fan-out:       LoadModel feeds two LoadWellPath instances in parallel
- Linear chains: LoadWellPath -> AddPerforation (x2)
- Fan-in:        ExportCompletions merges LoadModel + both AddPerforation instances

Per-task configuration: Each task declares what it needs as explicit input
model fields. A JobConfiguration provides per-task config values, and the
Runner merges upstream outputs with config values to construct task inputs.

Run:
    source .venv/bin/activate
    python examples/resinsight/pipeline.py              # Python API
    python examples/resinsight/pipeline.py --yaml       # YAML config (defaults)
    python examples/resinsight/pipeline.py --yaml workflow.yaml --input input.yaml
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import rips
from pydantic import BaseModel

from taskekrabbe import (
    EmptyConfig,
    ExecutionContext,
    Job,
    JobConfiguration,
    ObjectModel,
    Runner,
    Task,
    Workflow,
)
from taskekrabbe.hooks import LoggingHook, TimingHook
from taskekrabbe.hooks.base import BaseHook

# ---------------------------------------------------------------------------
# Custom hook — prints task start/end timestamps to stdout
# ---------------------------------------------------------------------------


class PrintTimestampHook(BaseHook):
    """A minimal custom hook that prints task start and end times to stdout."""

    def on_task_start(self, job: Job[Any], task: Task[Any, Any]) -> None:
        now = datetime.now(tz=UTC).isoformat(timespec="milliseconds")
        print(f"[START] {task.name}  at {now}")

    def on_task_complete(self, job: Job[Any], task: Task[Any, Any], output: BaseModel) -> None:
        now = datetime.now(tz=UTC).isoformat(timespec="milliseconds")
        print(f"[END]   {task.name}  at {now}")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ConnectionOutput(BaseModel):
    """Output of ConnectToResInsight: the gRPC port."""

    port: int


class FilePath(BaseModel):
    """A file path provided via config."""

    path: str


class GridCase(ObjectModel[rips.EclipseCase]):
    pass


class WellPath(ObjectModel[rips.WellPath]):
    pass


class AddPerforationInput(ObjectModel[rips.WellPath]):
    """Input for AddPerforation: well from upstream, MD range from config."""

    event_date: str
    start_md: float
    end_md: float


class PerforationOutput(ObjectModel[rips.WellPath]):
    """Output of AddPerforation: perforation interval details."""

    start_md: float
    end_md: float


# ---------------------------------------------------------------------------
# Tasks — each task gets all its data from explicit input fields
# ---------------------------------------------------------------------------


class ConnectToResInsight(Task[EmptyConfig, ConnectionOutput]):
    """Connect to a running ResInsight instance and register it in context."""

    name = "connect_to_resinsight"

    def run(self, input: EmptyConfig, ctx: ExecutionContext) -> ConnectionOutput:
        ctx.logger.info("Connecting to ResInsight...")
        instance = rips.Instance.find()
        ctx.register("resinsight", instance)
        port = int(instance.location.split(":")[1])
        ctx.logger.info("Connected to ResInsight on port %d", port)
        return ConnectionOutput(port=port)


class LoadModel(Task[FilePath, GridCase]):
    """Load the reservoir model (.egrid) into ResInsight."""

    name = "load_model"

    def run(self, input: FilePath, ctx: ExecutionContext) -> GridCase:
        instance: rips.Instance = ctx.resolve("resinsight")
        ctx.logger.info("Loading model from %s", input.path)
        grid_case = instance.project.load_case(input.path)
        ctx.logger.info("Loaded case '%s' (id=%d)", grid_case.name, grid_case.id)
        return GridCase(value=grid_case)


class LoadWellPath(Task[FilePath, WellPath]):
    """Import a well path file into ResInsight."""

    name = "load_well_path"

    def run(self, input: FilePath, ctx: ExecutionContext) -> WellPath:
        instance: rips.Instance = ctx.resolve("resinsight")
        ctx.logger.info("Importing well path from %s", input.path)
        collection = instance.project.well_path_collection()
        well_path = collection.import_well_path(file_name=input.path)
        ctx.logger.info("Imported well path '%s'", well_path.name)
        return WellPath(value=well_path)


class AddPerforation(Task[AddPerforationInput, PerforationOutput]):
    """Add perforation events to a well path."""

    name = "add_perforation"

    def run(self, input: AddPerforationInput, ctx: ExecutionContext) -> PerforationOutput:
        instance: rips.Instance = ctx.resolve("resinsight")
        ctx.logger.info(
            "Adding perforation to '%s' at MD %.1f-%.1f on %s",
            input.value.name,
            input.start_md,
            input.end_md,
            input.event_date,
        )
        collection = instance.project.descendants(rips.WellPathCollection)[0]
        timeline = collection.event_timeline()
        timeline.add_perf_event(
            event_date=input.event_date,
            well_path=input.value,
            start_md=input.start_md,
            end_md=input.end_md,
            diameter=0.1,
            skin_factor=0.5,
            state="OPEN",
        )

        return PerforationOutput(
            value=input.value,
            start_md=input.start_md,
            end_md=input.end_md,
        )


class ExportCompletions(Task):  # type: ignore[type-arg]
    """Fan-in: merge case and perforations, then export completions.

    Uses inline Inputs/Outputs inner classes to demonstrate the
    named-ports fan-in pattern. export_path comes from config.
    """

    name = "export_completions"

    class Inputs(BaseModel):
        grid_case: GridCase
        perforation_1: PerforationOutput
        perforation_2: PerforationOutput
        event_date: str
        export_path: str

    class Outputs(BaseModel):
        export_file: str
        well_path_names: list[str]

    def run(self, input: Inputs, ctx: ExecutionContext) -> Outputs:
        eclipse_case = input.grid_case.value
        well_path_names = [
            input.perforation_1.value.name,
            input.perforation_2.value.name,
        ]
        ctx.logger.info(
            "Exporting completions for wells %s to %s",
            well_path_names,
            input.export_path,
        )

        instance: rips.Instance = ctx.resolve("resinsight")
        collection = instance.project.descendants(rips.WellPathCollection)[0]
        timeline = collection.event_timeline()
        timeline.set_timestamp(timestamp=input.event_date)

        eclipse_case.export_well_path_completions(
            time_step=0,
            well_path_names=well_path_names,
            file_split="UNIFIED_FILE",
            include_perforations=True,
            custom_file_name=input.export_path,
        )
        ctx.logger.info("Export complete: %s", input.export_path)
        return self.Outputs(
            export_file=input.export_path,
            well_path_names=well_path_names,
        )


# ---------------------------------------------------------------------------
# Workflow — uses named instances and config_fields
# ---------------------------------------------------------------------------

workflow = (
    Workflow.builder(name="resinsight_completions")
    .add_task(ConnectToResInsight)
    .add_task(
        LoadModel,
        depends_on=ConnectToResInsight,
        config_fields=["path"],
    )
    .add_task(
        LoadWellPath,
        name="load_well_path_1",
        depends_on=LoadModel,
        config_fields=["path"],
    )
    .add_task(
        LoadWellPath,
        name="load_well_path_2",
        depends_on=LoadModel,
        config_fields=["path"],
    )
    .add_task(
        AddPerforation,
        name="add_perf_1",
        depends_on="load_well_path_1",
        config_fields=["event_date", "start_md", "end_md"],
    )
    .add_task(
        AddPerforation,
        name="add_perf_2",
        depends_on="load_well_path_2",
        config_fields=["event_date", "start_md", "end_md"],
    )
    .add_task(
        ExportCompletions,
        depends_on={
            "grid_case": LoadModel,
            "perforation_1": "add_perf_1",
            "perforation_2": "add_perf_2",
        },
        config_fields=["event_date", "export_path"],
    )
    .build()
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def print_results(result: Job[Any], timing: TimingHook, wf: Workflow) -> None:
    """Print pipeline results, timings, and Mermaid diagram."""
    print("=" * 60)
    print("  ResInsight Completions Pipeline - Results")
    print("=" * 60)
    print(f"\n  Job status:     {result.status}")
    if result.result is not None:
        output: ExportCompletions.Outputs = result.result  # type: ignore[assignment]
        print(f"  Export file:    {output.export_file}")
        print(f"  Well paths:     {output.well_path_names}")
    if result.error:
        print(f"  Error:          {result.error}")
        print(f"  Failed task:    {result.failed_task}")

    print(f"\n  Total duration: {timing.job_duration:.4f}s")
    print("  Task timings:")
    for name, duration in timing.task_timings.items():
        print(f"    {name:30s} {duration:.4f}s")

    print("\nMermaid diagram:")
    print("```mermaid")
    print(wf.to_mermaid(), end="")
    print("```")


def run_python_mode() -> None:
    """Run the pipeline using the Python API."""
    job_config = JobConfiguration(
        {
            "connect_to_resinsight": {},
            "load_model": {
                "path": "/path/to/model/NORNE_ATW2013.EGRID",
            },
            "load_well_path_1": {
                "path": "/path/to/wells/well_1.dev",
            },
            "load_well_path_2": {
                "path": "/path/to/wells/well_2.dev",
            },
            "add_perf_1": {
                "event_date": "2024-01-01",
                "start_md": 3000.0,
                "end_md": 3500.0,
            },
            "add_perf_2": {
                "event_date": "2024-02-01",
                "start_md": 2800.0,
                "end_md": 3200.0,
            },
            "export_completions": {
                "event_date": "2024-05-01",
                "export_path": "/path/to/output/completions.sch",
            },
        }
    )

    job: Job[EmptyConfig] = Job(
        workflow=workflow,
        config=EmptyConfig(),
        job_configuration=job_config,
    )
    ctx = ExecutionContext()

    timing = TimingHook()
    runner = Runner(hooks=[LoggingHook(), timing, PrintTimestampHook()])
    result = runner.run(job, ctx=ctx)

    print_results(result, timing, workflow)


def run_yaml_mode(workflow_path: str, input_path: str) -> None:
    """Run the pipeline from YAML workflow and input files."""
    from taskekrabbe.yaml_config import load_workflow_from_yaml

    loaded = load_workflow_from_yaml(workflow_path, input_path)
    result = loaded.run()

    timing = next(
        (h for h in loaded.runner.hooks if isinstance(h, TimingHook)),
        None,
    )
    assert timing is not None, "TimingHook not found in loaded runner hooks"

    print_results(result, timing, loaded.workflow)


def main() -> None:
    import argparse
    from pathlib import Path

    _dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="ResInsight Completions Pipeline example")
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
