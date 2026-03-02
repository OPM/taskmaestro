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

Run:
    source .venv/bin/activate
    python example_resinsight.py              # Python API
    python example_resinsight.py --yaml       # YAML config (defaults)
    python example_resinsight.py --yaml example_resinsight.yaml \
        --input example_resinsight_input.yaml
"""

from __future__ import annotations

import rips
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


class ResInsightConfig(BaseModel):
    """Job config: file paths and perforation parameters."""

    egrid_path: str
    well_path_file_1: str
    well_path_file_2: str
    perf_1_start_md: float
    perf_1_end_md: float
    perf_2_start_md: float
    perf_2_end_md: float
    export_path: str


class ConnectionOutput(BaseModel):
    """Output of ConnectToResInsight: the gRPC port."""

    port: int


class LoadModelOutput(BaseModel):
    """Output of LoadModel: case metadata."""

    case_id: int
    case_name: str


class WellPathOutput(BaseModel):
    """Output of LoadWellPath: imported well path name."""

    well_path_name: str


class PerforationOutput(BaseModel):
    """Output of AddPerforation: perforation interval details."""

    well_path_name: str
    start_md: float
    end_md: float


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class ConnectToResInsight(Task[ResInsightConfig, ConnectionOutput]):
    """Connect to a running ResInsight instance and register it in context."""

    name = "connect_to_resinsight"

    def run(self, input: ResInsightConfig, ctx: ExecutionContext) -> ConnectionOutput:
        ctx.logger.info("Connecting to ResInsight...")
        instance = rips.Instance.find()
        ctx.register("resinsight", instance)
        ctx.register("egrid_path", input.egrid_path)
        ctx.register("well_path_file_1", input.well_path_file_1)
        ctx.register("well_path_file_2", input.well_path_file_2)
        ctx.register("perf_1_start_md", input.perf_1_start_md)
        ctx.register("perf_1_end_md", input.perf_1_end_md)
        ctx.register("perf_2_start_md", input.perf_2_start_md)
        ctx.register("perf_2_end_md", input.perf_2_end_md)
        ctx.register("export_path", input.export_path)
        port = int(instance.location.split(":")[1])
        ctx.logger.info("Connected to ResInsight on port %d", port)
        return ConnectionOutput(port=port)


class LoadModel(Task[ConnectionOutput, LoadModelOutput]):
    """Load the reservoir model (.egrid) into ResInsight."""

    name = "load_model"

    def run(self, input: ConnectionOutput, ctx: ExecutionContext) -> LoadModelOutput:
        instance: rips.Instance = ctx.resolve("resinsight")
        egrid_path: str = ctx.resolve("egrid_path")
        ctx.logger.info("Loading model from %s", egrid_path)
        case = instance.project.load_case(egrid_path)
        ctx.register("case", case)
        ctx.logger.info("Loaded case '%s' (id=%d)", case.name, case.id)
        return LoadModelOutput(case_id=case.id, case_name=case.name)


class LoadWellPath(Task[LoadModelOutput, WellPathOutput]):
    """Import a well path file into ResInsight."""

    name = "load_well_path"
    _ctx_key: str = "well_path_file_1"  # default; overridden via context

    def run(self, input: LoadModelOutput, ctx: ExecutionContext) -> WellPathOutput:
        instance: rips.Instance = ctx.resolve("resinsight")
        # Use the instance name to look up the right context key
        well_path_key = f"well_path_file_{self.name.split('_')[-1]}"
        well_path_file: str = ctx.resolve(well_path_key)
        ctx.logger.info("Importing well path from %s", well_path_file)
        collection = instance.project.well_path_collection()
        well_path = collection.import_well_path(file_name=well_path_file)
        ctx.register(f"well_path_{self.name}", well_path)
        ctx.logger.info("Imported well path '%s'", well_path.name)
        return WellPathOutput(well_path_name=well_path.name)


class AddPerforation(Task[WellPathOutput, PerforationOutput]):
    """Add perforation events to a well path."""

    name = "add_perforation"

    def run(self, input: WellPathOutput, ctx: ExecutionContext) -> PerforationOutput:
        instance: rips.Instance = ctx.resolve("resinsight")
        # Derive context keys from instance name (e.g. "add_perf_1" -> "1")
        suffix = self.name.split("_")[-1]
        well_path = ctx.resolve(f"well_path_load_well_path_{suffix}")
        start_md: float = ctx.resolve(f"perf_{suffix}_start_md")
        end_md: float = ctx.resolve(f"perf_{suffix}_end_md")
        ctx.logger.info(
            "Adding perforation to '%s' at MD %.1f-%.1f",
            input.well_path_name,
            start_md,
            end_md,
        )
        collection = instance.project.descendants(rips.WellPathCollection)[0]
        timeline = collection.event_timeline()
        timeline.add_perf_event(
            well_path=well_path,
            start_md=start_md,
            end_md=end_md,
        )
        return PerforationOutput(
            well_path_name=input.well_path_name,
            start_md=start_md,
            end_md=end_md,
        )


class ExportCompletions(Task):  # type: ignore[type-arg]
    """Fan-in: merge case and perforations, then export completions.

    Uses inline Inputs/Outputs inner classes to demonstrate the
    named-ports fan-in pattern.
    """

    name = "export_completions"

    class Inputs(BaseModel):
        case: LoadModelOutput
        perforation_1: PerforationOutput
        perforation_2: PerforationOutput

    class Outputs(BaseModel):
        export_file: str
        well_path_names: list[str]

    def run(self, input: Inputs, ctx: ExecutionContext) -> Outputs:
        case = ctx.resolve("case")
        export_path: str = ctx.resolve("export_path")
        well_path_names = [
            input.perforation_1.well_path_name,
            input.perforation_2.well_path_name,
        ]
        ctx.logger.info(
            "Exporting completions for wells %s to %s",
            well_path_names,
            export_path,
        )
        case.export_well_path_completions(
            time_step=0,
            well_path_names=well_path_names,
            file_split="UNIFIED_FILE",
            include_perforations=True,
            custom_file_name=export_path,
        )
        ctx.logger.info("Export complete: %s", export_path)
        return self.Outputs(
            export_file=export_path,
            well_path_names=well_path_names,
        )


# ---------------------------------------------------------------------------
# Workflow — uses named instances instead of subclasses
# ---------------------------------------------------------------------------

workflow = (
    Workflow.builder(name="resinsight_completions")
    .add_task(ConnectToResInsight)
    .add_task(LoadModel, depends_on=ConnectToResInsight)
    .add_task(LoadWellPath, name="load_well_path_1", depends_on=LoadModel)
    .add_task(LoadWellPath, name="load_well_path_2", depends_on=LoadModel)
    .add_task(AddPerforation, name="add_perf_1", depends_on="load_well_path_1")
    .add_task(AddPerforation, name="add_perf_2", depends_on="load_well_path_2")
    .add_task(
        ExportCompletions,
        depends_on={
            "case": LoadModel,
            "perforation_1": "add_perf_1",
            "perforation_2": "add_perf_2",
        },
    )
    .build()
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def print_results(result: Job[ResInsightConfig], timing: TimingHook, wf: Workflow) -> None:
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
    config = ResInsightConfig(
        egrid_path="/path/to/model/NORNE_ATW2013.EGRID",
        well_path_file_1="/path/to/wells/well_1.dev",
        well_path_file_2="/path/to/wells/well_2.dev",
        perf_1_start_md=3000.0,
        perf_1_end_md=3500.0,
        perf_2_start_md=2800.0,
        perf_2_end_md=3200.0,
        export_path="/path/to/output/completions.sch",
    )

    job: Job[ResInsightConfig] = Job(workflow=workflow, config=config)
    ctx = ExecutionContext()

    timing = TimingHook()
    runner = Runner(hooks=[LoggingHook(), timing])
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

    parser = argparse.ArgumentParser(description="ResInsight Completions Pipeline example")
    parser.add_argument(
        "--yaml",
        metavar="FILE",
        nargs="?",
        const="example_resinsight.yaml",
        help="Load workflow from a YAML config file (default: example_resinsight.yaml)",
    )
    parser.add_argument(
        "--input",
        metavar="FILE",
        default="example_resinsight_input.yaml",
        help="Input YAML file (default: example_resinsight_input.yaml)",
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
