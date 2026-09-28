"""Command-line interface for validating, visualizing, and running workflows."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from typing import Any

from taskmaestro.exceptions import ConfigLoadError
from taskmaestro.job import JobStatus
from taskmaestro.yaml_config import LoadedWorkflow, load_workflow_from_yaml


def _add_workflow_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("workflow", help="Path to the workflow YAML file")
    parser.add_argument("--input", required=True, help="Path to the input YAML file")


def _load(args: argparse.Namespace) -> LoadedWorkflow:
    return load_workflow_from_yaml(args.workflow, args.input)


def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s — %(message)s",
        force=True,
    )
    result = _load(args).run()
    if result.status == JobStatus.FAILED:
        print(
            f"Workflow failed at {result.failed_task}: {result.error}",
            file=sys.stderr,
        )
        return 1
    assert result.result is not None
    print(result.result.model_dump_json(indent=2))
    return 0


def _validate(args: argparse.Namespace) -> int:
    loaded = _load(args)
    print(f"Workflow '{loaded.workflow.name}' is valid")
    return 0


def _graph(args: argparse.Namespace) -> int:
    loaded = _load(args)
    print(
        loaded.workflow.to_mermaid(
            job_configuration=loaded.job.job_configuration,
        ),
        end="",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line parser."""
    parser = argparse.ArgumentParser(prog="taskmaestro")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a YAML workflow")
    _add_workflow_arguments(run_parser)
    run_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    run_parser.set_defaults(handler=_run)

    validate_parser = subparsers.add_parser("validate", help="Validate a YAML workflow")
    _add_workflow_arguments(validate_parser)
    validate_parser.set_defaults(handler=_validate)

    graph_parser = subparsers.add_parser("graph", help="Print a Mermaid workflow graph")
    _add_workflow_arguments(graph_parser)
    graph_parser.set_defaults(handler=_graph)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Taskmaestro command-line interface."""
    args = build_parser().parse_args(argv)
    try:
        handler: Any = args.handler
        return int(handler(args))
    except ConfigLoadError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
