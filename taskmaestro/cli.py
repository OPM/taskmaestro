"""Command-line interface for validating, visualizing, and running workflows."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic.errors import PydanticInvalidForJsonSchema, PydanticSchemaGenerationError
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema

from taskmaestro.discovery import get_registered_task, registered_task_names
from taskmaestro.exceptions import ConfigLoadError, PluginLoadError
from taskmaestro.job import JobStatus
from taskmaestro.task import get_input_type, get_output_type
from taskmaestro.yaml_config import LoadedWorkflow, load_workflow_from_yaml


def _add_workflow_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("workflow", help="Path to the workflow YAML file")
    parser.add_argument("--input", required=True, help="Path to the input YAML file")


def _load(args: argparse.Namespace) -> LoadedWorkflow:
    """Load YAML with its directory available for local task imports."""
    workflow_dir = str(Path(args.workflow).resolve().parent)
    original_path = sys.path.copy()
    sys.path.insert(0, workflow_dir)
    try:
        return load_workflow_from_yaml(args.workflow, args.input)
    finally:
        sys.path[:] = original_path


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


def _tasks_list(args: argparse.Namespace) -> int:
    names = sorted(registered_task_names())
    if args.json:
        print(json.dumps({"tasks": names}))
    else:
        print("\n".join(names))
    return 0


class _TaskSchemaGenerator(GenerateJsonSchema):
    """Describe runtime-only Python objects without pretending they accept JSON."""

    def is_instance_schema(self, schema: core_schema.IsInstanceSchema) -> JsonSchemaValue:
        cls = schema["cls"]
        return {
            "not": {},  # No JSON value can satisfy an isinstance check for this object.
            "x-taskmaestro-opaque": True,
            "x-taskmaestro-python-type": f"{cls.__module__}.{cls.__qualname__}",
        }


def _tasks_describe(args: argparse.Namespace) -> int:
    task = get_registered_task(args.name)
    try:
        input_type = get_input_type(task)
        output_type = get_output_type(task)
        description = {
            "identifier": args.name,
            "name": task.name,
            "timeout_seconds": task.timeout_seconds,
            "input_schema": input_type.model_json_schema(schema_generator=_TaskSchemaGenerator),
            "output_schema": output_type.model_json_schema(schema_generator=_TaskSchemaGenerator),
        }
    except (
        TypeError,
        ValueError,
        PydanticInvalidForJsonSchema,
        PydanticSchemaGenerationError,
    ) as exc:
        raise PluginLoadError(f"Cannot describe task '{args.name}': {exc}") from exc
    print(json.dumps(description, indent=None if args.json else 2))
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

    tasks_parser = subparsers.add_parser("tasks", help="Discover installed task plugins")
    tasks_subparsers = tasks_parser.add_subparsers(dest="tasks_command", required=True)
    list_parser = tasks_subparsers.add_parser("list", help="List registered task identifiers")
    list_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    list_parser.set_defaults(handler=_tasks_list)
    describe_parser = tasks_subparsers.add_parser("describe", help="Describe a registered task")
    describe_parser.add_argument("name", help="Registered task identifier (not a class path)")
    describe_parser.add_argument("--json", action="store_true", help="Print single-line JSON")
    describe_parser.set_defaults(handler=_tasks_describe)

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
    except PluginLoadError as exc:
        print(f"Plugin error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
