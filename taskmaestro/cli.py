"""Command-line interface for validating, visualizing, and running workflows."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from pydantic.errors import PydanticInvalidForJsonSchema, PydanticSchemaGenerationError
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema

from taskmaestro.discovery import get_registered_task, registered_task_names
from taskmaestro.exceptions import ConfigLoadError, PluginLoadError, WorkflowDefinitionError
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


def _error(code: str, exc: Exception, message: str, *, task: str | None = None) -> dict[str, Any]:
    """Return safe diagnostics from an exception and its causes."""
    issues: list[dict[str, str]] = []
    cause: BaseException | None = exc
    while cause is not None:
        if isinstance(cause, ValidationError):
            issues = [
                {"field": ".".join(map(str, issue["loc"])), "code": issue["type"]}
                for issue in cause.errors(include_input=False, include_context=False)
            ]
            break
        if isinstance(cause, WorkflowDefinitionError):
            task = task or cause.task_name
            issues = [{"field": field, "code": "missing"} for field in cause.fields]
            if issues:
                break
        cause = cause.__cause__
    return {
        "code": code,
        "type": type(exc).__name__,
        "message": message,
        "task": task,
        "field": issues[0]["field"] if issues else None,
        "issues": issues,
    }


def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s — %(message)s",
        force=True,
    )
    if args.json:
        # Plugin imports and user tasks may print; reserve stdout for one JSON document.
        with redirect_stdout(sys.stderr):
            result = _load(args).run()
    else:
        result = _load(args).run()
    if args.json:
        if result.status == JobStatus.FAILED:
            assert result.exception is not None
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "workflow": result.workflow.name,
                        "failed_task": result.failed_task,
                        "error": _error(
                            "task_failed", result.exception, "Task failed", task=result.failed_task
                        ),
                    }
                )
            )
            return 1
        assert result.result is not None
        try:
            with redirect_stdout(sys.stderr):
                output = json.loads(result.result.model_dump_json())
        except Exception as exc:
            # Tasks may return Python-only objects; keep stdout valid JSON even then.
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "workflow": result.workflow.name,
                        "failed_task": None,
                        "error": _error(
                            "serialization_error", exc, "Result is not JSON serializable"
                        ),
                    }
                )
            )
            return 1
        print(
            json.dumps({"status": "completed", "workflow": result.workflow.name, "result": output})
        )
        return 0
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
    if args.json:
        with redirect_stdout(sys.stderr):
            loaded = _load(args)
    else:
        loaded = _load(args)
    if args.json:
        print(json.dumps({"status": "valid", "workflow": loaded.workflow.name}))
    else:
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
    run_parser.add_argument(
        "--json", action="store_true", help="Print structured JSON result or error"
    )
    run_parser.set_defaults(handler=_run)

    validate_parser = subparsers.add_parser("validate", help="Validate a YAML workflow")
    _add_workflow_arguments(validate_parser)
    validate_parser.add_argument(
        "--json", action="store_true", help="Print structured JSON result or error"
    )
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
        if getattr(args, "json", False) and args.command in ("run", "validate"):
            print(
                json.dumps(
                    {
                        "status": "invalid",
                        "error": _error(
                            "configuration_error", exc, "Workflow configuration is invalid"
                        ),
                    }
                )
            )
        else:
            print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except PluginLoadError as exc:
        print(f"Plugin error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
