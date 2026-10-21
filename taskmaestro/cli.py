"""Command-line interface for validating, visualizing, and running workflows."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any

from pydantic.errors import PydanticInvalidForJsonSchema, PydanticSchemaGenerationError

from taskmaestro.discovery import (
    get_registered_task,
    registered_task_names,
    registered_workflow_names,
)
from taskmaestro.exceptions import ConfigLoadError, PluginLoadError, WorkflowLookupError
from taskmaestro.introspection import (
    TaskSchemaGenerator,
    dependency_spec,
    describe_registered_workflow,
    describe_yaml_workflow,
    error_payload,
    jsonable,
)
from taskmaestro.job import JobStatus
from taskmaestro.task import get_input_type, get_output_type
from taskmaestro.yaml_config import (
    LoadedWorkflow,
    load_registered_workflow,
    load_workflow_from_yaml,
    workflow_imports,
)

WORKFLOW_ARGUMENT_HELP = "Workflow YAML file, or the name of a registered workflow"


def _add_workflow_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("workflow", help=WORKFLOW_ARGUMENT_HELP)
    parser.add_argument("--input", required=True, help="Path to the input YAML file")


def _is_workflow_path(workflow: str) -> bool:
    """Return True if a workflow argument is a YAML path rather than a registered name."""
    return Path(workflow).suffix.lower() in (".yaml", ".yml") or Path(workflow).is_file()


@contextmanager
def _workflow_imports(path: str) -> Iterator[None]:
    """Make task modules next to a YAML file importable for the duration of loading."""
    with workflow_imports(Path(path).resolve().parent):
        yield


def _load(args: argparse.Namespace) -> LoadedWorkflow:
    if not _is_workflow_path(args.workflow):
        return load_registered_workflow(args.workflow, args.input)
    with _workflow_imports(args.workflow):
        return load_workflow_from_yaml(args.workflow, args.input)


_error = error_payload
_jsonable = jsonable
_dependency_spec = dependency_spec
_TaskSchemaGenerator = TaskSchemaGenerator


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


def _workflow_list(args: argparse.Namespace) -> int:
    names = sorted(registered_workflow_names())
    if args.json:
        print(json.dumps({"workflows": names}))
    else:
        print("\n".join(names))
    return 0


def _workflow_describe(args: argparse.Namespace) -> int:
    input_path = Path(args.input) if args.input else None
    if _is_workflow_path(args.workflow):
        description = describe_yaml_workflow(Path(args.workflow), input_path)
    else:
        description = describe_registered_workflow(args.workflow, input_path)
    print(json.dumps(description, indent=None if args.json else 2))
    return 0


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

    workflow_parser = subparsers.add_parser(
        "workflow", help="Inspect YAML and registered workflows"
    )
    workflow_subparsers = workflow_parser.add_subparsers(dest="workflow_command", required=True)
    workflow_list_parser = workflow_subparsers.add_parser(
        "list", help="List registered workflow names"
    )
    workflow_list_parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON"
    )
    workflow_list_parser.set_defaults(handler=_workflow_list)
    workflow_describe_parser = workflow_subparsers.add_parser(
        "describe", help="Describe a workflow without running tasks"
    )
    workflow_describe_parser.add_argument("workflow", help=WORKFLOW_ARGUMENT_HELP)
    workflow_describe_parser.add_argument(
        "--input",
        help="Optional input YAML; reports provided, missing and configured values per task",
    )
    workflow_describe_parser.add_argument(
        "--json", action="store_true", help="Print single-line JSON"
    )
    workflow_describe_parser.set_defaults(handler=_workflow_describe)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Taskmaestro command-line interface."""
    args = build_parser().parse_args(argv)
    try:
        handler: Any = args.handler
        return int(handler(args))
    except ConfigLoadError as exc:
        if getattr(args, "json", False) and (
            args.command in ("run", "validate")
            or (args.command == "workflow" and args.workflow_command == "describe")
        ):
            # A lookup message only names workflows, so it is safe to report.
            message = (
                str(exc)
                if isinstance(exc, WorkflowLookupError)
                else "Workflow configuration is invalid"
            )
            print(
                json.dumps(
                    {
                        "status": "invalid",
                        "error": _error("configuration_error", exc, message),
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
