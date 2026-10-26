"""Public helpers for describing workflows, tasks and configuration values as JSON.

The CLI's ``describe`` commands are built on these functions. Other front ends
(editors, services) can reuse them, and can extend the generated JSON schemas
by subclassing ``TaskSchemaGenerator`` and overriding ``type_annotations``.
"""

from __future__ import annotations

import inspect
import json
import math
import sys
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict
from datetime import date
from pathlib import Path, PurePath
from typing import Any

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic.errors import PydanticInvalidForJsonSchema, PydanticSchemaGenerationError
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema

from taskmaestro.conditions import Guard, plain_label
from taskmaestro.dependencies import CollectionRef, FirstOfRef, OutputRef
from taskmaestro.exceptions import ConfigLoadError, WorkflowDefinitionError
from taskmaestro.job import EmptyConfig, Job, JobConfiguration
from taskmaestro.task import get_input_type, get_output_type
from taskmaestro.workflow import Workflow
from taskmaestro.yaml_config import (
    load_input_configuration,
    load_workflow_only,
    registered_workflow,
    workflow_imports,
)


def python_type_name(cls: type[Any]) -> str:
    """Return the dotted ``module.QualifiedName`` of a class."""
    return f"{cls.__module__}.{cls.__qualname__}"


def error_payload(
    code: str, exc: Exception, message: str, *, task: str | None = None
) -> dict[str, Any]:
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


class TaskSchemaGenerator(GenerateJsonSchema):
    """Describe runtime-only Python objects without pretending they accept JSON.

    Override ``type_annotations`` to add extra keys to the schema of every
    opaque Python type and every Pydantic model.
    """

    @classmethod
    def type_annotations(cls, python_type: type[Any]) -> dict[str, Any]:
        return {}

    def is_instance_schema(self, schema: core_schema.IsInstanceSchema) -> JsonSchemaValue:
        python_type = schema["cls"]
        return {
            "not": {},  # No JSON value can satisfy an isinstance check for this object.
            "x-taskmaestro-opaque": True,
            "x-taskmaestro-python-type": python_type_name(python_type),
            **self.type_annotations(python_type),
        }

    def model_schema(self, schema: core_schema.ModelSchema) -> JsonSchemaValue:
        json_schema = super().model_schema(schema)
        json_schema.update(self.type_annotations(schema["cls"]))
        return json_schema


def model_json_schema(
    model: type[Any], schema_generator: type[GenerateJsonSchema] = TaskSchemaGenerator
) -> dict[str, Any]:
    """Return the JSON schema of a Pydantic model or any type Pydantic can adapt."""
    if isinstance(model, type) and issubclass(model, BaseModel):
        return model.model_json_schema(schema_generator=schema_generator)
    return TypeAdapter(model).json_schema(schema_generator=schema_generator)


def _output_ref(ref: OutputRef) -> dict[str, str | None]:
    return {"task": ref.task_name, "field": ref.output_field}


def dependency_spec(deps: Any) -> Any:
    """Render validated dependencies, including field routing and collections."""
    if deps is None:
        return None
    if isinstance(deps, str):
        return {"task": deps, "field": None}
    if isinstance(deps, tuple):
        return {"task": deps[0], "field": deps[1]}
    result: dict[str, Any] = {}
    for field, ref in deps.items():
        if isinstance(ref, CollectionRef):
            members: list[Any] | dict[str, Any]
            if ref.kind == "keyed":
                members = {key: _output_ref(item) for key, item in ref.keyed_members}
            else:
                members = [_output_ref(item) for item in ref.positional_members]
            result[field] = {"collect": {"kind": ref.kind, "members": members}}
        elif isinstance(ref, FirstOfRef):
            result[field] = {"first_of": [_output_ref(item) for item in ref.members]}
        elif isinstance(ref, tuple):
            result[field] = {"task": ref[0], "field": ref[1]}
        else:
            result[field] = {"task": ref, "field": None}
    return result


def guard_spec(guard: Guard | None) -> dict[str, Any] | None:
    """Render a task's guard: source task and field, operator and values.

    For ``otherwise`` guards ``values`` lists the labels the guard matches.
    """
    if guard is None:
        return None
    return {
        "task": guard.ref.task_name,
        "field": guard.ref.output_field,
        "op": guard.op,
        "values": [
            jsonable(plain_label(value), f"{guard.ref.task_name}.{guard.ref.output_field}")
            for value in guard.values
        ],
    }


def _jsonable_key(key: object, path: str) -> str:
    """Convert a YAML mapping key to the JSON object key it would get."""
    value = jsonable(key, path)
    if isinstance(value, str):
        return value
    if isinstance(value, bool | int | float) or value is None:
        return json.dumps(value)
    raise TypeError(f"Configuration value '{path}' has an unsupported mapping key type")


def jsonable(value: object, path: str) -> Any:
    """Convert a raw configuration value (as loaded from YAML) to plain JSON data.

    Dates and datetimes become ISO 8601 strings and paths become strings;
    everything else must already be JSON-compatible. ``path`` names the value
    in error messages, which deliberately never include the value itself.
    """
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Configuration value '{path}' is not a finite number")
        return value
    if isinstance(value, date):  # Includes datetime.
        return value.isoformat()
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            json_key = _jsonable_key(key, path)
            if json_key in result:
                raise ValueError(
                    f"Configuration value '{path}' has keys that collide as JSON key '{json_key}'"
                )
            result[json_key] = jsonable(item, f"{path}.{json_key}")
        return result
    if isinstance(value, list | tuple):
        return [jsonable(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, set | frozenset):
        items = [jsonable(item, f"{path}[]") for item in value]
        try:
            return sorted(items)
        except TypeError:
            # Mixed types: order by canonical JSON text so output is deterministic.
            return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    raise TypeError(f"Configuration value '{path}' has unsupported type {type(value).__name__}")


def missing_config_fields(workflow: Workflow, config: JobConfiguration) -> dict[str, list[str]]:
    """Return the declared config fields that the input does not supply, per task."""
    return {
        name: sorted(workflow.get_config_fields(name) - config.config_fields_for_task(name))
        for name, _task in workflow.topological_order()
    }


def _fill_missing(
    workflow: Workflow, config: JobConfiguration, missing: dict[str, list[str]]
) -> JobConfiguration:
    """Return a copy of ``config`` with placeholder values for missing config fields.

    ``Job`` only checks that these fields are present, so placeholders let it run
    its remaining validation. Map sources get no placeholder, so a missing one is
    still rejected with its own error.
    """
    filled: dict[str, dict[str, Any]] = {}
    for name, fields in missing.items():
        task_map = workflow.get_task_map(name)
        map_source = task_map.over if task_map is not None else None
        placeholders = {field: None for field in fields if field != map_source}
        filled[name] = {**placeholders, **config.get_config_for_task(name)}
    return JobConfiguration(filled)


@contextmanager
def _describe_errors() -> Iterator[None]:
    """Report describe failures as ConfigLoadError."""
    try:
        yield
    except WorkflowDefinitionError as exc:
        raise ConfigLoadError(f"Job validation failed: {exc}") from exc
    except (
        TypeError,
        ValueError,
        PydanticInvalidForJsonSchema,
        PydanticSchemaGenerationError,
    ) as exc:
        raise ConfigLoadError(f"Cannot describe workflow: {exc}") from exc


def _describe_loaded(
    workflow: Workflow,
    config: JobConfiguration | None,
    schema_generator: type[GenerateJsonSchema],
    source: dict[str, str],
) -> dict[str, Any]:
    """Describe a loaded workflow and, if given, its input configuration."""
    configured = None
    missing = None
    values = None
    if config is not None:
        missing = missing_config_fields(workflow, config)
        # Check the rest of the supplied config (map sources, root inputs)
        # without constructing hooks or executing any tasks. Missing config
        # fields are reported rather than rejected: the input may be a
        # template whose remaining fields the caller fills in.
        Job(
            workflow,
            EmptyConfig(),
            job_configuration=_fill_missing(workflow, config, missing),
        )
        configured = {
            name: sorted(config.config_fields_for_task(name))
            for name, _task in workflow.topological_order()
        }
        # Raw input values, before Pydantic validation, so opaque or
        # runtime-only markers pass through unchanged.
        values = {
            name: {
                field: jsonable(value, f"{name}.{field}")
                for field, value in sorted(config.get_config_for_task(name).items())
            }
            for name, _task in workflow.topological_order()
        }
    description = describe_workflow(
        workflow,
        configured=configured,
        missing=missing,
        values=values,
        schema_generator=schema_generator,
    )
    description["source"] = source
    return description


def describe_yaml_workflow(
    workflow_path: Path,
    input_path: Path | None = None,
    schema_generator: type[GenerateJsonSchema] = TaskSchemaGenerator,
) -> dict[str, Any]:
    """Describe a YAML workflow without running any task.

    With ``input_path`` the description also reports provided, missing and
    configured values per task. ``source`` is ``{"kind": "yaml", "path": ...}``.
    Raises ConfigLoadError if the workflow or input is invalid; output printed
    while importing task modules goes to stderr.
    """
    path = Path(workflow_path).resolve()
    with _describe_errors(), redirect_stdout(sys.stderr), workflow_imports(path.parent):
        workflow, config = load_workflow_only(Path(workflow_path), input_path)
        return _describe_loaded(
            workflow,
            config if input_path is not None else None,
            schema_generator,
            {"kind": "yaml", "path": str(path)},
        )


def describe_registered_workflow(
    name: str,
    input_path: Path | None = None,
    schema_generator: type[GenerateJsonSchema] = TaskSchemaGenerator,
) -> dict[str, Any]:
    """Describe a workflow registered in the ``taskmaestro.workflows`` entry-point group.

    Like ``describe_yaml_workflow``; ``source`` is
    ``{"kind": "entry_point", "name": ...}``. Raises ConfigLoadError for an
    unknown workflow or invalid input.
    """
    with _describe_errors(), redirect_stdout(sys.stderr):
        workflow = registered_workflow(name)
        config = load_input_configuration(workflow, input_path) if input_path else None
        return _describe_loaded(
            workflow, config, schema_generator, {"kind": "entry_point", "name": name}
        )


def task_descriptor(
    task_id: str,
    task: type[Any],
    schema_generator: type[GenerateJsonSchema] = TaskSchemaGenerator,
) -> dict[str, Any]:
    """Describe a task class: identity, timeout and input/output JSON schemas."""
    return {
        "id": task_id,
        "name": getattr(task, "name", task.__name__),
        "python_type": python_type_name(task),
        "description": inspect.getdoc(task) or "",
        "timeout_seconds": getattr(task, "timeout_seconds", None),
        "input_schema": model_json_schema(get_input_type(task), schema_generator),
        "output_schema": model_json_schema(get_output_type(task), schema_generator),
    }


def describe_workflow(
    workflow: Workflow,
    *,
    configured: dict[str, list[str]] | None = None,
    missing: dict[str, list[str]] | None = None,
    values: dict[str, dict[str, Any]] | None = None,
    schema_generator: type[GenerateJsonSchema] = TaskSchemaGenerator,
) -> dict[str, Any]:
    """Describe a workflow's tasks, dependencies and schemas without running it.

    ``configured``, ``missing`` and ``values`` report per-task input state and
    are ``None`` in the output when not supplied.
    """
    tasks: list[dict[str, Any]] = []
    for name, task in workflow.topological_order():
        input_type = get_input_type(task)
        output_type = workflow.get_output_annotation(name)
        task_map = workflow.get_task_map(name)
        tasks.append(
            {
                "name": name,
                "python_type": python_type_name(task),
                "depends_on": dependency_spec(workflow.get_dependencies(name)),
                "config_fields": sorted(workflow.get_config_fields(name)),
                "provided_config_fields": configured[name] if configured is not None else None,
                "missing_config_fields": missing[name] if missing is not None else None,
                "config_values": values[name] if values is not None else None,
                "required_input_fields": sorted(
                    field for field, info in input_type.model_fields.items() if info.is_required()
                ),
                "map": asdict(task_map) if task_map is not None else None,
                "when": guard_spec(workflow.get_guard(name)),
                "branches": (
                    [plain_label(label) for label in task.branches]  # type: ignore[attr-defined]
                    if workflow.is_switch(name)
                    else None
                ),
                "input_schema": model_json_schema(input_type, schema_generator),
                "output_schema": model_json_schema(output_type, schema_generator),
            }
        )
    return {"workflow": workflow.name, "result_task": workflow.result_task_name, "tasks": tasks}
