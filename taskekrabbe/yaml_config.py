"""YAML-based workflow configuration loader."""

from __future__ import annotations

import importlib
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from taskekrabbe.context import ExecutionContext
from taskekrabbe.exceptions import ConfigLoadError
from taskekrabbe.hooks.base import BaseHook
from taskekrabbe.job import Job
from taskekrabbe.runner import Runner
from taskekrabbe.task import Task, get_input_type
from taskekrabbe.workflow import Workflow, WorkflowBuilder

# --- Pydantic schema models for YAML validation ---


class TaskConfig(BaseModel):
    """A single task entry in the YAML workflow config."""

    task: str
    name: str | None = None
    depends_on: str | list[str] | dict[str, Any] | None = None


class HookConfig(BaseModel):
    """A single hook entry in the YAML runner config."""

    hook: str
    params: dict[str, Any] = Field(default_factory=dict)


class RunnerConfig(BaseModel):
    """Runner configuration section."""

    timeout_seconds: float | None = None
    hooks: list[HookConfig] = Field(default_factory=list)


class ContextConfig(BaseModel):
    """Context configuration section."""

    correlation_id: str | None = None
    scratch_dir: str | None = None
    services: dict[str, Any] = Field(default_factory=dict)


class WorkflowSectionConfig(BaseModel):
    """Workflow section of the YAML config."""

    name: str
    result_task: str | None = None
    tasks: list[TaskConfig] = Field(min_length=1)


class YamlWorkflowConfig(BaseModel):
    """Top-level YAML workflow configuration."""

    workflow: WorkflowSectionConfig
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)


# --- Utilities ---


def import_class(dotted_path: str) -> type[Any]:
    """Import a class from a dotted path like 'pkg.mod.ClassName'.

    Raises ConfigLoadError if the module or class cannot be found.
    """
    parts = dotted_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ConfigLoadError(f"Invalid import path '{dotted_path}': expected 'module.ClassName'")
    module_path, class_name = parts
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ConfigLoadError(f"Cannot import module '{module_path}': {exc}") from exc
    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise ConfigLoadError(f"Module '{module_path}' has no attribute '{class_name}'") from None
    return cls  # type: ignore[no-any-return]


def _coerce_hook_params(hook_cls: type[Any], params: dict[str, Any]) -> dict[str, Any]:
    """Inspect hook __init__ signature and coerce string values to Path where annotated."""
    coerced = dict(params)
    try:
        hints = typing.get_type_hints(hook_cls.__init__)
    except Exception:
        return coerced
    for param_name, annotation in hints.items():
        if param_name == "return":
            continue
        if param_name in coerced and annotation is Path:
            coerced[param_name] = Path(coerced[param_name])
    return coerced


# --- LoadedWorkflow ---


@dataclass(frozen=True)
class LoadedWorkflow:
    """A fully resolved workflow ready to execute."""

    workflow: Workflow
    runner: Runner
    job: Job[Any]
    context: ExecutionContext
    _timeout: float | None = None

    def run(self) -> Job[Any]:
        """Execute the loaded workflow."""
        return self.runner.run(
            self.job,
            ctx=self.context,
            timeout_seconds=self._timeout,
        )

    @property
    def _timeout_seconds(self) -> float | None:
        """Access timeout value."""
        return self._timeout


# --- Main loader ---


def load_workflow_from_yaml(workflow_path: str | Path, input_path: str | Path) -> LoadedWorkflow:
    """Load a complete workflow configuration from a workflow YAML file and an input YAML file.

    Returns a LoadedWorkflow with workflow, runner, job, and context
    ready to execute.

    Raises ConfigLoadError for any loading or validation failure.
    """
    workflow_path = Path(workflow_path)
    input_path = Path(input_path)

    # 1. Parse workflow YAML
    try:
        raw = yaml.safe_load(workflow_path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"YAML parse error: {exc}") from exc
    except OSError as exc:
        raise ConfigLoadError(f"Cannot read file '{workflow_path}': {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigLoadError("YAML file must contain a mapping at top level")

    # 2. Parse input YAML
    try:
        raw_input = yaml.safe_load(input_path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"Input YAML parse error: {exc}") from exc
    except OSError as exc:
        raise ConfigLoadError(f"Cannot read input file '{input_path}': {exc}") from exc

    if not isinstance(raw_input, dict):
        raise ConfigLoadError("Input YAML file must contain a mapping")

    # 3. Schema validation
    try:
        config = YamlWorkflowConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigLoadError(f"YAML schema validation error: {exc}") from exc

    # 4. Resolve task import paths
    task_classes: dict[str, type[Task[Any, Any]]] = {}
    for task_config in config.workflow.tasks:
        cls = import_class(task_config.task)
        if not (isinstance(cls, type) and issubclass(cls, Task)):
            raise ConfigLoadError(f"'{task_config.task}' is not a Task subclass")
        task_classes[task_config.task] = cls

    # 5. Build a lookup from instance names and import paths to registered names.
    # The registered name is task_config.name (if set) or the class's default name.
    # This lookup is used for resolving depends_on strings and result_task.
    name_lookup: dict[str, str] = {}
    for task_config in config.workflow.tasks:
        registered_name = (
            task_config.name if task_config.name else task_classes[task_config.task].name
        )
        # Map import path to registered name
        name_lookup[task_config.task] = registered_name
        # Map instance name to itself
        if task_config.name:
            name_lookup[task_config.name] = registered_name

    def _resolve_yaml_dep(dep_str: str, context_task: str) -> str:
        """Resolve a YAML dependency string to a registered task name."""
        if dep_str in name_lookup:
            return name_lookup[dep_str]
        raise ConfigLoadError(f"Dependency '{dep_str}' for task '{context_task}' not found")

    # 6. Detect linear vs DAG mode
    has_depends_on = any(tc.depends_on is not None for tc in config.workflow.tasks)

    # 7. Resolve result_task
    result_task_name: str | None = None
    if config.workflow.result_task:
        if config.workflow.result_task in name_lookup:
            result_task_name = name_lookup[config.workflow.result_task]
        else:
            raise ConfigLoadError(f"result_task '{config.workflow.result_task}' not found")

    # 8. Build Workflow
    if not has_depends_on:
        # Linear mode: chain tasks in list order
        task_list = [task_classes[tc.task] for tc in config.workflow.tasks]
        result_task_cls = (
            task_classes[config.workflow.result_task] if config.workflow.result_task else None
        )
        workflow = Workflow(
            name=config.workflow.name,
            tasks=task_list,
            result_task=result_task_cls,
        )
    else:
        # DAG mode: use WorkflowBuilder
        builder = WorkflowBuilder(
            config.workflow.name,
            result_task=result_task_name,
        )
        for task_config in config.workflow.tasks:
            cls = task_classes[task_config.task]
            deps = task_config.depends_on
            registered_name = name_lookup[task_config.task]
            # Use task_config.name to pass to add_task (None means use class default)
            instance_name = task_config.name
            if deps is None:
                builder.add_task(cls, name=instance_name)
            elif isinstance(deps, str):
                resolved_dep = _resolve_yaml_dep(deps, task_config.task)
                builder.add_task(cls, name=instance_name, depends_on=resolved_dep)
            elif isinstance(deps, list):
                # Field-ref: [task_path, field_name]
                if len(deps) != 2 or not all(isinstance(e, str) for e in deps):
                    raise ConfigLoadError(
                        f"List depends_on must be [task_path, field_name], "
                        f"got {deps!r} for task '{task_config.task}'"
                    )
                upstream_path, field_name = deps
                resolved_dep = _resolve_yaml_dep(upstream_path, task_config.task)
                builder.add_task(
                    cls,
                    name=instance_name,
                    depends_on=(resolved_dep, field_name),
                )
            elif isinstance(deps, dict):
                fan_in: dict[
                    str, type[Task[Any, Any]] | str | tuple[type[Task[Any, Any]] | str, str]
                ] = {}
                for field_name, upstream_ref in deps.items():
                    if isinstance(upstream_ref, list):
                        if len(upstream_ref) != 2 or not all(
                            isinstance(e, str) for e in upstream_ref
                        ):
                            raise ConfigLoadError(
                                f"List dep must be [task_path, field_name], "
                                f"got {upstream_ref!r} for field '{field_name}' "
                                f"on task '{task_config.task}'"
                            )
                        up_path, up_field = upstream_ref
                        resolved_dep = _resolve_yaml_dep(up_path, task_config.task)
                        fan_in[field_name] = (resolved_dep, up_field)
                    else:
                        resolved_dep = _resolve_yaml_dep(upstream_ref, task_config.task)
                        fan_in[field_name] = resolved_dep
                builder.add_task(cls, name=instance_name, depends_on=fan_in)
        try:
            workflow = builder.build()
        except Exception as exc:
            raise ConfigLoadError(f"Workflow validation failed: {exc}") from exc

    # 9. Validate input against root task input type
    root_tasks = [
        task_classes[tc.task]
        for tc in config.workflow.tasks
        if tc.depends_on is None and has_depends_on
    ]
    if not has_depends_on:
        # Linear mode: first task is the root
        root_tasks = [task_classes[config.workflow.tasks[0].task]]

    input_type = get_input_type(root_tasks[0])
    try:
        validated_input = input_type.model_validate(raw_input)
    except ValidationError as exc:
        raise ConfigLoadError(f"Input validation error: {exc}") from exc

    # 10. Build Job
    job: Job[Any] = Job(workflow, validated_input)

    # 11. Instantiate hooks
    hooks: list[BaseHook] = []
    for hook_config in config.runner.hooks:
        hook_cls = import_class(hook_config.hook)
        if not (isinstance(hook_cls, type) and issubclass(hook_cls, BaseHook)):
            raise ConfigLoadError(f"'{hook_config.hook}' is not a BaseHook subclass")
        coerced_params = _coerce_hook_params(hook_cls, hook_config.params)
        try:
            hooks.append(hook_cls(**coerced_params))
        except TypeError as exc:
            raise ConfigLoadError(f"Cannot instantiate hook '{hook_config.hook}': {exc}") from exc

    runner = Runner(hooks=hooks)

    # 12. Build ExecutionContext
    ctx_kwargs: dict[str, Any] = {}
    if config.context.correlation_id:
        ctx_kwargs["correlation_id"] = config.context.correlation_id
    if config.context.scratch_dir:
        ctx_kwargs["scratch_dir"] = Path(config.context.scratch_dir)
    context = ExecutionContext(**ctx_kwargs)
    for key, value in config.context.services.items():
        context.register(key, value)

    # 13. Return LoadedWorkflow
    return LoadedWorkflow(
        workflow=workflow,
        runner=runner,
        job=job,
        context=context,
        _timeout=config.runner.timeout_seconds,
    )


def run_workflow_from_yaml(workflow_path: str | Path, input_path: str | Path) -> Job[Any]:
    """Load and run a workflow from a workflow YAML file and an input YAML file in one step."""
    loaded = load_workflow_from_yaml(workflow_path, input_path)
    return loaded.run()
