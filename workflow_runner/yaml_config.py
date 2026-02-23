"""YAML-based workflow configuration loader."""

from __future__ import annotations

import importlib
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from workflow_runner.context import ExecutionContext
from workflow_runner.exceptions import ConfigLoadError
from workflow_runner.hooks.base import BaseHook
from workflow_runner.job import Job
from workflow_runner.runner import Runner
from workflow_runner.task import Task, get_input_type
from workflow_runner.workflow import Workflow, WorkflowBuilder

# --- Pydantic schema models for YAML validation ---


class TaskConfig(BaseModel):
    """A single task entry in the YAML workflow config."""

    task: str
    depends_on: str | dict[str, str] | None = None


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
    input: dict[str, Any]


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


def load_workflow_from_yaml(path: str | Path) -> LoadedWorkflow:
    """Load a complete workflow configuration from a YAML file.

    Returns a LoadedWorkflow with workflow, runner, job, and context
    ready to execute.

    Raises ConfigLoadError for any loading or validation failure.
    """
    path = Path(path)

    # 1. Parse YAML
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"YAML parse error: {exc}") from exc
    except OSError as exc:
        raise ConfigLoadError(f"Cannot read file '{path}': {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigLoadError("YAML file must contain a mapping at top level")

    # 2. Schema validation
    try:
        config = YamlWorkflowConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigLoadError(f"YAML schema validation error: {exc}") from exc

    # 3. Resolve task import paths
    task_classes: dict[str, type[Task[Any, Any]]] = {}
    for task_config in config.workflow.tasks:
        cls = import_class(task_config.task)
        if not (isinstance(cls, type) and issubclass(cls, Task)):
            raise ConfigLoadError(f"'{task_config.task}' is not a Task subclass")
        task_classes[task_config.task] = cls

    # 4. Detect linear vs DAG mode
    has_depends_on = any(tc.depends_on is not None for tc in config.workflow.tasks)

    # 5. Build Workflow
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
        result_task_cls = (
            task_classes[config.workflow.result_task] if config.workflow.result_task else None
        )
        builder = WorkflowBuilder(config.workflow.name, result_task=result_task_cls)
        for task_config in config.workflow.tasks:
            cls = task_classes[task_config.task]
            deps = task_config.depends_on
            if deps is None:
                builder.add_task(cls)
            elif isinstance(deps, str):
                if deps not in task_classes:
                    raise ConfigLoadError(
                        f"Dependency '{deps}' for task '{task_config.task}' not found"
                    )
                builder.add_task(cls, depends_on=task_classes[deps])
            elif isinstance(deps, dict):
                fan_in: dict[str, type[Task[Any, Any]]] = {}
                for field_name, upstream_path in deps.items():
                    if upstream_path not in task_classes:
                        raise ConfigLoadError(
                            f"Fan-in dependency '{upstream_path}' for field "
                            f"'{field_name}' on task '{task_config.task}' not found"
                        )
                    fan_in[field_name] = task_classes[upstream_path]
                builder.add_task(cls, depends_on=fan_in)
        try:
            workflow = builder.build()
        except Exception as exc:
            raise ConfigLoadError(f"Workflow validation failed: {exc}") from exc

    # 6. Validate input against root task input type
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
        validated_input = input_type.model_validate(config.input)
    except ValidationError as exc:
        raise ConfigLoadError(f"Input validation error: {exc}") from exc

    # 7. Build Job
    job: Job[Any] = Job(workflow, validated_input)

    # 8. Instantiate hooks
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

    # 9. Build ExecutionContext
    ctx_kwargs: dict[str, Any] = {}
    if config.context.correlation_id:
        ctx_kwargs["correlation_id"] = config.context.correlation_id
    if config.context.scratch_dir:
        ctx_kwargs["scratch_dir"] = Path(config.context.scratch_dir)
    context = ExecutionContext(**ctx_kwargs)
    for key, value in config.context.services.items():
        context.register(key, value)

    # 10. Return LoadedWorkflow
    return LoadedWorkflow(
        workflow=workflow,
        runner=runner,
        job=job,
        context=context,
        _timeout=config.runner.timeout_seconds,
    )


def run_workflow_from_yaml(path: str | Path) -> Job[Any]:
    """Load and run a workflow from a YAML config file in one step."""
    loaded = load_workflow_from_yaml(path)
    return loaded.run()
