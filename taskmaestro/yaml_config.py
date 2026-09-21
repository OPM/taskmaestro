"""YAML-based workflow configuration loader."""

from __future__ import annotations

import importlib
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from taskmaestro.context import ExecutionContext
from taskmaestro.dependencies import CollectionDependency, OutputReference, collect
from taskmaestro.discovery import get_registered_task, registered_task_names
from taskmaestro.exceptions import ConfigLoadError, PluginLoadError, WorkflowDefinitionError
from taskmaestro.hooks.base import BaseHook
from taskmaestro.job import EmptyConfig, Job, JobConfiguration
from taskmaestro.mapping import TaskMap
from taskmaestro.runner import Runner
from taskmaestro.task import Task, get_input_type
from taskmaestro.workflow import Workflow, WorkflowBuilder

# --- Pydantic schema models for YAML validation ---


class TaskMapConfig(BaseModel):
    """Mapped execution settings for a YAML task entry."""

    over: str
    key_as: str
    value_as: str
    error_mode: typing.Literal["fail_fast", "collect_all"] = "fail_fast"


class TaskConfig(BaseModel):
    """A single task entry in the YAML workflow config."""

    task: str | None = None
    workflow: str | None = None
    workflow_input: str | None = None
    name: str | None = None
    depends_on: str | list[str] | dict[str, Any] | None = None
    config_fields: list[str] | None = None
    map: TaskMapConfig | None = None

    @model_validator(mode="after")
    def _check_task_or_workflow(self) -> TaskConfig:
        if self.task and self.workflow:
            raise ValueError("Specify either 'task' or 'workflow', not both")
        if not self.task and not self.workflow:
            raise ValueError("Must specify either 'task' or 'workflow'")
        if self.workflow_input and not self.workflow:
            raise ValueError("'workflow_input' requires 'workflow'")
        return self


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


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._checked_mappings: set[yaml.nodes.MappingNode] = set()

    def flatten_mapping(self, node: yaml.nodes.MappingNode) -> None:
        # Check declarations before merges add inherited keys. Anchors can reuse
        # already-flattened nodes, whose override keys are legitimately repeated.
        if node in self._checked_mappings:
            return
        self._checked_mappings.add(node)
        keys: set[Any] = set()
        for key_node, _value_node in node.value:
            key = (
                "<<"
                if key_node.tag == "tag:yaml.org,2002:merge"
                else self.construct_object(key_node)
            )
            if key in keys:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            keys.add(key)
        super().flatten_mapping(node)


def _yaml_load(text: str) -> Any:
    """Safely parse YAML while rejecting duplicate mapping keys."""
    return yaml.load(text, Loader=_UniqueKeyLoader)


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
class _LinearDep:
    """An already-registered upstream name produced by linear-mode chaining."""

    upstream: str


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


def _load_workflow_only(
    workflow_path: Path,
    input_path: Path | None = None,
    *,
    _ancestors: frozenset[Path] = frozenset(),
) -> tuple[Workflow, JobConfiguration | None]:
    """Build a Workflow and optional JobConfiguration from YAML files.

    This is the core logic shared by ``load_workflow_from_yaml`` and
    recursive ``workflow:`` references in YAML configs.  ``_ancestors`` holds
    the resolved paths of every enclosing workflow file so that a self- or
    mutually-referencing ``workflow:`` entry is rejected instead of recursing
    without bound.

    Returns (workflow, job_configuration).
    """
    from taskmaestro.workflow_task import workflow_task as _workflow_task

    resolved_path = workflow_path.resolve()
    if resolved_path in _ancestors:
        chain = " -> ".join(str(p) for p in (*sorted(_ancestors), resolved_path))
        raise ConfigLoadError(f"Recursive workflow reference: {chain}")
    ancestors = _ancestors | {resolved_path}

    # 1. Parse workflow YAML
    try:
        raw = _yaml_load(workflow_path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"YAML parse error: {exc}") from exc
    except OSError as exc:
        raise ConfigLoadError(f"Cannot read file '{workflow_path}': {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigLoadError("YAML file must contain a mapping at top level")

    # 2. Parse input YAML (optional for inner workflows)
    raw_input: dict[str, Any] = {}
    if input_path is not None:
        try:
            raw_input = _yaml_load(input_path.read_text())
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

    # 4. Resolve task import paths (handles both task: and workflow: entries)
    base_dir = workflow_path.parent
    task_classes: dict[str, type[Task[Any, Any]]] = {}
    installed_task_names = registered_task_names()
    for task_config in config.workflow.tasks:
        if task_config.workflow:
            # Recursive workflow reference
            inner_wf_path = base_dir / task_config.workflow
            inner_input_path = (
                base_dir / task_config.workflow_input if task_config.workflow_input else None
            )
            inner_wf, inner_jc = _load_workflow_only(
                inner_wf_path, inner_input_path, _ancestors=ancestors
            )
            inner_name = task_config.name if task_config.name else inner_wf.name
            wrapped_cls = _workflow_task(inner_wf, name=inner_name, job_configuration=inner_jc)
            # Use a synthetic key for this entry (the workflow path)
            task_classes[task_config.workflow] = wrapped_cls
        else:
            assert task_config.task is not None
            try:
                if task_config.task in installed_task_names:
                    cls = get_registered_task(task_config.task)
                else:
                    cls = import_class(task_config.task)
            except PluginLoadError as exc:
                raise ConfigLoadError(str(exc)) from exc
            if not (isinstance(cls, type) and issubclass(cls, Task)):
                raise ConfigLoadError(f"'{task_config.task}' is not a Task subclass")
            task_classes[task_config.task] = cls

    # Helper to get the lookup key for a task config entry
    def _task_key(tc: TaskConfig) -> str:
        return tc.workflow if tc.workflow else tc.task  # type: ignore[return-value]

    # 5. Build a lookup from instance names and import paths to registered names.
    name_lookup: dict[str, str] = {}
    for task_config in config.workflow.tasks:
        key = _task_key(task_config)
        registered_name = task_config.name if task_config.name else task_classes[key].name
        name_lookup[key] = registered_name
        if task_config.name:
            name_lookup[task_config.name] = registered_name

    def _resolve_yaml_dep(dep_str: str, context_task: str) -> str:
        """Resolve a YAML dependency string to a registered task name."""
        if dep_str in name_lookup:
            return name_lookup[dep_str]
        raise ConfigLoadError(f"Dependency '{dep_str}' for task '{context_task}' not found")

    def _resolve_yaml_output_ref(raw_ref: Any, context_task: str) -> OutputReference:
        """Resolve a YAML task or ``[task, field]`` output reference."""
        if isinstance(raw_ref, str):
            return _resolve_yaml_dep(raw_ref, context_task)
        if isinstance(raw_ref, list):
            if len(raw_ref) != 2 or not all(isinstance(item, str) for item in raw_ref):
                raise ConfigLoadError(
                    f"Collection member must be a task name or [task, field], "
                    f"got {raw_ref!r} for task '{context_task}'"
                )
            return (_resolve_yaml_dep(raw_ref[0], context_task), raw_ref[1])
        raise ConfigLoadError(
            f"Collection member must be a task name or [task, field], "
            f"got {raw_ref!r} for task '{context_task}'"
        )

    def _resolve_yaml_collection(raw_collection: Any, context_task: str) -> CollectionDependency:
        """Resolve a YAML collect list or mapping."""
        if isinstance(raw_collection, list):
            return collect(
                *(_resolve_yaml_output_ref(member, context_task) for member in raw_collection)
            )
        if isinstance(raw_collection, dict):
            if not all(isinstance(key, str) for key in raw_collection):
                raise ConfigLoadError(f"Collection keys must be strings for task '{context_task}'")
            return collect(
                {
                    key: _resolve_yaml_output_ref(member, context_task)
                    for key, member in raw_collection.items()
                }
            )
        raise ConfigLoadError(
            f"'collect' must contain a list or mapping for task '{context_task}'"
        )

    # 6. Detect linear vs DAG mode
    has_depends_on = any(
        tc.depends_on is not None or tc.map is not None for tc in config.workflow.tasks
    )

    # 6b. Detect per-task config format early (before building workflow)
    all_registered_names: set[str] = set()
    for task_config in config.workflow.tasks:
        key = _task_key(task_config)
        registered_name = task_config.name if task_config.name else task_classes[key].name
        all_registered_names.add(registered_name)

    is_per_task_config = bool(raw_input) and all(
        key in all_registered_names and isinstance(raw_input[key], (dict, type(None)))
        for key in raw_input
    )

    per_task_data: dict[str, dict[str, Any]] = {}
    per_task_cfg_fields: dict[str, list[str]] = {}
    if is_per_task_config:
        for task_name, task_values in raw_input.items():
            per_task_data[task_name] = dict(task_values) if task_values else {}
            if task_values:
                task_config = next(
                    tc
                    for tc in config.workflow.tasks
                    if (tc.name or task_classes[_task_key(tc)].name) == task_name
                )
                map_source = task_config.map.over if task_config.map is not None else None
                per_task_cfg_fields[task_name] = [
                    field_name for field_name in task_values if field_name != map_source
                ]

    # 7. Resolve result_task
    result_task_name: str | None = None
    if config.workflow.result_task:
        if config.workflow.result_task in name_lookup:
            result_task_name = name_lookup[config.workflow.result_task]
        else:
            raise ConfigLoadError(f"result_task '{config.workflow.result_task}' not found")

    # 8. Build Workflow. Both linear and DAG configs go through the builder so
    #    that ``name:`` overrides, config_fields and validation behave the same.
    #    In linear mode each task depends on the whole output of the previous one.
    builder = WorkflowBuilder(config.workflow.name, result_task=result_task_name)
    previous_registered: str | None = None
    for task_config in config.workflow.tasks:
        key = _task_key(task_config)
        cls = task_classes[key]
        registered_name = name_lookup[key] if not task_config.name else task_config.name
        instance_name = task_config.name
        cfg_fields = task_config.config_fields or per_task_cfg_fields.get(registered_name)
        mapped_over = (
            TaskMap(**task_config.map.model_dump()) if task_config.map is not None else None
        )
        deps: str | list[str] | dict[str, Any] | _LinearDep | None = task_config.depends_on
        if not has_depends_on and previous_registered is not None:
            # Linear mode: chain on the previous task's registered name, which
            # the builder accepts verbatim as a string dependency.
            deps = _LinearDep(previous_registered)
        previous_registered = registered_name

        try:
            if deps is None:
                builder.add_task(
                    cls,
                    name=instance_name,
                    config_fields=cfg_fields,
                    mapped_over=mapped_over,
                )
            elif isinstance(deps, _LinearDep):
                builder.add_task(
                    cls,
                    name=instance_name,
                    depends_on=deps.upstream,
                    config_fields=cfg_fields,
                    mapped_over=mapped_over,
                )
            elif isinstance(deps, str):
                resolved_dep = _resolve_yaml_dep(deps, key)
                builder.add_task(
                    cls,
                    name=instance_name,
                    depends_on=resolved_dep,
                    config_fields=cfg_fields,
                    mapped_over=mapped_over,
                )
            elif isinstance(deps, list):
                if len(deps) != 2 or not all(isinstance(e, str) for e in deps):
                    raise ConfigLoadError(
                        f"List depends_on must be [task_path, field_name], "
                        f"got {deps!r} for task '{key}'"
                    )
                upstream_path, field_name = deps
                resolved_dep = _resolve_yaml_dep(upstream_path, key)
                builder.add_task(
                    cls,
                    name=instance_name,
                    depends_on=(resolved_dep, field_name),
                    config_fields=cfg_fields,
                    mapped_over=mapped_over,
                )
            elif isinstance(deps, dict):
                fan_in: dict[
                    str,
                    type[Task[Any, Any]]
                    | str
                    | tuple[type[Task[Any, Any]] | str, str]
                    | CollectionDependency,
                ] = {}
                for field_name, upstream_ref in deps.items():
                    if isinstance(upstream_ref, dict) and set(upstream_ref) == {"collect"}:
                        fan_in[field_name] = _resolve_yaml_collection(upstream_ref["collect"], key)
                    elif isinstance(upstream_ref, list):
                        if len(upstream_ref) != 2 or not all(
                            isinstance(e, str) for e in upstream_ref
                        ):
                            raise ConfigLoadError(
                                f"List dep must be [task_path, field_name], "
                                f"got {upstream_ref!r} for field '{field_name}' "
                                f"on task '{key}'"
                            )
                        up_path, up_field = upstream_ref
                        resolved_dep = _resolve_yaml_dep(up_path, key)
                        fan_in[field_name] = (resolved_dep, up_field)
                    elif isinstance(upstream_ref, str):
                        resolved_dep = _resolve_yaml_dep(upstream_ref, key)
                        fan_in[field_name] = resolved_dep
                    else:
                        raise ConfigLoadError(
                            f"Invalid dependency {upstream_ref!r} for field "
                            f"'{field_name}' on task '{key}'"
                        )
                builder.add_task(
                    cls,
                    name=instance_name,
                    depends_on=fan_in,
                    config_fields=cfg_fields,
                    mapped_over=mapped_over,
                )
        except WorkflowDefinitionError as exc:
            raise ConfigLoadError(f"Workflow validation failed: {exc}") from exc
    try:
        workflow = builder.build()
    except Exception as exc:
        raise ConfigLoadError(f"Workflow validation failed: {exc}") from exc

    # 9. Build JobConfiguration if per-task config detected
    job_configuration: JobConfiguration | None = None
    if is_per_task_config:
        job_configuration = JobConfiguration(per_task_data)

    return workflow, job_configuration


def load_workflow_from_yaml(workflow_path: str | Path, input_path: str | Path) -> LoadedWorkflow:
    """Load a complete workflow configuration from a workflow YAML file and an input YAML file.

    Returns a LoadedWorkflow with workflow, runner, job, and context
    ready to execute.

    Raises ConfigLoadError for any loading or validation failure.
    """
    workflow_path = Path(workflow_path)
    input_path = Path(input_path)

    # 1. Parse workflow YAML (needed for runner/context config)
    try:
        raw = _yaml_load(workflow_path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"YAML parse error: {exc}") from exc
    except OSError as exc:
        raise ConfigLoadError(f"Cannot read file '{workflow_path}': {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigLoadError("YAML file must contain a mapping at top level")

    # 2. Parse input YAML
    try:
        raw_input = _yaml_load(input_path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"Input YAML parse error: {exc}") from exc
    except OSError as exc:
        raise ConfigLoadError(f"Cannot read input file '{input_path}': {exc}") from exc

    if not isinstance(raw_input, dict):
        raise ConfigLoadError("Input YAML file must contain a mapping")

    # 3. Schema validation (for runner/context sections)
    try:
        config = YamlWorkflowConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigLoadError(f"YAML schema validation error: {exc}") from exc

    # 4. Build workflow and job_configuration via shared helper
    workflow, job_configuration = _load_workflow_only(workflow_path, input_path)

    # 5. Validate input and build Job
    job: Job[Any]
    try:
        if job_configuration is not None:
            job = Job(workflow, EmptyConfig(), job_configuration=job_configuration)
        else:
            # Flat config mode: find root tasks from the built workflow
            root_task_classes = [
                workflow._tasks[task_name]
                for task_name, deps in workflow._dependencies.items()
                if deps is None and not workflow.get_config_fields(task_name)
            ]
            if not root_task_classes and all(
                deps is not None for deps in workflow._dependencies.values()
            ):
                # The workflow is self-contained, for example a task fed only by
                # explicitly empty collection dependencies.
                job = Job(workflow, EmptyConfig())
            elif not root_task_classes:
                raise ConfigLoadError(
                    "Workflow has configured root tasks but no per-task input configuration"
                )
            else:
                input_type = get_input_type(root_task_classes[0])
                try:
                    validated_input = input_type.model_validate(raw_input)
                except ValidationError as exc:
                    raise ConfigLoadError(f"Input validation error: {exc}") from exc

                job = Job(workflow, validated_input)
    except WorkflowDefinitionError as exc:
        raise ConfigLoadError(f"Job validation failed: {exc}") from exc

    # 6. Instantiate hooks
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

    # 7. Build ExecutionContext
    ctx_kwargs: dict[str, Any] = {}
    if config.context.correlation_id:
        ctx_kwargs["correlation_id"] = config.context.correlation_id
    if config.context.scratch_dir:
        ctx_kwargs["scratch_dir"] = Path(config.context.scratch_dir)
    context = ExecutionContext(**ctx_kwargs)
    for key, value in config.context.services.items():
        context.register(key, value)

    # 8. Return LoadedWorkflow
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
