"""YAML-based workflow configuration loader."""

from __future__ import annotations

import importlib
import sys
import typing
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from taskmaestro.conditions import Condition, when
from taskmaestro.context import ExecutionContext
from taskmaestro.dependencies import (
    CollectionDependency,
    FirstOfDependency,
    OutputReference,
    collect,
    first_of,
)
from taskmaestro.discovery import (
    get_registered_task,
    get_registered_workflow,
    registered_task_names,
    registered_workflow_names,
)
from taskmaestro.exceptions import (
    ConfigLoadError,
    PluginLoadError,
    WorkflowDefinitionError,
    WorkflowLookupError,
)
from taskmaestro.hooks.base import BaseHook
from taskmaestro.job import EmptyConfig, Job, JobConfiguration
from taskmaestro.mapping import TaskMap
from taskmaestro.runner import Runner
from taskmaestro.task import Task
from taskmaestro.workflow import Workflow, WorkflowBuilder

# --- Pydantic schema models for YAML validation ---


class TaskMapConfig(BaseModel):
    """Mapped execution settings for a YAML task entry."""

    over: str
    key_as: str
    value_as: str
    error_mode: typing.Literal["fail_fast", "collect_all"] = "fail_fast"


class WhenConfig(BaseModel):
    """The ``when:`` condition of a YAML task entry.

    Either a switch case, ``{switch: route, case: express}`` (``case`` may be a
    list) or ``{switch: route, otherwise: true}``, or a condition on an output
    field: ``{ref: [task, field], equals: value}`` with ``in: [...]`` or
    ``truthy: true|false`` instead of ``equals``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    switch: str | None = None
    case: Any = None
    otherwise: bool = False
    ref: str | list[str] | None = None
    equals: Any = None
    in_: list[Any] | None = Field(default=None, alias="in")
    truthy: bool | None = None

    @model_validator(mode="after")
    def _check_form(self) -> WhenConfig:
        given = self.model_fields_set
        field_ops = given & {"equals", "in_", "truthy"}
        if self.switch is not None:
            if self.ref is not None or field_ops:
                raise ValueError("'when' with 'switch' accepts only 'case' or 'otherwise'")
            if ("case" in given) == self.otherwise:
                raise ValueError("'when' with 'switch' needs either 'case' or 'otherwise: true'")
            if "case" in given and (self.case is None or self.case == []):
                raise ValueError("'when' 'case' needs at least one branch label")
            return self
        if self.ref is not None:
            if given & {"case", "otherwise"}:
                raise ValueError("'when' with 'ref' does not accept 'case' or 'otherwise'")
            if len(field_ops) != 1:
                raise ValueError("'when' with 'ref' needs exactly one of 'equals', 'in', 'truthy'")
            if "truthy" in given and self.truthy is None:
                raise ValueError("'when' 'truthy' must be true or false")
            if "in_" in given and not self.in_:
                raise ValueError("'when' 'in' needs at least one value")
            if isinstance(self.ref, list) and (
                len(self.ref) != 2 or not all(isinstance(item, str) for item in self.ref)
            ):
                raise ValueError("'when' 'ref' must be a task name or [task, field]")
            return self
        raise ValueError("'when' needs either 'switch' or 'ref'")


class TaskConfig(BaseModel):
    """A single task entry in the YAML workflow config."""

    task: str | None = None
    workflow: str | None = None
    workflow_input: str | None = None
    name: str | None = None
    depends_on: str | list[str] | dict[str, Any] | None = None
    config_fields: list[str] | None = None
    map: TaskMapConfig | None = None
    when: WhenConfig | None = None

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

    model_config = ConfigDict(extra="forbid")

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
            try:
                is_duplicate = key in keys
            except TypeError as exc:
                # Match PyYAML's own error for unhashable keys (e.g. ``? [a, b]``).
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found unhashable key ({exc})",
                    key_node.start_mark,
                ) from exc
            if is_duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            keys.add(key)
        super().flatten_mapping(node)


def yaml_load(text: str) -> Any:
    """Safely parse YAML while rejecting duplicate mapping keys."""
    return yaml.load(text, Loader=_UniqueKeyLoader)


_yaml_load = yaml_load


def resolve_task(
    task_id: str, installed_task_names: set[str] | None = None
) -> type[Task[Any, Any]]:
    """Resolve the ``task:`` value of a YAML file to a Task class.

    A registered entry-point name wins; otherwise ``task_id`` is imported as a
    dotted class path. Raises ConfigLoadError if it cannot be resolved.
    """
    if installed_task_names is None:
        installed_task_names = registered_task_names()
    try:
        if task_id in installed_task_names:
            cls: type[Task[Any, Any]] = get_registered_task(task_id)
        else:
            cls = import_class(task_id)
    except PluginLoadError as exc:
        raise ConfigLoadError(str(exc)) from exc
    if not (isinstance(cls, type) and issubclass(cls, Task)):
        raise ConfigLoadError(f"'{task_id}' is not a Task subclass")
    return cls


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
        if exc.name is not None and _is_module_or_parent(exc.name, module_path):
            raise ConfigLoadError(f"Cannot import module '{module_path}': {exc}") from exc
        # The module exists but one of its own imports is missing.
        raise ConfigLoadError(f"Error while importing module '{module_path}': {exc!r}") from exc
    except Exception as exc:
        # SyntaxError, ImportError, or any error raised by module-level code.
        raise ConfigLoadError(f"Error while importing module '{module_path}': {exc!r}") from exc
    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise ConfigLoadError(f"Module '{module_path}' has no attribute '{class_name}'") from None
    return cls  # type: ignore[no-any-return]


def _is_module_or_parent(missing: str, module_path: str) -> bool:
    """Return whether *missing* is *module_path* itself or one of its parent packages."""
    return module_path == missing or module_path.startswith(f"{missing}.")


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
class _Entry:
    """One resolved ``tasks:`` entry, kept positionally."""

    config: TaskConfig
    cls: type[Task[Any, Any]]
    key: str  # import path or inner-workflow path as written in YAML
    registered_name: str


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


def _read_input_yaml(input_path: Path) -> dict[str, Any]:
    """Parse an input YAML file, which must contain a mapping."""
    try:
        raw_input = _yaml_load(input_path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"Input YAML parse error: {exc}") from exc
    except OSError as exc:
        raise ConfigLoadError(f"Cannot read input file '{input_path}': {exc}") from exc
    if not isinstance(raw_input, dict):
        raise ConfigLoadError("Input YAML file must contain a mapping")
    return raw_input


def _check_input_keys(raw_input: dict[str, Any], task_names: typing.Collection[str]) -> None:
    """Check that input keys are task names and their values mappings or null."""
    for key, value in raw_input.items():
        if key not in task_names:
            raise ConfigLoadError(
                f"Input top-level key '{key}' is not a task name "
                f"(known tasks: {sorted(task_names)})"
            )
        if value is not None and not isinstance(value, dict):
            raise ConfigLoadError(
                f"Input value for task '{key}' must be a mapping or null "
                f"(got {type(value).__name__})"
            )


def _load_workflow_only(
    workflow_path: Path,
    input_path: Path | None = None,
    *,
    _ancestors: tuple[Path, ...] = (),
) -> tuple[Workflow, JobConfiguration | None]:
    """Build a Workflow and optional JobConfiguration from YAML files.

    This is the core logic shared by ``load_workflow_from_yaml`` and
    recursive ``workflow:`` references in YAML configs.  ``_ancestors`` holds
    the resolved paths of every enclosing workflow file, outermost first, so
    that the reported reference chain is in nesting order and a self- or
    mutually-referencing ``workflow:`` entry is rejected instead of recursing
    without bound.

    Returns (workflow, job_configuration).
    """
    from taskmaestro.workflow_task import workflow_task as _workflow_task

    resolved_path = workflow_path.resolve()
    if resolved_path in _ancestors:
        chain = " -> ".join(str(p) for p in (*_ancestors, resolved_path))
        raise ConfigLoadError(f"Recursive workflow reference: {chain}")
    ancestors = (*_ancestors, resolved_path)

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
        raw_input = _read_input_yaml(input_path)

    # 3. Schema validation
    try:
        config = YamlWorkflowConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigLoadError(f"YAML schema validation error: {exc}") from exc

    # 4. Resolve task import paths (handles both task: and workflow: entries).
    #    Entries are kept positionally: the same class path or inner YAML file
    #    may legitimately appear more than once under different ``name:``s.
    base_dir = workflow_path.parent
    entries: list[_Entry] = []
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
            cls: type[Task[Any, Any]] = _workflow_task(
                inner_wf, name=inner_name, job_configuration=inner_jc
            )
            key = task_config.workflow
        else:
            assert task_config.task is not None
            cls = resolve_task(task_config.task, installed_task_names)
            key = task_config.task
        registered_name = task_config.name if task_config.name else cls.name
        entries.append(_Entry(task_config, cls, key, registered_name))

    # 5. Build a lookup from instance names and import paths to registered names.
    #    A key that maps to more than one registered name is ambiguous and is
    #    rejected when used, with the candidates listed.
    candidates: dict[str, set[str]] = {}
    for entry in entries:
        candidates.setdefault(entry.key, set()).add(entry.registered_name)
        if entry.config.name:
            candidates.setdefault(entry.config.name, set()).add(entry.registered_name)

    def _resolve_yaml_dep(dep_str: str, context_task: str, *, what: str = "Dependency") -> str:
        """Resolve a YAML dependency string to a registered task name."""
        where = f" for task '{context_task}'" if context_task else ""
        names = candidates.get(dep_str)
        if names is None:
            raise ConfigLoadError(f"{what} '{dep_str}'{where} not found")
        if len(names) > 1:
            raise ConfigLoadError(
                f"{what} '{dep_str}'{where} is ambiguous; it matches "
                f"{sorted(names)}. Use the instance name."
            )
        return next(iter(names))

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

    def _resolve_yaml_alternatives(raw_members: Any, context_task: str) -> FirstOfDependency:
        """Resolve a YAML first_of list."""
        if not isinstance(raw_members, list) or len(raw_members) < 2:
            raise ConfigLoadError(
                f"'first_of' must contain a list of at least two members for task '{context_task}'"
            )
        return first_of(
            *(_resolve_yaml_output_ref(member, context_task) for member in raw_members)
        )

    def _resolve_yaml_condition(raw: WhenConfig | None, context_task: str) -> Condition | None:
        """Resolve a YAML ``when:`` entry to a condition."""
        if raw is None:
            return None
        if raw.switch is not None:
            source = _resolve_yaml_dep(raw.switch, context_task, what="Condition switch")
            if raw.otherwise:
                return Condition(source, "otherwise")
            cases = raw.case if isinstance(raw.case, list) else [raw.case]
            return Condition(source, "in", tuple(cases))
        assert raw.ref is not None
        ref: OutputReference
        if isinstance(raw.ref, list):
            ref = (_resolve_yaml_dep(raw.ref[0], context_task, what="Condition"), raw.ref[1])
        else:
            ref = _resolve_yaml_dep(raw.ref, context_task, what="Condition")
        given = raw.model_fields_set
        if "equals" in given:
            return when(ref, equals=raw.equals)
        if "in_" in given:
            assert raw.in_  # checked by WhenConfig
            return when(ref, in_=raw.in_)
        assert raw.truthy is not None  # checked by WhenConfig
        return when(ref, truthy=raw.truthy)

    # 6. Detect linear vs DAG mode. Conditions refer to other tasks, so a
    #    workflow with a 'when' entry is a DAG as well.
    has_depends_on = any(
        tc.depends_on is not None or tc.map is not None or tc.when is not None
        for tc in config.workflow.tasks
    )

    # 6b. Input YAML always uses per-task configuration. Top-level keys are
    # registered task instance names and each value is a mapping or null.
    entry_by_name = {entry.registered_name: entry for entry in entries}
    _check_input_keys(raw_input, entry_by_name)

    per_task_data: dict[str, dict[str, Any]] = {}
    per_task_cfg_fields: dict[str, list[str]] = {}
    for task_name, task_values in raw_input.items():
        per_task_data[task_name] = dict(task_values) if task_values else {}
        if task_values:
            task_config = entry_by_name[task_name].config
            map_source = task_config.map.over if task_config.map is not None else None
            per_task_cfg_fields[task_name] = [
                field_name for field_name in task_values if field_name != map_source
            ]

    # 7. Resolve result_task
    result_task_name: str | None = None
    if config.workflow.result_task:
        result_task_name = _resolve_yaml_dep(config.workflow.result_task, "", what="result_task")

    # 8. Build Workflow. Both linear and DAG configs go through the builder so
    #    that ``name:`` overrides, config_fields and validation behave the same.
    #    In linear mode each task depends on the whole output of the previous one.
    builder = WorkflowBuilder(config.workflow.name, result_task=result_task_name)
    previous_registered: str | None = None
    for entry in entries:
        task_config = entry.config
        key = entry.key
        cls = entry.cls
        registered_name = entry.registered_name
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
            condition = _resolve_yaml_condition(task_config.when, key)
            if deps is None:
                builder.add_task(
                    cls,
                    name=instance_name,
                    config_fields=cfg_fields,
                    mapped_over=mapped_over,
                    when=condition,
                )
            elif isinstance(deps, _LinearDep):
                builder.add_task(
                    cls,
                    name=instance_name,
                    depends_on=deps.upstream,
                    config_fields=cfg_fields,
                    mapped_over=mapped_over,
                    when=condition,
                )
            elif isinstance(deps, str):
                resolved_dep = _resolve_yaml_dep(deps, key)
                builder.add_task(
                    cls,
                    name=instance_name,
                    depends_on=resolved_dep,
                    config_fields=cfg_fields,
                    mapped_over=mapped_over,
                    when=condition,
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
                    when=condition,
                )
            elif isinstance(deps, dict):
                fan_in: dict[
                    str,
                    type[Task[Any, Any]]
                    | str
                    | tuple[type[Task[Any, Any]] | str, str]
                    | CollectionDependency
                    | FirstOfDependency,
                ] = {}
                for field_name, upstream_ref in deps.items():
                    if isinstance(upstream_ref, dict) and set(upstream_ref) == {"collect"}:
                        fan_in[field_name] = _resolve_yaml_collection(upstream_ref["collect"], key)
                    elif isinstance(upstream_ref, dict) and set(upstream_ref) == {"first_of"}:
                        fan_in[field_name] = _resolve_yaml_alternatives(
                            upstream_ref["first_of"], key
                        )
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
                    when=condition,
                )
        except WorkflowDefinitionError as exc:
            raise ConfigLoadError(f"Workflow validation failed: {exc}") from exc
    try:
        workflow = builder.build()
    except Exception as exc:
        raise ConfigLoadError(f"Workflow validation failed: {exc}") from exc

    # 9. Input YAML always becomes per-task job configuration.
    return workflow, JobConfiguration(per_task_data)


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
    _read_input_yaml(input_path)

    # 3. Schema validation (for runner/context sections)
    try:
        config = YamlWorkflowConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigLoadError(f"YAML schema validation error: {exc}") from exc

    # 4. Build workflow and job_configuration via shared helper
    workflow, job_configuration = _load_workflow_only(workflow_path, input_path)

    # 5. Validate per-task input and build Job. The root input is EmptyConfig
    # because all external YAML values are delivered through JobConfiguration.
    assert job_configuration is not None
    try:
        job = Job(workflow, EmptyConfig(), job_configuration=job_configuration)
    except WorkflowDefinitionError as exc:
        raise ConfigLoadError(f"Job validation failed: {exc}") from exc

    # 6. Instantiate hooks
    hooks: list[BaseHook] = []
    for hook_config in config.runner.hooks:
        hook_cls = import_class(hook_config.hook)
        if not (isinstance(hook_cls, type) and issubclass(hook_cls, BaseHook)):
            raise ConfigLoadError(f"'{hook_config.hook}' is not a BaseHook subclass")
        try:
            coerced_params = _coerce_hook_params(hook_cls, hook_config.params)
            hooks.append(hook_cls(**coerced_params))
        except Exception as exc:
            # TypeError for bad parameters, but a hook's __init__ may raise anything.
            raise ConfigLoadError(
                f"Cannot instantiate hook '{hook_config.hook}': {type(exc).__name__}: {exc}"
            ) from exc

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


load_workflow_only = _load_workflow_only


def registered_workflow(name: str) -> Workflow:
    """Load a registered workflow, raising WorkflowLookupError if it is unknown or invalid.

    The error message lists the registered workflow names.
    """
    try:
        return get_registered_workflow(name)
    except PluginLoadError as exc:
        names = sorted(registered_workflow_names())
        available = ", ".join(names) if names else "none"
        raise WorkflowLookupError(f"{exc} (registered workflows: {available})") from exc


def load_input_configuration(workflow: Workflow, input_path: str | Path) -> JobConfiguration:
    """Read an input YAML file as per-task configuration for ``workflow``.

    The input is checked the same way as for YAML workflows: top-level keys
    must be task names and each value a mapping or null.
    """
    raw_input = _read_input_yaml(Path(input_path))
    _check_input_keys(raw_input, {name for name, _task in workflow.topological_order()})
    return JobConfiguration({name: dict(values or {}) for name, values in raw_input.items()})


def load_registered_workflow(name: str, input_path: str | Path) -> LoadedWorkflow:
    """Load a registered workflow with an input YAML file, ready to run.

    Registered workflows have no ``runner:`` or ``context:`` sections, so the
    workflow runs with a default ``Runner`` and ``ExecutionContext``.
    Raises ConfigLoadError for an unknown workflow or invalid input.
    """
    workflow = registered_workflow(name)
    job_configuration = load_input_configuration(workflow, input_path)
    try:
        job = Job(workflow, EmptyConfig(), job_configuration=job_configuration)
    except WorkflowDefinitionError as exc:
        raise ConfigLoadError(f"Job validation failed: {exc}") from exc
    return LoadedWorkflow(workflow=workflow, runner=Runner(), job=job, context=ExecutionContext())


@contextmanager
def workflow_imports(directory: str | Path) -> Iterator[None]:
    """Make task modules in ``directory`` importable for the duration of loading."""
    original_path = sys.path.copy()
    sys.path.insert(0, str(Path(directory).resolve()))
    try:
        yield
    finally:
        sys.path[:] = original_path
