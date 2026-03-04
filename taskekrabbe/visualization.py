"""Mermaid diagram visualization for workflows."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from taskekrabbe.task import get_input_type, get_output_type

if TYPE_CHECKING:
    from taskekrabbe.job import JobConfiguration
    from taskekrabbe.workflow import Workflow


def _safe_type_name(tp: type, context_cls: type | None = None) -> str:
    """Return a Mermaid-safe type name, resolving module-level aliases.

    When *context_cls* is provided, its module namespace is scanned for a
    variable that refers to *tp*, so that ``GridCase = ObjectModel[X]``
    renders as ``GridCase`` instead of ``ObjectModel[X]``.
    """
    name = tp.__name__ if hasattr(tp, "__name__") else str(tp)
    if "[" not in name:
        return name
    # Try to find a module-level alias name for this generic type
    module_name = getattr(context_cls, "__module__", None) if context_cls else None
    if module_name and module_name in sys.modules:
        mod = sys.modules[module_name]
        for attr_name, attr_val in list(vars(mod).items()):
            if attr_val is tp and not attr_name.startswith("_"):
                return attr_name
    return name.replace("[", "&lsaquo;").replace("]", "&rsaquo;")


def _field_type_label(task_by_name: dict[str, type], upstream_name: str, field_name: str) -> str:
    """Return ``'.field: FieldType'`` for a field-ref edge."""
    upstream_cls = task_by_name[upstream_name]
    output_model = get_output_type(upstream_cls)
    field_info = output_model.model_fields[field_name]
    annotation = field_info.annotation
    type_label = _safe_type_name(annotation, upstream_cls) if annotation is not None else "Any"
    return f".{field_name}: {type_label}"


def to_mermaid(
    workflow: Workflow,
    *,
    job_configuration: JobConfiguration | None = None,
) -> str:
    """Generate a Mermaid diagram string for a workflow.

    Returns a ``graph TD`` block with a start node, task nodes, and edges
    labeled with the data types flowing between them.

    When *job_configuration* is provided, adds a ``_job_config_`` node with
    dashed edges to each configured task.
    """
    from taskekrabbe.workflow import _extract_upstream_names

    lines: list[str] = ["---", f"title: {workflow.name}", "---", "graph TD"]

    tasks = workflow.topological_order()
    task_by_name: dict[str, type] = {name: cls for name, cls in tasks}

    # Find sink tasks (no downstream dependents)
    has_dependents: set[str] = set()
    for task_name, _task_cls in tasks:
        deps = workflow.get_dependencies(task_name)
        has_dependents.update(_extract_upstream_names(deps))
    sinks = [(name, cls) for name, cls in tasks if name not in has_dependents]

    # Collect tasks with config_fields
    configured_tasks = {name for name, _cls in tasks if workflow.get_config_fields(name)}

    # Start and end nodes
    lines.append('    _start_(("start"))')
    lines.append('    _end_(("end"))')

    # JobConfiguration node (if there are configured tasks)
    if configured_tasks:
        lines.append('    _job_config_[("JobConfiguration")]')

    # Task node definitions (plain labels, no type info)
    for task_name, _task_cls in tasks:
        lines.append(f'    {task_name}["{task_name}"]')

    # Edge definitions
    for task_name, task_cls in tasks:
        deps = workflow.get_dependencies(task_name)
        if deps is None:
            if task_name not in configured_tasks:
                # Root task: edge from start, labeled with input type
                input_name = _safe_type_name(get_input_type(task_cls), task_cls)
                lines.append(f"    _start_ -->|{input_name}| {task_name}")
        elif isinstance(deps, str):
            # Single dependency: labeled with upstream output type
            upstream_cls = task_by_name[deps]
            output_name = _safe_type_name(get_output_type(upstream_cls), upstream_cls)
            lines.append(f"    {deps} -->|{output_name}| {task_name}")
        elif isinstance(deps, tuple):
            # Single dependency, specific output field
            upstream_name, field_name = deps
            label = _field_type_label(task_by_name, upstream_name, field_name)
            lines.append(f"    {upstream_name} -->|{label}| {task_name}")
        elif isinstance(deps, dict):
            # Fan-in: labeled with field: upstream output type
            for down_field, upstream_ref in sorted(deps.items()):
                if isinstance(upstream_ref, tuple):
                    upstream_name, up_field = upstream_ref
                    label = _field_type_label(task_by_name, upstream_name, up_field)
                    lines.append(f"    {upstream_name} -->|{down_field}: {label}| {task_name}")
                else:
                    up_cls = task_by_name[upstream_ref]
                    output_name = _safe_type_name(get_output_type(up_cls), up_cls)
                    lines.append(
                        f"    {upstream_ref} -->|{down_field}: {output_name}| {task_name}"
                    )

    # JobConfiguration dashed edges to configured tasks
    if configured_tasks:
        for task_name in sorted(configured_tasks):
            cf = workflow.get_config_fields(task_name)
            label = ", ".join(sorted(cf))
            lines.append(f"    _job_config_ -.->|{label}| {task_name}")

    # Sink tasks: edge to end, labeled with output type
    for task_name, task_cls in sinks:
        output_name = _safe_type_name(get_output_type(task_cls), task_cls)
        lines.append(f"    {task_name} -->|{output_name}| _end_")

    return "\n".join(lines) + "\n"
