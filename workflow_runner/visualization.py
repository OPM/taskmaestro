"""Mermaid diagram visualization for workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from workflow_runner.task import get_input_type, get_output_type

if TYPE_CHECKING:
    from workflow_runner.workflow import Workflow


def _field_type_label(task_by_name: dict[str, type], upstream_name: str, field_name: str) -> str:
    """Return ``'.field: FieldType'`` for a field-ref edge."""
    output_model = get_output_type(task_by_name[upstream_name])
    field_info = output_model.model_fields[field_name]
    annotation = field_info.annotation
    type_label = annotation.__name__ if annotation is not None else "Any"
    return f".{field_name}: {type_label}"


def to_mermaid(workflow: Workflow) -> str:
    """Generate a Mermaid diagram string for a workflow.

    Returns a ``graph TD`` block with a start node, task nodes, and edges
    labeled with the data types flowing between them.
    """
    from workflow_runner.workflow import _extract_upstream_names

    lines: list[str] = ["---", f"title: {workflow.name}", "---", "graph TD"]

    tasks = workflow.topological_order()
    task_by_name: dict[str, type] = {t.name: t for t in tasks}

    # Find sink tasks (no downstream dependents)
    has_dependents: set[str] = set()
    for task_cls in tasks:
        deps = workflow.get_dependencies(task_cls.name)
        has_dependents.update(_extract_upstream_names(deps))
    sinks = [t for t in tasks if t.name not in has_dependents]

    # Start and end nodes
    lines.append('    _start_(("start"))')
    lines.append('    _end_(("end"))')

    # Task node definitions (plain labels, no type info)
    for task_cls in tasks:
        lines.append(f'    {task_cls.name}["{task_cls.name}"]')

    # Edge definitions
    for task_cls in tasks:
        deps = workflow.get_dependencies(task_cls.name)
        if deps is None:
            # Root task: edge from start, labeled with input type
            input_name = get_input_type(task_cls).__name__
            lines.append(f"    _start_ -->|{input_name}| {task_cls.name}")
        elif isinstance(deps, str):
            # Single dependency: labeled with upstream output type
            output_name = get_output_type(task_by_name[deps]).__name__
            lines.append(f"    {deps} -->|{output_name}| {task_cls.name}")
        elif isinstance(deps, tuple):
            # Single dependency, specific output field
            upstream_name, field_name = deps
            label = _field_type_label(task_by_name, upstream_name, field_name)
            lines.append(f"    {upstream_name} -->|{label}| {task_cls.name}")
        elif isinstance(deps, dict):
            # Fan-in: labeled with field: upstream output type
            for down_field, upstream_ref in sorted(deps.items()):
                if isinstance(upstream_ref, tuple):
                    upstream_name, up_field = upstream_ref
                    label = _field_type_label(task_by_name, upstream_name, up_field)
                    lines.append(f"    {upstream_name} -->|{down_field}: {label}| {task_cls.name}")
                else:
                    output_name = get_output_type(task_by_name[upstream_ref]).__name__
                    lines.append(
                        f"    {upstream_ref} -->|{down_field}: {output_name}| {task_cls.name}"
                    )

    # Sink tasks: edge to end, labeled with output type
    for task_cls in sinks:
        output_name = get_output_type(task_cls).__name__
        lines.append(f"    {task_cls.name} -->|{output_name}| _end_")

    return "\n".join(lines) + "\n"
