"""Mermaid diagram visualization for workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from workflow_runner.task import get_input_type, get_output_type

if TYPE_CHECKING:
    from workflow_runner.workflow import Workflow


def to_mermaid(workflow: Workflow) -> str:
    """Generate a Mermaid diagram string for a workflow.

    Returns a ``graph TD`` block with a start node, task nodes, and edges
    labeled with the data types flowing between them.
    """
    lines: list[str] = ["---", f"title: {workflow.name}", "---", "graph TD"]

    tasks = workflow.topological_order()
    task_by_name = {t.name: t for t in tasks}

    # Find sink tasks (no downstream dependents)
    has_dependents: set[str] = set()
    for task_cls in tasks:
        deps: dict[str, str] | str | None = workflow.get_dependencies(task_cls.name)
        if isinstance(deps, str):
            has_dependents.add(deps)
        elif isinstance(deps, dict):
            has_dependents.update(deps.values())
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
        elif isinstance(deps, dict):
            # Fan-in: labeled with field: upstream output type
            for field_name, upstream_name in sorted(deps.items()):
                output_name = get_output_type(task_by_name[upstream_name]).__name__
                lines.append(
                    f"    {upstream_name} -->|{field_name}: {output_name}| {task_cls.name}"
                )

    # Sink tasks: edge to end, labeled with output type
    for task_cls in sinks:
        output_name = get_output_type(task_cls).__name__
        lines.append(f"    {task_cls.name} -->|{output_name}| _end_")

    return "\n".join(lines) + "\n"
