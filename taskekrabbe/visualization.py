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


def _apply_redirect(name: str, redirect: dict[str, str]) -> str:
    """Replace *name* with its redirect target if one exists."""
    return redirect.get(name, name)


def _emit_edges(
    tasks: list[tuple[str, type]],
    workflow: Workflow,
    task_by_name: dict[str, type],
    configured_tasks: set[str],
    lines: list[str],
    indent: str,
    source_redirect: dict[str, str],
    target_redirect: dict[str, str],
    *,
    skip_start: bool = False,
    skip_end_sinks: bool = False,
) -> None:
    """Emit edge lines for a set of tasks, applying redirects."""
    for task_name, task_cls in tasks:
        deps = workflow.get_dependencies(task_name)
        tgt_name = _apply_redirect(task_name, target_redirect)

        if deps is None:
            if not skip_start and task_name not in configured_tasks:
                input_name = _safe_type_name(get_input_type(task_cls), task_cls)
                lines.append(f"{indent}_start_ -->|{input_name}| {tgt_name}")
        elif isinstance(deps, str):
            upstream_src = _apply_redirect(deps, source_redirect)
            upstream_cls = task_by_name[deps]
            output_name = _safe_type_name(get_output_type(upstream_cls), upstream_cls)
            lines.append(f"{indent}{upstream_src} -->|{output_name}| {tgt_name}")
        elif isinstance(deps, tuple):
            upstream_name, field_name = deps
            upstream_src = _apply_redirect(upstream_name, source_redirect)
            label = _field_type_label(task_by_name, upstream_name, field_name)
            lines.append(f"{indent}{upstream_src} -->|{label}| {tgt_name}")
        elif isinstance(deps, dict):
            for down_field, upstream_ref in sorted(deps.items()):
                if isinstance(upstream_ref, tuple):
                    upstream_name, up_field = upstream_ref
                    upstream_src = _apply_redirect(upstream_name, source_redirect)
                    label = _field_type_label(task_by_name, upstream_name, up_field)
                    lines.append(f"{indent}{upstream_src} -->|{down_field}: {label}| {tgt_name}")
                else:
                    upstream_src = _apply_redirect(upstream_ref, source_redirect)
                    up_cls = task_by_name[upstream_ref]
                    output_name = _safe_type_name(get_output_type(up_cls), up_cls)
                    lines.append(
                        f"{indent}{upstream_src} -->|{down_field}: {output_name}| {tgt_name}"
                    )


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

    Tasks created via ``workflow_task`` are expanded into Mermaid subgraphs
    showing the inner workflow's structure.
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

    # Detect workflow_task nodes and build redirect maps
    source_redirect: dict[str, str] = {}
    target_redirect: dict[str, str] = {}
    workflow_task_nodes: dict[str, Workflow] = {}

    for task_name, task_cls in tasks:
        inner_wf = getattr(task_cls, "_inner_workflow", None)
        if inner_wf is not None:
            workflow_task_nodes[task_name] = inner_wf
            # Find inner root(s): tasks with no dependencies and no config_fields
            inner_tasks = inner_wf.topological_order()
            for iname, _icls in inner_tasks:
                ideps = inner_wf.get_dependencies(iname)
                if ideps is None and not inner_wf.get_config_fields(iname):
                    target_redirect[task_name] = f"{task_name}__{iname}"
                    break
            # Result task → source redirect
            source_redirect[task_name] = f"{task_name}__{inner_wf.result_task_name}"

    # Start and end nodes
    lines.append('    _start_(("start"))')
    lines.append('    _end_(("end"))')

    # JobConfiguration node (if there are configured tasks)
    if configured_tasks:
        lines.append('    _job_config_[("JobConfiguration")]')

    # Task node definitions
    for task_name, _task_cls in tasks:
        if task_name in workflow_task_nodes:
            # Render as subgraph
            inner_wf = workflow_task_nodes[task_name]
            inner_tasks = inner_wf.topological_order()
            inner_task_by_name: dict[str, type] = {n: c for n, c in inner_tasks}
            inner_configured = {n for n, _c in inner_tasks if inner_wf.get_config_fields(n)}

            lines.append(f'    subgraph {task_name}["{task_name}"]')

            # Inner node definitions
            for iname, _icls in inner_tasks:
                prefixed = f"{task_name}__{iname}"
                lines.append(f'        {prefixed}["{iname}"]')

            # Inner edges (skip start/end — handled by outer redirects)
            inner_source_redirect: dict[str, str] = {
                n: f"{task_name}__{n}" for n, _c in inner_tasks
            }
            inner_target_redirect: dict[str, str] = {
                n: f"{task_name}__{n}" for n, _c in inner_tasks
            }
            _emit_edges(
                inner_tasks,
                inner_wf,
                inner_task_by_name,
                inner_configured,
                lines,
                "        ",
                inner_source_redirect,
                inner_target_redirect,
                skip_start=True,
                skip_end_sinks=True,
            )

            lines.append("    end")
        else:
            lines.append(f'    {task_name}["{task_name}"]')

    # Outer edge definitions
    _emit_edges(
        tasks,
        workflow,
        task_by_name,
        configured_tasks,
        lines,
        "    ",
        source_redirect,
        target_redirect,
    )

    # JobConfiguration dashed edges to configured tasks
    if configured_tasks:
        for task_name in sorted(configured_tasks):
            cf = workflow.get_config_fields(task_name)
            label = ", ".join(sorted(cf))
            lines.append(f"    _job_config_ -.->|{label}| {task_name}")

    # Sink tasks: edge to end, labeled with output type
    for task_name, task_cls in sinks:
        src = _apply_redirect(task_name, source_redirect)
        output_name = _safe_type_name(get_output_type(task_cls), task_cls)
        lines.append(f"    {src} -->|{output_name}| _end_")

    return "\n".join(lines) + "\n"
