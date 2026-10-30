"""Mermaid diagram visualization for workflows."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, get_args, get_origin

from taskmaestro.conditions import BRANCH_FIELD, Guard, plain_label
from taskmaestro.dependencies import CollectionRef, FirstOfRef, OutputRef
from taskmaestro.task import get_input_type

if TYPE_CHECKING:
    from taskmaestro.job import JobConfiguration
    from taskmaestro.workflow import Workflow


def _safe_type_name(tp: Any, context_cls: type | None = None) -> str:
    """Return a Mermaid-safe type name, resolving module-level aliases.

    When *context_cls* is provided, its module namespace is scanned for a
    variable that refers to *tp*, so that ``GridCase = ObjectModel[X]``
    renders as ``GridCase`` instead of ``ObjectModel[X]``.
    """
    origin = get_origin(tp)
    if origin is not None:
        origin_name = getattr(origin, "__name__", str(origin))
        args = ", ".join(_safe_type_name(arg, context_cls) for arg in get_args(tp))
        return f"{origin_name}&lsaquo;{args}&rsaquo;"
    name = tp.__name__ if hasattr(tp, "__name__") else str(tp)
    if "[" not in name:
        return name
    # Try to find a module-level alias name for this generic type
    module_name = getattr(context_cls, "__module__", None) if context_cls else None
    if module_name and module_name in sys.modules:
        mod = sys.modules[module_name]
        for attr_name, attr_val in list(vars(mod).items()):
            # Pydantic may also register the model under its bracketed name.
            if attr_val is tp and attr_name.isidentifier() and not attr_name.startswith("_"):
                return attr_name
    return name.replace("[", "&lsaquo;").replace("]", "&rsaquo;")


def _field_type_label(
    workflow: Workflow,
    task_by_name: dict[str, type],
    upstream_name: str,
    field_name: str,
) -> str:
    """Return ``'.field: FieldType'`` for a field-ref edge."""
    upstream_cls = task_by_name[upstream_name]
    output_model = workflow.get_output_annotation(upstream_name)
    field_info = output_model.model_fields[field_name]
    annotation = field_info.annotation
    type_label = _safe_type_name(annotation, upstream_cls) if annotation is not None else "Any"
    return f".{field_name}: {type_label}"


def _output_ref_label(workflow: Workflow, task_by_name: dict[str, type], ref: OutputRef) -> str:
    """Return the type label for a resolved output reference."""
    task_cls = task_by_name[ref.task_name]
    if ref.output_field is None:
        return _safe_type_name(workflow.get_output_annotation(ref.task_name), task_cls)
    return _field_type_label(workflow, task_by_name, ref.task_name, ref.output_field)


def _label_text(value: Any) -> str:
    """Render a guard value for an edge label."""
    value = plain_label(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace('"', "'").replace("|", "/")


def _guard_label(workflow: Workflow, guard: Guard) -> str:
    """Return the label of a dashed control edge for a guard."""
    switch_case = (
        workflow.is_switch(guard.ref.task_name) and guard.ref.output_field == BRANCH_FIELD
    )
    if switch_case:
        if guard.op == "otherwise":
            return "otherwise"
        if guard.op == "in":
            return " / ".join(_label_text(value) for value in guard.values)
    field = f".{guard.ref.output_field}"
    if guard.op == "truthy":
        return field if not switch_case else "true"
    if guard.op == "falsy":
        return f"not {field}" if not switch_case else "false"
    if len(guard.values) == 1:
        return f"{field} = {_label_text(guard.values[0])}"
    return f"{field} in {' / '.join(_label_text(value) for value in guard.values)}"


def _node_line(workflow: Workflow, task_name: str, node_id: str, label: str) -> str:
    """Return a node definition; switch tasks are drawn as diamonds."""
    if workflow.is_switch(task_name):
        return f'{node_id}{{"{label}"}}'
    return f'{node_id}["{label}"]'


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
            output_name = _safe_type_name(workflow.get_output_annotation(deps), upstream_cls)
            lines.append(f"{indent}{upstream_src} -->|{output_name}| {tgt_name}")
        elif isinstance(deps, tuple):
            upstream_name, field_name = deps
            upstream_src = _apply_redirect(upstream_name, source_redirect)
            label = _field_type_label(workflow, task_by_name, upstream_name, field_name)
            lines.append(f"{indent}{upstream_src} -->|{label}| {tgt_name}")
        elif isinstance(deps, dict):
            for down_field, upstream_ref in sorted(deps.items()):
                if isinstance(upstream_ref, FirstOfRef):
                    first_node = f"_first_of_{tgt_name}_{down_field}_"
                    lines.append(f'{indent}{first_node}(["first_of {down_field}"])')
                    for ref in upstream_ref.members:
                        upstream_src = _apply_redirect(ref.task_name, source_redirect)
                        label = _output_ref_label(workflow, task_by_name, ref)
                        lines.append(f"{indent}{upstream_src} -->|{label}| {first_node}")
                    input_model = get_input_type(task_cls)
                    annotation = input_model.model_fields[down_field].annotation
                    field_type = _safe_type_name(annotation, task_cls)
                    lines.append(
                        f"{indent}{first_node} -->|{down_field}: {field_type}| {tgt_name}"
                    )
                elif isinstance(upstream_ref, CollectionRef):
                    collection_node = f"_collect_{tgt_name}_{down_field}_"
                    lines.append(f'{indent}{collection_node}{{{{"collect {down_field}"}}}}')
                    for member_label, ref, mode in upstream_ref.labeled_members():
                        upstream_src = _apply_redirect(ref.task_name, source_redirect)
                        label = _output_ref_label(workflow, task_by_name, ref)
                        if mode != "item":
                            member_label = f"*{member_label}"
                        lines.append(
                            f"{indent}{upstream_src} -->|{member_label}: {label}| "
                            f"{collection_node}"
                        )
                    input_model = get_input_type(task_cls)
                    annotation = input_model.model_fields[down_field].annotation
                    collection_type = _safe_type_name(annotation, task_cls)
                    lines.append(
                        f"{indent}{collection_node} -->|{down_field}: {collection_type}| "
                        f"{tgt_name}"
                    )
                elif isinstance(upstream_ref, tuple):
                    upstream_name, up_field = upstream_ref
                    upstream_src = _apply_redirect(upstream_name, source_redirect)
                    label = _field_type_label(workflow, task_by_name, upstream_name, up_field)
                    lines.append(f"{indent}{upstream_src} -->|{down_field}: {label}| {tgt_name}")
                else:
                    upstream_src = _apply_redirect(upstream_ref, source_redirect)
                    up_cls = task_by_name[upstream_ref]
                    output_name = _safe_type_name(
                        workflow.get_output_annotation(upstream_ref), up_cls
                    )
                    lines.append(
                        f"{indent}{upstream_src} -->|{down_field}: {output_name}| {tgt_name}"
                    )

        guard = workflow.get_guard(task_name)
        if guard is not None:
            guard_src = _apply_redirect(guard.ref.task_name, source_redirect)
            lines.append(f"{indent}{guard_src} -.->|{_guard_label(workflow, guard)}| {tgt_name}")


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
    lines: list[str] = ["---", f"title: {workflow.name}", "---", "graph TD"]

    tasks = workflow.topological_order()
    task_by_name: dict[str, type] = {name: cls for name, cls in tasks}

    # Find sink tasks (no downstream data or condition dependents)
    sink_names = set(workflow._find_sinks())
    sinks = [(name, cls) for name, cls in tasks if name in sink_names]

    # Collect tasks with config_fields
    configured_tasks = {
        name
        for name, _cls in tasks
        if workflow.get_config_fields(name) or workflow.is_mapped_task(name)
    }

    # Detect workflow_task nodes and build redirect maps
    source_redirect: dict[str, str] = {}
    target_redirect: dict[str, str] = {}
    workflow_task_nodes: dict[str, Workflow] = {}

    for task_name, task_cls in tasks:
        inner_wf = getattr(task_cls, "_inner_workflow", None)
        if inner_wf is not None:
            workflow_task_nodes[task_name] = inner_wf
            # Edges into the subgraph enter at the inner task consuming the
            # wrapper's input (the same root workflow_task derives its input
            # type from); configured and mapped roots are fed by config.
            inner_roots = inner_wf.input_root_names()
            if inner_roots:
                target_redirect[task_name] = f"{task_name}__{inner_roots[0]}"
            # Result task → source redirect
            source_redirect[task_name] = f"{task_name}__{inner_wf.result_task_name}"

    # A start node represents external job input. Fully configured and
    # self-contained workflows have no such input, so avoid an orphan node.
    has_start_node = any(
        workflow.get_dependencies(name) is None and name not in configured_tasks
        for name, _cls in tasks
    )
    if has_start_node:
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
                lines.append(f"        {_node_line(inner_wf, iname, prefixed, iname)}")

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
            task_map = workflow.get_task_map(task_name)
            label = (
                f"{task_name}<br/>map over: {task_map.over}" if task_map is not None else task_name
            )
            lines.append(f"    {_node_line(workflow, task_name, task_name, label)}")

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
            cf = set(workflow.get_config_fields(task_name))
            task_map = workflow.get_task_map(task_name)
            if task_map is not None:
                cf.add(task_map.over)
            label = ", ".join(sorted(cf))
            lines.append(f"    _job_config_ -.->|{label}| {task_name}")

    # Sink tasks: edge to end, labeled with output type
    for task_name, task_cls in sinks:
        src = _apply_redirect(task_name, source_redirect)
        output_name = _safe_type_name(workflow.get_output_annotation(task_name), task_cls)
        lines.append(f"    {src} -->|{output_name}| _end_")

    return "\n".join(lines) + "\n"
