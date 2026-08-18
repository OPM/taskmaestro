"""Factory for wrapping a Workflow as a single Task."""

from __future__ import annotations

from typing import Any

from taskmaestro.context import ExecutionContext
from taskmaestro.exceptions import WorkflowDefinitionError
from taskmaestro.job import EmptyConfig, Job, JobConfiguration, JobStatus
from taskmaestro.runner import Runner
from taskmaestro.task import Task, get_input_type, get_output_type
from taskmaestro.workflow import Workflow


def workflow_task(
    workflow: Workflow,
    *,
    name: str | None = None,
    job_configuration: JobConfiguration | None = None,
) -> type[Task[Any, Any]]:
    """Create a Task subclass that wraps an entire workflow as a single task.

    The generated task's input type is derived from the inner workflow's sole
    unconfigured root task, and its output type from the inner workflow's result
    task. Prefer ``workflow.as_task()`` in application code; this function remains
    available when a factory-style API is more convenient.

    Args:
        workflow: The inner Workflow to wrap.
        name: Optional name override; defaults to workflow.name.
        job_configuration: Optional JobConfiguration for inner tasks with config_fields.

    Returns:
        A new Task subclass that runs the inner workflow with the outer task's
        ``ExecutionContext``. The inner workflow is an opaque lifecycle boundary:
        hooks on the outer runner observe the generated task, not its inner tasks.

    Raises:
        WorkflowDefinitionError: If the inner workflow does not have exactly one
            root task without config_fields (unless all roots are covered by
            job_configuration).
    """
    # Find root tasks: tasks with deps=None and no config_fields
    roots: list[tuple[str, type[Task[Any, Any]]]] = []
    for task_name, deps in workflow._dependencies.items():
        if deps is None:
            config_fields = workflow.get_config_fields(task_name)
            if not config_fields:
                roots.append((task_name, workflow._tasks[task_name]))

    all_roots_configured = False

    if len(roots) == 0:
        if job_configuration is not None:
            # All roots have config_fields covered by job_configuration.
            # The wrapper is self-contained — use EmptyConfig as input type.
            all_roots_configured = True
            input_type: type[Any] = EmptyConfig
        else:
            raise WorkflowDefinitionError(
                f"Inner workflow '{workflow.name}' has no root tasks without config_fields; "
                f"cannot derive input type for workflow_task"
            )
    elif len(roots) > 1:
        root_names = [r[0] for r in roots]
        raise WorkflowDefinitionError(
            f"Inner workflow '{workflow.name}' has multiple root tasks without "
            f"config_fields ({root_names}); workflow_task requires exactly one"
        )
    else:
        input_type = get_input_type(roots[0][1])

    result_task_cls = workflow.result_task
    output_type = get_output_type(result_task_cls)

    resolved_name = name if name is not None else workflow.name
    inner_wf = workflow
    inner_jc = job_configuration
    _all_configured = all_roots_configured

    class _WorkflowTask(Task[input_type, output_type]):  # type: ignore[valid-type]
        name = resolved_name
        _inner_workflow = inner_wf

        def run(self, input: Any, ctx: ExecutionContext) -> Any:
            cfg = EmptyConfig() if _all_configured else input
            job = Job(workflow=inner_wf, config=cfg, job_configuration=inner_jc)
            result_job = Runner().run(job, ctx=ctx)
            if result_job.status == JobStatus.FAILED:
                raise RuntimeError(
                    f"Inner workflow '{inner_wf.name}' failed at task "
                    f"'{result_job.failed_task}': {result_job.error}"
                )
            return result_job.result

    _WorkflowTask.__name__ = f"WorkflowTask_{resolved_name}"
    _WorkflowTask.__qualname__ = f"WorkflowTask_{resolved_name}"

    return _WorkflowTask
