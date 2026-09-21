"""Result persistence hook that writes task outputs to JSON files."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel

from taskmaestro.hooks.base import BaseHook
from taskmaestro.job import Job
from taskmaestro.task import Task


class ResultPersistenceHook(BaseHook):
    """Writes {task_name}.json per completed task to an output directory.

    Task names and mapped-item keys are percent-encoded so that a name such
    as ``../evil`` or ``a/b`` can never escape ``output_dir``.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def on_task_complete(self, job: Job[Any], task: Task[Any, Any], output: BaseModel) -> None:
        self._write(f"{_safe(task.name)}.json", output)

    def on_map_item_complete(
        self, job: Job[Any], task: Task[Any, Any], key: str, output: BaseModel
    ) -> None:
        self._write(f"{_safe(task.name)}[{_safe(key)}].json", output)

    def _write(self, filename: str, output: BaseModel) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / filename).write_text(output.model_dump_json(indent=2))


def _safe(component: str) -> str:
    """Return a single, traversal-free filename component.

    Escapes ``%`` too, so distinct inputs cannot collapse onto the same name.
    """
    return quote(component, safe="")
