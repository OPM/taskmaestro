"""Result persistence hook that writes task outputs to JSON files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from taskmaestro.hooks.base import BaseHook
from taskmaestro.job import Job
from taskmaestro.task import Task


class ResultPersistenceHook(BaseHook):
    """Writes {task_name}.json per completed task to an output directory."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def on_task_complete(self, job: Job[Any], task: Task[Any, Any], output: BaseModel) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{task.name}.json"
        output_path.write_text(output.model_dump_json(indent=2))
