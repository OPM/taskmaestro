"""Configuration for expanding one task over a configured mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, RootModel

O = TypeVar("O", bound=BaseModel)


class MappedOutput(RootModel[dict[str, O]], Generic[O]):
    """Typed aggregate output produced by a mapped workflow task."""


@dataclass(frozen=True)
class TaskMap:
    """Describe how mapping keys and values populate task input fields."""

    over: str
    key_as: str
    value_as: str
    error_mode: Literal["fail_fast", "collect_all"] = "fail_fast"

    def __post_init__(self) -> None:
        for field_name, value in (
            ("over", self.over),
            ("key_as", self.key_as),
            ("value_as", self.value_as),
        ):
            if not value:
                raise ValueError(f"TaskMap.{field_name} must be a non-empty string")
        if self.key_as == self.value_as:
            raise ValueError("TaskMap.key_as and TaskMap.value_as must be different")
        if self.error_mode not in ("fail_fast", "collect_all"):
            raise ValueError("TaskMap.error_mode must be 'fail_fast' or 'collect_all'")
