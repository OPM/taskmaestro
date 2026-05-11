"""Generic base model for wrapping arbitrary (non-Pydantic) objects."""

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class ObjectModel(BaseModel, Generic[T]):
    """Base model for wrapping arbitrary (non-Pydantic) objects.

    Enables arbitrary_types_allowed so fields can hold native library
    objects like database connections, API clients, etc.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: T
