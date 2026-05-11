"""Built-in hooks for the workflow runner."""

from taskmaestro.hooks.base import BaseHook, Event, Hook
from taskmaestro.hooks.logging import LoggingHook
from taskmaestro.hooks.persistence import ResultPersistenceHook
from taskmaestro.hooks.timing import TimingHook

__all__ = [
    "BaseHook",
    "Event",
    "Hook",
    "LoggingHook",
    "ResultPersistenceHook",
    "TimingHook",
]
