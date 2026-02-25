"""Built-in hooks for the workflow runner."""

from taskekrabbe.hooks.base import BaseHook, Event, Hook
from taskekrabbe.hooks.logging import LoggingHook
from taskekrabbe.hooks.persistence import ResultPersistenceHook
from taskekrabbe.hooks.timing import TimingHook

__all__ = [
    "BaseHook",
    "Event",
    "Hook",
    "LoggingHook",
    "ResultPersistenceHook",
    "TimingHook",
]
