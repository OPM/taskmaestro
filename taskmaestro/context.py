"""Execution context passed to every task during execution."""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any


class ExecutionContext:
    """Cross-cutting state passed to every task during execution.

    Carries a correlation ID, logger, scratch directory, and a simple
    service registry for injecting dependencies (DB connections, HTTP
    clients, etc.) into tasks.
    """

    def __init__(
        self,
        correlation_id: str | None = None,
        logger: logging.Logger | None = None,
        scratch_dir: Path | None = None,
    ) -> None:
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = logger or logging.getLogger("taskmaestro")
        self.scratch_dir = scratch_dir or Path(tempfile.gettempdir()) / self.correlation_id
        self._registry: dict[str, Any] = {}

    def register(self, key: str, service: Any) -> None:
        """Register a service by key."""
        self._registry[key] = service

    def resolve(self, key: str) -> Any:
        """Retrieve a registered service. Raises KeyError if not found."""
        return self._registry[key]
