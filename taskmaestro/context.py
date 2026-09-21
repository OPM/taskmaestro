"""Execution context passed to every task during execution."""

from __future__ import annotations

import hashlib
import logging
import re
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
        *,
        parent_correlation_id: str | None = None,
    ) -> None:
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.parent_correlation_id = parent_correlation_id
        self.logger = logger or logging.getLogger("taskmaestro")
        self.scratch_dir = scratch_dir or Path(tempfile.gettempdir()) / self.correlation_id
        self._registry: dict[str, Any] = {}

    def register(self, key: str, service: Any) -> None:
        """Register a service by key."""
        self._registry[key] = service

    def resolve(self, key: str) -> Any:
        """Retrieve a registered service. Raises KeyError if not found."""
        return self._registry[key]

    def child(self, *, task_name: str, item_key: str) -> ExecutionContext:
        """Create a mapped-item context sharing this context's services."""
        raw_suffix = f"{task_name}:{item_key}"
        safe_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_suffix).strip("_") or "item"
        digest = hashlib.sha256(raw_suffix.encode()).hexdigest()[:8]
        child_id = f"{self.correlation_id}:{safe_suffix}:{digest}"
        child = ExecutionContext(
            correlation_id=child_id,
            logger=self.logger,
            scratch_dir=self.scratch_dir / f"{safe_suffix}-{digest}",
            parent_correlation_id=self.correlation_id,
        )
        child._registry = self._registry
        return child
