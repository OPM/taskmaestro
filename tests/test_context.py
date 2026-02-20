"""Tests for ExecutionContext."""

import logging
import tempfile
from pathlib import Path

import pytest

from workflow_runner.context import ExecutionContext


class TestExecutionContext:
    def test_default_correlation_id(self) -> None:
        ctx = ExecutionContext()
        assert ctx.correlation_id is not None
        assert len(ctx.correlation_id) > 0

    def test_custom_correlation_id(self) -> None:
        ctx = ExecutionContext(correlation_id="my-id")
        assert ctx.correlation_id == "my-id"

    def test_default_logger(self) -> None:
        ctx = ExecutionContext()
        assert ctx.logger.name == "workflow_runner"

    def test_custom_logger(self) -> None:
        logger = logging.getLogger("custom")
        ctx = ExecutionContext(logger=logger)
        assert ctx.logger is logger

    def test_default_scratch_dir(self) -> None:
        ctx = ExecutionContext(correlation_id="test-id")
        expected = Path(tempfile.gettempdir()) / "test-id"
        assert ctx.scratch_dir == expected

    def test_custom_scratch_dir(self) -> None:
        custom = Path("/my/scratch")
        ctx = ExecutionContext(scratch_dir=custom)
        assert ctx.scratch_dir == custom

    def test_register_and_resolve(self) -> None:
        ctx = ExecutionContext()
        ctx.register("db", "fake_db_connection")
        assert ctx.resolve("db") == "fake_db_connection"

    def test_resolve_missing_key(self) -> None:
        ctx = ExecutionContext()
        with pytest.raises(KeyError):
            ctx.resolve("nonexistent")

    def test_register_overwrite(self) -> None:
        ctx = ExecutionContext()
        ctx.register("key", "value1")
        ctx.register("key", "value2")
        assert ctx.resolve("key") == "value2"
