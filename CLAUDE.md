# CLAUDE.md

Typed DAG task workflow library with Pydantic models, lifecycle hooks, and fail-fast semantics.

## Commands

```bash
source .venv/bin/activate          # activate venv
pip install -e ".[dev]"            # install with dev deps
pytest -v                          # run tests
ruff check taskekrabbe/ tests/ # lint
ruff format taskekrabbe/ tests/ # format
mypy taskekrabbe               # type check (strict mode)
```

## Architecture

| File | Responsibility |
|---|---|
| `taskekrabbe/exceptions.py` | Exception hierarchy (no internal deps) |
| `taskekrabbe/context.py` | `ExecutionContext` with correlation ID, logger, scratch dir, service registry |
| `taskekrabbe/task.py` | `Task[I, O]` ABC, type introspection (`get_input_type`, `get_output_type`) |
| `taskekrabbe/workflow.py` | `Workflow` (linear + DAG), `WorkflowBuilder`, validation (cycles, types, fan-in) |
| `taskekrabbe/job.py` | `Job[C]`, `JobStatus`, `TaskStatus`, `TaskResult` dataclass |
| `taskekrabbe/runner.py` | `Runner` — topological execution, timeout via `signal.alarm`, hook dispatch |
| `taskekrabbe/hooks/base.py` | `Event` StrEnum, `Hook` protocol, `BaseHook` no-op base |
| `taskekrabbe/hooks/logging.py` | `LoggingHook` — logs events via `logging` module |
| `taskekrabbe/hooks/timing.py` | `TimingHook` — records durations via `time.monotonic()` |
| `taskekrabbe/hooks/persistence.py` | `ResultPersistenceHook` — writes `{task_name}.json` per task |

## Key Patterns

- **Type introspection**: Walk MRO via `__orig_bases__` + `typing.get_args()` to extract concrete `I`/`O` types
- **Fan-in**: Downstream task input model fields mapped to upstream outputs via `model_fields` (Pydantic v2)
- **Timeouts**: `signal.alarm` (Unix only, main thread); gracefully warns if unavailable
- **Hook error swallowing**: `_emit()` wraps each hook call in try/except, reports via `warnings.warn()`
- **Validation order**: unique names → acyclic (DFS) → type chain → result task detection

## Testing Conventions

- Shared fixtures and reusable tasks/models in `tests/conftest.py`
- Tests organized by module: `test_exceptions`, `test_context`, `test_task`, `test_workflow`, `test_job`, `test_runner`, `test_hooks`
- Timeout tests skip on non-Unix (no `signal.SIGALRM`)
- Use `RecordingHook` pattern to assert event sequences
