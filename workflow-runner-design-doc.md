# Workflow Runner — Design Document

## 1. Overview

A Python 3.12+ library for defining and executing task workflows — both sequential pipelines and DAG (directed acyclic graph) workflows — with strongly-typed inputs/outputs, class-based task definitions, and a lifecycle event hook system. Designed for single-machine execution with a synchronous runner.

### Design Principles

- **Type safety first**: Task inputs and outputs are Pydantic models, validated at boundaries.
- **Fail fast**: A task failure immediately aborts the job — no retries, no partial recovery.
- **Code-first**: Workflows and configurations are composed in Python, not config files.
- **Observable**: A hook system provides visibility into every lifecycle transition.

---

## 2. Domain Model

| Concept | Description |
|---|---|
| **Task** | A single unit of work. Declares typed `Input` and `Output` models. Implemented as a class inheriting from `Task[I, O]`. |
| **Workflow** | A directed acyclic graph (DAG) of tasks. Linear pipelines are a special case where each task depends on the previous. Fan-in is supported via Pydantic field mapping. |
| **JobConfig** | A Pydantic model specifying the initial input to a workflow. Validated before execution begins. |
| **Job** | A workflow bound to a specific `JobConfig`, ready to execute. Tracks status and results. |

### Relationships

```
Linear:  JobConfig ──► Job ──► Workflow ──► Task₁ → Task₂ → ... → Taskₙ
                         │
                         └── status, result

DAG:     JobConfig ──► Job ──► Workflow ──► TaskA ──► TaskC
                         │              └──► TaskB ──┘
                         └── status, result
```

---

## 3. Task Design

### 3.1 Base Class

Tasks inherit from a generic base class parameterized by input and output types:

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Generic, TypeVar

I = TypeVar("I", bound=BaseModel)
O = TypeVar("O", bound=BaseModel)

class Task(ABC, Generic[I, O]):
    """Base class for all tasks."""

    name: str  # Human-readable name, defaults to class name
    timeout_seconds: float | None = None  # Optional per-task timeout

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "name" not in cls.__dict__:
            cls.name = cls.__name__

    @abstractmethod
    def run(self, input: I, ctx: ExecutionContext) -> O:
        """Execute the task. Receives validated input and context, returns validated output."""
        ...
```

### 3.2 Example Tasks

```python
class FileContent(BaseModel):
    path: str
    text: str

class ReadFile(Task[JobInput, FileContent]):
    name = "read_file"

    def run(self, input: JobInput, ctx: ExecutionContext) -> FileContent:
        ctx.logger.info("Reading file %s", input.file_path)
        text = Path(input.file_path).read_text()
        return FileContent(path=input.file_path, text=text)

class WordCount(BaseModel):
    path: str
    count: int

class CountWords(Task[FileContent, WordCount]):
    name = "count_words"

    def run(self, input: FileContent, ctx: ExecutionContext) -> WordCount:
        return WordCount(path=input.path, count=len(input.text.split()))
```

### 3.3 Type Chain Validation

The workflow validates at **definition time** (not runtime) that the output type of task N matches the input type of task N+1. This is done by inspecting the generic type arguments on each task class.

```python
# This should raise a WorkflowDefinitionError at construction time:
Workflow(name="bad", tasks=[ReadFile, ReadFile])  # FileContent != JobInput
```

---

## 4. Workflow

A `Workflow` is a DAG of tasks. Linear pipelines are a convenient special case.

### 4.1 Construction Modes

**Linear shorthand** (backward compatible) — an ordered list auto-chains each task to the next:

```python
# ReadCsv (root) → Summarize (depends on ReadCsv)
workflow = Workflow(name="csv_summary", tasks=[ReadCsv, Summarize])
```

**Builder for DAGs** — explicit dependency declarations supporting fan-in and fan-out:

```python
workflow = (
    Workflow.builder(name="enrich_user")
    .add_task(FetchUser)                                                   # root
    .add_task(FetchOrders)                                                 # root
    .add_task(EnrichUser, depends_on={"user": FetchUser, "orders": FetchOrders})
    .build()
)
```

### 4.2 Dependency Declaration

`add_task(task_cls, depends_on=...)` accepts three forms:

| Form | Meaning | Type validation |
|---|---|---|
| No `depends_on` | Root task — receives `job.config` | `input_type(task) == config type` (validated at `Job` creation) |
| `depends_on=TaskA` | Single dependency — receives TaskA's output directly | `output_type(TaskA) == input_type(task)` |
| `depends_on={"field": TaskA, ...}` | Fan-in — each field of the input model is sourced from a different upstream | Each `output_type(upstream) == field_annotation` |

Fan-out happens implicitly: multiple tasks can declare `depends_on` pointing to the same upstream.

### 4.3 Definition

```python
class Workflow:
    """A DAG of tasks. Linear pipelines are a special case."""

    def __init__(
        self,
        name: str,
        tasks: list[type[Task]] | None = None,
        *,
        result_task: type[Task] | None = None,
    ):
        """Linear shorthand: auto-chain tasks[0] → tasks[1] → ... → tasks[N].

        For DAG construction, use Workflow.builder() instead.
        """
        self.name = name
        self._tasks: dict[str, type[Task]] = {}
        self._dependencies: dict[str, dict[str, str] | str | None] = {}
        self._result_task_name: str | None = None

        if tasks:
            # Linear shorthand: auto-generate dependency chain
            for i, task_cls in enumerate(tasks):
                self._tasks[task_cls.name] = task_cls
                if i == 0:
                    self._dependencies[task_cls.name] = None  # root
                else:
                    prev = tasks[i - 1]
                    self._dependencies[task_cls.name] = prev.name  # single dep
            self._result_task_name = tasks[-1].name
            self._validate()

        if result_task is not None:
            self._result_task_name = result_task.name

    @classmethod
    def builder(cls, name: str, *, result_task: type[Task] | None = None) -> WorkflowBuilder:
        """Return a builder for DAG construction."""
        return WorkflowBuilder(name, result_task=result_task)

    # --- Query methods ---

    def topological_order(self) -> list[type[Task]]:
        """Return task classes in a valid execution order (Kahn's algorithm)."""
        ...

    @property
    def result_task(self) -> type[Task]:
        """The task whose output becomes job.result."""
        return self._tasks[self._result_task_name]

    def get_dependencies(self, task_name: str) -> dict[str, str] | str | None:
        """Return the dependency spec for a task."""
        return self._dependencies[task_name]

    # --- Validation ---

    def _validate(self) -> None:
        self._validate_unique_names()
        self._validate_acyclic()
        self._validate_types()
        self._validate_result_task()

    def _validate_unique_names(self) -> None:
        """Raise WorkflowDefinitionError on duplicate task names."""
        ...

    def _validate_acyclic(self) -> None:
        """Detect cycles via DFS. Raise CycleDetectedError."""
        ...

    def _validate_types(self) -> None:
        """Validate type compatibility for all edges.

        - Single dep: output_type(upstream) == input_type(downstream)
        - Fan-in dict: for each {field: upstream}, output_type(upstream)
          must match the Pydantic field annotation on the downstream input model.
          All required fields must be covered (IncompleteInputError if not).
        """
        ...

    def _validate_result_task(self) -> None:
        """Ensure result_task is set. Default to sole sink; raise if ambiguous."""
        sinks = self._find_sinks()
        if self._result_task_name is None:
            if len(sinks) == 1:
                self._result_task_name = sinks[0]
            else:
                raise WorkflowDefinitionError(
                    f"Workflow '{self.name}' has {len(sinks)} sink tasks "
                    f"({sinks}); specify result_task explicitly"
                )

    def _find_sinks(self) -> list[str]:
        """Return task names with no downstream dependents."""
        has_dependents: set[str] = set()
        for deps in self._dependencies.values():
            if isinstance(deps, str):
                has_dependents.add(deps)
            elif isinstance(deps, dict):
                has_dependents.update(deps.values())
        return [name for name in self._tasks if name not in has_dependents]


class WorkflowBuilder:
    """Fluent builder for DAG workflows."""

    def __init__(self, name: str, *, result_task: type[Task] | None = None):
        self._workflow = Workflow.__new__(Workflow)
        self._workflow.name = name
        self._workflow._tasks = {}
        self._workflow._dependencies = {}
        self._workflow._result_task_name = (
            result_task.name if result_task else None
        )

    def add_task(
        self,
        task_cls: type[Task],
        *,
        depends_on: type[Task] | dict[str, type[Task]] | None = None,
    ) -> WorkflowBuilder:
        """Add a task to the DAG. Returns self for chaining."""
        wf = self._workflow
        if task_cls.name in wf._tasks:
            raise WorkflowDefinitionError(
                f"Duplicate task name '{task_cls.name}'"
            )
        wf._tasks[task_cls.name] = task_cls

        if depends_on is None:
            wf._dependencies[task_cls.name] = None
        elif isinstance(depends_on, dict):
            wf._dependencies[task_cls.name] = {
                field: dep.name for field, dep in depends_on.items()
            }
        else:
            wf._dependencies[task_cls.name] = depends_on.name

        return self

    def build(self) -> Workflow:
        """Finalize and validate the workflow. Returns an immutable Workflow."""
        self._workflow._validate()
        return self._workflow
```

### 4.4 Type Introspection

Use `typing.get_args()` on each task's `__orig_bases__` to extract the concrete `Input` and `Output` types. For linear chains, the validation checks `output[i] == input[i+1]`. For fan-in edges, the validation checks each upstream's output type against the corresponding Pydantic field annotation on the downstream input model using `model_fields`.

---

## 5. Job Configuration

Job configs are Pydantic models, giving you validation, serialization, and schema generation for free:

```python
class MyJobConfig(BaseModel):
    file_path: str
    encoding: str = "utf-8"
    max_lines: int | None = None
```

The job config type must match the input type of **all root tasks** (tasks with no upstream dependencies). For linear pipelines this is the first task; for DAGs it may be multiple tasks. This is validated when creating a `Job`.

---

## 6. Task Context

Every task receives an `ExecutionContext` alongside its input. The context carries cross-cutting concerns so that tasks don't need to construct their own loggers, manage correlation IDs, or look up shared services.

```python
import logging
import uuid
from pathlib import Path
from typing import Any

class ExecutionContext:
    """Cross-cutting state passed to every task during execution."""

    def __init__(
        self,
        correlation_id: str | None = None,
        logger: logging.Logger | None = None,
        scratch_dir: Path | None = None,
    ):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = logger or logging.getLogger("workflow_runner")
        self.scratch_dir = scratch_dir or Path("/tmp") / self.correlation_id
        self._registry: dict[str, Any] = {}

    # --- Dependency registry ---

    def register(self, key: str, service: Any) -> None:
        """Register a service (DB connection, HTTP client, etc.) by key."""
        self._registry[key] = service

    def resolve(self, key: str) -> Any:
        """Retrieve a registered service. Raises KeyError if not found."""
        return self._registry[key]
```

**Usage pattern**: The runner creates the context once per job and passes it to every `task.run()` call. Users can register dependencies before calling `runner.run()`:

```python
ctx = ExecutionContext()
ctx.register("db", my_database_connection)
ctx.register("http", httpx.Client())
result = runner.run(job, ctx=ctx)
```

Tasks access services via the context:

```python
def run(self, input: MyInput, ctx: ExecutionContext) -> MyOutput:
    db = ctx.resolve("db")
    ...
```

---

## 7. Job

A `Job` binds a workflow to a configuration and tracks execution state:

```python
from enum import StrEnum
from datetime import datetime

class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

C = TypeVar("C", bound=BaseModel)

class Job(Generic[C]):
    """A workflow bound to a specific config, ready to execute.

    Generic over C so that `job.config` retains its concrete type.
    """
    def __init__(self, workflow: Workflow, config: C):
        self.workflow = workflow
        self.config: C = config
        self.status: JobStatus = JobStatus.PENDING
        self.result: BaseModel | None = None
        self.error: str | None = None
        self.failed_task: str | None = None
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.task_results: list[TaskResult] = []  # ordered by execution (topological)

class TaskStatus(StrEnum):
    """Status of an individual task execution (distinct from JobStatus)."""
    COMPLETED = "completed"
    FAILED = "failed"

class TaskResult:
    """Record of a single task's execution within a job."""
    task_name: str
    status: TaskStatus
    output: BaseModel | None
    started_at: datetime
    duration_seconds: float
    error: str | None = None  # Serializable error message, not Exception
```

---

## 8. Runner

The runner is the execution engine. It handles both linear and DAG workflows using a single algorithm: iterate tasks in topological order, assembling each task's input from the outputs of its upstream dependencies.

```python
class Runner:
    def __init__(self, hooks: list[Hook] | None = None):
        self.hooks = hooks or []

    def run(
        self,
        job: Job,
        ctx: ExecutionContext | None = None,
        timeout_seconds: float | None = None,
    ) -> Job:
        """Execute all tasks in topological order. Stops on first failure.

        Args:
            job: The job to execute. Must be in PENDING status.
            ctx: Execution context passed to each task. Created automatically
                 if not supplied.
            timeout_seconds: Optional wall-clock deadline for the entire job.
        """
        if job.status != JobStatus.PENDING:
            raise JobStateError(
                f"Cannot run job with status '{job.status}'; expected 'pending'"
            )

        ctx = ctx or ExecutionContext()
        workflow = job.workflow

        self._emit(Event.JOB_START, job)
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()

        outputs: dict[str, BaseModel] = {}  # task_name -> output

        for task_cls in workflow.topological_order():
            task = task_cls()
            deps = workflow.get_dependencies(task.name)

            # Assemble input based on dependency type
            if deps is None:
                # Root task: receives job config
                task_input = job.config
            elif isinstance(deps, str):
                # Single dependency: receives upstream output directly
                task_input = outputs[deps]
            elif isinstance(deps, dict):
                # Fan-in: construct input model from upstream outputs
                input_type = _get_input_type(task_cls)
                field_values = {
                    field: outputs[upstream_name]
                    for field, upstream_name in deps.items()
                }
                task_input = input_type(**field_values)

            task_started = datetime.now()
            self._emit(Event.TASK_START, job, task)
            try:
                output = task.run(task_input, ctx)

                # Validate output matches declared type
                expected_output_type = _get_output_type(task_cls)
                if not isinstance(output, expected_output_type):
                    raise TaskOutputTypeError(
                        f"Task '{task.name}' returned {type(output).__name__}, "
                        f"expected {expected_output_type.__name__}"
                    )

                duration = (datetime.now() - task_started).total_seconds()
                outputs[task.name] = output
                job.task_results.append(TaskResult(
                    task_name=task.name,
                    status=TaskStatus.COMPLETED,
                    output=output,
                    started_at=task_started,
                    duration_seconds=duration,
                ))
                self._emit(Event.TASK_COMPLETE, job, task, output)
            except Exception as exc:
                duration = (datetime.now() - task_started).total_seconds()
                job.status = JobStatus.FAILED
                job.error = str(exc)
                job.failed_task = task.name
                job.completed_at = datetime.now()
                job.task_results.append(TaskResult(
                    task_name=task.name,
                    status=TaskStatus.FAILED,
                    output=None,
                    started_at=task_started,
                    duration_seconds=duration,
                    error=str(exc),
                ))
                self._emit(Event.TASK_FAIL, job, task, exc)
                self._emit(Event.JOB_FAIL, job)
                return job

        job.status = JobStatus.COMPLETED
        job.result = outputs[workflow.result_task.name]
        job.completed_at = datetime.now()
        self._emit(Event.JOB_COMPLETE, job)
        return job

    def _emit(self, event: Event, *args: object) -> None:
        """Dispatch event to all hooks, swallowing any hook errors."""
        for hook in self.hooks:
            handler = getattr(hook, f"on_{event}", None)
            if handler is not None:
                try:
                    handler(*args)
                except Exception:
                    import warnings
                    warnings.warn(
                        f"Hook {type(hook).__name__} raised during {event}",
                        stacklevel=2,
                    )
```

### 8.1 Execution Flow

```
Runner.run(job, ctx?, timeout_seconds?)
  │
  ├── guard: job.status must be PENDING
  ├── create ExecutionContext (if not supplied)
  ├── emit JOB_START
  ├── for each task in workflow.topological_order():
  │     ├── assemble input:
  │     │     ├── root task → job.config
  │     │     ├── single dep → outputs[upstream]
  │     │     └── fan-in dict → InputModel(**{field: outputs[upstream], ...})
  │     ├── emit TASK_START
  │     ├── task.run(input, ctx) -> output
  │     ├── validate output type (isinstance check)
  │     ├── store in outputs[task.name]
  │     ├── record TaskResult
  │     └── emit TASK_COMPLETE
  │
  ├── on success: job.result = outputs[result_task], emit JOB_COMPLETE
  └── on failure: set completed_at, emit TASK_FAIL, JOB_FAIL, return immediately
```

### 8.2 Timeouts

Timeouts provide a safety net against tasks that hang indefinitely.

- **Per-task timeout**: Set `timeout_seconds` as an optional class attribute on the task (default `None` = no timeout).
- **Per-job timeout**: Passed as a parameter to `Runner.run(job, timeout_seconds=...)`.

The runner enforces timeouts by wrapping `task.run()` with `signal.alarm` (Unix) or a threading deadline. On timeout, the runner raises a `TaskTimeoutError` (subclass of `TaskExecutionError`) and follows the standard failure path.

> **Note**: Because the v1 runner is synchronous and single-threaded, `signal.alarm` is the preferred mechanism on Unix. For cross-platform support, a future version may use `concurrent.futures` with a thread-pool timeout.

```python
class TaskTimeoutError(TaskExecutionError):
    """Raised when a task exceeds its timeout_seconds."""
    pass
```

---

## 9. Event Hook System

Hooks fire in topological execution order — for linear pipelines this is the sequential task order; for DAGs, tasks fire as they execute in their topological order.

### 9.1 Events

```python
from enum import StrEnum

class Event(StrEnum):
    JOB_START = "job_start"
    JOB_COMPLETE = "job_complete"
    JOB_FAIL = "job_fail"
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    TASK_FAIL = "task_fail"
```

### 9.2 Hook Protocol

Hooks implement a simple protocol. Each method is optional (default no-op):

```python
from typing import Protocol

class Hook(Protocol):
    def on_job_start(self, job: Job) -> None: ...
    def on_job_complete(self, job: Job) -> None: ...
    def on_job_fail(self, job: Job) -> None: ...
    def on_task_start(self, job: Job, task: Task) -> None: ...
    def on_task_complete(self, job: Job, task: Task, output: BaseModel) -> None: ...
    def on_task_fail(self, job: Job, task: Task, error: Exception) -> None: ...
```

### 9.3 Built-in Hooks

| Hook | Purpose |
|---|---|
| `LoggingHook` | Logs all events via Python `logging` at configurable levels. |
| `TimingHook` | Records wall-clock duration per task and total job time. |
| `ResultPersistenceHook` | Optionally serializes each task's output to disk (JSON). |

### 9.4 Custom Hooks

Users implement the `Hook` protocol (or inherit from a `BaseHook` with no-op defaults):

```python
class SlackNotificationHook(BaseHook):
    def on_job_fail(self, job: Job) -> None:
        send_slack_message(f"Job {job.workflow.name} failed at {job.failed_task}")
```

### 9.5 Hook Error Handling

Hook exceptions **never** abort the pipeline. The runner's `_emit` method wraps every hook call in a `try/except`:

- The exception is caught and reported via `warnings.warn()` (or a fallback logger).
- Execution continues with the next hook and the next pipeline step.
- This ensures that a broken monitoring hook cannot take down a production job.

```python
def _emit(self, event: Event, *args: object) -> None:
    for hook in self.hooks:
        handler = getattr(hook, f"on_{event}", None)
        if handler is not None:
            try:
                handler(*args)
            except Exception:
                import warnings
                warnings.warn(
                    f"Hook {type(hook).__name__} raised during {event}",
                    stacklevel=2,
                )
```

---

## 10. Error Handling

The strategy is **fail fast**: the first unhandled exception in a task aborts the entire job. In a DAG, this means that if any task fails, no further tasks execute — even independent branches that could theoretically continue.

- The runner catches the exception, records it as a string on the `Job` (for serialization), and emits `TASK_FAIL` + `JOB_FAIL` events.
- Task authors may handle recoverable errors internally (e.g., retry an HTTP call), but unhandled exceptions propagate.
- Pydantic `ValidationError` on task output is treated the same as any other failure — the type contract was violated.
- `TaskOutputTypeError` is raised when `task.run()` returns a value whose type doesn't match the declared output type.
- `TaskTimeoutError` is raised when a task exceeds its `timeout_seconds`.
- `JobStateError` is raised when `Runner.run()` is called on a job that is not in `PENDING` status.
- `CycleDetectedError` is raised at workflow construction if the dependency graph contains a cycle.
- `IncompleteInputError` is raised at workflow construction if a fan-in task has required input fields not mapped to any upstream task.

### Exception Hierarchy

```
WorkflowRunnerError (base)
├── WorkflowDefinitionError       # Raised at workflow construction time
│   ├── CycleDetectedError        # Dependency graph contains a cycle
│   └── IncompleteInputError      # Fan-in input has unmapped required fields
├── JobStateError                  # Job not in expected state (e.g., re-run)
└── TaskExecutionError             # Raised during task execution
    ├── TaskOutputTypeError        # Output type mismatch
    └── TaskTimeoutError           # Task exceeded timeout
```

---

## 11. Project Structure

```
workflow_runner/
├── __init__.py
├── task.py            # Task base class, I/O type introspection
├── context.py         # ExecutionContext, dependency registry
├── workflow.py        # Workflow (DAG-based, with linear shorthand), WorkflowBuilder
├── job.py             # Job, JobStatus, TaskStatus, TaskResult
├── runner.py          # Runner (handles linear and DAG workflows)
├── hooks/
│   ├── __init__.py
│   ├── base.py        # Hook protocol, BaseHook
│   ├── logging.py     # LoggingHook
│   ├── timing.py      # TimingHook
│   └── persistence.py # ResultPersistenceHook
└── exceptions.py      # WorkflowRunnerError, CycleDetectedError, IncompleteInputError, etc.
```

---

## 12. End-to-End Examples

### 12.1 Linear Pipeline

```python
from pydantic import BaseModel
from workflow_runner import Task, Workflow, Job, Runner, ExecutionContext
from workflow_runner.hooks import LoggingHook, TimingHook

# --- Models ---
class CsvInput(BaseModel):
    file_path: str
    delimiter: str = ","

class ParsedData(BaseModel):
    headers: list[str]
    rows: list[list[str]]

class Summary(BaseModel):
    row_count: int
    column_count: int
    headers: list[str]

# --- Tasks ---
class ReadCsv(Task[CsvInput, ParsedData]):
    name = "read_csv"

    def run(self, input: CsvInput, ctx: ExecutionContext) -> ParsedData:
        ctx.logger.info("Parsing CSV: %s", input.file_path)
        import csv
        with open(input.file_path) as f:
            reader = csv.reader(f, delimiter=input.delimiter)
            headers = next(reader)
            rows = list(reader)
        return ParsedData(headers=headers, rows=rows)

class Summarize(Task[ParsedData, Summary]):
    name = "summarize"

    def run(self, input: ParsedData, ctx: ExecutionContext) -> Summary:
        return Summary(
            row_count=len(input.rows),
            column_count=len(input.headers),
            headers=input.headers,
        )

# --- Compose & Run (linear shorthand) ---
workflow = Workflow(name="csv_summary", tasks=[ReadCsv, Summarize])

config = CsvInput(file_path="data.csv")
job = Job(workflow=workflow, config=config)

ctx = ExecutionContext()
runner = Runner(hooks=[LoggingHook(), TimingHook()])
result = runner.run(job, ctx=ctx)

print(result.status)   # "completed"
print(result.result)   # Summary(row_count=100, column_count=5, headers=[...])
```

### 12.2 DAG Workflow (Fan-In)

```python
from pydantic import BaseModel
from workflow_runner import Task, Workflow, Job, Runner, ExecutionContext
from workflow_runner.hooks import LoggingHook

# --- Models ---
class UserQuery(BaseModel):
    user_id: str

class UserProfile(BaseModel):
    user_id: str
    name: str
    email: str

class OrderHistory(BaseModel):
    user_id: str
    orders: list[dict]

class EnrichmentInput(BaseModel):
    """Fan-in: fields sourced from different upstream tasks."""
    user: UserProfile
    orders: OrderHistory

class EnrichedUser(BaseModel):
    user_id: str
    name: str
    email: str
    order_count: int
    total_spent: float

# --- Tasks ---
class FetchUser(Task[UserQuery, UserProfile]):
    name = "fetch_user"

    def run(self, input: UserQuery, ctx: ExecutionContext) -> UserProfile:
        db = ctx.resolve("db")
        return db.get_user(input.user_id)

class FetchOrders(Task[UserQuery, OrderHistory]):
    name = "fetch_orders"

    def run(self, input: UserQuery, ctx: ExecutionContext) -> OrderHistory:
        db = ctx.resolve("db")
        return db.get_orders(input.user_id)

class EnrichUser(Task[EnrichmentInput, EnrichedUser]):
    name = "enrich_user"

    def run(self, input: EnrichmentInput, ctx: ExecutionContext) -> EnrichedUser:
        return EnrichedUser(
            user_id=input.user.user_id,
            name=input.user.name,
            email=input.user.email,
            order_count=len(input.orders.orders),
            total_spent=sum(o["amount"] for o in input.orders.orders),
        )

# --- Compose DAG & Run ---
workflow = (
    Workflow.builder(name="enrich_user_pipeline")
    .add_task(FetchUser)       # root: input = UserQuery (job config)
    .add_task(FetchOrders)     # root: input = UserQuery (job config)
    .add_task(EnrichUser, depends_on={
        "user": FetchUser,
        "orders": FetchOrders,
    })
    .build()
)

config = UserQuery(user_id="u-123")
job = Job(workflow=workflow, config=config)

ctx = ExecutionContext()
ctx.register("db", my_database)

runner = Runner(hooks=[LoggingHook()])
result = runner.run(job, ctx=ctx)

print(result.status)   # "completed"
print(result.result)   # EnrichedUser(user_id="u-123", name=..., order_count=5, ...)
```

---

## 13. Testing

### 13.1 Unit Testing Tasks

Tasks are plain classes with a single `run()` method, making them straightforward to test in isolation:

```python
def test_count_words():
    task = CountWords()
    ctx = ExecutionContext()  # Lightweight; no external services needed
    result = task.run(FileContent(path="test.txt", text="hello world"), ctx)
    assert result.count == 2
```

- Construct the task directly, pass a minimal `ExecutionContext`, and assert on the output.
- For tasks that use `ctx.resolve()`, register test doubles before calling `run()`:

```python
def test_task_with_db():
    ctx = ExecutionContext()
    ctx.register("db", FakeDatabase())
    result = MyTask().run(input_data, ctx)
    ...
```

### 13.2 Integration Testing Workflows

Test a full pipeline by composing real (or mock) tasks into a workflow and running it through the `Runner`:

```python
def test_csv_pipeline():
    workflow = Workflow(name="test", tasks=[ReadCsv, Summarize])
    config = CsvInput(file_path="fixtures/sample.csv")
    job = Job(workflow=workflow, config=config)
    result = Runner().run(job)
    assert result.status == JobStatus.COMPLETED
    assert result.result.row_count == 3
```

### 13.3 Testing with Mock Tasks

Replace expensive tasks with lightweight stubs to keep integration tests fast:

```python
class StubReadCsv(Task[CsvInput, ParsedData]):
    name = "read_csv"

    def run(self, input: CsvInput, ctx: ExecutionContext) -> ParsedData:
        return ParsedData(headers=["a", "b"], rows=[["1", "2"]])

workflow = Workflow(name="test", tasks=[StubReadCsv, Summarize])
```

### 13.4 Testing Hooks

Verify hooks are called correctly by using a recording hook:

```python
class RecordingHook(BaseHook):
    def __init__(self):
        self.events: list[str] = []

    def on_task_start(self, job, task):
        self.events.append(f"start:{task.name}")

    def on_task_complete(self, job, task, output):
        self.events.append(f"complete:{task.name}")

hook = RecordingHook()
Runner(hooks=[hook]).run(job)
assert hook.events == ["start:read_csv", "complete:read_csv", "start:summarize", "complete:summarize"]
```

### 13.5 Testing DAG Workflows

Test fan-in workflows by constructing the DAG and verifying the merged input is assembled correctly:

```python
def test_fan_in_workflow():
    workflow = (
        Workflow.builder(name="test_fan_in")
        .add_task(FetchUser)
        .add_task(FetchOrders)
        .add_task(EnrichUser, depends_on={"user": FetchUser, "orders": FetchOrders})
        .build()
    )
    config = UserQuery(user_id="u-1")
    job = Job(workflow=workflow, config=config)

    ctx = ExecutionContext()
    ctx.register("db", FakeDatabase())
    result = Runner().run(job, ctx=ctx)

    assert result.status == JobStatus.COMPLETED
    assert result.result.order_count == 3
```

Test that invalid DAGs are rejected at construction time:

```python
def test_cycle_detection():
    with pytest.raises(CycleDetectedError):
        (
            Workflow.builder(name="bad")
            .add_task(TaskA, depends_on=TaskB)
            .add_task(TaskB, depends_on=TaskA)
            .build()
        )

def test_incomplete_fan_in():
    with pytest.raises(IncompleteInputError):
        (
            Workflow.builder(name="bad")
            .add_task(FetchUser)
            # Missing FetchOrders — "orders" field not mapped
            .add_task(EnrichUser, depends_on={"user": FetchUser})
            .build()
        )
```

---

## 14. Future Considerations

These are explicitly **out of scope** for v1 but worth noting for future iterations:

- **Async runner**: An `AsyncRunner` that `await`s async task implementations.
- **Parallel DAG execution**: Run independent tasks concurrently via `concurrent.futures` or an async event loop.
- **Retry policies**: Configurable per-task retry with backoff.
- **Job resumption**: Persist checkpoints so a failed job can resume from the last successful task.
- **YAML/JSON workflow definitions**: A config layer that maps to the Python code model.
- **Distributed execution**: Farm tasks out to worker processes or remote machines.

---

## 15. Key Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| Execution model | DAG (with linear shorthand) | Supports both sequential pipelines and fan-in/fan-out; linear pipelines are a degenerate DAG. |
| Task definition | Class-based with generics | Clear contracts, IDE support, easy to test in isolation. |
| Type safety | Pydantic models for I/O | Validation, serialization, and schema introspection for free. |
| Error handling | Fail fast | Predictable behavior; tasks can handle retries internally if needed. |
| Workflow definition | Python code only | No config parsing layer; full IDE support and type checking. |
| DAG support | Single `Workflow` class (DAG-native) | Linear pipeline is a degenerate DAG; fan-in via Pydantic field mapping keeps type safety. |
| Runner | Synchronous, topological order | One algorithm handles both linear and DAG workflows. |
| Observability | Hook protocol | Decoupled from runner; composable; easy to extend. |
| Python version | 3.12+ | Modern generics syntax, `StrEnum`, performance improvements. |
