# Taskekrabbe

<p align="center">
  <img src="taskekrabbe.png" alt="Taskekrabbe" width="300">
</p>

A Python 3.12+ library for defining and executing typed DAG task workflows with Pydantic models, lifecycle hooks, and fail-fast semantics.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

### Linear Pipeline

```python
from pydantic import BaseModel
from taskekrabbe import Task, Workflow, Job, Runner, ExecutionContext

class NumberInput(BaseModel):
    value: int

class NumberOutput(BaseModel):
    value: int

class AddOne(Task[NumberInput, NumberOutput]):
    def run(self, input: NumberInput, ctx: ExecutionContext) -> NumberOutput:
        return NumberOutput(value=input.value + 1)

class Double(Task[NumberOutput, NumberOutput]):
    def run(self, input: NumberOutput, ctx: ExecutionContext) -> NumberOutput:
        return NumberOutput(value=input.value * 2)

workflow = Workflow(name="math", tasks=[AddOne, Double])
job = Job(workflow=workflow, config=NumberInput(value=5))
result = Runner().run(job)

print(result.status)        # "completed"
print(result.result.value)  # 12
```

### DAG Workflow (Fan-In)

```python
from pydantic import BaseModel
from taskekrabbe import Task, Workflow, Job, Runner, ExecutionContext

class Input(BaseModel):
    value: int

class Output(BaseModel):
    value: int

class MergedInput(BaseModel):
    a: Output
    b: Output

class MergedOutput(BaseModel):
    total: int

class BranchA(Task[Input, Output]):
    def run(self, input: Input, ctx: ExecutionContext) -> Output:
        return Output(value=input.value + 1)

class BranchB(Task[Input, Output]):
    def run(self, input: Input, ctx: ExecutionContext) -> Output:
        return Output(value=input.value * 2)

class Merge(Task[MergedInput, MergedOutput]):
    def run(self, input: MergedInput, ctx: ExecutionContext) -> MergedOutput:
        return MergedOutput(total=input.a.value + input.b.value)

workflow = (
    Workflow.builder(name="fan_in")
    .add_task(BranchA)
    .add_task(BranchB)
    .add_task(Merge, depends_on={"a": BranchA, "b": BranchB})
    .build()
)

job = Job(workflow=workflow, config=Input(value=5))
result = Runner().run(job)

print(result.status)        # "completed"
print(result.result.total)  # 16 (6 + 10)
```

## Core Concepts

You define **Tasks** (typed units of work), compose them into a **Workflow** (linear chain or DAG), bind input data via a **Job**, and hand it to a **Runner** for execution. Type safety is enforced at build time — input/output models are validated across the entire graph. Fail-fast semantics stop execution on the first error, and **Hooks** provide cross-cutting lifecycle observations without coupling to task logic.

| Concept | Description |
|---|---|
| **Task** | Subclass `Task[I, O]` with Pydantic models for input and output, then implement `run(input, ctx)`. Each task can declare an optional `timeout_seconds`. For tasks with multiple named outputs, use inline `Inputs`/`Outputs` classes inside the task body. |
| **Workflow** | Build a linear pipeline with `Workflow(tasks=[...])` or a DAG with `Workflow.builder()`. The builder accepts `depends_on` for single dependencies, fan-in dicts (`{"field": UpstreamTask}`), and `(Task, "field")` tuples for output field routing. Workflows are validated at build time for cycles, type compatibility, and input completeness. |
| **Job** | Binds a Workflow to a typed config (the root task's input). Tracks `status` (`pending` → `running` → `completed`/`failed`), the final `result`, any `error`, and per-task `task_results`. A job can only be run once. |
| **Runner** | Executes tasks in topological order, stopping on the first failure (fail-fast). Supports per-task and per-job timeouts via `signal.alarm` (Unix only). Dispatches lifecycle events to registered hooks. |
| **ExecutionContext** | Passed to every `run()` call. Provides a `logger`, an auto-generated `correlation_id` (UUID), a `scratch_dir` (temporary directory), and a service registry (`register()`/`resolve()`) for injecting shared resources like DB connections. |
| **Hooks** | Subclass `BaseHook` and override methods like `on_job_start`, `on_task_complete`, etc. Hook errors are swallowed and reported via `warnings.warn()`, so they never crash the job. Built-ins: `LoggingHook`, `TimingHook`, `ResultPersistenceHook`. |

## Error Handling

```
WorkflowRunnerError (base)
├── WorkflowDefinitionError       # Invalid workflow definition
│   ├── CycleDetectedError        # Dependency cycle
│   └── IncompleteInputError      # Missing fan-in field mappings
├── JobStateError                 # e.g., re-running a completed job
└── TaskExecutionError            # Runtime task failure
    ├── TaskOutputTypeError       # Output type mismatch
    └── TaskTimeoutError          # Task exceeded timeout
```

## Development

```bash
source .venv/bin/activate
pytest -v                  # run tests
ruff check .               # lint
ruff format .              # format
mypy taskekrabbe       # type check (strict)
```
