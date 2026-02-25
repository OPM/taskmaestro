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

| Concept | Description |
|---|---|
| **Task** | A unit of work with typed `Input` and `Output` (Pydantic models). Subclass `Task[I, O]` and implement `run()`. |
| **Workflow** | A DAG of tasks. Use `Workflow(tasks=[...])` for linear chains or `Workflow.builder()` for DAGs. |
| **Job** | A workflow bound to a config (input data). Tracks status and results. |
| **Runner** | Executes jobs by iterating tasks in topological order. Supports hooks and timeouts. |
| **ExecutionContext** | Cross-cutting state (logger, correlation ID, service registry) passed to every task. |
| **Hooks** | Lifecycle observers (job/task start/complete/fail). Built-in: `LoggingHook`, `TimingHook`, `ResultPersistenceHook`. |

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
