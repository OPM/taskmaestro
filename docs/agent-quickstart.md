# Agent quickstart

Use this as a short, reproducible path from a typed task to a validated workflow. Commands below run from the **repository root**. The example uses only Taskmaestro's declared dependencies; it needs no API keys or external services.

## Set up

Python 3.12+ is required. In a fresh checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

If you already have Taskmaestro installed in another environment, use that environment's Python instead of `.venv/bin/python` below.

## Inspect the example

- [`examples/agent_quickstart/tasks.py`](../examples/agent_quickstart/tasks.py) defines `AddOne` and `Double`. Each accepts and returns a Pydantic `Number` model. Tasks implement `run(input, ctx)`.
- [`examples/agent_quickstart/workflow.yaml`](../examples/agent_quickstart/workflow.yaml) gives the tasks stable instance names, configures `add_one.value` from the input file, and routes `add_one`'s output to `double`.
- [`examples/agent_quickstart/input.yaml`](../examples/agent_quickstart/input.yaml) supplies `value: 5` to `add_one`.

The YAML loader imports `tasks.AddOne` and `tasks.Double` from the workflow file's directory when invoked through the CLI. No package installation or Python path changes are needed for these local tasks.

## Validate, inspect, run

```bash
.venv/bin/python -m taskmaestro validate examples/agent_quickstart/workflow.yaml --input examples/agent_quickstart/input.yaml
.venv/bin/python -m taskmaestro graph examples/agent_quickstart/workflow.yaml --input examples/agent_quickstart/input.yaml
.venv/bin/python -m taskmaestro run examples/agent_quickstart/workflow.yaml --input examples/agent_quickstart/input.yaml
```

`validate` prints `Workflow 'agent_quickstart' is valid`; `graph` prints Mermaid text with `add_one -->|Number| double`; `run` prints JSON:

```json
{
  "value": 12
}
```

The calculation is `(5 + 1) * 2`. The CLI writes errors to stderr and returns a nonzero exit code on failure. `validate` loads and checks the workflow and job configuration **without executing tasks**; `run` executes them.

## Fix a validation error

To reproduce a missing configuration field without changing the checked-in input, create a temporary empty input file:

```bash
printf '{}\n' > /tmp/taskmaestro-agent-bad-input.yaml
.venv/bin/python -m taskmaestro validate examples/agent_quickstart/workflow.yaml --input /tmp/taskmaestro-agent-bad-input.yaml
```

This exits with code 2 and reports:

```text
Configuration error: Job validation failed: Task 'add_one' is missing configuration fields ['value']
```

The workflow declares `value` in `config_fields`, so the input must include `add_one: {value: 5}`. Validate again with the checked-in input, then run. Remove the temporary file when done:

```bash
rm /tmp/taskmaestro-agent-bad-input.yaml
```

## When generating your own workflow

1. Define each task's input and output as Pydantic models in a Python module. Subclass `Task[InputModel, OutputModel]` and implement `run(input, ctx)`.
2. List tasks in a workflow YAML file. Prefer explicit `name:` values; dependency references and top-level input YAML keys refer to **instance names**. Use `depends_on` to connect tasks and `config_fields` for fields supplied from input YAML.
3. Put input data under the task instance name in a separate YAML file. Run `validate`, then `graph`, then `run`. When validation fails, correct the named task/field before retrying.
4. For runtime failures, check stderr for the failing task. A successful validation does not test the task's `run()` logic or guarantee external services are available.

## Discover installed task plugins

If tasks are published by an installed package through the `taskmaestro.tasks` entry-point group, inspect their identifiers and model schemas before generating a workflow:

```bash
.venv/bin/python -m taskmaestro tasks list --json
.venv/bin/python -m taskmaestro tasks describe acme.prepare --json
```

The list command returns a sorted JSON object such as `{"tasks": ["acme.prepare"]}`. The describe command returns the chosen task's `identifier`, `name`, `timeout_seconds`, and Pydantic `input_schema` / `output_schema` JSON Schema objects. Fields containing runtime-only Python objects are marked `x-taskmaestro-opaque` and `x-taskmaestro-python-type`, with `"not": {}` because no JSON value can satisfy them; route these values from upstream tasks rather than inventing JSON input. Replace `acme.prepare` with an identifier from your list; if you have no installed task plugins, the list is empty. The example tasks above are **local Python classes**, not installed plugins, so they will not appear in `tasks list`.

## Parse results and errors as JSON

Use `--json` with `validate` and `run` when you need a stable response instead of parsing text from stderr:

```bash
.venv/bin/python -m taskmaestro validate examples/agent_quickstart/workflow.yaml --input examples/agent_quickstart/input.yaml --json
.venv/bin/python -m taskmaestro run examples/agent_quickstart/workflow.yaml --input examples/agent_quickstart/input.yaml --json
```

Validation returns `{"status": "valid", "workflow": "agent_quickstart"}`; a successful run returns `{"status": "completed", "workflow": "agent_quickstart", "result": {"value": 12}}`. On failure, stdout contains a single JSON object with status `invalid` (configuration error) or `failed` (execution/serialization error), and an `error` containing `code`, `type`, `message`, `task`, `field`, and `issues`. Missing metadata is `null` or an empty list. Exit codes are 0 for success, 1 for a task/serialization failure, and 2 for an invalid configuration. Error messages omit raw exception details and input values; stderr may still contain application logs or prints. See the [CLI section of the README](../README.md#command-line-interface) for the full contract.

For fan-in, collections, mapping, nested workflows, and the Python builder API, see the [README](../README.md) and the other examples under `examples/`.
