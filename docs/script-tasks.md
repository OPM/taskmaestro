# Script and command tasks (issue #1)

## Problem
Users need small steps in a workflow: post-processing a result, copying files, or
calling an existing script or executable. Before this change, each such step needed a
`Task[I, O]` subclass with its own Pydantic input and output models. Issue #1 asks for
custom tasks without that overhead. The design follows the proposal in the issue
comments, with the changes listed under "Decisions".

## Agreed design

### Built-in tasks
Both tasks are registered in the `taskmaestro.tasks` entry-point group. They appear in
`taskmaestro tasks list`, and `tasks describe` reports their schemas.

| Task id | Class | Runs |
|---|---|---|
| `taskmaestro.python` | `PythonScript` | `<sys.executable> <script> <args...>` in a subprocess |
| `taskmaestro.command` | `Command` | `command` as a string through the shell, or as a list run directly, without a shell |

Input fields:

| Field | Type | Default | Tasks |
|---|---|---|---|
| `script` | path | required | python |
| `args` | list of str | `[]` | python |
| `command` | str \| list of str | required | command |
| `inputs` | `dict[str, Any]` | `{}` | both |
| `env` | mapping str → str | `{}` | both, added to the inherited environment |
| `cwd` | path | current dir | both |
| `timeout` | float > 0 (s) | none | both, kills the process |
| `check` | bool | `true` | both, fails on a nonzero exit code |

Output (`ScriptOutput`): `result: Any` (the JSON from `$TASKMAESTRO_OUTPUT`, or `null`),
`returncode`, `stdout`, `stderr`.

### Data exchange
Each run gets a fresh directory under `ctx.scratch_dir` (`script-*`). The directory is
deleted when the task finishes, also when it fails: `result`, `stdout`, `stderr` and
`returncode` are returned in `ScriptOutput` or `ScriptError`, and `inputs` may hold
sensitive data. Processes left running in the background must not rely on these files
after the task returns.

- `TASKMAESTRO_INPUT` is set only when `inputs` is not empty. It holds the path to
  `input.json`, which is `inputs` serialized with `pydantic_core.to_json`, so Pydantic
  models become objects.
- `TASKMAESTRO_OUTPUT` is always set. The process may write one JSON value there. A
  missing or empty file gives `result = null`. Invalid JSON fails the task.

### `inputs` is `dict[str, Any]`
Upstream data is connected with a keyed `collect`:
`inputs: {collect: {export: [exporter, result]}}`. A `[task, field]` reference whose
type is `dict[str, ...]` also works. Scripts therefore always receive a JSON object with
known keys. This type needs no change to the type checker, because keyed collections
into `dict[str, T]` were already supported.

Alternatives that were rejected:
- `Any` would accept a whole model, a list or a dict. Collections into `Any` would need a
  core validation change, and scripts would get an input whose shape varies.
- `str` would only accept string-typed fields. Most upstream outputs are models.

`result` is `Any`, so it can be connected to any downstream field. The downstream input
model validates the value at run time.

### Configuration
The settings are ordinary input fields. They are given per task in `input.yaml`, like
any other task configuration, and need no new YAML syntax.

The proposal also had a `params:` key on workflow task entries. It would hold fixed
settings, such as the script path, in `workflow.yaml`. It was left out because it is a
general YAML feature rather than something scripts need, and it can be proposed
separately.

### Errors and timeouts
Failures raise `ScriptError`, a `TaskExecutionError` with `returncode`, `stdout` and
`stderr`:
- A missing script is reported before anything runs.
- Inputs that cannot be converted to JSON fail the task with a message suggesting a
  regular Task. This applies to runtime-only objects such as `ObjectModel[rips.EclipseCase]`.
- A process that cannot start (`OSError`) fails the task.
- A `timeout` kills the process and fails the task.
- A nonzero exit code with `check: true` fails the task. The message includes the exit
  code and the last 20 lines of stderr.
- When a runner or task timeout (`SIGALRM`) fires while a process runs,
  `subprocess.run` kills and reaps the child before the timeout error propagates.

### Paths
Absolute paths are used as given. Relative `script` and `cwd` paths resolve against the
current directory of the process that runs the workflow, not the workflow file's
directory. `script` is made absolute before the process starts, so `cwd` does not
affect it. Paths inside `command` or `args` are interpreted by the child process
in `cwd`.

## Decisions
- **Function (in-process) mode is postponed.** The proposal had a default `function`
  mode that calls `main(inputs, options, ctx)` in the workflow's interpreter. That mode
  could receive live objects. If it is added later, it should be an opt-in `mode`
  field that defaults to `subprocess`, which keeps existing workflows unchanged.
- **No `params:` key** (see Configuration).
- No new dependencies.

## ResInsight integration notes
- `result` has the empty JSON schema `{}`. ResInsight must accept edges out of a field
  with that schema.
- `inputs` is an object schema whose values can be anything. ResInsight connects to it
  with keyed `collect` edges (`"collect": "dict"`).
- ResInsight could warn when an opaque (`x-taskmaestro-opaque`) output is collected into
  `inputs`, since that fails at run time.
