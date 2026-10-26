# Conditional branch execution (issue #16)

## Problem
The runner executes every task in topological order and stops at the first failure. Nothing lets a workflow pick a path at runtime. Issue #16 asks for conditional switching between branches.

## Agreed design

### Concepts
- **`when` guard**: a *control edge* on a task. It adds ordering and a runtime predicate, but it carries no data. The guarded task still gets its input through `depends_on`.
- **`SwitchTask[I, L]`**: the user implements `decide(input, ctx) -> L`, where `L` is a `Literal[...]` or `StrEnum`. The base class's final `run()` wraps the returned label in `Branch[L]`, a generic Pydantic model with one field, `branch: L`. `__init_subclass__` sets `Outputs = Branch[L]`, and the existing `get_output_type` honours that, so introspection does not change. The label set comes from `L`.
- **`ConditionTask[I]`**: a `SwitchTask` with branches `True`/`False`. `decide -> bool`.
- **Handle sugar**: `handle.on(*labels)` returns a guard (`in`). `handle.otherwise()` matches any label that no other guard on the same switch claims. For a ConditionTask: `on(True)` and `on(False)`, plus `if_true()`/`if_false()` aliases.
- **Generic field guards**: `when(ref, equals=…)`, `when(ref, in_=[…])`, `when(ref, truthy=True|False)` on any upstream output field. `ref` is an `OutputHandle`, a `(task, field)` tuple, or a name string.
- **Join**: `first_of(a, b, …)` is a dependency value like `collect()`. It resolves to the first member, in declaration order, that completed.
- Labels are chosen over target task ids. They are validated at build time, one label can enable many tasks, and renaming a task does not break routing.

### Runtime semantics
- New `TaskStatus.SKIPPED`. A skipped task gets a `TaskResult(status=SKIPPED, output=None, error=None)` plus a reason, and `Event.TASK_SKIP` / `on_task_skip(job, task, reason)` is emitted. A skipped task does **not** emit `task_start`.
- A task is skipped when any of these holds:
  1. Its guard evaluates false.
  2. The guard's source task was skipped.
  3. Any plain data dependency (whole output, field ref, or a direct fan-in value) was skipped.
  4. Every member of one of its `first_of` dependencies was skipped.
- `collect()` drops skipped members. A list gets shorter; a dict loses those keys. If all members are skipped, the result is an empty list or dict, which is not a skip.
- Skips propagate transitively because tasks run in topological order.
- If the result task is skipped, the job **fails**. The new `ResultTaskSkippedError` goes in `job.exception`/`job.error`, `failed_task = result task`, and `JOB_FAIL` is emitted.
- If a guarded task is mapped and its guard is false, the whole node is skipped. No map items run.
- Nested workflows: if the inner result is skipped, the inner job fails, which raises `WorkflowTaskError` through the existing path.

### Static validation (`Workflow._validate`)
New step after references: `_validate_guards`. The order becomes: unique names → references (now including guard and `first_of` refs) → acyclic (including control edges) → task maps → guards → types → result task.
- The guard source task and field must exist. A guard cannot read a field on a mapped task's output (`MappedOutput`).
- `on()`/`equals`/`in` values must be compatible with the field annotation. When the field is a `Literal` or enum, a value outside it is an error, which catches typos.
- `truthy`/`falsy` guards are only allowed on `bool` or Optional fields.
- For each switch: every declared label is consumed by some `on()` or covered by `otherwise()`, else build error. `otherwise()` matches the labels not claimed by any `on()` on that switch. Several tasks may use `otherwise()`, and they all mean the same thing. If every label is already claimed, `otherwise()` is an error because it is unreachable.
- `first_of` needs at least 2 members, and each member type must be compatible with the destination field (reuses `_is_type_compatible`).
- `topological_order`, `_find_sinks`, and cycle detection include guard sources as upstreams. A switch that is consumed only by guards is therefore not a sink.

### Stored representation (`workflow.py`)
- `Workflow._guards: dict[str, Guard]`, where `Guard` is a frozen dataclass in a new `taskmaestro/conditions.py`: `ref: OutputRef`, `op: Literal["in","truthy","falsy","otherwise"]`, `values: tuple[Any, ...]`. `equals` is normalized to `in` with one value. The user-facing helpers `when()` and `first_of()` and the classes `SwitchTask`, `ConditionTask`, `Branch` also live there.
- `FirstOfRef(members: tuple[OutputRef, ...])` is stored as a `FanInValue`, next to `CollectionRef`. Update `_extract_upstream_names`, `_validate_types`, and `build()` copying.
- New public accessors: `get_guard(name)`, `is_conditional(name)`.

## Surfaces to update
| Area | Change |
|---|---|
| `conditions.py` (new) | `Branch`, `SwitchTask`, `ConditionTask`, `Guard`, `Condition` (unresolved), `when()`; `first_of()`/`FirstOfRef` live in `dependencies.py` |
| `dependencies.py` | `TaskHandle.on()`, `.otherwise()`, `.if_true()`, `.if_false()`; `OutputHandle.equals()/.is_in()/.truthy()/.falsy()` sugar |
| `workflow.py` | `when=` kwarg on `add_task`/`task`/`map_task`; resolution of guard and `first_of` refs; validation; topo/sinks/cycles; snapshot copy in `build()` |
| `runner.py` | evaluate guard and skip rules before `TASK_START`; skipped set; `first_of` resolution; `collect()` dropping skipped members; result-task-skipped failure |
| `job.py` | `TaskStatus.SKIPPED`; `TaskResult.skip_reason: str \| None = None`; `Job.skipped_tasks` convenience property |
| `exceptions.py` | `ResultTaskSkippedError(TaskExecutionError)` (check hierarchy) |
| `hooks/base.py` | `Event.TASK_SKIP`, `on_task_skip` on the Protocol and `BaseHook` |
| `hooks/logging.py`, `timing.py`, `persistence.py` | log the skip; timing records nothing (or 0); persistence writes nothing for skipped tasks |
| `yaml_config.py` | `TaskConfig.when`: `{switch: <task>, case: <label> \| [labels]}`, `{switch: <task>, otherwise: true}` (not `case: default`, since `default` could be a real label) or `{ref: <task> \| [task, field], equals \| in \| truthy: …}` (`truthy: false` means falsy); dependency value `{first_of: [refs]}` |
| `visualization.py` | dashed control edge `route -.->\|express\| fast`; `first_of` rendered as a rounded (stadium) join node; switch nodes drawn as diamonds `id{"label"}` |
| `introspection.py` | `describe_workflow` adds per-task `when` (JSON form) and the switch `branches` list; `dependency_spec` renders `first_of` |
| `definition.py` | `when` on nodes (or `control` edges with `kind: "when"`), `first_of` edges (`"first_of": true` with order); load and write round-trip; `_check_saveable` |
| `cli.py` | `run --json`: include skipped tasks in output if task statuses are reported; check the existing contract |
| `__init__.py` | export `SwitchTask`, `ConditionTask`, `Branch`, `when`, `first_of`, `ResultTaskSkippedError` |
| examples | `examples/conditional_if/` (`ConditionTask` + `first_of`) and `examples/conditional_switch/` (`SwitchTask` + `on`/`otherwise` + `first_of`), each with `pipeline.py`, `workflow.yaml`, `input.yaml` |
| docs | README "Conditional Branches" section plus a YAML snippet; AGENTS.md architecture row and key-pattern bullet (skip semantics, validation order) |

## Todos
1. **conditions-core**: `conditions.py` types (`Branch`, `SwitchTask`, `ConditionTask`, `Guard`, `FirstOfRef`, `when`, `first_of`) and handle sugar in `dependencies.py`.
2. **workflow-model**: builder `when=` and `first_of` resolution, storage, accessors, topo/sinks/cycles, `build()` copy.
3. **workflow-validation**: `_validate_guards`, `first_of` type checks, label coverage and `otherwise` rules.
4. **job-hooks**: `TaskStatus.SKIPPED`, `TaskResult.skip_reason`, `Event.TASK_SKIP`, hook implementations, `ResultTaskSkippedError`.
5. **runner**: skip evaluation, `first_of`/`collect` resolution with skips, result-skipped failure.
6. **yaml**: `when`/`first_of` in YAML schema and loader.
7. **visualization**: Mermaid control edges, switch nodes, `first_of` nodes.
8. **introspection**: describe JSON and `dependency_spec`.
9. **definition**: round-trip of `when` and `first_of` in the editable graph form.
10. **exports-cli**: `__init__` exports; CLI JSON output for skipped tasks.
11. **example-docs**: example directory, README, AGENTS.md.
12. **tests**: focused tests in existing files (`test_workflow`, `test_runner`, `test_hooks`, `test_yaml_config`, `test_visualization`, `test_definition`, `test_cli`, `test_workflow_task`) plus `test_conditions.py`. Run `pytest`, `ruff check`, `ruff format`, `mypy taskmaestro`.

## Notes / risks
- Adding `on_task_skip` to the `runtime_checkable` `Hook` Protocol changes `isinstance(x, Hook)` for third-party hooks that do not subclass `BaseHook`. Check how `Hook` is used. If it is used for isinstance checks, keep the Protocol method optional or document the change.
- `Branch[L]` is a generic Pydantic model. Check that `_safe_type_name`, JSON schema, and `_is_type_compatible` handle `Literal`/`StrEnum` field annotations. For an `in` check against a `Literal`, compare against `get_args`.
- A bool `ConditionTask` gives YAML case labels `true`/`false`. YAML parses them as booleans, which matches the `Literal[True, False]` values.
- `first_of` with a member that comes before the guard in topo order: "first completed in declaration order" stays deterministic.
- Cross-cutting: `Job.task_results` now contains skipped entries. Check consumers that assume every result is COMPLETED or FAILED (CLI, persistence, examples).
