"""Tests for YAML workflow configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from taskekrabbe import (
    ConfigLoadError,
    ExecutionContext,
    JobStatus,
    Task,
)
from taskekrabbe.yaml_config import (
    YamlWorkflowConfig,
    _coerce_hook_params,
    import_class,
    load_workflow_from_yaml,
    run_workflow_from_yaml,
)

# --- Test task/model definitions used by YAML configs ---


class TextInput(BaseModel):
    text: str


class TextOutput(BaseModel):
    text: str


class UpperText(Task[TextInput, TextOutput]):
    name = "upper_text"

    def run(self, input: TextInput, ctx: ExecutionContext) -> TextOutput:
        return TextOutput(text=input.text.upper())


class ReverseText(Task[TextOutput, TextOutput]):
    name = "reverse_text"

    def run(self, input: TextOutput, ctx: ExecutionContext) -> TextOutput:
        return TextOutput(text=input.text[::-1])


class LengthOutput(BaseModel):
    length: int


class TextLength(Task[TextOutput, LengthOutput]):
    name = "text_length"

    def run(self, input: TextOutput, ctx: ExecutionContext) -> LengthOutput:
        return LengthOutput(length=len(input.text))


class FanInInput(BaseModel):
    reversed: TextOutput
    length: LengthOutput


class FanInOutput(BaseModel):
    summary: str


class CombineResults(Task[FanInInput, FanInOutput]):
    name = "combine_results"

    def run(self, input: FanInInput, ctx: ExecutionContext) -> FanInOutput:
        return FanInOutput(summary=f"{input.reversed.text} ({input.length.length} chars)")


THIS_MODULE = "tests.test_yaml_config"


def _write_workflow_yaml(tmp_path: Path, content: str) -> Path:
    """Write workflow YAML content to a temp file and return the path."""
    p = tmp_path / "workflow.yaml"
    p.write_text(content)
    return p


def _write_input_yaml(tmp_path: Path, content: str) -> Path:
    """Write input YAML content to a temp file and return the path."""
    p = tmp_path / "input.yaml"
    p.write_text(content)
    return p


# ============================================================
# TestImportClass
# ============================================================


class TestImportClass:
    def test_valid_import(self) -> None:
        cls = import_class(f"{THIS_MODULE}.UpperText")
        assert cls is UpperText

    def test_nonexistent_module(self) -> None:
        with pytest.raises(ConfigLoadError, match="Cannot import module"):
            import_class("nonexistent.module.ClassName")

    def test_nonexistent_class(self) -> None:
        with pytest.raises(ConfigLoadError, match="has no attribute"):
            import_class(f"{THIS_MODULE}.NonexistentClass")

    def test_invalid_path_no_dot(self) -> None:
        with pytest.raises(ConfigLoadError, match="Invalid import path"):
            import_class("NoDotPath")

    def test_imports_non_task(self) -> None:
        # import_class itself doesn't check Task subclass, just imports
        cls = import_class(f"{THIS_MODULE}.TextInput")
        assert cls is TextInput


# ============================================================
# TestYamlSchemaValidation
# ============================================================


class TestYamlSchemaValidation:
    def test_valid_minimal_config(self) -> None:
        raw = {
            "workflow": {
                "name": "test",
                "tasks": [{"task": "some.module.Task"}],
            },
        }
        config = YamlWorkflowConfig.model_validate(raw)
        assert config.workflow.name == "test"
        assert len(config.workflow.tasks) == 1
        assert config.runner.timeout_seconds is None
        assert config.runner.hooks == []
        assert config.context.correlation_id is None
        assert config.context.services == {}

    def test_missing_workflow_key(self) -> None:
        raw: dict[str, object] = {}
        with pytest.raises(ValidationError):
            YamlWorkflowConfig.model_validate(raw)

    def test_empty_tasks_list(self) -> None:
        raw = {
            "workflow": {"name": "test", "tasks": []},
        }
        with pytest.raises(ValidationError):
            YamlWorkflowConfig.model_validate(raw)

    def test_full_config_with_all_sections(self) -> None:
        raw = {
            "workflow": {
                "name": "full",
                "result_task": "mod.ResultTask",
                "tasks": [
                    {"task": "mod.A"},
                    {"task": "mod.B", "depends_on": "mod.A"},
                ],
            },
            "runner": {
                "timeout_seconds": 300,
                "hooks": [
                    {"hook": "mod.Hook", "params": {"level": 20}},
                ],
            },
            "context": {
                "correlation_id": "abc-123",
                "scratch_dir": "/tmp/test",
                "services": {"multiplier": 3},
            },
        }
        config = YamlWorkflowConfig.model_validate(raw)
        assert config.workflow.result_task == "mod.ResultTask"
        assert config.runner.timeout_seconds == 300
        assert len(config.runner.hooks) == 1
        assert config.runner.hooks[0].params == {"level": 20}
        assert config.context.correlation_id == "abc-123"
        assert config.context.services == {"multiplier": 3}

    def test_depends_on_as_dict(self) -> None:
        raw = {
            "workflow": {
                "name": "test",
                "tasks": [
                    {"task": "mod.A"},
                    {"task": "mod.B"},
                    {
                        "task": "mod.C",
                        "depends_on": {"x": "mod.A", "y": "mod.B"},
                    },
                ],
            },
        }
        config = YamlWorkflowConfig.model_validate(raw)
        assert config.workflow.tasks[2].depends_on == {"x": "mod.A", "y": "mod.B"}


# ============================================================
# TestLoadWorkflowFromYaml
# ============================================================


class TestLoadWorkflowFromYaml:
    def test_linear_workflow_end_to_end(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: linear_test
  tasks:
    - task: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.ReverseText
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)

        assert loaded.workflow.name == "linear_test"
        assert loaded.job.status == JobStatus.PENDING

        result = loaded.run()
        assert result.status == JobStatus.COMPLETED
        assert result.result is not None
        assert result.result.text == "OLLEH"  # type: ignore[union-attr]

    def test_dag_workflow_end_to_end(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: dag_test
  tasks:
    - task: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.ReverseText
      depends_on: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.TextLength
      depends_on: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.CombineResults
      depends_on:
        reversed: {THIS_MODULE}.ReverseText
        length: {THIS_MODULE}.TextLength
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)
        result = loaded.run()

        assert result.status == JobStatus.COMPLETED
        assert result.result is not None
        assert "OLLEH" in result.result.summary  # type: ignore[union-attr]
        assert "5 chars" in result.result.summary  # type: ignore[union-attr]

    def test_minimal_config(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: minimal
  tasks:
    - task: {THIS_MODULE}.UpperText
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)
        result = loaded.run()
        assert result.status == JobStatus.COMPLETED
        assert result.result.text == "HELLO"  # type: ignore[union-attr]

    def test_hooks_with_params(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: hooks_test
  tasks:
    - task: {THIS_MODULE}.UpperText
runner:
  hooks:
    - hook: taskekrabbe.hooks.logging.LoggingHook
      params:
        level: 20
    - hook: taskekrabbe.hooks.timing.TimingHook
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)
        assert len(loaded.runner.hooks) == 2

    def test_persistence_hook_path_coercion(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "results"
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: persist_test
  tasks:
    - task: {THIS_MODULE}.UpperText
runner:
  hooks:
    - hook: taskekrabbe.hooks.persistence.ResultPersistenceHook
      params:
        output_dir: "{output_dir}"
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)
        result = loaded.run()
        assert result.status == JobStatus.COMPLETED
        assert (output_dir / "upper_text.json").exists()

    def test_context_services(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: ctx_test
  tasks:
    - task: {THIS_MODULE}.UpperText
context:
  correlation_id: test-run-42
  services:
    multiplier: 3
    name: test
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)
        assert loaded.context.correlation_id == "test-run-42"
        assert loaded.context.resolve("multiplier") == 3
        assert loaded.context.resolve("name") == "test"

    def test_context_scratch_dir(self, tmp_path: Path) -> None:
        scratch = tmp_path / "scratch"
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: scratch_test
  tasks:
    - task: {THIS_MODULE}.UpperText
context:
  scratch_dir: "{scratch}"
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)
        assert loaded.context.scratch_dir == scratch

    def test_bad_import_path(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            """\
workflow:
  name: bad
  tasks:
    - task: nonexistent.module.BadTask
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        with pytest.raises(ConfigLoadError, match="Cannot import module"):
            load_workflow_from_yaml(wf_path, in_path)

    def test_not_a_task_class(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: bad
  tasks:
    - task: {THIS_MODULE}.TextInput
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        with pytest.raises(ConfigLoadError, match="not a Task subclass"):
            load_workflow_from_yaml(wf_path, in_path)

    def test_bad_yaml_syntax(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            """\
workflow:
  name: bad
  tasks:
    - task: [invalid yaml
input:
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        with pytest.raises(ConfigLoadError, match="YAML parse error"):
            load_workflow_from_yaml(wf_path, in_path)

    def test_input_validation_error(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: bad_input
  tasks:
    - task: {THIS_MODULE}.UpperText
""",
        )
        in_path = _write_input_yaml(tmp_path, "wrong_field: 123\n")
        with pytest.raises(ConfigLoadError, match="Input validation error"):
            load_workflow_from_yaml(wf_path, in_path)

    def test_file_not_found(self, tmp_path: Path) -> None:
        wf_path = tmp_path / "nonexistent.yaml"
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        with pytest.raises(ConfigLoadError, match="Cannot read file"):
            load_workflow_from_yaml(wf_path, in_path)

    def test_input_file_not_found(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: test
  tasks:
    - task: {THIS_MODULE}.UpperText
""",
        )
        in_path = tmp_path / "nonexistent_input.yaml"
        with pytest.raises(ConfigLoadError, match="Cannot read input file"):
            load_workflow_from_yaml(wf_path, in_path)

    def test_input_bad_yaml_syntax(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: test
  tasks:
    - task: {THIS_MODULE}.UpperText
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: [invalid yaml\n")
        with pytest.raises(ConfigLoadError, match="Input YAML parse error"):
            load_workflow_from_yaml(wf_path, in_path)

    def test_input_not_a_mapping(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: test
  tasks:
    - task: {THIS_MODULE}.UpperText
""",
        )
        in_path = _write_input_yaml(tmp_path, "- item1\n- item2\n")
        with pytest.raises(ConfigLoadError, match="Input YAML file must contain a mapping"):
            load_workflow_from_yaml(wf_path, in_path)

    def test_explicit_result_task(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: explicit_result
  result_task: {THIS_MODULE}.UpperText
  tasks:
    - task: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.ReverseText
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)
        assert loaded.workflow.result_task.name == "upper_text"

    def test_dependency_not_found(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: bad_dep
  tasks:
    - task: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.ReverseText
      depends_on: nonexistent.module.Missing
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        with pytest.raises(ConfigLoadError, match="not found"):
            load_workflow_from_yaml(wf_path, in_path)

    def test_not_a_hook_class(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: bad_hook
  tasks:
    - task: {THIS_MODULE}.UpperText
runner:
  hooks:
    - hook: {THIS_MODULE}.TextInput
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        with pytest.raises(ConfigLoadError, match="not a BaseHook subclass"):
            load_workflow_from_yaml(wf_path, in_path)

    def test_hook_bad_params(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: bad_hook_params
  tasks:
    - task: {THIS_MODULE}.UpperText
runner:
  hooks:
    - hook: taskekrabbe.hooks.logging.LoggingHook
      params:
        nonexistent_param: true
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        with pytest.raises(ConfigLoadError, match="Cannot instantiate hook"):
            load_workflow_from_yaml(wf_path, in_path)

    def test_timeout_seconds_passed(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: timeout_test
  tasks:
    - task: {THIS_MODULE}.UpperText
runner:
  timeout_seconds: 60
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)
        assert loaded._timeout_seconds == 60


# ============================================================
# TestRunWorkflowFromYaml
# ============================================================


class TestYamlFieldRouting:
    """Tests for YAML list-form depends_on (field routing)."""

    def test_list_depends_on_end_to_end(self, tmp_path: Path) -> None:
        """Field routing via list-form depends_on works end-to-end."""
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: yaml_field_ref
  tasks:
    - task: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.TextLength
      depends_on: {THIS_MODULE}.UpperText
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)
        result = loaded.run()
        assert result.status == JobStatus.COMPLETED
        assert result.result.length == 5  # type: ignore[union-attr]

    def test_dict_with_list_field_ref(self, tmp_path: Path) -> None:
        """Fan-in with list-form field refs in YAML dict."""
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: yaml_mixed
  tasks:
    - task: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.ReverseText
      depends_on: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.TextLength
      depends_on: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.CombineResults
      depends_on:
        reversed: {THIS_MODULE}.ReverseText
        length: {THIS_MODULE}.TextLength
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)
        result = loaded.run()
        assert result.status == JobStatus.COMPLETED
        assert "OLLEH" in result.result.summary  # type: ignore[union-attr]

    def test_invalid_list_length_raises(self, tmp_path: Path) -> None:
        """A list with != 2 elements raises ConfigLoadError."""
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: bad
  tasks:
    - task: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.ReverseText
      depends_on:
        - {THIS_MODULE}.UpperText
        - text
        - extra
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        with pytest.raises(ConfigLoadError, match="List depends_on must be"):
            load_workflow_from_yaml(wf_path, in_path)

    def test_list_dep_not_found_raises(self, tmp_path: Path) -> None:
        """List-form depends_on with unknown task raises ConfigLoadError."""
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: bad
  tasks:
    - task: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.ReverseText
      depends_on:
        - nonexistent.module.Task
        - text
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        with pytest.raises(ConfigLoadError, match="not found"):
            load_workflow_from_yaml(wf_path, in_path)


class TestRunWorkflowFromYaml:
    def test_convenience_function(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: convenience
  tasks:
    - task: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.ReverseText
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        job = run_workflow_from_yaml(wf_path, in_path)
        assert job.status == JobStatus.COMPLETED
        assert job.result is not None
        assert job.result.text == "OLLEH"  # type: ignore[union-attr]


# ============================================================
# TestLoadedWorkflow
# ============================================================


class TestLoadedWorkflow:
    def test_run_method(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: run_test
  tasks:
    - task: {THIS_MODULE}.UpperText
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)
        result = loaded.run()
        assert result.status == JobStatus.COMPLETED

    def test_components_accessible(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: access_test
  tasks:
    - task: {THIS_MODULE}.UpperText
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)
        assert loaded.workflow is not None
        assert loaded.runner is not None
        assert loaded.job is not None
        assert loaded.context is not None

    def test_frozen(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: frozen_test
  tasks:
    - task: {THIS_MODULE}.UpperText
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)
        with pytest.raises(AttributeError):
            loaded.workflow = None  # type: ignore[misc]


# ============================================================
# TestCoerceHookParams
# ============================================================


class TestCoerceHookParams:
    def test_path_coercion(self) -> None:
        from taskekrabbe.hooks.persistence import ResultPersistenceHook

        params = _coerce_hook_params(ResultPersistenceHook, {"output_dir": "/tmp/out"})
        assert isinstance(params["output_dir"], Path)
        assert params["output_dir"] == Path("/tmp/out")

    def test_passthrough_non_path(self) -> None:
        from taskekrabbe.hooks.logging import LoggingHook

        params = _coerce_hook_params(LoggingHook, {"level": 20})
        assert params["level"] == 20
        assert isinstance(params["level"], int)

    def test_empty_params(self) -> None:
        from taskekrabbe.hooks.timing import TimingHook

        params = _coerce_hook_params(TimingHook, {})
        assert params == {}

    def test_extra_params_error(self, tmp_path: Path) -> None:
        """Extra params should cause TypeError when instantiating the hook."""
        from taskekrabbe.hooks.logging import LoggingHook

        params = _coerce_hook_params(LoggingHook, {"level": 20, "unknown": True})
        with pytest.raises(TypeError):
            LoggingHook(**params)
