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
    TaskConfig,
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


class WrappedOutput(BaseModel):
    inner: TextOutput
    count: int


class WrapText(Task[TextInput, WrappedOutput]):
    name = "wrap_text"

    def run(self, input: TextInput, ctx: ExecutionContext) -> WrappedOutput:
        return WrappedOutput(inner=TextOutput(text=input.text.upper()), count=len(input.text))


class FieldFanInInput(BaseModel):
    inner: TextOutput
    length: LengthOutput


class FieldFanInOutput(BaseModel):
    summary: str


class FieldFanInTask(Task[FieldFanInInput, FieldFanInOutput]):
    name = "field_fan_in"

    def run(self, input: FieldFanInInput, ctx: ExecutionContext) -> FieldFanInOutput:
        return FieldFanInOutput(summary=f"{input.inner.text} ({input.length.length} chars)")


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

    def test_workflow_yaml_not_a_mapping(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(tmp_path, "- item1\n- item2\n")
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        with pytest.raises(ConfigLoadError, match="must contain a mapping"):
            load_workflow_from_yaml(wf_path, in_path)

    def test_schema_validation_error(self, tmp_path: Path) -> None:
        wf_path = _write_workflow_yaml(tmp_path, "workflow:\n  name: test\n")
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        with pytest.raises(ConfigLoadError, match="YAML schema validation error"):
            load_workflow_from_yaml(wf_path, in_path)

    def test_list_depends_on_field_routing(self, tmp_path: Path) -> None:
        """List-form depends_on for single-upstream field routing in DAG mode."""
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: list_field_ref
  tasks:
    - task: {THIS_MODULE}.WrapText
    - task: {THIS_MODULE}.ReverseText
      depends_on:
        - {THIS_MODULE}.WrapText
        - inner
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)
        result = loaded.run()
        assert result.status == JobStatus.COMPLETED
        assert result.result.text == "OLLEH"  # type: ignore[union-attr]

    def test_dict_fan_in_with_list_field_ref(self, tmp_path: Path) -> None:
        """Dict depends_on with list-form field refs (fan-in + field routing)."""
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: dict_list_ref
  tasks:
    - task: {THIS_MODULE}.WrapText
    - task: {THIS_MODULE}.TextLength
      depends_on:
        - {THIS_MODULE}.WrapText
        - inner
    - task: {THIS_MODULE}.FieldFanInTask
      depends_on:
        inner:
          - {THIS_MODULE}.WrapText
          - inner
        length: {THIS_MODULE}.TextLength
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)
        result = loaded.run()
        assert result.status == JobStatus.COMPLETED
        assert "HELLO" in result.result.summary  # type: ignore[union-attr]

    def test_dict_fan_in_list_ref_invalid_length(self, tmp_path: Path) -> None:
        """List field ref in dict fan-in with != 2 elements raises ConfigLoadError."""
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: bad
  tasks:
    - task: {THIS_MODULE}.WrapText
    - task: {THIS_MODULE}.TextLength
      depends_on: {THIS_MODULE}.WrapText
    - task: {THIS_MODULE}.FieldFanInTask
      depends_on:
        inner:
          - {THIS_MODULE}.WrapText
          - inner
          - extra
        length: {THIS_MODULE}.TextLength
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        with pytest.raises(ConfigLoadError, match="List dep must be"):
            load_workflow_from_yaml(wf_path, in_path)

    def test_dict_fan_in_list_ref_not_found(self, tmp_path: Path) -> None:
        """List field ref in dict fan-in with unknown task raises ConfigLoadError."""
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: bad
  tasks:
    - task: {THIS_MODULE}.WrapText
    - task: {THIS_MODULE}.TextLength
      depends_on: {THIS_MODULE}.WrapText
    - task: {THIS_MODULE}.FieldFanInTask
      depends_on:
        inner:
          - nonexistent.module.Task
          - inner
        length: {THIS_MODULE}.TextLength
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        with pytest.raises(ConfigLoadError, match="not found"):
            load_workflow_from_yaml(wf_path, in_path)

    def test_fan_in_string_dep_not_found(self, tmp_path: Path) -> None:
        """Fan-in with an unknown string dependency raises ConfigLoadError."""
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: bad
  tasks:
    - task: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.CombineResults
      depends_on:
        reversed: nonexistent.module.Task
        length: {THIS_MODULE}.UpperText
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        with pytest.raises(ConfigLoadError, match="not found"):
            load_workflow_from_yaml(wf_path, in_path)

    def test_workflow_validation_failed(self, tmp_path: Path) -> None:
        """Workflow validation error during build() is wrapped in ConfigLoadError."""
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: ambiguous
  tasks:
    - task: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.ReverseText
      depends_on: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.TextLength
      depends_on: {THIS_MODULE}.UpperText
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        with pytest.raises(ConfigLoadError, match="Workflow validation failed"):
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


class TestYamlNamedInstances:
    """Tests for YAML configs with name: field on tasks."""

    def test_named_instances_yaml(self, tmp_path: Path) -> None:
        """YAML with name: field on tasks loads and resolves dependencies correctly."""
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: named_yaml
  result_task: reverse_it
  tasks:
    - task: {THIS_MODULE}.UpperText
      name: upper_1
    - task: {THIS_MODULE}.UpperText
      name: upper_2
    - task: {THIS_MODULE}.ReverseText
      name: reverse_it
      depends_on: upper_1
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)
        result = loaded.run()
        assert result.status == JobStatus.COMPLETED
        # ReverseText reverses the output of upper_1 ("HELLO" -> "OLLEH")
        assert result.result.text == "OLLEH"  # type: ignore[union-attr]

    def test_named_instances_fan_in(self, tmp_path: Path) -> None:
        """Fan-in referencing named instances by name."""
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: named_fan_in
  tasks:
    - task: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.ReverseText
      name: reverse_1
      depends_on: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.TextLength
      name: length_1
      depends_on: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.CombineResults
      depends_on:
        reversed: reverse_1
        length: length_1
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)
        result = loaded.run()
        assert result.status == JobStatus.COMPLETED
        assert "OLLEH" in result.result.summary  # type: ignore[union-attr]

    def test_result_task_not_found_raises(self, tmp_path: Path) -> None:
        """result_task referencing a nonexistent name raises ConfigLoadError."""
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: bad_result
  result_task: nonexistent_task
  tasks:
    - task: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.ReverseText
      depends_on: {THIS_MODULE}.UpperText
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        with pytest.raises(ConfigLoadError, match=r"result_task.*not found"):
            load_workflow_from_yaml(wf_path, in_path)

    def test_named_result_task(self, tmp_path: Path) -> None:
        """result_task references a named instance."""
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: named_result
  result_task: my_upper
  tasks:
    - task: {THIS_MODULE}.UpperText
      name: my_upper
    - task: {THIS_MODULE}.ReverseText
      depends_on: my_upper
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)
        assert loaded.workflow.result_task_name == "my_upper"


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

    def test_get_type_hints_failure_returns_uncoerced(self) -> None:
        """When get_type_hints() raises, params are returned unmodified."""
        from taskekrabbe.hooks.base import BaseHook

        class BadAnnotationHook(BaseHook):
            def __init__(self, x: "NonexistentType") -> None:  # type: ignore[name-defined] # noqa: F821, UP037
                pass

        params = {"x": "/some/path"}
        result = _coerce_hook_params(BadAnnotationHook, params)
        assert result == {"x": "/some/path"}
        assert isinstance(result["x"], str)

    def test_extra_params_error(self, tmp_path: Path) -> None:
        """Extra params should cause TypeError when instantiating the hook."""
        from taskekrabbe.hooks.logging import LoggingHook

        params = _coerce_hook_params(LoggingHook, {"level": 20, "unknown": True})
        with pytest.raises(TypeError):
            LoggingHook(**params)


# ============================================================
# TestPerTaskConfig
# ============================================================


class PerTaskInput(BaseModel):
    """Input for per-task config test: root task taking config values."""

    egrid_path: str
    flag: bool = False


class PerTaskOutput(BaseModel):
    path: str
    flag: bool


class PerTaskRoot(Task[PerTaskInput, PerTaskOutput]):
    name = "per_task_root"

    def run(self, input: PerTaskInput, ctx: ExecutionContext) -> PerTaskOutput:
        return PerTaskOutput(path=input.egrid_path, flag=input.flag)


class DownstreamInput(BaseModel):
    path: str
    flag: bool
    label: str


class DownstreamOutput(BaseModel):
    result: str


class PerTaskDownstream(Task[DownstreamInput, DownstreamOutput]):
    name = "per_task_downstream"

    def run(self, input: DownstreamInput, ctx: ExecutionContext) -> DownstreamOutput:
        return DownstreamOutput(result=f"{input.label}:{input.path}:{input.flag}")


class TestPerTaskConfig:
    """Tests for per-task YAML config format."""

    def test_per_task_format_detection(self, tmp_path: Path) -> None:
        """Input YAML with task-name keys triggers per-task config mode."""
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: per_task_test
  tasks:
    - task: {THIS_MODULE}.PerTaskRoot
""",
        )
        in_path = _write_input_yaml(
            tmp_path,
            """\
per_task_root:
  egrid_path: "/data/test.egrid"
""",
        )
        loaded = load_workflow_from_yaml(wf_path, in_path)
        result = loaded.run()
        assert result.status == JobStatus.COMPLETED
        assert result.result.path == "/data/test.egrid"  # type: ignore[union-attr]

    def test_per_task_with_dag(self, tmp_path: Path) -> None:
        """Per-task config with a DAG workflow merging upstream + config."""
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: per_task_dag
  tasks:
    - task: {THIS_MODULE}.PerTaskRoot
    - task: {THIS_MODULE}.PerTaskDownstream
      depends_on: {THIS_MODULE}.PerTaskRoot
""",
        )
        in_path = _write_input_yaml(
            tmp_path,
            """\
per_task_root:
  egrid_path: "/data/model.egrid"
per_task_downstream:
  label: "my_label"
""",
        )
        loaded = load_workflow_from_yaml(wf_path, in_path)
        result = loaded.run()
        assert result.status == JobStatus.COMPLETED
        assert result.result.result == "my_label:/data/model.egrid:False"  # type: ignore[union-attr]

    def test_flat_config_backward_compat(self, tmp_path: Path) -> None:
        """Flat config format still works when keys don't match task names."""
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: flat_compat
  tasks:
    - task: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.ReverseText
""",
        )
        in_path = _write_input_yaml(tmp_path, "text: hello\n")
        loaded = load_workflow_from_yaml(wf_path, in_path)
        result = loaded.run()
        assert result.status == JobStatus.COMPLETED
        assert result.result.text == "OLLEH"  # type: ignore[union-attr]

    def test_per_task_empty_config(self, tmp_path: Path) -> None:
        """Task with empty config dict ({}) in per-task format works."""
        wf_path = _write_workflow_yaml(
            tmp_path,
            f"""\
workflow:
  name: empty_cfg
  tasks:
    - task: {THIS_MODULE}.PerTaskRoot
""",
        )
        # null values treated same as empty dict
        in_path = _write_input_yaml(
            tmp_path,
            """\
per_task_root:
  egrid_path: "/data/test.egrid"
""",
        )
        loaded = load_workflow_from_yaml(wf_path, in_path)
        result = loaded.run()
        assert result.status == JobStatus.COMPLETED


# ============================================================
# TestWorkflowTaskYaml
# ============================================================


class TestWorkflowTaskYaml:
    """Tests for YAML workflow: references (workflow_task via YAML)."""

    def _write_yaml(self, path: Path, content: str) -> Path:
        path.write_text(content)
        return path

    def test_workflow_ref_basic(self, tmp_path: Path) -> None:
        """Outer YAML references inner YAML via workflow:, end-to-end."""
        self._write_yaml(
            tmp_path / "inner.yaml",
            f"""\
workflow:
  name: inner_pipeline
  tasks:
    - task: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.ReverseText
""",
        )
        outer_path = self._write_yaml(
            tmp_path / "outer.yaml",
            f"""\
workflow:
  name: outer_pipeline
  tasks:
    - workflow: inner.yaml
      name: sub_pipeline
    - task: {THIS_MODULE}.TextLength
      depends_on: sub_pipeline
""",
        )
        in_path = self._write_yaml(tmp_path / "input.yaml", "text: hello\n")
        loaded = load_workflow_from_yaml(outer_path, in_path)
        result = loaded.run()

        assert result.status == JobStatus.COMPLETED
        # inner: "hello" -> "HELLO" -> "OLLEH" (TextOutput)
        # TextLength: len("OLLEH") = 5
        assert result.result.length == 5  # type: ignore[union-attr]

    def test_workflow_ref_with_input(self, tmp_path: Path) -> None:
        """Inner YAML + workflow_input: for config_fields."""
        self._write_yaml(
            tmp_path / "inner.yaml",
            f"""\
workflow:
  name: inner_cfg
  tasks:
    - task: {THIS_MODULE}.UpperText
    - task: {THIS_MODULE}.ReverseText
""",
        )
        self._write_yaml(
            tmp_path / "inner_input.yaml",
            """\
upper_text:
  text: configured_hello
""",
        )
        outer_path = self._write_yaml(
            tmp_path / "outer.yaml",
            f"""\
workflow:
  name: outer_cfg
  tasks:
    - workflow: inner.yaml
      workflow_input: inner_input.yaml
      name: sub_cfg
    - task: {THIS_MODULE}.TextLength
      depends_on: sub_cfg
""",
        )
        # Outer root is sub_cfg (EmptyConfig input), so outer input is empty
        in_path = self._write_yaml(tmp_path / "input.yaml", "{}\n")
        loaded = load_workflow_from_yaml(outer_path, in_path)
        result = loaded.run()

        assert result.status == JobStatus.COMPLETED
        # inner: per-task config "configured_hello" -> "CONFIGURED_HELLO" -> "OLLEH_DERUGIFINOC"
        # TextLength: len("OLLEH_DERUGIFINOC") = 16
        assert result.result.length == 16  # type: ignore[union-attr]

    def test_workflow_ref_path_resolution(self, tmp_path: Path) -> None:
        """Paths resolve relative to outer YAML file's directory."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        self._write_yaml(
            subdir / "inner.yaml",
            f"""\
workflow:
  name: inner
  tasks:
    - task: {THIS_MODULE}.UpperText
""",
        )
        outer_path = self._write_yaml(
            tmp_path / "outer.yaml",
            """\
workflow:
  name: outer
  tasks:
    - workflow: subdir/inner.yaml
      name: sub
""",
        )
        in_path = self._write_yaml(tmp_path / "input.yaml", "text: world\n")
        loaded = load_workflow_from_yaml(outer_path, in_path)
        result = loaded.run()

        assert result.status == JobStatus.COMPLETED
        assert result.result.text == "WORLD"  # type: ignore[union-attr]

    def test_task_and_workflow_both_set_rejected(self) -> None:
        """Validator rejects entry with both task: and workflow:."""
        with pytest.raises(ValidationError, match="Specify either"):
            TaskConfig(task="mod.Task", workflow="inner.yaml")

    def test_neither_task_nor_workflow_rejected(self) -> None:
        """Validator rejects entry with neither task: nor workflow:."""
        with pytest.raises(ValidationError, match="Must specify either"):
            TaskConfig()

    def test_workflow_input_without_workflow_rejected(self) -> None:
        """Validator rejects workflow_input: without workflow:."""
        with pytest.raises(ValidationError, match="'workflow_input' requires 'workflow'"):
            TaskConfig(task="mod.Task", workflow_input="input.yaml")

    def test_inner_workflow_bad_yaml(self, tmp_path: Path) -> None:
        """Inner workflow: reference with bad YAML syntax raises ConfigLoadError."""
        self._write_yaml(tmp_path / "inner.yaml", "tasks: [invalid yaml\n")
        outer_path = self._write_yaml(
            tmp_path / "outer.yaml",
            f"""\
workflow:
  name: outer
  tasks:
    - workflow: inner.yaml
      name: sub
    - task: {THIS_MODULE}.TextLength
      depends_on: sub
""",
        )
        in_path = self._write_yaml(tmp_path / "input.yaml", "text: hello\n")
        with pytest.raises(ConfigLoadError, match="YAML parse error"):
            load_workflow_from_yaml(outer_path, in_path)

    def test_inner_workflow_file_not_found(self, tmp_path: Path) -> None:
        """Inner workflow: reference to nonexistent file raises ConfigLoadError."""
        outer_path = self._write_yaml(
            tmp_path / "outer.yaml",
            f"""\
workflow:
  name: outer
  tasks:
    - workflow: nonexistent.yaml
      name: sub
    - task: {THIS_MODULE}.TextLength
      depends_on: sub
""",
        )
        in_path = self._write_yaml(tmp_path / "input.yaml", "text: hello\n")
        with pytest.raises(ConfigLoadError, match="Cannot read file"):
            load_workflow_from_yaml(outer_path, in_path)

    def test_inner_workflow_not_a_mapping(self, tmp_path: Path) -> None:
        """Inner workflow: YAML that is not a mapping raises ConfigLoadError."""
        self._write_yaml(tmp_path / "inner.yaml", "- item1\n- item2\n")
        outer_path = self._write_yaml(
            tmp_path / "outer.yaml",
            f"""\
workflow:
  name: outer
  tasks:
    - workflow: inner.yaml
      name: sub
    - task: {THIS_MODULE}.TextLength
      depends_on: sub
""",
        )
        in_path = self._write_yaml(tmp_path / "input.yaml", "text: hello\n")
        with pytest.raises(ConfigLoadError, match="must contain a mapping"):
            load_workflow_from_yaml(outer_path, in_path)

    def test_inner_workflow_schema_error(self, tmp_path: Path) -> None:
        """Inner workflow: YAML with schema error raises ConfigLoadError."""
        self._write_yaml(tmp_path / "inner.yaml", "workflow:\n  name: inner\n")
        outer_path = self._write_yaml(
            tmp_path / "outer.yaml",
            f"""\
workflow:
  name: outer
  tasks:
    - workflow: inner.yaml
      name: sub
    - task: {THIS_MODULE}.TextLength
      depends_on: sub
""",
        )
        in_path = self._write_yaml(tmp_path / "input.yaml", "text: hello\n")
        with pytest.raises(ConfigLoadError, match="YAML schema validation error"):
            load_workflow_from_yaml(outer_path, in_path)

    def test_inner_workflow_input_bad_yaml(self, tmp_path: Path) -> None:
        """Inner workflow_input with bad YAML syntax raises ConfigLoadError."""
        self._write_yaml(
            tmp_path / "inner.yaml",
            f"""\
workflow:
  name: inner
  tasks:
    - task: {THIS_MODULE}.UpperText
""",
        )
        self._write_yaml(tmp_path / "inner_input.yaml", "key: [bad yaml\n")
        outer_path = self._write_yaml(
            tmp_path / "outer.yaml",
            """\
workflow:
  name: outer
  tasks:
    - workflow: inner.yaml
      workflow_input: inner_input.yaml
      name: sub
""",
        )
        in_path = self._write_yaml(tmp_path / "input.yaml", "{}\n")
        with pytest.raises(ConfigLoadError, match="Input YAML parse error"):
            load_workflow_from_yaml(outer_path, in_path)

    def test_inner_workflow_input_file_not_found(self, tmp_path: Path) -> None:
        """Inner workflow_input referencing nonexistent file raises ConfigLoadError."""
        self._write_yaml(
            tmp_path / "inner.yaml",
            f"""\
workflow:
  name: inner
  tasks:
    - task: {THIS_MODULE}.UpperText
""",
        )
        outer_path = self._write_yaml(
            tmp_path / "outer.yaml",
            """\
workflow:
  name: outer
  tasks:
    - workflow: inner.yaml
      workflow_input: nonexistent_input.yaml
      name: sub
""",
        )
        in_path = self._write_yaml(tmp_path / "input.yaml", "{}\n")
        with pytest.raises(ConfigLoadError, match="Cannot read input file"):
            load_workflow_from_yaml(outer_path, in_path)

    def test_inner_workflow_input_not_a_mapping(self, tmp_path: Path) -> None:
        """Inner workflow_input that is not a mapping raises ConfigLoadError."""
        self._write_yaml(
            tmp_path / "inner.yaml",
            f"""\
workflow:
  name: inner
  tasks:
    - task: {THIS_MODULE}.UpperText
""",
        )
        self._write_yaml(tmp_path / "inner_input.yaml", "- item1\n- item2\n")
        outer_path = self._write_yaml(
            tmp_path / "outer.yaml",
            """\
workflow:
  name: outer
  tasks:
    - workflow: inner.yaml
      workflow_input: inner_input.yaml
      name: sub
""",
        )
        in_path = self._write_yaml(tmp_path / "input.yaml", "{}\n")
        with pytest.raises(ConfigLoadError, match="Input YAML file must contain a mapping"):
            load_workflow_from_yaml(outer_path, in_path)
