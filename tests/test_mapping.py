"""Tests for sequential mapped task expansion."""

from __future__ import annotations

import signal
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, Field

from taskmaestro import (
    EmptyConfig,
    ExecutionContext,
    Job,
    JobConfiguration,
    JobStatus,
    MappedOutput,
    MappedTaskExecutionError,
    Runner,
    Task,
    TaskMap,
    Workflow,
    WorkflowDefinitionError,
    collect,
    workflow_task,
)
from taskmaestro.hooks import LoggingHook, ResultPersistenceHook, TimingHook
from taskmaestro.hooks.base import BaseHook
from taskmaestro.yaml_config import ConfigLoadError, load_workflow_from_yaml
from tests.conftest import AddOne, MergeTask, NumberInput, NumberOutput, StringOutput


class MappedInput(BaseModel):
    base: NumberOutput
    item_name: str
    amount: int
    multiplier: int


class MappedNumber(Task[MappedInput, NumberOutput]):
    name = "mapped_number"
    seen: ClassVar[list[tuple[int, str, str]]] = []

    def run(self, input: MappedInput, ctx: ExecutionContext) -> NumberOutput:
        self.seen.append((id(self), input.item_name, ctx.correlation_id))
        if input.amount < 0:
            raise ValueError(f"negative amount for {input.item_name}")
        return NumberOutput(value=input.base.value + input.amount * input.multiplier)


class EnvelopeOutput(BaseModel):
    number: NumberOutput


class ProduceEnvelope(Task[NumberInput, EnvelopeOutput]):
    name = "produce_envelope_for_map"

    def run(self, input: NumberInput, ctx: ExecutionContext) -> EnvelopeOutput:
        return EnvelopeOutput(number=NumberOutput(value=input.value))


class MappedCollectionInput(BaseModel):
    bases: list[NumberOutput]
    item_name: str
    amount: int


class MappedCollection(Task[MappedCollectionInput, NumberOutput]):
    name = "mapped_collection"

    def run(self, input: MappedCollectionInput, ctx: ExecutionContext) -> NumberOutput:
        return NumberOutput(value=sum(item.value for item in input.bases) + input.amount)


class MappedOnlyInput(BaseModel):
    item_name: str
    amount: int


class MappedOnly(Task[MappedOnlyInput, NumberOutput]):
    name = "mapped_only"

    def run(self, input: MappedOnlyInput, ctx: ExecutionContext) -> NumberOutput:
        return NumberOutput(value=input.amount)


class MappedWrongOutput(Task[MappedOnlyInput, NumberOutput]):
    name = "mapped_wrong_output"

    def run(self, input: MappedOnlyInput, ctx: ExecutionContext) -> NumberOutput:
        return StringOutput(text="wrong")  # type: ignore[return-value]


class MappedSlow(Task[MappedOnlyInput, NumberOutput]):
    name = "mapped_slow"
    timeout_seconds = 10

    def run(self, input: MappedOnlyInput, ctx: ExecutionContext) -> NumberOutput:
        return NumberOutput(value=input.amount)


class AggregateInput(BaseModel):
    values: dict[str, NumberOutput]


class SumAggregate(Task[AggregateInput, NumberOutput]):
    name = "sum_aggregate"

    def run(self, input: AggregateInput, ctx: ExecutionContext) -> NumberOutput:
        return NumberOutput(value=sum(value.value for value in input.values.values()))


def _mapped_workflow(*, error_mode: str = "fail_fast") -> Workflow:
    return (
        Workflow.builder("mapped", result_task=SumAggregate)
        .add_task(AddOne)
        .add_task(
            MappedNumber,
            depends_on={"base": AddOne},
            config_fields=["multiplier"],
            mapped_over=TaskMap(
                over="items",
                key_as="item_name",
                value_as="amount",
                error_mode=error_mode,  # type: ignore[arg-type]
            ),
        )
        .add_task(SumAggregate, depends_on={"values": (MappedNumber, "root")})
        .build()
    )


def _mapped_job(workflow: Workflow, items: dict[Any, Any]) -> Job[NumberInput]:
    return Job(
        workflow,
        NumberInput(value=10),
        job_configuration=JobConfiguration({"mapped_number": {"items": items, "multiplier": 2}}),
    )


class RecordingMapHook(BaseHook):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_map_item_start(self, job: Job[Any], task: Task[Any, Any], key: str) -> None:
        self.events.append(f"start:{task.name}[{key}]")

    def on_map_item_complete(
        self, job: Job[Any], task: Task[Any, Any], key: str, output: object
    ) -> None:
        self.events.append(f"complete:{task.name}[{key}]")

    def on_map_item_fail(
        self, job: Job[Any], task: Task[Any, Any], key: str, error: Exception
    ) -> None:
        self.events.append(f"fail:{task.name}[{key}]")


class TestTaskMap:
    @pytest.mark.parametrize("field", ["over", "key_as", "value_as"])
    def test_fields_must_not_be_empty(self, field: str) -> None:
        values = {"over": "items", "key_as": "item_name", "value_as": "amount"}
        values[field] = ""
        with pytest.raises(ValueError, match=f"TaskMap.{field}"):
            TaskMap(**values)  # type: ignore[arg-type]

    def test_injected_fields_must_differ(self) -> None:
        with pytest.raises(ValueError, match="must be different"):
            TaskMap(over="items", key_as="item", value_as="item")

    def test_error_mode_is_validated_at_runtime(self) -> None:
        with pytest.raises(ValueError, match="error_mode"):
            TaskMap(
                over="items",
                key_as="key",
                value_as="value",
                error_mode="invalid",  # type: ignore[arg-type]
            )


class TestMappedWorkflowValidation:
    def test_mapping_metadata_and_effective_output(self) -> None:
        workflow = _mapped_workflow()

        assert workflow.is_mapped_task("mapped_number")
        assert workflow.get_task_map("mapped_number") is not None
        assert workflow.get_task_map("add_one") is None
        assert workflow.get_output_annotation("mapped_number") == MappedOutput[NumberOutput]
        assert workflow.get_output_annotation("add_one") is NumberOutput

    @pytest.mark.parametrize("key_as,value_as", [("missing", "amount"), ("item_name", "missing")])
    def test_map_fields_must_exist(self, key_as: str, value_as: str) -> None:
        with pytest.raises(WorkflowDefinitionError, match="Map field 'missing'"):
            (
                Workflow.builder("bad")
                .add_task(
                    MappedOnly,
                    mapped_over=TaskMap(over="items", key_as=key_as, value_as=value_as),
                )
                .build()
            )

    def test_key_field_must_accept_strings(self) -> None:
        class NumericKeyInput(BaseModel):
            key: int
            amount: int

        class NumericKeyTask(Task[NumericKeyInput, NumberOutput]):
            def run(self, input: NumericKeyInput, ctx: ExecutionContext) -> NumberOutput:
                return NumberOutput(value=input.amount)

        with pytest.raises(WorkflowDefinitionError, match="must accept strings"):
            (
                Workflow.builder("bad")
                .add_task(
                    NumericKeyTask,
                    mapped_over=TaskMap(over="items", key_as="key", value_as="amount"),
                )
                .build()
            )

    def test_map_fields_cannot_be_config_fields(self) -> None:
        with pytest.raises(WorkflowDefinitionError, match="cannot also be config_fields"):
            (
                Workflow.builder("bad")
                .add_task(
                    MappedOnly,
                    config_fields=["amount"],
                    mapped_over=TaskMap(over="items", key_as="item_name", value_as="amount"),
                )
                .build()
            )

    def test_map_fields_cannot_be_dependencies(self) -> None:
        with pytest.raises(WorkflowDefinitionError, match="cannot also be dependencies"):
            (
                Workflow.builder("bad")
                .add_task(AddOne)
                .add_task(
                    MappedNumber,
                    depends_on={"amount": AddOne, "base": AddOne},
                    config_fields=["multiplier"],
                    mapped_over=TaskMap(over="items", key_as="item_name", value_as="amount"),
                )
                .build()
            )

    def test_all_required_fields_must_be_covered(self) -> None:
        with pytest.raises(WorkflowDefinitionError, match=r"multiplier.*not covered"):
            (
                Workflow.builder("bad")
                .add_task(AddOne)
                .add_task(
                    MappedNumber,
                    depends_on={"base": AddOne},
                    mapped_over=TaskMap(over="items", key_as="item_name", value_as="amount"),
                )
                .build()
            )

    def test_mapped_task_requires_named_dependencies(self) -> None:
        with pytest.raises(WorkflowDefinitionError, match="requires named field dependencies"):
            (
                Workflow.builder("bad")
                .add_task(AddOne)
                .add_task(
                    MappedNumber,
                    depends_on=AddOne,
                    config_fields=["multiplier"],
                    mapped_over=TaskMap(over="items", key_as="item_name", value_as="amount"),
                )
                .build()
            )

    def test_mapped_output_type_is_checked_downstream(self) -> None:
        class BadAggregateInput(BaseModel):
            values: dict[str, StringOutput]

        class BadAggregate(Task[BadAggregateInput, NumberOutput]):
            def run(self, input: BadAggregateInput, ctx: ExecutionContext) -> NumberOutput:
                return NumberOutput(value=0)

        with pytest.raises(WorkflowDefinitionError, match="Fan-in type mismatch"):
            (
                Workflow.builder("bad")
                .add_task(
                    MappedOnly,
                    mapped_over=TaskMap(over="items", key_as="item_name", value_as="amount"),
                )
                .add_task(BadAggregate, depends_on={"values": (MappedOnly, "root")})
                .build()
            )

    def test_can_route_root_dictionary_from_mapped_output(self) -> None:
        workflow = (
            Workflow.builder("mapped_root_route")
            .add_task(
                MappedOnly,
                mapped_over=TaskMap(over="items", key_as="item_name", value_as="amount"),
            )
            .add_task(SumAggregate, depends_on={"values": (MappedOnly, "root")})
            .build()
        )
        assert workflow.result_task is SumAggregate

    def test_mapped_upstream_with_single_dependency_and_config_is_rejected(self) -> None:
        with pytest.raises(WorkflowDefinitionError, match="must be connected through"):
            (
                Workflow.builder("bad", result_task=MergeTask)
                .add_task(
                    MappedOnly,
                    mapped_over=TaskMap(over="items", key_as="item_name", value_as="amount"),
                )
                .add_task(MergeTask, depends_on=MappedOnly, config_fields=["label"])
                .build()
            )

    def test_unknown_mapped_output_field_is_rejected(self) -> None:
        with pytest.raises(WorkflowDefinitionError, match="Field 'value' not found"):
            (
                Workflow.builder("bad")
                .add_task(
                    MappedOnly,
                    mapped_over=TaskMap(over="items", key_as="item_name", value_as="amount"),
                )
                .add_task(
                    SumAggregate,
                    depends_on={"values": (MappedOnly, "value")},
                )
                .build()
            )

    def test_collection_can_route_root_from_mapped_output(self) -> None:
        class NestedAggregateInput(BaseModel):
            values: list[dict[str, NumberOutput]]

        class NestedAggregate(Task[NestedAggregateInput, NumberOutput]):
            def run(self, input: NestedAggregateInput, ctx: ExecutionContext) -> NumberOutput:
                return NumberOutput(value=0)

        workflow = (
            Workflow.builder("mapped_collection_route")
            .add_task(
                MappedOnly,
                mapped_over=TaskMap(over="items", key_as="item_name", value_as="amount"),
            )
            .add_task(
                NestedAggregate,
                depends_on={"values": collect((MappedOnly, "root"))},
            )
            .build()
        )
        assert workflow.result_task is NestedAggregate

    def test_mapped_result_workflow_can_be_wrapped(self) -> None:
        workflow = (
            Workflow.builder("mapped_result")
            .add_task(
                MappedOnly,
                mapped_over=TaskMap(over="items", key_as="item_name", value_as="amount"),
            )
            .build()
        )

        Wrapped = workflow_task(
            workflow,
            job_configuration=JobConfiguration({"mapped_only": {"items": {"one": 1}}}),
        )
        outer = Workflow("outer", [Wrapped])

        result = Runner().run(Job(outer, EmptyConfig()))

        assert result.status == JobStatus.COMPLETED
        assert result.result == MappedOutput[NumberOutput](root={"one": NumberOutput(value=1)})


class TestMappedJobValidation:
    def test_job_configuration_is_required(self) -> None:
        workflow = _mapped_workflow()
        with pytest.raises(WorkflowDefinitionError, match="requires JobConfiguration"):
            Job(workflow, NumberInput(value=1))

    def test_map_source_is_required(self) -> None:
        workflow = _mapped_workflow()
        config = JobConfiguration({"mapped_number": {"multiplier": 2}})
        with pytest.raises(WorkflowDefinitionError, match="requires configuration field 'items'"):
            Job(workflow, NumberInput(value=1), job_configuration=config)

    def test_map_source_must_be_mapping(self) -> None:
        workflow = _mapped_workflow()
        config = JobConfiguration({"mapped_number": {"items": [1, 2], "multiplier": 2}})
        with pytest.raises(WorkflowDefinitionError, match="must be a mapping"):
            Job(workflow, NumberInput(value=1), job_configuration=config)

    def test_map_keys_must_be_strings(self) -> None:
        workflow = _mapped_workflow()
        with pytest.raises(WorkflowDefinitionError, match="must be strings"):
            _mapped_job(workflow, {1: 2})

    def test_map_values_are_validated(self) -> None:
        workflow = _mapped_workflow()
        with pytest.raises(WorkflowDefinitionError, match="Invalid mapping item 'bad'"):
            _mapped_job(workflow, {"bad": "not-an-int"})

    def test_map_values_preserve_model_config_and_nested_models(self) -> None:
        class Resource:
            pass

        class ResourceInput(BaseModel):
            model_config = ConfigDict(arbitrary_types_allowed=True)
            key: str
            value: Resource | NumberOutput

        seen: list[Resource | NumberOutput] = []

        class ResourceTask(Task[ResourceInput, NumberOutput]):
            def run(self, input: ResourceInput, ctx: ExecutionContext) -> NumberOutput:
                seen.append(input.value)
                return NumberOutput(value=1)

        resource = Resource()
        workflow = (
            Workflow.builder("resources")
            .add_task(ResourceTask, mapped_over=TaskMap("items", "key", "value"))
            .build()
        )
        job = Job(
            workflow,
            EmptyConfig(),
            job_configuration=JobConfiguration(
                {"ResourceTask": {"items": {"object": resource, "model": {"value": 3}}}}
            ),
        )

        assert Runner().run(job).status == JobStatus.COMPLETED
        assert seen == [resource, NumberOutput(value=3)]

    def test_map_values_preserve_field_constraints(self) -> None:
        class PositiveInput(BaseModel):
            key: str
            value: int = Field(gt=0)

        class PositiveTask(Task[PositiveInput, NumberOutput]):
            def run(self, input: PositiveInput, ctx: ExecutionContext) -> NumberOutput:
                return NumberOutput(value=input.value)

        workflow = (
            Workflow.builder("positive")
            .add_task(PositiveTask, mapped_over=TaskMap("items", "key", "value"))
            .build()
        )
        with pytest.raises(WorkflowDefinitionError, match="Invalid mapping item 'bad'"):
            Job(
                workflow,
                EmptyConfig(),
                job_configuration=JobConfiguration({"PositiveTask": {"items": {"bad": -1}}}),
            )


class TestMappedExecution:
    def setup_method(self) -> None:
        MappedNumber.seen = []

    def test_only_map_source_is_stripped_from_config_values(self) -> None:
        """Mapped items receive every configured value except the map source."""
        seen: list[dict[str, Any]] = []

        class OpenInput(BaseModel):
            model_config = ConfigDict(extra="allow")
            item_name: str
            amount: int

        class OpenMapped(Task[OpenInput, NumberOutput]):
            name = "open_mapped"

            def run(self, input: OpenInput, ctx: ExecutionContext) -> NumberOutput:
                seen.append(input.model_extra or {})
                return NumberOutput(value=input.amount)

        workflow = (
            Workflow.builder("open")
            .add_task(
                OpenMapped,
                mapped_over=TaskMap(over="items", key_as="item_name", value_as="amount"),
            )
            .build()
        )
        job = Job(
            workflow,
            EmptyConfig(),
            job_configuration=JobConfiguration(
                {"open_mapped": {"items": {"only": 1}, "passthrough": "yes"}}
            ),
        )

        result = Runner().run(job)

        assert result.status == JobStatus.COMPLETED
        assert seen == [{"passthrough": "yes"}]

    def test_executes_sequentially_and_aggregates_output(self) -> None:
        workflow = _mapped_workflow()
        job = _mapped_job(workflow, {"first": 1, "second": 2, "third": 3})
        hook = RecordingMapHook()

        result = Runner(hooks=[hook]).run(job)

        assert result.status == JobStatus.COMPLETED
        assert result.result == NumberOutput(value=45)
        mapped_result = next(r for r in result.task_results if r.task_name == "mapped_number")
        assert isinstance(mapped_result.output, MappedOutput)
        assert list(mapped_result.output.root) == ["first", "second", "third"]
        assert [name for _instance, name, _ctx in MappedNumber.seen] == [
            "first",
            "second",
            "third",
        ]
        assert len({instance for instance, _name, _ctx in MappedNumber.seen}) == 3
        assert hook.events == [
            "start:mapped_number[first]",
            "complete:mapped_number[first]",
            "start:mapped_number[second]",
            "complete:mapped_number[second]",
            "start:mapped_number[third]",
            "complete:mapped_number[third]",
        ]
        assert [r.task_name for r in result.mapped_item_results["mapped_number"]] == [
            "mapped_number[first]",
            "mapped_number[second]",
            "mapped_number[third]",
        ]
        correlation_ids = [ctx_id for _instance, _name, ctx_id in MappedNumber.seen]
        assert len(set(correlation_ids)) == 3
        assert all(
            ctx_id.startswith(job.task_results[0].task_name) is False for ctx_id in correlation_ids
        )

    def test_routed_and_collection_dependencies_are_shared_by_items(self) -> None:
        routed_workflow = (
            Workflow.builder("routed_map")
            .add_task(ProduceEnvelope)
            .add_task(
                MappedNumber,
                depends_on={"base": (ProduceEnvelope, "number")},
                config_fields=["multiplier"],
                mapped_over=TaskMap(over="items", key_as="item_name", value_as="amount"),
            )
            .build()
        )
        routed_job = Job(
            routed_workflow,
            NumberInput(value=5),
            job_configuration=JobConfiguration(
                {"mapped_number": {"items": {"one": 2}, "multiplier": 3}}
            ),
        )
        assert Runner().run(routed_job).result == MappedOutput[NumberOutput](
            root={"one": NumberOutput(value=11)}
        )

        collection_workflow = (
            Workflow.builder("collection_map")
            .add_task(AddOne, name="first")
            .add_task(AddOne, name="second")
            .add_task(
                MappedCollection,
                depends_on={"bases": collect("first", "second")},
                mapped_over=TaskMap(over="items", key_as="item_name", value_as="amount"),
            )
            .build()
        )
        collection_job = Job(
            collection_workflow,
            NumberInput(value=4),
            job_configuration=JobConfiguration({"mapped_collection": {"items": {"one": 1}}}),
        )
        assert Runner().run(collection_job).result == MappedOutput[NumberOutput](
            root={"one": NumberOutput(value=11)}
        )

    def test_empty_mapping_produces_empty_dictionary(self) -> None:
        workflow = (
            Workflow.builder("empty_map")
            .add_task(
                MappedOnly,
                mapped_over=TaskMap(over="items", key_as="item_name", value_as="amount"),
            )
            .build()
        )
        job = Job(
            workflow,
            EmptyConfig(),
            job_configuration=JobConfiguration({"mapped_only": {"items": {}}}),
        )

        result = Runner().run(job)

        assert result.status == JobStatus.COMPLETED
        assert result.result == MappedOutput[NumberOutput](root={})
        assert result.mapped_item_results["mapped_only"] == []

    def test_fail_fast_stops_after_first_failure(self) -> None:
        workflow = _mapped_workflow()
        job = _mapped_job(workflow, {"good": 1, "bad": -1, "later": 3})
        hook = RecordingMapHook()

        result = Runner(hooks=[hook]).run(job)

        assert result.status == JobStatus.FAILED
        assert result.failed_task == "mapped_number"
        assert "bad" in (result.error or "")
        assert [r.task_name for r in result.mapped_item_results["mapped_number"]] == [
            "mapped_number[good]",
            "mapped_number[bad]",
        ]
        assert "start:mapped_number[later]" not in hook.events

    def test_collect_all_records_every_failure(self) -> None:
        workflow = _mapped_workflow(error_mode="collect_all")
        job = _mapped_job(workflow, {"bad_one": -1, "good": 2, "bad_two": -2})

        result = Runner().run(job)

        assert result.status == JobStatus.FAILED
        assert "bad_one" in (result.error or "")
        assert "bad_two" in (result.error or "")
        assert len(result.mapped_item_results["mapped_number"]) == 3

    @pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="SIGALRM unavailable")
    @pytest.mark.parametrize("job_timeout", [True, False])
    def test_collect_all_stops_only_for_job_timeout(self, job_timeout: bool) -> None:
        seen: list[str] = []

        class AlarmTask(Task[MappedOnlyInput, NumberOutput]):
            timeout_seconds = None if job_timeout else 60

            def run(self, input: MappedOnlyInput, ctx: ExecutionContext) -> NumberOutput:
                seen.append(input.item_name)
                if input.item_name == "first":
                    # Exercise the installed handler without waiting for a real deadline.
                    signal.raise_signal(signal.SIGALRM)
                return NumberOutput(value=input.amount)

        workflow = (
            Workflow.builder("timeout")
            .add_task(
                AlarmTask,
                mapped_over=TaskMap("items", "item_name", "amount", "collect_all"),
            )
            .build()
        )
        job = Job(
            workflow,
            EmptyConfig(),
            job_configuration=JobConfiguration(
                {"AlarmTask": {"items": {"first": 1, "second": 2}}}
            ),
        )
        previous_handler = signal.getsignal(signal.SIGALRM)
        try:
            result = Runner().run(job, timeout_seconds=60 if job_timeout else None)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)

        assert result.status == JobStatus.FAILED
        assert "timed out" in (result.error or "")
        assert seen == (["first"] if job_timeout else ["first", "second"])
        assert len(result.mapped_item_results["AlarmTask"]) == len(seen)

    def test_item_input_validation_is_recorded_as_item_failure(self) -> None:
        workflow = _mapped_workflow()
        job = Job(
            workflow,
            NumberInput(value=1),
            job_configuration=JobConfiguration({"mapped_number": {"items": {"one": 1}}}),
        )
        hook = RecordingMapHook()

        result = Runner(hooks=[hook]).run(job)

        assert result.status == JobStatus.FAILED
        assert result.mapped_item_results["mapped_number"][0].status.value == "failed"
        assert hook.events == [
            "start:mapped_number[one]",
            "fail:mapped_number[one]",
        ]

    def test_wrong_item_output_fails_mapped_task(self) -> None:
        workflow = (
            Workflow.builder("wrong_output")
            .add_task(
                MappedWrongOutput,
                mapped_over=TaskMap(over="items", key_as="item_name", value_as="amount"),
            )
            .build()
        )
        job = Job(
            workflow,
            EmptyConfig(),
            job_configuration=JobConfiguration({"mapped_wrong_output": {"items": {"one": 1}}}),
        )

        result = Runner().run(job)

        assert result.status == JobStatus.FAILED
        assert "expected NumberOutput" in (result.error or "")

    def test_item_timeout_setup_and_cleanup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        workflow = (
            Workflow.builder("mapped_timeout")
            .add_task(
                MappedSlow,
                mapped_over=TaskMap(over="items", key_as="item_name", value_as="amount"),
            )
            .build()
        )
        job = Job(
            workflow,
            EmptyConfig(),
            job_configuration=JobConfiguration({"mapped_slow": {"items": {"one": 1}}}),
        )
        calls: list[tuple[float, str]] = []

        def fake_alarm(seconds: float, label: str, **_kwargs: object) -> bool:
            calls.append((seconds, label))
            return True

        monkeypatch.setattr(Runner, "_set_alarm", staticmethod(fake_alarm))
        result = Runner().run(job)

        assert result.status == JobStatus.COMPLETED
        assert calls == [(10, "mapped_slow[one]")]

    def test_child_context_shares_services_and_has_safe_unique_paths(self, tmp_path: Path) -> None:
        parent = ExecutionContext(correlation_id="parent", scratch_dir=tmp_path)
        service = object()
        parent.register("service", service)

        first = parent.child(task_name="load surfaces", item_key="a/b")
        second = parent.child(task_name="load surfaces", item_key="a_b")

        assert first.parent_correlation_id == "parent"
        assert first.resolve("service") is service
        assert first.logger is parent.logger
        assert first.scratch_dir != second.scratch_dir
        assert first.correlation_id.startswith("parent:load_surfaces_a_b:")

    def test_mapped_exception_retains_errors(self) -> None:
        error = ValueError("bad")
        exc = MappedTaskExecutionError("mapped", {"item": error})
        assert exc.errors == {"item": error}
        assert str(exc) == "Mapped task 'mapped' failed: item: bad"


class TestMappedHooks:
    def test_base_hook_handles_mapped_completion_and_failure(self) -> None:
        success = _mapped_job(_mapped_workflow(), {"good": 1})
        failure = _mapped_job(_mapped_workflow(), {"bad": -1})

        assert Runner(hooks=[BaseHook()]).run(success).status == JobStatus.COMPLETED
        assert Runner(hooks=[BaseHook()]).run(failure).status == JobStatus.FAILED

    def test_builtin_hooks_record_and_persist_items(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        workflow = _mapped_workflow()
        job = _mapped_job(workflow, {"one/unsafe": 1})
        timing = TimingHook()
        persistence = ResultPersistenceHook(tmp_path)

        with caplog.at_level("INFO", logger="taskmaestro.hooks.logging"):
            result = Runner(hooks=[LoggingHook(), timing, persistence]).run(job)

        assert result.status == JobStatus.COMPLETED
        assert "one/unsafe" in timing.mapped_item_timings["mapped_number"]
        assert (tmp_path / "mapped_number[one%2Funsafe].json").exists()
        assert (tmp_path / "mapped_number.json").exists()
        assert any("Map item started: mapped_number[one/unsafe]" in m for m in caplog.messages)
        assert any("Map item completed: mapped_number[one/unsafe]" in m for m in caplog.messages)

    def test_persisted_items_do_not_collide_even_when_parent_fails(self, tmp_path: Path) -> None:
        job = _mapped_job(
            _mapped_workflow(), {"a/b": 1, "a\\b": 2, "a_b": 3, "a%2Fb": 4, "bad": -1}
        )

        result = Runner(hooks=[ResultPersistenceHook(tmp_path)]).run(job)

        assert result.status == JobStatus.FAILED
        assert not (tmp_path / "mapped_number.json").exists()
        for filename_key, value in [("a%2Fb", 13), ("a%5Cb", 15), ("a_b", 17), ("a%252Fb", 19)]:
            path = tmp_path / f"mapped_number[{filename_key}].json"
            assert NumberOutput.model_validate_json(path.read_text()) == NumberOutput(value=value)

    def test_builtin_hooks_record_item_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        workflow = _mapped_workflow()
        job = _mapped_job(workflow, {"bad": -1})
        timing = TimingHook()

        with caplog.at_level("INFO", logger="taskmaestro.hooks.logging"):
            Runner(hooks=[LoggingHook(), timing]).run(job)

        assert "bad" in timing.mapped_item_timings["mapped_number"]
        assert any("Map item failed: mapped_number[bad]" in m for m in caplog.messages)


class TestMappedYamlAndVisualization:
    def test_yaml_mapped_task_end_to_end(self, tmp_path: Path) -> None:
        module = "tests.test_mapping"
        workflow_path = tmp_path / "workflow.yaml"
        workflow_path.write_text(
            f"""\
workflow:
  name: yaml_map
  tasks:
    - task: {module}.MappedOnly
      map:
        over: items
        key_as: item_name
        value_as: amount
        error_mode: fail_fast
"""
        )
        input_path = tmp_path / "input.yaml"
        input_path.write_text(
            """\
mapped_only:
  items:
    first: 1
    second: 2
"""
        )

        loaded = load_workflow_from_yaml(workflow_path, input_path)
        result = loaded.run()

        assert loaded.workflow.get_config_fields("mapped_only") == set()
        assert result.status == JobStatus.COMPLETED
        assert result.result == MappedOutput[NumberOutput](
            root={
                "first": NumberOutput(value=1),
                "second": NumberOutput(value=2),
            }
        )

    def test_yaml_rejects_invalid_error_mode(self, tmp_path: Path) -> None:
        module = "tests.test_mapping"
        workflow_path = tmp_path / "workflow.yaml"
        workflow_path.write_text(
            f"""\
workflow:
  name: bad_yaml_map
  tasks:
    - task: {module}.MappedOnly
      map:
        over: items
        key_as: item_name
        value_as: amount
        error_mode: invalid
"""
        )
        input_path = tmp_path / "input.yaml"
        input_path.write_text("mapped_only: {items: {}}\n")

        with pytest.raises(ConfigLoadError, match="YAML schema validation error"):
            load_workflow_from_yaml(workflow_path, input_path)

    def test_mermaid_marks_mapped_node_and_output_type(self) -> None:
        workflow = _mapped_workflow()

        diagram = workflow.to_mermaid(
            job_configuration=JobConfiguration(
                {"mapped_number": {"items": {"one": 1}, "multiplier": 2}}
            )
        )

        assert 'mapped_number["mapped_number<br/>map over: items"]' in diagram
        assert "items, multiplier" in diagram
        assert ".root: dict&lsaquo;str, NumberOutput&rsaquo;" in diagram
