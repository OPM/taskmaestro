"""Tests for Job creation and validation."""

from __future__ import annotations

import pytest

from taskmaestro import (
    EmptyConfig,
    Job,
    JobConfiguration,
    JobStatus,
    Workflow,
    WorkflowDefinitionError,
)
from tests.conftest import (
    AddOne,
    AddOneB,
    ConfigOnlyTask,
    Double,
    FanInTask,
    NumberInput,
    NumberOutput,
)


class TestJobCreation:
    def test_valid_creation(self) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double])
        job = Job(workflow=wf, config=NumberInput(value=5))
        assert job.status == JobStatus.PENDING
        assert job.config.value == 5

    def test_initial_state(self) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double])
        job = Job(workflow=wf, config=NumberInput(value=1))
        assert job.result is None
        assert job.error is None
        assert job.failed_task is None
        assert job.started_at is None
        assert job.completed_at is None
        assert job.task_results == []

    def test_root_task_type_mismatch(self) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double])
        with pytest.raises(WorkflowDefinitionError, match="Root task"):
            Job(workflow=wf, config=NumberOutput(value=5))

    def test_multiple_root_tasks_valid(self) -> None:
        wf = (
            Workflow.builder(name="multi_root")
            .add_task(AddOne)
            .add_task(AddOneB)
            .add_task(FanInTask, depends_on={"a": AddOne, "b": AddOneB})
            .build()
        )
        job = Job(workflow=wf, config=NumberInput(value=3))
        assert job.status == JobStatus.PENDING

    def test_multiple_root_tasks_type_mismatch(self) -> None:
        wf = (
            Workflow.builder(name="multi_root")
            .add_task(AddOne)
            .add_task(AddOneB)
            .add_task(FanInTask, depends_on={"a": AddOne, "b": AddOneB})
            .build()
        )
        with pytest.raises(WorkflowDefinitionError, match="Root task"):
            Job(workflow=wf, config=NumberOutput(value=3))

    def test_config_retained(self) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double])
        config = NumberInput(value=42)
        job = Job(workflow=wf, config=config)
        assert job.config is config


class TestJobConfiguration:
    def test_creation(self) -> None:
        jc = JobConfiguration({"task_a": {"x": 1, "y": "hello"}})
        assert jc.get_config_for_task("task_a") == {"x": 1, "y": "hello"}

    def test_configured_tasks(self) -> None:
        jc = JobConfiguration({"a": {"x": 1}, "b": {"y": 2}})
        assert jc.configured_tasks() == {"a", "b"}

    def test_config_fields_for_task(self) -> None:
        jc = JobConfiguration({"a": {"x": 1, "y": 2}})
        assert jc.config_fields_for_task("a") == {"x", "y"}

    def test_missing_task_returns_empty(self) -> None:
        jc = JobConfiguration({"a": {"x": 1}})
        assert jc.get_config_for_task("missing") == {}
        assert jc.config_fields_for_task("missing") == set()

    def test_get_config_returns_copy(self) -> None:
        jc = JobConfiguration({"a": {"x": 1}})
        d = jc.get_config_for_task("a")
        d["x"] = 999
        assert jc.get_config_for_task("a") == {"x": 1}


class TestEmptyConfig:
    def test_empty_config_is_basemodel(self) -> None:
        ec = EmptyConfig()
        assert ec.model_dump() == {}


class TestJobWithConfiguration:
    def test_root_task_with_config_fields_skips_validation(self) -> None:
        """Root tasks with config_fields should not be validated against job.config."""
        wf = (
            Workflow.builder(name="cfg")
            .add_task(ConfigOnlyTask, config_fields=["path", "count"])
            .build()
        )
        jc = JobConfiguration({"config_only_task": {"path": "/tmp", "count": 3}})
        job = Job(workflow=wf, config=EmptyConfig(), job_configuration=jc)
        assert job.status == JobStatus.PENDING

    def test_job_configuration_stored(self) -> None:
        wf = (
            Workflow.builder(name="cfg")
            .add_task(ConfigOnlyTask, config_fields=["path", "count"])
            .build()
        )
        jc = JobConfiguration({"config_only_task": {"path": "/tmp", "count": 3}})
        job = Job(workflow=wf, config=EmptyConfig(), job_configuration=jc)
        assert job.job_configuration is jc

    def test_no_job_configuration_default(self) -> None:
        wf = Workflow(name="test", tasks=[AddOne, Double])
        job = Job(workflow=wf, config=NumberInput(value=5))
        assert job.job_configuration is None
