"""Tests for Job creation and validation."""

from __future__ import annotations

import pytest

from taskekrabbe import Job, JobStatus, Workflow, WorkflowDefinitionError
from tests.conftest import AddOne, AddOneB, Double, FanInTask, NumberInput, NumberOutput


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
