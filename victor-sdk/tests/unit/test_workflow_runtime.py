"""Tests for SDK workflow runtime host adapters."""

from victor_sdk.workflow_runtime import (
    BaseYAMLWorkflowProvider,
    WorkflowBuilder,
    WorkflowDefinition,
    workflow,
)


def test_workflow_runtime_exports_host_types() -> None:
    assert BaseYAMLWorkflowProvider.__name__ == "BaseYAMLWorkflowProvider"
    assert WorkflowBuilder.__name__ == "WorkflowBuilder"
    assert WorkflowDefinition.__name__ == "WorkflowDefinition"
    assert callable(workflow)
