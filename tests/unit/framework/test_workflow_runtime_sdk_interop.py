"""Interop tests for SDK workflow runtime host adapters."""

from victor.framework.workflows import BaseYAMLWorkflowProvider as CoreBaseYAMLWorkflowProvider
from victor.workflows.definition import (
    WorkflowBuilder as CoreWorkflowBuilder,
    WorkflowDefinition as CoreWorkflowDefinition,
    workflow as core_workflow,
)
from victor_sdk.workflow_runtime import (
    BaseYAMLWorkflowProvider as SdkBaseYAMLWorkflowProvider,
    WorkflowBuilder as SdkWorkflowBuilder,
    WorkflowDefinition as SdkWorkflowDefinition,
    workflow as sdk_workflow,
)


def test_workflow_runtime_identity_is_shared() -> None:
    assert CoreBaseYAMLWorkflowProvider is SdkBaseYAMLWorkflowProvider
    assert CoreWorkflowBuilder is SdkWorkflowBuilder
    assert CoreWorkflowDefinition is SdkWorkflowDefinition
    assert core_workflow is sdk_workflow
