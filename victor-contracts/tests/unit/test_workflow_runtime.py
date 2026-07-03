"""Tests for SDK workflow runtime host adapters."""

import pytest

pytest.importorskip("victor", reason="host runtime adapters require the victor-ai package")

from victor_contracts.workflow_runtime import (
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
