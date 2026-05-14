from victor_contracts.workflows import ExecutorNodeStatus as SDKExecutorNodeStatus
from victor_contracts.workflows import NodeResult as SDKNodeResult
from victor.workflows.executor import ExecutorNodeStatus, NodeResult


def test_workflow_executor_reuses_sdk_result_types() -> None:
    assert ExecutorNodeStatus is SDKExecutorNodeStatus
    assert NodeResult is SDKNodeResult
