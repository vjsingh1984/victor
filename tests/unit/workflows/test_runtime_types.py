from victor.workflows import (
    CompiledGraphNodeResult,
    WorkflowState as PublicWorkflowState,
    WorkflowNodeExecutorRegistration as PublicWorkflowNodeExecutorRegistration,
    WorkflowNodeExecutorRegistry as PublicWorkflowNodeExecutorRegistry,
    create_initial_workflow_state as public_create_initial_workflow_state,
    clear_registered_workflow_node_executors as public_clear_registered_workflow_node_executors,
    get_workflow_node_executor_registry as public_get_workflow_node_executor_registry,
    register_workflow_node_executor as public_register_workflow_node_executor,
)
from victor.workflows.executors import (
    WorkflowNodeExecutorRegistration as ExecutorWorkflowNodeExecutorRegistration,
    WorkflowNodeExecutorRegistry as ExecutorWorkflowNodeExecutorRegistry,
    clear_registered_workflow_node_executors as executor_clear_registered_workflow_node_executors,
    get_workflow_node_executor_registry as executor_get_workflow_node_executor_registry,
    register_workflow_node_executor as executor_register_workflow_node_executor,
)
from victor.workflows.executors.registry import (
    WorkflowNodeExecutorRegistration,
    WorkflowNodeExecutorRegistry,
    clear_registered_workflow_node_executors,
    get_workflow_node_executor_registry,
    register_workflow_node_executor,
)
from victor.workflows.runtime_types import GraphNodeResult, WorkflowState
from victor.workflows.runtime_types import create_initial_workflow_state
from victor.workflows.unified_compiler import NodeExecutionResult as UnifiedNodeExecutionResult
from victor.workflows.yaml_to_graph_compiler import (
    GraphNodeResult as YAMLGraphNodeResult,
    WorkflowState as YAMLWorkflowState,
)


def test_yaml_compiler_reexports_shared_runtime_types() -> None:
    assert YAMLGraphNodeResult is GraphNodeResult
    assert YAMLWorkflowState is WorkflowState


def test_workflows_package_reexports_shared_runtime_types() -> None:
    assert CompiledGraphNodeResult is GraphNodeResult
    assert PublicWorkflowState is WorkflowState


def test_unified_compiler_reexports_shared_node_result_type() -> None:
    assert UnifiedNodeExecutionResult is GraphNodeResult


def test_executor_package_reexports_workflow_node_executor_registry_helpers() -> None:
    assert ExecutorWorkflowNodeExecutorRegistration is WorkflowNodeExecutorRegistration
    assert ExecutorWorkflowNodeExecutorRegistry is WorkflowNodeExecutorRegistry
    assert executor_register_workflow_node_executor is register_workflow_node_executor
    assert executor_get_workflow_node_executor_registry is get_workflow_node_executor_registry
    assert (
        executor_clear_registered_workflow_node_executors
        is clear_registered_workflow_node_executors
    )


def test_workflows_package_reexports_workflow_node_executor_registry_helpers() -> None:
    assert PublicWorkflowNodeExecutorRegistration is WorkflowNodeExecutorRegistration
    assert PublicWorkflowNodeExecutorRegistry is WorkflowNodeExecutorRegistry
    assert public_register_workflow_node_executor is register_workflow_node_executor
    assert public_get_workflow_node_executor_registry is get_workflow_node_executor_registry
    assert (
        public_clear_registered_workflow_node_executors is clear_registered_workflow_node_executors
    )


def test_workflows_package_reexports_initial_workflow_state_helper() -> None:
    assert public_create_initial_workflow_state is create_initial_workflow_state


def test_create_initial_workflow_state_uses_explicit_workflow_id() -> None:
    state = create_initial_workflow_state(
        workflow_id="thread-123",
        workflow_name="example-workflow",
        current_node="start",
        initial_state={"input": "value"},
    )

    assert state["_workflow_id"] == "thread-123"
    assert state["_workflow_name"] == "example-workflow"
    assert state["_current_node"] == "start"
    assert state["_node_results"] == {}
    assert state["_parallel_results"] == {}
    assert state["_hitl_pending"] is False
    assert state["_hitl_response"] is None
    assert state["input"] == "value"


def test_create_initial_workflow_state_generates_workflow_id() -> None:
    state = create_initial_workflow_state()

    assert state["_workflow_id"]
    assert state["_workflow_name"] == ""
    assert state["_current_node"] == ""
    assert state["_node_results"] == {}
    assert state["_parallel_results"] == {}
