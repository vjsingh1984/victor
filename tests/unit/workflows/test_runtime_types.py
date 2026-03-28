from victor.workflows import (
    CompiledGraphNodeResult,
    WorkflowState as PublicWorkflowState,
    WorkflowNodeExecutorRegistration as PublicWorkflowNodeExecutorRegistration,
    WorkflowNodeExecutorRegistry as PublicWorkflowNodeExecutorRegistry,
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
        public_clear_registered_workflow_node_executors
        is clear_registered_workflow_node_executors
    )
