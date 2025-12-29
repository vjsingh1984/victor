"""Workflow protocol definitions for graph-based workflow execution.

This module defines the core protocols (interfaces) for the workflow system,
following SOLID principles and enabling LangGraph-like workflow capabilities.

Protocols defined:
- IWorkflowNode: Interface for workflow nodes (steps)
- IWorkflowEdge: Interface for edges connecting nodes
- IWorkflowGraph: Interface for the workflow graph structure
- ICheckpointStore: Interface for state persistence
- IWorkflowExecutor: Interface for workflow execution
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type, runtime_checkable


class NodeStatus(Enum):
    """Status of a workflow node execution.

    Attributes:
        PENDING: Node has not started execution.
        RUNNING: Node is currently executing.
        COMPLETED: Node finished successfully.
        FAILED: Node execution failed.
        SKIPPED: Node was skipped (e.g., due to conditional edge).
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RetryPolicy:
    """Configuration for node retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts.
        delay_seconds: Base delay between retries.
        exponential_backoff: Whether to use exponential backoff.
        retry_on_exceptions: Tuple of exception types to retry on.
    """

    max_retries: int = 3
    delay_seconds: float = 1.0
    exponential_backoff: bool = True
    retry_on_exceptions: Tuple[Type[Exception], ...] = (Exception,)


@dataclass
class NodeResult:
    """Result of a node execution.

    Attributes:
        status: The execution status.
        output: The output data from the node.
        error: The exception if the node failed.
        metadata: Additional metadata about the execution.
    """

    status: NodeStatus
    output: Any = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class IWorkflowNode(Protocol):
    """Protocol for workflow nodes.

    A workflow node represents a single step in a workflow graph.
    Nodes are executed by the workflow executor and can produce
    outputs that affect workflow state.

    Example:
        class MyNode:
            @property
            def id(self) -> str:
                return "my_node"

            @property
            def name(self) -> str:
                return "My Custom Node"

            @property
            def retry_policy(self) -> RetryPolicy:
                return RetryPolicy(max_retries=3)

            async def execute(
                self,
                state: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None
            ) -> NodeResult:
                # Process state and return result
                return NodeResult(
                    status=NodeStatus.COMPLETED,
                    output={"processed": True}
                )
    """

    @property
    def id(self) -> str:
        """Unique identifier for the node."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name for the node."""
        ...

    @property
    def retry_policy(self) -> RetryPolicy:
        """Retry policy for this node."""
        ...

    async def execute(
        self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> NodeResult:
        """Execute the node with the given state.

        Args:
            state: Current workflow state.
            context: Optional execution context.

        Returns:
            NodeResult with status and output.
        """
        ...


@runtime_checkable
class IWorkflowEdge(Protocol):
    """Protocol for workflow edges.

    An edge connects two nodes and can include conditional logic
    to determine whether traversal should occur based on state.

    Example:
        class MyEdge:
            @property
            def source_id(self) -> str:
                return "node_a"

            @property
            def target_id(self) -> str:
                return "node_b"

            def should_traverse(self, state: Dict[str, Any]) -> bool:
                return state.get("continue", True)
    """

    @property
    def source_id(self) -> str:
        """ID of the source node."""
        ...

    @property
    def target_id(self) -> str:
        """ID of the target node."""
        ...

    def should_traverse(self, state: Dict[str, Any]) -> bool:
        """Determine if this edge should be traversed.

        Args:
            state: Current workflow state.

        Returns:
            True if the edge should be traversed.
        """
        ...


@runtime_checkable
class IWorkflowGraph(Protocol):
    """Protocol for workflow graphs.

    A workflow graph is a directed graph of nodes connected by edges.
    It defines the structure and flow of a workflow.

    Example:
        graph = MyGraph()
        graph.add_node(start_node)
        graph.add_node(process_node)
        graph.add_edge(simple_edge)
        errors = graph.validate()
    """

    def add_node(self, node: IWorkflowNode) -> "IWorkflowGraph":
        """Add a node to the graph.

        Args:
            node: The node to add.

        Returns:
            Self for fluent interface.
        """
        ...

    def add_edge(self, edge: IWorkflowEdge) -> "IWorkflowGraph":
        """Add an edge to the graph.

        Args:
            edge: The edge to add.

        Returns:
            Self for fluent interface.
        """
        ...

    def get_node(self, node_id: str) -> Optional[IWorkflowNode]:
        """Get a node by its ID.

        Args:
            node_id: The node's unique identifier.

        Returns:
            The node if found, None otherwise.
        """
        ...

    def get_entry_node(self) -> Optional[IWorkflowNode]:
        """Get the entry node of the graph.

        Returns:
            The entry node if set, None otherwise.
        """
        ...

    def get_next_nodes(
        self, node_id: str, state: Dict[str, Any]
    ) -> List[IWorkflowNode]:
        """Get the next nodes to execute after the given node.

        Args:
            node_id: The current node's ID.
            state: Current workflow state (for conditional edges).

        Returns:
            List of nodes to execute next.
        """
        ...

    def validate(self) -> List[str]:
        """Validate the graph structure.

        Returns:
            List of validation error messages (empty if valid).
        """
        ...


@runtime_checkable
class ICheckpointStore(Protocol):
    """Protocol for checkpoint storage.

    A checkpoint store persists workflow state for resumption,
    debugging, and fault tolerance.

    Example:
        store = MyCheckpointStore()
        await store.save("wf_123", "cp_1", {"step": 3, "data": [1,2,3]})
        state = await store.load("wf_123", "cp_1")
    """

    async def save(
        self,
        workflow_id: str,
        checkpoint_id: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a checkpoint.

        Args:
            workflow_id: Unique workflow instance identifier.
            checkpoint_id: Unique checkpoint identifier.
            state: The state to save.
            metadata: Optional metadata about the checkpoint.
        """
        ...

    async def load(
        self, workflow_id: str, checkpoint_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load a checkpoint.

        Args:
            workflow_id: Unique workflow instance identifier.
            checkpoint_id: Unique checkpoint identifier.

        Returns:
            The saved state if found, None otherwise.
        """
        ...

    async def list_checkpoints(self, workflow_id: str) -> List[str]:
        """List all checkpoints for a workflow.

        Args:
            workflow_id: Unique workflow instance identifier.

        Returns:
            List of checkpoint IDs.
        """
        ...

    async def delete(self, workflow_id: str, checkpoint_id: str) -> bool:
        """Delete a checkpoint.

        Args:
            workflow_id: Unique workflow instance identifier.
            checkpoint_id: Unique checkpoint identifier.

        Returns:
            True if deleted, False if not found.
        """
        ...


@runtime_checkable
class IWorkflowExecutor(Protocol):
    """Protocol for workflow execution.

    A workflow executor runs a workflow graph, managing state
    transitions, error handling, and checkpointing.

    Example:
        executor = MyExecutor()
        result = await executor.execute(graph, {"input": "data"})
    """

    async def execute(
        self,
        graph: IWorkflowGraph,
        initial_state: Dict[str, Any],
        checkpoint_store: Optional[ICheckpointStore] = None,
    ) -> Dict[str, Any]:
        """Execute a workflow graph.

        Args:
            graph: The workflow graph to execute.
            initial_state: Initial workflow state.
            checkpoint_store: Optional store for checkpointing.

        Returns:
            Final workflow state.
        """
        ...

    async def resume(
        self,
        graph: IWorkflowGraph,
        checkpoint_store: ICheckpointStore,
        workflow_id: str,
        checkpoint_id: str,
    ) -> Dict[str, Any]:
        """Resume a workflow from a checkpoint.

        Args:
            graph: The workflow graph to execute.
            checkpoint_store: Store containing the checkpoint.
            workflow_id: Workflow instance identifier.
            checkpoint_id: Checkpoint to resume from.

        Returns:
            Final workflow state.
        """
        ...

    def cancel(self) -> None:
        """Cancel the currently executing workflow."""
        ...
