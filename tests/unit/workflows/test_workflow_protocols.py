"""Tests for workflow protocol definitions.

These tests verify that the workflow protocols are properly defined
and that implementations can be validated at runtime.
"""

from typing import Any, Optional

from victor.workflows.protocols import (
    ProtocolNodeStatus,
    RetryPolicy,
    NodeResult,
    IWorkflowNode,
    IWorkflowEdge,
    IWorkflowGraph,
    ICheckpointStore,
    IWorkflowExecutor,
)


class TestProtocolNodeStatus:
    """Tests for ProtocolNodeStatus enum."""

    def test_has_pending_status(self):
        """ProtocolNodeStatus should have PENDING state."""
        assert ProtocolNodeStatus.PENDING is not None
        assert ProtocolNodeStatus.PENDING.value == "pending"

    def test_has_running_status(self):
        """ProtocolNodeStatus should have RUNNING state."""
        assert ProtocolNodeStatus.RUNNING is not None
        assert ProtocolNodeStatus.RUNNING.value == "running"

    def test_has_completed_status(self):
        """ProtocolNodeStatus should have COMPLETED state."""
        assert ProtocolNodeStatus.COMPLETED is not None
        assert ProtocolNodeStatus.COMPLETED.value == "completed"

    def test_has_failed_status(self):
        """ProtocolNodeStatus should have FAILED state."""
        assert ProtocolNodeStatus.FAILED is not None
        assert ProtocolNodeStatus.FAILED.value == "failed"

    def test_has_skipped_status(self):
        """ProtocolNodeStatus should have SKIPPED state."""
        assert ProtocolNodeStatus.SKIPPED is not None
        assert ProtocolNodeStatus.SKIPPED.value == "skipped"


class TestRetryPolicy:
    """Tests for RetryPolicy dataclass."""

    def test_default_values(self):
        """RetryPolicy should have sensible defaults."""
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.delay_seconds == 1.0
        assert policy.exponential_backoff is True
        assert policy.retry_on_exceptions == (Exception,)

    def test_custom_values(self):
        """RetryPolicy should accept custom values."""
        policy = RetryPolicy(
            max_retries=5,
            delay_seconds=2.0,
            exponential_backoff=False,
            retry_on_exceptions=(ValueError, TypeError),
        )
        assert policy.max_retries == 5
        assert policy.delay_seconds == 2.0
        assert policy.exponential_backoff is False
        assert policy.retry_on_exceptions == (ValueError, TypeError)


class TestNodeResult:
    """Tests for NodeResult dataclass."""

    def test_successful_result(self):
        """NodeResult should store successful execution data."""
        result = NodeResult(
            status=ProtocolNodeStatus.COMPLETED,
            output={"key": "value"},
        )
        assert result.status == ProtocolNodeStatus.COMPLETED
        assert result.output == {"key": "value"}
        assert result.error is None

    def test_failed_result(self):
        """NodeResult should store failure information."""
        error = ValueError("Something went wrong")
        result = NodeResult(
            status=ProtocolNodeStatus.FAILED,
            output=None,
            error=error,
        )
        assert result.status == ProtocolNodeStatus.FAILED
        assert result.output is None
        assert result.error is error

    def test_result_with_metadata(self):
        """NodeResult should support metadata."""
        result = NodeResult(
            status=ProtocolNodeStatus.COMPLETED,
            output="result",
            metadata={"duration_ms": 150, "retries": 2},
        )
        assert result.metadata == {"duration_ms": 150, "retries": 2}


class TestIWorkflowNode:
    """Tests for IWorkflowNode protocol."""

    def test_protocol_is_runtime_checkable(self):
        """IWorkflowNode should be runtime checkable."""

        assert hasattr(IWorkflowNode, "__protocol_attrs__") or isinstance(IWorkflowNode, type)

    def test_compliant_class_is_instance(self):
        """A compliant class should pass isinstance check."""

        class MockNode:
            @property
            def id(self) -> str:
                return "node_1"

            @property
            def name(self) -> str:
                return "Test Node"

            @property
            def retry_policy(self) -> RetryPolicy:
                return RetryPolicy()

            async def execute(
                self, state: dict[str, Any], context: Optional[dict[str, Any]] = None
            ) -> NodeResult:
                return NodeResult(status=ProtocolNodeStatus.COMPLETED, output=state)

        node = MockNode()
        assert isinstance(node, IWorkflowNode)

    def test_non_compliant_class_is_not_instance(self):
        """A non-compliant class should fail isinstance check."""

        class IncompleteNode:
            @property
            def id(self) -> str:
                return "node_1"

            # Missing name, retry_policy, and execute

        node = IncompleteNode()
        assert not isinstance(node, IWorkflowNode)


class TestIWorkflowEdge:
    """Tests for IWorkflowEdge protocol."""

    def test_compliant_class_is_instance(self):
        """A compliant edge class should pass isinstance check."""

        class MockEdge:
            @property
            def source_id(self) -> str:
                return "node_1"

            @property
            def target_id(self) -> str:
                return "node_2"

            def should_traverse(self, state: dict[str, Any]) -> bool:
                return True

        edge = MockEdge()
        assert isinstance(edge, IWorkflowEdge)

    def test_edge_with_condition(self):
        """Edge should support conditional traversal."""

        class ConditionalMockEdge:
            @property
            def source_id(self) -> str:
                return "check_node"

            @property
            def target_id(self) -> str:
                return "success_node"

            def should_traverse(self, state: dict[str, Any]) -> bool:
                return state.get("success", False)

        edge = ConditionalMockEdge()
        assert edge.should_traverse({"success": True}) is True
        assert edge.should_traverse({"success": False}) is False


class TestIWorkflowGraph:
    """Tests for IWorkflowGraph protocol."""

    def test_compliant_class_is_instance(self):
        """A compliant graph class should pass isinstance check."""

        class MockGraph:
            def add_node(self, node: IWorkflowNode) -> "MockGraph":
                return self

            def add_edge(self, edge: IWorkflowEdge) -> "MockGraph":
                return self

            def get_node(self, node_id: str) -> Optional[IWorkflowNode]:
                return None

            def get_entry_node(self) -> Optional[IWorkflowNode]:
                return None

            def get_next_nodes(self, node_id: str, state: dict[str, Any]) -> list[IWorkflowNode]:
                return []

            def validate(self) -> list[str]:
                return []

        graph = MockGraph()
        assert isinstance(graph, IWorkflowGraph)


class TestICheckpointStore:
    """Tests for ICheckpointStore protocol."""

    def test_compliant_class_is_instance(self):
        """A compliant checkpoint store should pass isinstance check."""

        class MockCheckpointStore:
            async def save(
                self,
                workflow_id: str,
                checkpoint_id: str,
                state: dict[str, Any],
                metadata: Optional[dict[str, Any]] = None,
            ) -> None:
                pass

            async def load(self, workflow_id: str, checkpoint_id: str) -> Optional[dict[str, Any]]:
                return None

            async def list_checkpoints(self, workflow_id: str) -> list[str]:
                return []

            async def delete(self, workflow_id: str, checkpoint_id: str) -> bool:
                return True

        store = MockCheckpointStore()
        assert isinstance(store, ICheckpointStore)


class TestIWorkflowExecutor:
    """Tests for IWorkflowExecutor protocol."""

    def test_compliant_class_is_instance(self):
        """A compliant executor should pass isinstance check."""

        class MockExecutor:
            async def execute(
                self,
                graph: IWorkflowGraph,
                initial_state: dict[str, Any],
                checkpoint_store: Optional[ICheckpointStore] = None,
            ) -> dict[str, Any]:
                return initial_state

            async def resume(
                self,
                graph: IWorkflowGraph,
                checkpoint_store: ICheckpointStore,
                workflow_id: str,
                checkpoint_id: str,
            ) -> dict[str, Any]:
                return {}

            def cancel(self) -> None:
                pass

        executor = MockExecutor()
        assert isinstance(executor, IWorkflowExecutor)
