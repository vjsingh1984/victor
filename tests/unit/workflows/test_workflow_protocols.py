"""Tests for workflow protocol definitions.

These tests verify that the workflow protocols are properly defined
and that implementations can be validated at runtime.
"""

import pytest
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from victor.workflows.protocols import (
    NodeStatus,
    RetryPolicy,
    NodeResult,
    IWorkflowNode,
    IWorkflowEdge,
    IWorkflowGraph,
    ICheckpointStore,
    IWorkflowExecutor,
)


class TestNodeStatus:
    """Tests for NodeStatus enum."""

    def test_has_pending_status(self):
        """NodeStatus should have PENDING state."""
        assert NodeStatus.PENDING is not None
        assert NodeStatus.PENDING.value == "pending"

    def test_has_running_status(self):
        """NodeStatus should have RUNNING state."""
        assert NodeStatus.RUNNING is not None
        assert NodeStatus.RUNNING.value == "running"

    def test_has_completed_status(self):
        """NodeStatus should have COMPLETED state."""
        assert NodeStatus.COMPLETED is not None
        assert NodeStatus.COMPLETED.value == "completed"

    def test_has_failed_status(self):
        """NodeStatus should have FAILED state."""
        assert NodeStatus.FAILED is not None
        assert NodeStatus.FAILED.value == "failed"

    def test_has_skipped_status(self):
        """NodeStatus should have SKIPPED state."""
        assert NodeStatus.SKIPPED is not None
        assert NodeStatus.SKIPPED.value == "skipped"


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
            status=NodeStatus.COMPLETED,
            output={"key": "value"},
        )
        assert result.status == NodeStatus.COMPLETED
        assert result.output == {"key": "value"}
        assert result.error is None

    def test_failed_result(self):
        """NodeResult should store failure information."""
        error = ValueError("Something went wrong")
        result = NodeResult(
            status=NodeStatus.FAILED,
            output=None,
            error=error,
        )
        assert result.status == NodeStatus.FAILED
        assert result.output is None
        assert result.error is error

    def test_result_with_metadata(self):
        """NodeResult should support metadata."""
        result = NodeResult(
            status=NodeStatus.COMPLETED,
            output="result",
            metadata={"duration_ms": 150, "retries": 2},
        )
        assert result.metadata == {"duration_ms": 150, "retries": 2}


class TestIWorkflowNode:
    """Tests for IWorkflowNode protocol."""

    def test_protocol_is_runtime_checkable(self):
        """IWorkflowNode should be runtime checkable."""
        from typing import runtime_checkable, Protocol

        assert hasattr(IWorkflowNode, "__protocol_attrs__") or isinstance(
            IWorkflowNode, type
        )

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
                self, state: Dict[str, Any], context: Optional[Dict[str, Any]] = None
            ) -> NodeResult:
                return NodeResult(status=NodeStatus.COMPLETED, output=state)

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

            def should_traverse(self, state: Dict[str, Any]) -> bool:
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

            def should_traverse(self, state: Dict[str, Any]) -> bool:
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

            def get_next_nodes(
                self, node_id: str, state: Dict[str, Any]
            ) -> List[IWorkflowNode]:
                return []

            def validate(self) -> List[str]:
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
                state: Dict[str, Any],
                metadata: Optional[Dict[str, Any]] = None,
            ) -> None:
                pass

            async def load(
                self, workflow_id: str, checkpoint_id: str
            ) -> Optional[Dict[str, Any]]:
                return None

            async def list_checkpoints(self, workflow_id: str) -> List[str]:
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
                initial_state: Dict[str, Any],
                checkpoint_store: Optional[ICheckpointStore] = None,
            ) -> Dict[str, Any]:
                return initial_state

            async def resume(
                self,
                graph: IWorkflowGraph,
                checkpoint_store: ICheckpointStore,
                workflow_id: str,
                checkpoint_id: str,
            ) -> Dict[str, Any]:
                return {}

            def cancel(self) -> None:
                pass

        executor = MockExecutor()
        assert isinstance(executor, IWorkflowExecutor)
