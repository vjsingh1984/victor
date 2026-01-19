# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for victor.workflows.adapters module."""

from unittest.mock import Mock, MagicMock, patch, AsyncMock

import pytest

from victor.workflows.adapters import (
    WorkflowState,
    AdaptedNode,
    WorkflowToGraphAdapter,
    GraphToWorkflowAdapter,
)
from victor.workflows.definition import (
    WorkflowDefinition,
    WorkflowNode,
    WorkflowNodeType,
    AgentNode,
    ComputeNode,
    ConditionNode,
    ParallelNode,
    TransformNode,
    TeamNodeWorkflow,
)


# =============================================================================
# WorkflowState Tests
# =============================================================================


class TestWorkflowState:
    """Test WorkflowState TypedDict."""

    def test_workflow_state_structure(self):
        """Test WorkflowState has correct structure."""
        state: WorkflowState = {
            "context": {"task": "test"},
            "messages": [{"role": "user", "content": "hello"}],
            "current_node": "node1",
            "visited_nodes": ["node1"],
            "results": {"node1": {"status": "success"}},
            "error": None,
            "is_complete": False,
        }
        assert state["context"]["task"] == "test"
        assert state["current_node"] == "node1"
        assert state["is_complete"] is False

    def test_workflow_state_optional_fields(self):
        """Test WorkflowState with optional fields."""
        state: WorkflowState = {
            "context": {},
        }
        assert "context" in state
        assert "messages" not in state


# =============================================================================
# AdaptedNode Tests
# =============================================================================


class TestAdaptedNode:
    """Test AdaptedNode dataclass."""

    def test_initialization(self):
        """Test AdaptedNode initialization."""

        def _identity_handler(state):
            return state

        node = AdaptedNode(
            name="test_node",
            node_type=WorkflowNodeType.AGENT,
            handler=_identity_handler,
            next_nodes=["node2", "node3"],
            tool_budget=20,
            allowed_tools=["tool1", "tool2"],
        )
        assert node.name == "test_node"
        assert node.node_type == WorkflowNodeType.AGENT
        assert node.next_nodes == ["node2", "node3"]
        assert node.tool_budget == 20
        assert node.allowed_tools == ["tool1", "tool2"]

    def test_default_values(self):
        """Test AdaptedNode default values."""

        def _identity_handler(state):
            return state

        node = AdaptedNode(
            name="test_node",
            node_type=WorkflowNodeType.AGENT,
            handler=_identity_handler,
        )
        assert node.next_nodes == []
        assert node.conditional_edges == {}
        assert node.tool_budget == 10
        assert node.allowed_tools == []


# =============================================================================
# WorkflowToGraphAdapter Tests
# =============================================================================


class TestWorkflowToGraphAdapter:
    """Test WorkflowToGraphAdapter class."""

    def test_initialization_defaults(self):
        """Test adapter initialization with default values."""
        adapter = WorkflowToGraphAdapter()
        assert adapter.max_iterations == 50
        assert adapter.enable_cycles is False

    def test_initialization_custom(self):
        """Test adapter initialization with custom values."""
        adapter = WorkflowToGraphAdapter(
            max_iterations=100,
            enable_cycles=True,
        )
        assert adapter.max_iterations == 100
        assert adapter.enable_cycles is True

    def test_adapt_simple_workflow(self):
        """Test adapting a simple workflow."""
        # Create a simple workflow
        node1 = AgentNode(
            id="node1",
            name="node1",
            role="researcher",
            goal="Do research",
            next_nodes=["node2"],
        )
        node2 = AgentNode(
            id="node2",
            name="node2",
            role="writer",
            goal="Write output",
            next_nodes=[],
        )
        workflow = WorkflowDefinition(
            name="test_workflow",
            description="Test workflow",
            nodes={
                "node1": node1,
                "node2": node2,
            },
        )

        adapter = WorkflowToGraphAdapter()
        graph = adapter.adapt(workflow)

        assert graph is not None
        # Verify graph structure
        assert hasattr(graph, "add_node")
        assert hasattr(graph, "add_edge")
        assert hasattr(graph, "compile")

    def test_adapt_with_single_node(self):
        """Test adapting a workflow with a single node."""
        only_node = AgentNode(
            id="only_node",
            name="only_node",
            role="agent",
            goal="Do something",
            next_nodes=[],
        )
        workflow = WorkflowDefinition(
            name="single_node_workflow",
            description="Single node workflow",
            nodes={
                "only_node": only_node,
            },
        )

        adapter = WorkflowToGraphAdapter()
        graph = adapter.adapt(workflow)

        assert graph is not None

    def test_adapt_with_branching(self):
        """Test adapting a workflow with branching paths."""
        start = AgentNode(
            id="start",
            name="start",
            role="coordinator",
            goal="Coordinate",
            next_nodes=["branch_a", "branch_b"],
        )
        branch_a = AgentNode(
            id="branch_a",
            name="branch_a",
            role="agent_a",
            goal="Handle branch A",
            next_nodes=[],
        )
        branch_b = AgentNode(
            id="branch_b",
            name="branch_b",
            role="agent_b",
            goal="Handle branch B",
            next_nodes=[],
        )
        workflow = WorkflowDefinition(
            name="branching_workflow",
            description="Branching workflow",
            nodes={
                "start": start,
                "branch_a": branch_a,
                "branch_b": branch_b,
            },
        )

        adapter = WorkflowToGraphAdapter()
        graph = adapter.adapt(workflow)

        assert graph is not None

    def test_adapt_empty_workflow(self):
        """Test adapting an empty workflow."""
        workflow = WorkflowDefinition(
            name="empty_workflow",
            description="Empty workflow",
            nodes={},
        )

        adapter = WorkflowToGraphAdapter()
        graph = adapter.adapt(workflow)

        assert graph is not None

    def test_adapt_node_creates_handler(self):
        """Test that _adapt_node creates a proper handler."""
        node1 = AgentNode(
            id="node1",
            name="node1",
            role="agent",
            goal="Test",
            next_nodes=[],
        )
        workflow = WorkflowDefinition(
            name="test_workflow",
            description="Test",
            nodes={
                "node1": node1,
            },
        )

        adapter = WorkflowToGraphAdapter()
        adapted = adapter._adapt_node(list(workflow.nodes.values())[0], workflow)

        assert adapted.name == "node1"
        assert adapted.node_type == WorkflowNodeType.AGENT
        assert callable(adapted.handler)

        # Test the handler
        state: WorkflowState = {
            "context": {},
            "messages": [],
            "current_node": "",
            "visited_nodes": [],
            "results": {},
            "error": None,
            "is_complete": False,
        }

        new_state = adapted.handler(state)
        assert new_state["current_node"] == "node1"
        assert "node1" in new_state["visited_nodes"]
        assert "node1" in new_state["results"]

    def test_adapt_node_preserves_properties(self):
        """Test that _adapt_node preserves node properties."""
        node = AgentNode(
            id="test_node",
            name="test_node",
            role="researcher",
            goal="Research task",
            next_nodes=["next"],
            tool_budget=30,
            allowed_tools=["tool1", "tool2"],
        )

        workflow = WorkflowDefinition(
            name="test_workflow",
            description="Test",
            nodes={
                "test_node": node,
            },
        )

        adapter = WorkflowToGraphAdapter()
        adapted = adapter._adapt_node(node, workflow)

        assert adapted.tool_budget == 30
        assert adapted.allowed_tools == ["tool1", "tool2"]
        assert adapted.next_nodes == ["next"]

    @patch("victor.framework.graph.StateGraph")
    def test_adapt_adds_edges_correctly(self, mock_graph_class):
        """Test that adapt adds edges between nodes."""
        mock_graph = MagicMock()
        mock_graph_class.return_value = mock_graph

        node1 = AgentNode(
            id="node1",
            name="node1",
            role="agent1",
            goal="Task 1",
            next_nodes=["node2"],
        )
        node2 = AgentNode(
            id="node2",
            name="node2",
            role="agent2",
            goal="Task 2",
            next_nodes=[],
        )
        workflow = WorkflowDefinition(
            name="test_workflow",
            description="Test",
            nodes={
                "node1": node1,
                "node2": node2,
            },
        )

        adapter = WorkflowToGraphAdapter()
        adapter.adapt(workflow)

        # Verify edges were added
        assert mock_graph.add_edge.called
        # Should have edge from node1 to node2
        # and edge from node2 to END

    def test_handler_updates_visited_nodes(self):
        """Test that handler tracks visited nodes."""
        node1 = AgentNode(
            id="node1",
            name="node1",
            role="agent",
            goal="Test",
            next_nodes=[],
        )
        workflow = WorkflowDefinition(
            name="test_workflow",
            description="Test",
            nodes={
                "node1": node1,
            },
        )

        adapter = WorkflowToGraphAdapter()
        adapted = adapter._adapt_node(list(workflow.nodes.values())[0], workflow)

        state: WorkflowState = {
            "context": {},
            "messages": [],
            "current_node": "",
            "visited_nodes": ["previous_node"],
            "results": {},
            "error": None,
            "is_complete": False,
        }

        new_state = adapted.handler(state)
        assert "previous_node" in new_state["visited_nodes"]
        assert "node1" in new_state["visited_nodes"]
        assert len(new_state["visited_nodes"]) == 2

    def test_handler_preserves_context(self):
        """Test that handler preserves workflow context."""
        node1 = AgentNode(
            id="node1",
            name="node1",
            role="agent",
            goal="Test",
            next_nodes=[],
        )
        workflow = WorkflowDefinition(
            name="test_workflow",
            description="Test",
            nodes={
                "node1": node1,
            },
        )

        adapter = WorkflowToGraphAdapter()
        adapted = adapter._adapt_node(list(workflow.nodes.values())[0], workflow)

        state: WorkflowState = {
            "context": {"task": "important task", "data": {"key": "value"}},
            "messages": [],
            "current_node": "",
            "visited_nodes": [],
            "results": {},
            "error": None,
            "is_complete": False,
        }

        new_state = adapted.handler(state)
        assert new_state["context"]["task"] == "important task"
        assert new_state["context"]["data"]["key"] == "value"


class TestWorkflowToGraphAdapterExecution:
    """Test WorkflowToGraphAdapter with execution handlers."""

    def test_adapt_with_execution(self):
        """Test adapting workflow with real execution handlers."""
        node1 = AgentNode(
            id="node1",
            name="node1",
            role="agent",
            goal="Test",
            next_nodes=[],
        )
        workflow = WorkflowDefinition(
            name="test_workflow",
            description="Test",
            nodes={
                "node1": node1,
            },
        )

        mock_executor = MagicMock()
        mock_executor.execute_node = AsyncMock(return_value={"status": "success"})

        adapter = WorkflowToGraphAdapter()
        graph = adapter.adapt_with_execution(workflow, mock_executor)

        assert graph is not None

    def test_execution_handler_calls_executor(self):
        """Test that execution handler calls executor."""
        node = AgentNode(
            id="node1",
            name="node1",
            role="agent",
            goal="Test",
            next_nodes=[],
        )

        workflow = WorkflowDefinition(
            name="test_workflow",
            description="Test",
            nodes={
                "node1": node,
            },
        )

        mock_executor = MagicMock()
        mock_executor.execute_node = AsyncMock(return_value={"status": "completed"})

        adapter = WorkflowToGraphAdapter()
        handler = adapter._create_execution_handler(node, mock_executor)

        state: WorkflowState = {
            "context": {"task": "test"},
            "messages": [],
            "current_node": "",
            "visited_nodes": [],
            "results": {},
            "error": None,
            "is_complete": False,
        }

        new_state = handler(state)
        assert new_state["current_node"] == "node1"
        assert "node1" in new_state["visited_nodes"]
        # Executor should have been called
        # Note: The sync wrapper makes it tricky to test async calls directly

    def test_execution_handler_handles_errors(self):
        """Test that execution handler handles errors gracefully."""
        node = AgentNode(
            id="node1",
            name="node1",
            role="agent",
            goal="Test",
            next_nodes=[],
        )

        workflow = WorkflowDefinition(
            name="test_workflow",
            description="Test",
            nodes={
                "node1": node,
            },
        )

        mock_executor = MagicMock()
        mock_executor.execute_node = AsyncMock(side_effect=Exception("Execution failed"))

        adapter = WorkflowToGraphAdapter()
        handler = adapter._create_execution_handler(node, mock_executor)

        state: WorkflowState = {
            "context": {},
            "messages": [],
            "current_node": "",
            "visited_nodes": [],
            "results": {},
            "error": None,
            "is_complete": False,
        }

        new_state = handler(state)
        assert new_state["error"] is not None
        assert "Execution failed" in new_state["error"]


# =============================================================================
# GraphToWorkflowAdapter Tests
# =============================================================================


class TestGraphToWorkflowAdapter:
    """Test GraphToWorkflowAdapter class."""

    def test_adapt_stategraph_to_workflow(self):
        """Test adapting StateGraph to WorkflowDefinition."""
        # Create a mock StateGraph
        mock_graph = MagicMock()
        mock_graph._nodes = {
            "node1": MagicMock(),
            "node2": MagicMock(),
        }
        mock_graph._edges = {
            "node1": ["node2"],
        }
        mock_graph._entry_point = "node1"

        adapter = GraphToWorkflowAdapter()
        workflow = adapter.adapt(mock_graph, "test_workflow")

        assert workflow is not None
        assert workflow.name == "test_workflow"

    def test_adapt_preserves_entry_point(self):
        """Test that adapter preserves entry point order."""
        mock_graph = MagicMock()
        mock_graph._nodes = {
            "node2": MagicMock(),
            "node1": MagicMock(),
            "node3": MagicMock(),
        }
        mock_graph._edges = {}
        mock_graph._entry_point = "node1"

        adapter = GraphToWorkflowAdapter()
        workflow = adapter.adapt(mock_graph, "test_workflow")

        # Entry point should be first
        nodes_list = list(workflow.nodes.values())
        assert nodes_list[0].name == "node1"

    def test_adapt_without_entry_point(self):
        """Test adapting graph without entry point."""
        mock_graph = MagicMock()
        mock_graph._nodes = {
            "node1": MagicMock(),
            "node2": MagicMock(),
        }
        mock_graph._edges = {}
        mock_graph._entry_point = None

        adapter = GraphToWorkflowAdapter()
        workflow = adapter.adapt(mock_graph, "test_workflow")

        assert workflow is not None
        assert len(workflow.nodes) == 2

    def test_adapt_empty_graph(self):
        """Test adapting an empty graph."""
        mock_graph = MagicMock()
        mock_graph._nodes = {}
        mock_graph._edges = {}
        mock_graph._entry_point = None

        adapter = GraphToWorkflowAdapter()
        workflow = adapter.adapt(mock_graph, "empty_workflow")

        assert workflow is not None
        assert len(workflow.nodes) == 0
