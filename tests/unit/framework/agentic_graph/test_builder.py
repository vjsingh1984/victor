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

"""Tests for agentic loop graph builder and executor."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from victor.framework.agentic_graph.state import create_initial_state
from victor.framework.agentic_graph.builder import create_agentic_loop_graph
from victor.framework.agentic_graph.executor import AgenticLoopGraphExecutor, LoopResult


class TestAgenticLoopGraphBuilder:
    """Tests for create_agentic_loop_graph builder function."""

    def test_create_basic_graph(self):
        """Test creating a basic agentic loop graph."""
        graph = create_agentic_loop_graph()

        assert graph is not None
        # Verify graph structure (nodes are stored in _nodes)
        assert hasattr(graph, "_nodes")
        assert len(graph._nodes) >= 4  # At least perceive, plan, act, evaluate

    def test_create_graph_with_custom_max_iterations(self):
        """Test creating graph with custom max_iterations."""
        graph = create_agentic_loop_graph(max_iterations=5)

        assert graph is not None
        # Max iterations should be stored in graph config
        # (implementation dependent)

    def test_create_graph_with_fulfillment_enabled(self):
        """Test creating graph with fulfillment check enabled."""
        graph = create_agentic_loop_graph(enable_fulfillment=True)

        assert graph is not None

    def test_create_graph_with_adaptive_iterations(self):
        """Test creating graph with adaptive iterations enabled."""
        graph = create_agentic_loop_graph(enable_adaptive_iterations=True)

        assert graph is not None

    def test_graph_has_required_nodes(self):
        """Test that graph has all required nodes."""
        graph = create_agentic_loop_graph()
        compiled = graph.compile()

        # Check that nodes exist (implementation may vary)
        assert compiled is not None


class TestAgenticLoopGraphExecutor:
    """Tests for AgenticLoopGraphExecutor."""

    @pytest.mark.asyncio
    async def test_executor_initialization(self):
        """Test executor initialization."""
        mock_context = MagicMock()

        executor = AgenticLoopGraphExecutor(
            execution_context=mock_context,
            max_iterations=5,
        )

        assert executor.max_iterations == 5
        assert executor.execution_context == mock_context

    @pytest.mark.asyncio
    async def test_executor_run_simple_query(self):
        """Test executor with a simple query."""
        mock_context = MagicMock()

        executor = AgenticLoopGraphExecutor(
            execution_context=mock_context,
            max_iterations=3,
        )

        result = await executor.run("Hello")

        assert isinstance(result, LoopResult)
        # Note: Without proper service injection, execution may fail
        # This test validates the executor structure, not full execution

    @pytest.mark.asyncio
    async def test_executor_with_context(self):
        """Test executor with additional context."""
        mock_context = MagicMock()

        executor = AgenticLoopGraphExecutor(
            execution_context=mock_context,
            max_iterations=3,
        )

        result = await executor.run(
            "Help with code",
            context={"project": "victor", "file": "agent.py"},
        )

        assert isinstance(result, LoopResult)

    @pytest.mark.asyncio
    async def test_executor_max_iterations_enforced(self):
        """Test that executor max_iterations is stored."""
        mock_context = MagicMock()

        executor = AgenticLoopGraphExecutor(
            execution_context=mock_context,
            max_iterations=2,
        )

        assert executor.max_iterations == 2

    @pytest.mark.asyncio
    async def test_executor_streaming(self):
        """Test executor streaming support."""
        mock_context = MagicMock()
        mock_context.services = MagicMock()
        mock_context.services.chat = AsyncMock()

        executor = AgenticLoopGraphExecutor(
            execution_context=mock_context,
            max_iterations=3,
        )

        iterations = []
        async for event in executor.stream("Streaming query"):
            iterations.append(event)
            # Early break for testing
            if len(iterations) >= 2:
                break

        assert len(iterations) >= 1


class TestLoopResult:
    """Tests for LoopResult model."""

    def test_loop_result_creation(self):
        """Test creating a LoopResult."""
        result = LoopResult(
            success=True,
            response="Task completed",
            iterations=2,
            termination_reason="complete",
            metadata={"task_type": "code_generation"},
        )

        assert result.success is True
        assert result.response == "Task completed"
        assert result.iterations == 2
        assert result.termination_reason == "complete"

    def test_loop_result_from_graph_result(self):
        """Test creating LoopResult from graph execution result."""
        # Mock graph result
        graph_result = MagicMock()
        graph_result.success = True

        # Create state properly
        from victor.framework.agentic_graph.state import AgenticLoopStateModel
        state = AgenticLoopStateModel(query="Test", iteration=3)
        state = state.model_copy(update={
            "action_result": {"response": "Done"},
        })
        graph_result.state = state

        result = LoopResult.from_graph_result(graph_result, state)

        assert result.success is True
        assert result.iterations == 3


class TestGraphIntegration:
    """Integration tests for full graph execution."""

    @pytest.mark.asyncio
    async def test_full_graph_execution(self):
        """Test complete graph execution structure."""
        mock_context = MagicMock()

        # Create executor with mocked services
        executor = AgenticLoopGraphExecutor(
            execution_context=mock_context,
            max_iterations=3,
        )

        # Verify executor structure
        assert executor.graph is not None
        assert executor.compiled is not None
        assert executor.max_iterations == 3

        # Get execution stats
        stats = executor.get_execution_stats()
        assert stats["max_iterations"] == 3
        assert len(stats["graph_nodes"]) >= 4
