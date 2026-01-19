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

"""Tests for tool execution engine."""

import pytest

from victor.agent.tool_engine import ToolExecutionEngine
from victor.agent.tool_graph import ToolExecutionGraph, ToolExecutionNode


class MockToolCallResult:
    """Mock tool call result."""

    def __init__(self, success: bool = True):
        self.tool_name = "test"
        self.arguments = {}
        self.success = success
        self.result = "test result"
        self.error = None
        self.execution_time_ms = 10.0
        self.cached = False
        self.skipped = False


class MockToolPipeline:
    """Mock tool pipeline for testing."""

    def __init__(self):
        self.executed_calls = []

    async def _execute_single_call(self, tool_call: dict, context: dict):
        """Mock execution."""
        self.executed_calls.append(tool_call)
        return MockToolCallResult(success=True)


@pytest.fixture
def mock_pipeline():
    """Create mock tool pipeline."""
    return MockToolPipeline()


@pytest.fixture
def sample_graph():
    """Create sample execution graph."""
    nodes = [
        ToolExecutionNode(tool_name="read"),
        ToolExecutionNode(tool_name="grep"),
    ]
    return ToolExecutionGraph(nodes=nodes)


class TestToolExecutionEngine:
    """Tests for ToolExecutionEngine."""

    @pytest.mark.asyncio
    async def test_execute_graph(self, mock_pipeline, sample_graph):
        """Test executing a graph."""
        engine = ToolExecutionEngine(mock_pipeline)

        result = await engine.execute(sample_graph, {})

        assert result.total_calls == 2
        assert result.successful_calls == 2
        assert result.failed_calls == 0
        assert len(result.results) == 2
        assert len(mock_pipeline.executed_calls) == 2

    @pytest.mark.asyncio
    async def test_execute_graph_with_failures(self, mock_pipeline):
        """Test executing graph with failures."""

        async def failing_execute(tool_call, context):
            return MockToolCallResult(success=False)

        mock_pipeline._execute_single_call = failing_execute

        nodes = [ToolExecutionNode(tool_name="read")]
        graph = ToolExecutionGraph(nodes=nodes)

        engine = ToolExecutionEngine(mock_pipeline)
        result = await engine.execute(graph, {})

        assert result.total_calls == 1
        assert result.successful_calls == 0
        assert result.failed_calls == 1

    def test_cache_graph(self, mock_pipeline, sample_graph):
        """Test caching a graph."""
        engine = ToolExecutionEngine(mock_pipeline)

        engine.cache_graph("test_key", sample_graph)

        cached = engine.get_cached_graph("test_key")
        assert cached is not None
        assert cached.nodes[0].tool_name == "read"

    def test_get_cached_graph_miss(self, mock_pipeline):
        """Test getting non-existent cached graph."""
        engine = ToolExecutionEngine(mock_pipeline)

        cached = engine.get_cached_graph("nonexistent")
        assert cached is None

    def test_clear_cache(self, mock_pipeline, sample_graph):
        """Test clearing cache."""
        engine = ToolExecutionEngine(mock_pipeline)

        engine.cache_graph("key1", sample_graph)
        engine.cache_graph("key2", sample_graph)

        assert engine.get_cached_graph("key1") is not None
        assert engine.get_cached_graph("key2") is not None

        engine.clear_cache()

        assert engine.get_cached_graph("key1") is None
        assert engine.get_cached_graph("key2") is None

    def test_get_cache_stats(self, mock_pipeline, sample_graph):
        """Test getting cache statistics."""
        engine = ToolExecutionEngine(mock_pipeline)

        engine.cache_graph("key1", sample_graph)
        engine.cache_graph("key2", sample_graph)

        stats = engine.get_cache_stats()

        assert stats["cached_graphs"] == 2
        assert "key1" in stats["cache_keys"]
        assert "key2" in stats["cache_keys"]
        assert len(stats["cache_keys"]) == 2

    def test_cache_overwrite(self, mock_pipeline, sample_graph):
        """Test overwriting cached graph."""
        engine = ToolExecutionEngine(mock_pipeline)

        # Cache initial graph
        engine.cache_graph("test_key", sample_graph)
        initial = engine.get_cached_graph("test_key")

        # Create new graph with different node
        new_nodes = [ToolExecutionNode(tool_name="write")]
        new_graph = ToolExecutionGraph(nodes=new_nodes)

        # Overwrite
        engine.cache_graph("test_key", new_graph)
        updated = engine.get_cached_graph("test_key")

        assert initial.nodes[0].tool_name == "read"
        assert updated.nodes[0].tool_name == "write"

    @pytest.mark.asyncio
    async def test_execute_empty_graph(self, mock_pipeline):
        """Test executing empty graph raises error."""
        # Empty graphs should fail validation
        with pytest.raises(ValueError, match="must have at least one node"):
            ToolExecutionGraph(nodes=[])

    @pytest.mark.asyncio
    async def test_execute_preserves_context(self, mock_pipeline, sample_graph):
        """Test that context is passed through execution."""
        engine = ToolExecutionEngine(mock_pipeline)

        test_context = {"user": "test", "session": "123"}
        await engine.execute(sample_graph, test_context)

        # Verify context was passed
        assert len(mock_pipeline.executed_calls) > 0
