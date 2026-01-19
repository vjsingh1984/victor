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

"""Tests for tool execution compiler."""

import pytest

from victor.agent.tool_compiler import ToolExecutionCompiler
from victor.agent.tool_graph import CacheStrategy


class MockTool:
    """Mock tool for testing."""

    def __init__(self, name: str, description: str = "Test tool"):
        self.name = name
        self.description = description
        self.category = "test"
        self.cost_tier = "low"
        self.parameters = {
            "properties": {
                "path": {"type": "string"},
                "pattern": {"type": "string"},
            },
            "required": ["path"],
        }


class MockToolRegistry:
    """Mock tool registry for testing."""

    def __init__(self):
        self._tools = {
            "read": MockTool("read", "Read file"),
            "grep": MockTool("grep", "Search file"),
            "write": MockTool("write", "Write file"),
        }

    def get_tool(self, name: str):
        """Get tool by name."""
        return self._tools.get(name)


@pytest.fixture
def mock_registry():
    """Create mock tool registry."""
    return MockToolRegistry()


class TestToolExecutionCompiler:
    """Tests for ToolExecutionCompiler."""

    def test_compile_single_tool(self, mock_registry):
        """Test compiling single tool call."""
        compiler = ToolExecutionCompiler(mock_registry)

        tool_calls = [{"name": "read", "arguments": {"path": "/test"}}]

        graph = compiler.compile(tool_calls)

        assert len(graph.nodes) == 1
        assert graph.nodes[0].tool_name == "read"
        assert len(graph.edges) == 0

    def test_compile_multiple_tools(self, mock_registry):
        """Test compiling multiple tool calls."""
        compiler = ToolExecutionCompiler(mock_registry)

        tool_calls = [
            {"name": "read", "arguments": {"path": "/test"}},
            {"name": "grep", "arguments": {"pattern": "foo"}},
        ]

        graph = compiler.compile(tool_calls)

        assert len(graph.nodes) == 2
        assert graph.nodes[0].tool_name == "read"
        assert graph.nodes[1].tool_name == "grep"

    def test_compile_deduplicates_tools(self, mock_registry):
        """Test that duplicate tool names are deduplicated."""
        compiler = ToolExecutionCompiler(mock_registry)

        tool_calls = [
            {"name": "read", "arguments": {"path": "/test1"}},
            {"name": "read", "arguments": {"path": "/test2"}},
        ]

        graph = compiler.compile(tool_calls)

        # Should only have one node for "read"
        assert len(graph.nodes) == 1
        assert graph.nodes[0].tool_name == "read"

    def test_compile_skips_invalid_calls(self, mock_registry):
        """Test that invalid tool calls are skipped."""
        compiler = ToolExecutionCompiler(mock_registry)

        tool_calls = [
            {"name": "read", "arguments": {"path": "/test"}},
            None,  # Invalid
            "invalid",  # Invalid type
            {},  # Missing name
        ]

        graph = compiler.compile(tool_calls)

        # Should only have the valid call
        assert len(graph.nodes) == 1
        assert graph.nodes[0].tool_name == "read"

    def test_compile_adds_validation_rules(self, mock_registry):
        """Test that validation rules are added to nodes."""
        compiler = ToolExecutionCompiler(mock_registry)

        tool_calls = [{"name": "read", "arguments": {"path": "/test"}}]

        graph = compiler.compile(tool_calls)

        node = graph.nodes[0]
        assert len(node.validation_rules) > 0

        # Check for required parameter rule
        required_rules = [r for r in node.validation_rules if r.parameter == "path"]
        assert len(required_rules) > 0

    def test_compile_sets_cache_policy(self, mock_registry):
        """Test that cache policy is set correctly."""
        compiler = ToolExecutionCompiler(mock_registry)

        tool_calls = [{"name": "read", "arguments": {"path": "/test"}}]

        graph = compiler.compile(tool_calls)

        node = graph.nodes[0]
        # read is typically idempotent
        assert node.cache_policy in ("idempotent", "default")

    def test_compute_hash_consistency(self, mock_registry):
        """Test that hash computation is consistent."""
        compiler = ToolExecutionCompiler(mock_registry)

        tool_calls = [{"name": "read", "arguments": {"path": "/test"}}]

        hash1 = compiler.compute_graph_hash(tool_calls)
        hash2 = compiler.compute_graph_hash(tool_calls)

        assert hash1 == hash2
        assert len(hash1) == 16  # SHA256 truncated to 16 chars

    def test_compute_hash_order_independence(self, mock_registry):
        """Test that hash is independent of order."""
        compiler = ToolExecutionCompiler(mock_registry)

        tool_calls1 = [
            {"name": "read", "arguments": {"path": "/test"}},
            {"name": "grep", "arguments": {"pattern": "foo"}},
        ]

        tool_calls2 = [
            {"name": "grep", "arguments": {"pattern": "foo"}},
            {"name": "read", "arguments": {"path": "/test"}},
        ]

        hash1 = compiler.compute_graph_hash(tool_calls1)
        hash2 = compiler.compute_graph_hash(tool_calls2)

        # Same tools, different order should produce same hash
        # because we sort before hashing
        assert hash1 == hash2

    def test_determine_cache_strategy_all_idempotent(self, mock_registry):
        """Test cache strategy when all tools are idempotent."""
        compiler = ToolExecutionCompiler(mock_registry)

        from victor.agent.tool_graph import ToolExecutionNode

        nodes = [
            ToolExecutionNode(tool_name="read", cache_policy="idempotent"),
            ToolExecutionNode(tool_name="grep", cache_policy="idempotent"),
        ]

        strategy = compiler._determine_cache_strategy(nodes)
        assert strategy == CacheStrategy.TTL

    def test_determine_cache_strategy_mixed(self, mock_registry):
        """Test cache strategy with mixed policies."""
        compiler = ToolExecutionCompiler(mock_registry)

        from victor.agent.tool_graph import ToolExecutionNode

        nodes = [
            ToolExecutionNode(tool_name="read", cache_policy="idempotent"),
            ToolExecutionNode(tool_name="write", cache_policy="default"),
        ]

        strategy = compiler._determine_cache_strategy(nodes)
        assert strategy == CacheStrategy.ADAPTIVE

    def test_get_tool_metadata(self, mock_registry):
        """Test that tool metadata is extracted correctly."""
        compiler = ToolExecutionCompiler(mock_registry)

        metadata = compiler._get_tool_metadata("read")

        assert "description" in metadata
        assert metadata["description"] == "Read file"
        assert metadata["category"] == "test"
        assert metadata["cost_tier"] == "low"

    def test_get_tool_metadata_unknown_tool(self, mock_registry):
        """Test metadata for unknown tool."""
        compiler = ToolExecutionCompiler(mock_registry)

        metadata = compiler._get_tool_metadata("unknown")

        assert metadata == {}
