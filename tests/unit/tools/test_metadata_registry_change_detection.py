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

"""Tests for ToolMetadataRegistry change detection and performance improvements.

Tests that ToolMetadataRegistry uses hash-based change detection to skip
reindexing when tools haven't changed, improving startup performance.
"""

import pytest
from unittest.mock import Mock

from victor.tools.metadata_registry import ToolMetadataRegistry, get_global_registry
from victor.tools.base import BaseTool


class MockTool(BaseTool):
    """Mock tool for testing."""

    def __init__(self, name: str, description: str = "Test tool"):
        self._name = name
        self._description = description
        self._parameters = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return self._parameters

    def execute(self, **kwargs):
        return {"result": "ok"}


class TestToolMetadataRegistryChangeDetection:
    """Tests for hash-based change detection in ToolMetadataRegistry."""

    def test_refresh_from_tools_skips_reindex_when_unchanged(self, reset_singletons):
        """Registry should skip reindexing when tools haven't changed.

        Phase 3 implementation: refresh_from_tools should compute hash of
        tool definitions and skip reindexing if hash matches cached value.
        """
        # Create registry
        registry = ToolMetadataRegistry()

        # Create initial tools
        tools = [
            MockTool("tool1", "First tool"),
            MockTool("tool2", "Second tool"),
        ]

        # First call should reindex
        reindexed = registry.refresh_from_tools(tools)
        assert reindexed is True, "First call should reindex"
        assert len(registry._entries) == 2, "Should have 2 tools registered"

        # Second call with same tools should skip reindexing
        reindexed = registry.refresh_from_tools(tools)
        assert reindexed is False, "Second call with unchanged tools should skip reindexing"

    def test_refresh_from_tools_reindexes_when_tools_change(self, reset_singletons):
        """Registry should reindex when tools are added/removed/modified.

        Phase 3 implementation: Hash should detect any change to tool
        definitions including count, names, or descriptions.
        """
        registry = ToolMetadataRegistry()

        # Initial tools
        tools = [
            MockTool("tool1", "First tool"),
            MockTool("tool2", "Second tool"),
        ]

        # First call
        reindexed = registry.refresh_from_tools(tools)
        assert reindexed is True

        # Second call with same tools - should skip
        reindexed = registry.refresh_from_tools(tools)
        assert reindexed is False

        # Add a new tool - should reindex
        tools.append(MockTool("tool3", "Third tool"))
        reindexed = registry.refresh_from_tools(tools)
        assert reindexed is True, "Should reindex when tool is added"
        assert len(registry._entries) == 3

        # Modify tool description - should reindex
        tools[0]._description = "Modified description"
        reindexed = registry.refresh_from_tools(tools)
        assert reindexed is True, "Should reindex when tool description changes"

        # Remove a tool - should reindex
        tools.pop()
        reindexed = registry.refresh_from_tools(tools)
        assert reindexed is True, "Should reindex when tool is removed"
        assert len(registry._entries) == 2

    def test_refresh_from_tools_returns_statistics(self, reset_singletons):
        """Registry should provide statistics after reindexing.

        Phase 3 implementation: get_statistics should return counts of
        tools, categories, keywords, and other metadata.
        """
        registry = ToolMetadataRegistry()

        # Create tools with different properties
        tools = [
            MockTool("tool1", "Git tool"),
            MockTool("tool2", "Testing tool"),
            MockTool("tool3", "File tool"),
        ]

        # Refresh and verify statistics
        registry.refresh_from_tools(tools)
        stats = registry.get_statistics()

        assert "total_tools" in stats
        assert stats["total_tools"] == 3
        assert "total_categories" in stats
        assert "total_keywords" in stats

    def test_global_registry_has_get_instance_alias(self, reset_singletons):
        """ToolMetadataRegistry should provide get_instance() class method.

        Phase 3 implementation: Add get_instance() as alias to get_global_registry()
        for API consistency with semantic_selector expectations.
        """
        # This test verifies the API exists
        # The implementation can be either a class method or a module-level alias
        try:
            from victor.tools.metadata_registry import ToolMetadataRegistry

            # Try to call get_instance() - it should exist
            registry = ToolMetadataRegistry.get_instance()
            assert registry is not None
            assert isinstance(registry, ToolMetadataRegistry)
        except AttributeError:
            # If get_instance doesn't exist as class method,
            # check if there's a module-level function
            try:
                from victor.tools.metadata_registry import get_instance

                registry = get_instance()
                assert registry is not None
                assert isinstance(registry, ToolMetadataRegistry)
            except ImportError:
                pytest.fail("ToolMetadataRegistry.get_instance() not found - API gap")

    def test_refresh_from_tools_handles_empty_tool_list(self, reset_singletons):
        """Registry should handle empty tool list gracefully.

        Edge case: Empty tool list should not cause errors.
        """
        registry = ToolMetadataRegistry()

        # Empty list
        reindexed = registry.refresh_from_tools([])
        assert reindexed is True  # First call is always a reindex

        # Second call with empty list - should skip
        reindexed = registry.refresh_from_tools([])
        assert reindexed is False

        stats = registry.get_statistics()
        assert stats["total_tools"] == 0

    def test_refresh_from_tools_idempotent(self, reset_singletons):
        """Multiple calls with same tools should be idempotent.

        Phase 3: Hash-based change detection ensures idempotence.
        """
        registry = ToolMetadataRegistry()

        tools = [MockTool("tool1", "Tool"), MockTool("tool2", "Tool")]

        # Call multiple times
        results = []
        for i in range(5):
            reindexed = registry.refresh_from_tools(tools)
            results.append(reindexed)

        # First call should reindex, rest should skip
        assert results[0] is True, "First call should reindex"
        assert all(not r for r in results[1:]), "Subsequent calls should skip reindexing"

        # Registry state should be consistent
        assert len(registry._entries) == 2
        stats = registry.get_statistics()
        assert stats["total_tools"] == 2


class TestToolMetadataRegistryPerformance:
    """Tests for performance improvements in ToolMetadataRegistry."""

    def test_statistics_includes_performance_metrics(self, reset_singletons):
        """get_statistics should include performance-relevant metrics.

        Phase 3: Statistics should help users understand registry state
        and detect performance issues.
        """
        registry = ToolMetadataRegistry()

        tools = [MockTool(f"tool{i}", f"Tool {i}") for i in range(10)]

        registry.refresh_from_tools(tools)
        stats = registry.get_statistics()

        # Verify all expected keys
        expected_keys = {
            "total_tools",
            "total_categories",
            "total_keywords",
            "total_stages",
            "indexed_by_priority",
            "indexed_by_access_mode",
        }
        assert expected_keys.issubset(stats.keys()), f"Stats should include {expected_keys}"

        # Verify counts
        assert stats["total_tools"] == 10

    def test_reindex_performance_improvement(self, reset_singletons):
        """Skipping reindex should significantly improve performance.

        Phase 3: Demonstrate performance benefit of hash-based change detection.
        """
        import time

        registry = ToolMetadataRegistry()

        # Create many tools
        tools = [MockTool(f"tool{i}", f"Tool {i}") for i in range(100)]

        # First call - should reindex (slower)
        start = time.time()
        registry.refresh_from_tools(tools)
        first_duration = time.time() - start

        # Second call - should skip reindexing (much faster)
        start = time.time()
        reindexed = registry.refresh_from_tools(tools)
        second_duration = time.time() - start

        assert reindexed is False, "Second call should skip reindexing"
        # Skipping should be at least 2x faster (in practice, much more)
        assert (
            second_duration < first_duration / 2
        ), f"Skipping reindex should be faster: {second_duration:.4f}s vs {first_duration:.4f}s"
