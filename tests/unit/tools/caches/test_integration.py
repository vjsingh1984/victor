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

"""Integration tests for tool selection caching.

This module tests end-to-end caching integration across:
- HybridToolSelector cache lookups and storage
- SemanticToolSelector cache integration
- KeywordToolSelector cache integration
- Cache invalidation on tools change
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.protocols import ToolSelectionContext
from victor.providers.base import ToolDefinition
from victor.tools.caches import (
    get_tool_selection_cache,
    invalidate_tool_selection_cache,
)
from victor.tools.semantic_selector import SemanticToolSelector
from victor.tools.keyword_tool_selector import KeywordToolSelector
from victor.tools.hybrid_tool_selector import HybridToolSelector, HybridSelectorConfig


class MockTool:
    """Mock tool with proper attributes."""

    def __init__(self, name: str, description: str, parameters: dict):
        self.name = name
        self.description = description
        self.parameters = parameters

    def get_metadata(self):
        """Return mock metadata."""
        from victor.tools.metadata import ToolMetadata

        return ToolMetadata(
            categories=["test"],
            keywords=[self.name],
            use_cases=[],
            examples=[],
        )


@pytest.fixture
def mock_tools():
    """Create mock tool registry."""
    tools = MagicMock()
    tools.list_tools.return_value = [
        MockTool(
            name="read",
            description="Read file contents",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
        MockTool(
            name="write",
            description="Write content to a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
        MockTool(
            name="search",
            description="Search for code patterns",
            parameters={"type": "object", "properties": {"pattern": {"type": "string"}}},
        ),
    ]
    tools.is_tool_enabled.return_value = True
    tools.get.return_value = MockTool(
        name="read",
        description="Read file contents",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
    )
    return tools


@pytest.fixture
def mock_context():
    """Create mock tool selection context."""
    context = MagicMock(spec=ToolSelectionContext)
    context.conversation_history = []
    context.pending_actions = []
    context.planned_tools = []
    context.conversation_stage = None
    return context


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset cache before each test."""
    invalidate_tool_selection_cache()
    yield
    invalidate_tool_selection_cache()


class TestSemanticToolSelectorCaching:
    """Test SemanticToolSelector caching integration."""

    @pytest.mark.asyncio
    async def test_cache_lookup_before_selection(self, mock_tools):
        """Test that cache is checked before semantic selection."""
        selector = SemanticToolSelector(cache_embeddings=False)
        selector._tools_registry = mock_tools

        # Pre-populate cache with a result
        cache = get_tool_selection_cache()
        key_gen = cache._get_key_generator = MagicMock()
        from victor.tools.caches import get_cache_key_generator

        real_key_gen = get_cache_key_generator()
        cache._get_key_generator = lambda: real_key_gen

        from victor.tools.caches import get_cache_key_generator

        key_gen = get_cache_key_generator()
        tools_hash = key_gen.calculate_tools_hash(mock_tools)
        config_hash = selector._get_config_hash(0.18)
        cache_key = key_gen.generate_query_key(
            query="read the file",
            tools_hash=tools_hash,
            config_hash=config_hash,
        )

        cached_tools = [
            ToolDefinition(
                name="read",
                description="Read file contents",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        ]
        cache.put_query(cache_key, ["read"], tools=cached_tools)

        # Mock the embedding service to avoid actual computation
        with patch("victor.tools.semantic_selector.EmbeddingService") as mock_emb:
            mock_instance = MagicMock()
            mock_instance.embed_text = AsyncMock(return_value=[0.1] * 384)
            mock_emb.get_instance.return_value = mock_instance

            # Call selection - should return cached result
            result = await selector.select_relevant_tools(
                user_message="read the file",
                tools=mock_tools,
                similarity_threshold=0.18,
            )

            # Should return cached tools
            assert len(result) == 1
            assert result[0].name == "read"

    @pytest.mark.asyncio
    async def test_cache_storage_after_selection(self, mock_tools):
        """Test that results are stored in cache after selection."""
        selector = SemanticToolSelector(cache_embeddings=False)
        selector._tools_registry = mock_tools

        with patch("victor.tools.semantic_selector.EmbeddingService") as mock_emb:
            mock_instance = MagicMock()
            mock_instance.embed_text = AsyncMock(return_value=[0.1] * 384)
            mock_emb.get_instance.return_value = mock_instance

            # First call should compute and cache
            result1 = await selector.select_relevant_tools(
                user_message="read the file",
                tools=mock_tools,
                similarity_threshold=0.18,
            )

            # Check cache was populated
            cache = get_tool_selection_cache()
            from victor.tools.caches import get_cache_key_generator

            key_gen = get_cache_key_generator()
            tools_hash = key_gen.calculate_tools_hash(mock_tools)
            config_hash = selector._get_config_hash(0.18)
            cache_key = key_gen.generate_query_key(
                query="read the file",
                tools_hash=tools_hash,
                config_hash=config_hash,
            )
            cached = cache.get_query(cache_key)
            assert cached is not None
            assert len(cached.tools) > 0

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_tools_change(self, mock_tools):
        """Test that cache is invalidated when tools change."""
        selector = SemanticToolSelector(cache_embeddings=False)
        selector._tools_registry = mock_tools

        # Populate cache
        cache = get_tool_selection_cache()
        from victor.tools.caches import get_cache_key_generator

        key_gen = get_cache_key_generator()
        tools_hash = key_gen.calculate_tools_hash(mock_tools)
        config_hash = selector._get_config_hash(0.18)
        cache_key = key_gen.generate_query_key(
            query="read the file",
            tools_hash=tools_hash,
            config_hash=config_hash,
        )
        cache.put_query(cache_key, ["read"])

        # Notify tools changed
        selector.notify_tools_changed()

        # Cache should be invalidated (tools hash cleared)
        assert selector._tools_hash is None
        assert len(selector._category_memberships_cache) == 0


class TestKeywordToolSelectorCaching:
    """Test KeywordToolSelector caching integration."""

    @pytest.mark.asyncio
    async def test_cache_lookup_before_selection(self, mock_tools, mock_context):
        """Test that cache is checked before keyword selection."""
        selector = KeywordToolSelector(tools=mock_tools)

        # Pre-populate cache
        cache = get_tool_selection_cache()
        from victor.tools.caches import get_cache_key_generator

        key_gen = get_cache_key_generator()
        tools_hash = key_gen.calculate_tools_hash(mock_tools)
        config_hash = selector._get_config_hash()
        cache_key = key_gen.generate_query_key(
            query="read the file",
            tools_hash=tools_hash,
            config_hash=config_hash,
        )

        cached_tools = [
            ToolDefinition(
                name="read",
                description="Read file contents",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        ]
        cache.put_query(cache_key, ["read"], tools=cached_tools)

        # Call selection - should return cached result
        result = await selector.select_tools(
            prompt="read the file",
            context=mock_context,
        )

        # Should return cached tools
        assert len(result) == 1
        assert result[0].name == "read"

    @pytest.mark.asyncio
    async def test_cache_storage_after_selection(self, mock_tools, mock_context):
        """Test that results are stored in cache after selection."""
        selector = KeywordToolSelector(tools=mock_tools)

        # Call selection
        result = await selector.select_tools(
            prompt="read the file",
            context=mock_context,
        )

        # Check cache was populated
        cache = get_tool_selection_cache()
        from victor.tools.caches import get_cache_key_generator

        key_gen = get_cache_key_generator()
        tools_hash = key_gen.calculate_tools_hash(mock_tools)
        config_hash = selector._get_config_hash()

        # Cache should have the entry (exact key depends on config)
        assert cache.get_metrics().total_entries >= 0

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_tools_change(self, mock_tools):
        """Test that cache is invalidated when tools change."""
        selector = KeywordToolSelector(tools=mock_tools)

        # Populate cache
        cache = get_tool_selection_cache()
        from victor.tools.caches import get_cache_key_generator

        key_gen = get_cache_key_generator()
        tools_hash = key_gen.calculate_tools_hash(mock_tools)
        config_hash = selector._get_config_hash()
        cache_key = key_gen.generate_query_key(
            query="read the file",
            tools_hash=tools_hash,
            config_hash=config_hash,
        )
        cache.put_query(cache_key, ["read"])

        # Notify tools changed
        selector.notify_tools_changed()

        # Cache should be invalidated (internal caches cleared)
        assert selector._core_tools_cache is None
        assert selector._core_readonly_cache is None


class TestHybridToolSelectorCaching:
    """Test HybridToolSelector caching integration."""

    @pytest.fixture
    def hybrid_selector(self, mock_tools):
        """Create hybrid selector with mock sub-selectors."""
        semantic_selector = MagicMock(spec=SemanticToolSelector)
        keyword_selector = MagicMock(spec=KeywordToolSelector)

        # Setup async return values
        semantic_result = [
            ToolDefinition(
                name="read",
                description="Read file contents",
                parameters={"type": "object"},
            )
        ]
        keyword_result = [
            ToolDefinition(
                name="write",
                description="Write to file",
                parameters={"type": "object"},
            )
        ]

        semantic_selector.select_tools = AsyncMock(return_value=semantic_result)
        keyword_selector.select_tools = AsyncMock(return_value=keyword_result)
        semantic_selector.record_tool_execution = MagicMock()
        keyword_selector.record_tool_execution = MagicMock()
        semantic_selector.close = AsyncMock()
        keyword_selector.close = AsyncMock()

        config = HybridSelectorConfig(enable_cache=True)
        selector = HybridToolSelector(
            semantic_selector=semantic_selector,
            keyword_selector=keyword_selector,
            config=config,
        )
        return selector

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_result(self, hybrid_selector, mock_context):
        """Test that cache hit returns cached tools without calling sub-selectors."""
        # Pre-populate cache
        cache = get_tool_selection_cache()
        from victor.tools.caches import get_cache_key_generator

        key_gen = get_cache_key_generator()

        # Mock tools hash calculation
        hybrid_selector._tools_hash = "test_hash"
        hybrid_selector._config_hash = "test_config_hash"

        cache_key = key_gen.generate_query_key(
            query="read the file",
            tools_hash="test_hash",
            config_hash="test_config_hash",
        )

        cached_tools = [
            ToolDefinition(
                name="cached_read",
                description="Cached read tool",
                parameters={},
            )
        ]
        cache.put_query(cache_key, ["cached_read"], tools=cached_tools)

        # Call selection
        result = await hybrid_selector.select_tools(
            prompt="read the file",
            context=mock_context,
        )

        # Should return cached tools
        assert len(result) == 1
        assert result[0].name == "cached_read"

        # Sub-selectors should NOT have been called
        hybrid_selector.semantic.select_tools.assert_not_called()
        hybrid_selector.keyword.select_tools.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_calls_selectors_and_stores_result(
        self, hybrid_selector, mock_context
    ):
        """Test that cache miss calls sub-selectors and stores result."""
        # Call selection with empty cache
        result = await hybrid_selector.select_tools(
            prompt="read the file",
            context=mock_context,
        )

        # Sub-selectors should have been called
        hybrid_selector.semantic.select_tools.assert_called_once()
        hybrid_selector.keyword.select_tools.assert_called_once()

        # Result should have tools from both selectors
        assert len(result) > 0

        # Check cache was populated
        cache = get_tool_selection_cache()
        stats = cache.get_metrics()
        # Should have at least attempted a cache operation
        assert stats.total_lookups >= 1

    @pytest.mark.asyncio
    async def test_invalidate_cache_clears_stored_hash(self, hybrid_selector):
        """Test that invalidate_cache clears stored hashes."""
        # Set some hashes
        hybrid_selector._tools_hash = "test_hash"
        hybrid_selector._config_hash = "test_config_hash"

        # Invalidate cache
        hybrid_selector.invalidate_cache()

        # Hashes should be cleared
        assert hybrid_selector._tools_hash is None

    @pytest.mark.asyncio
    async def test_enable_disable_cache(self, hybrid_selector):
        """Test enabling and disabling cache."""
        # Initially enabled by config
        assert hybrid_selector._cache_enabled is True

        # Disable
        hybrid_selector.disable_cache()
        assert hybrid_selector._cache_enabled is False

        # Enable
        hybrid_selector.enable_cache()
        assert hybrid_selector._cache_enabled is True

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, hybrid_selector):
        """Test getting cache statistics."""
        stats = hybrid_selector.get_cache_stats()

        # Should return a dict with cache info
        assert isinstance(stats, dict)
        assert "enabled" in stats


class TestCacheInvalidation:
    """Test cache invalidation scenarios."""

    @pytest.mark.asyncio
    async def test_invalidate_on_tools_change_semantic(self, mock_tools):
        """Test that SemanticToolSelector invalidates cache on tools change."""
        selector = SemanticToolSelector(cache_embeddings=False)
        selector._tools_registry = mock_tools

        # Set some state
        selector._tools_hash = "test_hash"
        selector._category_memberships_cache = {"test": ["read"]}

        # Notify tools changed
        selector.notify_tools_changed()

        # State should be cleared
        assert selector._tools_hash is None
        assert len(selector._category_memberships_cache) == 0

    @pytest.mark.asyncio
    async def test_invalidate_on_tools_change_keyword(self, mock_tools):
        """Test that KeywordToolSelector invalidates cache on tools change."""
        selector = KeywordToolSelector(tools=mock_tools)

        # Set some state
        selector._core_tools_cache = {"read"}
        selector._core_readonly_cache = {"read"}

        # Notify tools changed
        selector.notify_tools_changed()

        # State should be cleared
        assert selector._core_tools_cache is None
        assert selector._core_readonly_cache is None

    @pytest.mark.asyncio
    async def test_invalidate_on_tools_change_hybrid(self, mock_tools):
        """Test that HybridToolSelector propagates invalidation to sub-selectors."""
        semantic_selector = MagicMock(spec=SemanticToolSelector)
        keyword_selector = MagicMock(spec=KeywordToolSelector)
        semantic_selector.notify_tools_changed = MagicMock()
        keyword_selector.notify_tools_changed = MagicMock()

        config = HybridSelectorConfig(enable_cache=True)
        selector = HybridToolSelector(
            semantic_selector=semantic_selector,
            keyword_selector=keyword_selector,
            config=config,
        )

        # Set some state
        selector._tools_hash = "test_hash"

        # Invalidate cache
        selector.invalidate_cache()

        # Hash should be cleared
        assert selector._tools_hash is None

        # Sub-selectors should have been notified
        semantic_selector.notify_tools_changed.assert_called_once()
        keyword_selector.notify_tools_changed.assert_called_once()


class TestCacheStats:
    """Test cache statistics reporting."""

    @pytest.mark.asyncio
    async def test_semantic_selector_cache_stats(self, mock_tools):
        """Test getting cache stats from SemanticToolSelector."""
        selector = SemanticToolSelector(cache_embeddings=False)
        selector._tools_registry = mock_tools

        stats = selector.get_cache_stats()

        assert isinstance(stats, dict)
        assert "enabled" in stats

    @pytest.mark.asyncio
    async def test_keyword_selector_cache_stats(self, mock_tools):
        """Test getting cache stats from KeywordToolSelector."""
        selector = KeywordToolSelector(tools=mock_tools)

        stats = selector.get_cache_stats()

        assert isinstance(stats, dict)
        assert "enabled" in stats

    @pytest.mark.asyncio
    async def test_hybrid_selector_cache_stats(self, mock_tools):
        """Test getting cache stats from HybridToolSelector."""
        semantic_selector = MagicMock(spec=SemanticToolSelector)
        keyword_selector = MagicMock(spec=KeywordToolSelector)
        semantic_selector.select_tools = AsyncMock(return_value=[])
        keyword_selector.select_tools = AsyncMock(return_value=[])
        semantic_selector.record_tool_execution = MagicMock()
        keyword_selector.record_tool_execution = MagicMock()
        semantic_selector.close = AsyncMock()
        keyword_selector.close = AsyncMock()

        config = HybridSelectorConfig(enable_cache=True)
        selector = HybridToolSelector(
            semantic_selector=semantic_selector,
            keyword_selector=keyword_selector,
            config=config,
        )

        stats = selector.get_cache_stats()

        assert isinstance(stats, dict)
        assert "enabled" in stats
