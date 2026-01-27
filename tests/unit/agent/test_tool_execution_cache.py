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

"""Tests for ToolExecutionDecisionCache.

Tests the hot path optimization cache for tool execution decisions.
"""
import pytest
from unittest.mock import Mock

from victor.agent.tool_execution_cache import (
    ToolExecutionDecisionCache,
    ToolValidationResult,
    ToolNormalizationResult,
)


class TestToolValidationResult:
    """Tests for ToolValidationResult dataclass."""

    def test_creation(self):
        """Test creating a validation result."""
        result = ToolValidationResult(is_valid=True, tool_exists=True, is_enabled=True)
        assert result.is_valid is True
        assert result.tool_exists is True
        assert result.is_enabled is True

    def test_creation_invalid(self):
        """Test creating an invalid validation result."""
        result = ToolValidationResult(is_valid=False, tool_exists=False, is_enabled=False)
        assert result.is_valid is False
        assert result.tool_exists is False
        assert result.is_enabled is False


class MockToolRegistry:
    """Mock tool registry for testing."""

    def __init__(self, enabled_tools=None):
        self._tools = enabled_tools or ["test_tool", "read", "write"]

    def is_tool_enabled(self, name):
        return name in self._tools

    def has_tool(self, name):
        return name in self._tools

    def get(self, name):
        """Get tool by name, returns mock object if exists."""
        if name in self._tools:
            return Mock()  # Return a mock tool object
        return None


class MockNormalizer:
    """Mock argument normalizer for testing."""

    def __init__(self):
        self.call_count = 0

    def normalize_arguments(self, args, tool_name):
        self.call_count += 1
        return args, "direct"


class TestToolExecutionDecisionCache:
    """Tests for ToolExecutionDecisionCache."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = ToolExecutionDecisionCache(max_size=100)
        assert cache._max_size == 100
        assert cache._hits == 0
        assert cache._misses == 0
        assert len(cache._validation_cache) == 0
        assert len(cache._normalization_cache) == 0

    def test_cache_validation_miss_then_hit(self):
        """Test validation caching: first call is miss, second is hit."""
        cache = ToolExecutionDecisionCache()
        registry = MockToolRegistry()

        # First call - cache miss
        result1 = cache.is_valid_tool("test_tool", registry)
        assert result1.is_valid is True
        assert result1.tool_exists is True
        assert result1.is_enabled is True

        # Second call - cache hit
        result2 = cache.is_valid_tool("test_tool", registry)
        assert result2.is_valid is True
        assert result2.tool_exists is True
        assert result2.is_enabled is True

        # Check stats
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["validation_cache_size"] == 1

    def test_cache_validation_invalid_tool(self):
        """Test validation caching with invalid tool."""
        cache = ToolExecutionDecisionCache()
        registry = MockToolRegistry(enabled_tools=["test_tool"])

        # Query non-existent tool
        result = cache.is_valid_tool("nonexistent_tool", registry)
        assert result.is_valid is False
        assert result.tool_exists is False
        assert result.is_enabled is False

        # Check stats
        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["validation_cache_size"] == 1

    def test_cache_normalization_miss_then_hit(self):
        """Test normalization caching: first call is miss, second is hit."""
        cache = ToolExecutionDecisionCache()
        normalizer = MockNormalizer()

        args = {"path": "/test", "pattern": "foo"}

        # First call - cache miss
        result1 = cache.get_normalized_args("test_tool", args, normalizer)
        assert result1.normalized_args == args
        assert result1.strategy == "direct"
        assert len(result1.signature) > 0  # Signature should be non-empty
        assert "test_tool" in result1.signature  # Signature contains tool name
        assert normalizer.call_count == 1

        # Second call - cache hit (normalizer not called again)
        result2 = cache.get_normalized_args("test_tool", args, normalizer)
        assert result2.normalized_args == args
        assert result2.strategy == "direct"
        assert normalizer.call_count == 1  # Still 1, not called again

        # Check stats
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["normalization_cache_size"] == 1

    def test_cache_normalization_different_args(self):
        """Test normalization caching with different arguments."""
        cache = ToolExecutionDecisionCache()
        normalizer = MockNormalizer()

        args1 = {"path": "/test1"}
        args2 = {"path": "/test2"}

        # First call
        result1 = cache.get_normalized_args("test_tool", args1, normalizer)
        assert result1.normalized_args == args1

        # Second call with different args - cache miss
        result2 = cache.get_normalized_args("test_tool", args2, normalizer)
        assert result2.normalized_args == args2

        # Check stats - both should be misses
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 2
        assert stats["normalization_cache_size"] == 2

    def test_cache_normalization_different_tools(self):
        """Test normalization caching with different tools."""
        cache = ToolExecutionDecisionCache()
        normalizer = MockNormalizer()

        args = {"path": "/test"}

        # Call with different tool names
        result1 = cache.get_normalized_args("read", args, normalizer)
        result2 = cache.get_normalized_args("write", args, normalizer)

        assert result1.normalized_args == args
        assert result2.normalized_args == args

        # Should be 2 separate cache entries
        stats = cache.get_stats()
        assert stats["misses"] == 2
        assert stats["normalization_cache_size"] == 2

    def test_cache_eviction_validation(self):
        """Test FIFO eviction when validation cache exceeds max size."""
        cache = ToolExecutionDecisionCache(max_size=2)
        registry = MockToolRegistry(enabled_tools=["tool1", "tool2", "tool3"])

        # Add 3 items (should evict first)
        cache.is_valid_tool("tool1", registry)
        cache.is_valid_tool("tool2", registry)
        cache.is_valid_tool("tool3", registry)

        stats = cache.get_stats()
        assert stats["validation_cache_size"] == 2

        # Query tool1 again - should be a miss (was evicted)
        cache.is_valid_tool("tool1", registry)
        stats = cache.get_stats()
        assert stats["misses"] == 4  # 3 initial + 1 re-query

    def test_cache_eviction_normalization(self):
        """Test FIFO eviction when normalization cache exceeds max size."""
        cache = ToolExecutionDecisionCache(max_size=2)
        normalizer = MockNormalizer()

        # Add 3 items with different args
        cache.get_normalized_args("tool", {"path": "/test1"}, normalizer)
        cache.get_normalized_args("tool", {"path": "/test2"}, normalizer)
        cache.get_normalized_args("tool", {"path": "/test3"}, normalizer)

        stats = cache.get_stats()
        assert stats["normalization_cache_size"] == 2

    def test_clear(self):
        """Test clearing the cache."""
        cache = ToolExecutionDecisionCache()
        registry = MockToolRegistry()
        normalizer = MockNormalizer()

        # Add some entries
        cache.is_valid_tool("test_tool", registry)
        cache.get_normalized_args("test_tool", {"path": "/test"}, normalizer)

        assert cache.get_stats()["validation_cache_size"] == 1
        assert cache.get_stats()["normalization_cache_size"] == 1
        assert cache.get_stats()["hits"] == 0
        assert cache.get_stats()["misses"] == 2

        # Clear cache
        cache.clear()

        assert cache.get_stats()["validation_cache_size"] == 0
        assert cache.get_stats()["normalization_cache_size"] == 0
        assert cache.get_stats()["hits"] == 0
        assert cache.get_stats()["misses"] == 0

    def test_get_stats_comprehensive(self):
        """Test get_stats returns all expected fields."""
        cache = ToolExecutionDecisionCache()
        registry = MockToolRegistry()
        normalizer = MockNormalizer()

        # Add some entries
        cache.is_valid_tool("test_tool", registry)
        cache.is_valid_tool("test_tool", registry)  # Hit
        cache.get_normalized_args("test_tool", {"path": "/test"}, normalizer)

        stats = cache.get_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "validation_cache_size" in stats
        assert "normalization_cache_size" in stats

        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert 0.0 <= stats["hit_rate"] <= 1.0

    def test_empty_cache_hit_rate(self):
        """Test hit rate is 0.0 for empty cache."""
        cache = ToolExecutionDecisionCache()
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.0

    def test_normalization_with_complex_args(self):
        """Test normalization caching with complex argument structures."""
        cache = ToolExecutionDecisionCache()
        normalizer = MockNormalizer()

        # Test with nested args
        args = {
            "path": "/test",
            "options": {"recursive": True, "pattern": "*.py"},
            "files": ["file1.py", "file2.py"],
        }

        result = cache.get_normalized_args("test_tool", args, normalizer)
        assert result.normalized_args == args
        assert len(result.signature) > 0  # Signature should be non-empty
        assert "test_tool" in result.signature  # Signature contains tool name

    def test_normalization_with_unhashable_args(self):
        """Test normalization caching handles unhashable values gracefully."""
        cache = ToolExecutionDecisionCache()
        normalizer = MockNormalizer()

        # Args with list values (normally unhashable)
        args = {"files": ["file1.py", "file2.py"]}

        # Should not raise an error
        result = cache.get_normalized_args("test_tool", args, normalizer)
        assert result.normalized_args == args

        # Second call should hit cache
        result2 = cache.get_normalized_args("test_tool", args, normalizer)
        assert result2.normalized_args == args

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
