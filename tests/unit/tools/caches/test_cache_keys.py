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

"""Tests for cache key generation."""

import pytest

from victor.tools.caches.cache_keys import (
    CacheKeyGenerator,
    calculate_tools_hash,
    generate_context_key,
    generate_query_key,
    get_cache_key_generator,
)


class TestCacheKeyGenerator:
    """Tests for CacheKeyGenerator class."""

    def test_init(self):
        """Test CacheKeyGenerator initialization."""
        gen = CacheKeyGenerator()
        assert gen is not None
        assert gen._tools_hash_cache is None
        assert gen.HASH_TRUNCATE_LENGTH == 16

    def test_generate_query_key(self):
        """Test query key generation."""
        gen = CacheKeyGenerator()

        key1 = gen.generate_query_key("read the file", "abc123", "def456")
        key2 = gen.generate_query_key("read the file", "abc123", "def456")
        key3 = gen.generate_query_key("write the file", "abc123", "def456")

        # Same inputs should produce same key
        assert key1 == key2
        assert len(key1) == 16

        # Different query should produce different key
        assert key1 != key3

    def test_generate_query_key_case_insensitive(self):
        """Test that query keys are case-insensitive."""
        gen = CacheKeyGenerator()

        key1 = gen.generate_query_key("Read the File", "abc123", "def456")
        key2 = gen.generate_query_key("read the file", "abc123", "def456")

        assert key1 == key2

    def test_generate_query_key_whitespace_normalized(self):
        """Test that query keys normalize whitespace."""
        gen = CacheKeyGenerator()

        key1 = gen.generate_query_key("read the file", "abc123", "def456")
        key2 = gen.generate_query_key("  read the file  ", "abc123", "def456")

        assert key1 == key2

    def test_generate_context_key(self):
        """Test context-aware key generation."""
        gen = CacheKeyGenerator()

        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

        key1 = gen.generate_context_key("read the file", "abc123", history, ["edit"])
        key2 = gen.generate_context_key("read the file", "abc123", history, ["edit"])
        key3 = gen.generate_context_key("read the file", "abc123", None, None)

        # Same inputs should produce same key
        assert key1 == key2
        assert len(key1) == 16

        # Different context should produce different key
        assert key1 != key3

    def test_generate_context_key_empty_history(self):
        """Test context key with empty history."""
        gen = CacheKeyGenerator()

        key1 = gen.generate_context_key("read the file", "abc123", None, None)
        key2 = gen.generate_context_key("read the file", "abc123", [], None)

        # Empty history and None should produce same result
        assert key1 == key2

    def test_generate_context_key_history_truncation(self):
        """Test that context key only uses last N messages."""
        gen = CacheKeyGenerator()

        short_history = [{"role": "user", "content": "last message"}]
        long_history = [
            {"role": "user", "content": f"message {i}"} for i in range(100)
        ]

        key1 = gen.generate_context_key("read the file", "abc123", long_history, None)
        key2 = gen.generate_context_key("read the file", "abc123", short_history, None)

        # Different histories should produce different keys
        assert key1 != key2

        # But keys should be deterministic
        key3 = gen.generate_context_key("read the file", "abc123", long_history, None)
        assert key1 == key3

    def test_generate_rl_key(self):
        """Test RL ranking key generation."""
        gen = CacheKeyGenerator()

        key1 = gen.generate_rl_key("analysis", "abc123", 12)
        key2 = gen.generate_rl_key("analysis", "abc123", 12)
        key3 = gen.generate_rl_key("action", "abc123", 12)

        # Same inputs should produce same key
        assert key1 == key2
        assert len(key1) == 16

        # Different task type should produce different key
        assert key1 != key3

    def test_generate_rl_key_hour_bucket(self):
        """Test that RL key includes hour bucket."""
        gen = CacheKeyGenerator()

        key1 = gen.generate_rl_key("analysis", "abc123", 10)
        key2 = gen.generate_rl_key("analysis", "abc123", 11)

        # Different hour buckets should produce different keys
        assert key1 != key2

    def test_calculate_config_hash(self):
        """Test configuration hash calculation."""
        gen = CacheKeyGenerator()

        hash1 = gen.calculate_config_hash(0.7, 0.3, 10, 0.18)
        hash2 = gen.calculate_config_hash(0.7, 0.3, 10, 0.18)
        hash3 = gen.calculate_config_hash(0.5, 0.5, 10, 0.18)

        # Same config should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 16

        # Different config should produce different hash
        assert hash1 != hash3


class TestGlobalFunctions:
    """Tests for global convenience functions."""

    def test_get_cache_key_generator_singleton(self):
        """Test that global key generator is a singleton."""
        gen1 = get_cache_key_generator()
        gen2 = get_cache_key_generator()

        assert gen1 is gen2

    def test_generate_query_key_function(self):
        """Test global generate_query_key function."""
        key1 = generate_query_key("read the file", "abc123", "def456")
        key2 = generate_query_key("read the file", "abc123", "def456")

        assert key1 == key2
        assert len(key1) == 16

    def test_generate_context_key_function(self):
        """Test global generate_context_key function."""
        history = [{"role": "user", "content": "test"}]

        key1 = generate_context_key("read the file", "abc123", history, None)
        key2 = generate_context_key("read the file", "abc123", history, None)

        assert key1 == key2
        assert len(key1) == 16


class TestToolsHashCalculation:
    """Tests for tools registry hash calculation."""

    def test_calculate_tools_hash_caching(self):
        """Test that tools hash is cached."""
        gen = CacheKeyGenerator()

        # Create a mock tool registry
        class MockTool:
            def __init__(self, name: str, description: str = "", parameters: dict = None):
                self.name = name
                self.description = description
                self.parameters = parameters or {}

        class MockRegistry:
            def __init__(self, tools):
                self._tools = tools

            def list_tools(self):
                return self._tools

        tools = [
            MockTool("read", "Read a file", {}),
            MockTool("write", "Write a file", {}),
        ]
        registry = MockRegistry(tools)

        # First call should calculate hash
        hash1 = gen.calculate_tools_hash(registry)

        # Second call should return cached hash
        hash2 = gen.calculate_tools_hash(registry)

        assert hash1 == hash2

        # Verify cache was used by checking internal state
        assert gen._tools_hash_cache is not None

    def test_calculate_tools_hash_different_registries(self):
        """Test that different registries produce different hashes."""
        gen = CacheKeyGenerator()

        class MockTool:
            def __init__(self, name: str):
                self.name = name
                self.description = ""
                self.parameters = {}

        class MockRegistry:
            def __init__(self, tools):
                self._tools = tools

            def list_tools(self):
                return self._tools

        registry1 = MockRegistry([MockTool("read"), MockTool("write")])
        registry2 = MockRegistry([MockTool("read"), MockTool("edit")])

        hash1 = gen.calculate_tools_hash(registry1)

        # Invalidate cache
        gen.invalidate_tools_cache()

        hash2 = gen.calculate_tools_hash(registry2)

        assert hash1 != hash2

    def test_invalidate_tools_cache(self):
        """Test tools cache invalidation."""
        gen = CacheKeyGenerator()

        class MockTool:
            def __init__(self, name: str):
                self.name = name
                self.description = ""
                self.parameters = {}

        class MockRegistry:
            def __init__(self, tools):
                self._tools = tools

            def list_tools(self):
                return self._tools

        registry = MockRegistry([MockTool("read")])

        # Calculate hash
        hash1 = gen.calculate_tools_hash(registry)
        assert gen._tools_hash_cache is not None

        # Invalidate
        gen.invalidate_tools_cache()
        assert gen._tools_hash_cache is None
        assert gen._tools_hash_registry_id is None
