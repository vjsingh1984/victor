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

"""Tests for response cache optimization."""

import asyncio
import pytest
import time

from victor.core.cache import (
    ResponseCache,
    CacheEntry,
    CacheStats,
    get_response_cache,
    reset_response_cache,
)
from victor.providers.base import Message, CompletionResponse


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        response = CompletionResponse(content="Test response")
        entry = CacheEntry(
            key="test_key",
            response=response,
            ttl=3600,
        )

        assert entry.key == "test_key"
        assert entry.response.content == "Test response"
        assert entry.ttl == 3600
        assert entry.access_count == 0
        assert entry.timestamp > 0

    def test_cache_entry_expiration(self):
        """Test cache entry expiration."""
        response = CompletionResponse(content="Test response")
        entry = CacheEntry(
            key="test_key",
            response=response,
            ttl=1,  # 1 second TTL
            timestamp=time.time() - 2,  # Created 2 seconds ago
        )

        assert entry.is_expired()

        entry_no_ttl = CacheEntry(
            key="test_key",
            response=response,
            ttl=None,  # No expiration
        )

        assert not entry_no_ttl.is_expired()

    def test_cache_entry_touch(self):
        """Test updating access statistics."""
        response = CompletionResponse(content="Test response")
        entry = CacheEntry(
            key="test_key",
            response=response,
        )

        initial_count = entry.access_count
        entry.touch()

        assert entry.access_count == initial_count + 1
        assert entry.last_access > entry.timestamp


class TestCacheStats:
    """Test CacheStats thread-safe statistics."""

    def test_cache_stats_initialization(self):
        """Test stats initialization."""
        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.semantic_hits == 0

    def test_cache_stats_record_hit(self):
        """Test recording cache hits."""
        stats = CacheStats()

        stats.record_hit(semantic=False)
        assert stats.hits == 1
        assert stats.semantic_hits == 0

        stats.record_hit(semantic=True)
        assert stats.hits == 2
        assert stats.semantic_hits == 1

    def test_cache_stats_record_miss(self):
        """Test recording cache misses."""
        stats = CacheStats()

        stats.record_miss()
        assert stats.misses == 1

    def test_cache_stats_hit_rate(self):
        """Test hit rate calculation."""
        stats = CacheStats()

        assert stats.get_hit_rate() == 0.0

        stats.record_hit()
        stats.record_miss()
        assert stats.get_hit_rate() == 0.5

    def test_cache_stats_thread_safety(self):
        """Test that stats are thread-safe."""
        import threading

        stats = CacheStats()

        def record_hits():
            for _ in range(100):
                stats.record_hit()

        def record_misses():
            for _ in range(100):
                stats.record_miss()

        threads = [
            threading.Thread(target=record_hits),
            threading.Thread(target=record_misses),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert stats.hits == 100
        assert stats.misses == 100


class TestResponseCache:
    """Test ResponseCache implementation."""

    @pytest.fixture
    def cache(self):
        """Create a test cache instance."""
        return ResponseCache(
            max_size=10,
            default_ttl=3600,
            enable_semantic=False,  # Disable for faster tests
        )

    @pytest.fixture
    def sample_messages(self):
        """Create sample messages."""
        return [
            Message(role="user", content="What is 2+2?"),
        ]

    @pytest.fixture
    def sample_response(self):
        """Create sample response."""
        return CompletionResponse(
            content="2+2 equals 4.",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )

    @pytest.mark.asyncio
    async def test_cache_put_and_get(self, cache, sample_messages, sample_response):
        """Test basic cache put and get operations."""
        # Put in cache
        key = await cache.put(sample_messages, sample_response)
        assert key is not None

        # Get from cache
        cached_response = await cache.get(sample_messages)

        assert cached_response is not None
        assert cached_response.content == sample_response.content
        assert cached_response.model == sample_response.model

    @pytest.mark.asyncio
    async def test_cache_miss(self, cache, sample_messages):
        """Test cache miss."""
        response = await cache.get(sample_messages)

        assert response is None
        assert cache.stats.misses == 1

    @pytest.mark.asyncio
    async def test_cache_hit(self, cache, sample_messages, sample_response):
        """Test cache hit."""
        await cache.put(sample_messages, sample_response)
        await cache.get(sample_messages)

        assert cache.stats.hits == 1

    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache, sample_messages, sample_response):
        """Test cache entry expiration."""
        # Put with short TTL
        await cache.put(sample_messages, sample_response, ttl=1)

        # Wait for expiration
        await asyncio.sleep(1.5)

        # Should be expired
        response = await cache.get(sample_messages)
        assert response is None

    @pytest.mark.asyncio
    async def test_cache_eviction(self, cache, sample_messages, sample_response):
        """Test LRU eviction when cache is full."""
        # Fill cache to max size
        for i in range(cache.max_size + 5):
            messages = [Message(role="user", content=f"Message {i}")]
            await cache.put(messages, sample_response)

        # Cache should be at max size
        assert cache.get_size() == cache.max_size

        # Oldest entries should be evicted
        old_messages = [Message(role="user", content="Message 0")]
        response = await cache.get(old_messages)
        assert response is None

    @pytest.mark.asyncio
    async def test_cache_clear(self, cache, sample_messages, sample_response):
        """Test clearing the cache."""
        await cache.put(sample_messages, sample_response)
        assert cache.get_size() > 0

        cache.clear()
        assert cache.get_size() == 0

    @pytest.mark.asyncio
    async def test_cache_stats(self, cache, sample_messages, sample_response):
        """Test cache statistics."""
        # Add some entries
        await cache.put(sample_messages, sample_response)

        # Hit
        await cache.get(sample_messages)

        # Miss
        await cache.get([Message(role="user", content="Different message")])

        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, cache):
        """Test that cache keys are generated consistently."""
        messages1 = [
            Message(role="user", content="Test"),
        ]
        messages2 = [
            Message(role="user", content="Test"),
        ]

        key1 = cache._generate_key(messages1)
        key2 = cache._generate_key(messages2)

        assert key1 == key2

        # Different content should produce different key
        messages3 = [Message(role="user", content="Different")]
        key3 = cache._generate_key(messages3)

        assert key1 != key3


class TestResponseCacheWithSemantic:
    """Test ResponseCache with semantic similarity."""

    @pytest.fixture
    def semantic_cache(self):
        """Create a cache with semantic matching enabled."""
        return ResponseCache(
            max_size=10,
            default_ttl=3600,
            enable_semantic=True,
            semantic_threshold=0.85,
        )

    @pytest.mark.asyncio
    async def test_semantic_similarity_disabled(self, semantic_cache):
        """Test that semantic matching is disabled when embedding service fails."""
        # Mock embedding service to return None
        semantic_cache._embedding_service = None

        messages = [Message(role="user", content="Test message")]
        response = CompletionResponse(content="Test response")

        await semantic_cache.put(messages, response)

        # Should fall back to exact match
        cached = await semantic_cache.get(messages)
        assert cached is not None

        # Similar message should not match without embeddings
        similar_messages = [Message(role="user", content="Similar test message")]
        similar_cached = await semantic_cache.get_similar(similar_messages)
        assert similar_cached is None


class TestGlobalResponseCache:
    """Test global response cache instance."""

    def test_get_global_cache(self):
        """Test getting global cache instance."""
        reset_response_cache()

        cache1 = get_response_cache()
        cache2 = get_response_cache()

        assert cache1 is cache2

    def test_reset_global_cache(self):
        """Test resetting global cache."""
        cache = get_response_cache()
        cache.clear()

        reset_response_cache()

        new_cache = get_response_cache()
        assert new_cache is not cache
        assert new_cache.get_size() == 0


@pytest.mark.integration
class TestResponseCacheIntegration:
    """Integration tests for response cache."""

    @pytest.mark.asyncio
    async def test_cache_with_real_embeddings(self):
        """Test cache with real embedding service (if available)."""
        try:
            from victor.agents.embeddings import EmbeddingService

            cache = ResponseCache(
                max_size=10,
                enable_semantic=True,
            )

            messages = [Message(role="user", content="What is Python?")]
            response = CompletionResponse(content="Python is a programming language.")

            await cache.put(messages, response)

            # Exact match should work
            cached = await cache.get(messages)
            assert cached is not None

            # Semantic match might work if embeddings available
            similar_messages = [Message(role="user", content="Tell me about Python")]
            similar_cached = await cache.get_similar(similar_messages)

            # May or may not match depending on threshold
            # Just ensure it doesn't error
            assert similar_cached is None or isinstance(similar_cached, CompletionResponse)

        except ImportError:
            pytest.skip("Embedding service not available")
