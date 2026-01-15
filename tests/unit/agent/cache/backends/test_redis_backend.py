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

"""Tests for Redis cache backend implementation.

Tests the RedisCacheBackend with mocked Redis client to avoid
requiring a real Redis instance for unit tests.
"""

import json
import pickle
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from uuid import uuid4

import pytest

from victor.agent.cache.backends.redis import RedisCacheBackend


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_redis(mock_pubsub):
    """Create a mock Redis client."""
    # Create an empty async iterator for default scan_iter
    class EmptyAsyncIterator:
        def __aiter__(self):
            return self
        async def __anext__(self):
            raise StopAsyncIteration

    redis = AsyncMock()
    redis.ping = AsyncMock(return_value=True)
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.setex = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.info = AsyncMock(return_value={"dbkeys": 10, "used_memory": 1024})
    # scan_iter returns an async iterator, not awaitable, so use Mock not AsyncMock
    redis.scan_iter = Mock(return_value=EmptyAsyncIterator())
    redis.publish = AsyncMock(return_value=1)
    redis.pubsub = Mock(return_value=mock_pubsub)
    return redis


@pytest.fixture
def mock_pubsub():
    """Create a mock Redis pubsub client."""
    # Create an empty async iterator for default listen
    class EmptyAsyncIterator:
        def __aiter__(self):
            return self
        async def __anext__(self):
            raise StopAsyncIteration

    pubsub = AsyncMock()
    pubsub.subscribe = AsyncMock(return_value=None)
    pubsub.psubscribe = AsyncMock(return_value=None)  # Pattern subscribe
    # listen returns an async iterator, not awaitable, so use Mock not AsyncMock
    pubsub.listen = Mock(return_value=EmptyAsyncIterator())
    pubsub.close = AsyncMock(return_value=None)
    return pubsub


@pytest.fixture
def redis_backend(mock_redis, mock_pubsub):
    """Create a Redis backend with mocked dependencies.

    Note: This fixture is deprecated in favor of creating backends directly
    in tests with proper patching. The with-context approach doesn't work
    with fixtures because the patch is undone when the fixture returns.
    """
    # Create backend without patching
    backend = RedisCacheBackend(
        redis_url="redis://localhost:6379/0",
        key_prefix="victor",
        default_ttl_seconds=3600,
    )
    # Store mocks for tests to use
    backend._mock_redis = mock_redis
    backend._mock_pubsub = mock_pubsub
    # Manually set pubsub
    backend._pubsub = mock_pubsub
    # Override connect to use mocks directly without calling original
    async def mocked_connect():
        # Skip actual connection, just mark as connected
        backend._is_connected = True
        backend._redis = mock_redis
        backend._pubsub = mock_pubsub

    backend.connect = mocked_connect
    return backend


# =============================================================================
# Connection Lifecycle Tests
# =============================================================================


class TestConnectionLifecycle:
    """Tests for connect/disconnect lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_establishes_connection(self, mock_redis):
        """Test that connect() establishes Redis connection."""
        with patch("victor.agent.cache.backends.redis.aioredis.from_url", new=AsyncMock(return_value=mock_redis)):
            backend = RedisCacheBackend(redis_url="redis://localhost:6379/0")

            await backend.connect()

            assert backend._redis is not None
            assert backend._pubsub is not None
            mock_redis.pubsub.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_creates_pubsub(self, mock_redis, mock_pubsub):
        """Test that connect() creates pubsub client."""
        with patch("victor.agent.cache.backends.redis.aioredis.from_url", new=AsyncMock(return_value=mock_redis)):
            backend = RedisCacheBackend(redis_url="redis://localhost:6379/0")

            await backend.connect()

            mock_redis.pubsub.assert_called_once()
            assert backend._pubsub is not None

    @pytest.mark.asyncio
    async def test_disconnect_closes_pubsub(self, mock_redis, mock_pubsub):
        """Test that disconnect() closes pubsub."""
        with patch("victor.agent.cache.backends.redis.aioredis.from_url", new=AsyncMock(return_value=mock_redis)):
            backend = RedisCacheBackend(
                redis_url="redis://localhost:6379/0",
                key_prefix="victor",
                default_ttl_seconds=3600,
            )
            backend._pubsub = mock_pubsub

            await backend.connect()
            await backend.disconnect()

            mock_pubsub.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_closes_redis(self, mock_redis, mock_pubsub):
        """Test that disconnect() closes Redis connection."""
        with patch("victor.agent.cache.backends.redis.aioredis.from_url", new=AsyncMock(return_value=mock_redis)):
            backend = RedisCacheBackend(
                redis_url="redis://localhost:6379/0",
                key_prefix="victor",
                default_ttl_seconds=3600,
            )
            backend._pubsub = mock_pubsub

            await backend.connect()
            await backend.disconnect()

            mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_connect_calls_idempotent(self, mock_redis, mock_pubsub):
        """Test that multiple connect() calls are idempotent."""
        with patch("victor.agent.cache.backends.redis.aioredis.from_url", new=AsyncMock(return_value=mock_redis)):
            backend = RedisCacheBackend(
                redis_url="redis://localhost:6379/0",
                key_prefix="victor",
                default_ttl_seconds=3600,
            )
            backend._pubsub = mock_pubsub

            await backend.connect()
            first_redis = backend._redis

            await backend.connect()

            assert backend._redis is first_redis

    @pytest.mark.asyncio
    async def test_multiple_disconnect_calls_safe(self, mock_redis, mock_pubsub):
        """Test that multiple disconnect() calls are safe."""
        with patch("victor.agent.cache.backends.redis.aioredis.from_url", new=AsyncMock(return_value=mock_redis)):
            backend = RedisCacheBackend(
                redis_url="redis://localhost:6379/0",
                key_prefix="victor",
                default_ttl_seconds=3600,
            )
            backend._pubsub = mock_pubsub

            await backend.connect()
            await backend.disconnect()

            # Should not raise
            await backend.disconnect()

            # Close only called once
            assert mock_pubsub.close.call_count == 1


# =============================================================================
# Basic Cache Operations Tests
# =============================================================================


class TestBasicCacheOperations:
    """Tests for basic cache operations (get, set, delete)."""

    @pytest.mark.asyncio
    async def test_get_returns_cached_value(self, redis_backend, mock_redis):
        """Test that get() returns cached value."""
        await redis_backend.connect()

        value = {"result": "data"}
        mock_redis.get.return_value = pickle.dumps(value)

        result = await redis_backend.get("key1", "test_namespace")

        assert result == value
        mock_redis.get.assert_called_once_with("victor:test_namespace:key1")

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing_key(self, redis_backend, mock_redis):
        """Test that get() returns None when key doesn't exist."""
        await redis_backend.connect()

        mock_redis.get.return_value = None

        result = await redis_backend.get("missing_key", "test_namespace")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_stores_value(self, redis_backend, mock_redis):
        """Test that set() stores value in Redis."""
        await redis_backend.connect()

        value = {"result": "data"}

        await redis_backend.set("key1", value, "test_namespace")

        # Check that setex was called with TTL
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        key, ttl, serialized = call_args[0]
        assert key == "victor:test_namespace:key1"
        assert ttl == 3600  # default TTL
        assert pickle.loads(serialized) == value

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self, redis_backend, mock_redis):
        """Test that set() uses custom TTL when provided."""
        await redis_backend.connect()

        value = {"result": "data"}

        await redis_backend.set("key1", value, "test_namespace", ttl_seconds=600)

        # Check that setex was called with custom TTL
        call_args = mock_redis.setex.call_args
        ttl = call_args[0][1]
        assert ttl == 600

    @pytest.mark.asyncio
    async def test_delete_removes_key(self, redis_backend, mock_redis):
        """Test that delete() removes key from Redis."""
        await redis_backend.connect()

        mock_redis.delete.return_value = 1

        result = await redis_backend.delete("key1", "test_namespace")

        assert result is True
        mock_redis.delete.assert_called_once_with("victor:test_namespace:key1")

    @pytest.mark.asyncio
    async def test_delete_returns_false_for_missing_key(self, redis_backend, mock_redis):
        """Test that delete() returns False when key doesn't exist."""
        await redis_backend.connect()

        mock_redis.delete.return_value = 0

        result = await redis_backend.delete("missing_key", "test_namespace")

        assert result is False


# =============================================================================
# Namespace Management Tests
# =============================================================================


class TestNamespaceManagement:
    """Tests for namespace isolation and management."""

    @pytest.mark.asyncio
    async def test_clear_namespace_deletes_all_keys(self, redis_backend, mock_redis):
        """Test that clear_namespace() deletes all keys in namespace."""
        await redis_backend.connect()

        # Mock scan_iter to return keys (async iterator)
        keys = [
            "victor:test_namespace:key1",
            "victor:test_namespace:key2",
            "victor:test_namespace:key3",
        ]

        # Create an async iterator
        class AsyncIterator:
            def __init__(self, items):
                self.items = items

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.items:
                    return self.items.pop(0)
                raise StopAsyncIteration

        mock_redis.scan_iter.return_value = AsyncIterator(keys[:])  # Copy to avoid mutation
        mock_redis.delete.return_value = 3

        count = await redis_backend.clear_namespace("test_namespace")

        assert count == 3
        mock_redis.delete.assert_called_once_with(*keys)

    @pytest.mark.asyncio
    async def test_clear_namespace_returns_zero_for_empty_namespace(self, redis_backend, mock_redis):
        """Test that clear_namespace() returns 0 when namespace is empty."""
        await redis_backend.connect()

        # Create an empty async iterator
        class AsyncIterator:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        mock_redis.scan_iter.return_value = AsyncIterator()

        count = await redis_backend.clear_namespace("empty_namespace")

        assert count == 0

    @pytest.mark.asyncio
    async def test_namespace_isolation(self, redis_backend, mock_redis):
        """Test that different namespaces don't interfere."""
        await redis_backend.connect()

        # Set same key in different namespaces
        await redis_backend.set("key1", "value1", "namespace1")
        await redis_backend.set("key1", "value2", "namespace2")

        # Verify different Redis keys were used
        call_args_list = mock_redis.setex.call_args_list
        keys = [call[0][0] for call in call_args_list]

        assert "victor:namespace1:key1" in keys
        assert "victor:namespace2:key1" in keys


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for cache statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_returns_backend_type(self, redis_backend, mock_redis):
        """Test that get_stats() includes backend type."""
        await redis_backend.connect()

        mock_redis.info.return_value = {"dbkeys": 10, "used_memory": 1024}

        stats = await redis_backend.get_stats()

        assert stats["backend_type"] == "redis"

    @pytest.mark.asyncio
    async def test_get_stats_returns_key_count(self, redis_backend, mock_redis):
        """Test that get_stats() includes key count."""
        await redis_backend.connect()

        # Create async iterator that returns 42 keys
        class AsyncIterator:
            def __init__(self, count):
                self.count = count

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.count > 0:
                    self.count -= 1
                    return f"key_{self.count}"
                raise StopAsyncIteration

        mock_redis.scan_iter.return_value = AsyncIterator(42)
        mock_redis.info.return_value = {"used_memory": 1024}

        stats = await redis_backend.get_stats()

        assert stats["keys"] == 42

    @pytest.mark.asyncio
    async def test_get_stats_returns_memory_usage(self, redis_backend, mock_redis):
        """Test that get_stats() includes memory usage."""
        await redis_backend.connect()

        mock_redis.info.return_value = {"dbkeys": 10, "used_memory": 2048}

        stats = await redis_backend.get_stats()

        assert stats["memory_used"] == 2048


# =============================================================================
# Distributed Invalidation Tests
# =============================================================================


class TestDistributedInvalidation:
    """Tests for distributed cache invalidation via pub/sub."""

    @pytest.mark.asyncio
    async def test_invalidate_publish_sends_message(self, redis_backend, mock_redis):
        """Test that invalidate_publish() sends invalidation message."""
        await redis_backend.connect()

        await redis_backend.invalidate_publish("key1", "test_namespace")

        # Verify publish was called
        mock_redis.publish.assert_called_once()
        call_args = mock_redis.publish.call_args
        channel = call_args[0][0]
        message = json.loads(call_args[0][1])

        assert channel == "victor:invalidate:test_namespace"
        assert message["key"] == "key1"

    @pytest.mark.asyncio
    async def test_listen_for_invalidation_calls_callback(self, redis_backend, mock_pubsub):
        """Test that listen_for_invalidation() calls callback for events."""
        await redis_backend.connect()

        # Create mock messages
        messages = [
            {
                "type": "pmessage",  # Pattern message, not regular message
                "channel": b"victor:invalidate:test_namespace",
                "data": json.dumps({"key": "key1"}).encode(),
            },
            {
                "type": "pmessage",
                "channel": b"victor:invalidate:test_namespace",
                "data": json.dumps({"key": "key2"}).encode(),
            },
        ]

        # Create async iterator for messages
        class AsyncIterator:
            def __init__(self, items):
                self.items = items

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.items:
                    return self.items.pop(0)
                raise StopAsyncIteration

        mock_pubsub.listen.return_value = AsyncIterator(messages[:])

        # Create callback tracker
        invalidated_keys = []

        async def callback(key: str, namespace: str) -> None:
            invalidated_keys.append((key, namespace))
            # Stop after 2 messages
            if len(invalidated_keys) >= 2:
                raise StopAsyncIteration()

        # Listen (will stop after 2 messages)
        try:
            await redis_backend.listen_for_invalidation(callback)
        except StopAsyncIteration:
            pass

        assert len(invalidated_keys) == 2
        assert ("key1", "test_namespace") in invalidated_keys
        assert ("key2", "test_namespace") in invalidated_keys

    @pytest.mark.asyncio
    async def test_listen_for_invalidation_ignores_non_message_types(self, redis_backend, mock_pubsub):
        """Test that listen_for_invalidation() ignores non-message events."""
        await redis_backend.connect()

        # Mix of message types
        messages = [
            {"type": "subscribe", "channel": b"victor:invalidate:test_namespace", "data": 1},
            {"type": "pmessage", "channel": b"victor:invalidate:test_namespace", "data": json.dumps({"key": "key1"}).encode()},
        ]

        # Create async iterator for messages
        class AsyncIterator:
            def __init__(self, items):
                self.items = items

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.items:
                    return self.items.pop(0)
                raise StopAsyncIteration

        mock_pubsub.listen.return_value = AsyncIterator(messages[:])

        callback_called = []

        async def callback(key: str, namespace: str) -> None:
            callback_called.append((key, namespace))
            raise StopAsyncIteration()

        try:
            await redis_backend.listen_for_invalidation(callback)
        except StopAsyncIteration:
            pass

        # Only one message should trigger callback
        assert len(callback_called) == 1
        assert callback_called[0] == ("key1", "test_namespace")


# =============================================================================
# Key Prefix Tests
# =============================================================================


class TestKeyPrefix:
    """Tests for key prefix functionality."""

    @pytest.mark.asyncio
    async def test_key_prefix_applied_to_all_operations(self, mock_redis):
        """Test that key prefix is applied to all Redis operations."""
        with patch("victor.agent.cache.backends.redis.aioredis.from_url", new=AsyncMock(return_value=mock_redis)):
            backend = RedisCacheBackend(redis_url="redis://localhost:6379/0", key_prefix="testapp")
            await backend.connect()

            # Set a value
            await backend.set("key1", "value1", "test_namespace")

            # Check key prefix
            call_args = mock_redis.setex.call_args
            key = call_args[0][0]
            assert key.startswith("testapp:")
            assert "test_namespace" in key

    @pytest.mark.asyncio
    async def test_default_key_prefix(self, mock_redis):
        """Test that default key prefix is 'victor'."""
        with patch("victor.agent.cache.backends.redis.aioredis.from_url", new=AsyncMock(return_value=mock_redis)):
            backend = RedisCacheBackend(redis_url="redis://localhost:6379/0")
            await backend.connect()

            await backend.set("key1", "value1", "test_namespace")

            call_args = mock_redis.setex.call_args
            key = call_args[0][0]
            assert key.startswith("victor:")


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_get_before_connect_raises_error(self, redis_backend):
        """Test that get() before connect() raises appropriate error."""
        with pytest.raises(Exception):  # Could be more specific
            await redis_backend.get("key1", "test_namespace")

    @pytest.mark.asyncio
    async def test_set_before_connect_raises_error(self, redis_backend):
        """Test that set() before connect() raises appropriate error."""
        with pytest.raises(Exception):
            await redis_backend.set("key1", "value1", "test_namespace")


# =============================================================================
# Helper Classes
# =============================================================================


class StopAsyncIteration(StopAsyncIteration):
    """Helper to stop async iteration."""
    pass
