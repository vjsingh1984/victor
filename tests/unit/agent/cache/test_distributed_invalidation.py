"""Tests for distributed cache invalidation.

Tests the DistributedCacheInvalidator integration with Redis Pub/Sub
and local caches.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.cache.distributed_invalidation import (
    DistributedCacheInvalidator,
    DistributedInvalidationConfig,
    InvalidationStats,
    GRAPH_CACHE_NAMESPACE,
)


class MockTieredCache:
    """Mock TieredCache for testing."""

    def __init__(self):
        self.deleted_keys = []
        self.cleared_namespaces = []

    def delete(self, key: str, namespace: str = "default") -> bool:
        self.deleted_keys.append((key, namespace))
        return True

    def clear(self, namespace: str = None) -> int:
        self.cleared_namespaces.append(namespace)
        return 5  # Pretend we cleared 5 entries


class MockGraphCache:
    """Mock CompiledGraphCache for testing."""

    def __init__(self):
        self.invalidate_all_count = 0

    def invalidate_all(self) -> int:
        self.invalidate_all_count += 1
        return 3  # Pretend we cleared 3 entries


class MockRedisBackend:
    """Mock RedisCacheBackend for testing."""

    def __init__(self):
        self.connected = False
        self.published = []
        self.invalidation_callback = None

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def invalidate_publish(self, key: str, namespace: str):
        if not self.connected:
            raise RuntimeError("Not connected")
        self.published.append((key, namespace))

    async def listen_for_invalidation(self, callback):
        self.invalidation_callback = callback
        # Simulate running indefinitely until cancelled
        try:
            while True:
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise


class TestDistributedInvalidationConfig:
    """Tests for DistributedInvalidationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DistributedInvalidationConfig()
        assert config.redis_url == "redis://localhost:6379/0"
        assert config.key_prefix == "victor"
        assert config.publish_local_invalidations is True
        assert config.listen_for_remote is True
        assert config.invalidation_namespaces is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = DistributedInvalidationConfig(
            redis_url="redis://redis.example.com:6380/1",
            key_prefix="myapp",
            publish_local_invalidations=False,
            listen_for_remote=False,
        )
        assert config.redis_url == "redis://redis.example.com:6380/1"
        assert config.key_prefix == "myapp"
        assert config.publish_local_invalidations is False
        assert config.listen_for_remote is False


class TestInvalidationStats:
    """Tests for InvalidationStats."""

    def test_default_stats(self):
        """Test default stats are zero."""
        stats = InvalidationStats()
        assert stats.local_invalidations == 0
        assert stats.remote_invalidations == 0
        assert stats.publish_errors == 0
        assert stats.tiered_cache_clears == 0
        assert stats.graph_cache_clears == 0


class TestDistributedCacheInvalidator:
    """Tests for DistributedCacheInvalidator."""

    @pytest.fixture
    def mock_redis_backend(self):
        """Create mock Redis backend."""
        return MockRedisBackend()

    @pytest.fixture
    def mock_tiered_cache(self):
        """Create mock TieredCache."""
        return MockTieredCache()

    @pytest.fixture
    def mock_graph_cache(self):
        """Create mock GraphCache."""
        return MockGraphCache()

    @pytest.fixture
    def invalidator(self):
        """Create invalidator with default config."""
        return DistributedCacheInvalidator()

    @pytest.mark.asyncio
    async def test_register_tiered_cache(self, invalidator, mock_tiered_cache):
        """Test registering a TieredCache."""
        invalidator.register_tiered_cache(mock_tiered_cache)
        assert mock_tiered_cache in invalidator._tiered_caches

        # Registering again shouldn't duplicate
        invalidator.register_tiered_cache(mock_tiered_cache)
        assert invalidator._tiered_caches.count(mock_tiered_cache) == 1

    @pytest.mark.asyncio
    async def test_unregister_tiered_cache(self, invalidator, mock_tiered_cache):
        """Test unregistering a TieredCache."""
        invalidator.register_tiered_cache(mock_tiered_cache)
        invalidator.unregister_tiered_cache(mock_tiered_cache)
        assert mock_tiered_cache not in invalidator._tiered_caches

    @pytest.mark.asyncio
    async def test_register_graph_cache(self, invalidator, mock_graph_cache):
        """Test registering a GraphCache."""
        invalidator.register_graph_cache(mock_graph_cache)
        assert mock_graph_cache in invalidator._graph_caches

    @pytest.mark.asyncio
    async def test_local_invalidation(self, invalidator, mock_tiered_cache, mock_redis_backend):
        """Test local invalidation clears local caches."""
        invalidator._redis_backend = mock_redis_backend
        mock_redis_backend.connected = True
        invalidator.register_tiered_cache(mock_tiered_cache)

        await invalidator.invalidate("test_key", "test_namespace")

        assert ("test_key", "test_namespace") in mock_tiered_cache.deleted_keys
        assert invalidator._stats.local_invalidations == 1

    @pytest.mark.asyncio
    async def test_local_invalidation_publishes(self, invalidator, mock_redis_backend):
        """Test local invalidation publishes to Redis."""
        invalidator._redis_backend = mock_redis_backend
        mock_redis_backend.connected = True

        await invalidator.invalidate("key", "namespace")

        # Wait briefly for async operations
        await asyncio.sleep(0.1)

        assert ("key", "namespace") in mock_redis_backend.published

    @pytest.mark.asyncio
    async def test_local_only_invalidation(
        self, invalidator, mock_tiered_cache, mock_redis_backend
    ):
        """Test local_only=True doesn't publish."""
        invalidator._redis_backend = mock_redis_backend
        mock_redis_backend.connected = True
        invalidator.register_tiered_cache(mock_tiered_cache)

        await invalidator.invalidate("key", "ns", local_only=True)

        assert ("key", "ns") in mock_tiered_cache.deleted_keys
        assert len(mock_redis_backend.published) == 0

    @pytest.mark.asyncio
    async def test_namespace_invalidation(self, invalidator, mock_tiered_cache, mock_redis_backend):
        """Test namespace-wide invalidation."""
        invalidator._redis_backend = mock_redis_backend
        mock_redis_backend.connected = True
        invalidator.register_tiered_cache(mock_tiered_cache)

        await invalidator.invalidate_namespace("tools")

        assert "tools" in mock_tiered_cache.cleared_namespaces
        assert ("*", "tools") in mock_redis_backend.published

    @pytest.mark.asyncio
    async def test_graph_cache_invalidation(
        self, invalidator, mock_graph_cache, mock_redis_backend
    ):
        """Test graph cache invalidation."""
        invalidator._redis_backend = mock_redis_backend
        mock_redis_backend.connected = True
        invalidator.register_graph_cache(mock_graph_cache)

        await invalidator.invalidate_graph_cache()

        assert mock_graph_cache.invalidate_all_count == 1
        assert ("*", GRAPH_CACHE_NAMESPACE) in mock_redis_backend.published

    @pytest.mark.asyncio
    async def test_remote_invalidation_handling(self, invalidator, mock_tiered_cache):
        """Test handling of remote invalidation events."""
        invalidator.register_tiered_cache(mock_tiered_cache)

        # Simulate receiving a remote invalidation
        await invalidator._handle_remote_invalidation("remote_key", "remote_ns")

        assert ("remote_key", "remote_ns") in mock_tiered_cache.deleted_keys
        assert invalidator._stats.remote_invalidations == 1

    @pytest.mark.asyncio
    async def test_echo_prevention(self, invalidator, mock_tiered_cache):
        """Test that recently published invalidations are ignored."""
        invalidator.register_tiered_cache(mock_tiered_cache)

        # Add to recently published
        async with invalidator._recently_published_lock:
            invalidator._recently_published.add("ns:key")

        # Try to handle as remote (should be ignored)
        await invalidator._handle_remote_invalidation("key", "ns")

        # Should not have deleted (echo prevention)
        assert ("key", "ns") not in mock_tiered_cache.deleted_keys
        assert invalidator._stats.remote_invalidations == 0

    @pytest.mark.asyncio
    async def test_custom_callback(self, invalidator, mock_tiered_cache):
        """Test custom invalidation callbacks."""
        callback_calls = []

        def custom_callback(key: str, namespace: str):
            callback_calls.append((key, namespace))

        invalidator.register_tiered_cache(mock_tiered_cache)
        invalidator.register_callback(custom_callback)

        await invalidator._clear_local_caches("key", "ns")

        assert ("key", "ns") in callback_calls

    @pytest.mark.asyncio
    async def test_get_stats(self, invalidator, mock_tiered_cache, mock_graph_cache):
        """Test get_stats returns expected information."""
        invalidator.register_tiered_cache(mock_tiered_cache)
        invalidator.register_graph_cache(mock_graph_cache)

        stats = invalidator.get_stats()

        assert stats["local_invalidations"] == 0
        assert stats["remote_invalidations"] == 0
        assert stats["registered_tiered_caches"] == 1
        assert stats["registered_graph_caches"] == 1
        assert stats["listener_active"] is False

    @pytest.mark.asyncio
    async def test_is_connected(self, invalidator, mock_redis_backend):
        """Test is_connected property."""
        assert invalidator.is_connected is False

        invalidator._redis_backend = mock_redis_backend
        assert invalidator.is_connected is True

    @pytest.mark.asyncio
    async def test_publish_disabled(self, invalidator, mock_redis_backend):
        """Test that publish can be disabled via config."""
        config = DistributedInvalidationConfig(
            publish_local_invalidations=False,
        )
        inv = DistributedCacheInvalidator(config)
        inv._redis_backend = mock_redis_backend
        mock_redis_backend.connected = True

        await inv.invalidate("key", "ns")

        assert len(mock_redis_backend.published) == 0

    @pytest.mark.asyncio
    async def test_start_requires_connection(self, invalidator):
        """Test that start() requires prior connect()."""
        with pytest.raises(RuntimeError, match="Not connected"):
            await invalidator.start()


class TestDistributedCacheInvalidatorWithMockedRedis:
    """Tests with mocked RedisCacheBackend import."""

    @pytest.mark.asyncio
    async def test_connect_creates_backend(self):
        """Test that connect creates Redis backend."""
        with patch("victor.agent.cache.backends.redis.RedisCacheBackend") as MockBackend:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            MockBackend.return_value = mock_instance

            invalidator = DistributedCacheInvalidator(
                DistributedInvalidationConfig(
                    redis_url="redis://test:6379/0",
                    key_prefix="test",
                )
            )

            # Actually test connect() creates and connects to the backend
            await invalidator.connect()

            MockBackend.assert_called_once_with(
                redis_url="redis://test:6379/0",
                key_prefix="test",
            )
            mock_instance.connect.assert_called_once()


class TestDistributedCacheInvalidatorMultipleCaches:
    """Tests with multiple registered caches."""

    @pytest.mark.asyncio
    async def test_multiple_tiered_caches(self):
        """Test invalidation with multiple TieredCaches."""
        cache1 = MockTieredCache()
        cache2 = MockTieredCache()

        invalidator = DistributedCacheInvalidator()
        invalidator.register_tiered_cache(cache1)
        invalidator.register_tiered_cache(cache2)

        await invalidator._clear_local_caches("key", "ns")

        assert ("key", "ns") in cache1.deleted_keys
        assert ("key", "ns") in cache2.deleted_keys

    @pytest.mark.asyncio
    async def test_multiple_graph_caches(self):
        """Test invalidation with multiple GraphCaches."""
        cache1 = MockGraphCache()
        cache2 = MockGraphCache()

        invalidator = DistributedCacheInvalidator()
        invalidator.register_graph_cache(cache1)
        invalidator.register_graph_cache(cache2)

        await invalidator._clear_local_caches("*", GRAPH_CACHE_NAMESPACE)

        assert cache1.invalidate_all_count == 1
        assert cache2.invalidate_all_count == 1


class TestDistributedCacheInvalidatorContextManager:
    """Tests for async context manager usage."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager calls connect/start/stop/disconnect."""
        invalidator = DistributedCacheInvalidator(
            DistributedInvalidationConfig(listen_for_remote=False)
        )

        # Mock the methods
        invalidator.connect = AsyncMock()
        invalidator.disconnect = AsyncMock()
        invalidator.start = AsyncMock()
        invalidator.stop = AsyncMock()

        async with invalidator:
            invalidator.connect.assert_called_once()
            invalidator.start.assert_called_once()

        invalidator.stop.assert_called_once()
        invalidator.disconnect.assert_called_once()
