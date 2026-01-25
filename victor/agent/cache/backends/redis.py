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

"""Redis cache backend implementation with distributed invalidation.

This module provides RedisCacheBackend, a distributed cache backend
implementing ICacheBackend protocol with Redis pub/sub for distributed
cache invalidation.

Features:
    - Distributed caching across multiple processes/servers
    - Pub/sub for distributed invalidation events
    - Connection pooling and automatic reconnection
    - TLS/SSL support for secure connections
    - Key namespace isolation with configurable prefix
    - TTL support with automatic expiration
    - Statistics tracking (key count, memory usage)

Use Cases:
    - Multi-process deployments (e.g., Gunicorn, uWSGI)
    - Multi-server deployments (shared cache)
    - Distributed systems requiring coordination
    - Production environments requiring persistence

Limitations:
    - Requires Redis server
    - Network latency (slower than in-memory)
    - Serialization overhead (pickle)

Example:
    ```python
    backend = RedisCacheBackend(
        redis_url="redis://localhost:6379/0",
        key_prefix="victor",
        default_ttl_seconds=3600,
    )

    # Connect to Redis
    await backend.connect()

    # Use cache
    await backend.set("result_123", computation_result, "tool", ttl_seconds=300)
    value = await backend.get("result_123", "tool")

    # Publish invalidation
    await backend.delete("result_123", "tool")
    await backend.invalidate_publish("result_123", "tool")

    # Listen for invalidations (run in background)
    async def handle_invalidation(key: str, namespace: str) -> None:
        await backend.delete(key, namespace)

    task = asyncio.create_task(
        backend.listen_for_invalidation(handle_invalidation)
    )

    # Cleanup
    await backend.disconnect()
    ```
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
from typing import Any, Awaitable, Callable, Dict, Optional

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None  # type: ignore

from victor.agent.cache.backends.protocol import ICacheBackend


logger = logging.getLogger(__name__)


class RedisCacheBackend(ICacheBackend):
    """Redis-based distributed cache backend.

    This backend provides distributed caching with Redis and supports
    pub/sub for distributed cache invalidation across multiple processes
    or servers.

    Features:
        - Distributed caching (shared across processes/servers)
        - Pub/sub for distributed invalidation
        - Connection pooling with auto-reconnect
        - TLS/SSL support
        - Namespace isolation with key prefix
        - TTL support with auto-expiration
        - Statistics tracking

    Configuration:
        redis_url: Redis connection URL (e.g., "redis://localhost:6379/0")
        key_prefix: Prefix for all keys (default: "victor")
        default_ttl_seconds: Default TTL for cache entries (default: 3600)
        connection_pool_size: Size of connection pool (default: 10)
        encoding: Encoding for string values (default: "utf-8")

    Example:
        backend = RedisCacheBackend(
            redis_url="redis://localhost:6379/0",
            key_prefix="myapp",
            default_ttl_seconds=1800,
        )

        await backend.connect()

        # Cache a value
        await backend.set("result_123", computation_result, "tool", ttl_seconds=300)

        # Retrieve it
        value = await backend.get("result_123", "tool")

        # Invalidate across all processes
        await backend.delete("result_123", "tool")
        await backend.invalidate_publish("result_123", "tool")

        await backend.disconnect()
    """

    def __init__(
        self,
        redis_url: str,
        key_prefix: str = "victor",
        default_ttl_seconds: int = 3600,
        connection_pool_size: int = 10,
        encoding: str = "utf-8",
    ):
        """Initialize the Redis cache backend.

        Args:
            redis_url: Redis connection URL
                Examples:
                    - "redis://localhost:6379/0"
                    - "rediss://localhost:6379/0" (TLS)
                    - "redis://:password@localhost:6379/0" (with password)
            key_prefix: Prefix for all cache keys (default: "victor")
            default_ttl_seconds: Default TTL for entries (default: 3600)
            connection_pool_size: Size of connection pool (default: 10)
            encoding: Encoding for string values (default: "utf-8")

        Raises:
            ImportError: If redis package is not installed
        """
        if aioredis is None:
            raise ImportError(
                "redis package is required for RedisCacheBackend. "
                "Install it with: pip install redis>=4.0"
            )

        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._default_ttl = default_ttl_seconds
        self._connection_pool_size = connection_pool_size
        self._encoding = encoding

        # Connection objects (set in connect())
        self._redis: Optional["aioredis.Redis"] = None
        self._pubsub: Optional["aioredis.client.PubSub"] = None

        # Connection state
        self._is_connected = False

    async def connect(self) -> None:
        """Establish connection to Redis.

        This method creates the Redis connection and pubsub client.
        It should be called before any cache operations.

        Raises:
            ConnectionError: If connection to Redis fails
        """
        if self._is_connected:
            # Already connected
            return

        try:
            # Create Redis connection with connection pooling
            from redis.asyncio.connection import Connection
            self._redis = await aioredis.from_url(  # type: ignore[no-untyped-call]
                self._redis_url,
                max_connections=self._connection_pool_size,
                decode_responses=False,  # We handle encoding ourselves
            )

            # Ping to verify connection
            ping_result = await self._redis.ping()
            if not ping_result or not isinstance(ping_result, bool):
                raise ConnectionError("Redis ping failed")

            # Create pubsub client for distributed invalidation
            self._pubsub = self._redis.pubsub()

            self._is_connected = True
            logger.info(f"Connected to Redis at {self._redis_url}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise ConnectionError(f"Failed to connect to Redis: {e}") from e

    async def disconnect(self) -> None:
        """Close Redis connection and release resources.

        This method closes the pubsub client and Redis connection.
        It should be called when shutting down.
        """
        if not self._is_connected:
            return

        try:
            # Close pubsub
            if self._pubsub:
                await self._pubsub.close()
                self._pubsub = None

            # Close Redis connection
            if self._redis:
                await self._redis.close()
                self._redis = None

            self._is_connected = False
            logger.info("Disconnected from Redis")

        except Exception as e:
            logger.error(f"Error disconnecting from Redis: {e}")
            # Don't raise - cleanup should be best-effort

    def _make_key(self, key: str, namespace: str) -> str:
        """Create full Redis key with prefix and namespace.

        Args:
            key: Cache key
            namespace: Namespace for isolation

        Returns:
            Full Redis key (e.g., "victor:tool:result_123")
        """
        return f"{self._key_prefix}:{namespace}:{key}"

    def _make_invalidation_channel(self, namespace: str) -> str:
        """Create Redis pub/sub channel for invalidation events.

        Args:
            namespace: Namespace for isolation

        Returns:
            Channel name (e.g., "victor:invalidate:tool")
        """
        return f"{self._key_prefix}:invalidate:{namespace}"

    async def get(self, key: str, namespace: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key
            namespace: Namespace for isolation

        Returns:
            Cached value or None if not found or expired

        Raises:
            RuntimeError: If not connected to Redis
        """
        if not self._is_connected or self._redis is None:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        redis_key = self._make_key(key, namespace)

        try:
            data = await self._redis.get(redis_key)

            if data is None:
                return None

            # Deserialize
            value = pickle.loads(data)
            return value

        except Exception as e:
            logger.error(f"Error getting key {redis_key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        namespace: str,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be pickle-able)
            namespace: Namespace for isolation
            ttl_seconds: Time-to-live in seconds (None = use default)

        Raises:
            RuntimeError: If not connected to Redis
            TypeError: If value is not pickle-able
        """
        if not self._is_connected or self._redis is None:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        redis_key = self._make_key(key, namespace)
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl

        try:
            # Serialize value
            serialized = pickle.dumps(value)

            # Store with TTL
            await self._redis.setex(redis_key, ttl, serialized)

            logger.debug(f"Set key {redis_key} with TTL {ttl}s")

        except Exception as e:
            logger.error(f"Error setting key {redis_key}: {e}")
            raise

    async def delete(self, key: str, namespace: str) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key to delete
            namespace: Namespace of the key

        Returns:
            True if key was deleted, False if not found

        Raises:
            RuntimeError: If not connected to Redis
        """
        if not self._is_connected or self._redis is None:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        redis_key = self._make_key(key, namespace)

        try:
            result = await self._redis.delete(redis_key)
            return bool(result > 0)

        except Exception as e:
            logger.error(f"Error deleting key {redis_key}: {e}")
            return False

    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace.

        This method uses SCAN to avoid blocking the Redis server.

        Args:
            namespace: Namespace to clear

        Returns:
            Number of keys deleted

        Raises:
            RuntimeError: If not connected to Redis
        """
        if not self._is_connected or self._redis is None:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        pattern = self._make_key("*", namespace)

        try:
            # Use SCAN to avoid blocking
            keys = []
            async for key in self._redis.scan_iter(match=pattern):
                keys.append(key)

            # Delete in batches
            count = 0
            if keys:
                await self._redis.delete(*keys)
                count = len(keys)

            logger.info(f"Cleared {count} keys in namespace {namespace}")
            return count

        except Exception as e:
            logger.error(f"Error clearing namespace {namespace}: {e}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with backend statistics

        Raises:
            RuntimeError: If not connected to Redis
        """
        if not self._is_connected or self._redis is None:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        try:
            # Get Redis info
            info = await self._redis.info()

            # Count keys with our prefix
            pattern = f"{self._key_prefix}:*"
            key_count = 0
            async for _ in self._redis.scan_iter(match=pattern):
                key_count += 1

            return {
                "backend_type": "redis",
                "keys": key_count,
                "memory_used": info.get("used_memory", 0),
                "connected_clients": info.get("connected_clients", 0),
                "redis_version": info.get("redis_version", "unknown"),
            }

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "backend_type": "redis",
                "keys": 0,
                "memory_used": 0,
                "error": str(e),
            }

    async def invalidate_publish(self, key: str, namespace: str) -> None:
        """Publish cache invalidation event for distribution.

        This publishes an invalidation message to the Redis pub/sub channel,
        allowing other processes/nodes to invalidate their local caches.

        Args:
            key: Cache key that was invalidated
            namespace: Namespace of the invalidated key

        Raises:
            RuntimeError: If not connected to Redis
        """
        if not self._is_connected or self._redis is None:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        if self._pubsub is None:
            raise RuntimeError("Pubsub client not initialized")

        channel = self._make_invalidation_channel(namespace)
        message = json.dumps({"key": key})

        try:
            await self._redis.publish(channel, message)
            logger.debug(f"Published invalidation for {namespace}:{key}")

        except Exception as e:
            logger.error(f"Error publishing invalidation: {e}")
            raise

    async def listen_for_invalidation(
        self,
        callback: Callable[[str, str], Awaitable[None]],
    ) -> None:
        """Listen for cache invalidation events from other processes.

        This method subscribes to the invalidation channel and calls
        the provided callback for each invalidation event.

        The callback receives:
            - key: The cache key that was invalidated
            - namespace: The namespace of the invalidated key

        This method runs indefinitely until cancelled.

        Args:
            callback: Async function to call for each invalidation event

        Raises:
            RuntimeError: If not connected to Redis

        Example:
            async def handle_invalidation(key: str, namespace: str) -> None:
                await backend.delete(key, namespace)

            # Run in background
            task = asyncio.create_task(
                backend.listen_for_invalidation(handle_invalidation)
            )
        """
        if not self._is_connected or self._pubsub is None:
            raise RuntimeError("Not connected to Redis. Call connect() first.")

        # Subscribe to all invalidation channels
        pattern = f"{self._make_invalidation_channel('*')}"
        await self._pubsub.psubscribe(pattern)

        logger.info(f"Listening for invalidation events on {pattern}")

        try:
            async for message in self._pubsub.listen():
                if message["type"] == "pmessage":
                    try:
                        # Parse message
                        channel = message["channel"].decode(self._encoding)
                        data = json.loads(message["data"])

                        # Extract namespace from channel
                        # Format: victor:invalidate:{namespace}
                        namespace = channel.split(":")[-1]
                        key = data["key"]

                        # Call callback
                        await callback(key, namespace)

                    except (KeyError, ValueError, json.JSONDecodeError) as e:
                        logger.warning(f"Invalid invalidation message: {e}")
                        continue

        except asyncio.CancelledError:
            logger.info("Invalidation listener cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in invalidation listener: {e}")
            raise


__all__ = [
    "RedisCacheBackend",
]
