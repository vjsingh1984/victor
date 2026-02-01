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

"""Cache backend protocols with distributed cache support.

This module defines the protocol for cache backends with support for:
- Basic cache operations (get, set, delete, clear_namespace)
- Distributed invalidation via pub/sub
- Connection lifecycle management
- Statistics and monitoring

The protocol extends the base ICacheBackend from victor.protocols.cache
with additional methods for distributed deployments.

Design Principles:
    - DIP: High-level modules depend on this protocol, not concrete backends
    - ISP: Focused protocol - implement only what you need
    - OCP: New backends can be added without modifying existing code
    - Strategy Pattern: Different backends for different use cases

Supported Backends:
    - Memory: Fast in-memory caching (default, single-process)
    - Redis: Distributed caching with pub/sub (multi-process)
    - SQLite: Persistent caching (across restarts)
    - Custom: User-provided implementations

Usage:
    # Implement a cache backend
    class RedisCacheBackend(ICacheBackend):
        def __init__(self, redis_url: str):
            self._redis_url = redis_url
            self._pubsub = None

        async def connect(self) -> None:
            self._redis = await aioredis.from_url(self._redis_url)
            self._pubsub = self._redis.pubsub()

        async def disconnect(self) -> None:
            if self._pubsub:
                await self._pubsub.close()
            await self._redis.close()

        async def invalidate_publish(self, key: str, namespace: str) -> None:
            await self._redis.publish(
                f"cache:invalidate:{namespace}",
                json.dumps({"key": key})
            )

        async def listen_for_invalidation(
            self,
            callback: Callable[[str, str], Awaitable[None]]
        ) -> None:
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    await callback(data["key"], namespace)
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable
from collections.abc import Awaitable, Callable


# =============================================================================
# Enhanced Cache Backend Protocol
# =============================================================================


@runtime_checkable
class ICacheBackend(Protocol):
    """Protocol for cache storage backends with distributed support.

    This protocol extends the base ICacheBackend with methods for
    distributed cache invalidation and connection lifecycle management.

    Core Methods (from base ICacheBackend):
        - get(key, namespace): Get value from cache
        - set(key, value, namespace, ttl_seconds): Set value in cache
        - delete(key, namespace): Delete key from cache
        - clear_namespace(namespace): Clear all keys in namespace
        - get_stats(): Get cache statistics

    Distributed Methods (new in this protocol):
        - connect(): Establish connection to backend
        - disconnect(): Close connection and release resources
        - invalidate_publish(key, namespace): Publish invalidation event
        - listen_for_invalidation(callback): Subscribe to invalidation events

    Implementation Guidelines:
        - All methods must be async for non-blocking operation
        - connect() should be called before any cache operations
        - disconnect() should be called when shutting down
        - invalidate_publish() is optional (returns NotImplementedError if not supported)
        - listen_for_invalidation() should run in background task

    Use Cases:
        Memory Backend:
            - Single-process deployments
            - Testing and development
            - Fast access with no I/O
            - No distributed invalidation needed

        Redis Backend:
            - Multi-process deployments
            - Distributed cache invalidation
            - Shared cache across servers
            - Pub/sub for coordination

        SQLite Backend:
            - Persistent caching across restarts
            - Single-process deployments
            - No distributed features needed
            - Disk-based storage

    Example Implementation (Redis):
        class RedisCacheBackend(ICacheBackend):
            def __init__(self, redis_url: str):
                self._redis_url = redis_url
                self._redis = None
                self._pubsub = None

            async def connect(self) -> None:
                self._redis = await aioredis.from_url(self._redis_url)
                self._pubsub = self._redis.pubsub()
                await self._pubsub.subscribe("cache:invalidate:*")

            async def disconnect(self) -> None:
                if self._pubsub:
                    await self._pubsub.close()
                if self._redis:
                    await self._redis.close()

            async def get(self, key: str, namespace: str) -> Optional[Any]:
                data = await self._redis.get(f"{namespace}:{key}")
                return pickle.loads(data) if data else None

            async def set(
                self,
                key: str,
                value: Any,
                namespace: str,
                ttl_seconds: Optional[int] = None
            ) -> None:
                serialized = pickle.dumps(value)
                ttl = ttl_seconds or 3600
                await self._redis.setex(f"{namespace}:{key}", ttl, serialized)

            async def delete(self, key: str, namespace: str) -> bool:
                result = await self._redis.delete(f"{namespace}:{key}")
                return result > 0

            async def clear_namespace(self, namespace: str) -> int:
                pattern = f"{namespace}:*"
                keys = []
                async for key in self._redis.scan_iter(match=pattern):
                    keys.append(key)
                if keys:
                    await self._redis.delete(*keys)
                return len(keys)

            async def get_stats(self) -> Dict[str, Any]:
                info = await self._redis.info()
                return {
                    "backend_type": "redis",
                    "keys": info.get("dbkeys", 0),
                    "memory_used": info.get("used_memory", 0),
                }

            async def invalidate_publish(self, key: str, namespace: str) -> None:
                await self._redis.publish(
                    f"cache:invalidate:{namespace}",
                    json.dumps({"key": key})
                )

            async def listen_for_invalidation(
                self,
                callback: Callable[[str, str], Awaitable[None]]
            ) -> None:
                async for message in self._pubsub.listen():
                    if message["type"] == "message":
                        channel = message["channel"]
                        namespace = channel.decode().split(":")[-1]
                        data = json.loads(message["data"])
                        await callback(data["key"], namespace)
    """

    # -------------------------------------------------------------------------
    # Core cache operations (from base ICacheBackend)
    # -------------------------------------------------------------------------

    async def get(self, key: str, namespace: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key
            namespace: Namespace for isolation

        Returns:
            Cached value or None if not found or expired

        Example:
            value = await backend.get("result_123", "tool")
        """
        ...

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
            value: Value to cache
            namespace: Namespace for isolation
            ttl_seconds: Time-to-live in seconds (None = use default)

        Example:
            await backend.set("result_123", computation_result, "tool", ttl_seconds=300)
        """
        ...

    async def delete(self, key: str, namespace: str) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key to delete
            namespace: Namespace of the key

        Returns:
            True if key was deleted, False if not found

        Example:
            deleted = await backend.delete("result_123", "tool")
        """
        ...

    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace.

        Args:
            namespace: Namespace to clear

        Returns:
            Number of keys deleted

        Example:
            count = await backend.clear_namespace("tool")
        """
        ...

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with backend statistics

        Common keys:
            - backend_type: Type of backend ('memory', 'redis', 'sqlite')
            - keys: Number of cached items
            - memory_used: Memory usage in bytes (if tracked)
            - hit_rate: Cache hit rate (if tracked)

        Example:
            stats = await backend.get_stats()
        """
        ...

    # -------------------------------------------------------------------------
    # Distributed cache methods (new in this protocol)
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Establish connection to cache backend.

        This method should be called before any cache operations.
        For in-memory backends, this may be a no-op.

        Example:
            await backend.connect()
        """
        ...

    async def disconnect(self) -> None:
        """Close connection and release resources.

        This method should be called when shutting down.
        For in-memory backends, this may be a no-op.

        Example:
            await backend.disconnect()
        """
        ...

    async def invalidate_publish(self, key: str, namespace: str) -> None:
        """Publish cache invalidation event for distribution.

        This method is used to notify other processes/nodes that
        a cache entry has been invalidated. Implementations that
        don't support distributed invalidation should raise
        NotImplementedError.

        Args:
            key: Cache key that was invalidated
            namespace: Namespace of the invalidated key

        Raises:
            NotImplementedError: If backend doesn't support pub/sub

        Example:
            # After deleting a key, publish invalidation
            await backend.delete("result_123", "tool")
            await backend.invalidate_publish("result_123", "tool")
        """
        ...

    async def listen_for_invalidation(
        self,
        callback: Callable[[str, str], Awaitable[None]],
    ) -> None:
        """Listen for cache invalidation events from other processes.

        This method should run indefinitely (until cancelled) and
        call the provided callback for each invalidation event.

        The callback receives:
            - key: The cache key that was invalidated
            - namespace: The namespace of the invalidated key

        Implementations that don't support distributed invalidation
        should raise NotImplementedError.

        This is typically run as a background task:
            task = asyncio.create_task(
                backend.listen_for_invalidation(my_callback)
            )

        Args:
            callback: Async function to call for each invalidation event

        Raises:
            NotImplementedError: If backend doesn't support pub/sub

        Example:
            async def handle_invalidation(key: str, namespace: str) -> None:
                await backend.delete(key, namespace)

            # Run in background
            task = asyncio.create_task(
                backend.listen_for_invalidation(handle_invalidation)
            )
        """
        ...


__all__ = [
    "ICacheBackend",
]
