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

"""Distributed cache invalidation using Redis Pub/Sub.

This module provides the DistributedCacheInvalidator class that integrates
Redis Pub/Sub with local caches (TieredCache, CompiledGraphCache) for
distributed cache invalidation across multiple processes or nodes.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Node 1                                    │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │ TieredCache │  │ GraphCache  │  │ DistributedCache    │  │
    │  │ (L1/L2)     │  │ (compiled)  │  │ Invalidator         │  │
    │  └─────┬───────┘  └──────┬──────┘  └──────────┬──────────┘  │
    │        │                 │                    │              │
    │        └────────────────>│<───────────────────┘              │
    │                          │                                   │
    └──────────────────────────│───────────────────────────────────┘
                               │
                               ▼ Redis Pub/Sub
    ┌──────────────────────────│───────────────────────────────────┐
    │                    Node 2                                    │
    │                          │                                   │
    │        ┌────────────────>│<───────────────────┐              │
    │        │                 │                    │              │
    │  ┌─────┴───────┐  ┌──────┴──────┐  ┌──────────┴──────────┐  │
    │  │ TieredCache │  │ GraphCache  │  │ DistributedCache    │  │
    │  │ (L1/L2)     │  │ (compiled)  │  │ Invalidator         │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
    └──────────────────────────────────────────────────────────────┘

Features:
    - Automatic propagation of cache invalidations across nodes
    - Support for TieredCache (L1/L2) and CompiledGraphCache
    - Namespace-based invalidation for fine-grained control
    - Background listener for receiving remote invalidations
    - Graceful shutdown with pending flush

Example:
    from victor.agent.cache.distributed_invalidation import (
        DistributedCacheInvalidator,
        get_distributed_invalidator,
    )

    # Initialize
    invalidator = await get_distributed_invalidator(
        redis_url="redis://localhost:6379/0",
    )

    # Register local caches
    invalidator.register_tiered_cache(tiered_cache)
    invalidator.register_graph_cache(graph_cache)

    # Start listening for remote invalidations
    await invalidator.start()

    # Invalidate (propagates to all nodes)
    await invalidator.invalidate("key", "namespace")

    # Cleanup
    await invalidator.stop()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

if TYPE_CHECKING:
    from victor.agent.cache.backends.redis import RedisCacheBackend
    from victor.framework.graph_cache import CompiledGraphCache
    from victor.storage.cache.tiered_cache import TieredCache

logger = logging.getLogger(__name__)


# Special namespace for graph cache invalidations
GRAPH_CACHE_NAMESPACE = "__graph_cache__"


@dataclass
class DistributedInvalidationConfig:
    """Configuration for distributed cache invalidation.

    Attributes:
        redis_url: Redis connection URL (default: redis://localhost:6379/0)
        key_prefix: Prefix for Redis keys and channels (default: victor)
        publish_local_invalidations: Whether to publish local invalidations
                                    to Redis (default: True)
        listen_for_remote: Whether to listen for remote invalidations
                          (default: True)
        invalidation_namespaces: Namespaces to listen for (default: all)
    """

    redis_url: str = "redis://localhost:6379/0"
    key_prefix: str = "victor"
    publish_local_invalidations: bool = True
    listen_for_remote: bool = True
    invalidation_namespaces: Optional[Set[str]] = None


@dataclass
class InvalidationStats:
    """Statistics for distributed invalidation.

    Attributes:
        local_invalidations: Count of locally-triggered invalidations
        remote_invalidations: Count of remotely-received invalidations
        publish_errors: Count of publish failures
        tiered_cache_clears: Count of TieredCache clears
        graph_cache_clears: Count of GraphCache clears
    """

    local_invalidations: int = 0
    remote_invalidations: int = 0
    publish_errors: int = 0
    tiered_cache_clears: int = 0
    graph_cache_clears: int = 0


class DistributedCacheInvalidator:
    """Distributed cache invalidation coordinator.

    Integrates Redis Pub/Sub with local caches to propagate cache
    invalidations across multiple processes or nodes.

    When a cache entry is invalidated locally:
    1. The local cache is cleared
    2. An invalidation event is published to Redis
    3. Other nodes receive the event and clear their local caches

    Thread Safety:
        This class is async-safe but not thread-safe. Use from a single
        asyncio event loop.

    Example:
        invalidator = DistributedCacheInvalidator(config)
        await invalidator.connect()
        invalidator.register_tiered_cache(tiered_cache)
        await invalidator.start()

        # Invalidate locally and propagate
        await invalidator.invalidate("key", "tool")

        await invalidator.stop()
        await invalidator.disconnect()
    """

    def __init__(
        self,
        config: Optional[DistributedInvalidationConfig] = None,
    ) -> None:
        """Initialize distributed cache invalidator.

        Args:
            config: Configuration (uses defaults if None)
        """
        self._config = config or DistributedInvalidationConfig()
        self._redis_backend: Optional[RedisCacheBackend] = None

        # Registered local caches
        self._tiered_caches: List[TieredCache] = []
        self._graph_caches: List[CompiledGraphCache] = []

        # Custom invalidation callbacks
        self._callbacks: List[Callable[[str, str], None]] = []

        # Background listener task
        self._listener_task: Optional[asyncio.Task] = None
        self._shutdown = False

        # Statistics
        self._stats = InvalidationStats()

        # Track recently published to avoid echo
        self._recently_published: Set[str] = set()
        self._recently_published_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Connect to Redis.

        Must be called before start() or invalidate().

        Raises:
            ImportError: If redis package is not installed
            ConnectionError: If Redis connection fails
        """
        from victor.agent.cache.backends.redis import RedisCacheBackend

        self._redis_backend = RedisCacheBackend(
            redis_url=self._config.redis_url,
            key_prefix=self._config.key_prefix,
        )
        await self._redis_backend.connect()
        logger.info(f"Connected to Redis for distributed invalidation")

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis_backend is not None:
            await self._redis_backend.disconnect()
            self._redis_backend = None
            logger.info("Disconnected from Redis")

    def register_tiered_cache(self, cache: "TieredCache") -> None:
        """Register a TieredCache for distributed invalidation.

        Registered caches will be cleared when remote invalidation
        events are received.

        Args:
            cache: TieredCache instance to register
        """
        if cache not in self._tiered_caches:
            self._tiered_caches.append(cache)
            logger.debug(f"Registered TieredCache for invalidation")

    def unregister_tiered_cache(self, cache: "TieredCache") -> None:
        """Unregister a TieredCache.

        Args:
            cache: TieredCache instance to unregister
        """
        if cache in self._tiered_caches:
            self._tiered_caches.remove(cache)

    def register_graph_cache(self, cache: "CompiledGraphCache") -> None:
        """Register a CompiledGraphCache for distributed invalidation.

        Args:
            cache: CompiledGraphCache instance to register
        """
        if cache not in self._graph_caches:
            self._graph_caches.append(cache)
            logger.debug(f"Registered CompiledGraphCache for invalidation")

    def unregister_graph_cache(self, cache: "CompiledGraphCache") -> None:
        """Unregister a CompiledGraphCache.

        Args:
            cache: CompiledGraphCache instance to unregister
        """
        if cache in self._graph_caches:
            self._graph_caches.remove(cache)

    def register_callback(
        self,
        callback: Callable[[str, str], None],
    ) -> None:
        """Register a custom invalidation callback.

        The callback will be called for each invalidation event
        (both local and remote).

        Args:
            callback: Function(key, namespace) to call on invalidation
        """
        self._callbacks.append(callback)

    async def start(self) -> None:
        """Start listening for remote invalidation events.

        Must call connect() first.

        Starts a background task that listens for Redis Pub/Sub
        messages and clears local caches accordingly.
        """
        if self._redis_backend is None:
            raise RuntimeError("Not connected. Call connect() first.")

        if not self._config.listen_for_remote:
            logger.debug("Remote invalidation listening disabled")
            return

        if self._listener_task is not None:
            logger.warning("Listener already running")
            return

        self._shutdown = False
        self._listener_task = asyncio.create_task(self._listen_loop())
        logger.info("Started distributed invalidation listener")

    async def stop(self) -> None:
        """Stop listening for remote invalidation events.

        Cancels the background listener task.
        """
        self._shutdown = True

        if self._listener_task is not None:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None
            logger.info("Stopped distributed invalidation listener")

    async def invalidate(
        self,
        key: str,
        namespace: str = "default",
        local_only: bool = False,
    ) -> None:
        """Invalidate a cache entry locally and propagate to other nodes.

        Args:
            key: Cache key to invalidate
            namespace: Cache namespace
            local_only: If True, don't publish to Redis (default: False)
        """
        # Clear local caches
        await self._clear_local_caches(key, namespace)
        self._stats.local_invalidations += 1

        # Publish to Redis (if enabled and not local_only)
        if (
            self._config.publish_local_invalidations
            and not local_only
            and self._redis_backend is not None
        ):
            # Track to avoid echo
            cache_key = f"{namespace}:{key}"
            async with self._recently_published_lock:
                self._recently_published.add(cache_key)

            try:
                await self._redis_backend.invalidate_publish(key, namespace)
                logger.debug(f"Published invalidation: {namespace}:{key}")
            except Exception as e:
                self._stats.publish_errors += 1
                logger.warning(f"Failed to publish invalidation: {e}")

            # Remove from recently published after short delay
            asyncio.create_task(self._remove_from_recent(cache_key))

    async def invalidate_namespace(
        self,
        namespace: str,
        local_only: bool = False,
    ) -> None:
        """Invalidate all entries in a namespace.

        Args:
            namespace: Namespace to clear
            local_only: If True, don't publish to Redis
        """
        # Clear local caches for namespace
        for cache in self._tiered_caches:
            cleared = cache.clear(namespace)
            self._stats.tiered_cache_clears += cleared

        self._stats.local_invalidations += 1

        # Publish namespace invalidation
        if (
            self._config.publish_local_invalidations
            and not local_only
            and self._redis_backend is not None
        ):
            try:
                # Use special key to indicate namespace-wide invalidation
                await self._redis_backend.invalidate_publish("*", namespace)
            except Exception as e:
                self._stats.publish_errors += 1
                logger.warning(f"Failed to publish namespace invalidation: {e}")

    async def invalidate_graph_cache(self, local_only: bool = False) -> None:
        """Invalidate all compiled graph caches.

        Args:
            local_only: If True, don't publish to Redis
        """
        # Clear local graph caches
        for cache in self._graph_caches:
            cleared = cache.invalidate_all()
            self._stats.graph_cache_clears += cleared

        self._stats.local_invalidations += 1

        # Publish graph cache invalidation
        if (
            self._config.publish_local_invalidations
            and not local_only
            and self._redis_backend is not None
        ):
            try:
                await self._redis_backend.invalidate_publish(
                    "*", GRAPH_CACHE_NAMESPACE
                )
            except Exception as e:
                self._stats.publish_errors += 1
                logger.warning(f"Failed to publish graph cache invalidation: {e}")

    async def _clear_local_caches(self, key: str, namespace: str) -> None:
        """Clear a key from all local caches.

        Args:
            key: Cache key
            namespace: Cache namespace
        """
        # Handle graph cache namespace specially
        if namespace == GRAPH_CACHE_NAMESPACE:
            for cache in self._graph_caches:
                cleared = cache.invalidate_all()
                self._stats.graph_cache_clears += cleared
            return

        # Handle namespace-wide invalidation
        if key == "*":
            for cache in self._tiered_caches:
                cleared = cache.clear(namespace)
                self._stats.tiered_cache_clears += cleared
            return

        # Handle specific key invalidation
        for cache in self._tiered_caches:
            if cache.delete(key, namespace):
                self._stats.tiered_cache_clears += 1

        # Call custom callbacks
        for callback in self._callbacks:
            try:
                callback(key, namespace)
            except Exception as e:
                logger.warning(f"Invalidation callback error: {e}")

    async def _listen_loop(self) -> None:
        """Background loop for listening to Redis invalidation events."""
        if self._redis_backend is None:
            return

        try:
            await self._redis_backend.listen_for_invalidation(
                self._handle_remote_invalidation
            )
        except asyncio.CancelledError:
            logger.debug("Invalidation listener cancelled")
            raise
        except Exception as e:
            if not self._shutdown:
                logger.error(f"Invalidation listener error: {e}")

    async def _handle_remote_invalidation(self, key: str, namespace: str) -> None:
        """Handle an invalidation event received from Redis.

        Args:
            key: Invalidated cache key
            namespace: Cache namespace
        """
        cache_key = f"{namespace}:{key}"

        # Check if this was recently published by us (avoid echo)
        async with self._recently_published_lock:
            if cache_key in self._recently_published:
                logger.debug(f"Ignoring echo: {cache_key}")
                return

        logger.debug(f"Received remote invalidation: {cache_key}")
        self._stats.remote_invalidations += 1

        # Clear local caches
        await self._clear_local_caches(key, namespace)

    async def _remove_from_recent(self, cache_key: str) -> None:
        """Remove key from recently published set after delay."""
        await asyncio.sleep(1.0)  # 1 second delay
        async with self._recently_published_lock:
            self._recently_published.discard(cache_key)

    def get_stats(self) -> Dict[str, Any]:
        """Get invalidation statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "local_invalidations": self._stats.local_invalidations,
            "remote_invalidations": self._stats.remote_invalidations,
            "publish_errors": self._stats.publish_errors,
            "tiered_cache_clears": self._stats.tiered_cache_clears,
            "graph_cache_clears": self._stats.graph_cache_clears,
            "registered_tiered_caches": len(self._tiered_caches),
            "registered_graph_caches": len(self._graph_caches),
            "listener_active": self._listener_task is not None,
        }

    @property
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        return self._redis_backend is not None

    @property
    def is_listening(self) -> bool:
        """Check if listening for remote invalidations."""
        return self._listener_task is not None and not self._listener_task.done()

    async def __aenter__(self) -> "DistributedCacheInvalidator":
        """Async context manager entry."""
        await self.connect()
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
        await self.disconnect()


# Singleton instance
_invalidator: Optional[DistributedCacheInvalidator] = None
_invalidator_lock = asyncio.Lock()


async def get_distributed_invalidator(
    config: Optional[DistributedInvalidationConfig] = None,
    redis_url: Optional[str] = None,
) -> DistributedCacheInvalidator:
    """Get or create the singleton DistributedCacheInvalidator.

    Args:
        config: Configuration (used only on first call)
        redis_url: Redis URL (shortcut for config.redis_url)

    Returns:
        DistributedCacheInvalidator instance
    """
    global _invalidator

    async with _invalidator_lock:
        if _invalidator is None:
            if config is None and redis_url is not None:
                config = DistributedInvalidationConfig(redis_url=redis_url)
            _invalidator = DistributedCacheInvalidator(config)
            await _invalidator.connect()

        return _invalidator


async def shutdown_distributed_invalidator() -> None:
    """Shutdown the singleton invalidator."""
    global _invalidator

    async with _invalidator_lock:
        if _invalidator is not None:
            await _invalidator.stop()
            await _invalidator.disconnect()
            _invalidator = None


__all__ = [
    "DistributedCacheInvalidator",
    "DistributedInvalidationConfig",
    "InvalidationStats",
    "GRAPH_CACHE_NAMESPACE",
    "get_distributed_invalidator",
    "shutdown_distributed_invalidator",
]
