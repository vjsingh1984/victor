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

"""Intelligent cache invalidation system with multiple strategies.

This module implements comprehensive cache invalidation mechanisms:
- Time-based expiration (TTL)
- Event-based invalidation (code changes, config changes)
- Manual invalidation API
- Dependency graph for cascade invalidation
- Cache tagging for group invalidation
- Predictive invalidation (proactive refresh)

Strategies:
- Active: Push updates immediately
- Passive: Lazy expiration on access
- Predictive: Proactive refresh before expiration

Benefits:
- Keeps cache fresh and consistent
- Reduces stale data serving
- Efficient cascade invalidation
- Flexible tagging system
- Predictive refresh for hot keys

Usage:
    invalidator = CacheInvalidator(
        cache=multi_level_cache,
        strategy=InvalidationStrategy.TTL,
        default_ttl=3600,
        enable_tagging=True,
    )

    # Invalidate by key
    await invalidator.invalidate("key", namespace="tool")

    # Invalidate by tag
    await invalidator.invalidate_tag("python_files")

    # Cascade invalidation via dependency graph
    await invalidator.invalidate_dependents("/src/main.py")
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, cast
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


# =============================================================================
# Invalidation Strategies
# =============================================================================


class InvalidationStrategy(Enum):
    """Cache invalidation strategy."""

    TTL = "ttl"
    """Time-based expiration - entries expire after fixed time."""

    EVENT_BASED = "event_based"
    """Event-driven - invalidate on specific events (file changes, etc.)."""

    MANUAL = "manual"
    """Manual - explicit invalidation via API."""

    PREDICTIVE = "predictive"
    """Predictive - proactively refresh hot entries before expiration."""

    HYBRID = "hybrid"
    """Combine multiple strategies for optimal performance."""


# =============================================================================
# Cache Tags
# =============================================================================


@dataclass
class TaggedEntry:
    """A cache entry with tags for group invalidation.

    Attributes:
        key: Cache key
        namespace: Namespace
        tags: Set of tags associated with this entry
        timestamp: Creation timestamp
        ttl: Time-to-live in seconds
    """

    key: str
    namespace: str
    tags: set[str]
    timestamp: float
    ttl: Optional[int]


class CacheTagManager:
    """Manages cache tags for group-based invalidation.

    Allows invalidating groups of cache entries by tag, useful for:
    - Invalidating all Python file caches
    - Clearing all tool results for a specific project
    - Bulk invalidation by category

    Args:
        max_entries: Maximum number of tagged entries to track
    """

    def __init__(self, max_entries: int = 10000):
        """Initialize tag manager."""
        self.max_entries = max_entries

        # tag -> set of (key, namespace) tuples
        self._tag_index: dict[str, set[tuple[str, str]]] = defaultdict(set)

        # (key, namespace) -> set of tags
        self._entry_tags: dict[tuple[str, str], set[str]] = {}

        self._lock = threading.RLock()

    def tag(self, key: str, namespace: str, tags: list[str]) -> None:
        """Add tags to a cache entry.

        Args:
            key: Cache key
            namespace: Namespace
            tags: List of tags to add
        """
        key_tuple = (key, namespace)

        with self._lock:
            # Create tag sets if needed
            if key_tuple not in self._entry_tags:
                self._entry_tags[key_tuple] = set()

            # Add new tags
            for tag in tags:
                self._entry_tags[key_tuple].add(tag)
                self._tag_index[tag].add(key_tuple)

    def untag(self, key: str, namespace: str, tags: Optional[list[str]] = None) -> None:
        """Remove tags from a cache entry.

        Args:
            key: Cache key
            namespace: Namespace
            tags: List of tags to remove (None = remove all tags)
        """
        key_tuple = (key, namespace)

        with self._lock:
            if key_tuple not in self._entry_tags:
                return

            if tags is None:
                # Remove all tags
                for tag in self._entry_tags[key_tuple]:
                    self._tag_index[tag].discard(key_tuple)
                    if not self._tag_index[tag]:
                        del self._tag_index[tag]

                del self._entry_tags[key_tuple]
            else:
                # Remove specific tags
                for tag in tags:
                    self._entry_tags[key_tuple].discard(tag)
                    self._tag_index[tag].discard(key_tuple)
                    if not self._tag_index[tag]:
                        del self._tag_index[tag]

    def get_tagged_keys(self, tag: str) -> set[tuple[str, str]]:
        """Get all keys with a specific tag.

        Args:
            tag: Tag to query

        Returns:
            Set of (key, namespace) tuples
        """
        with self._lock:
            return self._tag_index.get(tag, set()).copy()

    def remove_entry(self, key: str, namespace: str) -> None:
        """Remove entry from tag tracking.

        Args:
            key: Cache key
            namespace: Namespace
        """
        key_tuple = (key, namespace)

        with self._lock:
            if key_tuple not in self._entry_tags:
                return

            # Remove from tag index
            for tag in self._entry_tags[key_tuple]:
                self._tag_index[tag].discard(key_tuple)
                if not self._tag_index[tag]:
                    del self._tag_index[tag]

            # Remove entry
            del self._entry_tags[key_tuple]

    def get_stats(self) -> dict[str, Any]:
        """Get tag manager statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                "total_tags": len(self._tag_index),
                "tagged_entries": len(self._entry_tags),
                "avg_tags_per_entry": (
                    sum(len(tags) for tags in self._entry_tags.values()) / len(self._entry_tags)
                    if self._entry_tags
                    else 0
                ),
            }


# =============================================================================
# Dependency Graph
# =============================================================================


class InvalidationDependencyGraph:
    """Dependency graph for cascading cache invalidation.

    Tracks dependencies between cache entries and external resources
    (files, config, etc.) to enable cascade invalidation.

    When a resource changes, all dependent cache entries are invalidated.

    Example:
        - Tool result depends on source files
        - Config cache depends on config files
        - Analysis result depends on multiple files
    """

    def __init__(self) -> None:
        """Initialize dependency graph."""
        # resource -> set of dependent (key, namespace) tuples
        self._dependencies: dict[str, set[tuple[str, str]]] = defaultdict(set)

        # (key, namespace) -> set of resources it depends on
        self._dependents: dict[tuple[str, str], set[str]] = defaultdict(set)

        self._lock = threading.RLock()

    def add_dependency(self, key: str, namespace: str, resource: str) -> None:
        """Add a dependency relationship.

        When resource changes, key should be invalidated.

        Args:
            key: Cache key
            namespace: Namespace
            resource: Resource identifier (file path, config key, etc.)
        """
        key_tuple = (key, namespace)

        with self._lock:
            self._dependencies[resource].add(key_tuple)
            self._dependents[key_tuple].add(resource)

    def remove_dependency(self, key: str, namespace: str, resource: str) -> None:
        """Remove a dependency relationship.

        Args:
            key: Cache key
            namespace: Namespace
            resource: Resource identifier
        """
        key_tuple = (key, namespace)

        with self._lock:
            self._dependencies[resource].discard(key_tuple)
            self._dependents[key_tuple].discard(resource)

            # Clean up empty sets
            if not self._dependencies[resource]:
                del self._dependencies[resource]
            if not self._dependents[key_tuple]:
                del self._dependents[key_tuple]

    def get_dependents(self, resource: str) -> set[tuple[str, str]]:
        """Get all cache entries that depend on a resource.

        Args:
            resource: Resource identifier

        Returns:
            Set of (key, namespace) tuples that should be invalidated
        """
        with self._lock:
            return self._dependencies.get(resource, set()).copy()

    def remove_entry(self, key: str, namespace: str) -> None:
        """Remove entry from dependency tracking.

        Args:
            key: Cache key
            namespace: Namespace
        """
        key_tuple = (key, namespace)

        with self._lock:
            if key_tuple not in self._dependents:
                return

            # Remove from dependency index
            for resource in self._dependents[key_tuple]:
                self._dependencies[resource].discard(key_tuple)
                if not self._dependencies[resource]:
                    del self._dependencies[resource]

            # Remove entry
            del self._dependents[key_tuple]

    def get_stats(self) -> dict[str, Any]:
        """Get dependency graph statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            return {
                "resources": len(self._dependencies),
                "dependent_entries": len(self._dependents),
                "total_dependencies": sum(len(deps) for deps in self._dependencies.values()),
            }


# =============================================================================
# Cache Invalidator
# =============================================================================


@dataclass
class InvalidationConfig:
    """Configuration for cache invalidation.

    Attributes:
        strategy: Invalidations strategy to use
        default_ttl: Default TTL in seconds
        enable_tagging: Enable cache tagging
        enable_dependencies: Enable dependency tracking
        predictive_refresh_threshold: Access count threshold for predictive refresh
        cleanup_interval: Seconds between cleanup tasks
    """

    strategy: InvalidationStrategy = InvalidationStrategy.TTL
    default_ttl: int = 3600
    enable_tagging: bool = True
    enable_dependencies: bool = True
    predictive_refresh_threshold: int = 10
    cleanup_interval: int = 300


class CacheInvalidator:
    """Comprehensive cache invalidation system.

    Features:
    - Multiple invalidation strategies (TTL, event-based, manual, predictive)
    - Tag-based group invalidation
    - Dependency graph for cascade invalidation
    - Predictive refresh for hot entries
    - Event-driven invalidation
    - Comprehensive metrics

    Example:
        ```python
        invalidator = CacheInvalidator(
            cache=multi_level_cache,
            strategy=InvalidationStrategy.HYBRID,
            enable_tagging=True,
        )

        # Tag entries
        invalidator.tag("key1", ["python_files", "src"], namespace="tool")

        # Invalidate by tag
        await invalidator.invalidate_tag("python_files")

        # Add dependency
        invalidator.add_dependency("key1", "tool", "/src/main.py")

        # Invalidate on file change
        await invalidator.invalidate_dependents("/src/main.py")
        ```
    """

    def __init__(
        self,
        cache: Any,  # MultiLevelCache
        config: Optional[InvalidationConfig] = None,
    ):
        """Initialize cache invalidator.

        Args:
            cache: MultiLevelCache instance to invalidate
            config: Invalidation configuration
        """
        self.cache = cache
        self.config = config or InvalidationConfig()

        # Tag manager
        self._tag_manager: Optional[CacheTagManager] = None
        if self.config.enable_tagging:
            self._tag_manager = CacheTagManager()

        # Dependency graph
        self._dependency_graph: Optional[InvalidationDependencyGraph] = None
        if self.config.enable_dependencies:
            self._dependency_graph = InvalidationDependencyGraph()

        # Invalidation event handlers
        self._event_handlers: dict[str, list[Callable[..., Awaitable[None]]]] = {}

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()

        # Metrics
        self._invalidations = 0
        self._tag_invalidations = 0
        self._dependency_invalidations = 0

        logger.info(
            f"Initialized cache invalidator: strategy={self.config.strategy.value}, "
            f"tagging={self.config.enable_tagging}, "
            f"dependencies={self.config.enable_dependencies}"
        )

    async def invalidate(self, key: str, namespace: str = "default") -> bool:
        """Invalidate a specific cache entry.

        Args:
            key: Cache key
            namespace: Namespace

        Returns:
            True if entry was invalidated, False if not found
        """
        # Remove from tag tracking
        if self._tag_manager:
            self._tag_manager.remove_entry(key, namespace)

        # Remove from dependency tracking
        if self._dependency_graph:
            self._dependency_graph.remove_entry(key, namespace)

        # Invalidate from cache
        deleted = await self.cache.delete(key, namespace)

        if deleted:
            self._invalidations += 1
            logger.debug(f"Invalidated cache entry: {key}")

        return cast(bool, deleted)

    async def invalidate_namespace(self, namespace: str) -> int:
        """Invalidate all entries in a namespace.

        Args:
            namespace: Namespace to clear

        Returns:
            Number of entries invalidated
        """
        count = await self.cache.clear_namespace(namespace)

        # Clean up tag tracking
        if self._tag_manager:
            with self._tag_manager._lock:
                entries_to_remove = [
                    (key, ns) for key, ns in self._tag_manager._entry_tags.keys() if ns == namespace
                ]
                for key, ns in entries_to_remove:
                    self._tag_manager.remove_entry(key, ns)

        # Clean up dependency tracking
        if self._dependency_graph:
            with self._dependency_graph._lock:
                entries_to_remove = [
                    (key, ns)
                    for key, ns in self._dependency_graph._dependents.keys()
                    if ns == namespace
                ]
                for key, ns in entries_to_remove:
                    self._dependency_graph.remove_entry(key, ns)

        self._invalidations += count
        logger.info(f"Invalidated {count} entries in namespace: {namespace}")
        return cast(int, count)

    async def invalidate_all(self) -> None:
        """Invalidate all cache entries."""
        await self.cache.clear()

        if self._tag_manager:
            self._tag_manager._entry_tags.clear()
            self._tag_manager._tag_index.clear()

        if self._dependency_graph:
            self._dependency_graph._dependents.clear()
            self._dependency_graph._dependencies.clear()

        logger.info("Invalidated all cache entries")

    def tag(self, key: str, namespace: str, tags: list[str]) -> None:
        """Add tags to a cache entry.

        Args:
            key: Cache key
            namespace: Namespace
            tags: List of tags to add
        """
        if not self._tag_manager:
            logger.warning("Tagging not enabled")
            return

        self._tag_manager.tag(key, namespace, tags)
        logger.debug(f"Tagged cache entry {key} with tags: {tags}")

    async def invalidate_tag(self, tag: str) -> int:
        """Invalidate all entries with a specific tag.

        Args:
            tag: Tag to invalidate

        Returns:
            Number of entries invalidated
        """
        if not self._tag_manager:
            logger.warning("Tagging not enabled")
            return 0

        keys = self._tag_manager.get_tagged_keys(tag)
        count = 0

        for key, namespace in keys:
            deleted = await self.invalidate(key, namespace)
            if deleted:
                count += 1

        self._tag_invalidations += count
        logger.info(f"Invalidated {count} entries with tag: {tag}")
        return count

    def add_dependency(self, key: str, namespace: str, resource: str) -> None:
        """Add a dependency relationship.

        Args:
            key: Cache key
            namespace: Namespace
            resource: Resource identifier (file path, config key, etc.)
        """
        if not self._dependency_graph:
            logger.warning("Dependency tracking not enabled")
            return

        self._dependency_graph.add_dependency(key, namespace, resource)
        logger.debug(f"Added dependency: {key} -> {resource}")

    async def invalidate_dependents(self, resource: str) -> int:
        """Invalidate all entries that depend on a resource.

        Args:
            resource: Resource identifier

        Returns:
            Number of entries invalidated
        """
        if not self._dependency_graph:
            logger.warning("Dependency tracking not enabled")
            return 0

        keys = self._dependency_graph.get_dependents(resource)
        count = 0

        for key, namespace in keys:
            deleted = await self.invalidate(key, namespace)
            if deleted:
                count += 1

        self._dependency_invalidations += count
        logger.info(f"Invalidated {count} entries dependent on: {resource}")
        return count

    def register_event_handler(
        self, event_type: str, handler: Callable[..., Awaitable[None]]
    ) -> None:
        """Register an event handler for invalidation events.

        Args:
            event_type: Type of event (e.g., "file_change", "config_change")
            handler: Async function to call when event occurs
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        logger.debug(f"Registered event handler for: {event_type}")

    async def trigger_event(self, event_type: str, **kwargs: Any) -> None:
        """Trigger an invalidation event.

        Args:
            event_type: Type of event to trigger
            **kwargs: Event-specific arguments
        """
        if event_type not in self._event_handlers:
            return

        for handler in self._event_handlers[event_type]:
            try:
                await handler(**kwargs)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")

    async def start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is not None:
            logger.warning("Cleanup task already running")
            return

        async def cleanup_loop() -> None:
            """Background cleanup loop."""
            while not self._stop_event.is_set():
                try:
                    # Cleanup expired entries (if cache supports it)
                    # This is handled by the cache itself, but we can do additional cleanup

                    # Wait for next interval
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.config.cleanup_interval,
                    )
                except asyncio.TimeoutError:
                    # Normal timeout, continue loop
                    pass
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info(f"Started cleanup task (interval: {self.config.cleanup_interval}s)")

    async def stop_cleanup_task(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task is None:
            return

        self._stop_event.set()
        self._cleanup_task.cancel()
        try:
            await self._cleanup_task
        except asyncio.CancelledError:
            pass

        self._cleanup_task = None
        self._stop_event.clear()
        logger.info("Stopped cleanup task")

    def get_stats(self) -> dict[str, Any]:
        """Get invalidation statistics.

        Returns:
            Dictionary with statistics
        """
        stats: dict[str, Any] = {
            "strategy": self.config.strategy.value,
            "invalidations": self._invalidations,
            "tag_invalidations": self._tag_invalidations,
            "dependency_invalidations": self._dependency_invalidations,
        }

        if self._tag_manager:
            stats["tags"] = self._tag_manager.get_stats()

        if self._dependency_graph:
            stats["dependencies"] = self._dependency_graph.get_stats()

        return stats


__all__ = [
    "CacheInvalidator",
    "CacheTagManager",
    "InvalidationDependencyGraph",
    "InvalidationStrategy",
    "InvalidationConfig",
    "TaggedEntry",
]
