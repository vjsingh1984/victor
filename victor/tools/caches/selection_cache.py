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

"""Tool selection cache for performance optimization.

This module provides a high-performance caching layer for tool selection operations.
It extends UniversalRegistry with TTL-based caching specifically optimized for
tool selection use cases.

Cache Types:
    1. Query Selection Cache: Caches selections based on query + tools + config
    2. Context-Aware Cache: Caches selections including conversation context
    3. RL Ranking Cache: Caches RL-based tool rankings

Expected Performance Improvement:
    - 30-50% reduction in tool selection latency
    - 40-50% hit rate for query cache
    - 30-40% hit rate for context cache
    - 60-70% hit rate for RL ranking cache

Memory Usage:
    - Approximately 100-200MB for 1000 cached selections
    - Automatic LRU eviction when limit reached

Example:
    from victor.tools.caches import ToolSelectionCache

    cache = ToolSelectionCache()

    # Store selection result
    cache.put_query(
        cache_key="abc123...",
        tools=["read", "write", "edit"],
        ttl=3600  # 1 hour
    )

    # Retrieve selection result
    result = cache.get_query("abc123...")
    if result is not None:
        # Cache hit - use cached tools
        selected_tools = result.value
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from victor.core.registries import CacheStrategy, UniversalRegistry
from victor.providers.base import ToolDefinition

if TYPE_CHECKING:
    from victor.tools.caches.cache_keys import CacheKeyGenerator

logger = logging.getLogger(__name__)


@dataclass
class CachedSelection:
    """A cached tool selection result.

    Attributes:
        value: List of selected tool names
        tools: Full ToolDefinition objects (optional, for complete cache)
        timestamp: When this selection was cached
        hit_count: Number of times this cache entry was accessed
        ttl: Time-to-live in seconds
        metadata: Additional metadata (selection time, scores, latency, etc.)
    """

    value: List[str]
    tools: List[ToolDefinition] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0
    ttl: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    selection_latency_ms: float = 0.0  # Track selection time for performance metrics

    def is_expired(self) -> bool:
        """Check if this cached selection has expired.

        Returns:
            True if TTL has elapsed
        """
        if not self.ttl:
            return False
        return (time.time() - self.timestamp) > self.ttl

    def record_hit(self) -> None:
        """Record a cache hit for metrics."""
        self.hit_count += 1

    def get_age_seconds(self) -> float:
        """Get age of this cache entry in seconds.

        Returns:
            Age in seconds since caching
        """
        return time.time() - self.timestamp


@dataclass
class CacheMetrics:
    """Metrics for cache performance tracking.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        evictions: Number of entries evicted
        total_lookups: Total number of cache lookups
        total_entries: Current number of cached entries
        memory_usage_bytes: Estimated memory usage
        total_latency_saved_ms: Total latency saved by cache hits (ms)
        avg_latency_per_hit_ms: Average latency saved per cache hit (ms)
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_lookups: int = 0
    total_entries: int = 0
    memory_usage_bytes: int = 0
    total_latency_saved_ms: float = 0.0
    avg_latency_per_hit_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate as a percentage (0.0 - 1.0)
        """
        if self.total_lookups == 0:
            return 0.0
        return self.hits / self.total_lookups

    def record_hit(self, latency_saved_ms: float = 0.0) -> None:
        """Record a cache hit.

        Args:
            latency_saved_ms: Latency saved by this cache hit (ms)
        """
        self.hits += 1
        self.total_lookups += 1
        if latency_saved_ms > 0:
            self.total_latency_saved_ms += latency_saved_ms
            self.avg_latency_per_hit_ms = self.total_latency_saved_ms / self.hits

    def record_miss(self) -> None:
        """Record a cache miss."""
        self.misses += 1
        self.total_lookups += 1

    def record_eviction(self) -> None:
        """Record a cache eviction."""
        self.evictions += 1

    def reset(self) -> None:
        """Reset all metrics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_lookups = 0


class ToolSelectionCache:
    """High-performance cache for tool selection operations.

    Provides three separate cache namespaces:
        1. "query": Query-based selections (1 hour TTL)
        2. "context": Context-aware selections (5 minutes TTL)
        3. "rl": RL-based rankings (1 hour TTL)

    Thread-safe with LRU eviction and comprehensive metrics.

    Example:
        cache = ToolSelectionCache(max_size=1000)

        # Query cache
        cache.put_query(key="abc123", tools=["read", "write"], ttl=3600)
        result = cache.get("abc123", namespace="query")

        # Context cache
        cache.put_context(key="def456", tools=["read", "edit"], ttl=300)
        result = cache.get("def456", namespace="context")

        # Get metrics
        metrics = cache.get_metrics()
        print(f"Hit rate: {metrics.hit_rate:.1%}")
    """

    # Default TTL values (in seconds)
    DEFAULT_QUERY_TTL: int = 3600  # 1 hour
    DEFAULT_CONTEXT_TTL: int = 300  # 5 minutes
    DEFAULT_RL_TTL: int = 3600  # 1 hour

    # Namespace names
    NAMESPACE_QUERY: str = "query"
    NAMESPACE_CONTEXT: str = "context"
    NAMESPACE_RL: str = "rl"

    def __init__(
        self,
        max_size: int = 1000,
        query_ttl: int = DEFAULT_QUERY_TTL,
        context_ttl: int = DEFAULT_CONTEXT_TTL,
        rl_ttl: int = DEFAULT_RL_TTL,
        enabled: bool = True,
        use_cache_config: bool = True,
    ):
        """Initialize tool selection cache.

        Args:
            max_size: Maximum number of entries per namespace (deprecated, use cache_config)
            query_ttl: Default TTL for query cache (seconds) (deprecated, use cache_config)
            context_ttl: Default TTL for context cache (seconds) (deprecated, use cache_config)
            rl_ttl: Default TTL for RL cache (seconds) (deprecated, use cache_config)
            enabled: Whether caching is enabled
            use_cache_config: Use centralized cache_config module (recommended)
        """
        self._enabled = enabled

        # Always store max_size for backward compatibility and stats
        self._max_size = max_size
        self._query_ttl = query_ttl
        self._context_ttl = context_ttl
        self._rl_ttl = rl_ttl

        # Use centralized cache configuration if requested (recommended for Phase 6)
        if use_cache_config:
            from victor.core.registries.cache_config import get_cache_config_manager

            cache_manager = get_cache_config_manager()

            # Configure tool_selection_query registry
            try:
                query_config = cache_manager.get_config(
                    "tool_selection_query", env_prefix="VICTOR_CACHE_"
                )
                self._query_registry = cache_manager.configure_registry(
                    UniversalRegistry, "tool_selection_query", env_prefix="VICTOR_CACHE_"
                )
                # Update max_size from config if available
                if query_config.max_size is not None:
                    self._max_size = query_config.max_size
                # Store TTL for validation
                self._query_ttl = query_config.ttl if query_config.ttl else query_ttl
            except Exception as e:
                logger.warning(
                    f"Failed to use cache config for tool_selection_query: {e}. Using defaults."
                )
                self._query_registry = UniversalRegistry.get_registry(
                    "tool_selection_query",
                    cache_strategy=CacheStrategy.LRU,
                    max_size=max_size,
                )
                self._query_ttl = query_ttl

            # Configure tool_selection_context registry
            try:
                context_config = cache_manager.get_config(
                    "tool_selection_context", env_prefix="VICTOR_CACHE_"
                )
                self._context_registry = cache_manager.configure_registry(
                    UniversalRegistry, "tool_selection_context", env_prefix="VICTOR_CACHE_"
                )
                # Update max_size from config if available (and larger)
                if context_config.max_size is not None and context_config.max_size > self._max_size:
                    self._max_size = context_config.max_size
                self._context_ttl = context_config.ttl if context_config.ttl else context_ttl
            except Exception as e:
                logger.warning(
                    f"Failed to use cache config for tool_selection_context: {e}. Using defaults."
                )
                self._context_registry = UniversalRegistry.get_registry(
                    "tool_selection_context",
                    cache_strategy=CacheStrategy.LRU,
                    max_size=max_size,
                )
                self._context_ttl = context_ttl

            # Configure tool_selection_rl registry
            try:
                rl_config = cache_manager.get_config(
                    "tool_selection_rl", env_prefix="VICTOR_CACHE_"
                )
                self._rl_registry = cache_manager.configure_registry(
                    UniversalRegistry, "tool_selection_rl", env_prefix="VICTOR_CACHE_"
                )
                # Update max_size from config if available (and larger)
                if rl_config.max_size is not None and rl_config.max_size > self._max_size:
                    self._max_size = rl_config.max_size
                self._rl_ttl = rl_config.ttl if rl_config.ttl else rl_ttl
            except Exception as e:
                logger.warning(
                    f"Failed to use cache config for tool_selection_rl: {e}. Using defaults."
                )
                self._rl_registry = UniversalRegistry.get_registry(
                    "tool_selection_rl",
                    cache_strategy=CacheStrategy.LRU,
                    max_size=max_size,
                )
                self._rl_ttl = rl_ttl

            logger.info(
                f"ToolSelectionCache initialized using centralized cache_config: enabled={enabled}, max_size={self._max_size}"
            )
        else:
            # Legacy initialization with hardcoded values
            # (max_size and TTLs already set above)

            # Create registries for each namespace
            self._query_registry = UniversalRegistry.get_registry(
                "tool_selection_query",
                cache_strategy=CacheStrategy.LRU,
                max_size=max_size,
            )
            self._context_registry = UniversalRegistry.get_registry(
                "tool_selection_context",
                cache_strategy=CacheStrategy.LRU,
                max_size=max_size,
            )
            self._rl_registry = UniversalRegistry.get_registry(
                "tool_selection_rl",
                cache_strategy=CacheStrategy.LRU,
                max_size=max_size,
            )

            logger.info(
                f"ToolSelectionCache initialized (legacy): enabled={enabled}, max_size={max_size}, "
                f"TTL(query={query_ttl}s, context={context_ttl}s, rl={rl_ttl}s)"
            )

        # Metrics per namespace
        self._metrics: Dict[str, CacheMetrics] = {
            self.NAMESPACE_QUERY: CacheMetrics(),
            self.NAMESPACE_CONTEXT: CacheMetrics(),
            self.NAMESPACE_RL: CacheMetrics(),
        }

        # Lock for metrics updates
        self._metrics_lock = threading.RLock()

    # ========================================================================
    # Cache Access Methods
    # ========================================================================

    def get(
        self,
        key: str,
        namespace: str = NAMESPACE_QUERY,
    ) -> Optional[CachedSelection]:
        """Get cached selection by key.

        Args:
            key: Cache key
            namespace: Cache namespace (query, context, rl)

        Returns:
            CachedSelection if found and not expired, None otherwise
        """
        if not self._enabled:
            return None

        registry = self._get_registry(namespace)
        if not registry:
            logger.warning(f"Unknown cache namespace: {namespace}")
            return None

        entry = registry.get(key, default=None)
        if not entry:
            self._record_miss(namespace)
            return None

        # Record hit with latency saved
        latency_saved = entry.selection_latency_ms
        entry.record_hit()
        self._record_hit(namespace, latency_saved)

        logger.debug(
            f"Cache hit: namespace={namespace}, key={key[:8]}..., " f"saved={latency_saved:.2f}ms"
        )
        return entry

    def put(
        self,
        key: str,
        value: List[str],
        tools: Optional[List[ToolDefinition]] = None,
        namespace: str = NAMESPACE_QUERY,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        selection_latency_ms: float = 0.0,
    ) -> None:
        """Store selection in cache.

        Args:
            key: Cache key
            value: List of tool names
            tools: Optional full ToolDefinition objects
            namespace: Cache namespace (query, context, rl)
            ttl: Time-to-live in seconds (None for namespace default)
            metadata: Optional metadata (scores, latency, etc.)
            selection_latency_ms: Time taken for this selection (for metrics)
        """
        if not self._enabled:
            return

        registry = self._get_registry(namespace)
        if not registry:
            logger.warning(f"Unknown cache namespace: {namespace}")
            return

        # Use namespace default TTL if not specified
        if ttl is None:
            ttl = self._get_default_ttl(namespace)

        # Create cached selection with latency tracking
        cached = CachedSelection(
            value=value,
            tools=tools or [],
            ttl=ttl,
            metadata=metadata or {},
            selection_latency_ms=selection_latency_ms,
        )

        # Store in registry
        registry.register(key, cached, ttl=ttl)
        logger.debug(
            f"Cache put: namespace={namespace}, key={key[:8]}..., "
            f"tools={len(value)}, latency={selection_latency_ms:.2f}ms"
        )

    # ========================================================================
    # Convenience Methods for Each Namespace
    # ========================================================================

    def get_query(self, key: str) -> Optional[CachedSelection]:
        """Get cached query-based selection.

        Args:
            key: Cache key

        Returns:
            CachedSelection if found, None otherwise
        """
        return self.get(key, namespace=self.NAMESPACE_QUERY)

    def put_query(
        self,
        key: str,
        value: List[str],
        tools: Optional[List[ToolDefinition]] = None,
        ttl: Optional[int] = None,
        selection_latency_ms: float = 0.0,
    ) -> None:
        """Store query-based selection in cache.

        Args:
            key: Cache key
            value: List of tool names
            tools: Optional full ToolDefinition objects
            ttl: Time-to-live (defaults to DEFAULT_QUERY_TTL)
            selection_latency_ms: Time taken for this selection (for metrics)
        """
        self.put(
            key,
            value,
            tools,
            namespace=self.NAMESPACE_QUERY,
            ttl=ttl,
            selection_latency_ms=selection_latency_ms,
        )

    def get_context(self, key: str) -> Optional[CachedSelection]:
        """Get cached context-aware selection.

        Args:
            key: Cache key

        Returns:
            CachedSelection if found, None otherwise
        """
        return self.get(key, namespace=self.NAMESPACE_CONTEXT)

    def put_context(
        self,
        key: str,
        value: List[str],
        tools: Optional[List[ToolDefinition]] = None,
        ttl: Optional[int] = None,
        selection_latency_ms: float = 0.0,
    ) -> None:
        """Store context-aware selection in cache.

        Args:
            key: Cache key
            value: List of tool names
            tools: Optional full ToolDefinition objects
            ttl: Time-to-live (defaults to DEFAULT_CONTEXT_TTL)
            selection_latency_ms: Time taken for this selection (for metrics)
        """
        self.put(
            key,
            value,
            tools,
            namespace=self.NAMESPACE_CONTEXT,
            ttl=ttl,
            selection_latency_ms=selection_latency_ms,
        )

    def get_rl(self, key: str) -> Optional[CachedSelection]:
        """Get cached RL-based ranking.

        Args:
            key: Cache key

        Returns:
            CachedSelection if found, None otherwise
        """
        return self.get(key, namespace=self.NAMESPACE_RL)

    def put_rl(
        self,
        key: str,
        value: List[str],
        tools: Optional[List[ToolDefinition]] = None,
        ttl: Optional[int] = None,
        selection_latency_ms: float = 0.0,
    ) -> None:
        """Store RL-based ranking in cache.

        Args:
            key: Cache key
            value: List of tool names (ranked)
            tools: Optional full ToolDefinition objects
            ttl: Time-to-live (defaults to DEFAULT_RL_TTL)
            selection_latency_ms: Time taken for this selection (for metrics)
        """
        self.put(
            key,
            value,
            tools,
            namespace=self.NAMESPACE_RL,
            ttl=ttl,
            selection_latency_ms=selection_latency_ms,
        )

    # ========================================================================
    # Cache Invalidation
    # ========================================================================

    def invalidate(
        self,
        key: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> int:
        """Invalidate cache entries.

        Args:
            key: Specific key to invalidate (None = invalidate by namespace)
            namespace: Namespace to invalidate (None = invalidate all)

        Returns:
            Number of entries invalidated

        Example:
            # Invalidate specific key
            cache.invalidate(key="abc123", namespace="query")

            # Invalidate entire namespace
            cache.invalidate(namespace="query")

            # Invalidate all caches
            cache.invalidate()
        """
        if namespace:
            registry = self._get_registry(namespace)
            if registry:
                count = registry.invalidate(key=key)
                logger.info(f"Invalidated {count} entries in namespace '{namespace}'")
                return count
        else:
            # Invalidate all namespaces
            total = 0
            for ns in [self.NAMESPACE_QUERY, self.NAMESPACE_CONTEXT, self.NAMESPACE_RL]:
                registry = self._get_registry(ns)
                if registry:
                    total += registry.invalidate()
            logger.info(f"Invalidated all {total} cache entries")
            return total

        return 0

    def invalidate_on_tools_change(self) -> None:
        """Invalidate all caches when tools registry changes.

        Call this when:
        - Tools are added/removed
        - Tool definitions are modified
        - Tool metadata is updated
        """
        self.invalidate()
        logger.info("All caches invalidated due to tools registry change")

    # ========================================================================
    # Metrics
    # ========================================================================

    def get_metrics(self, namespace: Optional[str] = None) -> CacheMetrics:
        """Get cache metrics.

        Args:
            namespace: Optional namespace to get metrics for

        Returns:
            CacheMetrics for the namespace (or combined if None)
        """
        with self._metrics_lock:
            if namespace:
                # Update entry count on-demand
                metrics = self._metrics.get(namespace, CacheMetrics())
                registry = self._get_registry(namespace)
                if registry:
                    stats = registry.get_stats()
                    metrics.total_entries = stats.get("total_entries", 0)
                return metrics

            # Combine metrics from all namespaces
            combined = CacheMetrics()
            for ns in [self.NAMESPACE_QUERY, self.NAMESPACE_CONTEXT, self.NAMESPACE_RL]:
                metrics = self._metrics.get(ns, CacheMetrics())
                registry = self._get_registry(ns)
                if registry:
                    stats = registry.get_stats()
                    metrics.total_entries = stats.get("total_entries", 0)
                combined.hits += metrics.hits
                combined.misses += metrics.misses
                combined.evictions += metrics.evictions
                combined.total_lookups += metrics.total_lookups
                combined.total_entries += metrics.total_entries
            return combined

    def reset_metrics(self, namespace: Optional[str] = None) -> None:
        """Reset cache metrics.

        Args:
            namespace: Optional namespace to reset (all if None)
        """
        with self._metrics_lock:
            if namespace:
                if namespace in self._metrics:
                    self._metrics[namespace].reset()
            else:
                for metrics in self._metrics.values():
                    metrics.reset()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dictionary with cache stats including latency metrics
        """
        stats = {
            "enabled": self._enabled,
            "max_size": self._max_size,
            "namespaces": {},
        }

        for ns in [self.NAMESPACE_QUERY, self.NAMESPACE_CONTEXT, self.NAMESPACE_RL]:
            metrics = self.get_metrics(ns)
            registry = self._get_registry(ns)
            registry_stats = registry.get_stats() if registry else {}

            stats["namespaces"][ns] = {
                "ttl": self._get_default_ttl(ns),
                "hits": metrics.hits,
                "misses": metrics.misses,
                "hit_rate": metrics.hit_rate,
                "evictions": metrics.evictions,
                "total_entries": metrics.total_entries,
                "utilization": registry_stats.get("utilization", 0.0),
                "total_latency_saved_ms": metrics.total_latency_saved_ms,
                "avg_latency_per_hit_ms": metrics.avg_latency_per_hit_ms,
            }

        # Add combined stats
        combined = self.get_metrics()
        stats["combined"] = {
            "hits": combined.hits,
            "misses": combined.misses,
            "hit_rate": combined.hit_rate,
            "evictions": combined.evictions,
            "total_entries": combined.total_entries,
            "total_latency_saved_ms": combined.total_latency_saved_ms,
            "avg_latency_per_hit_ms": combined.avg_latency_per_hit_ms,
        }

        return stats

    # ========================================================================
    # Configuration
    # ========================================================================

    @property
    def enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable caching."""
        self._enabled = True
        logger.info("Tool selection caching enabled")

    def disable(self) -> None:
        """Disable caching."""
        self._enabled = False
        logger.info("Tool selection caching disabled")

    # ========================================================================
    # Internal Methods
    # ========================================================================

    def _get_registry(self, namespace: str) -> Optional[UniversalRegistry]:
        """Get registry for namespace.

        Args:
            namespace: Cache namespace

        Returns:
            UniversalRegistry or None if unknown namespace
        """
        registries = {
            self.NAMESPACE_QUERY: self._query_registry,
            self.NAMESPACE_CONTEXT: self._context_registry,
            self.NAMESPACE_RL: self._rl_registry,
        }
        return registries.get(namespace)

    def _get_default_ttl(self, namespace: str) -> int:
        """Get default TTL for namespace.

        Args:
            namespace: Cache namespace

        Returns:
            Default TTL in seconds
        """
        ttls = {
            self.NAMESPACE_QUERY: self._query_ttl,
            self.NAMESPACE_CONTEXT: self._context_ttl,
            self.NAMESPACE_RL: self._rl_ttl,
        }
        return ttls.get(namespace, self._query_ttl)

    def _record_hit(self, namespace: str, latency_saved_ms: float = 0.0) -> None:
        """Record a cache hit.

        Args:
            namespace: Cache namespace
            latency_saved_ms: Latency saved by this cache hit (ms)
        """
        with self._metrics_lock:
            if namespace in self._metrics:
                self._metrics[namespace].record_hit(latency_saved_ms)

    def _record_miss(self, namespace: str) -> None:
        """Record a cache miss.

        Args:
            namespace: Cache namespace
        """
        with self._metrics_lock:
            if namespace in self._metrics:
                self._metrics[namespace].record_miss()


# Global singleton instance
_global_cache: Optional[ToolSelectionCache] = None
_cache_lock = threading.Lock()


def get_tool_selection_cache(
    max_size: int = 1000,
    query_ttl: int = ToolSelectionCache.DEFAULT_QUERY_TTL,
    context_ttl: int = ToolSelectionCache.DEFAULT_CONTEXT_TTL,
    rl_ttl: int = ToolSelectionCache.DEFAULT_RL_TTL,
    enabled: bool = True,
) -> ToolSelectionCache:
    """Get global tool selection cache instance.

    Creates cache on first call with specified configuration.

    Args:
        max_size: Maximum number of entries per namespace
        query_ttl: Default TTL for query cache (seconds)
        context_ttl: Default TTL for context cache (seconds)
        rl_ttl: Default TTL for RL cache (seconds)
        enabled: Whether caching is enabled

    Returns:
        Shared ToolSelectionCache instance
    """
    global _global_cache
    with _cache_lock:
        if _global_cache is None:
            _global_cache = ToolSelectionCache(
                max_size=max_size,
                query_ttl=query_ttl,
                context_ttl=context_ttl,
                rl_ttl=rl_ttl,
                enabled=enabled,
            )
        return _global_cache


def invalidate_tool_selection_cache() -> None:
    """Invalidate all tool selection caches.

    Convenience function for global cache invalidation.
    """
    cache = get_tool_selection_cache()
    cache.invalidate()


def reset_tool_selection_cache() -> None:
    """Reset the global tool selection cache singleton.

    This is primarily used for testing to ensure isolation between tests.
    After calling this, the next call to get_tool_selection_cache() will
    create a new instance.

    This also clears the underlying UniversalRegistries to ensure
    complete isolation between tests.
    """
    global _global_cache
    with _cache_lock:
        # First invalidate all caches in the current instance if it exists
        if _global_cache is not None:
            _global_cache.invalidate()

        # Reset the global cache instance
        _global_cache = None

        # Also clear the underlying UniversalRegistries
        # These are singletons that persist across cache instances
        for registry_name in [
            "tool_selection_query",
            "tool_selection_context",
            "tool_selection_rl",
        ]:
            try:
                registry = UniversalRegistry.get_registry(registry_name)
                registry.invalidate()
            except Exception:
                # Ignore any errors during cleanup
                pass


__all__ = [
    "CachedSelection",
    "CacheMetrics",
    "ToolSelectionCache",
    "get_tool_selection_cache",
    "invalidate_tool_selection_cache",
    "reset_tool_selection_cache",
]
