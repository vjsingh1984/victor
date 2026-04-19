"""Unified cache registry for coordinated invalidation (OBS-1).

Provides a single registry where all framework caches register themselves.
Enables:
- Monitoring: total cache size, hit rates, staleness across all caches
- Coordinated invalidation: clear all caches, clear by category, cascade
- Lifecycle: cleanup stale entries, enforce global size limits

Usage:
    from victor.runtime.cache_registry import CacheRegistry

    # Register a cache
    registry = CacheRegistry.get_instance()
    registry.register("tool_embeddings", my_cache, category="embedding")

    # Monitor
    status = registry.get_status()

    # Invalidate
    registry.invalidate_category("embedding")
    registry.invalidate_all()
"""

from __future__ import annotations

import logging
import threading
import time

from victor.core.registry_base import SingletonRegistry
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class CacheCategory(Enum):
    """Categories for cache grouping and selective invalidation."""

    EMBEDDING = "embedding"
    TOOL = "tool"
    VERTICAL = "vertical"
    SCHEMA = "schema"
    QUERY = "query"
    SESSION = "session"
    GENERAL = "general"


@runtime_checkable
class InvalidatableCache(Protocol):
    """Protocol for caches that support invalidation."""

    def invalidate(self, *args: Any, **kwargs: Any) -> None:
        """Invalidate some or all entries."""
        ...


@dataclass
class CacheEntry:
    """Metadata about a registered cache."""

    name: str
    cache: Any
    category: CacheCategory
    registered_at: float = field(default_factory=time.time)
    invalidate_fn: Optional[Callable[[], None]] = None

    def invalidate(self) -> bool:
        """Invalidate this cache. Returns True if successful."""
        try:
            if self.invalidate_fn:
                self.invalidate_fn()
                return True
            elif hasattr(self.cache, "invalidate"):
                self.cache.invalidate()
                return True
            elif hasattr(self.cache, "clear"):
                self.cache.clear()
                return True
            elif hasattr(self.cache, "clear_cache"):
                self.cache.clear_cache()
                return True
            return False
        except Exception as e:
            logger.warning("Failed to invalidate cache '%s': %s", self.name, e)
            return False

    def get_size(self) -> Optional[int]:
        """Get cache size if available."""
        for attr in ("__len__", "size"):
            fn = getattr(self.cache, attr, None)
            if callable(fn):
                try:
                    return fn()
                except Exception:
                    pass
        return None


@dataclass
class CacheRegistryStatus:
    """Aggregated status of all registered caches."""

    total_caches: int
    caches_by_category: Dict[str, int]
    total_known_size: int
    entries: List[Dict[str, Any]]


class CacheRegistry(SingletonRegistry["CacheRegistry"]):
    """Unified registry for all framework caches.

    Thread-safe singleton that tracks all caches for coordinated
    invalidation and monitoring.
    """

    def __init__(self) -> None:
        super().__init__()
        self._entries: Dict[str, CacheEntry] = {}
        self._entry_lock = threading.Lock()

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton. For testing only."""
        cls.reset_instance()

    def register(
        self,
        name: str,
        cache: Any,
        category: str | CacheCategory = CacheCategory.GENERAL,
        invalidate_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        """Register a cache for unified management.

        Args:
            name: Unique identifier for this cache
            cache: The cache object
            category: Category for grouped invalidation
            invalidate_fn: Optional custom invalidation function
        """
        if isinstance(category, str):
            try:
                category = CacheCategory(category)
            except ValueError:
                category = CacheCategory.GENERAL

        with self._entry_lock:
            self._entries[name] = CacheEntry(
                name=name,
                cache=cache,
                category=category,
                invalidate_fn=invalidate_fn,
            )
        logger.debug("Registered cache '%s' (category=%s)", name, category.value)

    def unregister(self, name: str) -> bool:
        """Remove a cache from the registry."""
        with self._entry_lock:
            return self._entries.pop(name, None) is not None

    def invalidate_all(self) -> int:
        """Invalidate all registered caches. Returns count of successful invalidations."""
        with self._entry_lock:
            entries = list(self._entries.values())
        count = sum(1 for e in entries if e.invalidate())
        logger.info("Invalidated %d/%d caches", count, len(entries))
        return count

    def invalidate_category(self, category: str | CacheCategory) -> int:
        """Invalidate all caches in a category."""
        if isinstance(category, str):
            try:
                category = CacheCategory(category)
            except ValueError:
                return 0
        with self._entry_lock:
            entries = [e for e in self._entries.values() if e.category == category]
        count = sum(1 for e in entries if e.invalidate())
        logger.info(
            "Invalidated %d/%d caches in category '%s'",
            count,
            len(entries),
            category.value,
        )
        return count

    def invalidate_by_name(self, name: str) -> bool:
        """Invalidate a specific cache by name."""
        with self._entry_lock:
            entry = self._entries.get(name)
        if entry:
            return entry.invalidate()
        return False

    def get_status(self) -> CacheRegistryStatus:
        """Get aggregated status of all registered caches."""
        with self._entry_lock:
            entries = list(self._entries.values())

        by_category: Dict[str, int] = {}
        total_size = 0
        entry_infos = []

        for e in entries:
            cat = e.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
            size = e.get_size()
            if size is not None:
                total_size += size
            entry_infos.append(
                {
                    "name": e.name,
                    "category": cat,
                    "size": size,
                    "age_seconds": time.time() - e.registered_at,
                }
            )

        return CacheRegistryStatus(
            total_caches=len(entries),
            caches_by_category=by_category,
            total_known_size=total_size,
            entries=entry_infos,
        )

    def list_caches(self) -> List[str]:
        """List all registered cache names."""
        with self._entry_lock:
            return list(self._entries.keys())
