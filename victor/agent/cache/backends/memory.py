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

"""In-memory cache backend implementation.

This module provides MemoryCacheBackend, a fast in-memory cache
implementing ICacheBackend protocol. Suitable for single-process
deployments and testing.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from victor.protocols import ICacheBackend


@dataclass
class CacheEntry:
    """A single cache entry.

    Attributes:
        value: The cached value
        expires_at: Optional expiration timestamp
        created_at: When the entry was created
        access_count: Number of times this entry was accessed
        last_accessed: Last access time
    """

    value: Any
    expires_at: Optional[float]
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: Optional[float] = None


class MemoryCacheBackend(ICacheBackend):
    """In-memory cache backend implementation.

    This backend stores cached values in process memory. It provides:
    - Fast access (no I/O)
    - Thread-safe operations
    - TTL-based expiration
    - Namespace isolation
    - Statistics tracking

    Limitations:
    - Not shared across processes
    - Lost on process restart
    - Limited by process memory

    Use Cases:
    - Single-process deployments
    - Testing and development
    - Caching small datasets
    - Short-lived cache entries

    Example:
        ```python
        backend = MemoryCacheBackend(default_ttl_seconds=3600)

        # Cache a value
        await backend.set("result_123", computation_result, "tool", ttl_seconds=300)

        # Retrieve it
        value = await backend.get("result_123", "tool")

        # Clear all tool caches
        count = await backend.clear_namespace("tool")
        ```
    """

    def __init__(self, default_ttl_seconds: int = 3600, enable_stats: bool = True):
        """Initialize the in-memory cache backend.

        Args:
            default_ttl_seconds: Default TTL for entries without explicit TTL
            enable_stats: Enable statistics tracking
        """
        self._default_ttl = default_ttl_seconds
        self._enable_stats = enable_stats

        # Namespace-isolated storage: {namespace: {key: CacheEntry}}
        self._store: dict[str, dict[str, CacheEntry]] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
        }

    async def get(self, key: str, namespace: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key
            namespace: Namespace for isolation

        Returns:
            Cached value or None if not found or expired
        """
        with self._lock:
            # Get namespace store
            namespace_store = self._store.get(namespace)
            if namespace_store is None:
                if self._enable_stats:
                    self._stats["misses"] += 1
                return None

            # Get entry
            entry = namespace_store.get(key)
            if entry is None:
                if self._enable_stats:
                    self._stats["misses"] += 1
                return None

            # Check expiration
            if entry.expires_at is not None and time.time() > entry.expires_at:
                # Expired - remove it
                del namespace_store[key]
                if self._enable_stats:
                    self._stats["misses"] += 1
                    self._stats["evictions"] += 1
                return None

            # Update access stats
            if self._enable_stats:
                entry.access_count += 1
                entry.last_accessed = time.time()
                self._stats["hits"] += 1

            return entry.value

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
        """
        with self._lock:
            # Get or create namespace store
            if namespace not in self._store:
                self._store[namespace] = {}

            # Calculate expiration
            ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
            expires_at = time.time() + ttl if ttl > 0 else None

            # Create entry
            entry = CacheEntry(
                value=value,
                expires_at=expires_at,
            )

            # Store it
            self._store[namespace][key] = entry

            if self._enable_stats:
                self._stats["sets"] += 1

    async def delete(self, key: str, namespace: str) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key to delete
            namespace: Namespace of the key

        Returns:
            True if key was deleted, False if not found
        """
        with self._lock:
            namespace_store = self._store.get(namespace)
            if namespace_store is None:
                return False

            if key in namespace_store:
                del namespace_store[key]
                if self._enable_stats:
                    self._stats["deletes"] += 1
                return True

            return False

    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace.

        Args:
            namespace: Namespace to clear

        Returns:
            Number of keys deleted
        """
        with self._lock:
            namespace_store = self._store.get(namespace)
            if namespace_store is None:
                return 0

            count = len(namespace_store)
            del self._store[namespace]

            return count

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with backend statistics
        """
        with self._lock:
            # Calculate total keys
            total_keys = sum(len(store) for store in self._store.values())

            stats = {
                "backend_type": "memory",
                "keys": total_keys,
            }

            if self._enable_stats:
                stats["hits"] = self._stats["hits"]
                stats["misses"] = self._stats["misses"]
                stats["sets"] = self._stats["sets"]
                stats["deletes"] = self._stats["deletes"]
                stats["evictions"] = self._stats["evictions"]

                # Calculate hit rate
                total_accesses = self._stats["hits"] + self._stats["misses"]
                if total_accesses > 0:
                    stats["hit_rate"] = self._stats["hits"] / total_accesses
                else:
                    stats["hit_rate"] = 0.0

            return stats

    def cleanup_expired(self) -> int:
        """Clean up expired entries (utility method).

        This method is not part of ICacheBackend but is useful
        for manual cleanup or testing.

        Returns:
            Number of entries removed
        """
        with self._lock:
            now = time.time()
            removed = 0

            for namespace_store in self._store.values():
                expired_keys = [
                    key
                    for key, entry in namespace_store.items()
                    if entry.expires_at is not None and entry.expires_at < now
                ]

                for key in expired_keys:
                    del namespace_store[key]
                    removed += 1

            if self._enable_stats:
                self._stats["evictions"] += removed

            return removed


__all__ = [
    "MemoryCacheBackend",
    "CacheEntry",
]
