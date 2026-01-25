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

"""Lazy invalidation cache for zero-latency cache marking.

This module provides a cache with lazy invalidation strategy where entries
are marked as stale immediately (zero latency) and cleaned up when accessed
or periodically.

Performance Benefits:
- Invalidation marking: 0ms (just set a flag)
- Cleanup deferred: Only when accessed or periodically
- No blocking on file edits
- Ideal for high-frequency file modifications

Design Patterns:
- Lazy Evaluation: Defer work until necessary
- Mark-and-Sweep: Mark stale, clean on access
- SRP: Focused only on lazy invalidation

Usage:
    cache = LazyInvalidationCache(ttl_seconds=300)

    # Set value
    cache.set("key", value, files=["/src/main.py"])

    # Mark as stale (zero latency)
    cache.mark_stale("/src/main.py")

    # Get will clean stale entries
    value = cache.get("key")  # Returns None if stale
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set

from victor.agent.cache.dependency_extractor import DependencyExtractor

logger = logging.getLogger(__name__)


class LazyInvalidationCache:
    """Cache with lazy invalidation (mark stale, clean on access).

    Instead of immediately removing cache entries, they are marked as stale
    and cleaned up later when accessed or during periodic cleanup.

    Performance:
    - Mark stale: 0ms (just set a flag)
    - Cleanup on access: Amortized O(1)
    - Periodic cleanup: O(n) but infrequent

    Thread-safe: All operations protected by threading.Lock.

    Attributes:
        _cache: OrderedDict maintaining insertion/access order
        _timestamps: Dict mapping keys to creation timestamps
        _stale_keys: Set of keys marked as stale
        _file_index: Reverse index mapping file paths to cache key sets
        _max_size: Maximum number of entries to cache
        _ttl_seconds: Time-to-live in seconds
        _cleanup_interval: Seconds between periodic cleanups
        _last_cleanup: Timestamp of last periodic cleanup

    Example:
        cache = LazyInvalidationCache(max_size=50, ttl_seconds=300)

        # Set value
        cache.set("key1", value, files=["/src/main.py"])

        # Mark as stale (zero latency)
        cache.mark_stale("/src/main.py")

        # Get will return None for stale entries
        value = cache.get("key1")  # Returns None
    """

    def __init__(
        self,
        max_size: int = 50,
        ttl_seconds: float = 300.0,
        cleanup_interval: float = 60.0,
        dependency_extractor: Optional[DependencyExtractor] = None,
    ):
        """Initialize the lazy invalidation cache.

        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live in seconds
            cleanup_interval: Seconds between periodic cleanups (default: 60s)
            dependency_extractor: Dependency extractor for auto-detecting files
        """
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._stale_keys: Set[str] = set()
        self._file_index: Dict[str, Set[str]] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        self._lock = threading.Lock()
        self._dependency_extractor = dependency_extractor or DependencyExtractor()

        # Metrics tracking
        self._stale_mark_count = 0
        self._lazy_cleanup_count = 0
        self._periodic_cleanup_count = 0

    def _is_expired(self, key: str) -> bool:
        """Check if a cache entry has expired.

        Args:
            key: Cache key to check

        Returns:
            True if entry is expired or timestamp missing
        """
        if key not in self._timestamps:
            return True
        age = time.time() - self._timestamps[key]
        return age > self._ttl_seconds

    def _is_stale(self, key: str) -> bool:
        """Check if a cache entry is marked as stale.

        Args:
            key: Cache key to check

        Returns:
            True if entry is marked as stale
        """
        return key in self._stale_keys

    def _extract_files(self, value: Any) -> Set[str]:
        """Extract file paths from cached value.

        Args:
            value: Cached value (ToolCallResult or dict)

        Returns:
            Set of file paths
        """
        files: Set[str] = set()

        # Handle ToolCallResult objects
        if hasattr(value, "arguments") and hasattr(value, "tool_name"):
            files.update(
                self._dependency_extractor.extract_file_dependencies(
                    value.tool_name,
                    value.arguments,
                )
            )

        # Handle dict values
        elif isinstance(value, dict):
            tool_name = value.get("tool_name", "")
            arguments = value.get("arguments", {})
            files.update(
                self._dependency_extractor.extract_file_dependencies(
                    tool_name,
                    arguments,
                )
            )

        return files

    def _add_to_file_index(self, key: str, value: Any) -> None:
        """Add cache entry to file reverse index.

        Args:
            key: Cache key
            value: Cached value
        """
        files = self._extract_files(value)

        for file_path in files:
            if file_path not in self._file_index:
                self._file_index[file_path] = set()
            self._file_index[file_path].add(key)

    def _remove_from_file_index(self, key: str) -> None:
        """Remove cache entry from file reverse index.

        Args:
            key: Cache key to remove
        """
        value = self._cache.get(key)
        if not value:
            return

        files = self._extract_files(value)

        for file_path in files:
            if file_path in self._file_index:
                self._file_index[file_path].discard(key)
                if not self._file_index[file_path]:
                    del self._file_index[file_path]

    def _cleanup_stale(self) -> int:
        """Clean up all stale entries from cache.

        Returns:
            Number of entries cleaned up
        """
        cleaned = 0

        for key in list(self._stale_keys):
            if key in self._cache:
                self._remove_from_file_index(key)
                del self._cache[key]
                self._timestamps.pop(key, None)
                cleaned += 1

        self._stale_keys.clear()
        return cleaned

    def _periodic_cleanupIfNeeded(self) -> None:
        """Run periodic cleanup if enough time has passed.

        Cleans both stale and expired entries.
        """
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        # Clean expired entries
        expired_keys = [k for k in self._cache if self._is_expired(k)]
        for key in expired_keys:
            self._remove_from_file_index(key)
            del self._cache[key]
            self._timestamps.pop(key, None)
            self._stale_keys.discard(key)

        # Clean stale entries
        cleaned = self._cleanup_stale()

        self._last_cleanup = now
        self._periodic_cleanup_count += cleaned

        if cleaned > 0 or len(expired_keys) > 0:
            logger.debug(
                f"Periodic cleanup: removed {len(expired_keys)} expired, "
                f"{cleaned} stale entries"
            )

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.

        Cleans stale entries on access. If entry is stale or expired,
        returns None and removes it.

        Thread-safe: Protected by threading.Lock.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found, stale, or expired
        """
        with self._lock:
            # Check periodic cleanup
            self._periodic_cleanupIfNeeded()

            if key not in self._cache:
                return None

            # Check if stale
            if self._is_stale(key):
                # Clean up stale entry
                self._remove_from_file_index(key)
                del self._cache[key]
                self._timestamps.pop(key, None)
                self._stale_keys.discard(key)
                self._lazy_cleanup_count += 1
                logger.debug(f"Cleaned stale entry on access: {key}")
                return None

            # Check TTL expiration
            if self._is_expired(key):
                self._remove_from_file_index(key)
                del self._cache[key]
                self._timestamps.pop(key, None)
                self._stale_keys.discard(key)
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

    def set(self, key: str, value: Any, files: Optional[List[str]] = None) -> None:
        """Set a value in the cache.

        Thread-safe: Protected by threading.Lock.

        Args:
            key: Cache key
            value: Value to cache
            files: Optional list of file paths (auto-extracted if None)
        """
        with self._lock:
            current_time = time.time()

            if key in self._cache:
                self._cache.move_to_end(key)
                self._remove_from_file_index(key)
                self._stale_keys.discard(key)  # Clear stale flag on update

            self._cache[key] = value
            self._timestamps[key] = current_time

            # Add to reverse index
            self._add_to_file_index(key, value)

            # Evict if over capacity (check both stale and expired)
            if len(self._cache) > self._max_size:
                # First, try to evict stale entries
                if self._stale_keys:
                    stale_key = next(iter(self._stale_keys))
                    self._remove_from_file_index(stale_key)
                    del self._cache[stale_key]
                    self._timestamps.pop(stale_key, None)
                    self._stale_keys.discard(stale_key)

                # Then try expired entries
                expired = [k for k in self._cache if self._is_expired(k)]
                for key in expired[: len(self._cache) - self._max_size]:
                    self._remove_from_file_index(key)
                    del self._cache[key]
                    self._timestamps.pop(key, None)
                    self._stale_keys.discard(key)

            # Finally, LRU eviction
            while len(self._cache) > self._max_size:
                oldest_key, _ = self._cache.popitem(last=False)
                self._remove_from_file_index(oldest_key)
                self._timestamps.pop(oldest_key, None)
                self._stale_keys.discard(oldest_key)

    def mark_stale(self, file_path: str) -> int:
        """Mark all cache entries for a file as stale (zero latency).

        This is O(k) where k is the number of keys for this file,
        but doesn't actually remove them from the cache.

        Thread-safe: Protected by threading.Lock.

        Args:
            file_path: Path of the modified file

        Returns:
            Number of cache entries marked as stale
        """
        with self._lock:
            if file_path not in self._file_index:
                return 0

            keys_to_mark = self._file_index[file_path]
            self._stale_keys.update(keys_to_mark)

            count = len(keys_to_mark)
            self._stale_mark_count += count

            logger.debug(
                f"Marked {count} cache entries as stale for {file_path} "
                f"(total stale: {len(self._stale_keys)})"
            )

            return count

    def mark_stale_batch(self, file_paths: List[str]) -> int:
        """Mark multiple files as stale in one operation.

        Thread-safe: Protected by threading.Lock.

        Args:
            file_paths: List of file paths to mark as stale

        Returns:
            Total number of cache entries marked as stale
        """
        with self._lock:
            total_marked = 0

            for file_path in file_paths:
                if file_path in self._file_index:
                    keys_to_mark = self._file_index[file_path]
                    self._stale_keys.update(keys_to_mark)
                    total_marked += len(keys_to_mark)

            self._stale_mark_count += total_marked

            logger.debug(
                f"Marked {total_marked} cache entries as stale for {len(file_paths)} files"
            )

            return total_marked

    def clear(self) -> None:
        """Clear all entries from the cache.

        Thread-safe: Protected by threading.Lock.
        """
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._stale_keys.clear()
            self._file_index.clear()

    def __len__(self) -> int:
        """Return the number of active (non-stale) entries in the cache.

        Thread-safe: Returns snapshot of current size.
        """
        with self._lock:
            return len(self._cache) - len(self._stale_keys)

    def items(self):
        """Return all active (non-stale, non-expired) items in the cache.

        Thread-safe: Returns snapshot of current items.
        """
        with self._lock:
            return [
                (k, v)
                for k, v in self._cache.items()
                if not self._is_stale(k) and not self._is_expired(k)
            ]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring.

        Thread-safe: Returns snapshot of current stats.

        Returns:
            Dict with cache stats including stale count and cleanup metrics
        """
        with self._lock:
            expired_count = sum(1 for k in self._cache if self._is_expired(k))
            active_count = len(self._cache) - len(self._stale_keys) - expired_count
            return {
                "size": len(self._cache),
                "active_count": active_count,
                "stale_count": len(self._stale_keys),
                "expired_count": expired_count,
                "max_size": self._max_size,
                "ttl_seconds": self._ttl_seconds,
                "file_index_size": len(self._file_index),
                # Cleanup metrics
                "stale_mark_count": self._stale_mark_count,
                "lazy_cleanup_count": self._lazy_cleanup_count,
                "periodic_cleanup_count": self._periodic_cleanup_count,
                "cleanup_interval_seconds": self._cleanup_interval,
            }


__all__ = ["LazyInvalidationCache"]
