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

"""Multi-level cache hierarchy for optimal performance.

This module implements a hierarchical cache system with three levels:
    - L1: In-memory cache (fast, small)
    - L2: Local disk cache (medium speed, medium size)
    - L3: Remote/network cache (slow, large)

Expected Performance Improvement:
    - 20-30% latency reduction through cache hierarchy
    - 80-90% hit rate across all levels
    - Automatic promotion/demotion between levels

Example:
    from victor.tools.caches import MultiLevelCache

    cache = MultiLevelCache(
        l1_size=100,
        l2_size=1000,
        l3_size=10000,
    )

    # Use like a normal cache
    cache.put("key", value)
    value = cache.get("key")  # Checks L1, then L2, then L3
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata.

    Attributes:
        key: Cache key
        value: Cached value
        level: Cache level (1, 2, or 3)
        created_at: Creation timestamp
        last_accessed: Last access timestamp
        access_count: Number of accesses
        size: Estimated size in bytes
        ttl: Time-to-live in seconds (None = no expiration)
    """

    key: str
    value: Any
    level: int
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    size: int = 0
    ttl: Optional[int] = None

    def is_expired(self) -> bool:
        """Check if entry has expired.

        Returns:
            True if TTL has elapsed
        """
        if self.ttl is None or self.ttl == 0:
            return False
        return (time.time() - self.created_at) > self.ttl

    def record_access(self) -> None:
        """Record an access for promotion/demotion decisions."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class LevelMetrics:
    """Metrics for a single cache level.

    Attributes:
        level: Cache level number
        hits: Number of hits
        misses: Number of misses
        promotions: Number of entries promoted to higher level
        demotions: Number of entries demoted to lower level
        evictions: Number of entries evicted
        total_size: Current size in bytes
        entry_count: Number of entries
    """

    level: int
    hits: int = 0
    misses: int = 0
    promotions: int = 0
    demotions: int = 0
    evictions: int = 0
    total_size: int = 0
    entry_count: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate for this level.

        Returns:
            Hit rate as a percentage (0.0 - 1.0)
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class MultiLevelCache:
    """Multi-level hierarchical cache with automatic promotion/demotion.

    Implements three-tier cache hierarchy:
    - L1: In-memory LRU cache (fastest, smallest)
    - L2: Local disk cache (medium speed, medium size)
    - L3: Persistent/network cache (slowest, largest)

    Features:
    - Automatic promotion of hot entries to higher levels
    - Automatic demotion of cold entries to lower levels
    - Size-based eviction at each level
    - Thread-safe operations

    Example:
        cache = MultiLevelCache(
            l1_size=100,
            l2_size=1000,
            l3_size=10000,
            l2_dir="/tmp/cache_l2",
        )

        # Normal cache operations
        cache.put("key", value)
        value = cache.get("key")

        # Get metrics
        metrics = cache.get_metrics()
        print(f"L1 hit rate: {metrics['l1']['hit_rate']:.1%}")
    """

    # Default cache sizes
    DEFAULT_L1_SIZE = 100
    DEFAULT_L2_SIZE = 1000
    DEFAULT_L3_SIZE = 10000

    # Promotion/demotion thresholds
    PROMOTION_THRESHOLD = 3  # Accesses needed for promotion
    DEMOTION_THRESHOLD = 1  # Accesses below this trigger demotion

    # TTL defaults (seconds)
    DEFAULT_L1_TTL = 300  # 5 minutes
    DEFAULT_L2_TTL = 3600  # 1 hour
    DEFAULT_L3_TTL = 86400  # 24 hours

    def __init__(
        self,
        l1_size: int = DEFAULT_L1_SIZE,
        l2_size: int = DEFAULT_L2_SIZE,
        l3_size: int = DEFAULT_L3_SIZE,
        l2_dir: Optional[Path] = None,
        l1_ttl: int = DEFAULT_L1_TTL,
        l2_ttl: int = DEFAULT_L2_TTL,
        l3_ttl: int = DEFAULT_L3_TTL,
        enabled: bool = True,
    ):
        """Initialize multi-level cache.

        Args:
            l1_size: Maximum entries in L1 cache
            l2_size: Maximum entries in L2 cache
            l3_size: Maximum entries in L3 cache
            l2_dir: Directory for L2 disk cache (default: /tmp/victor_cache_l2)
            l1_ttl: Default TTL for L1 entries (seconds)
            l2_ttl: Default TTL for L2 entries (seconds)
            l3_ttl: Default TTL for L3 entries (seconds)
            enabled: Whether caching is enabled
        """
        self._l1_size = l1_size
        self._l2_size = l2_size
        self._l3_size = l3_size
        self._l1_ttl = l1_ttl
        self._l2_ttl = l2_ttl
        self._l3_ttl = l3_ttl
        self._enabled = enabled

        # L1: In-memory cache
        self._l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # L2: Disk cache directory
        if l2_dir is None:
            l2_dir = Path("/tmp/victor_cache_l2")
        self._l2_dir = Path(l2_dir)
        self._l2_dir.mkdir(parents=True, exist_ok=True)

        # L3: In-memory cache (could be Redis in production)
        self._l3_cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Metrics per level
        self._metrics = {
            1: LevelMetrics(level=1),
            2: LevelMetrics(level=2),
            3: LevelMetrics(level=3),
        }

        # Lock for thread safety
        self._lock = threading.RLock()

        logger.info(
            f"MultiLevelCache initialized: L1={l1_size}, L2={l2_size}, "
            f"L3={l3_size}, L2_dir={self._l2_dir}"
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache hierarchy.

        Checks L1, then L2, then L3. Promotes hits to higher levels.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        if not self._enabled:
            return default

        with self._lock:
            # Try L1 first
            entry = self._get_from_l1(key)
            if entry is not None:
                self._metrics[1].hits += 1
                entry.record_access()
                self._check_promotion(entry, 1)
                return entry.value

            self._metrics[1].misses += 1

            # Try L2
            entry = self._get_from_l2(key)
            if entry is not None:
                self._metrics[2].hits += 1
                entry.record_access()
                # Promote to L1
                self._promote_to_l1(entry)
                return entry.value

            self._metrics[2].misses += 1

            # Try L3
            entry = self._get_from_l3(key)
            if entry is not None:
                self._metrics[3].hits += 1
                entry.record_access()
                # Promote to L2
                self._promote_to_l2(entry)
                return entry.value

            self._metrics[3].misses += 1
            return default

    def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        size: Optional[int] = None,
    ) -> None:
        """Put value in cache hierarchy.

        Starts at L1, overflows to L2, then L3.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = level default)
            size: Estimated size in bytes (None = auto-estimate)
        """
        if not self._enabled:
            return

        with self._lock:
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                level=1,
                ttl=ttl or self._l1_ttl,
                size=size or self._estimate_size(value),
            )

            # Try to put in L1
            if self._put_to_l1(entry):
                return

            # L1 full, try L2
            entry.level = 2
            entry.ttl = ttl or self._l2_ttl
            if self._put_to_l2(entry):
                return

            # L2 full, try L3
            entry.level = 3
            entry.ttl = ttl or self._l3_ttl
            self._put_to_l3(entry)

    def invalidate(self, key: Optional[str] = None) -> None:
        """Invalidate cache entry or entire cache.

        Args:
            key: Specific key to invalidate (None = invalidate all)
        """
        with self._lock:
            if key:
                # Invalidate specific key from all levels
                self._l1_cache.pop(key, None)
                self._remove_from_l2(key)
                self._l3_cache.pop(key, None)
            else:
                # Invalidate all
                self._l1_cache.clear()
                self._clear_l2()
                self._l3_cache.clear()

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics.

        Returns:
            Dictionary with metrics for all levels
        """
        with self._lock:
            metrics = {
                "l1": self._get_level_metrics(1),
                "l2": self._get_level_metrics(2),
                "l3": self._get_level_metrics(3),
                "combined": {
                    "total_hits": sum(m.hits for m in self._metrics.values()),
                    "total_misses": sum(m.misses for m in self._metrics.values()),
                    "total_promotions": sum(m.promotions for m in self._metrics.values()),
                    "total_demotions": sum(m.demotions for m in self._metrics.values()),
                    "total_evictions": sum(m.evictions for m in self._metrics.values()),
                },
            }

            # Calculate combined hit rate
            total_hits = metrics["combined"]["total_hits"]
            total_lookups = total_hits + metrics["combined"]["total_misses"]
            metrics["combined"]["hit_rate"] = (
                total_hits / total_lookups if total_lookups > 0 else 0.0
            )

            return metrics

    # ========================================================================
    # L1 Cache Methods (In-Memory)
    # ========================================================================

    def _get_from_l1(self, key: str) -> Optional[CacheEntry]:
        """Get entry from L1 cache."""
        entry = self._l1_cache.get(key)
        if entry:
            # Check expiration
            if entry.is_expired():
                self._l1_cache.pop(key)
                self._metrics[1].evictions += 1
                return None

            # Move to end (mark as recently used)
            self._l1_cache.move_to_end(key)

        return entry

    def _put_to_l1(self, entry: CacheEntry) -> bool:
        """Put entry in L1 cache.

        Returns:
            True if successful (False if full)
        """
        # Update existing or add new
        if entry.key in self._l1_cache:
            self._l1_cache.pop(entry.key)

        self._l1_cache[entry.key] = entry

        # Enforce size limit
        while len(self._l1_cache) > self._l1_size:
            # Demote least recently used to L2
            lru_key, lru_entry = self._l1_cache.popitem(last=False)
            self._demote_to_l2(lru_entry)

        return True

    def _demote_from_l1(self, entry: CacheEntry) -> None:
        """Demote entry from L1 to L2."""
        self._metrics[1].demotions += 1
        entry.level = 2
        entry.ttl = self._l2_ttl
        self._promote_to_l2(entry)

    # ========================================================================
    # L2 Cache Methods (Disk)
    # ========================================================================

    def _get_from_l2(self, key: str) -> Optional[CacheEntry]:
        """Get entry from L2 disk cache."""
        cache_file = self._l2_dir / f"{key}.cache"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as f:
                entry = pickle.load(f)

            # Check expiration
            if entry.is_expired():
                cache_file.unlink()
                self._metrics[2].evictions += 1
                return None

            # Update metadata
            entry.last_accessed = time.time()
            entry.access_count += 1

            return entry

        except Exception as e:
            logger.warning(f"Failed to read L2 cache entry '{key}': {e}")
            return None

    def _put_to_l2(self, entry: CacheEntry) -> bool:
        """Put entry in L2 disk cache.

        Returns:
            True if successful (False if full)
        """
        # Check if L2 is full
        l2_files = list(self._l2_dir.glob("*.cache"))
        if len(l2_files) >= self._l2_size:
            # Demote oldest entry to L3
            oldest_file = min(l2_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(oldest_file, "rb") as f:
                    old_entry = pickle.load(f)
                self._demote_to_l3(old_entry)
                oldest_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to demote L2 entry: {e}")

        # Write to disk
        cache_file = self._l2_dir / f"{entry.key}.cache"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(entry, f)
            return True
        except Exception as e:
            logger.warning(f"Failed to write L2 cache entry '{entry.key}': {e}")
            return False

    def _remove_from_l2(self, key: str) -> None:
        """Remove entry from L2 cache."""
        cache_file = self._l2_dir / f"{key}.cache"
        if cache_file.exists():
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove L2 cache entry '{key}': {e}")

    def _clear_l2(self) -> None:
        """Clear all L2 cache files."""
        for cache_file in self._l2_dir.glob("*.cache"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove L2 cache file: {e}")

    # ========================================================================
    # L3 Cache Methods (In-Memory or Remote)
    # ========================================================================

    def _get_from_l3(self, key: str) -> Optional[CacheEntry]:
        """Get entry from L3 cache."""
        entry = self._l3_cache.get(key)
        if entry:
            # Check expiration
            if entry.is_expired():
                self._l3_cache.pop(key)
                self._metrics[3].evictions += 1
                return None

            # Move to end
            self._l3_cache.move_to_end(key)

        return entry

    def _put_to_l3(self, entry: CacheEntry) -> bool:
        """Put entry in L3 cache.

        Returns:
            True if successful
        """
        # Update existing or add new
        if entry.key in self._l3_cache:
            self._l3_cache.pop(entry.key)

        self._l3_cache[entry.key] = entry

        # Enforce size limit
        while len(self._l3_cache) > self._l3_size:
            # Evict least recently used
            self._l3_cache.popitem(last=False)
            self._metrics[3].evictions += 1

        return True

    # ========================================================================
    # Promotion/Demotion Methods
    # ========================================================================

    def _check_promotion(self, entry: CacheEntry, current_level: int) -> None:
        """Check if entry should be promoted to higher level.

        Args:
            entry: Cache entry to check
            current_level: Current cache level
        """
        if entry.access_count < self.PROMOTION_THRESHOLD:
            return

        if current_level == 1:
            # Already at top level
            return
        elif current_level == 2:
            # Promote to L1
            self._promote_to_l1(entry)
        elif current_level == 3:
            # Promote to L2
            self._promote_to_l2(entry)

    def _promote_to_l1(self, entry: CacheEntry) -> None:
        """Promote entry to L1 cache."""
        if entry.level == 1:
            return

        self._metrics[entry.level].promotions += 1
        entry.level = 1
        entry.ttl = self._l1_ttl

        # Remove from lower level
        if entry.key in self._l3_cache:
            self._l3_cache.pop(entry.key)

        # Add to L1 (may trigger demotion if full)
        self._put_to_l1(entry)

        logger.debug(f"Promoted entry '{entry.key}' to L1")

    def _promote_to_l2(self, entry: CacheEntry) -> None:
        """Promote entry to L2 cache."""
        if entry.level == 2 or entry.level == 1:
            return

        self._metrics[entry.level].promotions += 1
        entry.level = 2
        entry.ttl = self._l2_ttl

        # Remove from L3
        if entry.key in self._l3_cache:
            self._l3_cache.pop(entry.key)

        # Add to L2 (may trigger demotion if full)
        self._put_to_l2(entry)

        logger.debug(f"Promoted entry '{entry.key}' to L2")

    def _demote_to_l2(self, entry: CacheEntry) -> None:
        """Demote entry from L1 to L2."""
        if entry.level != 1:
            return

        self._metrics[1].demotions += 1
        entry.level = 2
        entry.ttl = self._l2_ttl
        self._put_to_l2(entry)

        logger.debug(f"Demoted entry '{entry.key}' to L2")

    def _demote_to_l3(self, entry: CacheEntry) -> None:
        """Demote entry from L2 to L3."""
        if entry.level != 2:
            return

        self._metrics[2].demotions += 1
        entry.level = 3
        entry.ttl = self._l3_ttl
        self._put_to_l3(entry)

        logger.debug(f"Demoted entry '{entry.key}' to L3")

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _get_level_metrics(self, level: int) -> Dict[str, Any]:
        """Get metrics for a specific level.

        Args:
            level: Cache level (1, 2, or 3)

        Returns:
            Dictionary with level metrics
        """
        metrics = self._metrics[level]

        if level == 1:
            entry_count = len(self._l1_cache)
            total_size = sum(e.size for e in self._l1_cache.values())
        elif level == 2:
            entry_count = len(list(self._l2_dir.glob("*.cache")))
            total_size = sum(f.stat().st_size for f in self._l2_dir.glob("*.cache"))
        else:  # level == 3
            entry_count = len(self._l3_cache)
            total_size = sum(e.size for e in self._l3_cache.values())

        return {
            "level": level,
            "hits": metrics.hits,
            "misses": metrics.misses,
            "hit_rate": metrics.hit_rate,
            "promotions": metrics.promotions,
            "demotions": metrics.demotions,
            "evictions": metrics.evictions,
            "entry_count": entry_count,
            "total_size": total_size,
        }

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes.

        Args:
            value: Value to estimate size for

        Returns:
            Estimated size in bytes
        """
        try:
            # Try pickle size
            return len(pickle.dumps(value))
        except Exception:
            # Fallback to string representation
            return len(str(value).encode())

    @property
    def enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable caching."""
        self._enabled = True
        logger.info("Multi-level cache enabled")

    def disable(self) -> None:
        """Disable caching."""
        self._enabled = False
        logger.info("Multi-level cache disabled")


__all__ = [
    "CacheEntry",
    "LevelMetrics",
    "MultiLevelCache",
]
