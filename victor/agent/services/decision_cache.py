"""Multi-level decision cache for reducing redundant LLM calls.

Provides L1 (memory) and L2 (optional disk) caching of decision results
to avoid redundant computations. Uses LRU eviction with TTL-based expiration.

Architecture:
- L1 Cache: 100 entries, 5-minute TTL (in-memory)
- L2 Cache: 1000 entries, 1-hour TTL (optional disk backing)
- Cache promotion/demotion based on usage frequency
- TTL-based eviction with LRU fallback

Usage:
    cache = DecisionCache(l1_size=100, l1_ttl=300, l2_enabled=False)

    # Try to get cached result
    result = cache.get(decision_type, context)
    if result:
        return result

    # Compute decision and cache it
    result = await compute_decision(...)
    cache.put(decision_type, context, result)
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from victor.agent.decisions.schemas import DecisionType

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry with value and metadata."""

    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.monotonic)
    ttl: float = 300.0  # Default 5 minutes

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        age = time.monotonic() - self.timestamp
        return age > self.ttl

    def touch(self) -> None:
        """Update access time and count."""
        self.access_count += 1
        self.last_access = time.monotonic()


class LRUCache:
    """LRU cache with TTL-based expiration."""

    def __init__(self, max_size: int = 100, default_ttl: float = 300.0):
        """Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if exists and not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            return None

        entry = self._cache[key]

        # Check expiration
        if entry.is_expired():
            del self._cache[key]
            logger.debug("Cache entry expired: %s", key)
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.touch()

        logger.debug("Cache hit: %s (access_count=%d)", key, entry.access_count)
        return entry.value

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug("Evicted oldest entry: %s", oldest_key)

        # Create entry
        entry = CacheEntry(
            value=value,
            timestamp=time.monotonic(),
            ttl=ttl if ttl is not None else self.default_ttl,
        )

        self._cache[key] = entry
        self._cache.move_to_end(key)

        logger.debug("Cache put: %s (ttl=%.1fs)", key, entry.ttl)

    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry was found and removed
        """
        if key in self._cache:
            del self._cache[key]
            logger.debug("Cache invalidation: %s", key)
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        logger.debug("Cache cleared")

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.debug("Cleaned up %d expired entries", len(expired_keys))

        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_accesses = sum(entry.access_count for entry in self._cache.values())

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "total_accesses": total_accesses,
            "avg_access_count": total_accesses / len(self._cache) if self._cache else 0,
        }


class DiskCache:
    """Optional disk-backed L2 cache with persistence."""

    def __init__(
        self,
        cache_dir: Path,
        max_size: int = 1000,
        default_ttl: float = 3600.0,
    ):
        """Initialize disk cache.

        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.default_ttl = default_ttl

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory index of disk entries
        self._index: Dict[str, CacheEntry] = {}

        # Load existing index
        self._load_index()

    def _load_index(self) -> None:
        """Load cache index from disk."""
        index_file = self.cache_dir / "index.pkl"

        if index_file.exists():
            try:
                with open(index_file, "rb") as f:
                    self._index = pickle.load(f)
                logger.debug("Loaded disk cache index: %d entries", len(self._index))
            except Exception as e:
                logger.warning("Failed to load disk cache index: %s", e)
                self._index = {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        index_file = self.cache_dir / "index.pkl"

        try:
            with open(index_file, "wb") as f:
                pickle.dump(self._index, f)
        except Exception as e:
            logger.warning("Failed to save disk cache index: %s", e)

    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for a key.

        Args:
            key: Cache key

        Returns:
            Path to cache file
        """
        # Use hash of key as filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._index:
            return None

        entry = self._index[key]

        # Check expiration
        if entry.is_expired():
            self.invalidate(key)
            return None

        # Load value from disk
        cache_file = self._get_cache_file(key)

        if not cache_file.exists():
            # File missing, remove from index
            del self._index[key]
            return None

        try:
            with open(cache_file, "rb") as f:
                value = pickle.load(f)

            # Update access stats
            entry.touch()
            self._save_index()

            logger.debug("Disk cache hit: %s", key)
            return value

        except Exception as e:
            logger.warning("Failed to load disk cache entry %s: %s", key, e)
            self.invalidate(key)
            return None

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value in disk cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        # Evict oldest if at capacity
        if len(self._index) >= self.max_size and key not in self._index:
            oldest_key = min(self._index.keys(), key=lambda k: self._index[k].timestamp)
            self.invalidate(oldest_key)

        # Create entry
        entry = CacheEntry(
            value=None,  # Value stored on disk
            timestamp=time.monotonic(),
            ttl=ttl if ttl is not None else self.default_ttl,
        )

        # Save value to disk
        cache_file = self._get_cache_file(key)

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(value, f)

            self._index[key] = entry
            self._save_index()

            logger.debug("Disk cache put: %s (ttl=%.1fs)", key, entry.ttl)

        except Exception as e:
            logger.warning("Failed to save disk cache entry %s: %s", key, e)

    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry was found and removed
        """
        if key not in self._index:
            return False

        # Delete cache file
        cache_file = self._get_cache_file(key)

        try:
            if cache_file.exists():
                cache_file.unlink()
        except Exception as e:
            logger.warning("Failed to delete disk cache file %s: %s", cache_file, e)

        # Remove from index
        del self._index[key]
        self._save_index()

        logger.debug("Disk cache invalidation: %s", key)
        return True

    def clear(self) -> None:
        """Clear all cache entries."""
        # Delete all cache files
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning("Failed to delete cache file %s: %s", cache_file, e)

        # Clear index
        self._index.clear()
        self._save_index()

        logger.debug("Disk cache cleared")

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = [key for key, entry in self._index.items() if entry.is_expired()]

        for key in expired_keys:
            self.invalidate(key)

        if expired_keys:
            logger.debug("Cleaned up %d expired disk cache entries", len(expired_keys))

        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_accesses = sum(entry.access_count for entry in self._index.values())

        return {
            "size": len(self._index),
            "max_size": self.max_size,
            "total_accesses": total_accesses,
            "avg_access_count": total_accesses / len(self._index) if self._index else 0,
        }


class DecisionCache:
    """Multi-level decision cache with L1 (memory) and L2 (disk) tiers.

    Provides automatic promotion/demotion between cache levels based on
    access frequency. L1 is fast but small, L2 is slower but larger.

    Usage:
        cache = DecisionCache(l1_size=100, l2_enabled=False)
        result = cache.get(decision_type, context)
        if result is None:
            result = compute_decision(...)
            cache.put(decision_type, context, result)
    """

    def __init__(
        self,
        l1_size: int = 100,
        l1_ttl: float = 300.0,
        l2_enabled: bool = False,
        l2_size: int = 1000,
        l2_ttl: float = 3600.0,
        l2_cache_dir: Optional[Path] = None,
        promotion_threshold: int = 3,
    ):
        """Initialize decision cache.

        Args:
            l1_size: Maximum L1 cache entries
            l1_ttl: L1 TTL in seconds (default: 5 minutes)
            l2_enabled: Whether to enable L2 disk cache
            l2_size: Maximum L2 cache entries
            l2_ttl: L2 TTL in seconds (default: 1 hour)
            l2_cache_dir: Directory for L2 cache (default: ~/.victor/cache/decisions)
            promotion_threshold: Access count before promoting to L2
        """
        self.l1 = LRUCache(max_size=l1_size, default_ttl=l1_ttl)
        self.l2_enabled = l2_enabled
        self.promotion_threshold = promotion_threshold

        # Initialize L2 cache if enabled
        self.l2: Optional[DiskCache] = None
        if l2_enabled:
            cache_dir = l2_cache_dir or Path.home() / ".victor" / "cache" / "decisions"
            self.l2 = DiskCache(
                cache_dir=cache_dir,
                max_size=l2_size,
                default_ttl=l2_ttl,
            )

        # Statistics
        self._l1_hits = 0
        self._l2_hits = 0
        self._misses = 0

        logger.info(
            "DecisionCache initialized: L1=%d/%d, L2=%s",
            l1_size,
            l1_ttl,
            f"{l2_size}/{l2_ttl}" if l2_enabled else "disabled",
        )

    def get(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
    ) -> Optional[Any]:
        """Get cached decision result.

        Args:
            decision_type: Type of decision
            context: Decision context dict

        Returns:
            Cached result or None if not found
        """
        key = self._make_key(decision_type, context)

        # Try L1 first
        result = self.l1.get(key)
        if result is not None:
            self._l1_hits += 1

            # Check if should promote to L2
            if self.l2_enabled and self._should_promote(key):
                self._promote_to_l2(key, result)

            return result

        # Try L2 if enabled
        if self.l2_enabled and self.l2:
            result = self.l2.get(key)
            if result is not None:
                self._l2_hits += 1

                # Demote to L1 on access (L1 is faster)
                self.l1.put(key, result)

                return result

        # Cache miss
        self._misses += 1
        return None

    def put(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        result: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """Cache a decision result.

        Args:
            decision_type: Type of decision
            context: Decision context dict
            result: Result to cache
            ttl: Time-to-live in seconds (uses L1 default if None)
        """
        key = self._make_key(decision_type, context)

        # Always put in L1
        self.l1.put(key, result, ttl=ttl)

        # Promote to L2 if frequently accessed
        if self.l2_enabled and self._should_promote(key):
            self._promote_to_l2(key, result)

    def invalidate(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
    ) -> bool:
        """Invalidate a cached decision.

        Args:
            decision_type: Type of decision
            context: Decision context dict

        Returns:
            True if entry was found and invalidated
        """
        key = self._make_key(decision_type, context)

        invalidated = False

        # Invalidate from L1
        if self.l1.invalidate(key):
            invalidated = True

        # Invalidate from L2
        if self.l2_enabled and self.l2 and self.l2.invalidate(key):
            invalidated = True

        return invalidated

    def clear(self) -> None:
        """Clear all cached decisions."""
        self.l1.clear()
        if self.l2_enabled and self.l2:
            self.l2.clear()

        # Reset statistics
        self._l1_hits = 0
        self._l2_hits = 0
        self._misses = 0

        logger.debug("Decision cache cleared")

    def cleanup(self) -> int:
        """Clean up expired entries in both cache levels.

        Returns:
            Total number of entries removed
        """
        total = 0

        total += self.l1.cleanup_expired()

        if self.l2_enabled and self.l2:
            total += self.l2.cleanup_expired()

        return total

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self._l1_hits + self._l2_hits + self._misses

        return {
            "l1_hits": self._l1_hits,
            "l2_hits": self._l2_hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_rate": (
                (self._l1_hits + self._l2_hits) / total_requests if total_requests > 0 else 0.0
            ),
            "l1_stats": self.l1.get_stats(),
            "l2_stats": self.l2.get_stats() if self.l2 else None,
        }

    def _make_key(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
    ) -> str:
        """Create a cache key from decision type and context.

        Args:
            decision_type: Type of decision
            context: Decision context dict

        Returns:
            Cache key string
        """
        # Normalize context for consistent keys
        normalized_context = json.dumps(context, sort_keys=True, default=str)
        key_data = f"{decision_type.value}:{normalized_context}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _should_promote(self, key: str) -> bool:
        """Check if an entry should be promoted to L2.

        Args:
            key: Cache key

        Returns:
            True if should promote
        """
        if not self.l2_enabled or not self.l2:
            return False

        # Get L1 entry
        if key not in self.l1._cache:
            return False

        entry = self.l1._cache[key]

        # Promote if accessed frequently
        return entry.access_count >= self.promotion_threshold

    def _promote_to_l2(self, key: str, result: Any) -> None:
        """Promote an entry to L2 cache.

        Args:
            key: Cache key
            result: Result to cache in L2
        """
        if not self.l2_enabled or not self.l2:
            return

        # Get L1 entry to check TTL
        if key not in self.l1._cache:
            return

        l1_entry = self.l1._cache[key]

        # Put in L2 with same TTL
        self.l2.put(key, result, ttl=l1_entry.ttl)

        logger.debug("Promoted entry to L2: %s", key)


def create_decision_cache(
    l1_size: int = 100,
    l1_ttl: int = 300,
    l2_enabled: bool = False,
    **kwargs,
) -> DecisionCache:
    """Factory function to create a decision cache.

    Args:
        l1_size: Maximum L1 cache entries
        l1_ttl: L1 TTL in seconds
        l2_enabled: Whether to enable L2 disk cache
        **kwargs: Additional arguments to pass to DecisionCache

    Returns:
        Configured DecisionCache instance
    """
    return DecisionCache(
        l1_size=l1_size,
        l1_ttl=float(l1_ttl),
        l2_enabled=l2_enabled,
        **kwargs,
    )
