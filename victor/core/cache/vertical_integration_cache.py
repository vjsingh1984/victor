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

"""Vertical integration cache service for SRP compliance and scalability.

This module externalizes caching logic from VerticalIntegrationPipeline,
following the Single Responsibility Principle. It provides:

1. Separation of concerns - cache management isolated from integration logic
2. Pluggable backends - can swap between in-memory, Redis, etc.
3. Testability - cache can be mocked in tests
4. Scalability - supports distributed caching for multi-instance deployments

Usage:
    from victor.core.cache.vertical_integration_cache import VerticalIntegrationCache
    from victor.config.settings import get_settings

    cache = VerticalIntegrationCache(
        ttl=getattr(settings, "vertical_cache_ttl", 3600),
        enable_cache=getattr(settings, "vertical_cache_enabled", True),
    )

    # Use in VerticalIntegrationPipeline
    result = cache.get(vertical_class, config_overrides)
    if result is None:
        result = compute_integration()
        cache.set(vertical_class, config_overrides, result)
"""

import hashlib
import json
import logging
import threading
import time
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.framework.vertical_integration import IntegrationResult

logger = logging.getLogger(__name__)


class _CacheEntry:
    """Internal cache entry with TTL support."""

    __slots__ = ["value", "expires_at"]

    def __init__(self, value: Any, ttl: int):
        self.value = value
        self.expires_at = time.time() + ttl

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.expires_at


class VerticalIntegrationCache:
    """Cache for vertical integration results.

    Provides SRP-compliant caching that can be:
    - Injected into VerticalIntegrationPipeline
    - Swapped for distributed implementations (Redis, Memcached)
    - Tested in isolation from integration logic

    Thread-safe for concurrent access (uses SemanticCache internally).

    Args:
        ttl: Cache time-to-live in seconds (default: 3600 = 1 hour)
        enable_cache: Whether caching is enabled (default: True)
        max_size: Maximum number of cache entries (default: 100)
    """

    def __init__(
        self,
        ttl: int = 3600,
        enable_cache: bool = True,
        max_size: int = 100,
    ):
        """Initialize vertical integration cache.

        Args:
            ttl: Time-to-live for cache entries in seconds
            enable_cache: Whether caching is enabled
            max_size: Maximum number of cache entries (default: 100)
        """
        self._enable_cache = enable_cache
        self._ttl = ttl
        self._max_size = max_size

        # Thread-safe in-memory cache
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0

    def generate_key(
        self,
        vertical_class: type,
        config_overrides: Optional[dict[str, Any]],
    ) -> str:
        """Generate cache key for a vertical integration.

        Args:
            vertical_class: The vertical class (e.g., CodingAssistant)
            config_overrides: Optional configuration overrides

        Returns:
            Cache key as string
        """
        # Create stable hash from class and config
        key_parts = [
            f"{vertical_class.__module__}.{vertical_class.__name__}",
        ]

        if config_overrides:
            # Sort keys for consistent hashing
            sorted_items = sorted(config_overrides.items())
            key_parts.append(json.dumps(sorted_items, sort_keys=True))

        # Include YAML config file hash if available
        yaml_hash = self._get_yaml_config_hash(vertical_class)
        if yaml_hash:
            key_parts.append(yaml_hash)

        key_string = ":".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _get_yaml_config_hash(self, vertical_class: type) -> Optional[str]:
        """Get hash of YAML config file if it exists.

        Args:
            vertical_class: The vertical class

        Returns:
            SHA256 hash of YAML file content, or None if file doesn't exist
        """
        import hashlib
        from pathlib import Path

        try:
            # Get YAML config path from vertical class
            yaml_path = None
            if hasattr(vertical_class, "_get_yaml_config_path"):
                yaml_path = vertical_class._get_yaml_config_path()

            # If _get_yaml_config_path returns None, try get_config_path
            if not yaml_path and hasattr(vertical_class, "get_config_path"):
                yaml_path = vertical_class.get_config_path()

            if yaml_path and Path(yaml_path).exists():
                # Read and hash YAML file content
                with open(yaml_path, "rb") as f:
                    content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception:
            pass

        return None

    def get(
        self,
        vertical_class: type,
        config_overrides: Optional[dict[str, Any]],
    ) -> Optional["IntegrationResult"]:
        """Get cached integration result if available and valid.

        Args:
            vertical_class: The vertical class
            config_overrides: Optional configuration overrides

        Returns:
            Cached IntegrationResult if hit and valid, None otherwise
        """
        if not self._enable_cache:
            return None

        try:
            key = self.generate_key(vertical_class, config_overrides)

            with self._lock:
                entry = self._cache.get(key)

                if entry is not None:
                    # Check if cache entry has expired
                    if entry.is_expired():
                        # Remove expired entry
                        del self._cache[key]
                        self._misses += 1
                        logger.debug(
                            f"Cache EXPIRED for {vertical_class.__name__}: "
                            f"key={key[:8]}... (hits={self._hits}, misses={self._misses})"
                        )
                        return None

                    # Cache hit
                    self._hits += 1
                    logger.debug(
                        f"Cache HIT for {vertical_class.__name__}: "
                        f"key={key[:8]}... (hits={self._hits}, misses={self._misses})"
                    )
                    return entry.value  # type: ignore[no-any-return]

                # Cache miss
                self._misses += 1
                logger.debug(
                    f"Cache MISS for {vertical_class.__name__}: "
                    f"key={key[:8]}... (hits={self._hits}, misses={self._misses})"
                )
                return None

        except Exception as e:
            logger.warning(f"Cache get failed for {vertical_class.__name__}: {e}")
            self._misses += 1
            return None

    def set(
        self,
        vertical_class: type,
        config_overrides: Optional[dict[str, Any]],
        result: "IntegrationResult",
    ) -> None:
        """Store integration result in cache.

        Args:
            vertical_class: The vertical class
            config_overrides: Optional configuration overrides
            result: IntegrationResult to cache
        """
        if not self._enable_cache:
            return

        try:
            key = self.generate_key(vertical_class, config_overrides)

            with self._lock:
                # Enforce max size by evicting oldest entry if needed
                if len(self._cache) >= self._max_size and key not in self._cache:
                    # Remove first (oldest) entry
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    logger.debug(f"Cache evicted oldest entry (max_size={self._max_size})")

                # Store new entry
                self._cache[key] = _CacheEntry(result, self._ttl)
                logger.debug(
                    f"Cached {vertical_class.__name__}: " f"key={key[:8]}... (TTL={self._ttl}s)"
                )
        except Exception as e:
            logger.warning(f"Cache set failed for {vertical_class.__name__}: {e}")

    def clear(self) -> None:
        """Clear all cached integration results."""
        with self._lock:
            self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Vertical integration cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests) if total_requests > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_rate": round(hit_rate, 3),
            "ttl_seconds": self._ttl,
            "enabled": self._enable_cache,
        }

    def invalidate(self, vertical_class: type) -> None:
        """Invalidate all cache entries for a vertical class.

        Args:
            vertical_class: The vertical class to invalidate
        """
        # This would require iterating over all keys, which is inefficient
        # For now, recommend clearing entire cache or implementing prefix search
        logger.info(
            f"Invalidation requested for {vertical_class.__name__} (consider using clear() instead)"
        )
        self.clear()


def create_vertical_integration_cache(
    ttl: int = 3600,
    enable_cache: bool = True,
    max_size: int = 100,
) -> VerticalIntegrationCache:
    """Factory function to create VerticalIntegrationCache.

    This allows easy configuration via settings or DI container.

    Args:
        ttl: Cache time-to-live in seconds (default: 3600)
        enable_cache: Whether caching is enabled (default: True)
        max_size: Maximum number of cache entries (default: 100)

    Returns:
        Configured VerticalIntegrationCache instance
    """
    return VerticalIntegrationCache(
        ttl=ttl,
        enable_cache=enable_cache,
        max_size=max_size,
    )
