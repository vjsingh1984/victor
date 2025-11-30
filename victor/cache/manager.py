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

"""Tiered cache manager with memory and disk caching."""

import hashlib
import json
import logging
from typing import Any, Optional, Dict
import threading

from cachetools import TTLCache  # type: ignore[import-untyped]
import diskcache

from victor.cache.config import CacheConfig

logger = logging.getLogger(__name__)


class CacheManager:
    """Tiered cache manager with L1 (memory) and L2 (disk) caching.

    Architecture:
    - L1: Fast in-memory cache using cachetools (TTL-based)
    - L2: Persistent disk cache using diskcache (survives restarts)

    Features:
    - Automatic tiering (checks memory first, then disk)
    - Thread-safe operations
    - TTL support at both levels
    - Size limits
    - Statistics tracking
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize cache manager.

        Args:
            config: Cache configuration (uses defaults if None)
        """
        self.config = config or CacheConfig()
        self._lock = threading.RLock()

        # Statistics
        self._stats: Dict[str, float | int] = {
            "memory_hits": 0,
            "memory_misses": 0,
            "disk_hits": 0,
            "disk_misses": 0,
            "sets": 0,
            "evictions": 0,
        }

        # Initialize L1 memory cache
        if self.config.enable_memory:
            self._memory_cache: Optional[TTLCache] = TTLCache(
                maxsize=self.config.memory_max_size,
                ttl=self.config.memory_ttl,
            )
        else:
            self._memory_cache = None

        # Initialize L2 disk cache
        if self.config.enable_disk:
            self._disk_cache: Optional[diskcache.Cache] = diskcache.Cache(
                directory=str(self.config.disk_path),
                size_limit=self.config.disk_max_size,
            )
        else:
            self._disk_cache = None

        logger.info(
            "Cache initialized: memory=%s, disk=%s",
            self.config.enable_memory,
            self.config.enable_disk,
        )

    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from cache.

        Checks L1 (memory) first, then L2 (disk).
        If found in disk, promotes to memory.

        Args:
            key: Cache key
            namespace: Cache namespace for organization

        Returns:
            Cached value or None if not found
        """
        cache_key = self._make_key(key, namespace)

        with self._lock:
            # Try L1 memory cache
            if self._memory_cache is not None:
                try:
                    value = self._memory_cache.get(cache_key)
                    if value is not None:
                        self._stats["memory_hits"] += 1
                        logger.debug("Memory cache hit: %s", cache_key)
                        return value
                    self._stats["memory_misses"] += 1
                except KeyError:
                    self._stats["memory_misses"] += 1

            # Try L2 disk cache
            if self._disk_cache is not None:
                try:
                    value = self._disk_cache.get(cache_key)
                    if value is not None:
                        self._stats["disk_hits"] += 1
                        logger.debug("Disk cache hit: %s", cache_key)

                        # Promote to memory cache
                        if self._memory_cache is not None:
                            self._memory_cache[cache_key] = value

                        return value
                    self._stats["disk_misses"] += 1
                except KeyError:
                    self._stats["disk_misses"] += 1

            logger.debug("Cache miss: %s", cache_key)
            return None

    def set(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in cache.

        Stores in both L1 (memory) and L2 (disk) if enabled.

        Args:
            key: Cache key
            value: Value to cache
            namespace: Cache namespace
            ttl: Custom TTL in seconds (uses config default if None)

        Returns:
            True if successfully cached
        """
        cache_key = self._make_key(key, namespace)

        with self._lock:
            success = False

            # Store in L1 memory cache
            if self._memory_cache is not None:
                try:
                    self._memory_cache[cache_key] = value
                    success = True
                except Exception as e:
                    logger.warning("Failed to store in memory cache: %s", e)

            # Store in L2 disk cache
            if self._disk_cache is not None:
                try:
                    expire_time = ttl or self.config.disk_ttl
                    self._disk_cache.set(cache_key, value, expire=expire_time)
                    success = True
                except Exception as e:
                    logger.warning("Failed to store in disk cache: %s", e)

            if success:
                self._stats["sets"] += 1
                logger.debug("Cached: %s", cache_key)

            return success

    def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete value from cache.

        Removes from both L1 and L2 if present.

        Args:
            key: Cache key
            namespace: Cache namespace

        Returns:
            True if deleted from at least one cache
        """
        cache_key = self._make_key(key, namespace)

        with self._lock:
            deleted = False

            # Remove from L1
            if self._memory_cache is not None:
                try:
                    del self._memory_cache[cache_key]
                    deleted = True
                except KeyError:
                    pass

            # Remove from L2
            if self._disk_cache is not None:
                try:
                    deleted = self._disk_cache.delete(cache_key) or deleted
                except Exception as e:
                    logger.warning("Failed to delete from disk cache: %s", e)

            if deleted:
                logger.debug("Deleted from cache: %s", cache_key)

            return deleted

    def clear(self, namespace: Optional[str] = None) -> int:
        """Clear cache entries.

        Args:
            namespace: If provided, clear only this namespace. Otherwise clear all.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = 0

            if namespace:
                # Clear specific namespace
                if self._memory_cache is not None:
                    keys_to_delete = [
                        k for k in self._memory_cache.keys() if k.startswith(f"{namespace}:")
                    ]
                    for key in keys_to_delete:
                        try:
                            del self._memory_cache[key]
                            count += 1
                        except KeyError:
                            pass

                if self._disk_cache is not None:
                    # diskcache namespace clear by iteration
                    to_delete = [
                        k for k in self._disk_cache.iterkeys() if k.startswith(f"{namespace}:")
                    ]
                    for key in to_delete:
                        try:
                            del self._disk_cache[key]
                            count += 1
                        except Exception:
                            pass
            else:
                # Clear all
                if self._memory_cache is not None:
                    count += len(self._memory_cache)
                    self._memory_cache.clear()

                if self._disk_cache is not None:
                    count += len(self._disk_cache)
                    self._disk_cache.clear()

            logger.info("Cleared %d cache entries", count)
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            stats = self._stats.copy()

            # Calculate hit rates
            total_memory = stats["memory_hits"] + stats["memory_misses"]
            total_disk = stats["disk_hits"] + stats["disk_misses"]

            stats["memory_hit_rate"] = (
                stats["memory_hits"] / total_memory if total_memory > 0 else 0
            )
            stats["disk_hit_rate"] = stats["disk_hits"] / total_disk if total_disk > 0 else 0

            # Add size info
            if self._memory_cache is not None:
                stats["memory_size"] = len(self._memory_cache)
                stats["memory_max_size"] = self.config.memory_max_size

            if self._disk_cache is not None:
                stats["disk_size"] = len(self._disk_cache)
                stats["disk_volume"] = self._disk_cache.volume()

            return stats

    def _make_key(self, key: str, namespace: str) -> str:
        """Create cache key with namespace.

        Args:
            key: Original key
            namespace: Namespace for organization

        Returns:
            Namespaced cache key
        """
        # Hash long keys to keep them manageable
        if len(key) > 200:
            key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
            return f"{namespace}:{key_hash}"

        return f"{namespace}:{key}"

    def warmup(self, data: Dict[str, Any], namespace: str = "default") -> int:
        """Warm up cache with data.

        Args:
            data: Dictionary of key-value pairs to cache
            namespace: Cache namespace

        Returns:
            Number of entries cached
        """
        count = 0
        for key, value in data.items():
            if self.set(key, value, namespace):
                count += 1

        logger.info("Warmed up cache with %d entries", count)
        return count

    def close(self) -> None:
        """Close cache connections and cleanup."""
        if self._disk_cache is not None:
            try:
                self._disk_cache.close()
                logger.info("Disk cache closed")
            except Exception as e:
                logger.warning("Error closing disk cache: %s", e)

    def __enter__(self) -> "CacheManager":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Context manager exit."""
        self.close()


class ResponseCache:
    """Specialized cache for LLM responses.

    Caches expensive LLM API calls to reduce costs and latency.
    """

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """Initialize response cache.

        Args:
            cache_manager: Cache manager instance (creates default if None)
        """
        self.cache = cache_manager or CacheManager()
        self.namespace = "responses"

    def get_response(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        """Get cached response.

        Args:
            prompt: Prompt text
            model: Model identifier
            temperature: Temperature setting

        Returns:
            Cached response or None
        """
        key = self._make_response_key(prompt, model, temperature)
        return self.cache.get(key, self.namespace)

    def cache_response(
        self,
        prompt: str,
        model: str,
        temperature: float,
        response: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache LLM response.

        Args:
            prompt: Prompt text
            model: Model identifier
            temperature: Temperature setting
            response: Response to cache
            ttl: Custom TTL (uses default if None)

        Returns:
            True if successfully cached
        """
        key = self._make_response_key(prompt, model, temperature)
        return self.cache.set(key, response, self.namespace, ttl)

    def _make_response_key(self, prompt: str, model: str, temperature: float) -> str:
        """Create cache key for response.

        Args:
            prompt: Prompt text
            model: Model identifier
            temperature: Temperature setting

        Returns:
            Cache key
        """
        # Include model and temperature in key for uniqueness
        key_data = {
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()


class EmbeddingCache:
    """Specialized cache for embeddings.

    Caches expensive embedding computations for semantic search.
    """

    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """Initialize embedding cache.

        Args:
            cache_manager: Cache manager instance
        """
        self.cache = cache_manager or CacheManager()
        self.namespace = "embeddings"

    def get_embedding(self, text: str, model: str) -> Optional[list]:
        """Get cached embedding.

        Args:
            text: Text to get embedding for
            model: Embedding model name

        Returns:
            Cached embedding vector or None
        """
        key = self._make_embedding_key(text, model)
        return self.cache.get(key, self.namespace)

    def cache_embedding(
        self, text: str, model: str, embedding: list, ttl: Optional[int] = None
    ) -> bool:
        """Cache embedding.

        Args:
            text: Text
            model: Embedding model
            embedding: Embedding vector
            ttl: Custom TTL

        Returns:
            True if successfully cached
        """
        key = self._make_embedding_key(text, model)
        return self.cache.set(key, embedding, self.namespace, ttl)

    def _make_embedding_key(self, text: str, model: str) -> str:
        """Create cache key for embedding.

        Args:
            text: Text
            model: Model name

        Returns:
            Cache key
        """
        key_str = f"{model}:{text}"
        return hashlib.sha256(key_str.encode()).hexdigest()
