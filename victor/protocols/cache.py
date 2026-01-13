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

"""Cache backend protocol for dependency inversion.

This module defines the ICacheBackend protocol that enables
dependency injection for cache storage, following the
Dependency Inversion Principle (DIP).

Design Principles:
    - DIP: CacheManager depends on this protocol, not concrete backends
    - OCP: New cache backends (Redis, Memcached) can be added without modification
    - Strategy Pattern: Different backends for different use cases

Supported Backends:
    - Memory: In-memory caching (default)
    - Redis: Distributed caching
    - SQLite: Persistent caching
    - Custom: User-provided implementations

Usage:
    class RedisCacheBackend(ICacheBackend):
        async def get(self, key: str, namespace: str) -> Optional[Any]:
            return await self._redis.get(f"{namespace}:{key}")

        async def set(self, key: str, value: Any, namespace: str, ttl_seconds: Optional[int]) -> None:
            await self._redis.setex(f"{namespace}:{key}", ttl_seconds or 3600, pickle.dumps(value))

        async def delete(self, key: str, namespace: str) -> bool:
            return await self._redis.delete(f"{namespace}:{key}") > 0

        async def clear_namespace(self, namespace: str) -> int:
            pattern = f"{namespace}:*"
            keys = await self._redis.keys(pattern)
            if keys:
                return await self._redis.delete(*keys)
            return 0

        async def get_stats(self) -> Dict[str, Any]:
            info = await self._redis.info()
            return {"backend_type": "redis", "keys": info.get("db0", {}).get("keys", 0)}
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class ICacheBackend(Protocol):
    """Protocol for cache storage backends.

    Implementations provide cache storage with namespace isolation,
    TTL support, and statistics tracking. The protocol enables
    pluggable backends for different deployment scenarios:

    - Memory: Fast, local caching (default)
    - Redis: Distributed caching for multi-process deployments
    - SQLite: Persistent caching across restarts
    - Custom: User-provided implementations

    Namespace isolation prevents cache key collisions between
    different components (tools, sessions, global, etc.).
    """

    async def get(self, key: str, namespace: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key
            namespace: Namespace for isolation (e.g., 'tool', 'session', 'global')

        Returns:
            Cached value or None if not found

        Example:
            value = await cache.get("result_123", "tool")
        """
        ...

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
            value: Value to cache (must be pickle-able)
            namespace: Namespace for isolation
            ttl_seconds: Time-to-live in seconds (None = default TTL)

        Example:
            await cache.set("result_123", tool_result, "tool", ttl_seconds=300)
        """
        ...

    async def delete(self, key: str, namespace: str) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key to delete
            namespace: Namespace of the key

        Returns:
            True if key was deleted, False if not found

        Example:
            deleted = await cache.delete("result_123", "tool")
        """
        ...

    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace.

        Args:
            namespace: Namespace to clear

        Returns:
            Number of keys deleted

        Example:
            count = await cache.clear_namespace("tool")  # Clear all tool caches
        """
        ...

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with backend-specific statistics

        Common keys:
            - backend_type: Type of backend ('memory', 'redis', 'sqlite')
            - keys: Number of cached items
            - memory_used: Memory usage (if applicable)
            - hit_rate: Cache hit rate (if tracked)

        Example:
            stats = await cache.get_stats()
            # {'backend_type': 'redis', 'keys': 1234, 'memory_used': '45.2MB'}
        """
        ...


@runtime_checkable
class IIdempotentTool(Protocol):
    """Protocol for tools that support idempotency tracking.

    Idempotent tools can be safely cached and retried without
    side effects. Implementations provide idempotency keys
    to identify duplicate calls.

    Examples of idempotent tools:
    - read_file: Reading the same file returns the same result
    - grep: Searching for the same pattern returns the same result
    - code_search: Semantic search with the same query returns same results

    Non-idempotent tools (should not use result caching):
    - write_file: Each call modifies state
    - shell_execute: Each execution may have different results
    - git_push: Each push changes remote state
    """

    def is_idempotent(self) -> bool:
        """Check if tool is idempotent.

        Returns:
            True if tool is idempotent, False otherwise

        Example:
            def is_idempotent(self) -> bool:
                # File reading is idempotent
                return True
        """
        ...

    def get_idempotency_key(self, **kwargs) -> str:
        """Generate key for idempotency tracking.

        The key uniquely identifies a tool call such that
        calls with the same key should return the same result.

        Args:
            **kwargs: Tool arguments

        Returns:
            String key for idempotency tracking

        Example:
            def get_idempotency_key(self, **kwargs) -> str:
                # For read_file, key is based on file path
                return f"read_file:{kwargs['path']}"

            def get_idempotency_key(self, **kwargs) -> str:
                # For grep, key is based on pattern and path
                return f"grep:{kwargs['pattern']}:{kwargs['path']}"
        """
        ...


__all__ = ["ICacheBackend", "IIdempotentTool"]
