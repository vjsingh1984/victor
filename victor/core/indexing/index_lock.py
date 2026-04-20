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

"""Index-level locking to prevent concurrent indexing of same path.

This module provides per-path async locking to prevent multiple concurrent
indexing operations on the same codebase path. Uses singleton pattern with
double-checked locking for thread-safe access.

Pattern: Singleton + Per-Resource Locking + Double-Checked Locking
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["IndexLockRegistry"]


class IndexLockRegistry:
    """Global registry of path-specific async locks.

    Prevents multiple concurrent indexing operations on the same path
    by providing a unique asyncio.Lock for each unique codebase path.

    Uses singleton pattern with double-checked locking:
    1. Check cache without lock (fast path)
    2. Acquire registry lock
    3. Double-check and create if needed
    4. Return path-specific lock

    Example:
        >>> registry = IndexLockRegistry.get_instance()
        >>> path_lock = await registry.acquire_lock(Path("/my/project"))
        >>> async with path_lock:
        ...     # Exclusive access to index this path
        ...     await build_index()

    Thread Safety:
        Fully thread-safe with asyncio.Lock for registry access
        and per-path asyncio.Lock instances for resource protection.
    """

    _instance: Optional["IndexLockRegistry"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self):
        """Initialize IndexLockRegistry."""
        self._path_locks: Dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()
        self._lock_stats: Dict[str, Dict] = {}  # Track lock usage stats

    @classmethod
    def get_instance(cls) -> "IndexLockRegistry":
        """Get singleton instance.

        Returns:
            IndexLockRegistry singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def acquire_lock(self, path: Path) -> asyncio.Lock:
        """Acquire or create lock for specific path.

        Uses double-checked locking pattern for performance:
        - Fast path: Check cache without registry lock
        - Slow path: Acquire registry lock and create if needed

        Args:
            path: File system path to lock

        Returns:
            asyncio.Lock for this specific path
        """
        path_str = str(path.resolve())

        # Fast path - check cache without lock
        if path_str in self._path_locks:
            # Update stats
            if path_str in self._lock_stats:
                self._lock_stats[path_str]["cache_hits"] += 1
            return self._path_locks[path_str]

        # Slow path - acquire registry lock
        start_time = time.time()
        async with self._registry_lock:
            # Double-check inside lock
            if path_str in self._path_locks:
                # Another task created it while we waited
                return self._path_locks[path_str]

            # Create new lock
            self._path_locks[path_str] = asyncio.Lock()

            # Initialize stats
            self._lock_stats[path_str] = {
                "created_at": time.time(),
                "cache_hits": 0,
                "waits": 0,
                "total_wait_time_ms": 0,
            }

            logger.info(
                f"[IndexLockRegistry] Created lock for {path_str} "
                f"(total locks: {len(self._path_locks)})"
            )

            # Track wait time
            wait_time_ms = (time.time() - start_time) * 1000
            self._lock_stats[path_str]["waits"] += 1
            self._lock_stats[path_str]["total_wait_time_ms"] += wait_time_ms

            return self._path_locks[path_str]

    async def cleanup_idle_locks(
        self,
        max_idle_seconds: int = 3600,
    ) -> int:
        """Remove locks that haven't been used recently.

        Args:
            max_idle_seconds: Remove locks idle longer than this (default: 1 hour)

        Returns:
            Number of locks removed
        """
        if max_idle_seconds <= 0:
            raise ValueError("max_idle_seconds must be positive")

        removed = 0
        current_time = time.time()

        async with self._registry_lock:
            # Find idle locks
            idle_paths = [
                path_str
                for path_str, stats in self._lock_stats.items()
                if current_time - stats.get("last_used", stats["created_at"]) > max_idle_seconds
            ]

            # Remove idle locks
            for path_str in idle_paths:
                del self._path_locks[path_str]
                del self._lock_stats[path_str]
                removed += 1

            if removed > 0:
                logger.info(
                    f"[IndexLockRegistry] Cleaned up {removed} idle locks "
                    f"(remaining: {len(self._path_locks)})"
                )

        return removed

    def mark_lock_used(self, path: Path) -> None:
        """Mark lock as recently used.

        Call this after successfully using a lock to track activity.

        Args:
            path: Path whose lock was used
        """
        path_str = str(path.resolve())

        if path_str in self._lock_stats:
            self._lock_stats[path_str]["last_used"] = time.time()

    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics about lock usage.

        Returns:
            Dict mapping path strings to lock statistics
        """
        return {
            "total_locks": len(self._path_locks),
            "lock_details": dict(self._lock_stats),
        }

    async def reset(self) -> None:
        """Reset all locks (mainly for testing).

        Warning: This will release all locks and clear the cache.
        Use with caution in production.
        """
        async with self._registry_lock:
            self._path_locks.clear()
            self._lock_stats.clear()
            logger.warning("[IndexLockRegistry] All locks cleared")
