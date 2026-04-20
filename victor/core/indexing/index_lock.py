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

Provides two levels of locking:
1. In-process locking (asyncio.Lock) - prevents concurrent indexing in same process
2. Cross-process locking (file-based) - prevents concurrent indexing across processes

Pattern: Singleton + Per-Resource Locking + Double-Checked Locking + File Locking
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["IndexLockRegistry", "FileLock"]


class FileLock:
    """Cross-process file-based lock using fcntl.

    Prevents multiple processes from indexing the same path concurrently.
    Uses exclusive file locking with automatic cleanup on process exit.

    Pattern: RAII (Resource Acquisition Is Initialization)

    Example:
        >>> lock = FileLock(Path("/my/project/.victor/index.lock"))
        >>> lock.acquire()
        >>> try:
        ...     # Exclusive access to index this path
        ...     await build_index()
        ... finally:
        ...     lock.release()
    """

    def __init__(self, lock_file: Path):
        """Initialize file lock.

        Args:
            lock_file: Path to lock file (will be created if needed)
        """
        self.lock_file = lock_file
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock_fd: Optional[int] = None

    def acquire(self, timeout: float = 300.0) -> bool:
        """Acquire exclusive lock on file.

        Args:
            timeout: Maximum time to wait for lock (default: 5 minutes)

        Returns:
            True if lock acquired, False if timeout

        Raises:
            IOError: If lock cannot be acquired due to system errors
        """
        start_time = time.time()

        while True:
            try:
                # Open file (create if needed)
                self._lock_fd = os.open(
                    self.lock_file,
                    os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
                    0o644,
                )

                # Try to acquire exclusive lock (non-blocking)
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

                # Write PID to lock file for debugging
                os.write(self._lock_fd, str(os.getpid()).encode())

                logger.debug(
                    f"[FileLock] Acquired lock {self.lock_file} (PID: {os.getpid()})"
                )
                return True

            except IOError as e:
                # Lock is held by another process
                if e.errno == errno.EWOULDBLOCK:
                    # Check timeout
                    if time.time() - start_time >= timeout:
                        logger.warning(
                            f"[FileLock] Timeout waiting for lock {self.lock_file}"
                        )
                        if self._lock_fd is not None:
                            os.close(self._lock_fd)
                            self._lock_fd = None
                        return False

                    # Wait a bit and retry
                    time.sleep(0.1)
                    continue

                # Other error
                if self._lock_fd is not None:
                    os.close(self._lock_fd)
                    self._lock_fd = None
                raise IOError(f"Failed to acquire lock: {e}")

    def release(self) -> None:
        """Release the lock and close file descriptor."""
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
                logger.debug(f"[FileLock] Released lock {self.lock_file}")
            except Exception as e:
                logger.warning(f"[FileLock] Error releasing lock: {e}")
            finally:
                self._lock_fd = None

                # Try to remove lock file
                try:
                    if self.lock_file.exists():
                        self.lock_file.unlink()
                except Exception:
                    pass  # Lock file may be in use by another process

    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

    def __del__(self):
        """Destructor - ensure lock is released."""
        self.release()


# Import errno for EWOULDBLOCK
import errno


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
        self._file_locks: Dict[str, FileLock] = {}
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

    async def acquire_lock(self, path: Path, use_file_lock: bool = True) -> asyncio.Lock:
        """Acquire or create lock for specific path.

        Uses double-checked locking pattern for performance:
        - Fast path: Check cache without registry lock
        - Slow path: Acquire registry lock and create if needed
        - Cross-process: Also acquires file lock for multi-process safety

        Args:
            path: File system path to lock
            use_file_lock: Whether to use file-based locking for cross-process safety

        Returns:
            asyncio.Lock for this specific path

        Raises:
            TimeoutError: If file lock cannot be acquired within timeout
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

            # Create file lock for cross-process protection
            if use_file_lock:
                from victor.config.settings import get_project_paths

                project_paths = get_project_paths(path)
                lock_file = project_paths.project_victor_dir / "index.lock"

                file_lock = FileLock(lock_file)
                # Acquire file lock in thread pool to avoid blocking event loop
                loop = asyncio.get_event_loop()
                acquired = await loop.run_in_executor(
                    None, lambda: file_lock.acquire(timeout=300.0)
                )

                if not acquired:
                    raise TimeoutError(
                        f"Failed to acquire index lock for {path_str} "
                        f"after 300 seconds (another process may be indexing)"
                    )

                self._file_locks[path_str] = file_lock
                logger.info(
                    f"[IndexLockRegistry] Acquired cross-process lock for {path_str}"
                )

            # Create new in-process lock
            self._path_locks[path_str] = asyncio.Lock()

            # Initialize stats
            self._lock_stats[path_str] = {
                "created_at": time.time(),
                "cache_hits": 0,
                "waits": 0,
                "total_wait_time_ms": 0,
                "has_file_lock": use_file_lock,
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

        Also releases file locks for cleaned up paths.

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
                if current_time - stats.get("last_used", stats["created_at"])
                > max_idle_seconds
            ]

            # Remove idle locks
            for path_str in idle_paths:
                # Release file lock if present
                if path_str in self._file_locks:
                    self._file_locks[path_str].release()
                    del self._file_locks[path_str]

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
            # Release all file locks
            for file_lock in self._file_locks.values():
                try:
                    file_lock.release()
                except Exception as e:
                    logger.warning(f"[IndexLockRegistry] Error releasing file lock: {e}")

            self._file_locks.clear()
            self._path_locks.clear()
            self._lock_stats.clear()
            logger.warning("[IndexLockRegistry] All locks cleared")
