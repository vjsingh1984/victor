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

"""SQLite cache backend implementation with persistent storage.

This module provides SQLiteCacheBackend, a persistent cache backend
implementing ICacheBackend protocol with SQLite for disk-based storage.

Features:
    - Persistent storage across process restarts
    - Thread-safe access (SQLite connections are thread-safe)
    - Automatic schema management
    - TTL support with cleanup
    - Namespace isolation
    - Statistics tracking

Use Cases:
    - Caching across process restarts
    - Long-term storage of computed results
    - Single-process deployments
    - Environments without Redis

Limitations:
    - Not distributed (single-process only)
    - Slower than in-memory (disk I/O)
    - No pub/sub support
    - Filesystem dependencies

Example:
    ```python
    backend = SQLiteCacheBackend(
        db_path="/var/cache/victor/cache.db",
        default_ttl_seconds=3600,
        cleanup_interval_seconds=300,
    )

    # Connect to database
    await backend.connect()

    # Use cache
    await backend.set("result_123", computation_result, "tool", ttl_seconds=300)
    value = await backend.get("result_123", "tool")

    # Cleanup expired entries
    removed = await backend.cleanup_expired()

    # Get statistics
    stats = await backend.get_stats()

    # Cleanup
    await backend.disconnect()
    ```
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from victor.agent.cache.backends.protocol import ICacheBackend
from victor.core.security import safe_pickle_dumps, safe_pickle_loads


logger = logging.getLogger(__name__)


class SQLiteCacheBackend(ICacheBackend):
    """SQLite-based persistent cache backend.

    This backend provides persistent caching with SQLite and supports
    cache persistence across process restarts.

    Features:
        - Persistent storage (survives process restarts)
        - Thread-safe operations
        - Automatic schema management
        - TTL support with cleanup task
        - Namespace isolation
        - Statistics tracking

    Configuration:
        db_path: Path to SQLite database file (default: ":memory:")
        default_ttl_seconds: Default TTL for cache entries (default: 3600)
        cleanup_interval_seconds: Interval for auto-cleanup (default: 300)
            Set to 0 to disable auto-cleanup
        enable_wal: Enable Write-Ahead Logging for better concurrency (default: True)

    Example:
        backend = SQLiteCacheBackend(
            db_path="/var/cache/victor/cache.db",
            default_ttl_seconds=1800,
        )

        await backend.connect()

        # Cache a value
        await backend.set("result_123", computation_result, "tool", ttl_seconds=300)

        # Retrieve it
        value = await backend.get("result_123", "tool")

        # Cleanup expired entries
        await backend.cleanup_expired()

        await backend.disconnect()
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        default_ttl_seconds: int = 3600,
        cleanup_interval_seconds: int = 300,
        enable_wal: bool = True,
    ):
        """Initialize the SQLite cache backend.

        Args:
            db_path: Path to SQLite database file
                Use ":memory:" for in-memory database (default)
                Use a file path for persistent storage
            default_ttl_seconds: Default TTL for entries (default: 3600)
            cleanup_interval_seconds: Auto-cleanup interval in seconds (default: 300)
                Set to 0 to disable automatic cleanup
            enable_wal: Enable Write-Ahead Logging (default: True)
                WAL improves concurrent read performance
        """
        self._db_path = db_path
        self._default_ttl = default_ttl_seconds
        self._cleanup_interval = cleanup_interval_seconds
        self._enable_wal = enable_wal

        # Connection objects (set in connect())
        self._conn: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None

        # Thread safety
        self._lock = threading.RLock()

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task[None]] = None

        # Connection state
        self._is_connected = False

    async def connect(self) -> None:
        """Establish connection to SQLite database.

        This method creates the database file and schema if they don't exist.
        It should be called before any cache operations.

        Raises:
            sqlite3.Error: If database connection fails
        """
        if self._is_connected:
            # Already connected
            return

        try:
            with self._lock:
                # Create connection
                self._conn = sqlite3.connect(
                    self._db_path,
                    check_same_thread=False,  # We handle thread safety with lock
                )
                self._cursor = self._conn.cursor()

                # Enable WAL mode for better concurrency
                if self._enable_wal:
                    self._cursor.execute("PRAGMA journal_mode=WAL")

                # Create schema
                self._create_schema()

                # Mark as connected before cleanup
                self._is_connected = True

                # Cleanup expired entries on startup
                await self.cleanup_expired()

                # Start cleanup task if enabled
                if self._cleanup_interval > 0:
                    self._cleanup_task = asyncio.create_task(self._cleanup_loop())

                logger.info(f"Connected to SQLite database at {self._db_path}")

        except sqlite3.Error as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            raise

    async def disconnect(self) -> None:
        """Close SQLite connection and release resources.

        This method stops the cleanup task and closes the database connection.
        It should be called when shutting down.
        """
        if not self._is_connected:
            return

        try:
            # Stop cleanup task
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            with self._lock:
                if self._cursor:
                    self._cursor.close()
                    self._cursor = None

                if self._conn:
                    self._conn.close()
                    self._conn = None

            self._is_connected = False
            logger.info("Disconnected from SQLite database")

        except Exception as e:
            logger.error(f"Error disconnecting from SQLite: {e}")
            # Don't raise - cleanup should be best-effort

    def _create_schema(self) -> None:
        """Create database schema if it doesn't exist.

        This creates the cache_entries table and indexes.
        Must be called with lock held.
        """
        if self._cursor is None:
            raise RuntimeError("Cursor not initialized")

        # Create cache entries table
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                namespace TEXT NOT NULL,
                value BLOB NOT NULL,
                expires_at REAL,
                created_at REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL,
                UNIQUE(key, namespace)
            )
        """
        )

        # Create index on key+namespace for lookups
        self._cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_key_namespace
            ON cache_entries(key, namespace)
        """
        )

        # Create index on expires_at for cleanup
        self._cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_expires_at
            ON cache_entries(expires_at)
        """
        )

        if self._conn:
            self._conn.commit()

    async def get(self, key: str, namespace: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key
            namespace: Namespace for isolation

        Returns:
            Cached value or None if not found or expired

        Raises:
            RuntimeError: If not connected to database
        """
        if not self._is_connected or self._cursor is None:
            raise RuntimeError("Not connected to SQLite. Call connect() first.")

        try:
            with self._lock:
                # Query database
                self._cursor.execute(
                    """
                    SELECT value, expires_at FROM cache_entries
                    WHERE key = ? AND namespace = ?
                """,
                    (key, namespace),
                )

                row = self._cursor.fetchone()

                if row is None:
                    return None

                value_blob, expires_at = row

                # Check expiration
                if expires_at is not None and time.time() > expires_at:
                    # Expired - delete it
                    self._cursor.execute(
                        "DELETE FROM cache_entries WHERE key = ? AND namespace = ?",
                        (key, namespace),
                    )
                    if self._conn:
                        self._conn.commit()
                    return None

                # Update access stats
                self._cursor.execute(
                    """
                    UPDATE cache_entries
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE key = ? AND namespace = ?
                """,
                    (time.time(), key, namespace),
                )
                if self._conn:
                    self._conn.commit()

                # Deserialize value (with HMAC signature verification)
                value = safe_pickle_loads(value_blob)
                return value

        except sqlite3.Error as e:
            logger.error(f"Error getting key {namespace}:{key}: {e}")
            return None

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
            ttl_seconds: Time-to-live in seconds (None = use default)

        Raises:
            RuntimeError: If not connected to database
            TypeError: If value is not pickle-able
        """
        if not self._is_connected or self._cursor is None:
            raise RuntimeError("Not connected to SQLite. Call connect() first.")

        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expires_at = time.time() + ttl if ttl > 0 else None

        try:
            with self._lock:
                # Serialize value (with HMAC signature)
                value_blob = safe_pickle_dumps(value)

                # Insert or replace (UPSERT)
                self._cursor.execute(
                    """
                    INSERT INTO cache_entries
                    (key, namespace, value, expires_at, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(key, namespace) DO UPDATE SET
                        value = excluded.value,
                        expires_at = excluded.expires_at,
                        created_at = excluded.created_at
                """,
                    (key, namespace, value_blob, expires_at, time.time()),
                )

                if self._conn:
                    self._conn.commit()

                logger.debug(f"Set key {namespace}:{key} with TTL {ttl}s")

        except (pickle.PicklingError, TypeError) as e:
            logger.error(f"Error serializing value for {namespace}:{key}: {e}")
            raise
        except sqlite3.Error as e:
            logger.error(f"Error setting key {namespace}:{key}: {e}")
            raise

    async def delete(self, key: str, namespace: str) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key to delete
            namespace: Namespace of the key

        Returns:
            True if key was deleted, False if not found

        Raises:
            RuntimeError: If not connected to database
        """
        if not self._is_connected or self._cursor is None:
            raise RuntimeError("Not connected to SQLite. Call connect() first.")

        try:
            with self._lock:
                self._cursor.execute(
                    "DELETE FROM cache_entries WHERE key = ? AND namespace = ?",
                    (key, namespace),
                )

                if self._conn:
                    self._conn.commit()

                deleted = self._cursor.rowcount > 0
                return deleted

        except sqlite3.Error as e:
            logger.error(f"Error deleting key {namespace}:{key}: {e}")
            return False

    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace.

        Args:
            namespace: Namespace to clear

        Returns:
            Number of keys deleted

        Raises:
            RuntimeError: If not connected to database
        """
        if not self._is_connected or self._cursor is None:
            raise RuntimeError("Not connected to SQLite. Call connect() first.")

        try:
            with self._lock:
                self._cursor.execute("DELETE FROM cache_entries WHERE namespace = ?", (namespace,))

                if self._conn:
                    self._conn.commit()

                count = self._cursor.rowcount
                logger.info(f"Cleared {count} keys in namespace {namespace}")
                return count

        except sqlite3.Error as e:
            logger.error(f"Error clearing namespace {namespace}: {e}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with backend statistics

        Raises:
            RuntimeError: If not connected to database
        """
        if not self._is_connected or self._cursor is None:
            raise RuntimeError("Not connected to SQLite. Call connect() first.")

        try:
            with self._lock:
                # Get total keys
                self._cursor.execute("SELECT COUNT(*) FROM cache_entries")
                total_keys = self._cursor.fetchone()[0]

                # Get database size (if file-based)
                db_size_bytes = 0
                if self._db_path != ":memory:":
                    db_size_bytes = Path(self._db_path).stat().st_size

                return {
                    "backend_type": "sqlite",
                    "keys": total_keys,
                    "db_size_bytes": db_size_bytes,
                    "db_path": self._db_path,
                }

        except sqlite3.Error as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "backend_type": "sqlite",
                "keys": 0,
                "db_size_bytes": 0,
                "error": str(e),
            }

    async def cleanup_expired(self) -> int:
        """Clean up expired entries.

        This removes all entries that have expired based on their TTL.
        It's called automatically on startup and periodically if
        cleanup_interval_seconds is configured.

        Returns:
            Number of entries removed

        Raises:
            RuntimeError: If not connected to database
        """
        if not self._is_connected or self._cursor is None:
            raise RuntimeError("Not connected to SQLite. Call connect() first.")

        try:
            with self._lock:
                now = time.time()

                self._cursor.execute(
                    "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (now,),
                )

                if self._conn:
                    self._conn.commit()

                removed = self._cursor.rowcount
                if removed > 0:
                    logger.info(f"Cleaned up {removed} expired entries")

                return removed

        except sqlite3.Error as e:
            logger.error(f"Error cleaning up expired entries: {e}")
            return 0

    async def _cleanup_loop(self) -> None:
        """Background task to periodically cleanup expired entries.

        This runs indefinitely until cancelled.
        """
        try:
            while True:
                await asyncio.sleep(self._cleanup_interval)
                await self.cleanup_expired()

        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
            raise

    # -------------------------------------------------------------------------
    # Distributed invalidation (not supported)
    # -------------------------------------------------------------------------

    async def invalidate_publish(self, key: str, namespace: str) -> None:
        """Publish cache invalidation event (not supported).

        SQLite backend doesn't support distributed invalidation.
        This method raises NotImplementedError.

        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError(
            "SQLite backend doesn't support distributed invalidation. "
            "Use Redis backend for distributed caching."
        )

    async def listen_for_invalidation(
        self,
        callback: Callable[[str, str], Awaitable[None]],
    ) -> None:
        """Listen for cache invalidation events (not supported).

        SQLite backend doesn't support distributed invalidation.
        This method raises NotImplementedError.

        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError(
            "SQLite backend doesn't support distributed invalidation. "
            "Use Redis backend for distributed caching."
        )


__all__ = [
    "SQLiteCacheBackend",
]
