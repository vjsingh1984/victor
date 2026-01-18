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

"""Tests for SQLite cache backend implementation.

Tests the SQLiteCacheBackend with in-memory database for fast testing.
"""

import asyncio
import pickle
import sqlite3
from pathlib import Path

import pytest

from victor.agent.cache.backends.sqlite import SQLiteCacheBackend


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def db_path(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test_cache.db"


@pytest.fixture
def sqlite_backend(db_path):
    """Create a SQLite backend with temporary database."""
    backend = SQLiteCacheBackend(
        db_path=str(db_path),
        default_ttl_seconds=3600,
    )
    return backend


@pytest.fixture
async def connected_backend(sqlite_backend):
    """Create a connected SQLite backend."""
    await sqlite_backend.connect()
    yield sqlite_backend
    await sqlite_backend.disconnect()


# =============================================================================
# Connection Lifecycle Tests
# =============================================================================


class TestConnectionLifecycle:
    """Tests for connect/disconnect lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_creates_database(self, db_path):
        """Test that connect() creates database file."""
        backend = SQLiteCacheBackend(db_path=str(db_path))
        await backend.connect()

        assert db_path.exists()

        await backend.disconnect()

    @pytest.mark.asyncio
    async def test_connect_creates_cache_table(self, connected_backend, db_path):
        """Test that connect() creates cache table."""
        # Connect creates the table
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check that table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cache_entries'")
        result = cursor.fetchone()

        assert result is not None
        conn.close()

    @pytest.mark.asyncio
    async def test_connect_creates_indexes(self, connected_backend, db_path):
        """Test that connect() creates performance indexes."""
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check that indexes exist
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='cache_entries'"
        )
        indexes = [row[0] for row in cursor.fetchall()]

        # Should have indexes on key_namespace and expires_at
        assert any("key_namespace" in idx for idx in indexes)
        assert any("expires_at" in idx for idx in indexes)

        conn.close()

    @pytest.mark.asyncio
    async def test_disconnect_closes_connection(self, connected_backend):
        """Test that disconnect() closes database connection."""
        await connected_backend.disconnect()

        assert connected_backend._conn is None
        assert connected_backend._cursor is None

    @pytest.mark.asyncio
    async def test_multiple_connect_calls_idempotent(self, connected_backend, db_path):
        """Test that multiple connect() calls are idempotent."""
        first_conn = connected_backend._conn

        await connected_backend.connect()

        assert connected_backend._conn is first_conn

    @pytest.mark.asyncio
    async def test_multiple_disconnect_calls_safe(self, connected_backend):
        """Test that multiple disconnect() calls are safe."""
        await connected_backend.disconnect()

        # Should not raise
        await connected_backend.disconnect()


# =============================================================================
# Basic Cache Operations Tests
# =============================================================================


class TestBasicCacheOperations:
    """Tests for basic cache operations (get, set, delete)."""

    @pytest.mark.asyncio
    async def test_get_returns_cached_value(self, connected_backend):
        """Test that get() returns cached value."""
        value = {"result": "data"}

        await connected_backend.set("key1", value, "test_namespace")
        result = await connected_backend.get("key1", "test_namespace")

        assert result == value

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing_key(self, connected_backend):
        """Test that get() returns None when key doesn't exist."""
        result = await connected_backend.get("missing_key", "test_namespace")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_stores_value(self, connected_backend):
        """Test that set() stores value in database."""
        value = [1, 2, 3, {"nested": "value"}]

        await connected_backend.set("key1", value, "test_namespace")
        result = await connected_backend.get("key1", "test_namespace")

        assert result == value

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self, connected_backend):
        """Test that set() uses custom TTL when provided."""
        value = "test_value"

        await connected_backend.set("key1", value, "test_namespace", ttl_seconds=1)

        # Should exist immediately
        result = await connected_backend.get("key1", "test_namespace")
        assert result == value

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired
        result = await connected_backend.get("key1", "test_namespace")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_without_ttl_uses_default(self, connected_backend):
        """Test that set() without TTL uses backend default."""
        value = "test_value"

        await connected_backend.set("key1", value, "test_namespace")

        # Should exist with default TTL
        result = await connected_backend.get("key1", "test_namespace")
        assert result == value

    @pytest.mark.asyncio
    async def test_delete_removes_key(self, connected_backend):
        """Test that delete() removes key from database."""
        await connected_backend.set("key1", "value1", "test_namespace")

        deleted = await connected_backend.delete("key1", "test_namespace")
        assert deleted is True

        result = await connected_backend.get("key1", "test_namespace")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_returns_false_for_missing_key(self, connected_backend):
        """Test that delete() returns False when key doesn't exist."""
        deleted = await connected_backend.delete("missing_key", "test_namespace")

        assert deleted is False


# =============================================================================
# Namespace Management Tests
# =============================================================================


class TestNamespaceManagement:
    """Tests for namespace isolation and management."""

    @pytest.mark.asyncio
    async def test_namespace_isolation(self, connected_backend):
        """Test that different namespaces don't interfere."""
        # Set same key in different namespaces
        await connected_backend.set("key1", "value1", "namespace1")
        await connected_backend.set("key1", "value2", "namespace2")

        value1 = await connected_backend.get("key1", "namespace1")
        value2 = await connected_backend.get("key1", "namespace2")

        assert value1 == "value1"
        assert value2 == "value2"

    @pytest.mark.asyncio
    async def test_clear_namespace_deletes_all_keys(self, connected_backend):
        """Test that clear_namespace() deletes all keys in namespace."""
        # Set multiple keys in namespace
        await connected_backend.set("key1", "value1", "test_namespace")
        await connected_backend.set("key2", "value2", "test_namespace")
        await connected_backend.set("key3", "value3", "other_namespace")

        # Clear test_namespace
        count = await connected_backend.clear_namespace("test_namespace")

        assert count == 2

        # Keys in test_namespace should be gone
        assert await connected_backend.get("key1", "test_namespace") is None
        assert await connected_backend.get("key2", "test_namespace") is None

        # Key in other_namespace should remain
        assert await connected_backend.get("key3", "other_namespace") == "value3"

    @pytest.mark.asyncio
    async def test_clear_namespace_returns_zero_for_empty_namespace(self, connected_backend):
        """Test that clear_namespace() returns 0 when namespace is empty."""
        count = await connected_backend.clear_namespace("empty_namespace")

        assert count == 0


# =============================================================================
# TTL and Expiration Tests
# =============================================================================


class TestTTLAndExpiration:
    """Tests for TTL-based expiration."""

    @pytest.mark.asyncio
    async def test_expired_entries_not_returned(self, connected_backend):
        """Test that expired entries are not returned."""
        # Set with short TTL
        await connected_backend.set("key1", "value1", "test_namespace", ttl_seconds=1)

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired
        result = await connected_backend.get("key1", "test_namespace")
        assert result is None

    @pytest.mark.asyncio
    async def test_cleanup_expired_removes_old_entries(self, connected_backend):
        """Test that cleanup_expired() removes expired entries."""
        # Set entries with different TTLs
        await connected_backend.set("key1", "value1", "test_namespace", ttl_seconds=1)
        await connected_backend.set("key2", "value2", "test_namespace", ttl_seconds=3600)

        # Wait for first to expire
        await asyncio.sleep(1.1)

        # Cleanup expired entries
        removed = await connected_backend.cleanup_expired()

        assert removed == 1

        # First key should be gone
        assert await connected_backend.get("key1", "test_namespace") is None

        # Second key should remain
        assert await connected_backend.get("key2", "test_namespace") == "value2"

    @pytest.mark.asyncio
    async def test_cleanup_expired_returns_zero_when_no_expired(self, connected_backend):
        """Test that cleanup_expired() returns 0 when no expired entries."""
        await connected_backend.set("key1", "value1", "test_namespace", ttl_seconds=3600)

        removed = await connected_backend.cleanup_expired()

        assert removed == 0


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for cache statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_returns_backend_type(self, connected_backend):
        """Test that get_stats() includes backend type."""
        stats = await connected_backend.get_stats()

        assert stats["backend_type"] == "sqlite"

    @pytest.mark.asyncio
    async def test_get_stats_returns_key_count(self, connected_backend):
        """Test that get_stats() includes key count."""
        await connected_backend.set("key1", "value1", "test_namespace")
        await connected_backend.set("key2", "value2", "test_namespace")

        stats = await connected_backend.get_stats()

        assert stats["keys"] == 2

    @pytest.mark.asyncio
    async def test_get_stats_returns_db_size(self, connected_backend, db_path):
        """Test that get_stats() includes database file size."""
        await connected_backend.set("key1", "value1", "test_namespace")

        stats = await connected_backend.get_stats()

        assert "db_size_bytes" in stats
        assert stats["db_size_bytes"] > 0


# =============================================================================
# Persistence Tests
# =============================================================================


class TestPersistence:
    """Tests for cache persistence across connections."""

    @pytest.mark.asyncio
    async def test_cache_persists_across_reconnects(self, db_path):
        """Test that cache data persists across disconnect/reconnect."""
        # Create backend and set value
        backend1 = SQLiteCacheBackend(db_path=str(db_path))
        await backend1.connect()
        await backend1.set("key1", "value1", "test_namespace")
        await backend1.disconnect()

        # Create new backend connection
        backend2 = SQLiteCacheBackend(db_path=str(db_path))
        await backend2.connect()

        # Value should still be there
        result = await backend2.get("key1", "test_namespace")
        assert result == "value1"

        await backend2.disconnect()

    @pytest.mark.asyncio
    async def test_expired_entries_cleanup_on_reconnect(self, db_path):
        """Test that expired entries are cleaned up on reconnect."""
        # Create backend and set value with short TTL
        backend1 = SQLiteCacheBackend(db_path=str(db_path))
        await backend1.connect()
        await backend1.set("key1", "value1", "test_namespace", ttl_seconds=1)
        await backend1.disconnect()

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Reconnect (should cleanup expired)
        backend2 = SQLiteCacheBackend(db_path=str(db_path))
        await backend2.connect()

        # Value should be expired
        result = await backend2.get("key1", "test_namespace")
        assert result is None

        await backend2.disconnect()


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_get_operations(self, connected_backend):
        """Test that concurrent get operations work correctly."""
        await connected_backend.set("key1", "value1", "test_namespace")

        # Run concurrent gets
        tasks = [connected_backend.get("key1", "test_namespace") for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should return the same value
        assert all(r == "value1" for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_set_operations(self, connected_backend):
        """Test that concurrent set operations work correctly."""
        # Run concurrent sets
        tasks = [connected_backend.set(f"key{i}", f"value{i}", "test_namespace") for i in range(10)]
        await asyncio.gather(*tasks)

        # All values should be stored
        for i in range(10):
            result = await connected_backend.get(f"key{i}", "test_namespace")
            assert result == f"value{i}"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_get_before_connect_raises_error(self, sqlite_backend):
        """Test that get() before connect() raises error."""
        with pytest.raises(RuntimeError, match="Not connected"):
            await sqlite_backend.get("key1", "test_namespace")

    @pytest.mark.asyncio
    async def test_set_before_connect_raises_error(self, sqlite_backend):
        """Test that set() before connect() raises error."""
        with pytest.raises(RuntimeError, match="Not connected"):
            await sqlite_backend.set("key1", "value1", "test_namespace")

    @pytest.mark.asyncio
    async def test_unpickleable_value_raises_error(self, connected_backend):
        """Test that unpickleable value raises appropriate error."""
        # Lambda functions are not pickle-able
        def _unpickleable(x):
            return x

        # pickle.dumps can raise AttributeError, PicklingError, or TypeError
        # depending on the object and Python version
        with pytest.raises((pickle.PicklingError, TypeError, AttributeError)):
            await connected_backend.set("key1", _unpickleable, "test_namespace")


# =============================================================================
# Distributed Invalidation Tests
# =============================================================================


class TestDistributedInvalidation:
    """Tests for distributed invalidation (not supported by SQLite)."""

    @pytest.mark.asyncio
    async def test_invalidate_publish_raises_not_implemented(self, connected_backend):
        """Test that invalidate_publish() raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await connected_backend.invalidate_publish("key1", "test_namespace")

    @pytest.mark.asyncio
    async def test_listen_for_invalidation_raises_not_implemented(self, connected_backend):
        """Test that listen_for_invalidation() raises NotImplementedError."""

        async def callback(key: str, namespace: str) -> None:
            pass

        with pytest.raises(NotImplementedError):
            await connected_backend.listen_for_invalidation(callback)
