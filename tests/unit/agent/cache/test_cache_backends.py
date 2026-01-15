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

"""Tests for cache backend implementations.

Tests cache storage backends implementing ICacheBackend protocol.
"""

import pytest

from victor.protocols import ICacheBackend, CacheNamespace


class TestMemoryCacheBackend:
    """Tests for MemoryCacheBackend implementation."""

    @pytest.mark.asyncio
    async def test_get_set_delete(self, backend):
        """Test basic get, set, delete operations."""
        # Set a value
        await backend.set("key1", "value1", "test_namespace")

        # Get it back
        value = await backend.get("key1", "test_namespace")
        assert value == "value1"

        # Delete it
        deleted = await backend.delete("key1", "test_namespace")
        assert deleted is True

        # Should be gone
        value = await backend.get("key1", "test_namespace")
        assert value is None

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, backend):
        """Test getting a key that doesn't exist returns None."""
        value = await backend.get("nonexistent", "test_namespace")
        assert value is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, backend):
        """Test deleting a nonexistent key returns False."""
        deleted = await backend.delete("nonexistent", "test_namespace")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, backend):
        """Test setting value with TTL expires after TTL seconds."""
        import asyncio

        # Set with short TTL
        await backend.set("key1", "value1", "test_namespace", ttl_seconds=1)

        # Should exist immediately
        value = await backend.get("key1", "test_namespace")
        assert value == "value1"

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired
        value = await backend.get("key1", "test_namespace")
        assert value is None

    @pytest.mark.asyncio
    async def test_set_without_ttl(self, backend):
        """Test setting value without TTL uses backend default."""
        # Set without TTL
        await backend.set("key1", "value1", "test_namespace")

        # Should exist
        value = await backend.get("key1", "test_namespace")
        assert value == "value1"

    @pytest.mark.asyncio
    async def test_namespace_isolation(self, backend):
        """Test that different namespaces don't interfere."""
        # Set same key in different namespaces
        await backend.set("key1", "value1", "namespace1")
        await backend.set("key1", "value2", "namespace2")

        # Should get different values
        value1 = await backend.get("key1", "namespace1")
        value2 = await backend.get("key1", "namespace2")

        assert value1 == "value1"
        assert value2 == "value2"

    @pytest.mark.asyncio
    async def test_clear_namespace(self, backend):
        """Test clearing a namespace removes all keys in that namespace."""
        # Set multiple keys in namespace
        await backend.set("key1", "value1", "test_namespace")
        await backend.set("key2", "value2", "test_namespace")
        await backend.set("key3", "value3", "other_namespace")

        # Clear test_namespace
        count = await backend.clear_namespace("test_namespace")
        assert count == 2

        # Keys in test_namespace should be gone
        assert await backend.get("key1", "test_namespace") is None
        assert await backend.get("key2", "test_namespace") is None

        # Key in other_namespace should remain
        assert await backend.get("key3", "other_namespace") == "value3"

    @pytest.mark.asyncio
    async def test_get_stats(self, backend):
        """Test getting cache statistics."""
        # Set some values
        await backend.set("key1", "value1", "test_namespace")
        await backend.set("key2", "value2", "test_namespace")

        # Get stats
        stats = await backend.get_stats()

        # Should have backend_type
        assert "backend_type" in stats
        assert stats["backend_type"] == "memory"

        # Should have keys count
        assert "keys" in stats
        assert stats["keys"] >= 2

    @pytest.mark.asyncio
    async def test_overwrite_existing_key(self, backend):
        """Test overwriting an existing key."""
        # Set initial value
        await backend.set("key1", "value1", "test_namespace")
        value = await backend.get("key1", "test_namespace")
        assert value == "value1"

        # Overwrite
        await backend.set("key1", "value2", "test_namespace")
        value = await backend.get("key1", "test_namespace")
        assert value == "value2"

    @pytest.mark.asyncio
    async def test_complex_values(self, backend):
        """Test caching complex (pickle-able) values."""
        import datetime

        # Dict
        value = {"a": 1, "b": [2, 3, 4]}
        await backend.set("dict_key", value, "test_namespace")
        assert await backend.get("dict_key", "test_namespace") == value

        # List
        value = [1, 2, 3, {"nested": "value"}]
        await backend.set("list_key", value, "test_namespace")
        assert await backend.get("list_key", "test_namespace") == value

        # Datetime
        value = datetime.datetime.now()
        await backend.set("dt_key", value, "test_namespace")
        retrieved = await backend.get("dt_key", "test_namespace")
        assert retrieved == value

    @pytest.mark.asyncio
    async def test_cache_namespace_hierarchy(self, backend):
        """Test cache namespace hierarchy (if supported)."""
        # Set in SESSION namespace
        await backend.set("session_key", "session_value", CacheNamespace.SESSION.value)

        # Set in REQUEST namespace
        await backend.set("request_key", "request_value", CacheNamespace.REQUEST.value)

        # Clear SESSION should clear REQUEST too (if hierarchy supported)
        # This is optional behavior - not all backends need to support it
        count = await backend.clear_namespace(CacheNamespace.SESSION.value)

        # SESSION key should be gone
        assert await backend.get("session_key", CacheNamespace.SESSION.value) is None

        # REQUEST key handling depends on backend implementation
        # For memory backend without hierarchy, REQUEST might remain


class TestCacheBackendProtocol:
    """Tests for ICacheBackend protocol compliance."""

    def test_memory_backend_implements_protocol(self):
        """Verify MemoryCacheBackend implements ICacheBackend."""
        from victor.agent.cache.backends.memory import MemoryCacheBackend

        backend = MemoryCacheBackend()
        assert isinstance(backend, ICacheBackend)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def backend():
    """Provide a cache backend instance for testing."""
    from victor.agent.cache.backends.memory import MemoryCacheBackend

    return MemoryCacheBackend()
