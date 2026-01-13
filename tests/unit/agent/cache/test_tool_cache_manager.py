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

"""Tests for ToolCacheManager.

Tests the hierarchical tool cache with dependency-aware invalidation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from victor.agent.cache.tool_cache_manager import (
    ToolCacheManager,
    CacheNamespace,
)
from victor.protocols import ICacheBackend


class MockCacheBackend(ICacheBackend):
    """Mock cache backend for testing."""

    def __init__(self):
        self._store = {}
        self._delete_calls = []
        self._clear_calls = []

    async def get(self, key: str, namespace: str):
        return self._store.get(f"{namespace}:{key}")

    async def set(self, key: str, value, namespace: str, ttl_seconds=None):
        self._store[f"{namespace}:{key}"] = value

    async def delete(self, key: str, namespace: str) -> bool:
        self._delete_calls.append((key, namespace))
        full_key = f"{namespace}:{key}"
        if full_key in self._store:
            del self._store[full_key]
            return True
        return False

    async def clear_namespace(self, namespace: str) -> int:
        self._clear_calls.append(namespace)
        count = 0
        to_delete = [k for k in self._store.keys() if k.startswith(f"{namespace}:")]
        for k in to_delete:
            del self._store[k]
            count += 1
        return count

    async def get_stats(self):
        return {"entries": len(self._store)}


class TestToolCacheManager:
    """Tests for ToolCacheManager."""

    @pytest.fixture
    def backend(self):
        """Create mock backend."""
        return MockCacheBackend()

    @pytest.fixture
    def manager(self, backend):
        """Create manager with mock backend."""
        return ToolCacheManager(backend=backend, default_ttl=3600)

    def test_init(self, backend):
        """Test initialization."""
        manager = ToolCacheManager(backend=backend, default_ttl=1800)

        assert manager._backend == backend
        assert manager._default_ttl == 1800
        assert manager._enable_dependency_tracking is True

    def test_init_without_dependency_tracking(self, backend):
        """Test initialization with dependency tracking disabled."""
        manager = ToolCacheManager(
            backend=backend, enable_dependency_tracking=False
        )

        assert manager._enable_dependency_tracking is False

    @pytest.mark.asyncio
    async def test_get_tool_result_miss(self, manager):
        """Test getting tool result when not in cache."""
        result = await manager.get_tool_result(
            tool_name="read",
            args={"path": "/src/main.py"},
            namespace=CacheNamespace.SESSION,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get_tool_result(self, manager):
        """Test setting and getting tool result."""
        args = {"path": "/src/main.py"}
        result_data = {"content": "print('hello')"}

        await manager.set_tool_result(
            tool_name="read",
            args=args,
            result=result_data,
            namespace=CacheNamespace.SESSION,
        )

        retrieved = await manager.get_tool_result(
            tool_name="read",
            args=args,
            namespace=CacheNamespace.SESSION,
        )

        assert retrieved == result_data

    @pytest.mark.asyncio
    async def test_set_with_file_dependencies(self, manager):
        """Test setting result with file dependencies."""
        await manager.set_tool_result(
            tool_name="code_search",
            args={"query": "auth"},
            result={"files": ["/src/auth.py"]},
            namespace=CacheNamespace.SESSION,
            file_dependencies={"/src/auth.py"},
        )

        # Check dependency graph
        dependents = manager._dependency_graph.get_file_dependents("/src/auth.py")
        assert "code_search" in dependents

    @pytest.mark.asyncio
    async def test_set_with_dependent_tools(self, manager):
        """Test setting result with dependent tools."""
        await manager.set_tool_result(
            tool_name="read",
            args={"path": "/src/main.py"},
            result={"content": "..."},
            namespace=CacheNamespace.SESSION,
            dependent_tools={"code_search", "ast_analysis"},
        )

        # Check dependency graph
        dependents = manager._dependency_graph.get_dependents("read")
        assert "code_search" in dependents
        assert "ast_analysis" in dependents

    @pytest.mark.asyncio
    async def test_invalidate_tool(self, manager, backend):
        """Test invalidating a tool."""
        # First cache a result
        await manager.set_tool_result(
            tool_name="read",
            args={"path": "/src/main.py"},
            result={"content": "..."},
            namespace=CacheNamespace.SESSION,
        )

        # Invalidate
        count = await manager.invalidate_tool("read", cascade=False)

        # Should have deleted from session namespace
        assert count == 1
        assert len(backend._delete_calls) > 0  # At least one delete call was made

    @pytest.mark.asyncio
    async def test_invalidate_tool_with_cascade(self, manager):
        """Test invalidating a tool with cascading."""
        # Set up dependencies: code_search depends on read
        await manager.set_tool_result(
            tool_name="read",
            args={"path": "/src/main.py"},
            result={"content": "..."},
            namespace=CacheNamespace.SESSION,
            dependent_tools={"code_search"},
        )

        # Cache code_search result
        await manager.set_tool_result(
            tool_name="code_search",
            args={"query": "auth"},
            result={"files": []},
            namespace=CacheNamespace.SESSION,
        )

        # Invalidate read with cascade
        count = await manager.invalidate_tool("read", cascade=True)

        # Should invalidate both read and code_search
        assert count >= 1

    @pytest.mark.asyncio
    async def test_invalidate_file_dependencies(self, manager):
        """Test invalidating tools that depend on a file."""
        # Set up file dependency
        await manager.set_tool_result(
            tool_name="code_search",
            args={"query": "auth"},
            result={"files": ["/src/auth.py"]},
            namespace=CacheNamespace.SESSION,
            file_dependencies={"/src/auth.py"},
        )

        # Invalidate file dependencies
        count = await manager.invalidate_file_dependencies("/src/auth.py")

        # Should have invalidated code_search
        assert count >= 1

    @pytest.mark.asyncio
    async def test_invalidate_namespace(self, manager, backend):
        """Test invalidating an entire namespace."""
        # Cache multiple results in session namespace
        await manager.set_tool_result(
            tool_name="read",
            args={"path": "/src/main.py"},
            result={"content": "..."},
            namespace=CacheNamespace.SESSION,
        )
        await manager.set_tool_result(
            tool_name="grep",
            args={"pattern": "test"},
            result={"matches": []},
            namespace=CacheNamespace.SESSION,
        )

        # Clear namespace
        count = await manager.invalidate_namespace(CacheNamespace.SESSION)

        assert count == 2
        assert "session" in backend._clear_calls

    @pytest.mark.asyncio
    async def test_clear_all(self, manager):
        """Test clearing all namespaces."""
        # Cache results in multiple namespaces
        await manager.set_tool_result(
            tool_name="read",
            args={"path": "/src/main.py"},
            result={"content": "..."},
            namespace=CacheNamespace.GLOBAL,
        )
        await manager.set_tool_result(
            tool_name="grep",
            args={"pattern": "test"},
            result={"matches": []},
            namespace=CacheNamespace.SESSION,
        )

        # Clear all
        count = await manager.clear_all()

        assert count >= 2

    @pytest.mark.asyncio
    async def test_get_stats(self, manager):
        """Test getting cache statistics."""
        stats = await manager.get_stats()

        assert "backend_stats" in stats
        assert "dependency_stats" in stats

    @pytest.mark.asyncio
    async def test_custom_ttl(self, manager):
        """Test setting result with custom TTL."""
        await manager.set_tool_result(
            tool_name="read",
            args={"path": "/src/main.py"},
            result={"content": "..."},
            namespace=CacheNamespace.SESSION,
            ttl_seconds=7200,
        )

        # Verify result was cached (TTL doesn't affect immediate retrieval)
        result = await manager.get_tool_result(
            tool_name="read",
            args={"path": "/src/main.py"},
            namespace=CacheNamespace.SESSION,
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_different_namespaces_isolated(self, manager):
        """Test that different namespaces are isolated."""
        args = {"path": "/src/main.py"}

        # Cache in session namespace
        await manager.set_tool_result(
            tool_name="read",
            args=args,
            result={"content": "session"},
            namespace=CacheNamespace.SESSION,
        )

        # Cache in global namespace
        await manager.set_tool_result(
            tool_name="read",
            args=args,
            result={"content": "global"},
            namespace=CacheNamespace.GLOBAL,
        )

        # Retrieve from each namespace
        session_result = await manager.get_tool_result(
            tool_name="read",
            args=args,
            namespace=CacheNamespace.SESSION,
        )
        global_result = await manager.get_tool_result(
            tool_name="read",
            args=args,
            namespace=CacheNamespace.GLOBAL,
        )

        assert session_result == {"content": "session"}
        assert global_result == {"content": "global"}

    @pytest.mark.asyncio
    async def test_invalidate_tool_specific_namespace(self, manager, backend):
        """Test invalidating tool in specific namespace only."""
        # Cache in multiple namespaces
        await manager.set_tool_result(
            tool_name="read",
            args={"path": "/src/main.py"},
            result={"content": "..."},
            namespace=CacheNamespace.SESSION,
        )
        await manager.set_tool_result(
            tool_name="read",
            args={"path": "/src/main.py"},
            result={"content": "..."},
            namespace=CacheNamespace.GLOBAL,
        )

        # Invalidate only session namespace
        count = await manager.invalidate_tool(
            "read", cascade=False, namespace=CacheNamespace.SESSION
        )

        # Should only invalidate session
        assert count == 1

    @pytest.mark.asyncio
    async def test_make_cache_key(self, manager):
        """Test cache key generation."""
        key1 = manager._make_cache_key("read", "abc123", CacheNamespace.SESSION)
        key2 = manager._make_cache_key("read", "abc123", CacheNamespace.SESSION)
        key3 = manager._make_cache_key("read", "xyz789", CacheNamespace.SESSION)
        key4 = manager._make_cache_key("read", "abc123", CacheNamespace.GLOBAL)

        assert key1 == key2
        assert key1 != key3
        assert key1 != key4

    @pytest.mark.asyncio
    async def test_hash_args_stable(self, manager):
        """Test that args hashing is stable."""
        args = {"path": "/src/main.py", "line": 10}
        hash1 = manager._hash_args(args)
        hash2 = manager._hash_args(args)

        # Same args should produce same hash
        assert hash1 == hash2

        # Different order should produce same hash (sorted keys)
        args_reordered = {"line": 10, "path": "/src/main.py"}
        hash3 = manager._hash_args(args_reordered)
        assert hash1 == hash3
