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

"""Tests for GraphManager."""

import asyncio
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from victor.core.indexing.graph_manager import GraphManager
from victor.core.indexing.file_watcher import FileChangeEvent, FileChangeType


@pytest.fixture(autouse=True)
async def reset_graph_manager():
    """Reset GraphManager singleton before each test."""
    manager = GraphManager.get_instance()
    await manager.clear_cache()
    manager._watcher_subscribed.clear()
    yield
    await manager.clear_cache()
    manager._watcher_subscribed.clear()


@pytest.fixture
def temp_codebase():
    """Create a temporary codebase directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Create some test files
        (root / "test.py").write_text("print('hello')")
        (root / "src").mkdir()
        (root / "src" / "module.py").write_text("def foo(): pass")
        yield root


class TestGraphManager:
    """Test GraphManager functionality."""

    @pytest.mark.asyncio
    async def test_singleton_instance(self):
        """Verify singleton pattern works."""
        manager1 = GraphManager.get_instance()
        manager2 = GraphManager.get_instance()

        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_get_stats_empty(self):
        """Verify stats are correct for empty manager."""
        manager = GraphManager.get_instance()

        stats = manager.get_stats()
        assert stats["total_graphs"] == 0
        assert stats["fresh_graphs"] == 0
        assert stats["stale_graphs"] == 0
        assert stats["watched_roots"] == 0

    @pytest.mark.asyncio
    async def test_cache_graph(self, temp_codebase):
        """Verify graph caching works."""
        manager = GraphManager.get_instance()

        # Mock graph result
        mock_graph = {"nodes": 5, "edges": 10}
        cache_key = f"{temp_codebase}:pagerank"

        manager._graph_cache[cache_key] = {
            "graph": mock_graph,
            "built_at": 1234567890.0,
            "stale": False,
        }

        stats = manager.get_stats()
        assert stats["total_graphs"] == 1
        assert stats["fresh_graphs"] == 1
        assert stats["stale_graphs"] == 0

    @pytest.mark.asyncio
    async def test_mark_stale_on_file_change(self, temp_codebase):
        """Verify file changes mark graphs as stale."""
        manager = GraphManager.get_instance()
        root_str = str(temp_codebase.resolve())

        # Cache some graphs
        for mode in ["pagerank", "centrality", "trace"]:
            cache_key = f"{root_str}:{mode}"
            manager._graph_cache[cache_key] = {
                "graph": {"mode": mode},
                "built_at": 1234567890.0,
                "stale": False,
            }

        # Simulate file change event
        event = FileChangeEvent(
            path=temp_codebase / "test.py",
            change_type=FileChangeType.MODIFIED,
            timestamp=None,
        )

        await manager._on_file_change(event, temp_codebase, None)

        # All graphs should be marked stale
        stats = manager.get_stats()
        assert stats["total_graphs"] == 3
        assert stats["fresh_graphs"] == 0
        assert stats["stale_graphs"] == 3

    @pytest.mark.asyncio
    async def test_invalidate_root(self, temp_codebase):
        """Verify invalidate_root marks all graphs for root as stale."""
        manager = GraphManager.get_instance()
        root_str = str(temp_codebase.resolve())

        # Cache some graphs
        for mode in ["pagerank", "centrality"]:
            cache_key = f"{root_str}:{mode}"
            manager._graph_cache[cache_key] = {
                "graph": {"mode": mode},
                "built_at": 1234567890.0,
                "stale": False,
            }

        # Invalidate root
        invalidated = await manager.invalidate_root(temp_codebase)

        assert invalidated == 2

        stats = manager.get_stats()
        assert stats["stale_graphs"] == 2

    @pytest.mark.asyncio
    async def test_clear_cache_all(self, temp_codebase):
        """Verify clear_cache removes all graphs."""
        manager = GraphManager.get_instance()

        # Cache some graphs
        for mode in ["pagerank", "centrality"]:
            cache_key = f"{temp_codebase}:{mode}"
            manager._graph_cache[cache_key] = {
                "graph": {"mode": mode},
                "built_at": 1234567890.0,
                "stale": False,
            }

        # Clear all
        cleared = await manager.clear_cache()

        assert cleared == 2

        stats = manager.get_stats()
        assert stats["total_graphs"] == 0

    @pytest.mark.asyncio
    async def test_clear_cache_specific_root(self, temp_codebase):
        """Verify clear_cache removes graphs for specific root."""
        manager = GraphManager.get_instance()
        root_str = str(temp_codebase.resolve())

        # Cache graphs for two different roots
        with tempfile.TemporaryDirectory() as tmpdir2:
            root2 = Path(tmpdir2)
            root2_str = str(root2.resolve())

            for i, root in enumerate([temp_codebase, root2]):
                root_str_current = str(root.resolve())
                for mode in ["pagerank", "centrality"]:
                    cache_key = f"{root_str_current}:{mode}"
                    manager._graph_cache[cache_key] = {
                        "graph": {"root": i},
                        "built_at": 1234567890.0,
                        "stale": False,
                    }

            # Clear only first root
            cleared = await manager.clear_cache(temp_codebase)

            assert cleared == 2

            stats = manager.get_stats()
            assert stats["total_graphs"] == 2  # Still 2 from root2

    @pytest.mark.asyncio
    async def test_watcher_subscription_tracking(self, temp_codebase):
        """Verify file watcher subscription is tracked."""
        manager = GraphManager.get_instance()

        # Initially not subscribed
        root_str = str(temp_codebase.resolve())
        assert root_str not in manager._watcher_subscribed

        # Ensure file watcher (subscribes)
        await manager._ensure_file_watcher(temp_codebase, None)

        # Should be subscribed now
        assert root_str in manager._watcher_subscribed

        # Call again - should not duplicate subscription
        await manager._ensure_file_watcher(temp_codebase, None)

        # Still only one subscription
        assert len(manager._watcher_subscribed) == 1

    @pytest.mark.asyncio
    async def test_file_deleted_marks_stale(self, temp_codebase):
        """Verify file deletion marks graphs as stale."""
        manager = GraphManager.get_instance()
        root_str = str(temp_codebase.resolve())

        # Cache a graph
        cache_key = f"{root_str}:pagerank"
        manager._graph_cache[cache_key] = {
            "graph": {"nodes": 5},
            "built_at": 1234567890.0,
            "stale": False,
        }

        # Simulate file deletion
        event = FileChangeEvent(
            path=temp_codebase / "test.py",
            change_type=FileChangeType.DELETED,
            timestamp=None,
        )

        await manager._on_file_change(event, temp_codebase, None)

        # Graph should be marked stale
        cache_entry = manager._graph_cache.get(cache_key)
        assert cache_entry is not None
        assert cache_entry["stale"] is True

    @pytest.mark.asyncio
    async def test_file_created_marks_stale(self, temp_codebase):
        """Verify file creation marks graphs as stale."""
        manager = GraphManager.get_instance()
        root_str = str(temp_codebase.resolve())

        # Cache a graph
        cache_key = f"{root_str}:pagerank"
        manager._graph_cache[cache_key] = {
            "graph": {"nodes": 5},
            "built_at": 1234567890.0,
            "stale": False,
        }

        # Simulate file creation
        event = FileChangeEvent(
            path=temp_codebase / "new.py",
            change_type=FileChangeType.CREATED,
            timestamp=None,
        )

        await manager._on_file_change(event, temp_codebase, None)

        # Graph should be marked stale
        cache_entry = manager._graph_cache.get(cache_key)
        assert cache_entry is not None
        assert cache_entry["stale"] is True

    @pytest.mark.asyncio
    async def test_different_roots_separate_caches(self, temp_codebase):
        """Verify different roots have separate graph caches."""
        manager = GraphManager.get_instance()

        with tempfile.TemporaryDirectory() as tmpdir2:
            root2 = Path(tmpdir2)

            # Cache graphs for both roots
            for i, root in enumerate([temp_codebase, root2]):
                root_str = str(root.resolve())
                cache_key = f"{root_str}:pagerank"
                manager._graph_cache[cache_key] = {
                    "graph": {"root": i},
                    "built_at": 1234567890.0,
                    "stale": False,
                }

            stats = manager.get_stats()
            assert stats["total_graphs"] == 2

            # Invalidate first root - should not affect second
            await manager.invalidate_root(temp_codebase)

            stats = manager.get_stats()
            assert stats["stale_graphs"] == 1  # Only first root

    @pytest.mark.asyncio
    async def test_different_modes_separate_caches(self, temp_codebase):
        """Verify different modes have separate caches."""
        manager = GraphManager.get_instance()

        # Cache graphs for different modes
        for mode in ["pagerank", "centrality", "trace"]:
            cache_key = f"{temp_codebase}:{mode}"
            manager._graph_cache[cache_key] = {
                "graph": {"mode": mode},
                "built_at": 1234567890.0,
                "stale": False,
            }

        stats = manager.get_stats()
        assert stats["total_graphs"] == 3

        # All should be fresh
        assert stats["fresh_graphs"] == 3
