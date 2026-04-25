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

"""Unit tests for code search indexing initialization tracking."""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.core.indexing.file_watcher import FileChangeEvent, FileChangeType
from victor.framework.search.codebase_embedding_bridge import (
    build_codebase_index_manifest,
    write_codebase_index_manifest,
)
from victor.tools.code_search_tool import (
    IntegrityProbeOutcome,
    _ensure_file_watcher_subscription,
    _build_codebase_embedding_config,
    _build_index_failure_key,
    _finalize_index_storage,
    _get_or_build_index,
    _latest_mtime,
    _on_file_change,
    _probe_index_integrity,
    _schedule_file_change_refresh,
    clear_index_cache,
)


class _NoOpAsyncLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _TrackingAsyncLock:
    def __init__(self, registry: "_TrackingIndexLockRegistry") -> None:
        self._registry = registry

    async def __aenter__(self):
        self._registry.in_lock = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._registry.in_lock = False
        return False


class _FakeIndexLockRegistry:
    def __init__(self) -> None:
        self.used_paths: list[Path] = []

    async def acquire_lock(self, root: Path) -> _NoOpAsyncLock:
        return _NoOpAsyncLock()

    def mark_lock_used(self, root: Path) -> None:
        self.used_paths.append(root)


class _TrackingIndexLockRegistry(_FakeIndexLockRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.in_lock = False

    async def acquire_lock(self, root: Path) -> _TrackingAsyncLock:
        self.used_paths.append(root)
        return _TrackingAsyncLock(self)


class _SerialIndexLockRegistry(_FakeIndexLockRegistry):
    def __init__(self) -> None:
        super().__init__()
        self._lock = asyncio.Lock()

    async def acquire_lock(self, root: Path) -> asyncio.Lock:
        self.used_paths.append(root)
        return self._lock


class _FakeCapabilityRegistry:
    def __init__(self, factory: object) -> None:
        self._factory = factory

    def ensure_bootstrapped(self) -> None:
        return None

    def get(self, protocol: object) -> object:
        del protocol
        return self._factory

    def is_enhanced(self, protocol: object) -> bool:
        del protocol
        return True


class TestIndexingFlagBehavior:
    """Tests for _is_indexing flag in CodebaseIndex."""

    def test_indexing_flag_attributes_exist(self):
        """Test that CodebaseIndex has _is_indexing and _indexing_start_time attributes."""
        # This test requires the actual victor-coding package
        pytest.importorskip("victor_coding")

        from victor_coding.codebase.indexer import CodebaseIndex

        # Create a temporary directory for testing
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            index = CodebaseIndex(root_path=tmpdir, use_embeddings=False)

            # Check that flags exist and are initialized correctly
            assert hasattr(index, "_is_indexing")
            assert hasattr(index, "_indexing_start_time")
            assert index._is_indexing is False
            assert index._indexing_start_time is None

    def test_indexing_flag_set_during_indexing(self):
        """Test that _is_indexing flag is set to True during index_codebase."""
        pytest.importorskip("victor_coding")

        from victor_coding.codebase.indexer import CodebaseIndex

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            index = CodebaseIndex(root_path=tmpdir, use_embeddings=False)

            # Mock the indexing process to just set the flag
            async def mock_index():
                index._is_indexing = True
                index._indexing_start_time = time.time()
                try:
                    await asyncio.sleep(0.1)  # Simulate some work
                finally:
                    index._is_indexing = False
                    index._indexing_start_time = None

            # Run the mock indexing
            asyncio.run(mock_index())

            # Verify flag is cleared after indexing
            assert index._is_indexing is False
            assert index._indexing_start_time is None


class TestCorruptionDetection:
    """Tests for index integrity checking during init."""

    @pytest.mark.asyncio
    async def test_corruption_check_skipped_during_indexing(self):
        """Test that integrity check is skipped when _is_indexing flag is True."""
        # Create a mock index with _is_indexing flag
        mock_index = MagicMock()
        mock_index._is_indexing = True
        mock_index._is_indexed = False

        # The integrity check should return False (no rebuild needed)
        result = await _probe_index_integrity(mock_index, timeout=5.0)

        assert result == IntegrityProbeOutcome()  # Should NOT trigger rebuild

    @pytest.mark.asyncio
    async def test_corruption_check_proceeds_when_not_indexing(self):
        """Test that integrity check proceeds when _is_indexing flag is False."""
        # Create a mock index that appears healthy
        mock_index = MagicMock()
        mock_index._is_indexing = False
        mock_index._is_indexed = True

        # Mock vector store with data
        mock_table = MagicMock()
        mock_table.count_rows.return_value = 1000
        mock_store = MagicMock()
        mock_store._table = mock_table
        mock_index._vector_store = mock_store

        # The integrity check should return False (healthy)
        result = await _probe_index_integrity(mock_index, timeout=5.0)

        assert result == IntegrityProbeOutcome()  # Should be healthy
        mock_table.count_rows.assert_called_once()

    @pytest.mark.asyncio
    async def test_transient_error_does_not_trigger_rebuild(self):
        """Test that transient init errors don't trigger rebuild."""
        # Create a mock index that throws a "locked" error
        mock_index = MagicMock()
        mock_index._is_indexing = False
        mock_index._is_indexed = True

        # Mock vector store to return no table (skip fast path)
        mock_store = MagicMock()
        mock_store._table = None  # No table, so it will try semantic_search
        mock_index._vector_store = mock_store

        # Mock semantic_search that raises a "locked" error
        async def mock_semantic_search_error(*args, **kwargs):
            raise Exception("database is locked")

        mock_index.semantic_search = mock_semantic_search_error

        # The integrity check should return False (no rebuild for transient errors)
        result = await _probe_index_integrity(mock_index, timeout=5.0)

        assert result == IntegrityProbeOutcome()  # Should NOT rebuild for transient errors

    @pytest.mark.asyncio
    async def test_actual_corruption_triggers_rebuild(self):
        """Test that actual corruption errors trigger background rebuild."""
        # Create a mock index that throws a non-transient error
        mock_index = MagicMock()
        mock_index._is_indexing = False
        mock_index._is_indexed = True

        # Mock semantic_search that raises a "corruption" error
        async def mock_semantic_search_corrupt(*args, **kwargs):
            raise ValueError("Invalid data format")

        mock_index.semantic_search = mock_semantic_search_corrupt
        mock_index.index_codebase = AsyncMock()

        # The integrity check should return rebuilt=True when inline rebuild succeeds.
        result = await _probe_index_integrity(mock_index, timeout=5.0)

        assert result == IntegrityProbeOutcome(rebuilt=True)  # Should trigger rebuild
        assert mock_index._is_indexed is False  # Flag should be cleared

    @pytest.mark.asyncio
    async def test_failed_rebuild_marks_index_stale(self, monkeypatch):
        """A corrupt persisted index should stay stale if its inline rebuild fails."""
        mock_index = MagicMock()
        mock_index._is_indexing = False
        mock_index._is_indexed = True

        async def mock_semantic_search_corrupt(*args, **kwargs):
            raise ValueError("Invalid data format")

        mock_index.semantic_search = mock_semantic_search_corrupt

        import victor.tools.code_search_tool as code_search_tool_module

        monkeypatch.setattr(
            code_search_tool_module,
            "_background_index_rebuild",
            AsyncMock(return_value=False),
        )

        result = await _probe_index_integrity(mock_index, timeout=5.0)

        assert result == IntegrityProbeOutcome(stale=True)
        assert mock_index._is_indexed is False

    @pytest.mark.asyncio
    async def test_probe_cancellation_cleans_up_semantic_search_task(self):
        """Cancelling the integrity probe should also cancel the spawned search task."""
        mock_index = MagicMock()
        mock_index._is_indexing = False
        mock_index._is_indexed = True

        mock_store = MagicMock()
        mock_store._table = None
        mock_index._vector_store = mock_store

        cancelled = asyncio.Event()

        async def mock_semantic_search(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        mock_index.semantic_search = mock_semantic_search

        probe = asyncio.create_task(_probe_index_integrity(mock_index, timeout=5.0))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        probe.cancel()

        with pytest.raises(asyncio.CancelledError):
            await probe

        assert cancelled.is_set()


class TestErrorClassification:
    """Tests for error classification in corruption detection."""

    @pytest.mark.asyncio
    async def test_empty_exception_string_classified(self):
        """Test that empty exception strings are handled gracefully."""
        mock_index = MagicMock()
        mock_index._is_indexing = False
        mock_index._is_indexed = True

        # Mock semantic_search that raises exception with empty string
        class EmptyError(Exception):
            def __str__(self):
                return ""

        async def mock_empty_error(*args, **kwargs):
            raise EmptyError()

        mock_index.semantic_search = mock_empty_error
        mock_index.index_codebase = AsyncMock()

        # Should use type name as error message
        result = await _probe_index_integrity(mock_index, timeout=5.0)

        assert result == IntegrityProbeOutcome(rebuilt=True)  # Should trigger rebuild

    @pytest.mark.asyncio
    async def test_timeout_error_classified_as_transient(self):
        """Test that timeout errors are classified as transient."""
        mock_index = MagicMock()
        mock_index._is_indexing = False
        mock_index._is_indexed = True

        # Mock vector store to return no table (skip fast path)
        mock_store = MagicMock()
        mock_store._table = None
        mock_index._vector_store = mock_store

        # Mock semantic_search that raises timeout
        async def mock_timeout(*args, **kwargs):
            raise TimeoutError("Operation timed out")

        mock_index.semantic_search = mock_timeout

        # Should NOT trigger rebuild for timeout errors
        result = await _probe_index_integrity(mock_index, timeout=5.0)

        assert result == IntegrityProbeOutcome()  # Should skip rebuild

    @pytest.mark.asyncio
    async def test_not_ready_error_classified_as_transient(self):
        """Test that "not ready" errors are classified as transient."""
        mock_index = MagicMock()
        mock_index._is_indexing = False
        mock_index._is_indexed = True

        # Mock vector store to return no table (skip fast path)
        mock_store = MagicMock()
        mock_store._table = None
        mock_index._vector_store = mock_store

        # Mock semantic_search that raises "not ready"
        async def mock_not_ready(*args, **kwargs):
            raise RuntimeError("Service not ready")

        mock_index.semantic_search = mock_not_ready

        # Should NOT trigger rebuild
        result = await _probe_index_integrity(mock_index, timeout=5.0)

        assert result == IntegrityProbeOutcome()  # Should skip rebuild


class TestBackgroundRebuildLogging:
    """Tests for background rebuild logging improvements."""

    @pytest.mark.asyncio
    async def test_rebuild_finalizes_structural_provider_writes(self):
        """Successful rebuilds should flush structural provider buffers."""
        from victor.tools.code_search_tool import _background_index_rebuild

        provider = SimpleNamespace(
            config=SimpleNamespace(vector_store="victor_structural_bridge"),
            get_stats=AsyncMock(return_value={"total_documents": 1}),
        )
        mock_index = SimpleNamespace(
            root=Path("/test/project"),
            index_codebase=AsyncMock(),
            embedding_provider=provider,
        )

        result = await _background_index_rebuild(mock_index, rebuild_timeout=10.0)

        assert result is True
        mock_index.index_codebase.assert_awaited_once()
        provider.get_stats.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rebuild_logging_includes_index_path(self, caplog):
        """Test that rebuild logging includes the index path."""
        from victor.tools.code_search_tool import _background_index_rebuild

        # Create a mock index
        mock_index = MagicMock()
        mock_index.root = Path("/test/project")

        # Mock index_codebase to succeed quickly
        async def mock_rebuild():
            pass

        mock_index.index_codebase = mock_rebuild

        # Run rebuild (should log with index path)
        import logging

        with caplog.at_level(logging.INFO):
            await _background_index_rebuild(mock_index, rebuild_timeout=10.0)

        # Check that log messages include the index path
        log_messages = [record.message for record in caplog.records]
        assert any("/test/project" in msg for msg in log_messages)

    @pytest.mark.asyncio
    async def test_rebuild_logging_includes_timing(self, caplog):
        """Test that rebuild logging includes timing information."""
        from victor.tools.code_search_tool import _background_index_rebuild

        # Create a mock index
        mock_index = MagicMock()
        mock_index.root = Path("/test/project")

        # Mock index_codebase to take some time
        async def mock_rebuild_with_delay():
            await asyncio.sleep(0.1)

        mock_index.index_codebase = mock_rebuild_with_delay

        # Run rebuild
        import logging

        with caplog.at_level(logging.INFO):
            await _background_index_rebuild(mock_index, rebuild_timeout=10.0)

        # Check that log messages include timing
        log_messages = [record.message for record in caplog.records]
        assert any("0." in msg or "0.1" in msg or "0.2" in msg for msg in log_messages)

    @pytest.mark.asyncio
    async def test_rebuild_failure_logging_includes_error_details(self, caplog):
        """Test that rebuild failures include error details."""
        from victor.tools.code_search_tool import _background_index_rebuild

        # Create a mock index
        mock_index = MagicMock()
        mock_index.root = Path("/test/project")

        # Mock index_codebase to fail
        async def mock_rebuild_failure():
            raise ValueError("Test rebuild error")

        mock_index.index_codebase = mock_rebuild_failure

        # Run rebuild
        import logging

        with caplog.at_level(logging.WARNING):
            result = await _background_index_rebuild(mock_index, rebuild_timeout=10.0)

        # Check that log includes error details
        assert result is False
        log_messages = [record.message for record in caplog.records]
        assert any("Test rebuild error" in msg for msg in log_messages)
        assert any("/test/project" in msg for msg in log_messages)


class TestStructuralIndexPersistence:
    """Tests for manifest-aware persistent index reuse."""

    @pytest.mark.asyncio
    async def test_get_or_build_index_reuses_cached_index_when_in_memory_manifest_matches(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "repo"
        root.mkdir()
        (root / "main.py").write_text("print('hello')\n", encoding="utf-8")

        settings = SimpleNamespace(
            codebase_vector_store="lancedb",
            codebase_embedding_provider="sentence-transformers",
            codebase_embedding_model="BAAI/bge-small-en-v1.5",
            codebase_persist_directory=str(tmp_path / "embeddings"),
            codebase_dimension=384,
            codebase_batch_size=32,
            codebase_structural_indexing_enabled=False,
            codebase_chunking_strategy="tree_sitter_structural",
            codebase_chunk_size=500,
            codebase_chunk_overlap=50,
            codebase_embedding_extra_config={},
            codebase_graph_store="sqlite",
            codebase_graph_path=None,
            unified_embedding_model="BAAI/bge-small-en-v1.5",
        )

        cached_index = SimpleNamespace(incremental_reindex=AsyncMock())
        index_manifest = build_codebase_index_manifest(
            _build_codebase_embedding_config(settings, root)
        )
        fake_cache: dict[str, dict[str, object]] = {
            str(root): {
                "index": cached_index,
                "latest_mtime": _latest_mtime(root),
                "indexed_at": time.time(),
                "index_manifest": index_manifest,
                "watcher_subscribed": True,
            }
        }

        factory = MagicMock()
        fake_factory = SimpleNamespace(create=factory)

        import victor.core.capability_registry as capability_registry_module
        import victor.core.indexing.index_lock as index_lock_module
        import victor.tools.code_search_tool as code_search_tool_module

        monkeypatch.setattr(
            capability_registry_module.CapabilityRegistry,
            "get_instance",
            staticmethod(lambda: _FakeCapabilityRegistry(fake_factory)),
        )
        monkeypatch.setattr(
            index_lock_module.IndexLockRegistry,
            "get_instance",
            staticmethod(lambda: _FakeIndexLockRegistry()),
        )
        monkeypatch.setattr(
            code_search_tool_module, "_get_index_cache", lambda exec_ctx=None: fake_cache
        )
        monkeypatch.setattr(_get_or_build_index, "_failure_cache", {}, raising=False)

        clear_index_cache()
        index, rebuilt = await _get_or_build_index(root=root, settings=settings)

        assert index is cached_index
        assert rebuilt is False
        factory.assert_not_called()
        cached_index.incremental_reindex.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_or_build_index_coalesces_concurrent_watcher_subscription(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "repo"
        root.mkdir()
        (root / "main.py").write_text("print('hello')\n", encoding="utf-8")

        settings = SimpleNamespace(
            codebase_vector_store="lancedb",
            codebase_embedding_provider="sentence-transformers",
            codebase_embedding_model="BAAI/bge-small-en-v1.5",
            codebase_persist_directory=str(tmp_path / "embeddings"),
            codebase_dimension=384,
            codebase_batch_size=32,
            codebase_structural_indexing_enabled=False,
            codebase_chunking_strategy="tree_sitter_structural",
            codebase_chunk_size=500,
            codebase_chunk_overlap=50,
            codebase_embedding_extra_config={},
            codebase_graph_store="sqlite",
            codebase_graph_path=None,
            unified_embedding_model="BAAI/bge-small-en-v1.5",
        )

        cached_index = SimpleNamespace(incremental_reindex=AsyncMock())
        index_manifest = build_codebase_index_manifest(
            _build_codebase_embedding_config(settings, root)
        )
        fake_cache: dict[str, dict[str, object]] = {
            str(root): {
                "index": cached_index,
                "latest_mtime": _latest_mtime(root),
                "indexed_at": time.time(),
                "index_manifest": index_manifest,
                "watcher_subscribed": False,
            }
        }

        factory = MagicMock()
        fake_factory = SimpleNamespace(create=factory)

        async def _delayed_subscribe(*args, **kwargs):
            await asyncio.sleep(0.01)
            return True

        import victor.core.capability_registry as capability_registry_module
        import victor.core.indexing.index_lock as index_lock_module
        import victor.tools.code_search_tool as code_search_tool_module

        subscribe_mock = AsyncMock(side_effect=_delayed_subscribe)

        monkeypatch.setattr(
            capability_registry_module.CapabilityRegistry,
            "get_instance",
            staticmethod(lambda: _FakeCapabilityRegistry(fake_factory)),
        )
        monkeypatch.setattr(
            index_lock_module.IndexLockRegistry,
            "get_instance",
            staticmethod(lambda: _FakeIndexLockRegistry()),
        )
        monkeypatch.setattr(
            code_search_tool_module, "_get_index_cache", lambda exec_ctx=None: fake_cache
        )
        monkeypatch.setattr(code_search_tool_module, "_subscribe_to_file_watcher", subscribe_mock)
        monkeypatch.setattr(_get_or_build_index, "_failure_cache", {}, raising=False)

        clear_index_cache()
        first, second = await asyncio.gather(
            _get_or_build_index(root=root, settings=settings),
            _get_or_build_index(root=root, settings=settings),
        )

        assert first == (cached_index, False)
        assert second == (cached_index, False)
        assert subscribe_mock.await_count == 1
        assert fake_cache[str(root)]["watcher_subscribed"] is True
        assert "watcher_subscription_task" not in fake_cache[str(root)]
        factory.assert_not_called()
        cached_index.incremental_reindex.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_or_build_index_reuses_completed_watcher_subscription_task(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "repo"
        root.mkdir()
        (root / "main.py").write_text("print('hello')\n", encoding="utf-8")

        settings = SimpleNamespace(
            codebase_vector_store="lancedb",
            codebase_embedding_provider="sentence-transformers",
            codebase_embedding_model="BAAI/bge-small-en-v1.5",
            codebase_persist_directory=str(tmp_path / "embeddings"),
            codebase_dimension=384,
            codebase_batch_size=32,
            codebase_structural_indexing_enabled=False,
            codebase_chunking_strategy="tree_sitter_structural",
            codebase_chunk_size=500,
            codebase_chunk_overlap=50,
            codebase_embedding_extra_config={},
            codebase_graph_store="sqlite",
            codebase_graph_path=None,
            unified_embedding_model="BAAI/bge-small-en-v1.5",
        )

        cached_index = SimpleNamespace(incremental_reindex=AsyncMock())
        index_manifest = build_codebase_index_manifest(
            _build_codebase_embedding_config(settings, root)
        )
        completed_task = asyncio.create_task(asyncio.sleep(0, result=True))
        await completed_task
        fake_cache: dict[str, dict[str, object]] = {
            str(root): {
                "index": cached_index,
                "latest_mtime": _latest_mtime(root),
                "indexed_at": time.time(),
                "index_manifest": index_manifest,
                "watcher_subscribed": False,
                "watcher_subscription_task": completed_task,
            }
        }

        factory = MagicMock()
        fake_factory = SimpleNamespace(create=factory)

        import victor.core.capability_registry as capability_registry_module
        import victor.core.indexing.index_lock as index_lock_module
        import victor.tools.code_search_tool as code_search_tool_module

        subscribe_mock = AsyncMock(return_value=True)

        monkeypatch.setattr(
            capability_registry_module.CapabilityRegistry,
            "get_instance",
            staticmethod(lambda: _FakeCapabilityRegistry(fake_factory)),
        )
        monkeypatch.setattr(
            index_lock_module.IndexLockRegistry,
            "get_instance",
            staticmethod(lambda: _FakeIndexLockRegistry()),
        )
        monkeypatch.setattr(
            code_search_tool_module, "_get_index_cache", lambda exec_ctx=None: fake_cache
        )
        monkeypatch.setattr(code_search_tool_module, "_subscribe_to_file_watcher", subscribe_mock)
        monkeypatch.setattr(_get_or_build_index, "_failure_cache", {}, raising=False)

        clear_index_cache()
        index, rebuilt = await _get_or_build_index(root=root, settings=settings)

        assert index is cached_index
        assert rebuilt is False
        assert subscribe_mock.await_count == 0
        assert fake_cache[str(root)]["watcher_subscribed"] is True
        assert "watcher_subscription_task" not in fake_cache[str(root)]
        factory.assert_not_called()
        cached_index.incremental_reindex.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_or_build_index_retries_failed_completed_watcher_subscription_task(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "repo"
        root.mkdir()
        (root / "main.py").write_text("print('hello')\n", encoding="utf-8")

        settings = SimpleNamespace(
            codebase_vector_store="lancedb",
            codebase_embedding_provider="sentence-transformers",
            codebase_embedding_model="BAAI/bge-small-en-v1.5",
            codebase_persist_directory=str(tmp_path / "embeddings"),
            codebase_dimension=384,
            codebase_batch_size=32,
            codebase_structural_indexing_enabled=False,
            codebase_chunking_strategy="tree_sitter_structural",
            codebase_chunk_size=500,
            codebase_chunk_overlap=50,
            codebase_embedding_extra_config={},
            codebase_graph_store="sqlite",
            codebase_graph_path=None,
            unified_embedding_model="BAAI/bge-small-en-v1.5",
        )

        cached_index = SimpleNamespace(incremental_reindex=AsyncMock())
        index_manifest = build_codebase_index_manifest(
            _build_codebase_embedding_config(settings, root)
        )

        async def _explode() -> bool:
            raise RuntimeError("watcher failed")

        failed_task = asyncio.create_task(_explode())
        with pytest.raises(RuntimeError):
            await failed_task

        fake_cache: dict[str, dict[str, object]] = {
            str(root): {
                "index": cached_index,
                "latest_mtime": _latest_mtime(root),
                "indexed_at": time.time(),
                "index_manifest": index_manifest,
                "watcher_subscribed": False,
                "watcher_subscription_task": failed_task,
            }
        }

        factory = MagicMock()
        fake_factory = SimpleNamespace(create=factory)

        import victor.core.capability_registry as capability_registry_module
        import victor.core.indexing.index_lock as index_lock_module
        import victor.tools.code_search_tool as code_search_tool_module

        subscribe_mock = AsyncMock(return_value=True)

        monkeypatch.setattr(
            capability_registry_module.CapabilityRegistry,
            "get_instance",
            staticmethod(lambda: _FakeCapabilityRegistry(fake_factory)),
        )
        monkeypatch.setattr(
            index_lock_module.IndexLockRegistry,
            "get_instance",
            staticmethod(lambda: _FakeIndexLockRegistry()),
        )
        monkeypatch.setattr(
            code_search_tool_module, "_get_index_cache", lambda exec_ctx=None: fake_cache
        )
        monkeypatch.setattr(code_search_tool_module, "_subscribe_to_file_watcher", subscribe_mock)
        monkeypatch.setattr(_get_or_build_index, "_failure_cache", {}, raising=False)

        clear_index_cache()
        index, rebuilt = await _get_or_build_index(root=root, settings=settings)

        assert index is cached_index
        assert rebuilt is False
        assert subscribe_mock.await_count == 1
        assert fake_cache[str(root)]["watcher_subscribed"] is True
        assert "watcher_subscription_task" not in fake_cache[str(root)]
        factory.assert_not_called()
        cached_index.incremental_reindex.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_watcher_subscription_survives_waiter_cancellation(self, tmp_path, monkeypatch):
        root = tmp_path / "repo"
        root.mkdir()
        cache_entry: dict[str, object] = {"watcher_subscribed": False}

        async def _delayed_subscribe(*args, **kwargs):
            await asyncio.sleep(0.02)
            return True

        import victor.tools.code_search_tool as code_search_tool_module

        subscribe_mock = AsyncMock(side_effect=_delayed_subscribe)
        monkeypatch.setattr(code_search_tool_module, "_subscribe_to_file_watcher", subscribe_mock)

        first_waiter = asyncio.create_task(_ensure_file_watcher_subscription(cache_entry, root))
        await asyncio.sleep(0)
        first_waiter.cancel()

        with pytest.raises(asyncio.CancelledError):
            await first_waiter

        assert isinstance(cache_entry.get("watcher_subscription_task"), asyncio.Task)

        second_result = await _ensure_file_watcher_subscription(cache_entry, root)

        assert second_result is True
        assert subscribe_mock.await_count == 1
        assert cache_entry["watcher_subscribed"] is True
        assert "watcher_subscription_task" not in cache_entry

    @pytest.mark.asyncio
    async def test_get_or_build_index_coalesces_incremental_refresh_under_lock(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "repo"
        root.mkdir()
        (root / "main.py").write_text("print('hello')\n", encoding="utf-8")

        settings = SimpleNamespace(
            codebase_vector_store="lancedb",
            codebase_embedding_provider="sentence-transformers",
            codebase_embedding_model="BAAI/bge-small-en-v1.5",
            codebase_persist_directory=str(tmp_path / "embeddings"),
            codebase_dimension=384,
            codebase_batch_size=32,
            codebase_structural_indexing_enabled=False,
            codebase_chunking_strategy="tree_sitter_structural",
            codebase_chunk_size=500,
            codebase_chunk_overlap=50,
            codebase_embedding_extra_config={},
            codebase_graph_store="sqlite",
            codebase_graph_path=None,
            unified_embedding_model="BAAI/bge-small-en-v1.5",
        )

        provider = SimpleNamespace(
            config=SimpleNamespace(vector_store="victor_structural_bridge"),
            get_stats=AsyncMock(return_value={"total_documents": 1}),
        )

        async def _delayed_incremental() -> None:
            await asyncio.sleep(0.01)

        cached_index = SimpleNamespace(
            incremental_reindex=AsyncMock(side_effect=_delayed_incremental),
            embedding_provider=provider,
        )
        index_manifest = build_codebase_index_manifest(
            _build_codebase_embedding_config(settings, root)
        )
        fake_cache: dict[str, dict[str, object]] = {
            str(root): {
                "index": cached_index,
                "latest_mtime": 0.0,
                "indexed_at": time.time(),
                "index_manifest": index_manifest,
                "watcher_subscribed": True,
            }
        }

        factory = MagicMock()
        fake_factory = SimpleNamespace(create=factory)
        lock_registry = _SerialIndexLockRegistry()

        import victor.core.capability_registry as capability_registry_module
        import victor.core.indexing.index_lock as index_lock_module
        import victor.tools.code_search_tool as code_search_tool_module

        monkeypatch.setattr(
            capability_registry_module.CapabilityRegistry,
            "get_instance",
            staticmethod(lambda: _FakeCapabilityRegistry(fake_factory)),
        )
        monkeypatch.setattr(
            index_lock_module.IndexLockRegistry,
            "get_instance",
            staticmethod(lambda: lock_registry),
        )
        monkeypatch.setattr(
            code_search_tool_module, "_get_index_cache", lambda exec_ctx=None: fake_cache
        )
        monkeypatch.setattr(_get_or_build_index, "_failure_cache", {}, raising=False)

        clear_index_cache()
        first, second = await asyncio.gather(
            _get_or_build_index(root=root, settings=settings),
            _get_or_build_index(root=root, settings=settings),
        )

        assert first == (cached_index, False)
        assert second == (cached_index, False)
        cached_index.incremental_reindex.assert_awaited_once()
        provider.get_stats.assert_awaited_once()
        factory.assert_not_called()
        assert fake_cache[str(root)]["latest_mtime"] == _latest_mtime(root)

    @pytest.mark.asyncio
    async def test_get_or_build_index_invalidates_cached_index_when_manifest_changes(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "repo"
        root.mkdir()
        (root / "main.py").write_text("print('hello')\n", encoding="utf-8")

        persist_dir = tmp_path / "embeddings"
        persist_dir.mkdir()
        (persist_dir / "existing.lance").mkdir()

        settings = SimpleNamespace(
            codebase_vector_store="lancedb",
            codebase_embedding_provider="sentence-transformers",
            codebase_embedding_model="BAAI/bge-small-en-v1.5",
            codebase_persist_directory=str(persist_dir),
            codebase_dimension=384,
            codebase_batch_size=32,
            codebase_structural_indexing_enabled=False,
            codebase_chunking_strategy="tree_sitter_structural",
            codebase_chunk_size=500,
            codebase_chunk_overlap=50,
            codebase_embedding_extra_config={},
            codebase_graph_store="sqlite",
            codebase_graph_path=None,
            unified_embedding_model="BAAI/bge-small-en-v1.5",
        )

        new_manifest = build_codebase_index_manifest(
            _build_codebase_embedding_config(settings, root)
        )
        write_codebase_index_manifest(persist_dir, new_manifest)

        cached_index = SimpleNamespace(incremental_reindex=AsyncMock())
        replacement_index = SimpleNamespace(index_codebase=AsyncMock(), _is_indexed=False)
        fake_cache: dict[str, dict[str, object]] = {
            str(root): {
                "index": cached_index,
                "latest_mtime": _latest_mtime(root),
                "indexed_at": time.time(),
                "index_manifest": {"schema_version": -1},
                "watcher_subscribed": True,
            }
        }
        fake_factory = SimpleNamespace(create=lambda **kwargs: replacement_index)
        probe_mock = AsyncMock(return_value=False)

        import victor.core.capability_registry as capability_registry_module
        import victor.core.indexing.index_lock as index_lock_module
        import victor.tools.code_search_tool as code_search_tool_module

        monkeypatch.setattr(
            capability_registry_module.CapabilityRegistry,
            "get_instance",
            staticmethod(lambda: _FakeCapabilityRegistry(fake_factory)),
        )
        monkeypatch.setattr(
            index_lock_module.IndexLockRegistry,
            "get_instance",
            staticmethod(lambda: _FakeIndexLockRegistry()),
        )
        monkeypatch.setattr(
            code_search_tool_module, "_get_index_cache", lambda exec_ctx=None: fake_cache
        )
        monkeypatch.setattr(code_search_tool_module, "_probe_index_integrity", probe_mock)

        clear_index_cache()
        index, rebuilt = await _get_or_build_index(root=root, settings=settings)

        assert index is replacement_index
        assert rebuilt is False
        cached_index.incremental_reindex.assert_not_awaited()
        replacement_index.index_codebase.assert_not_awaited()
        probe_mock.assert_awaited_once_with(replacement_index)
        assert fake_cache[str(root)]["index"] is replacement_index
        assert fake_cache[str(root)]["index_manifest"] == new_manifest

    @pytest.mark.asyncio
    async def test_get_or_build_index_rebuilds_stale_cache_inside_lock(self, tmp_path, monkeypatch):
        root = tmp_path / "repo"
        root.mkdir()
        (root / "main.py").write_text("print('hello')\n", encoding="utf-8")

        settings = SimpleNamespace(
            codebase_vector_store="lancedb",
            codebase_embedding_provider="sentence-transformers",
            codebase_embedding_model="BAAI/bge-small-en-v1.5",
            codebase_persist_directory=str(tmp_path / "embeddings"),
            codebase_dimension=384,
            codebase_batch_size=32,
            codebase_structural_indexing_enabled=False,
            codebase_chunking_strategy="tree_sitter_structural",
            codebase_chunk_size=500,
            codebase_chunk_overlap=50,
            codebase_embedding_extra_config={},
            codebase_graph_store="sqlite",
            codebase_graph_path=None,
            unified_embedding_model="BAAI/bge-small-en-v1.5",
        )

        cached_index = SimpleNamespace(incremental_reindex=AsyncMock())
        replacement_index = SimpleNamespace(index_codebase=AsyncMock(), _is_indexed=False)
        index_manifest = build_codebase_index_manifest(
            _build_codebase_embedding_config(settings, root)
        )
        fake_cache: dict[str, dict[str, object]] = {
            str(root): {
                "index": cached_index,
                "latest_mtime": _latest_mtime(root),
                "indexed_at": time.time(),
                "index_manifest": index_manifest,
                "watcher_subscribed": True,
                "stale": True,
            }
        }
        fake_factory = SimpleNamespace(create=lambda **kwargs: replacement_index)

        import victor.core.capability_registry as capability_registry_module
        import victor.core.indexing.index_lock as index_lock_module
        import victor.tools.code_search_tool as code_search_tool_module

        monkeypatch.setattr(
            capability_registry_module.CapabilityRegistry,
            "get_instance",
            staticmethod(lambda: _FakeCapabilityRegistry(fake_factory)),
        )
        monkeypatch.setattr(
            index_lock_module.IndexLockRegistry,
            "get_instance",
            staticmethod(lambda: _FakeIndexLockRegistry()),
        )
        monkeypatch.setattr(
            code_search_tool_module, "_get_index_cache", lambda exec_ctx=None: fake_cache
        )
        monkeypatch.setattr(_get_or_build_index, "_failure_cache", {}, raising=False)

        clear_index_cache()
        index, rebuilt = await _get_or_build_index(root=root, settings=settings)

        assert index is replacement_index
        assert rebuilt is True
        cached_index.incremental_reindex.assert_not_awaited()
        replacement_index.index_codebase.assert_awaited_once()
        assert fake_cache[str(root)]["index"] is replacement_index
        assert fake_cache[str(root)].get("stale") is not True

    @pytest.mark.asyncio
    async def test_get_or_build_index_finalizes_structural_provider_after_incremental_reindex(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "repo"
        root.mkdir()
        (root / "main.py").write_text("print('hello')\n", encoding="utf-8")

        settings = SimpleNamespace(
            codebase_vector_store="lancedb",
            codebase_embedding_provider="sentence-transformers",
            codebase_embedding_model="BAAI/bge-small-en-v1.5",
            codebase_persist_directory=str(tmp_path / "embeddings"),
            codebase_dimension=384,
            codebase_batch_size=32,
            codebase_structural_indexing_enabled=False,
            codebase_chunking_strategy="tree_sitter_structural",
            codebase_chunk_size=500,
            codebase_chunk_overlap=50,
            codebase_embedding_extra_config={},
            codebase_graph_store="sqlite",
            codebase_graph_path=None,
            unified_embedding_model="BAAI/bge-small-en-v1.5",
        )

        provider = SimpleNamespace(
            config=SimpleNamespace(vector_store="victor_structural_bridge"),
            get_stats=AsyncMock(return_value={"total_documents": 1}),
        )
        cached_index = SimpleNamespace(
            incremental_reindex=AsyncMock(),
            embedding_provider=provider,
        )
        index_manifest = build_codebase_index_manifest(
            _build_codebase_embedding_config(settings, root)
        )
        fake_cache: dict[str, dict[str, object]] = {
            str(root): {
                "index": cached_index,
                "latest_mtime": 0.0,
                "indexed_at": time.time(),
                "index_manifest": index_manifest,
                "watcher_subscribed": False,
            }
        }

        factory = MagicMock()
        subscribe_mock = AsyncMock()
        fake_factory = SimpleNamespace(create=factory)

        import victor.core.capability_registry as capability_registry_module
        import victor.core.indexing.index_lock as index_lock_module
        import victor.tools.code_search_tool as code_search_tool_module

        monkeypatch.setattr(
            capability_registry_module.CapabilityRegistry,
            "get_instance",
            staticmethod(lambda: _FakeCapabilityRegistry(fake_factory)),
        )
        monkeypatch.setattr(
            index_lock_module.IndexLockRegistry,
            "get_instance",
            staticmethod(lambda: _FakeIndexLockRegistry()),
        )
        monkeypatch.setattr(
            code_search_tool_module, "_get_index_cache", lambda exec_ctx=None: fake_cache
        )
        monkeypatch.setattr(code_search_tool_module, "_subscribe_to_file_watcher", subscribe_mock)
        monkeypatch.setattr(_get_or_build_index, "_failure_cache", {}, raising=False)

        clear_index_cache()
        index, rebuilt = await _get_or_build_index(root=root, settings=settings)

        assert index is cached_index
        assert rebuilt is False
        factory.assert_not_called()
        cached_index.incremental_reindex.assert_awaited_once()
        provider.get_stats.assert_awaited_once()
        subscribe_mock.assert_awaited_once()
        assert fake_cache[str(root)]["watcher_subscribed"] is True

    @pytest.mark.asyncio
    async def test_get_or_build_index_rebuilds_when_manifest_mismatch(self, tmp_path, monkeypatch):
        root = tmp_path / "repo"
        root.mkdir()
        (root / "main.py").write_text("print('hello')\n", encoding="utf-8")

        persist_dir = tmp_path / "embeddings"
        persist_dir.mkdir()
        (persist_dir / "existing.lance").mkdir()
        write_codebase_index_manifest(persist_dir, {"schema_version": -1})

        settings = SimpleNamespace(
            codebase_vector_store="lancedb",
            codebase_embedding_provider="sentence-transformers",
            codebase_embedding_model="BAAI/bge-small-en-v1.5",
            codebase_persist_directory=str(persist_dir),
            codebase_dimension=384,
            codebase_batch_size=32,
            codebase_structural_indexing_enabled=False,
            codebase_chunking_strategy="tree_sitter_structural",
            codebase_chunk_size=500,
            codebase_chunk_overlap=50,
            codebase_embedding_extra_config={},
            codebase_graph_store="sqlite",
            codebase_graph_path=None,
            unified_embedding_model="BAAI/bge-small-en-v1.5",
        )

        mock_index = SimpleNamespace(index_codebase=AsyncMock(), _is_indexed=False)
        fake_factory = SimpleNamespace(create=lambda **kwargs: mock_index)
        fake_cache: dict[str, dict[str, object]] = {}

        import victor.core.capability_registry as capability_registry_module
        import victor.core.indexing.index_lock as index_lock_module
        import victor.tools.code_search_tool as code_search_tool_module

        monkeypatch.setattr(
            capability_registry_module.CapabilityRegistry,
            "get_instance",
            staticmethod(lambda: _FakeCapabilityRegistry(fake_factory)),
        )
        monkeypatch.setattr(
            index_lock_module.IndexLockRegistry,
            "get_instance",
            staticmethod(lambda: _FakeIndexLockRegistry()),
        )
        monkeypatch.setattr(
            code_search_tool_module, "_get_index_cache", lambda exec_ctx=None: fake_cache
        )

        clear_index_cache()
        index, rebuilt = await _get_or_build_index(root=root, settings=settings)

        expected_manifest = build_codebase_index_manifest(
            _build_codebase_embedding_config(settings, root)
        )
        assert index is mock_index
        assert rebuilt is True
        mock_index.index_codebase.assert_awaited_once()
        assert fake_cache[str(root)]["index"] is mock_index
        assert (
            build_codebase_index_manifest(_build_codebase_embedding_config(settings, root))
            == expected_manifest
        )

    @pytest.mark.asyncio
    async def test_get_or_build_index_reuses_persistent_index_when_manifest_matches(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "repo"
        root.mkdir()
        (root / "main.py").write_text("print('hello')\n", encoding="utf-8")

        persist_dir = tmp_path / "embeddings"
        persist_dir.mkdir()
        (persist_dir / "existing.lance").mkdir()

        settings = SimpleNamespace(
            codebase_vector_store="lancedb",
            codebase_embedding_provider="sentence-transformers",
            codebase_embedding_model="BAAI/bge-small-en-v1.5",
            codebase_persist_directory=str(persist_dir),
            codebase_dimension=384,
            codebase_batch_size=32,
            codebase_structural_indexing_enabled=False,
            codebase_chunking_strategy="tree_sitter_structural",
            codebase_chunk_size=500,
            codebase_chunk_overlap=50,
            codebase_embedding_extra_config={},
            codebase_graph_store="sqlite",
            codebase_graph_path=None,
            unified_embedding_model="BAAI/bge-small-en-v1.5",
        )

        embedding_config = _build_codebase_embedding_config(settings, root)
        write_codebase_index_manifest(persist_dir, build_codebase_index_manifest(embedding_config))

        mock_index = SimpleNamespace(index_codebase=AsyncMock(), _is_indexed=False)
        fake_factory = SimpleNamespace(create=lambda **kwargs: mock_index)
        fake_cache: dict[str, dict[str, object]] = {}
        probe_mock = AsyncMock(return_value=False)

        import victor.core.capability_registry as capability_registry_module
        import victor.core.indexing.index_lock as index_lock_module
        import victor.tools.code_search_tool as code_search_tool_module

        monkeypatch.setattr(
            capability_registry_module.CapabilityRegistry,
            "get_instance",
            staticmethod(lambda: _FakeCapabilityRegistry(fake_factory)),
        )
        monkeypatch.setattr(
            index_lock_module.IndexLockRegistry,
            "get_instance",
            staticmethod(lambda: _FakeIndexLockRegistry()),
        )
        monkeypatch.setattr(
            code_search_tool_module, "_get_index_cache", lambda exec_ctx=None: fake_cache
        )
        monkeypatch.setattr(code_search_tool_module, "_probe_index_integrity", probe_mock)

        clear_index_cache()
        index, rebuilt = await _get_or_build_index(root=root, settings=settings)

        assert index is mock_index
        assert rebuilt is False
        assert mock_index._is_indexed is True
        mock_index.index_codebase.assert_not_awaited()
        probe_mock.assert_awaited_once_with(mock_index)

    @pytest.mark.asyncio
    async def test_get_or_build_index_marks_persisted_cache_stale_when_integrity_rebuild_fails(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "repo"
        root.mkdir()
        (root / "main.py").write_text("print('hello')\n", encoding="utf-8")

        persist_dir = tmp_path / "embeddings"
        persist_dir.mkdir()
        (persist_dir / "existing.lance").mkdir()

        settings = SimpleNamespace(
            codebase_vector_store="lancedb",
            codebase_embedding_provider="sentence-transformers",
            codebase_embedding_model="BAAI/bge-small-en-v1.5",
            codebase_persist_directory=str(persist_dir),
            codebase_dimension=384,
            codebase_batch_size=32,
            codebase_structural_indexing_enabled=False,
            codebase_chunking_strategy="tree_sitter_structural",
            codebase_chunk_size=500,
            codebase_chunk_overlap=50,
            codebase_embedding_extra_config={},
            codebase_graph_store="sqlite",
            codebase_graph_path=None,
            unified_embedding_model="BAAI/bge-small-en-v1.5",
        )

        embedding_config = _build_codebase_embedding_config(settings, root)
        write_codebase_index_manifest(persist_dir, build_codebase_index_manifest(embedding_config))

        mock_index = SimpleNamespace(index_codebase=AsyncMock(), _is_indexed=False)
        fake_factory = SimpleNamespace(create=lambda **kwargs: mock_index)
        fake_cache: dict[str, dict[str, object]] = {}

        async def _stale_probe(index):
            index._is_indexed = False
            return IntegrityProbeOutcome(stale=True)

        probe_mock = AsyncMock(side_effect=_stale_probe)

        import victor.core.capability_registry as capability_registry_module
        import victor.core.indexing.index_lock as index_lock_module
        import victor.tools.code_search_tool as code_search_tool_module

        monkeypatch.setattr(
            capability_registry_module.CapabilityRegistry,
            "get_instance",
            staticmethod(lambda: _FakeCapabilityRegistry(fake_factory)),
        )
        monkeypatch.setattr(
            index_lock_module.IndexLockRegistry,
            "get_instance",
            staticmethod(lambda: _FakeIndexLockRegistry()),
        )
        monkeypatch.setattr(
            code_search_tool_module, "_get_index_cache", lambda exec_ctx=None: fake_cache
        )
        monkeypatch.setattr(code_search_tool_module, "_probe_index_integrity", probe_mock)

        clear_index_cache()
        index, rebuilt = await _get_or_build_index(root=root, settings=settings)

        assert index is mock_index
        assert rebuilt is False
        assert fake_cache[str(root)]["stale"] is True
        assert mock_index._is_indexed is False
        mock_index.index_codebase.assert_not_awaited()
        probe_mock.assert_awaited_once_with(mock_index)

    @pytest.mark.asyncio
    async def test_get_or_build_index_ignores_failure_cache_from_different_manifest(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "repo"
        root.mkdir()
        (root / "main.py").write_text("print('hello')\n", encoding="utf-8")

        persist_dir = tmp_path / "embeddings"
        persist_dir.mkdir()
        (persist_dir / "existing.lance").mkdir()

        settings = SimpleNamespace(
            codebase_vector_store="lancedb",
            codebase_embedding_provider="sentence-transformers",
            codebase_embedding_model="BAAI/bge-small-en-v1.5",
            codebase_persist_directory=str(persist_dir),
            codebase_dimension=384,
            codebase_batch_size=32,
            codebase_structural_indexing_enabled=False,
            codebase_chunking_strategy="tree_sitter_structural",
            codebase_chunk_size=500,
            codebase_chunk_overlap=50,
            codebase_embedding_extra_config={},
            codebase_graph_store="sqlite",
            codebase_graph_path=None,
            unified_embedding_model="BAAI/bge-small-en-v1.5",
        )
        new_manifest = build_codebase_index_manifest(
            _build_codebase_embedding_config(settings, root)
        )
        write_codebase_index_manifest(persist_dir, new_manifest)

        old_settings = SimpleNamespace(**vars(settings))
        old_settings.codebase_chunking_strategy = "symbol_span"
        old_manifest = build_codebase_index_manifest(
            _build_codebase_embedding_config(old_settings, root)
        )

        failure_cache = {
            _build_index_failure_key(root, old_manifest): {
                "error": "old config failed",
                "timestamp": time.time(),
            }
        }
        mock_index = SimpleNamespace(index_codebase=AsyncMock(), _is_indexed=False)
        fake_factory = SimpleNamespace(create=lambda **kwargs: mock_index)
        fake_cache: dict[str, dict[str, object]] = {}
        probe_mock = AsyncMock(return_value=False)

        import victor.core.capability_registry as capability_registry_module
        import victor.core.indexing.index_lock as index_lock_module
        import victor.tools.code_search_tool as code_search_tool_module

        monkeypatch.setattr(
            capability_registry_module.CapabilityRegistry,
            "get_instance",
            staticmethod(lambda: _FakeCapabilityRegistry(fake_factory)),
        )
        monkeypatch.setattr(
            index_lock_module.IndexLockRegistry,
            "get_instance",
            staticmethod(lambda: _FakeIndexLockRegistry()),
        )
        monkeypatch.setattr(
            code_search_tool_module, "_get_index_cache", lambda exec_ctx=None: fake_cache
        )
        monkeypatch.setattr(_get_or_build_index, "_failure_cache", failure_cache, raising=False)
        monkeypatch.setattr(code_search_tool_module, "_probe_index_integrity", probe_mock)

        clear_index_cache()
        index, rebuilt = await _get_or_build_index(root=root, settings=settings)

        assert index is mock_index
        assert rebuilt is False
        probe_mock.assert_awaited_once_with(mock_index)

    @pytest.mark.asyncio
    async def test_get_or_build_index_clears_plain_dict_failure_cache_after_success(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "repo"
        root.mkdir()
        (root / "main.py").write_text("print('hello')\n", encoding="utf-8")

        persist_dir = tmp_path / "embeddings"
        persist_dir.mkdir()
        (persist_dir / "existing.lance").mkdir()

        settings = SimpleNamespace(
            codebase_vector_store="lancedb",
            codebase_embedding_provider="sentence-transformers",
            codebase_embedding_model="BAAI/bge-small-en-v1.5",
            codebase_persist_directory=str(persist_dir),
            codebase_dimension=384,
            codebase_batch_size=32,
            codebase_structural_indexing_enabled=False,
            codebase_chunking_strategy="tree_sitter_structural",
            codebase_chunk_size=500,
            codebase_chunk_overlap=50,
            codebase_embedding_extra_config={},
            codebase_graph_store="sqlite",
            codebase_graph_path=None,
            unified_embedding_model="BAAI/bge-small-en-v1.5",
        )
        manifest = build_codebase_index_manifest(_build_codebase_embedding_config(settings, root))
        write_codebase_index_manifest(persist_dir, manifest)

        failure_key = _build_index_failure_key(root, manifest)
        failure_cache = {
            failure_key: {
                "error": "temporary failure",
                "timestamp": time.time(),
            }
        }
        mock_index = SimpleNamespace(index_codebase=AsyncMock(), _is_indexed=False)
        fake_factory = SimpleNamespace(create=lambda **kwargs: mock_index)
        fake_cache: dict[str, dict[str, object]] = {}
        probe_mock = AsyncMock(return_value=False)

        import victor.core.capability_registry as capability_registry_module
        import victor.core.indexing.index_lock as index_lock_module
        import victor.tools.code_search_tool as code_search_tool_module

        monkeypatch.setattr(
            capability_registry_module.CapabilityRegistry,
            "get_instance",
            staticmethod(lambda: _FakeCapabilityRegistry(fake_factory)),
        )
        monkeypatch.setattr(
            index_lock_module.IndexLockRegistry,
            "get_instance",
            staticmethod(lambda: _FakeIndexLockRegistry()),
        )
        monkeypatch.setattr(
            code_search_tool_module, "_get_index_cache", lambda exec_ctx=None: fake_cache
        )
        monkeypatch.setattr(_get_or_build_index, "_failure_cache", failure_cache, raising=False)
        monkeypatch.setattr(code_search_tool_module, "_probe_index_integrity", probe_mock)

        clear_index_cache()
        index, rebuilt = await _get_or_build_index(root=root, settings=settings, force_reindex=True)

        assert index is mock_index
        assert rebuilt is True
        assert failure_key not in failure_cache

    @pytest.mark.asyncio
    async def test_finalize_index_storage_flushes_structural_provider(self):
        provider = SimpleNamespace(
            config=SimpleNamespace(vector_store="victor_structural_bridge"),
            get_stats=AsyncMock(return_value={"total_documents": 1}),
        )
        index = SimpleNamespace(embedding_provider=provider)

        await _finalize_index_storage(index)

        provider.get_stats.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_finalize_index_storage_skips_non_structural_provider(self):
        provider = SimpleNamespace(
            config=SimpleNamespace(vector_store="lancedb"),
            get_stats=AsyncMock(return_value={"total_documents": 1}),
        )
        index = SimpleNamespace(embedding_provider=provider)

        await _finalize_index_storage(index)

        provider.get_stats.assert_not_awaited()


class TestFileWatcherIncrementalUpdates:
    """Tests for watcher-driven incremental code_search refreshes."""

    @pytest.mark.asyncio
    async def test_schedule_file_change_refresh_swallows_handler_exceptions(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "repo"
        root.mkdir()
        changed_file = root / "main.py"
        changed_file.write_text("print('hello')\n", encoding="utf-8")

        import victor.tools.code_search_tool as code_search_tool_module

        async def _raise(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(code_search_tool_module, "_on_file_change", _raise)

        task = _schedule_file_change_refresh(
            FileChangeEvent(
                path=changed_file,
                change_type=FileChangeType.MODIFIED,
                timestamp=datetime.now(),
            ),
            root,
        )

        await task
        assert task.exception() is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("change_type", "mutate_paths"),
        [
            (FileChangeType.MODIFIED, "modify"),
            (FileChangeType.CREATED, "create"),
            (FileChangeType.DELETED, "delete"),
            (FileChangeType.RENAMED, "rename"),
        ],
    )
    async def test_on_file_change_incrementally_updates_supported_change_types(
        self, tmp_path, monkeypatch, change_type, mutate_paths
    ):
        root = tmp_path / "repo"
        root.mkdir()
        original_file = root / "main.py"
        original_file.write_text("print('hello')\n", encoding="utf-8")
        changed_file = original_file
        old_path = None

        if mutate_paths == "modify":
            original_file.write_text("print('updated')\n", encoding="utf-8")
        elif mutate_paths == "create":
            changed_file = root / "created.py"
            changed_file.write_text("print('created')\n", encoding="utf-8")
        elif mutate_paths == "delete":
            original_file.unlink()
        elif mutate_paths == "rename":
            changed_file = root / "renamed.py"
            original_file.rename(changed_file)
            old_path = original_file

        provider = SimpleNamespace(
            config=SimpleNamespace(vector_store="victor_structural_bridge"),
            get_stats=AsyncMock(return_value={"total_documents": 1}),
        )
        index = SimpleNamespace(
            incremental_reindex=AsyncMock(),
            embedding_provider=provider,
        )
        cache_entry = {
            "index": index,
            "latest_mtime": 0.0,
            "indexed_at": 0.0,
            "stale": True,
        }
        fake_cache = {str(root): cache_entry}

        import victor.tools.code_search_tool as code_search_tool_module

        monkeypatch.setattr(
            code_search_tool_module, "_get_index_cache", lambda exec_ctx=None: fake_cache
        )

        await _on_file_change(
            FileChangeEvent(
                path=changed_file,
                change_type=change_type,
                timestamp=datetime.now(),
                old_path=old_path,
            ),
            root,
        )

        index.incremental_reindex.assert_awaited_once()
        provider.get_stats.assert_awaited_once()
        assert cache_entry["latest_mtime"] == _latest_mtime(root)
        assert cache_entry["indexed_at"] > 0.0
        assert cache_entry["stale"] is False

    @pytest.mark.asyncio
    async def test_on_file_change_marks_cache_stale_when_incremental_update_fails(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "repo"
        root.mkdir()
        changed_file = root / "main.py"
        changed_file.write_text("print('hello')\n", encoding="utf-8")

        index = SimpleNamespace(
            incremental_reindex=AsyncMock(side_effect=RuntimeError("boom")),
        )
        cache_entry = {
            "index": index,
            "latest_mtime": _latest_mtime(root),
            "stale": False,
        }
        fake_cache = {str(root): cache_entry}

        import victor.tools.code_search_tool as code_search_tool_module

        monkeypatch.setattr(
            code_search_tool_module, "_get_index_cache", lambda exec_ctx=None: fake_cache
        )

        await _on_file_change(
            FileChangeEvent(
                path=changed_file,
                change_type=FileChangeType.MODIFIED,
                timestamp=datetime.now(),
            ),
            root,
        )

        index.incremental_reindex.assert_awaited_once()
        assert cache_entry["stale"] is True

    @pytest.mark.asyncio
    async def test_on_file_change_acquires_index_lock_before_incremental_reindex(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "repo"
        root.mkdir()
        changed_file = root / "main.py"
        changed_file.write_text("print('hello')\n", encoding="utf-8")

        tracking_registry = _TrackingIndexLockRegistry()

        async def _assert_locked() -> None:
            assert tracking_registry.in_lock is True

        index = SimpleNamespace(
            incremental_reindex=AsyncMock(side_effect=_assert_locked),
        )
        cache_entry = {
            "index": index,
            "latest_mtime": 0.0,
            "stale": True,
        }
        fake_cache = {str(root): cache_entry}

        import victor.core.indexing.index_lock as index_lock_module
        import victor.tools.code_search_tool as code_search_tool_module

        monkeypatch.setattr(
            index_lock_module.IndexLockRegistry,
            "get_instance",
            staticmethod(lambda: tracking_registry),
        )
        monkeypatch.setattr(
            code_search_tool_module, "_get_index_cache", lambda exec_ctx=None: fake_cache
        )

        await _on_file_change(
            FileChangeEvent(
                path=changed_file,
                change_type=FileChangeType.MODIFIED,
                timestamp=datetime.now(),
            ),
            root,
        )

        index.incremental_reindex.assert_awaited_once()
        assert tracking_registry.used_paths == [root]
