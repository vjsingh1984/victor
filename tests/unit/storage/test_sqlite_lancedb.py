"""Tests for SqliteLanceDBStore initialization and helpers."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.storage.unified.sqlite_lancedb import SqliteLanceDBStore
from victor.storage.unified.protocol import UnifiedId


@pytest.fixture
def store(tmp_path):
    return SqliteLanceDBStore(
        repo_root=tmp_path,
        persist_directory=tmp_path / ".victor",
    )


class TestInit:
    def test_defaults(self, store, tmp_path):
        assert store.repo_root == tmp_path.resolve()
        assert store._initialized is False
        assert store._graph_store is None
        assert store._vector_store is None

    def test_custom_persist_directory(self, tmp_path):
        custom = tmp_path / "custom_dir"
        s = SqliteLanceDBStore(repo_root=tmp_path, persist_directory=custom)
        assert s.persist_directory == custom

    @pytest.mark.asyncio
    async def test_initialize_uses_custom_project_db_path(self, tmp_path):
        custom = tmp_path / "custom_dir"
        store = SqliteLanceDBStore(repo_root=tmp_path, persist_directory=custom)

        fake_graph_store = MagicMock()
        fake_graph_store.initialize = AsyncMock()

        class _FakeEmbeddingModel:
            async def initialize(self):
                return None

        class _FakeFactory:
            def create_model(self, *_args, **_kwargs):
                return _FakeEmbeddingModel()

        class _FakeRegistry:
            def get(self, *_args, **_kwargs):
                return _FakeFactory()

        with (
            patch("victor.storage.graph.sqlite_store.SqliteGraphStore", return_value=fake_graph_store) as mock_graph_store,
            patch("victor.core.capability_registry.CapabilityRegistry.get_instance", return_value=_FakeRegistry()),
            patch.object(store, "_init_vector_store", AsyncMock()),
        ):
            await store.initialize()

        graph_path_arg = mock_graph_store.call_args.args[0]
        assert Path(graph_path_arg) == custom / "project.db"
        fake_graph_store.initialize.assert_awaited_once()


class TestUnifiedIdHelpers:
    def test_make_symbol_id(self, store):
        sid = store.make_symbol_id("src/foo.py", "MyClass")
        assert "src/foo.py" in sid
        assert "MyClass" in sid

    def test_make_file_id(self, store):
        fid = store.make_file_id("src/foo.py")
        assert "src/foo.py" in fid

    def test_parse_id_roundtrip(self, store):
        sid = store.make_symbol_id("src/foo.py", "bar")
        parsed = store.parse_id(sid)
        assert isinstance(parsed, UnifiedId)


class TestClose:
    async def test_close_resets_state(self, store):
        store._initialized = True
        store._graph_store = MagicMock()
        store._graph_store.close = AsyncMock()
        store._embedding_model = MagicMock()
        store._embedding_model.close = AsyncMock()
        store._vector_store = MagicMock()
        store._vector_table = MagicMock()

        await store.close()

        assert store._initialized is False
        assert store._graph_store is None
        assert store._embedding_model is None
        assert store._vector_store is None
        assert store._vector_table is None


class TestCombineScores:
    def test_semantic_only(self, store):
        score = store._combine_scores(
            semantic=0.8,
            keyword=None,
            graph=None,
            semantic_weight=0.6,
            graph_weight=0.2,
        )
        assert 0.7 < score < 0.9

    def test_all_scores(self, store):
        score = store._combine_scores(
            semantic=1.0,
            keyword=1.0,
            graph=1.0,
            semantic_weight=0.5,
            graph_weight=0.3,
        )
        assert 0.9 < score <= 1.0

    def test_no_scores(self, store):
        score = store._combine_scores(
            semantic=None,
            keyword=None,
            graph=None,
            semantic_weight=0.5,
            graph_weight=0.3,
        )
        assert score == 0.0
