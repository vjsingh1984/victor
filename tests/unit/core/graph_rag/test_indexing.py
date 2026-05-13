from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.core.graph_rag.config import GraphIndexConfig
from victor.core.graph_rag.indexing import (
    GraphIndexingPipeline,
    GraphIndexStats,
    run_indexing_with_lock,
)
from victor.storage.graph.protocol import GraphEdge, GraphNode


class _RecordingGraphStore:
    def __init__(self) -> None:
        self.in_write_batch = False
        self.write_batch_entries = 0
        self._write_batch_depth = 0
        self.calls: list[tuple[str, bool, int | None]] = []

    @asynccontextmanager
    async def write_batch(self):
        is_outermost = self._write_batch_depth == 0
        if is_outermost:
            self.in_write_batch = True
            self.write_batch_entries += 1
        self._write_batch_depth += 1
        try:
            yield
        finally:
            self._write_batch_depth -= 1
            if self._write_batch_depth == 0:
                self.in_write_batch = False

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def upsert_nodes(self, nodes):
        rows = list(nodes)
        self.calls.append(("nodes", self.in_write_batch, len(rows)))

    async def upsert_edges(self, edges):
        rows = list(edges)
        self.calls.append(("edges", self.in_write_batch, len(rows)))

    async def update_file_mtime(self, file: str, mtime: float) -> None:
        self.calls.append(("mtime", self.in_write_batch, None))

    async def delete_by_file(self, file: str) -> None:
        self.calls.append(("delete", self.in_write_batch, None))

    async def get_stale_files(self, file_mtimes):
        return []

    async def get_all_nodes(self):
        return []


class _FakeNode:
    def __init__(
        self,
        node_type: str,
        *,
        text: bytes | str = b"",
        named_children: list["_FakeNode"] | None = None,
        field_children: dict[str, "_FakeNode"] | None = None,
    ) -> None:
        self.type = node_type
        self.text = text
        self.named_children = named_children or []
        self.children = self.named_children
        self._field_children = field_children or {}

    def child_by_field_name(self, name: str):
        return self._field_children.get(name)


@pytest.mark.asyncio
async def test_graph_indexing_pipeline_uses_store_write_batch_for_file_writes(
    monkeypatch, tmp_path: Path
):
    file_path = tmp_path / "sample.py"
    file_path.write_text("def foo():\n    return 1\n", encoding="utf-8")

    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    async def _fake_extract_symbols(_file_path: Path, _language: str):
        return [
            GraphNode(
                node_id="symbol:sample.py:foo",
                type="function",
                name="foo",
                file=str(_file_path),
                line=1,
                lang="python",
            )
        ]

    async def _fake_build_symbol_edges(_nodes, _file_path: Path):
        return [
            GraphEdge(
                src="file:sample.py",
                dst="symbol:sample.py:foo",
                type="CONTAINS",
            )
        ]

    monkeypatch.setattr(pipeline, "_extract_symbols", _fake_extract_symbols)
    monkeypatch.setattr(pipeline, "_build_symbol_edges", _fake_build_symbol_edges)

    stats = await pipeline._process_file(file_path)

    assert stats.files_processed == 1
    assert graph_store.write_batch_entries == 1
    assert graph_store.calls == [
        ("nodes", True, 1),
        ("edges", True, 1),
        ("mtime", True, None),
    ]


def test_extract_name_from_node_prefers_direct_name_field(tmp_path: Path):
    pipeline = GraphIndexingPipeline(
        _RecordingGraphStore(),
        GraphIndexConfig(
            root_path=tmp_path,
            enable_ccg=False,
            enable_embeddings=False,
            enable_subgraph_cache=False,
        ),
    )

    name_node = _FakeNode("identifier", text=b"fast_name")
    method_node = _FakeNode(
        "function_definition",
        named_children=[name_node],
        field_children={"name": name_node},
    )

    assert pipeline._extract_name_from_node(method_node) == "fast_name"


@pytest.mark.asyncio
async def test_graph_indexing_pipeline_batches_multi_file_writes_with_single_store_transaction(
    monkeypatch, tmp_path: Path
):
    first_file = tmp_path / "first.py"
    second_file = tmp_path / "second.py"
    first_file.write_text("def first():\n    return 1\n", encoding="utf-8")
    second_file.write_text("def second():\n    return 2\n", encoding="utf-8")

    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    async def _fake_extract_symbols(file_path: Path, _language: str):
        stem = file_path.stem
        return [
            GraphNode(
                node_id=f"symbol:{stem}.py:{stem}",
                type="function",
                name=stem,
                file=str(file_path),
                line=1,
                lang="python",
            )
        ]

    async def _fake_build_symbol_edges(_nodes, file_path: Path):
        stem = file_path.stem
        return [
            GraphEdge(
                src=f"file:{stem}.py",
                dst=f"symbol:{stem}.py:{stem}",
                type="CONTAINS",
            )
        ]

    monkeypatch.setattr(pipeline, "_extract_symbols", _fake_extract_symbols)
    monkeypatch.setattr(pipeline, "_build_symbol_edges", _fake_build_symbol_edges)

    stats = await pipeline._process_batch([first_file, second_file])

    assert stats.files_processed == 2
    assert graph_store.write_batch_entries == 1
    assert graph_store.calls == [
        ("nodes", True, 1),
        ("edges", True, 1),
        ("mtime", True, None),
        ("nodes", True, 1),
        ("edges", True, 1),
        ("mtime", True, None),
    ]


@pytest.mark.asyncio
async def test_graph_indexing_pipeline_cleans_vanished_file_without_error(tmp_path: Path):
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    missing_file = tmp_path / "missing.py"
    stats = await pipeline._process_file(missing_file)

    assert stats.files_processed == 0
    assert stats.files_deleted == 1
    assert stats.error_count == 0
    assert graph_store.calls == [("delete", False, None)]


@pytest.mark.asyncio
async def test_graph_indexing_pipeline_excludes_root_level_coverage_temp_files(tmp_path: Path):
    (tmp_path / "module.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / ".coverage.Vijays-MacBook-Pro.local.69421.XNBbPQdx.c").write_text(
        "not really C source", encoding="utf-8"
    )

    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
        respect_gitignore=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    files = await pipeline._discover_files(tmp_path)

    assert files == [tmp_path / "module.py"]


@pytest.mark.asyncio
async def test_graph_indexing_pipeline_refreshes_module_metrics_after_changed_graph(
    monkeypatch,
    tmp_path: Path,
):
    store = MagicMock()
    store.initialize = AsyncMock()
    store.delete_by_repo = AsyncMock()
    events: list[tuple[str, object]] = []

    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
        incremental=False,
    )
    pipeline = GraphIndexingPipeline(store, config)
    monkeypatch.setattr(
        pipeline,
        "_discover_files",
        AsyncMock(return_value=[tmp_path / "sample.py"]),
    )
    monkeypatch.setattr(
        pipeline,
        "_process_batch",
        AsyncMock(return_value=GraphIndexStats(files_processed=1, nodes_created=1)),
    )

    class _FakeModuleAnalyzer:
        def __init__(self, *, project_path):
            events.append(("init", project_path))

        def compute_all(self):
            events.append(("compute", None))
            return ["metric"]

        def persist(self, metrics):
            events.append(("persist", metrics))

    monkeypatch.setattr(
        "victor.core.analysis.module_analyzer.ModuleAnalyzer",
        _FakeModuleAnalyzer,
    )

    stats = await pipeline.index_repository()

    assert stats.module_metrics_computed == 1
    assert events == [
        ("init", tmp_path.resolve()),
        ("compute", None),
        ("persist", ["metric"]),
    ]


@pytest.mark.asyncio
async def test_graph_indexing_pipeline_ignores_files_that_vanish_before_incremental_planning(
    tmp_path: Path,
):
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    missing_file = tmp_path / ".coverage.Vijays-MacBook-Pro.local.69421.XNBbPQdx.c"

    stats = await pipeline._prepare_incremental_work([missing_file])

    assert stats.files_deleted == 0
    assert stats.files_unchanged == 0
    assert pipeline._files_to_process == set()
    assert graph_store.calls == [("delete", False, None)]


@pytest.mark.asyncio
async def test_run_indexing_with_lock_uses_project_index_lock(monkeypatch, tmp_path: Path):
    lock_events: list[str] = []

    class _FakePathLock:
        async def __aenter__(self):
            lock_events.append("enter")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            lock_events.append("exit")
            return False

    class _FakeRegistry:
        async def acquire_lock(self, path, use_file_lock=True, timeout_seconds=300.0):
            assert use_file_lock is True
            assert timeout_seconds == 300.0
            lock_events.append(str(path))
            return _FakePathLock()

    async def _operation():
        lock_events.append("operation")
        return "indexed"

    monkeypatch.setattr(
        "victor.core.indexing.index_lock.IndexLockRegistry.get_instance",
        lambda: _FakeRegistry(),
    )

    result = await run_indexing_with_lock(tmp_path, _operation)

    assert result == "indexed"
    assert lock_events == [str(tmp_path.resolve()), "enter", "operation", "exit"]
