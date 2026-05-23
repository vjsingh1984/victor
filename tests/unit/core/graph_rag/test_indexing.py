from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.core.graph_rag.config import GraphIndexConfig
from victor.core.graph_rag.indexing import (
    GraphIndexingPipeline,
    GraphIndexStats,
    ParseResult,
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


def test_extract_name_from_node_uses_type_field_for_impl_blocks():
    """`impl Default for Foo` must extract as "Foo" (the implementing type),
    not "Default" (the trait). Tree-sitter exposes the implementing type via
    the ``type`` field; the trait via ``trait``. Without this, every
    Default::default() call binds to all 800k user-defined `fn default`
    methods because they're all keyed by impl_name="Default".

    Verifies the extractor consults child_by_field_name("type") for impl_item
    nodes before falling back to the first-identifier scan (which would
    return the trait name).
    """
    pytest.importorskip("tree_sitter_rust")
    import tree_sitter
    import tree_sitter_rust

    lang = tree_sitter.Language(tree_sitter_rust.language())
    parser = tree_sitter.Parser(lang)

    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=Path("."), enable_ccg=False, enable_embeddings=False, enable_subgraph_cache=False
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    cases = [
        (b"impl Foo { fn a() {} }", "Foo"),
        (b"impl Default for Foo { fn default() {} }", "Foo"),
        (b"impl<T> Display for Bar<T> { fn fmt() {} }", "Bar"),
    ]
    for src, expected in cases:
        tree = parser.parse(src)
        impl_node = next(c for c in tree.root_node.children if c.type == "impl_item")
        assert (
            pipeline._extract_name_from_node(impl_node) == expected
        ), f"src={src!r} expected={expected!r}"


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

    # Patch _parse_file_sync (runs in thread pool) — return pre-built symbol nodes.
    # CCG is disabled in config so ccg_nodes/ccg_edges are empty.
    def _fake_parse_file_sync(file_path: Path):
        stem = file_path.stem
        sym_node = GraphNode(
            node_id=f"symbol:{stem}.py:{stem}",
            type="function",
            name=stem,
            file=str(file_path),
            line=1,
            lang="python",
        )
        return ParseResult(
            file_path=file_path,
            language="python",
            symbol_nodes=[sym_node],
        )

    async def _fake_build_symbol_edges(_nodes, file_path: Path):
        stem = file_path.stem
        return [
            GraphEdge(
                src=f"file:{stem}.py",
                dst=f"symbol:{stem}.py:{stem}",
                type="CONTAINS",
            )
        ]

    monkeypatch.setattr(pipeline, "_parse_file_sync", _fake_parse_file_sync)
    monkeypatch.setattr(pipeline, "_build_symbol_edges", _fake_build_symbol_edges)

    stats = await pipeline._process_batch([first_file, second_file])

    assert stats.files_processed == 2
    # ONE write_batch transaction for the entire two-file batch
    assert graph_store.write_batch_entries == 1
    # Bulk write: all symbol nodes in one call, all edges in one call, then per-file mtimes
    assert graph_store.calls == [
        ("nodes", True, 2),  # both files' nodes merged into one upsert_nodes call
        ("edges", True, 2),  # both files' edges merged into one upsert_edges call
        ("mtime", True, None),  # first_file mtime
        ("mtime", True, None),  # second_file mtime
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
async def test_graph_indexing_pipeline_discovers_files_by_language_then_path(tmp_path: Path):
    (tmp_path / "z_python.py").write_text("def zed():\n    return 1\n", encoding="utf-8")
    (tmp_path / "a_typescript.ts").write_text("export const value = 1\n", encoding="utf-8")
    (tmp_path / "b_python.py").write_text("def bee():\n    return 2\n", encoding="utf-8")
    (tmp_path / "a_javascript.js").write_text("export const value = 1\n", encoding="utf-8")

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

    assert files == [
        tmp_path / "a_javascript.js",
        tmp_path / "b_python.py",
        tmp_path / "z_python.py",
        tmp_path / "a_typescript.ts",
    ]


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


class _FakeProjectDatabaseManager:
    """Stand-in for ProjectDatabaseManager.

    Returns ``name_rows`` for the project-wide leaf-name SELECT and
    ``impl_rows`` for the impl-type JOIN that powers receiver-typed
    resolution. Distinguishes the two by looking for ``JOIN`` in the query.

    ``name_rows`` entries may be 2-tuples ``(name, node_id)`` (legacy) or
    3-tuples ``(name, node_id, file)``. The resolver uses ``file`` for
    same-file preference in name-only resolution; 2-tuples are padded with
    None to keep older tests compatible.
    """

    def __init__(
        self,
        rows: list[tuple[str, str]] | None = None,
        *,
        name_rows: list[tuple] | None = None,
        impl_rows: list[tuple[str, str, str]] | None = None,
    ):
        raw = list(rows or name_rows or [])
        self._name_rows = [row if len(row) >= 3 else (*row, None) for row in raw]
        self._impl_rows = list(impl_rows or [])

    def __call__(self, _project_path):
        return self

    def _get_raw_connection(self):
        outer = self

        class _Conn:
            def execute(self, query):
                # Discriminator: the impl-type lookup selects "AS impl_type";
                # the leaf-name lookup never does. (Both queries now use JOIN
                # since the name query also LEFT-joins to filter trait
                # impls.)
                if "impl_type" in query:
                    return iter(outer._impl_rows)
                return iter(outer._name_rows)

        return _Conn()


@pytest.mark.asyncio
async def test_resolve_cross_file_calls_returns_zero_when_buffer_empty(tmp_path: Path):
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    assert emitted == 0
    assert graph_store.calls == []


@pytest.mark.asyncio
async def test_resolve_cross_file_calls_emits_edges_for_cross_file_callees(
    monkeypatch, tmp_path: Path
):
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    # Two raw call records: caller_a calls "shared_fn", caller_b calls "shared_fn".
    pipeline._pending_call_records = [
        ("caller_a", "shared_fn", None, False),
        ("caller_b", "shared_fn", None, False),
    ]

    # Project-wide name index has one callee, in a different file (cross-file).
    monkeypatch.setattr(
        "victor.core.database.ProjectDatabaseManager",
        _FakeProjectDatabaseManager([("shared_fn", "callee_x")]),
    )

    captured_edges: list[GraphEdge] = []

    async def _capture(edges):
        captured_edges.extend(edges)

    monkeypatch.setattr(graph_store, "upsert_edges", _capture)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    assert emitted == 2
    assert {(e.src, e.dst, e.type) for e in captured_edges} == {
        ("caller_a", "callee_x", "CALLS"),
        ("caller_b", "callee_x", "CALLS"),
    }
    # Buffer is drained after the pass so a subsequent index_repository starts clean.
    assert pipeline._pending_call_records == []


@pytest.mark.asyncio
async def test_resolve_cross_file_calls_skips_self_loops(monkeypatch, tmp_path: Path):
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)
    pipeline._pending_call_records = [("node_self", "recurse", None, False)]

    monkeypatch.setattr(
        "victor.core.database.ProjectDatabaseManager",
        _FakeProjectDatabaseManager([("recurse", "node_self")]),
    )

    captured: list[GraphEdge] = []
    monkeypatch.setattr(graph_store, "upsert_edges", lambda edges: captured.extend(edges))

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    assert emitted == 0
    assert captured == []


@pytest.mark.asyncio
async def test_resolve_cross_file_calls_respects_fanout_cap(monkeypatch, tmp_path: Path):
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    # Cap at 2 candidates per leaf name; ambiguous names beyond that skip resolution.
    config.cross_file_call_max_fanout = 2  # type: ignore[attr-defined]
    pipeline = GraphIndexingPipeline(graph_store, config)
    pipeline._pending_call_records = [
        ("caller", "popular", None, False),  # 3 candidates -> skipped (above cap)
        ("caller", "rare", None, False),  # 1 candidate  -> emitted
    ]

    monkeypatch.setattr(
        "victor.core.database.ProjectDatabaseManager",
        _FakeProjectDatabaseManager(
            [
                ("popular", "p1"),
                ("popular", "p2"),
                ("popular", "p3"),
                ("rare", "r1"),
            ]
        ),
    )

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    monkeypatch.setattr(graph_store, "upsert_edges", _capture)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    assert emitted == 1
    assert {(e.src, e.dst) for e in captured} == {("caller", "r1")}


def test_graph_index_stats_to_dict_includes_cross_file_calls_resolved():
    stats = GraphIndexStats(cross_file_calls_resolved=42)
    assert stats.to_dict()["cross_file_calls_resolved"] == 42


def test_tree_sitter_parser_cache_is_lru_one_per_language(monkeypatch, tmp_path: Path):
    """Switching languages evicts the prior language's parser (bounded memory)."""
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)

    constructed: list[str] = []

    def _fake_create(language: str):
        constructed.append(language)
        return MagicMock(name=f"parser-{language}")

    monkeypatch.setattr(pipeline, "_create_ts_parser", _fake_create)

    py = pipeline._get_tree_sitter_parser("python")
    # Same language -> cache hit, no new construction
    py_again = pipeline._get_tree_sitter_parser("python")
    assert py is py_again
    assert constructed == ["python"]
    assert list(pipeline._thread_local.parser_cache.keys()) == ["python"]

    # Switch language -> evict python, load rust
    rs = pipeline._get_tree_sitter_parser("rust")
    assert constructed == ["python", "rust"]
    assert list(pipeline._thread_local.parser_cache.keys()) == ["rust"]
    assert pipeline._thread_local.parser_cache["rust"] is rs

    # Switch back -> python must be reconstructed (it was evicted)
    pipeline._get_tree_sitter_parser("python")
    assert constructed == ["python", "rust", "python"]
    assert list(pipeline._thread_local.parser_cache.keys()) == ["python"]


# -----------------------------------------------------------------------------
# Receiver-typed resolution: a method call obj.method() with a known receiver
# type binds only to methods inside `impl T`, instead of fanning out to every
# `method` defined anywhere in the project.
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_filters_by_impl_type_when_receiver_known(monkeypatch, tmp_path: Path):
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)
    pipeline._pending_call_records = [("caller", "render", "Foo", True)]

    # Project-wide name index says "render" exists 4 times (above default fanout=25 we'd
    # still emit; below we keep it small to show impl-type filter wins anyway).
    name_rows = [
        ("render", "render_foo_id"),
        ("render", "render_bar_id"),
        ("render", "render_baz_id"),
        ("render", "render_qux_id"),
    ]
    # Impl-type join: only render_foo_id lives inside impl Foo.
    impl_rows = [
        ("Foo", "render", "render_foo_id"),
        ("Bar", "render", "render_bar_id"),
        ("Baz", "render", "render_baz_id"),
        ("Qux", "render", "render_qux_id"),
    ]
    monkeypatch.setattr(
        "victor.core.database.ProjectDatabaseManager",
        _FakeProjectDatabaseManager(name_rows=name_rows, impl_rows=impl_rows),
    )

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    monkeypatch.setattr(graph_store, "upsert_edges", _capture)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    assert emitted == 1
    assert {(e.src, e.dst) for e in captured} == {("caller", "render_foo_id")}


@pytest.mark.asyncio
async def test_resolve_does_not_fall_back_when_receiver_type_unmatched(monkeypatch, tmp_path: Path):
    """If the receiver type is set but no impl T::method matches, drop the call.

    The receiver type tells us the call targets a specific T::method. If T isn't
    in our graph (typically a stdlib type like Vec or HashMap, or an external
    crate), name-only fallback would fan out to every user-defined method with
    the same leaf name -- almost certainly the wrong type. That observed
    behavior produced large inflation on names like `iter`, `format`, `clone`
    where 10-20 user impls share the leaf name. Strict receiver-typed
    semantics: typed lookup is exact, and a miss means unresolved, not
    name-only.
    """
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)
    pipeline._pending_call_records = [("caller", "render", "MysteryType", True)]

    name_rows = [("render", "render_foo_id")]
    impl_rows = [("Foo", "render", "render_foo_id")]  # no impl MysteryType
    monkeypatch.setattr(
        "victor.core.database.ProjectDatabaseManager",
        _FakeProjectDatabaseManager(name_rows=name_rows, impl_rows=impl_rows),
    )

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    monkeypatch.setattr(graph_store, "upsert_edges", _capture)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    assert emitted == 0
    assert captured == []


@pytest.mark.asyncio
async def test_resolve_drops_method_calls_with_no_inferable_receiver(monkeypatch, tmp_path: Path):
    """Method-syntax call (`x.method()`) with receiver_type=None must NOT fall
    back to name-only. Reasoning: the user wrote dot-dispatch, so they wanted
    a specific impl; if we couldn't infer the type, binding to user-defined
    methods of unrelated types with the same leaf name is almost always wrong.

    Plain function calls (is_method_call=False) keep the name-only fallback
    because `func()` is globally unambiguous in well-named codebases.
    """
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)
    # 4-tuple buffer entries: (caller_id, callee_name, receiver_type, is_method_call)
    pipeline._pending_call_records = [
        ("caller", "collect", None, True),  # method call, no type -> drop
        ("caller", "free_fn", None, False),  # plain call -> name-only fallback works
    ]

    # 10 user-defined `collect` methods (below fanout cap, so old behavior
    # would emit edges to all 10). `free_fn` has exactly one definition.
    impl_rows = [(f"T{i}", "collect", f"collect_{i}") for i in range(10)] + [
        ("Helper", "free_fn", "free_fn_id"),
    ]
    name_rows = [("collect", f"collect_{i}") for i in range(10)] + [
        ("free_fn", "free_fn_id"),
    ]
    monkeypatch.setattr(
        "victor.core.database.ProjectDatabaseManager",
        _FakeProjectDatabaseManager(name_rows=name_rows, impl_rows=impl_rows),
    )

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    monkeypatch.setattr(graph_store, "upsert_edges", _capture)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    assert emitted == 1
    assert {(e.src, e.dst) for e in captured} == {("caller", "free_fn_id")}


@pytest.mark.asyncio
async def test_resolve_does_not_fanout_when_external_stdlib_receiver(monkeypatch, tmp_path: Path):
    """Regression: `vec.iter()` where vec: Vec (stdlib) must not fan out.

    Reproduces the inflation observed on proximaDB: one call site with
    receiver_type='Vec' was emitting edges to all 12 user-defined `iter`
    methods because name-only fallback found 12 candidates below the cap of
    25. With strict receiver-typed semantics, this call is unresolved (Vec
    isn't in user code) and zero edges are emitted.
    """
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)
    pipeline._pending_call_records = [("caller", "iter", "Vec", True)]

    # 12 user-defined iter methods on various unrelated types, all under fanout cap.
    impl_rows = [
        (t, "iter", f"iter_{t}")
        for t in (
            "BTree",
            "SkipList",
            "ResultSet",
            "ZeroOverhead",
            "Ultra",
            "QuantBatch",
            "PartSet",
            "Cache",
            "Lru",
            "TypedMeta",
            "LabelSet",
            "CapSet",
        )
    ]
    name_rows = [(row[1], row[2]) for row in impl_rows]
    monkeypatch.setattr(
        "victor.core.database.ProjectDatabaseManager",
        _FakeProjectDatabaseManager(name_rows=name_rows, impl_rows=impl_rows),
    )

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    monkeypatch.setattr(graph_store, "upsert_edges", _capture)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    # With strict semantics: 0 edges. (Old behavior would emit 12.)
    assert emitted == 0
    assert captured == []


@pytest.mark.asyncio
async def test_resolve_receiver_typed_match_bypasses_fanout_cap(monkeypatch, tmp_path: Path):
    """A receiver-typed match is precise enough that fanout cap shouldn't apply.

    If 30 impls of Foo all define `method`, the receiver-typed lookup should still
    emit edges to all of them (the user wrote `obj: Foo`; that's an unambiguous
    intent compared to a bare leaf-name match).
    """
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    config.cross_file_call_max_fanout = 2  # type: ignore[attr-defined]
    pipeline = GraphIndexingPipeline(graph_store, config)
    pipeline._pending_call_records = [("caller", "method", "Foo", True)]

    # Pathological: 5 different `method` definitions inside impl Foo (e.g., from
    # multiple `impl Foo` blocks scattered across files).
    impl_rows = [("Foo", "method", f"foo_method_{i}") for i in range(5)] + [
        ("Other", "method", "other_method_id"),
    ]
    name_rows = [(row[1], row[2]) for row in impl_rows]
    monkeypatch.setattr(
        "victor.core.database.ProjectDatabaseManager",
        _FakeProjectDatabaseManager(name_rows=name_rows, impl_rows=impl_rows),
    )

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    monkeypatch.setattr(graph_store, "upsert_edges", _capture)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    # All 5 Foo::method nodes are bound; Other::method is not.
    assert emitted == 5
    dsts = {e.dst for e in captured}
    assert dsts == {f"foo_method_{i}" for i in range(5)}


# -----------------------------------------------------------------------------
# Same-file preference for plain function calls: when name-only resolution
# finds candidates in the caller's file, restrict to those. Rust function
# name resolution is module-scoped; a local `fn inputs(t)` helper called
# inside a test module shouldn't fan out to every other module's `fn inputs`
# with the same leaf name. (Method calls don't need this because typed
# dispatch already routes them precisely.)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_name_only_resolution_prefers_same_file_candidate(monkeypatch, tmp_path: Path):
    """A plain function call inside file A binds only to the file-A candidate,
    not every same-leaf-name function elsewhere.

    Reproduces the audit finding: `inputs(&t)` (a test helper) had 11
    project-wide candidates, all bound to a single call site under the
    fanout cap. Same-file preference restricts to the local helper.
    """
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)
    # Plain function call: receiver_type=None, is_method_call=False.
    pipeline._pending_call_records = [("caller_a", "inputs", None, False)]

    # name_rows: (name, node_id, file). caller_a lives in file_a; there are
    # `inputs` definitions in file_a, file_b, and file_c.
    name_rows = [
        ("caller_a_self", "caller_a", "src/file_a.rs"),  # so caller's file is known
        ("inputs", "inputs_a", "src/file_a.rs"),
        ("inputs", "inputs_b", "src/file_b.rs"),
        ("inputs", "inputs_c", "src/file_c.rs"),
    ]
    monkeypatch.setattr(
        "victor.core.database.ProjectDatabaseManager",
        _FakeProjectDatabaseManager(name_rows=name_rows),
    )

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    monkeypatch.setattr(graph_store, "upsert_edges", _capture)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    assert emitted == 1
    assert {(e.src, e.dst) for e in captured} == {("caller_a", "inputs_a")}


@pytest.mark.asyncio
async def test_name_index_excludes_trait_impl_methods(monkeypatch, tmp_path: Path):
    """Trait method names like `drop`, `fmt`, `eq`, `hash` should not appear in
    the name-only candidate list. They're invoked by the compiler or by typed
    method dispatch, never by plain function calls. Without this filter,
    `drop(x)` (std::mem::drop) was fanning out across all 16 user `impl Drop
    for T { fn drop }` impls in proximaDB.

    Verifies the resolver's SQL query carries the trait-impl filter
    (signature NOT LIKE '% for %').
    """
    captured_queries: list[str] = []

    class _RecordingConn:
        def execute(self, query):
            captured_queries.append(query)
            return iter([])

    class _RecordingDB:
        def __call__(self, _path):
            return self

        def _get_raw_connection(self):
            return _RecordingConn()

    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)
    pipeline._pending_call_records = [("caller", "drop", None, False)]

    monkeypatch.setattr("victor.core.database.ProjectDatabaseManager", _RecordingDB())

    await pipeline._resolve_cross_file_calls(tmp_path)

    # The leaf-name query (no "impl_type" alias) must filter trait impls.
    leaf_queries = [q for q in captured_queries if "FROM graph_node" in q and "impl_type" not in q]
    assert leaf_queries, "expected at least one leaf-name SELECT against graph_node"
    leaf_query = leaf_queries[0]
    assert "NOT LIKE" in leaf_query, (
        "leaf-name query must exclude trait-impl methods via NOT LIKE on signature; "
        f"got: {leaf_query!r}"
    )
    assert "for" in leaf_query.lower(), (
        "leaf-name query must filter on ` for ` substring; " f"got: {leaf_query!r}"
    )


@pytest.mark.asyncio
async def test_name_only_falls_through_when_no_same_file_candidate(monkeypatch, tmp_path: Path):
    """If no candidate is in the caller's file, keep the original
    cross-file candidate set (subject to fanout cap). Conservative: we
    only restrict when same-file candidates exist."""
    graph_store = _RecordingGraphStore()
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    pipeline = GraphIndexingPipeline(graph_store, config)
    pipeline._pending_call_records = [("caller_a", "helper", None, False)]

    # caller in file_a, but helper only defined in file_b and file_c.
    name_rows = [
        ("caller_a_self", "caller_a", "src/file_a.rs"),
        ("helper", "helper_b", "src/file_b.rs"),
        ("helper", "helper_c", "src/file_c.rs"),
    ]
    monkeypatch.setattr(
        "victor.core.database.ProjectDatabaseManager",
        _FakeProjectDatabaseManager(name_rows=name_rows),
    )

    captured: list[GraphEdge] = []

    async def _capture(edges):
        captured.extend(edges)

    monkeypatch.setattr(graph_store, "upsert_edges", _capture)

    emitted = await pipeline._resolve_cross_file_calls(tmp_path)

    # No same-file match -> fall through; both cross-file candidates emitted
    # (2 < fanout cap of 25, so they pass).
    assert emitted == 2
    assert {e.dst for e in captured} == {"helper_b", "helper_c"}


# ────────────────────────────────────────────────────────────────────────
# TSA-4: TreeSitterAnalysisProtocol provider integration in _parse_file_sync
# ────────────────────────────────────────────────────────────────────────


class _FakeAnalysisProvider:
    """Test double for TreeSitterAnalysisProtocol."""

    def __init__(
        self,
        *,
        supported: set[str] | None = None,
        symbols: list[dict] | None = None,
        raise_on_extract: bool = False,
    ) -> None:
        self.supported = supported or {"python"}
        self.symbols = symbols or []
        self.raise_on_extract = raise_on_extract
        self.calls: list[str] = []

    def supports_language(self, language: str) -> bool:
        return language in self.supported

    def extract_symbols(self, content, language, *, file_path):
        self.calls.append(file_path)
        if self.raise_on_extract:
            raise RuntimeError("provider boom")
        return list(self.symbols)


def _make_pipeline(tmp_path: Path) -> GraphIndexingPipeline:
    config = GraphIndexConfig(
        root_path=tmp_path,
        enable_ccg=False,
        enable_embeddings=False,
        enable_subgraph_cache=False,
    )
    return GraphIndexingPipeline(_RecordingGraphStore(), config)


def test_parse_file_sync_uses_enhanced_provider_when_available(monkeypatch, tmp_path: Path) -> None:
    file_path = tmp_path / "a.py"
    file_path.write_text("def whatever(): pass\n", encoding="utf-8")

    pipeline = _make_pipeline(tmp_path)
    provider = _FakeAnalysisProvider(
        symbols=[
            {
                "name": "whatever",
                "symbol_type": "function",
                "file_path": str(file_path),
                "line_start": 1,
                "line_end": 1,
                "ast_kind": "function_definition",
            }
        ],
    )
    monkeypatch.setattr(pipeline, "_get_analysis_provider", lambda: (provider, True))

    # Guarantee a failure if the hardcoded path runs instead of the provider.
    def _should_not_run(*args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("_extract_definitions should not run when provider succeeds")

    monkeypatch.setattr(pipeline, "_extract_definitions", _should_not_run)

    result = pipeline._parse_file_sync(file_path)

    assert provider.calls == [str(file_path)]
    assert [n.name for n in result.symbol_nodes] == ["whatever"]
    assert result.symbol_nodes[0].lang == "python"
    assert result.symbol_nodes[0].ast_kind == "function_definition"
    assert result.provider_fallback is False


def test_parse_file_sync_falls_back_when_provider_raises(monkeypatch, tmp_path: Path) -> None:
    file_path = tmp_path / "b.py"
    file_path.write_text("def whatever(): pass\n", encoding="utf-8")

    pipeline = _make_pipeline(tmp_path)
    provider = _FakeAnalysisProvider(raise_on_extract=True)
    monkeypatch.setattr(pipeline, "_get_analysis_provider", lambda: (provider, True))

    # Replace the hardcoded path with a deterministic stub so the test does
    # not depend on tree-sitter wheel installation.
    fallback_node = GraphNode(
        node_id="fallback",
        type="function",
        name="whatever",
        file=str(file_path),
        line=1,
        lang="python",
    )
    monkeypatch.setattr(
        pipeline,
        "_extract_symbols_fallback",
        lambda *args, **kwargs: [fallback_node],
    )
    monkeypatch.setattr(pipeline, "_create_ts_parser", lambda lang: None)

    result = pipeline._parse_file_sync(file_path)

    assert result.provider_fallback is True
    assert [n.name for n in result.symbol_nodes] == ["whatever"]


def test_parse_file_sync_skips_provider_when_only_stub_registered(
    monkeypatch, tmp_path: Path
) -> None:
    file_path = tmp_path / "c.py"
    file_path.write_text("def whatever(): pass\n", encoding="utf-8")

    pipeline = _make_pipeline(tmp_path)
    provider = _FakeAnalysisProvider()
    # enhanced=False mirrors the null stub state at root bootstrap.
    monkeypatch.setattr(pipeline, "_get_analysis_provider", lambda: (provider, False))

    def _stub_extract(*args, **kwargs):
        return [
            GraphNode(
                node_id="legacy",
                type="function",
                name="whatever",
                file=str(file_path),
                line=1,
                lang="python",
            )
        ]

    monkeypatch.setattr(pipeline, "_extract_symbols_fallback", _stub_extract)
    monkeypatch.setattr(pipeline, "_create_ts_parser", lambda lang: None)

    result = pipeline._parse_file_sync(file_path)

    assert provider.calls == [], "provider must not be called when enhanced is False"
    assert [n.name for n in result.symbol_nodes] == ["whatever"]
    assert result.provider_fallback is False


def test_provider_symbols_to_graph_nodes_maps_dict_fields() -> None:
    pipeline = _make_pipeline(Path("/tmp"))
    symbols = [
        {
            "name": "Foo",
            "symbol_type": "class",
            "file_path": "/x.py",
            "line_start": 10,
            "line_end": 20,
            "ast_kind": "class_definition",
            "signature": "class Foo(Base):",
            "visibility": "public",
        }
    ]
    nodes = pipeline._provider_symbols_to_graph_nodes(symbols, Path("/x.py"), "python")
    assert len(nodes) == 1
    node = nodes[0]
    assert node.name == "Foo"
    assert node.file == "/x.py"
    assert node.line == 10
    assert node.end_line == 20
    assert node.signature == "class Foo(Base):"
    assert node.visibility == "public"
    assert node.ast_kind == "class_definition"


def test_provider_fallback_increments_stat_in_merge(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    target = GraphIndexStats()
    source = GraphIndexStats(provider_fallbacks=3)
    pipeline._merge_stats(target, source)
    assert target.provider_fallbacks == 3


# ────────────────────────────────────────────────────────────────────────
# TSA-5: cooperative yield between mini-batches
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_consume_yields_between_flushes(monkeypatch, tmp_path: Path) -> None:
    """``_consume`` should ``await asyncio.sleep(0)`` after each flush so
    long-running indexing runs do not starve other coroutines.
    """
    import asyncio

    from victor.core.graph_rag.indexing import _IndexingStreamPipeline

    pipeline = _make_pipeline(tmp_path)
    streaming = _IndexingStreamPipeline(pipeline, write_batch_size=1, queue_maxsize=8)

    # Replace _flush with a no-op so we can count cooperative yields
    # independent of the bulk-write machinery.
    flush_calls: list[int] = []

    async def _fake_flush(batch, done_offset, total_files, cb):
        flush_calls.append(len(batch))
        return GraphIndexStats(files_processed=len(batch))

    monkeypatch.setattr(streaming, "_flush", _fake_flush)

    # Count cooperative sleeps from inside the consume loop.
    sleep_count = {"n": 0}
    original_sleep = asyncio.sleep

    async def _counting_sleep(seconds):
        if seconds == 0:
            sleep_count["n"] += 1
        await original_sleep(seconds)

    monkeypatch.setattr(asyncio, "sleep", _counting_sleep)

    queue: asyncio.Queue = asyncio.Queue(maxsize=8)
    # Enqueue three real ParseResults plus the sentinel.
    for i in range(3):
        await queue.put(
            ParseResult(
                file_path=tmp_path / f"x{i}.py",
                language="python",
                symbol_nodes=[],
            )
        )
    await queue.put(streaming._STREAM_DONE)

    stats = await streaming._consume(queue, total_files=3, done_offset=0, progress_callback=None)

    # write_batch_size=1 -> flush runs per file. The 3rd flush happens on
    # the STREAM_DONE branch and intentionally does NOT yield (we're done),
    # so we expect exactly 2 cooperative yields from the in-loop branch.
    assert flush_calls == [1, 1, 1]
    assert sleep_count["n"] >= 2
    assert stats.files_processed == 3
