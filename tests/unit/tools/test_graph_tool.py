from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from victor.storage.graph.memory_store import MemoryGraphStore
from victor.storage.graph.protocol import GraphEdge, GraphNode
from victor.storage.graph.sqlite_store import SqliteGraphStore


async def _seed_graph(store: MemoryGraphStore) -> None:
    await store.upsert_nodes(
        [
            GraphNode(node_id="file:a.py", type="file", name="a.py", file="a.py"),
            GraphNode(node_id="file:b.py", type="file", name="b.py", file="b.py"),
            GraphNode(node_id="symbol:a.py:start", type="function", name="start", file="a.py"),
            GraphNode(node_id="symbol:a.py:mid", type="function", name="mid", file="a.py"),
            GraphNode(node_id="symbol:b.py:end", type="function", name="end", file="b.py"),
            GraphNode(node_id="module:pkg.alpha", type="module", name="pkg.alpha", file="a.py"),
            GraphNode(node_id="module:pkg.beta", type="module", name="pkg.beta", file="b.py"),
        ]
    )
    await store.upsert_edges(
        [
            GraphEdge(src="file:a.py", dst="symbol:a.py:start", type="CONTAINS"),
            GraphEdge(src="file:a.py", dst="symbol:a.py:mid", type="CONTAINS"),
            GraphEdge(src="file:b.py", dst="symbol:b.py:end", type="CONTAINS"),
            GraphEdge(src="symbol:a.py:start", dst="symbol:a.py:mid", type="CALLS"),
            GraphEdge(src="symbol:a.py:mid", dst="symbol:b.py:end", type="CALLS"),
            GraphEdge(src="file:a.py", dst="module:pkg.alpha", type="IMPORTS"),
            GraphEdge(src="file:b.py", dst="module:pkg.beta", type="IMPORTS"),
        ]
    )


@pytest.mark.asyncio
async def test_graph_tool_stats_and_path(monkeypatch, tmp_path: Path):
    from victor.tools import graph_tool as graph_tool_module

    store = MemoryGraphStore()
    await _seed_graph(store)
    fake_index = SimpleNamespace(graph_store=store, files={})

    async def _fake_get_or_build_index(*args, **kwargs):
        return fake_index, False

    monkeypatch.setattr(graph_tool_module, "_get_or_build_index", _fake_get_or_build_index)

    exec_ctx = {"settings": SimpleNamespace(codebase_graph_store="memory")}

    stats = await graph_tool_module.graph(
        mode="stats",
        path=str(tmp_path),
        _exec_ctx=exec_ctx,
    )
    path_result = await graph_tool_module.graph(
        mode="path",
        path=str(tmp_path),
        source="start",
        target="end",
        _exec_ctx=exec_ctx,
    )

    assert stats["success"] is True
    assert stats["mode"] == "stats"
    assert stats["result"]["nodes"] == 7
    assert stats["result"]["edges"] == 7

    assert path_result["success"] is True
    assert path_result["mode"] == "path"
    assert path_result["result"]["found"] is True
    assert [step["node_id"] for step in path_result["result"]["path"]] == [
        "symbol:a.py:start",
        "symbol:a.py:mid",
        "symbol:b.py:end",
    ]


@pytest.mark.asyncio
async def test_graph_tool_supports_overview_alias(monkeypatch, tmp_path: Path):
    from victor.tools import graph_tool as graph_tool_module

    store = MemoryGraphStore()
    await _seed_graph(store)
    fake_index = SimpleNamespace(graph_store=store, files={})

    async def _fake_get_or_build_index(*args, **kwargs):
        return fake_index, False

    monkeypatch.setattr(graph_tool_module, "_get_or_build_index", _fake_get_or_build_index)

    exec_ctx = {"settings": SimpleNamespace(codebase_graph_store="memory")}

    result = await graph_tool_module.graph(
        mode="overview",
        path=str(tmp_path),
        _exec_ctx=exec_ctx,
    )

    assert result["success"] is True
    assert result["mode"] == "overview"
    assert result["result"]["stats"]["nodes"] == 7
    assert "important_symbols" in result["result"]
    assert "important_modules" in result["result"]


@pytest.mark.asyncio
async def test_graph_tool_resolves_file_scoped_symbol_reference(monkeypatch, tmp_path: Path):
    from victor.tools import graph_tool as graph_tool_module

    store = MemoryGraphStore()
    await _seed_graph(store)
    fake_index = SimpleNamespace(graph_store=store, files={})

    async def _fake_get_or_build_index(*args, **kwargs):
        return fake_index, False

    monkeypatch.setattr(graph_tool_module, "_get_or_build_index", _fake_get_or_build_index)

    exec_ctx = {"settings": SimpleNamespace(codebase_graph_store="memory")}

    result = await graph_tool_module.graph(
        mode="neighbors",
        path=str(tmp_path),
        node="a.py:start",
        direction="out",
        depth=1,
        _exec_ctx=exec_ctx,
    )

    assert result["success"] is True
    assert result["result"]["source"] == "symbol:a.py:start"
    assert result["result"]["total_neighbors"] == 1


@pytest.mark.asyncio
async def test_graph_tool_file_dependencies_use_index_metadata(monkeypatch, tmp_path: Path):
    from victor.tools import graph_tool as graph_tool_module

    fake_index = SimpleNamespace(
        graph_store=MemoryGraphStore(),
        files={
            "a.py": SimpleNamespace(path="a.py", dependencies=["b.py"]),
            "b.py": SimpleNamespace(path="b.py", dependencies=[]),
        },
    )

    async def _fake_get_or_build_index(*args, **kwargs):
        return fake_index, False

    monkeypatch.setattr(graph_tool_module, "_get_or_build_index", _fake_get_or_build_index)

    exec_ctx = {"settings": SimpleNamespace(codebase_graph_store="memory")}

    dependents = await graph_tool_module.graph(
        mode="file_deps",
        path=str(tmp_path),
        file="b.py",
        direction="in",
        _exec_ctx=exec_ctx,
    )
    dependencies = await graph_tool_module.graph(
        mode="file_deps",
        path=str(tmp_path),
        file="a.py",
        direction="out",
        _exec_ctx=exec_ctx,
    )

    assert dependents["success"] is True
    assert dependents["result"]["file"] == "b.py"
    assert dependents["result"]["dependents"] == ["a.py"]

    assert dependencies["success"] is True
    assert dependencies["result"]["file"] == "a.py"
    assert dependencies["result"]["dependencies"] == ["b.py"]


@pytest.mark.asyncio
async def test_graph_tool_requires_graph_support(monkeypatch, tmp_path: Path):
    from victor.tools import graph_tool as graph_tool_module

    fake_index = SimpleNamespace(graph_store=None, files={})

    async def _fake_get_or_build_index(*args, **kwargs):
        return fake_index, False

    monkeypatch.setattr(graph_tool_module, "_get_or_build_index", _fake_get_or_build_index)

    exec_ctx = {"settings": SimpleNamespace(codebase_graph_store="memory")}

    result = await graph_tool_module.graph(
        mode="stats",
        path=str(tmp_path),
        _exec_ctx=exec_ctx,
    )

    assert result["success"] is False
    assert "graph_store" in result["error"]


@pytest.mark.asyncio
async def test_graph_tool_falls_back_to_project_graph_store(monkeypatch, tmp_path: Path):
    from victor.tools import graph_tool as graph_tool_module

    store = SqliteGraphStore(tmp_path)
    await _seed_graph(store)

    async def _missing_index(*args, **kwargs):
        raise ImportError("CodebaseIndex requires a codebase indexing provider")

    monkeypatch.setattr(graph_tool_module, "_get_or_build_index", _missing_index)

    exec_ctx = {"settings": SimpleNamespace(codebase_graph_store="sqlite")}

    stats = await graph_tool_module.graph(
        mode="stats",
        path=str(tmp_path),
        _exec_ctx=exec_ctx,
    )

    assert stats["success"] is True
    assert stats["mode"] == "stats"
    assert stats["result"]["nodes"] == 7
    assert stats["result"]["edges"] == 7
