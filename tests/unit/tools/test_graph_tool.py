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
    assert result["result"]["symbol_identity_basis"]
    assert "important_symbols" in result["result"]
    assert "important_modules" in result["result"]
    assert "hub_modules" in result["result"]


@pytest.mark.asyncio
async def test_graph_tool_symbol_rankings_keep_absolute_ids_and_qualified_names(
    monkeypatch, tmp_path: Path
):
    from victor.tools import graph_tool as graph_tool_module

    store = MemoryGraphStore()
    await store.upsert_nodes(
        [
            GraphNode(node_id="symbol:a.py:run", type="function", name="run", file="a.py"),
            GraphNode(node_id="symbol:b.py:run", type="function", name="run", file="b.py"),
            GraphNode(node_id="symbol:c.py:caller", type="function", name="caller", file="c.py"),
        ]
    )
    await store.upsert_edges(
        [
            GraphEdge(src="symbol:c.py:caller", dst="symbol:a.py:run", type="CALLS"),
            GraphEdge(src="symbol:c.py:caller", dst="symbol:b.py:run", type="CALLS"),
        ]
    )
    fake_index = SimpleNamespace(graph_store=store, files={})

    async def _fake_get_or_build_index(*args, **kwargs):
        return fake_index, False

    monkeypatch.setattr(graph_tool_module, "_get_or_build_index", _fake_get_or_build_index)

    exec_ctx = {"settings": SimpleNamespace(codebase_graph_store="memory")}

    result = await graph_tool_module.graph(
        mode="centrality",
        path=str(tmp_path),
        top_k=10,
        _exec_ctx=exec_ctx,
    )

    assert result["success"] is True
    run_rows = [item for item in result["result"] if item["name"] == "run"]
    assert len(run_rows) == 2
    assert {item["node_id"] for item in run_rows} == {"symbol:a.py:run", "symbol:b.py:run"}
    assert {item["qualified_name"] for item in run_rows} == {"a.py:run", "b.py:run"}
    assert {item["module"] for item in run_rows} == {"a", "b"}


@pytest.mark.asyncio
async def test_graph_tool_supports_hub_analysis_alias(monkeypatch, tmp_path: Path):
    from victor.tools import graph_tool as graph_tool_module

    store = MemoryGraphStore()
    await _seed_graph(store)
    fake_index = SimpleNamespace(graph_store=store, files={})

    async def _fake_get_or_build_index(*args, **kwargs):
        return fake_index, False

    monkeypatch.setattr(graph_tool_module, "_get_or_build_index", _fake_get_or_build_index)

    exec_ctx = {"settings": SimpleNamespace(codebase_graph_store="memory")}

    result = await graph_tool_module.graph(
        mode="hub_analysis",
        path=str(tmp_path),
        top_k=20,
        _exec_ctx=exec_ctx,
    )

    assert result["success"] is True
    assert result["requested_mode"] == "hub_analysis"
    assert result["mode"] == "overview"
    assert result["result"]["stats"]["nodes"] == 7


@pytest.mark.asyncio
async def test_graph_tool_supports_top_k_alias_for_search_queries(monkeypatch, tmp_path: Path):
    from victor.tools import graph_tool as graph_tool_module

    store = MemoryGraphStore()
    await _seed_graph(store)
    fake_index = SimpleNamespace(graph_store=store, files={})

    async def _fake_get_or_build_index(*args, **kwargs):
        return fake_index, False

    monkeypatch.setattr(graph_tool_module, "_get_or_build_index", _fake_get_or_build_index)

    exec_ctx = {"settings": SimpleNamespace(codebase_graph_store="memory")}

    result = await graph_tool_module.graph(
        mode="top_k",
        query="start",
        path=str(tmp_path),
        top_k=5,
        _exec_ctx=exec_ctx,
    )

    assert result["success"] is True
    assert result["requested_mode"] == "top_k"
    assert result["mode"] == "search"
    assert result["result"]["matches"]
    assert result["result"]["matches"][0]["node_id"] == "symbol:a.py:start"


@pytest.mark.asyncio
async def test_graph_tool_accepts_enum_mode_values(monkeypatch, tmp_path: Path):
    from victor.tools import graph_tool as graph_tool_module

    store = MemoryGraphStore()
    await _seed_graph(store)
    fake_index = SimpleNamespace(graph_store=store, files={})

    async def _fake_get_or_build_index(*args, **kwargs):
        return fake_index, False

    monkeypatch.setattr(graph_tool_module, "_get_or_build_index", _fake_get_or_build_index)

    exec_ctx = {"settings": SimpleNamespace(codebase_graph_store="memory")}

    result = await graph_tool_module.graph(
        mode=graph_tool_module.GraphMode.STATS,
        path=str(tmp_path),
        _exec_ctx=exec_ctx,
    )

    assert result["success"] is True
    assert result["requested_mode"] == "stats"
    assert result["mode"] == "stats"
    assert result["result"]["nodes"] == 7


def test_graph_tool_schema_exposes_mode_and_direction_enums():
    from victor.tools import graph_tool as graph_tool_module

    params = graph_tool_module.graph.Tool.parameters

    assert params["properties"]["mode"]["enum"]
    assert "overview" in params["properties"]["mode"]["enum"]
    assert "neighbors" in params["properties"]["mode"]["enum"]
    assert params["properties"]["mode"]["default"] == "neighbors"

    assert params["properties"]["direction"]["enum"] == ["out", "in", "both"]


@pytest.mark.asyncio
async def test_graph_tool_unsupported_mode_returns_follow_up_suggestions(
    monkeypatch, tmp_path: Path
):
    from victor.tools import graph_tool as graph_tool_module

    store = MemoryGraphStore()
    await _seed_graph(store)
    fake_index = SimpleNamespace(graph_store=store, files={})

    async def _fake_get_or_build_index(*args, **kwargs):
        return fake_index, False

    monkeypatch.setattr(graph_tool_module, "_get_or_build_index", _fake_get_or_build_index)

    exec_ctx = {"settings": SimpleNamespace(codebase_graph_store="memory")}

    result = await graph_tool_module.graph(
        mode="hubs",
        query="start",
        path=str(tmp_path),
        top_k=5,
        _exec_ctx=exec_ctx,
    )

    assert result["success"] is False
    assert result["requested_mode"] == "hubs"
    assert "metadata" in result
    suggestions = result["metadata"]["follow_up_suggestions"]
    assert suggestions
    assert any('graph(mode="search"' in suggestion["command"] for suggestion in suggestions)


@pytest.mark.asyncio
async def test_graph_tool_unresolved_node_returns_follow_up_suggestions(
    monkeypatch, tmp_path: Path
):
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
        node="missing_symbol",
        path=str(tmp_path),
        depth=2,
        _exec_ctx=exec_ctx,
    )

    assert result["success"] is False
    suggestions = result["metadata"]["follow_up_suggestions"]
    assert suggestions
    assert suggestions[0]["command"].startswith('graph(mode="search"')


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
async def test_graph_tool_file_deps_with_directory_path_falls_back_to_overview(
    monkeypatch, tmp_path: Path
):
    from victor.tools import graph_tool as graph_tool_module

    store = MemoryGraphStore()
    await _seed_graph(store)
    fake_index = SimpleNamespace(graph_store=store, files={})

    async def _fake_get_or_build_index(*args, **kwargs):
        return fake_index, False

    monkeypatch.setattr(graph_tool_module, "_get_or_build_index", _fake_get_or_build_index)

    exec_ctx = {"settings": SimpleNamespace(codebase_graph_store="memory")}

    result = await graph_tool_module.graph(
        mode="file_deps",
        path="victor/framework",
        _exec_ctx=exec_ctx,
    )

    assert result["success"] is True
    assert result["requested_mode"] == "file_deps"
    assert result["mode"] == "overview"
    assert result["result"]["recovered_from_mode"] == "file_deps"
    assert result["result"]["recovered_from_path"] == "victor/framework"
    assert result["result"]["stats"]["nodes"] == 7


@pytest.mark.asyncio
async def test_graph_tool_file_deps_with_file_path_uses_path_as_subject(
    monkeypatch, tmp_path: Path
):
    from victor.tools import graph_tool as graph_tool_module

    fake_index = SimpleNamespace(
        graph_store=MemoryGraphStore(),
        files={
            "agent.py": SimpleNamespace(
                path="agent.py",
                dependencies=["orchestrator.py"],
            ),
            "orchestrator.py": SimpleNamespace(path="orchestrator.py", dependencies=[]),
        },
    )

    async def _fake_get_or_build_index(*args, **kwargs):
        return fake_index, False

    monkeypatch.setattr(graph_tool_module, "_get_or_build_index", _fake_get_or_build_index)

    exec_ctx = {"settings": SimpleNamespace(codebase_graph_store="memory")}

    result = await graph_tool_module.graph(
        mode="file_deps",
        path="victor/framework/agent.py",
        _exec_ctx=exec_ctx,
    )

    assert result["success"] is True
    assert result["requested_mode"] == "file_deps"
    assert result["mode"] == "file_deps"
    assert result["result"]["file"] == "agent.py"
    assert result["result"]["dependencies"] == ["orchestrator.py"]
    assert result["result"]["recovered_from_mode"] == "file_deps"
    assert result["result"]["recovered_from_path"] == "victor/framework/agent.py"


@pytest.mark.asyncio
async def test_graph_tool_call_flow_with_file_falls_back_to_file_dependencies(
    monkeypatch, tmp_path: Path
):
    from victor.tools import graph_tool as graph_tool_module

    fake_index = SimpleNamespace(
        graph_store=MemoryGraphStore(),
        files={
            "victor/framework/agent.py": SimpleNamespace(
                path="victor/framework/agent.py",
                dependencies=["victor/agent/orchestrator.py"],
            ),
            "victor/agent/orchestrator.py": SimpleNamespace(
                path="victor/agent/orchestrator.py",
                dependencies=[],
            ),
        },
    )

    async def _fake_get_or_build_index(*args, **kwargs):
        return fake_index, False

    monkeypatch.setattr(graph_tool_module, "_get_or_build_index", _fake_get_or_build_index)

    exec_ctx = {"settings": SimpleNamespace(codebase_graph_store="memory")}

    result = await graph_tool_module.graph(
        mode="call_flow",
        path=str(tmp_path),
        file="victor/framework/agent.py",
        depth=3,
        _exec_ctx=exec_ctx,
    )

    assert result["success"] is True
    assert result["requested_mode"] == "call_flow"
    assert result["mode"] == "file_deps"
    assert result["result"]["file"] == "victor/framework/agent.py"
    assert result["result"]["dependencies"] == ["victor/agent/orchestrator.py"]
    assert result["result"]["recovered_from_mode"] == "call_flow"


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


def test_graph_tool_is_available_with_persisted_project_graph(monkeypatch, tmp_path: Path):
    from victor.tools import graph_tool as graph_tool_module

    class _MissingProviderRegistry:
        def ensure_bootstrapped(self) -> None:
            return None

        def get(self, protocol: object) -> None:
            del protocol
            return None

        def is_enhanced(self, protocol: object) -> bool:
            del protocol
            return False

    store = SqliteGraphStore(tmp_path)

    import asyncio
    import victor.core.capability_registry as capability_registry_module

    asyncio.run(_seed_graph(store))
    monkeypatch.setattr(
        capability_registry_module.CapabilityRegistry,
        "get_instance",
        staticmethod(lambda: _MissingProviderRegistry()),
    )
    monkeypatch.setattr(
        graph_tool_module,
        "get_project_paths",
        lambda: SimpleNamespace(project_root=tmp_path),
    )

    assert graph_tool_module.graph.Tool.is_available() is True


def test_graph_tool_is_unavailable_without_provider_or_graph_data(monkeypatch, tmp_path: Path):
    from victor.tools import graph_tool as graph_tool_module

    class _MissingProviderRegistry:
        def ensure_bootstrapped(self) -> None:
            return None

        def get(self, protocol: object) -> None:
            del protocol
            return None

        def is_enhanced(self, protocol: object) -> bool:
            del protocol
            return False

    import victor.core.capability_registry as capability_registry_module

    monkeypatch.setattr(
        capability_registry_module.CapabilityRegistry,
        "get_instance",
        staticmethod(lambda: _MissingProviderRegistry()),
    )
    monkeypatch.setattr(
        graph_tool_module,
        "get_project_paths",
        lambda: SimpleNamespace(project_root=tmp_path),
    )

    assert graph_tool_module.graph.Tool.is_available() is False


@pytest.mark.asyncio
async def test_graph_tool_semantic_mode_gracefully_skips_when_index_has_no_semantic_search(
    monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    from victor.tools import graph_tool as graph_tool_module

    store = MemoryGraphStore()
    await _seed_graph(store)
    fake_index = SimpleNamespace(graph_store=store, files={})

    async def _fake_get_or_build_index(*args, **kwargs):
        return fake_index, False

    monkeypatch.setattr(graph_tool_module, "_get_or_build_index", _fake_get_or_build_index)

    exec_ctx = {"settings": SimpleNamespace(codebase_graph_store="memory")}

    with caplog.at_level("WARNING", logger="victor.tools.graph_tool"):
        result = await graph_tool_module.graph(
            mode="semantic",
            node="start",
            path=str(tmp_path),
            top_k=5,
            _exec_ctx=exec_ctx,
        )

    assert result["success"] is True
    assert result["result"]["semantic_search_available"] is False
    assert result["result"]["potential_relationships"] == []
    assert "Semantic search failed during discovery" not in caplog.text
