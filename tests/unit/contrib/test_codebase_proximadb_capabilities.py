from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from victor.verticals.contrib.coding.codebase.indexer import CodebaseIndex


class FakeCapabilityProvider:
    def __init__(self) -> None:
        self._initialized = False
        self.initialize = AsyncMock(side_effect=self._mark_initialized)
        self.hybrid_search = AsyncMock(return_value=[{"id": "hybrid:1"}])
        self.find_callers = AsyncMock(return_value=[{"id": "caller:1"}])
        self.find_similar_bugs = AsyncMock(return_value=[{"id": "bug:1"}])

    async def _mark_initialized(self) -> None:
        self._initialized = True


def _make_index(provider: object, use_embeddings: bool = True) -> CodebaseIndex:
    index = object.__new__(CodebaseIndex)
    index.use_embeddings = use_embeddings
    index.embedding_provider = provider
    index.ensure_indexed = AsyncMock()
    return index


@pytest.mark.asyncio
async def test_codebase_index_hybrid_search_delegates_to_provider() -> None:
    provider = FakeCapabilityProvider()
    index = _make_index(provider)

    results = await index.hybrid_search(
        query="parse json",
        graph_query="MATCH (c)-[:CALLS]->(f)",
        document_filter={"language": "python"},
        top_k=5,
        auto_reindex=False,
    )

    index.ensure_indexed.assert_awaited_once_with(auto_reindex=False)
    provider.initialize.assert_awaited_once()
    provider.hybrid_search.assert_awaited_once_with(
        query="parse json",
        graph_query="MATCH (c)-[:CALLS]->(f)",
        document_filter={"language": "python"},
        time_range=None,
        top_k=5,
    )
    assert results == [{"id": "hybrid:1"}]


@pytest.mark.asyncio
async def test_codebase_index_find_callers_delegates_to_provider() -> None:
    provider = FakeCapabilityProvider()
    provider._initialized = True
    index = _make_index(provider)

    results = await index.find_callers(
        function_name="parse_json",
        file_path="src/main.py",
        edge_type="CALLS",
        max_depth=2,
    )

    index.ensure_indexed.assert_awaited_once_with(auto_reindex=True)
    provider.initialize.assert_not_awaited()
    provider.find_callers.assert_awaited_once_with(
        function_name="parse_json",
        file_path="src/main.py",
        edge_type="CALLS",
        max_depth=2,
    )
    assert results == [{"id": "caller:1"}]


@pytest.mark.asyncio
async def test_codebase_index_find_similar_bugs_delegates_to_provider() -> None:
    provider = FakeCapabilityProvider()
    provider._initialized = True
    index = _make_index(provider)

    results = await index.find_similar_bugs(
        bug_description="json parsing crash",
        language="python",
        top_k=7,
        include_graph_context=False,
        context_limit=1,
    )

    provider.find_similar_bugs.assert_awaited_once_with(
        bug_description="json parsing crash",
        language="python",
        top_k=7,
        include_graph_context=False,
        context_limit=1,
    )
    assert results == [{"id": "bug:1"}]


@pytest.mark.asyncio
async def test_codebase_index_capability_methods_raise_for_unsupported_provider() -> None:
    class MinimalProvider:
        def __init__(self) -> None:
            self._initialized = True

    index = _make_index(MinimalProvider())

    with pytest.raises(NotImplementedError, match="trace_execution_path"):
        await index.trace_execution_path("main")


@pytest.mark.asyncio
async def test_codebase_index_capability_methods_require_embeddings() -> None:
    index = _make_index(provider=None, use_embeddings=False)

    with pytest.raises(ValueError, match="Embeddings not enabled"):
        await index.find_callers("parse_json")
