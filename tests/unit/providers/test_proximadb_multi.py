"""Unit tests for the ProximaDB multi-model provider."""

from __future__ import annotations

import importlib.util
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

from victor.storage.vector_stores.base import EmbeddingConfig
from victor.storage.vector_stores.proximadb_multi import ProximaDBMultiModelProvider

if importlib.util.find_spec("proximadb_sdk") is None:
    pytest.skip("proximadb_sdk not installed", allow_module_level=True)


def test_proximadb_sdk_top_level_imports(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Smoke test the ProximaDB SDK imports Victor depends on."""

    monkeypatch.setenv("DSP_CACHEBOOL", "false")
    monkeypatch.setenv("DSPY_CACHEDIR", str(tmp_path / "dspy-cache"))

    import proximadb_sdk
    from proximadb_sdk import ProximaDBClient, ProximaDBGraph

    assert proximadb_sdk.__file__
    assert ProximaDBClient.__name__ == "ProximaDBClient"
    assert ProximaDBGraph.__name__ == "ProximaDBGraph"


class StubEmbeddingModel:
    """Minimal async embedding model used by provider tests."""

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension
        self.initialized = False

    async def initialize(self) -> None:
        self.initialized = True

    async def embed_text(self, text: str) -> List[float]:
        seed = float((len(text) % 7) + 1)
        return [seed, seed / 2.0, seed / 3.0, seed / 4.0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed_text(text) for text in texts]

    def get_dimension(self) -> int:
        return self._dimension

    async def close(self) -> None:
        self.initialized = False


@dataclass
class FakeSearchHit:
    """Search hit shape compatible with the provider."""

    id: str
    score: float
    source: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class FakeGraphNode:
    """Minimal graph node shape returned by helper-backed graph queries."""

    id: str
    labels: List[str]
    properties: Dict[str, Any]


class FakeProximaClient:
    """In-memory ProximaDB stand-in for unit tests."""

    def __init__(self) -> None:
        self.collections: Dict[str, List[Any]] = defaultdict(list)
        self.created_collections: List[str] = []
        self.created_graphs: List[str] = []
        self.deleted_collections: List[str] = []
        self.deleted_graphs: List[str] = []
        self.deleted_vectors: List[tuple[str, List[str]]] = []
        self.graph_nodes: List[Dict[str, Any]] = []
        self.graph_edges: List[Dict[str, Any]] = []
        self.query_node_rows: List[Dict[str, Any]] = []
        self.traversal_rows: Dict[str, Dict[str, Any]] = {}
        self.sql_rows: List[Dict[str, Any]] = []

    def create_collection(self, name: str, config: Any = None, **kwargs: Any) -> Dict[str, Any]:
        if name not in self.created_collections:
            self.created_collections.append(name)
        self.collections.setdefault(name, [])
        return {"name": name, "config": config, "kwargs": kwargs}

    def get_collection(self, name: str) -> Dict[str, Any]:
        return {"name": name, "record_count": len(self.collections.get(name, []))}

    def delete_collection(self, name: str) -> None:
        self.deleted_collections.append(name)
        self.collections.pop(name, None)

    def insert_vectors(self, collection: str, records: List[Any], **kwargs: Any) -> None:
        self.collections[collection].extend(records)

    def search(
        self,
        collection: str,
        vector: List[float],
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[FakeSearchHit]:
        del vector, kwargs
        records = list(self.collections.get(collection, []))
        hits = []
        for index, record in enumerate(records):
            metadata = dict(getattr(record, "metadata", {}) or {})
            if metadata_filter and any(
                metadata.get(key) != value for key, value in metadata_filter.items()
            ):
                continue
            hits.append(
                FakeSearchHit(
                    id=getattr(record, "id", f"record:{index}"),
                    score=1.0 - (index * 0.01),
                    source=getattr(record, "source", None),
                    metadata=metadata,
                )
            )
        return hits[:top_k]

    def delete_vectors(self, collection: str, ids: List[str]) -> None:
        self.deleted_vectors.append((collection, list(ids)))
        id_set = set(ids)
        self.collections[collection] = [
            record
            for record in self.collections.get(collection, [])
            if getattr(record, "id", None) not in id_set
        ]

    def create_graph(
        self,
        graph_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del name, description, schema
        if graph_id not in self.created_graphs:
            self.created_graphs.append(graph_id)
        return {"graph_id": graph_id}

    def get_graph(self, graph_id: str) -> Dict[str, Any]:
        return {
            "graph_id": graph_id,
            "node_count": len(self.graph_nodes),
            "edge_count": len(self.graph_edges),
        }

    def query_nodes(
        self,
        graph_id: Optional[str] = None,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del graph_id, kwargs
        labels = list(labels or [])
        properties = dict(properties or {})
        matches = []
        for node in self.query_node_rows:
            node_labels = list(node.get("labels", []))
            node_props = dict(node.get("properties", {}) or {})
            if labels and not all(label in node_labels for label in labels):
                continue
            if any(node_props.get(key) != value for key, value in properties.items()):
                continue
            matches.append(node)
        return {"nodes": matches}

    def delete_graph(self, graph_id: str) -> None:
        self.deleted_graphs.append(graph_id)

    def create_node(
        self,
        node_id: Optional[str] = None,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        graph_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del graph_id
        node_id = node_id or kwargs.get("id")
        labels = list(labels or kwargs.get("labels", []))
        properties = dict(properties or kwargs.get("properties", {}))
        self.graph_nodes.append({"id": node_id, "labels": labels, "properties": properties})
        return self.graph_nodes[-1]

    def create_edge(
        self,
        edge_id: Optional[str] = None,
        from_node_id: Optional[str] = None,
        to_node_id: Optional[str] = None,
        edge_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        graph_id: Optional[str] = None,
        from_node: Optional[str] = None,
        to_node: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del graph_id
        edge_id = edge_id or kwargs.get("id")
        from_node_id = from_node_id or from_node or kwargs.get("from_node")
        to_node_id = to_node_id or to_node or kwargs.get("to_node")
        edge_type = edge_type or kwargs.get("edge_type")
        properties = dict(properties or kwargs.get("properties", {}))
        self.graph_edges.append(
            {
                "id": edge_id,
                "from_node_id": from_node_id,
                "to_node_id": to_node_id,
                "edge_type": edge_type,
                "properties": properties,
            }
        )
        return self.graph_edges[-1]

    def execute_sql(
        self,
        query: str,
        parameters: Optional[List[Any]] = None,
        collection: Optional[str] = None,
    ) -> Dict[str, Any]:
        del query, parameters, collection
        return {"rows": list(self.sql_rows)}

    def traverse_graph(
        self,
        graph_id: Optional[str] = None,
        start_node_id: Optional[str] = None,
        max_depth: int = 1,
        edge_types: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del graph_id, max_depth, edge_types, kwargs
        if start_node_id is None:
            return {"nodes": [], "edges": []}
        return dict(self.traversal_rows.get(start_node_id, {"nodes": [], "edges": []}))


@pytest.fixture
def provider_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        vector_store="proximadb_multi",
        embedding_model_type="sentence-transformers",
        embedding_model_name="all-MiniLM-L12-v2",
        distance_metric="cosine",
        extra_config={
            "workspace": "victor_test_repo",
            "dimension": 4,
            "chunk_size": 8,
            "chunk_overlap": 1,
            "vector_index": "hnsw",
        },
    )


@pytest.fixture
def fake_client() -> FakeProximaClient:
    return FakeProximaClient()


@pytest.fixture
def stub_model() -> StubEmbeddingModel:
    return StubEmbeddingModel(dimension=4)


class TestProximaDBMultiModelProvider:
    """Focused tests for Victor's ProximaDB multi-model adapter."""

    @pytest.mark.asyncio
    async def test_initialize_creates_collections_and_graph(
        self,
        provider_config: EmbeddingConfig,
        fake_client: FakeProximaClient,
        stub_model: StubEmbeddingModel,
    ) -> None:
        with patch(
            "victor.storage.vector_stores.proximadb_multi.create_embedding_model",
            return_value=stub_model,
        ):
            provider = ProximaDBMultiModelProvider(provider_config, client=fake_client)
            await provider.initialize()

        assert provider._initialized
        assert fake_client.created_graphs == ["victor_test_repo_graph"]
        assert fake_client.created_collections == [
            "victor_test_repo_vectors",
            "victor_test_repo_documents",
            "victor_test_repo_metrics",
        ]

    @pytest.mark.asyncio
    async def test_index_code_file_populates_vectors_documents_graph_and_metrics(
        self,
        provider_config: EmbeddingConfig,
        fake_client: FakeProximaClient,
        stub_model: StubEmbeddingModel,
    ) -> None:
        code = """
import json

def main(data):
    return parse_json(data)

def parse_json(data):
    return json.loads(data)
""".strip()

        with patch(
            "victor.storage.vector_stores.proximadb_multi.create_embedding_model",
            return_value=stub_model,
        ):
            provider = ProximaDBMultiModelProvider(provider_config, client=fake_client)
            result = await provider.index_code_file("src/main.py", code, language="python")

        assert result["vectors"] >= 1
        assert result["document"] is True
        assert result["graph"]["functions"] == 2
        assert result["graph"]["calls"] == 1
        assert result["graph"]["imports"] == 1
        assert result["timeseries"] >= 4

        assert any("Module" in node["labels"] for node in fake_client.graph_nodes)
        assert any(node["properties"].get("name") == "main" for node in fake_client.graph_nodes)
        assert any(edge["edge_type"] == "CALLS" for edge in fake_client.graph_edges)
        assert len(fake_client.collections["victor_test_repo_documents"]) == 1
        assert len(fake_client.collections["victor_test_repo_metrics"]) >= 4

    @pytest.mark.asyncio
    async def test_search_similar_returns_embedding_search_results(
        self,
        provider_config: EmbeddingConfig,
        fake_client: FakeProximaClient,
        stub_model: StubEmbeddingModel,
    ) -> None:
        with patch(
            "victor.storage.vector_stores.proximadb_multi.create_embedding_model",
            return_value=stub_model,
        ):
            provider = ProximaDBMultiModelProvider(provider_config, client=fake_client)
            await provider.initialize()
            await provider.index_document(
                doc_id="chunk:1",
                content="def parse_json(data): return json.loads(data)",
                metadata={
                    "file_path": "src/main.py",
                    "symbol_name": "parse_json",
                    "start_line": 4,
                    "language": "python",
                },
            )
            results = await provider.search_similar("parse json", limit=5)

        assert len(results) == 1
        assert results[0].file_path == "src/main.py"
        assert results[0].symbol_name == "parse_json"
        assert "json.loads" in results[0].content

    @pytest.mark.asyncio
    async def test_hybrid_search_merges_vector_graph_and_document_hits(
        self,
        provider_config: EmbeddingConfig,
        fake_client: FakeProximaClient,
        stub_model: StubEmbeddingModel,
    ) -> None:
        fake_client.sql_rows = [
            {
                "node_id": "victor_test_repo:function:src/main.py:parse_json:4",
                "score": 0.8,
                "properties": {
                    "file_path": "src/main.py",
                    "name": "parse_json",
                    "qualified_name": "parse_json",
                    "language": "python",
                },
            }
        ]

        code = """
import json

def parse_json(data):
    return json.loads(data)
""".strip()

        with patch(
            "victor.storage.vector_stores.proximadb_multi.create_embedding_model",
            return_value=stub_model,
        ):
            provider = ProximaDBMultiModelProvider(provider_config, client=fake_client)
            await provider.index_code_file("src/main.py", code, language="python")
            results = await provider.hybrid_search(
                query="parse json",
                graph_query="SELECT * FROM graph_query",
                document_filter={"language": "python"},
                top_k=5,
            )

        assert results
        assert results[0]["file_path"] == "src/main.py"
        assert "vector" in results[0]["sources"]
        assert any("graph" in row["sources"] for row in results)

    @pytest.mark.asyncio
    async def test_find_callers_uses_graph_helper(
        self,
        provider_config: EmbeddingConfig,
        fake_client: FakeProximaClient,
        stub_model: StubEmbeddingModel,
    ) -> None:
        target_node_id = "victor_test_repo:function:src/main.py:parse_json:4"
        fake_client.query_node_rows = [
            {
                "id": target_node_id,
                "labels": ["Function"],
                "properties": {
                    "name": "parse_json",
                    "qualified_name": "parse_json",
                    "file_path": "src/main.py",
                    "line_start": 4,
                },
            }
        ]

        class FakeGraphHelper:
            def __init__(self) -> None:
                self.calls: List[tuple[str, str, int]] = []

            def find_callers(
                self,
                node_id: str,
                edge_type: str = "CALLS",
                max_depth: int = 1,
            ) -> List[FakeGraphNode]:
                self.calls.append((node_id, edge_type, max_depth))
                return [
                    FakeGraphNode(
                        id="victor_test_repo:function:src/main.py:main:2",
                        labels=["Function"],
                        properties={
                            "name": "main",
                            "qualified_name": "main",
                            "file_path": "src/main.py",
                            "line_start": 2,
                            "line_end": 3,
                        },
                    )
                ]

        with patch(
            "victor.storage.vector_stores.proximadb_multi.create_embedding_model",
            return_value=stub_model,
        ):
            provider = ProximaDBMultiModelProvider(provider_config, client=fake_client)
            await provider.initialize()
            helper = FakeGraphHelper()
            provider._graph_api = helper
            results = await provider.find_callers("parse_json", file_path="src/main.py")

        assert helper.calls == [(target_node_id, "CALLS", 1)]
        assert results == [
            {
                "id": "victor_test_repo:function:src/main.py:main:2",
                "name": "main",
                "file_path": "src/main.py",
                "line_start": 2,
                "line_end": 3,
                "labels": ["Function"],
                "metadata": {
                    "name": "main",
                    "qualified_name": "main",
                    "file_path": "src/main.py",
                    "line_start": 2,
                    "line_end": 3,
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_find_callees_uses_graph_traversal(
        self,
        provider_config: EmbeddingConfig,
        fake_client: FakeProximaClient,
        stub_model: StubEmbeddingModel,
    ) -> None:
        start_node_id = "victor_test_repo:function:src/main.py:main:2"
        callee_node_id = "victor_test_repo:function:src/main.py:parse_json:4"
        fake_client.query_node_rows = [
            {
                "id": start_node_id,
                "labels": ["Function"],
                "properties": {
                    "name": "main",
                    "qualified_name": "main",
                    "file_path": "src/main.py",
                    "line_start": 2,
                },
            }
        ]
        fake_client.traversal_rows[start_node_id] = {
            "nodes": [
                {
                    "id": start_node_id,
                    "labels": ["Function"],
                    "properties": {
                        "name": "main",
                        "qualified_name": "main",
                        "file_path": "src/main.py",
                        "line_start": 2,
                    },
                },
                {
                    "id": callee_node_id,
                    "labels": ["Function"],
                    "properties": {
                        "name": "parse_json",
                        "qualified_name": "parse_json",
                        "file_path": "src/main.py",
                        "line_start": 4,
                        "line_end": 5,
                    },
                },
            ],
            "edges": [
                {
                    "id": "edge:main:parse_json",
                    "from_node_id": start_node_id,
                    "to_node_id": callee_node_id,
                    "edge_type": "CALLS",
                    "properties": {"file_path": "src/main.py", "line_number": 3},
                }
            ],
        }

        with patch(
            "victor.storage.vector_stores.proximadb_multi.create_embedding_model",
            return_value=stub_model,
        ):
            provider = ProximaDBMultiModelProvider(provider_config, client=fake_client)
            await provider.initialize()
            results = await provider.find_callees("main", file_path="src/main.py")

        assert results == [
            {
                "id": callee_node_id,
                "name": "parse_json",
                "file_path": "src/main.py",
                "line_start": 4,
                "line_end": 5,
                "labels": ["Function"],
                "metadata": {
                    "name": "parse_json",
                    "qualified_name": "parse_json",
                    "file_path": "src/main.py",
                    "line_start": 4,
                    "line_end": 5,
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_trace_execution_path_returns_nodes_and_edges(
        self,
        provider_config: EmbeddingConfig,
        fake_client: FakeProximaClient,
        stub_model: StubEmbeddingModel,
    ) -> None:
        entry_node_id = "victor_test_repo:function:src/main.py:main:2"
        parse_node_id = "victor_test_repo:function:src/main.py:parse_json:4"
        loads_node_id = "victor_test_repo:external:function:json.loads"
        fake_client.query_node_rows = [
            {
                "id": entry_node_id,
                "labels": ["Function"],
                "properties": {
                    "name": "main",
                    "qualified_name": "main",
                    "file_path": "src/main.py",
                    "line_start": 2,
                    "line_end": 3,
                },
            }
        ]
        fake_client.traversal_rows[entry_node_id] = {
            "nodes": [
                {
                    "id": entry_node_id,
                    "labels": ["Function"],
                    "properties": {
                        "name": "main",
                        "qualified_name": "main",
                        "file_path": "src/main.py",
                        "line_start": 2,
                        "line_end": 3,
                    },
                },
                {
                    "id": parse_node_id,
                    "labels": ["Function"],
                    "properties": {
                        "name": "parse_json",
                        "qualified_name": "parse_json",
                        "file_path": "src/main.py",
                        "line_start": 4,
                        "line_end": 5,
                    },
                },
                {
                    "id": loads_node_id,
                    "labels": ["Function", "External"],
                    "properties": {
                        "name": "json.loads",
                        "qualified_name": "json.loads",
                    },
                },
            ],
            "edges": [
                {
                    "id": "edge:main:parse_json",
                    "from_node_id": entry_node_id,
                    "to_node_id": parse_node_id,
                    "edge_type": "CALLS",
                    "properties": {"file_path": "src/main.py", "line_number": 3},
                },
                {
                    "id": "edge:parse_json:json.loads",
                    "from_node_id": parse_node_id,
                    "to_node_id": loads_node_id,
                    "edge_type": "CALLS",
                    "properties": {"file_path": "src/main.py", "line_number": 5},
                },
            ],
        }

        with patch(
            "victor.storage.vector_stores.proximadb_multi.create_embedding_model",
            return_value=stub_model,
        ):
            provider = ProximaDBMultiModelProvider(provider_config, client=fake_client)
            await provider.initialize()
            result = await provider.trace_execution_path(
                "main", file_path="src/main.py", max_depth=2
            )

        assert result["entry_point"]["id"] == entry_node_id
        assert result["edge_type"] == "CALLS"
        assert result["max_depth"] == 2
        assert [edge["id"] for edge in result["edges"]] == [
            "edge:main:parse_json",
            "edge:parse_json:json.loads",
        ]
        assert {node["id"] for node in result["nodes"]} == {
            entry_node_id,
            parse_node_id,
            loads_node_id,
        }

    @pytest.mark.asyncio
    async def test_find_similar_bugs_enriches_hybrid_hits_with_graph_context(
        self,
        provider_config: EmbeddingConfig,
        fake_client: FakeProximaClient,
        stub_model: StubEmbeddingModel,
    ) -> None:
        with patch(
            "victor.storage.vector_stores.proximadb_multi.create_embedding_model",
            return_value=stub_model,
        ):
            provider = ProximaDBMultiModelProvider(provider_config, client=fake_client)
            await provider.initialize()

        provider.hybrid_search = AsyncMock(
            return_value=[
                {
                    "id": "hit:1",
                    "file_path": "src/main.py",
                    "symbol_name": "parse_json",
                    "content": "def parse_json(data): return json.loads(data)",
                    "score": 0.91,
                    "sources": ["vector", "document"],
                    "metadata": {
                        "file_path": "src/main.py",
                        "language": "python",
                        "qualified_name": "parse_json",
                    },
                }
            ]
        )
        provider.find_callers = AsyncMock(
            return_value=[
                {
                    "id": "caller:1",
                    "name": "main",
                    "file_path": "src/main.py",
                    "line_start": 2,
                    "line_end": 3,
                    "labels": ["Function"],
                    "metadata": {"name": "main"},
                }
            ]
        )
        provider.find_callees = AsyncMock(
            return_value=[
                {
                    "id": "callee:1",
                    "name": "json.loads",
                    "file_path": "src/json_utils.py",
                    "line_start": 8,
                    "line_end": 8,
                    "labels": ["Function", "External"],
                    "metadata": {"name": "json.loads"},
                }
            ]
        )

        results = await provider.find_similar_bugs(
            "json parsing crash",
            language="python",
            top_k=5,
            context_limit=2,
        )

        provider.hybrid_search.assert_awaited_once_with(
            query="json parsing crash",
            document_filter={"language": "python"},
            top_k=5,
        )
        provider.find_callers.assert_awaited_once_with("parse_json", file_path="src/main.py")
        provider.find_callees.assert_awaited_once_with("parse_json", file_path="src/main.py")
        assert results == [
            {
                "id": "hit:1",
                "file_path": "src/main.py",
                "symbol_name": "parse_json",
                "content": "def parse_json(data): return json.loads(data)",
                "score": 0.91,
                "sources": ["vector", "document"],
                "metadata": {
                    "file_path": "src/main.py",
                    "language": "python",
                    "qualified_name": "parse_json",
                },
                "graph_context": {
                    "callers": [
                        {
                            "id": "caller:1",
                            "name": "main",
                            "file_path": "src/main.py",
                            "line_start": 2,
                            "line_end": 3,
                            "labels": ["Function"],
                            "metadata": {"name": "main"},
                        }
                    ],
                    "callees": [
                        {
                            "id": "callee:1",
                            "name": "json.loads",
                            "file_path": "src/json_utils.py",
                            "line_start": 8,
                            "line_end": 8,
                            "labels": ["Function", "External"],
                            "metadata": {"name": "json.loads"},
                        }
                    ],
                    "related_files": ["src/json_utils.py", "src/main.py"],
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_localize_issue_returns_seed_and_graph_related_files(
        self,
        provider_config: EmbeddingConfig,
        fake_client: FakeProximaClient,
        stub_model: StubEmbeddingModel,
    ) -> None:
        with patch(
            "victor.storage.vector_stores.proximadb_multi.create_embedding_model",
            return_value=stub_model,
        ):
            provider = ProximaDBMultiModelProvider(provider_config, client=fake_client)
            await provider.initialize()

        provider.hybrid_search = AsyncMock(
            return_value=[
                {
                    "id": "hit:1",
                    "file_path": "src/repository.py",
                    "symbol_name": "BaseRepository.save",
                    "content": "class BaseRepository:\n    def save(self, entity): ...",
                    "score": 0.82,
                    "sources": ["vector", "document"],
                    "metadata": {
                        "file_path": "src/repository.py",
                        "language": "python",
                        "qualified_name": "BaseRepository.save",
                    },
                }
            ]
        )
        provider.find_callers = AsyncMock(
            return_value=[
                {
                    "id": "caller:1",
                    "name": "UserService.create_user",
                    "file_path": "src/service.py",
                    "line_start": 12,
                    "line_end": 18,
                    "labels": ["Function"],
                    "metadata": {"name": "UserService.create_user"},
                }
            ]
        )
        provider.find_callees = AsyncMock(
            return_value=[
                {
                    "id": "callee:1",
                    "name": "AuditLogger.log_save",
                    "file_path": "src/audit.py",
                    "line_start": 4,
                    "line_end": 8,
                    "labels": ["Function"],
                    "metadata": {"name": "AuditLogger.log_save"},
                }
            ]
        )

        results = await provider.localize_issue(
            "which files should I edit to add a logger parameter to BaseRepository.save",
            language="python",
            top_k=5,
            context_limit=2,
        )

        provider.hybrid_search.assert_awaited_once_with(
            query="which files should I edit to add a logger parameter to BaseRepository.save",
            document_filter={"language": "python"},
            top_k=10,
        )
        provider.find_callers.assert_awaited_once_with(
            "BaseRepository.save",
            file_path="src/repository.py",
            max_depth=1,
        )
        provider.find_callees.assert_awaited_once_with(
            "BaseRepository.save",
            file_path="src/repository.py",
            max_depth=1,
        )
        assert results[0]["file_path"] == "src/repository.py"
        assert {row["file_path"] for row in results} >= {
            "src/repository.py",
            "src/service.py",
            "src/audit.py",
        }
        localization = results[0]["metadata"]["localization"]
        assert localization["matched_hints"] == ["BaseRepository.save"]
        assert localization["graph_score"] > 0

    @pytest.mark.asyncio
    async def test_analyze_change_impact_returns_seed_and_impacted_neighbors(
        self,
        provider_config: EmbeddingConfig,
        fake_client: FakeProximaClient,
        stub_model: StubEmbeddingModel,
    ) -> None:
        with patch(
            "victor.storage.vector_stores.proximadb_multi.create_embedding_model",
            return_value=stub_model,
        ):
            provider = ProximaDBMultiModelProvider(provider_config, client=fake_client)
            await provider.initialize()

        provider.hybrid_search = AsyncMock(
            return_value=[
                {
                    "id": "hit:1",
                    "file_path": "src/repository.py",
                    "symbol_name": "BaseRepository.save",
                    "content": "class BaseRepository:\n    def save(self, entity): ...",
                    "score": 0.84,
                    "sources": ["vector", "document"],
                    "metadata": {
                        "file_path": "src/repository.py",
                        "language": "python",
                        "qualified_name": "BaseRepository.save",
                    },
                }
            ]
        )
        provider.find_callers = AsyncMock(
            return_value=[
                {
                    "id": "caller:1",
                    "name": "UserService.create_user",
                    "file_path": "src/service.py",
                    "line_start": 12,
                    "line_end": 18,
                    "labels": ["Function"],
                    "metadata": {"name": "UserService.create_user"},
                }
            ]
        )
        provider.find_callees = AsyncMock(
            return_value=[
                {
                    "id": "callee:1",
                    "name": "AuditLogger.log_save",
                    "file_path": "src/audit.py",
                    "line_start": 4,
                    "line_end": 8,
                    "labels": ["Function"],
                    "metadata": {"name": "AuditLogger.log_save"},
                }
            ]
        )

        results = await provider.analyze_change_impact(
            "what breaks if I change BaseRepository.save",
            language="python",
            top_k=5,
            context_limit=2,
        )

        provider.hybrid_search.assert_awaited_once_with(
            query="what breaks if I change BaseRepository.save",
            document_filter={"language": "python"},
            top_k=10,
        )
        provider.find_callers.assert_awaited_once_with(
            "BaseRepository.save",
            file_path="src/repository.py",
            max_depth=2,
        )
        provider.find_callees.assert_awaited_once_with(
            "BaseRepository.save",
            file_path="src/repository.py",
            max_depth=1,
        )
        assert results[0]["file_path"] == "src/repository.py"
        assert {row["file_path"] for row in results} >= {
            "src/repository.py",
            "src/service.py",
            "src/audit.py",
        }
        impact = results[0]["metadata"]["impact"]
        assert impact["matched_hints"] == ["BaseRepository.save"]
        assert impact["graph_score"] > 0
