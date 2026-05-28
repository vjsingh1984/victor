"""Unit tests for SQLite + LanceDB to ProximaDB migration helpers."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from victor.core.schema import Tables
from victor.storage.vector_stores.base import EmbeddingConfig
from victor.storage.vector_stores.proximadb_migration import SqliteLanceDBMigration
from victor.storage.vector_stores.proximadb_multi import ProximaDBMultiModelProvider


class StubEmbeddingModel:
    """Minimal async embedding model used by migration tests."""

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension

    async def initialize(self) -> None:
        return None

    async def embed_text(self, text: str) -> List[float]:
        seed = float((len(text) % 7) + 1)
        return [seed, seed / 2.0, seed / 3.0, seed / 4.0]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [await self.embed_text(text) for text in texts]

    def get_dimension(self) -> int:
        return self._dimension

    async def close(self) -> None:
        return None


@dataclass
class FakeSearchHit:
    id: str
    score: float
    source: Optional[str]
    metadata: Dict[str, Any]


class FakeProximaClient:
    """In-memory ProximaDB stand-in for migration tests."""

    def __init__(self) -> None:
        self.collections: Dict[str, List[Any]] = defaultdict(list)
        self.created_collections: List[str] = []
        self.created_graphs: List[str] = []
        self.graph_nodes: List[Dict[str, Any]] = []
        self.graph_edges: List[Dict[str, Any]] = []

    def create_collection(
        self, name: str, config: Any = None, **kwargs: Any
    ) -> Dict[str, Any]:
        del config, kwargs
        if name not in self.created_collections:
            self.created_collections.append(name)
        self.collections.setdefault(name, [])
        return {"name": name}

    def get_collection(self, name: str) -> Dict[str, Any]:
        return {"name": name, "record_count": len(self.collections.get(name, []))}

    def delete_collection(self, name: str) -> None:
        self.collections.pop(name, None)

    def insert_vectors(
        self, collection: str, records: List[Any], **kwargs: Any
    ) -> None:
        del kwargs
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
        hits = []
        for index, record in enumerate(self.collections.get(collection, [])):
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
        return {"graph_id": graph_id}

    def delete_graph(self, graph_id: str) -> None:
        del graph_id

    def create_node(
        self,
        node_id: str,
        labels: List[str],
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.graph_nodes.append(
            {
                "id": node_id,
                "labels": list(labels),
                "properties": dict(properties or {}),
            }
        )
        return self.graph_nodes[-1]

    def create_edge(
        self,
        edge_id: str,
        from_node_id: str,
        to_node_id: str,
        edge_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.graph_edges.append(
            {
                "id": edge_id,
                "from_node_id": from_node_id,
                "to_node_id": to_node_id,
                "edge_type": edge_type,
                "properties": dict(properties or {}),
            }
        )
        return self.graph_edges[-1]


class FakeLanceHead:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = rows

    def to_list(self) -> List[Dict[str, Any]]:
        return list(self._rows)


class FakeLanceTable:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = rows

    def count_rows(self) -> int:
        return len(self._rows)

    def head(self, size: int) -> FakeLanceHead:
        return FakeLanceHead(self._rows[:size])


class FakeLanceDB:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = rows

    def table_names(self) -> List[str]:
        return ["embeddings"]

    def open_table(self, name: str) -> FakeLanceTable:
        assert name == "embeddings"
        return FakeLanceTable(self._rows)


@pytest.fixture
def provider_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        vector_store="proximadb_multi",
        embedding_model_type="sentence-transformers",
        embedding_model_name="all-MiniLM-L12-v2",
        distance_metric="cosine",
        extra_config={
            "workspace": "victor_migration_repo",
            "dimension": 4,
            "chunk_size": 12,
            "chunk_overlap": 1,
        },
    )


def _create_graph_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(f"""
            CREATE TABLE {Tables.GRAPH_NODE} (
                node_id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                file TEXT NOT NULL,
                line INTEGER,
                end_line INTEGER,
                lang TEXT,
                signature TEXT,
                docstring TEXT,
                parent_id TEXT,
                embedding_ref TEXT,
                metadata TEXT
            );
            CREATE TABLE {Tables.GRAPH_EDGE} (
                src TEXT NOT NULL,
                dst TEXT NOT NULL,
                type TEXT NOT NULL,
                weight REAL,
                metadata TEXT
            );
            """)
        conn.executemany(
            f"""
            INSERT INTO {Tables.GRAPH_NODE}(
                node_id, type, name, file, line, end_line, lang,
                signature, docstring, parent_id, embedding_ref, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "function:src/main.py:main",
                    "function",
                    "main",
                    "src/main.py",
                    3,
                    4,
                    "python",
                    "main(data)",
                    None,
                    None,
                    "function:src/main.py:main",
                    '{"role": "entrypoint"}',
                ),
                (
                    "function:src/main.py:parse_json",
                    "function",
                    "parse_json",
                    "src/main.py",
                    6,
                    7,
                    "python",
                    "parse_json(data)",
                    None,
                    None,
                    "function:src/main.py:parse_json",
                    '{"role": "parser"}',
                ),
            ],
        )
        conn.execute(
            f"""
            INSERT INTO {Tables.GRAPH_EDGE}(src, dst, type, weight, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "function:src/main.py:main",
                "function:src/main.py:parse_json",
                "CALLS",
                1.0,
                '{"line_number": 4, "file_path": "src/main.py"}',
            ),
        )
        conn.commit()
    finally:
        conn.close()


class TestSqliteLanceDBMigration:
    @pytest.mark.asyncio
    async def test_migrate_transfers_graph_vectors_and_backfills_records(
        self,
        tmp_path: Path,
        provider_config: EmbeddingConfig,
    ) -> None:
        repo_root = tmp_path
        source_file = repo_root / "src" / "main.py"
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text(
            "import json\n\n"
            "def main(data):\n    return parse_json(data)\n\n"
            "def parse_json(data):\n    return json.loads(data)\n",
            encoding="utf-8",
        )

        graph_db = repo_root / ".victor" / "project.db"
        _create_graph_db(graph_db)
        (repo_root / ".victor" / "embeddings").mkdir(parents=True, exist_ok=True)

        fake_lance_rows = [
            {
                "id": "function:src/main.py:parse_json",
                "vector": [0.1, 0.2, 0.3, 0.4],
                "content": "def parse_json(data): return json.loads(data)",
                "file_path": "src/main.py",
                "symbol_name": "parse_json",
                "symbol_type": "function",
                "line_number": 6,
                "language": "python",
            }
        ]

        fake_client = FakeProximaClient()
        with (
            patch(
                "victor.storage.vector_stores.proximadb_multi.create_embedding_model",
                return_value=StubEmbeddingModel(dimension=4),
            ),
            patch("lancedb.connect", return_value=FakeLanceDB(fake_lance_rows)),
        ):
            provider = ProximaDBMultiModelProvider(provider_config, client=fake_client)
            migration = SqliteLanceDBMigration(
                provider=provider,
                repo_root=repo_root,
                graph_db_path=graph_db,
                lancedb_dir=repo_root / ".victor" / "embeddings",
                lancedb_table="embeddings",
            )
            summary = await migration.migrate()

        assert summary["graph_nodes_migrated"] == 2
        assert summary["graph_edges_migrated"] == 1
        assert summary["vector_records_migrated"] == 1
        assert summary["document_records_backfilled"] == 1
        assert summary["metric_records_backfilled"] >= 4
        assert summary["files_backfilled"] == 1
        assert summary["files_missing"] == 0

        migrated_vector_ids = {
            getattr(record, "id", None)
            for record in fake_client.collections["victor_migration_repo_vectors"]
        }
        assert "function:src/main.py:parse_json" in migrated_vector_ids
        assert any(
            node["id"] == "function:src/main.py:main"
            for node in fake_client.graph_nodes
        )
        assert any(edge["edge_type"] == "CALLS" for edge in fake_client.graph_edges)
        assert len(fake_client.collections["victor_migration_repo_documents"]) == 1

    @pytest.mark.asyncio
    async def test_migrate_handles_missing_legacy_stores(
        self,
        tmp_path: Path,
        provider_config: EmbeddingConfig,
    ) -> None:
        fake_client = FakeProximaClient()
        with patch(
            "victor.storage.vector_stores.proximadb_multi.create_embedding_model",
            return_value=StubEmbeddingModel(dimension=4),
        ):
            provider = ProximaDBMultiModelProvider(provider_config, client=fake_client)
            migration = SqliteLanceDBMigration(
                provider=provider,
                repo_root=tmp_path,
                graph_db_path=tmp_path / "missing_project.db",
                lancedb_dir=tmp_path / "missing_embeddings",
            )
            summary = await migration.migrate()

        assert summary["graph_nodes_migrated"] == 0
        assert summary["graph_edges_migrated"] == 0
        assert summary["vector_records_migrated"] == 0
        assert summary["document_records_backfilled"] == 0
        assert summary["metric_records_backfilled"] == 0
