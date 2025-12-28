# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest

from victor.graph.registry import create_graph_store
from victor.graph.sqlite_store import SqliteGraphStore
from victor.graph.memory_store import MemoryGraphStore

try:
    from victor.graph.duckdb_store import DuckDBGraphStore
except Exception:  # pragma: no cover - optional backend
    DuckDBGraphStore = None
try:
    import importlib.util

    DUCKDB_AVAILABLE = importlib.util.find_spec("duckdb") is not None
except Exception:  # pragma: no cover - defensive
    DUCKDB_AVAILABLE = False
try:
    from victor.graph.lancedb_store import LanceDBGraphStore
except Exception:  # pragma: no cover - optional backend
    LanceDBGraphStore = None
try:
    import importlib.util

    LANCEDB_AVAILABLE = importlib.util.find_spec("lancedb") is not None
except Exception:  # pragma: no cover - defensive
    LANCEDB_AVAILABLE = False
try:
    from victor.graph.neo4j_store import Neo4jGraphStore
except Exception:  # pragma: no cover - optional backend
    Neo4jGraphStore = None
try:
    import importlib.util

    NEO4J_AVAILABLE = importlib.util.find_spec("neo4j") is not None
except Exception:  # pragma: no cover - defensive
    NEO4J_AVAILABLE = False


def test_create_graph_store_sqlite(tmp_path: Path):
    db_path = tmp_path / "graph.db"
    store = create_graph_store("sqlite", db_path)
    assert isinstance(store, SqliteGraphStore)


def test_create_graph_store_invalid(tmp_path: Path):
    with pytest.raises(ValueError):
        create_graph_store("unknown", tmp_path / "db")


def test_create_graph_store_memory():
    store = create_graph_store("memory", Path(":memory:"))
    assert isinstance(store, MemoryGraphStore)


@pytest.mark.skipif(DuckDBGraphStore is None or not DUCKDB_AVAILABLE, reason="duckdb not installed")
def test_create_graph_store_duckdb(tmp_path: Path):
    db_path = tmp_path / "graph.db"
    store = create_graph_store("duckdb", db_path)
    assert isinstance(store, DuckDBGraphStore)


@pytest.mark.skipif(
    LanceDBGraphStore is None or not LANCEDB_AVAILABLE, reason="lancedb not installed"
)
def test_create_graph_store_lancedb(tmp_path: Path):
    db_path = tmp_path / "graph.db"
    with pytest.raises(NotImplementedError):
        create_graph_store("lancedb", db_path)


@pytest.mark.skipif(Neo4jGraphStore is None or not NEO4J_AVAILABLE, reason="neo4j not installed")
def test_create_graph_store_neo4j(tmp_path: Path):
    db_path = tmp_path / "graph.db"
    with pytest.raises(NotImplementedError):
        create_graph_store("neo4j", db_path)
