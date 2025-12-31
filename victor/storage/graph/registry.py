# Pluggable graph store factory.
from __future__ import annotations

from pathlib import Path
from typing import Optional

from victor.storage.graph.protocol import GraphStoreProtocol
from victor.storage.graph.sqlite_store import SqliteGraphStore
from victor.storage.graph.memory_store import MemoryGraphStore

try:
    from victor.storage.graph.duckdb_store import DuckDBGraphStore
except Exception:
    DuckDBGraphStore = None

try:
    from victor.storage.graph.lancedb_store import LanceDBGraphStore
except Exception:
    LanceDBGraphStore = None

try:
    from victor.storage.graph.neo4j_store import Neo4jGraphStore
except Exception:
    Neo4jGraphStore = None


def create_graph_store(
    name: str = "sqlite",
    project_path: Optional[Path] = None,
) -> GraphStoreProtocol:
    """Create a graph store by name.

    Args:
        name: Backend name (sqlite, memory, duckdb, lancedb, neo4j)
        project_path: Path to project root. If None, uses current directory.

    Returns:
        GraphStoreProtocol implementation
    """
    backend = (name or "sqlite").lower()
    if backend == "sqlite":
        return SqliteGraphStore(project_path=project_path)
    if backend == "memory":
        return MemoryGraphStore()
    if backend == "duckdb":
        if DuckDBGraphStore is None:
            raise ValueError("DuckDB graph backend requested but duckdb is not installed")
        return DuckDBGraphStore(project_path)
    if backend == "lancedb":
        if LanceDBGraphStore is None:
            raise ValueError("LanceDB graph backend requested but lancedb is not installed")
        return LanceDBGraphStore(project_path)
    if backend == "neo4j":
        if Neo4jGraphStore is None:
            raise ValueError("Neo4j graph backend requested but neo4j driver is not installed")
        return Neo4jGraphStore(project_path)
    raise ValueError(f"Unsupported graph store backend: {name}")
