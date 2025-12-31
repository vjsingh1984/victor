# Pluggable graph store factory.
from __future__ import annotations

from pathlib import Path

from victor.graph.protocol import GraphStoreProtocol
from victor.graph.sqlite_store import SqliteGraphStore
from victor.graph.memory_store import MemoryGraphStore

try:
    from victor.graph.duckdb_store import DuckDBGraphStore
except Exception:
    DuckDBGraphStore = None

try:
    from victor.graph.lancedb_store import LanceDBGraphStore
except Exception:
    LanceDBGraphStore = None

try:
    from victor.graph.neo4j_store import Neo4jGraphStore
except Exception:
    Neo4jGraphStore = None


def create_graph_store(name: str, path: Path) -> GraphStoreProtocol:
    """Create a graph store by name.

    Args:
        name: Backend name (currently supports "sqlite")
        path: Path to the underlying storage (e.g., DB file)

    Returns:
        GraphStoreProtocol implementation
    """
    backend = (name or "sqlite").lower()
    if backend == "sqlite":
        return SqliteGraphStore(path)
    if backend == "memory":
        return MemoryGraphStore()
    if backend == "duckdb":
        if DuckDBGraphStore is None:
            raise ValueError("DuckDB graph backend requested but duckdb is not installed")
        return DuckDBGraphStore(path)
    if backend == "lancedb":
        if LanceDBGraphStore is None:
            raise ValueError("LanceDB graph backend requested but lancedb is not installed")
        return LanceDBGraphStore(path)
    if backend == "neo4j":
        if Neo4jGraphStore is None:
            raise ValueError("Neo4j graph backend requested but neo4j driver is not installed")
        return Neo4jGraphStore(path)
    raise ValueError(f"Unsupported graph store backend: {name}")
