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


def create_graph_store(
    name: str = "sqlite",
    project_path: Optional[Path] = None,
    *,
    backend: Optional[str] = None,
) -> GraphStoreProtocol:
    """Create a graph store by name.

    Args:
        name: Backend name (sqlite, memory, duckdb)
        project_path: Path to project root. If None, uses current directory.
        backend: Alias for name (used by victor-coding shim). Takes priority if set.

    Returns:
        GraphStoreProtocol implementation
    """
    selected_backend = str(backend if backend is not None else name or "sqlite").lower()
    if selected_backend == "sqlite":
        return SqliteGraphStore(project_path=project_path)
    if selected_backend == "memory":
        return MemoryGraphStore()
    if selected_backend == "duckdb":
        if DuckDBGraphStore is None:
            raise ValueError("DuckDB graph backend requested but duckdb is not installed")
        return DuckDBGraphStore(project_path)
    raise ValueError(f"Unsupported graph store backend: {selected_backend}")
