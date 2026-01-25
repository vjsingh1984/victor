# Pluggable graph store factory.
from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

from victor.storage.graph.protocol import GraphStoreProtocol
from victor.storage.graph.sqlite_store import SqliteGraphStore
from victor.storage.graph.memory_store import MemoryGraphStore

if TYPE_CHECKING:
    from victor.storage.graph.duckdb_store import DuckDBGraphStore

try:
    from victor.storage.graph.duckdb_store import DuckDBGraphStore as _DuckDBGraphStore
    _duckdb_available: type[_DuckDBGraphStore] | None = _DuckDBGraphStore
except Exception:
    _duckdb_available = None

# Type alias for external use
if TYPE_CHECKING:
    DuckDBGraphStore = _DuckDBGraphStore
else:
    DuckDBGraphStore: Optional[Type[_DuckDBGraphStore]] = _duckdb_available


def create_graph_store(
    name: str = "sqlite",
    project_path: Optional[Path] = None,
) -> GraphStoreProtocol:
    """Create a graph store by name.

    Args:
        name: Backend name (sqlite, memory, duckdb)
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
        if _duckdb_available is None:
            raise ValueError("DuckDB graph backend requested but duckdb is not installed")
        if project_path is None:
            raise ValueError("DuckDB graph backend requires a project_path")
        return _duckdb_available(project_path)
    raise ValueError(f"Unsupported graph store backend: {name}")
