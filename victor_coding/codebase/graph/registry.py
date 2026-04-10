# Pluggable graph store factory.
# Isolates all victor.storage.graph imports to this single module.
# External code should use create_graph_store() and GraphStoreProtocol.
from __future__ import annotations

import logging
from typing import Any, Optional

# SDK protocol (zero victor-ai dependency)
from victor_sdk.verticals.protocols.storage import GraphStoreProtocol

logger = logging.getLogger(__name__)

# Concrete implementations (require victor-ai at runtime)
try:
    from victor.storage.graph.registry import (
        create_graph_store as _create_graph_store,
    )
    from victor.storage.graph.sqlite_store import SqliteGraphStore
    from victor.storage.graph.memory_store import MemoryGraphStore

    _VICTOR_AVAILABLE = True
except ImportError:
    _VICTOR_AVAILABLE = False
    SqliteGraphStore = None  # type: ignore[assignment,misc]
    MemoryGraphStore = None  # type: ignore[assignment,misc]


def create_graph_store(
    project_path: Optional[str] = None,
    backend: str = "sqlite",
    **kwargs: Any,
) -> Optional[Any]:
    """Create a graph store via victor's registry.

    Returns None if victor-ai is not installed.
    """
    if not _VICTOR_AVAILABLE:
        logger.debug("victor-ai not installed — graph store unavailable")
        return None
    return _create_graph_store(project_path=project_path, backend=backend, **kwargs)


__all__ = [
    "create_graph_store",
    "SqliteGraphStore",
    "MemoryGraphStore",
    "GraphStoreProtocol",
]
