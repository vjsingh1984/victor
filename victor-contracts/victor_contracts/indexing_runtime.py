"""Runtime bridge: indexing infrastructure."""

# ruff: noqa: F822

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "FileChangeEvent",
    "FileChangeType",
    "FileWatcherRegistry",
    "IndexLockRegistry",
    "ensure_project_graph_enriched",
    "GraphBuildCoordinator",
]

_LAZY_IMPORTS = {
    "FileChangeEvent": "victor.core.indexing.file_watcher",
    "FileChangeType": "victor.core.indexing.file_watcher",
    "FileWatcherRegistry": "victor.core.indexing.file_watcher",
    "IndexLockRegistry": "victor.core.indexing.index_lock",
    "ensure_project_graph_enriched": "victor.core.indexing.graph_enrichment",
    "GraphBuildCoordinator": "victor.core.indexing.graph_manager",
}


def __getattr__(name: str) -> Any:
    """Lazy-bridge to the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    return getattr(module, name)
