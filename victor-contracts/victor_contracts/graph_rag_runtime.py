"""Runtime bridge: graph RAG multi-hop retrieval."""

# ruff: noqa: F822

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["MultiHopRetriever", "RetrievalConfig"]

_LAZY_IMPORTS = {
    "MultiHopRetriever": "victor.core.graph_rag",
    "RetrievalConfig": "victor.core.graph_rag",
}


def __getattr__(name: str) -> Any:
    """Lazy-bridge to the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    return getattr(module, name)
