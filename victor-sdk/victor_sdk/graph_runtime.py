"""SDK host adapters for graph runtime helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from victor.framework.graph import END, StateGraph

__all__ = [
    "END",
    "StateGraph",
]

_LAZY_IMPORTS = {
    "END": "victor.framework.graph",
    "StateGraph": "victor.framework.graph",
}


def __getattr__(name: str) -> Any:
    """Resolve graph helpers lazily from the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'victor_sdk.graph_runtime' has no attribute {name!r}")

    module = importlib.import_module(module_name)
    return getattr(module, name)
