"""Runtime bridge: project database access."""

# ruff: noqa: F822

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["get_project_database"]

_LAZY_IMPORTS = {
    "get_project_database": "victor.core.database",
}


def __getattr__(name: str) -> Any:
    """Lazy-bridge to the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    return getattr(module, name)
