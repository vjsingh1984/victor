"""Shared compatibility helpers for codebase-analysis command surfaces."""

from __future__ import annotations

import importlib
from types import ModuleType

_CODEBASE_ANALYZER_MODULES = (
    "victor_coding.codebase_analyzer",
    "victor.verticals.contrib.coding.codebase_analyzer",
)


def load_codebase_analyzer_module() -> ModuleType:
    """Load the extracted codebase analyzer module, with legacy fallback."""

    last_error: ImportError | None = None
    for module_name in _CODEBASE_ANALYZER_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            last_error = exc
            continue
        return module

    raise ImportError(
        "codebase_analyzer requires the victor-coding package to expose "
        "'victor_coding.codebase_analyzer'"
    ) from last_error


def load_codebase_analyzer_attr(name: str) -> object:
    """Resolve a single analyzer export and raise ``ImportError`` when it is unavailable."""

    module = load_codebase_analyzer_module()
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise ImportError(
            f"codebase_analyzer does not expose required symbol '{name}'"
        ) from exc
