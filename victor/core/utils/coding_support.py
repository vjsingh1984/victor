"""Compatibility helpers for extracted coding-package integrations.

These helpers centralize the migration bridge from legacy in-tree coding
modules to the extracted ``victor-coding`` package. Discovery is done via
entry points and importlib — core never imports ``victor_coding`` directly.

Discovery chain for each capability:
1. Entry point: ``victor.sdk.capabilities`` group (registered by victor-coding)
2. Importlib: Try ``victor_coding.*`` via importlib.import_module (no static import)
3. Legacy: Try ``victor.verticals.contrib.coding.*`` fallback
4. Raise ImportError with helpful message
"""

from __future__ import annotations

import importlib
import logging
from types import ModuleType
from typing import Callable, TypeVar, cast

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _try_import(module_path: str) -> ModuleType:
    """Import a module by dotted path. Raises ImportError on failure."""
    return importlib.import_module(module_path)


def _try_entry_point(group: str, name: str) -> object:
    """Load a named entry point from a group. Returns None if not found."""
    try:
        from victor.framework.entry_point_registry import get_entry_point

        return get_entry_point(group, name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Codebase Analyzer
# ---------------------------------------------------------------------------

_CODEBASE_ANALYZER_MODULES = (
    "victor_coding.codebase_analyzer",
    "victor.verticals.contrib.coding.codebase_analyzer",
)


def load_codebase_analyzer_module() -> ModuleType:
    """Load the codebase analyzer module via discovery chain.

    Discovery order:
    1. Entry point ``victor.sdk.capabilities::codebase_analyzer``
    2. ``victor_coding.codebase_analyzer`` via importlib
    3. Legacy ``victor.verticals.contrib.coding.codebase_analyzer``
    """
    # Try entry point first
    ep = _try_entry_point("victor.sdk.capabilities", "codebase_analyzer")
    if ep is not None and hasattr(ep, "__module__"):
        try:
            return importlib.import_module(ep.__module__)
        except ImportError:
            pass

    # Try module paths
    last_error: ImportError | None = None
    for module_path in _CODEBASE_ANALYZER_MODULES:
        try:
            return _try_import(module_path)
        except ImportError as exc:
            last_error = exc
            continue

    raise ImportError(
        "codebase_analyzer requires the victor-coding package. "
        "Install with: pip install victor-coding"
    ) from last_error


def load_codebase_analyzer_attr(name: str) -> object:
    """Resolve a single analyzer export and raise ImportError when unavailable."""
    module = load_codebase_analyzer_module()
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise ImportError(f"codebase_analyzer does not expose required symbol '{name}'") from exc


# ---------------------------------------------------------------------------
# Tree-Sitter Manager
# ---------------------------------------------------------------------------

_TREE_SITTER_MODULES = (
    "victor_coding.codebase.tree_sitter_manager",
    "victor.verticals.contrib.coding.codebase.tree_sitter_manager",
)


def load_tree_sitter_get_parser() -> Callable[[str], object]:
    """Resolve the canonical get_parser function for tree-sitter support.

    Discovery order:
    1. Entry point ``victor.sdk.capabilities::tree_sitter``
    2. ``victor_coding.codebase.tree_sitter_manager`` via importlib
    3. Legacy ``victor.verticals.contrib.coding.codebase.tree_sitter_manager``
    """
    # Try entry point first
    ep = _try_entry_point("victor.sdk.capabilities", "tree_sitter")
    if ep is not None:
        get_parser = getattr(ep, "get_parser", None)
        if callable(get_parser):
            return cast(Callable[[str], object], get_parser)

    # Try module paths
    last_error: ImportError | None = None
    for module_path in _TREE_SITTER_MODULES:
        try:
            module = _try_import(module_path)
        except ImportError as exc:
            last_error = exc
            continue
        get_parser = getattr(module, "get_parser", None)
        if callable(get_parser):
            return cast(Callable[[str], object], get_parser)

    raise ImportError(
        "tree_sitter_manager requires the victor-coding package. "
        "Install with: pip install victor-coding"
    ) from last_error


# ---------------------------------------------------------------------------
# Coding Analyze CLI App
# ---------------------------------------------------------------------------

_ANALYZE_APP_MODULES = (
    "victor_coding.commands.analyze",
    "victor.verticals.contrib.coding.commands.analyze",
)


def load_coding_analyze_app() -> object:
    """Resolve the coding analyze CLI app.

    Discovery order:
    1. Entry point ``victor.commands::analyze``
    2. ``victor_coding.commands.analyze`` via importlib
    3. Legacy ``victor.verticals.contrib.coding.commands.analyze``
    """
    # Try entry point first
    ep = _try_entry_point("victor.commands", "analyze")
    if ep is not None:
        return ep

    # Try module paths
    last_error: ImportError | None = None
    for module_path in _ANALYZE_APP_MODULES:
        try:
            module = _try_import(module_path)
        except ImportError as exc:
            last_error = exc
            continue
        app = getattr(module, "app", None)
        if app is not None:
            return app

    raise ImportError(
        "coding analyze command requires victor-coding. " "Install with: pip install victor-coding"
    ) from last_error


__all__ = [
    "load_codebase_analyzer_module",
    "load_codebase_analyzer_attr",
    "load_tree_sitter_get_parser",
    "load_coding_analyze_app",
]
