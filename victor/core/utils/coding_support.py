"""Compatibility helpers for extracted coding-package integrations.

These helpers centralize the migration bridge from legacy in-tree coding
modules to the extracted ``victor-coding`` package so runtime/UI surfaces do
not duplicate fallback import logic.
"""

from __future__ import annotations

from types import ModuleType
from typing import Callable, TypeVar, cast

T = TypeVar("T")


def _load_extracted_codebase_analyzer() -> ModuleType:
    """Load the extracted analyzer from victor-coding."""

    from victor_coding import codebase_analyzer as module

    return module


def _load_legacy_codebase_analyzer() -> ModuleType:
    """Load the legacy in-tree analyzer as a compatibility fallback."""

    from victor.verticals.contrib.coding import codebase_analyzer as module

    return module


_CODEBASE_ANALYZER_LOADERS = (
    _load_extracted_codebase_analyzer,
    _load_legacy_codebase_analyzer,
)


def load_codebase_analyzer_module() -> ModuleType:
    """Load the extracted codebase analyzer module, with legacy fallback."""

    last_error: ImportError | None = None
    for loader in _CODEBASE_ANALYZER_LOADERS:
        try:
            module = loader()
        except ImportError as exc:
            last_error = exc
            continue
        return module

    raise ImportError(
        "codebase_analyzer requires the victor-coding package to expose "
        "'victor_coding.codebase_analyzer'"
    ) from last_error


def load_codebase_analyzer_attr(name: str) -> object:
    """Resolve a single analyzer export and raise ``ImportError`` when unavailable."""

    module = load_codebase_analyzer_module()
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise ImportError(f"codebase_analyzer does not expose required symbol '{name}'") from exc


def _load_extracted_tree_sitter_manager() -> ModuleType:
    """Load the extracted tree-sitter manager from victor-coding."""

    from victor_coding.codebase import tree_sitter_manager as module

    return module


def _load_legacy_tree_sitter_manager() -> ModuleType:
    """Load the legacy in-tree tree-sitter manager as a compatibility fallback."""

    from victor.verticals.contrib.coding.codebase import tree_sitter_manager as module

    return module


_TREE_SITTER_MANAGER_LOADERS = (
    _load_extracted_tree_sitter_manager,
    _load_legacy_tree_sitter_manager,
)


def _load_extracted_analyze_command_app() -> object:
    """Load the extracted coding analyze CLI app from victor-coding."""

    from victor_coding.commands.analyze import app as analyze_app

    return analyze_app


def _load_legacy_analyze_command_app() -> object:
    """Load the legacy in-tree coding analyze CLI app as a compatibility fallback."""

    from victor.verticals.contrib.coding.commands.analyze import app as analyze_app

    return analyze_app


_ANALYZE_COMMAND_APP_LOADERS = (
    _load_extracted_analyze_command_app,
    _load_legacy_analyze_command_app,
)


def load_tree_sitter_get_parser() -> Callable[[str], object]:
    """Resolve the canonical ``get_parser`` function for coding tree-sitter support."""

    last_error: ImportError | None = None
    for loader in _TREE_SITTER_MANAGER_LOADERS:
        try:
            module = loader()
        except ImportError as exc:
            last_error = exc
            continue

        get_parser = getattr(module, "get_parser", None)
        if callable(get_parser):
            return cast(Callable[[str], object], get_parser)

    raise ImportError(
        "tree_sitter_manager requires the victor-coding package to expose "
        "'victor_coding.codebase.tree_sitter_manager.get_parser'"
    ) from last_error


def load_coding_analyze_app() -> object:
    """Resolve the coding analyze CLI app, with extracted-first fallback order."""

    last_error: ImportError | None = None
    for loader in _ANALYZE_COMMAND_APP_LOADERS:
        try:
            return loader()
        except ImportError as exc:
            last_error = exc
            continue

    raise ImportError(
        "coding analyze command requires victor-coding to expose "
        "'victor_coding.commands.analyze.app'"
    ) from last_error


__all__ = [
    "load_codebase_analyzer_module",
    "load_codebase_analyzer_attr",
    "load_tree_sitter_get_parser",
    "load_coding_analyze_app",
]
