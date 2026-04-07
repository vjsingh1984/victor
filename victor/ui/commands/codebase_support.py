"""Shared compatibility helpers for codebase-analysis command surfaces."""

from __future__ import annotations

from types import ModuleType


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
    """Resolve a single analyzer export and raise ``ImportError`` when it is unavailable."""

    module = load_codebase_analyzer_module()
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise ImportError(f"codebase_analyzer does not expose required symbol '{name}'") from exc
