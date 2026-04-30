"""SDK host adapters for capability protocols and lazy capability helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from victor.framework.vertical_protocols import (
        CodebaseIndexFactoryProtocol,
        EditorProtocol,
        TreeSitterParserProtocol,
    )
    from victor.core.search.indexer import detect_enhanced_index_factory

__all__ = [
    "TreeSitterParserProtocol",
    "EditorProtocol",
    "CodebaseIndexFactoryProtocol",
    "create_lazy_capability_proxy",
    "detect_enhanced_index_factory",
]

_LAZY_IMPORTS = {
    "TreeSitterParserProtocol": "victor.framework.vertical_protocols",
    "EditorProtocol": "victor.framework.vertical_protocols",
    "CodebaseIndexFactoryProtocol": "victor.framework.vertical_protocols",
    "detect_enhanced_index_factory": "victor.core.search.indexer",
}


def __getattr__(name: str) -> Any:
    """Resolve capability helpers lazily from the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'victor_sdk.capability_runtime' has no attribute {name!r}")

    module = importlib.import_module(module_name)
    return getattr(module, name)


def create_lazy_capability_proxy(provider: Callable[[], Any] | Any) -> Any:
    """Wrap a lazy capability provider using the Victor host proxy implementation."""
    module = importlib.import_module("victor.core.plugins.context")
    proxy_type = module._LazyCapabilityProxy
    return proxy_type(provider)
