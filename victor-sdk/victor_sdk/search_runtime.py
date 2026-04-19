"""SDK host adapters for search runtime helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from victor.framework.search.query_expansion import (
        ExpandedQuery,
        QueryExpander,
        QueryExpansionConfig,
        QueryExpanderProtocol,
        create_query_expander,
    )

__all__ = [
    "ExpandedQuery",
    "QueryExpander",
    "QueryExpansionConfig",
    "QueryExpanderProtocol",
    "create_query_expander",
]

_LAZY_IMPORTS = {
    "ExpandedQuery": "victor.framework.search.query_expansion",
    "QueryExpander": "victor.framework.search.query_expansion",
    "QueryExpansionConfig": "victor.framework.search.query_expansion",
    "QueryExpanderProtocol": "victor.framework.search.query_expansion",
    "create_query_expander": "victor.framework.search.query_expansion",
}


def __getattr__(name: str):
    """Resolve search helpers lazily from the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'victor_sdk.search_runtime' has no attribute {name!r}")

    module = importlib.import_module(module_name)
    return getattr(module, name)
