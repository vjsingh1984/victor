"""Runtime bridge: feature flag access."""

# ruff: noqa: F822

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["get_feature_flag_manager", "FeatureFlag", "is_feature_enabled"]

_LAZY_IMPORTS = {
    "get_feature_flag_manager": "victor.core.feature_flags",
    "FeatureFlag": "victor.core.feature_flags",
    "is_feature_enabled": "victor.core.feature_flags",
}


def __getattr__(name: str) -> Any:
    """Lazy-bridge to the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    return getattr(module, name)
