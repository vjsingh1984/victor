"""SDK host adapters for framework safety policy helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from victor.framework.config import SafetyConfig, SafetyEnforcer, SafetyLevel, SafetyRule
    from victor.framework.safety import create_file_safety_rules, create_git_safety_rules

__all__ = [
    "SafetyConfig",
    "SafetyEnforcer",
    "SafetyLevel",
    "SafetyRule",
    "create_file_safety_rules",
    "create_git_safety_rules",
]

_LAZY_IMPORTS = {
    "SafetyConfig": "victor.framework.config",
    "SafetyEnforcer": "victor.framework.config",
    "SafetyLevel": "victor.framework.config",
    "SafetyRule": "victor.framework.config",
    "create_file_safety_rules": "victor.framework.safety",
    "create_git_safety_rules": "victor.framework.safety",
}


def __getattr__(name: str) -> Any:
    """Resolve safety policy helpers lazily from the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'victor_sdk.safety_policy' has no attribute {name!r}")

    module = importlib.import_module(module_name)
    return getattr(module, name)
