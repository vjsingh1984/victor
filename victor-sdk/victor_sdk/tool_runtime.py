"""SDK host adapters for runtime tool helper types."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from victor.framework.tools import ToolSet as RuntimeToolSet

__all__ = [
    "RuntimeToolSet",
]

_LAZY_IMPORTS = {
    "RuntimeToolSet": "victor.framework.tools",
}


def __getattr__(name: str) -> Any:
    """Resolve runtime tool helpers lazily from the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'victor_sdk.tool_runtime' has no attribute {name!r}")

    module = importlib.import_module(module_name)
    if name == "RuntimeToolSet":
        return module.ToolSet
    return getattr(module, name)
