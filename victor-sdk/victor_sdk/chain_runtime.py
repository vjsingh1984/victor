"""SDK host adapters for chain registry runtime helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from victor.framework.chains import ChainRegistry, get_chain_registry

__all__ = [
    "ChainRegistry",
    "get_chain_registry",
]

_LAZY_IMPORTS = {
    "ChainRegistry": "victor.framework.chains",
    "get_chain_registry": "victor.framework.chains",
}


def __getattr__(name: str) -> Any:
    """Resolve chain registry helpers lazily from the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'victor_sdk.chain_runtime' has no attribute {name!r}")

    module = importlib.import_module(module_name)
    return getattr(module, name)
