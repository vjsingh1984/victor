"""SDK host adapters for provider runtime helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from victor.providers.base import Message
    from victor.providers.registry import ProviderRegistry

__all__ = [
    "Message",
    "ProviderRegistry",
]

_LAZY_IMPORTS = {
    "Message": "victor.providers.base",
    "ProviderRegistry": "victor.providers.registry",
}


def __getattr__(name: str):
    """Resolve provider helpers lazily from the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'victor_sdk.provider_runtime' has no attribute {name!r}")

    module = importlib.import_module(module_name)
    return getattr(module, name)
