"""SDK host adapters for init synthesis runtime helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from victor.framework.init_synthesizer import InitSynthesizer

__all__ = ["InitSynthesizer"]

_LAZY_IMPORTS = {
    "InitSynthesizer": "victor.framework.init_synthesizer",
}


def __getattr__(name: str):
    """Resolve init synthesis helpers lazily from the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'victor_sdk.init_runtime' has no attribute {name!r}")

    module = importlib.import_module(module_name)
    return getattr(module, name)
