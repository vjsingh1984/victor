"""SDK host adapters for framework middleware helpers."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.code_correction_middleware import (
        CodeCorrectionConfig,
        CodeCorrectionMiddleware,
        CorrectionResult,
    )
    from victor.evaluation.correction.types import CodeValidationResult, Language
    from victor.framework.middleware import GitSafetyMiddleware, MiddlewareComposer

__all__ = [
    "CodeCorrectionConfig",
    "CodeCorrectionMiddleware",
    "CodeValidationResult",
    "CorrectionResult",
    "GitSafetyMiddleware",
    "Language",
    "MiddlewareComposer",
]

_LAZY_IMPORTS = {
    "CodeCorrectionConfig": "victor.agent.code_correction_middleware",
    "CodeCorrectionMiddleware": "victor.agent.code_correction_middleware",
    "CorrectionResult": "victor.agent.code_correction_middleware",
    "CodeValidationResult": "victor.evaluation.correction.types",
    "Language": "victor.evaluation.correction.types",
    "GitSafetyMiddleware": "victor.framework.middleware",
    "MiddlewareComposer": "victor.framework.middleware",
}


def __getattr__(name: str):
    """Resolve middleware helpers lazily from the Victor host runtime."""
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'victor_sdk.middleware_runtime' has no attribute {name!r}")

    module = importlib.import_module(module_name)
    return getattr(module, name)
