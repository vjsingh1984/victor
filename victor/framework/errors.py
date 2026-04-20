"""Backward-compatible re-exports from victor.core.errors.

All error classes now live in victor.core.errors. This module re-exports
them for backward compatibility and will be removed in v0.10.0.

Migrate imports:
    # Before
    from victor.framework.errors import AgentError, CancellationError

    # After
    from victor.core.errors import AgentError, CancellationError
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.framework.errors is deprecated and will be removed in v0.10.0. "
    "Import directly from victor.core.errors instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.core.errors import (  # noqa: E402, F401 (re-exports)
    AgentError,
    BudgetExhaustedError,
    CancellationError,
    ConfigurationError,
    EdgeResolutionError,
    ProviderError,
    StateTransitionError,
    ToolError,
)

__all__ = [
    "AgentError",
    "BudgetExhaustedError",
    "CancellationError",
    "ConfigurationError",
    "EdgeResolutionError",
    "ProviderError",
    "StateTransitionError",
    "ToolError",
]
