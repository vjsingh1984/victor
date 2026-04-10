"""Base Tool Dependency Provider — backward compatibility shim.

Canonical implementations have been promoted to victor-sdk.
This module re-exports them for backward compatibility.

Usage (preferred — import from SDK):
    from victor_sdk.verticals.tool_dependencies import BaseToolDependencyProvider

Usage (backward compat — still works):
    from victor.core.tool_dependency_base import BaseToolDependencyProvider
"""

from __future__ import annotations

# Re-export from SDK — canonical implementations now live in victor_sdk.
from victor_sdk.verticals.tool_dependencies import (  # noqa: F401
    BaseToolDependencyProvider,
    EmptyToolDependencyProvider,
    ToolDependencyConfig,
    ToolDependencyLoadError,
    create_vertical_tool_dependency_provider,
)

__all__ = [
    "BaseToolDependencyProvider",
    "EmptyToolDependencyProvider",
    "ToolDependencyConfig",
    "ToolDependencyLoadError",
    "create_vertical_tool_dependency_provider",
]
