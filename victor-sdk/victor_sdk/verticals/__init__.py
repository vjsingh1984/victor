"""Vertical protocols and base classes.

This module provides protocol definitions and base classes for creating
Victor verticals without runtime dependencies.
"""

from victor_sdk.verticals.protocols.base import VerticalBase
from victor_sdk.verticals.protocols import (
    ToolProvider,
    ToolSelectionStrategy,
    SafetyProvider,
    SafetyExtension,
    PromptProvider,
    PromptContributor,
    WorkflowProvider,
    HandlerProvider,
    TeamProvider,
    MiddlewareProvider,
    ModeConfigProvider,
    RLProvider,
    EnrichmentProvider,
    ServiceProvider,
)

__all__ = [
    "VerticalBase",
    "ToolProvider",
    "ToolSelectionStrategy",
    "SafetyProvider",
    "SafetyExtension",
    "PromptProvider",
    "PromptContributor",
    "WorkflowProvider",
    "HandlerProvider",
    "TeamProvider",
    "MiddlewareProvider",
    "ModeConfigProvider",
    "RLProvider",
    "EnrichmentProvider",
    "ServiceProvider",
]
