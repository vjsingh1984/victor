"""Vertical protocols and base classes.

This module provides protocol definitions and base classes for creating
Victor verticals without runtime dependencies.
"""

from victor_sdk.verticals.registration import register_vertical
from victor_sdk.verticals.protocols.base import VerticalBase
from victor_sdk.verticals.extensions import VerticalExtensions
from victor_sdk.verticals.protocols import (
    # SDK-native protocols
    CapabilityProvider,
    ToolDependency,
    ToolDependencyProviderProtocol,
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
    # Promoted protocols (from victor.core.verticals.protocols)
    MiddlewareProtocol,
    SafetyExtensionProtocol,
    PromptContributorProtocol,
    ModeConfigProviderProtocol,
    ToolSelectionStrategyProtocol,
    WorkflowProviderProtocol,
    ServiceProviderProtocol,
    RLConfigProviderProtocol,
    EnrichmentStrategyProtocol,
    CapabilityProviderProtocol,
    TeamSpecProviderProtocol,
    ChainProviderProtocol,
    PersonaProviderProtocol,
)
from victor_sdk.verticals.protocols.promoted_types import (
    MiddlewarePriority,
    MiddlewareResult,
    ModeConfig,
    SafetyPattern,
    TaskTypeHint,
)

__all__ = [
    "register_vertical",
    "VerticalBase",
    "VerticalExtensions",
    # SDK-native protocols
    "ToolDependency",
    "ToolDependencyProviderProtocol",
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
    "CapabilityProvider",
    # Promoted data helpers
    "MiddlewarePriority",
    "MiddlewareResult",
    "ModeConfig",
    "SafetyPattern",
    "TaskTypeHint",
    # Promoted protocols
    "MiddlewareProtocol",
    "SafetyExtensionProtocol",
    "PromptContributorProtocol",
    "ModeConfigProviderProtocol",
    "ToolSelectionStrategyProtocol",
    "WorkflowProviderProtocol",
    "ServiceProviderProtocol",
    "RLConfigProviderProtocol",
    "EnrichmentStrategyProtocol",
    "CapabilityProviderProtocol",
    "TeamSpecProviderProtocol",
    "ChainProviderProtocol",
    "PersonaProviderProtocol",
]
