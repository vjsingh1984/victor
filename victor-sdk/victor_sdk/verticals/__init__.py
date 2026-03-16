"""Vertical protocols and base classes.

This module provides protocol definitions and base classes for creating
Victor verticals without runtime dependencies.
"""

from victor_sdk.verticals.protocols.base import VerticalBase
from victor_sdk.verticals.protocols import (
    # SDK-native protocols
    CapabilityProvider,
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

__all__ = [
    "VerticalBase",
    # SDK-native protocols
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
