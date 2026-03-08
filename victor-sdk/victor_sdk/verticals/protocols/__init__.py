"""Protocol definitions for vertical extensions.

This module provides Protocol definitions for all vertical extension points.
These protocols use @runtime_checkable to support isinstance() checks.
"""

from victor_sdk.verticals.protocols.tools import (
    ToolProvider,
    ToolSelectionStrategy,
    TieredToolConfigProvider,
)
from victor_sdk.verticals.protocols.safety import (
    SafetyProvider,
    SafetyExtension,
    SafetyPattern,
)
from victor_sdk.verticals.protocols.prompts import (
    PromptProvider,
    PromptContributor,
    TaskTypeHint,
)
from victor_sdk.verticals.protocols.workflows import WorkflowProvider, HandlerProvider
from victor_sdk.verticals.protocols.teams import TeamProvider
from victor_sdk.verticals.protocols.middleware import MiddlewareProvider
from victor_sdk.verticals.protocols.modes import ModeConfigProvider
from victor_sdk.verticals.protocols.rl import RLProvider
from victor_sdk.verticals.protocols.enrichment import EnrichmentProvider
from victor_sdk.verticals.protocols.services import ServiceProvider
from victor_sdk.verticals.protocols.handlers import HandlerProvider as InputHandlerProvider
from victor_sdk.verticals.protocols.capabilities import CapabilityProvider

__all__ = [
    # Tools
    "ToolProvider",
    "ToolSelectionStrategy",
    "TieredToolConfigProvider",
    # Safety
    "SafetyProvider",
    "SafetyExtension",
    "SafetyPattern",
    # Prompts
    "PromptProvider",
    "PromptContributor",
    "TaskTypeHint",
    # Workflows
    "WorkflowProvider",
    "HandlerProvider",
    # Teams
    "TeamProvider",
    # Middleware
    "MiddlewareProvider",
    # Modes
    "ModeConfigProvider",
    # RL
    "RLProvider",
    # Enrichment
    "EnrichmentProvider",
    # Services
    "ServiceProvider",
    # Handlers
    "InputHandlerProvider",
    # Capabilities
    "CapabilityProvider",
]
