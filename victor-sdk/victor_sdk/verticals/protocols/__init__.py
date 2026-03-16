"""Protocol definitions for vertical extensions.

This module provides Protocol definitions for all vertical extension points.
These protocols use @runtime_checkable to support isinstance() checks.

Includes both SDK-native protocols and promoted protocols from
victor.core.verticals.protocols for external vertical compatibility.

Usage (external verticals):
    from victor_sdk.verticals.protocols import (
        MiddlewareProtocol,
        SafetyExtensionProtocol,
        PromptContributorProtocol,
        ModeConfigProviderProtocol,
    )
"""

# SDK-native protocols
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

# Promoted protocols from victor.core.verticals.protocols
# These allow external verticals to import from SDK instead of victor.core.*
from victor_sdk.verticals.protocols.promoted import (
    # Tool Selection
    ToolSelectionStrategyProtocol,
    VerticalToolSelectionProviderProtocol,
    TieredToolConfigProviderProtocol,
    VerticalTieredToolProviderProtocol,
    # Safety
    SafetyExtensionProtocol,
    # Team
    TeamSpecProviderProtocol,
    VerticalTeamProviderProtocol,
    # Middleware
    MiddlewareProtocol,
    # Prompt
    PromptContributorProtocol,
    # Mode
    ModeConfigProviderProtocol,
    # Workflow
    WorkflowProviderProtocol,
    VerticalWorkflowProviderProtocol,
    # Service
    ServiceProviderProtocol,
    # RL
    RLConfigProviderProtocol,
    VerticalRLProviderProtocol,
    # Enrichment
    EnrichmentStrategyProtocol,
    VerticalEnrichmentProviderProtocol,
    # Capability
    CapabilityProviderProtocol,
    ChainProviderProtocol,
    PersonaProviderProtocol,
    VerticalPersonaProviderProtocol,
    # Stage Contract
    StageContract,
    StageValidator,
    StageValidationResult,
    ValidationError,
    validate_stage_contract,
    StageContractMixin,
)

# Promoted data types
from victor_sdk.verticals.protocols.promoted_types import (
    SafetyPatternData,
    MiddlewarePriority,
    MiddlewareResult,
    TaskTypeHintData,
    ModeConfig,
    ToolSelectionContext,
    ToolSelectionResult,
)

__all__ = [
    # SDK-native protocols
    "ToolProvider",
    "ToolSelectionStrategy",
    "TieredToolConfigProvider",
    "SafetyProvider",
    "SafetyExtension",
    "SafetyPattern",
    "PromptProvider",
    "PromptContributor",
    "TaskTypeHint",
    "WorkflowProvider",
    "HandlerProvider",
    "TeamProvider",
    "MiddlewareProvider",
    "ModeConfigProvider",
    "RLProvider",
    "EnrichmentProvider",
    "ServiceProvider",
    "InputHandlerProvider",
    "CapabilityProvider",
    # Promoted protocols (from victor.core.verticals.protocols)
    "ToolSelectionStrategyProtocol",
    "VerticalToolSelectionProviderProtocol",
    "TieredToolConfigProviderProtocol",
    "VerticalTieredToolProviderProtocol",
    "SafetyExtensionProtocol",
    "TeamSpecProviderProtocol",
    "VerticalTeamProviderProtocol",
    "MiddlewareProtocol",
    "PromptContributorProtocol",
    "ModeConfigProviderProtocol",
    "WorkflowProviderProtocol",
    "VerticalWorkflowProviderProtocol",
    "ServiceProviderProtocol",
    "RLConfigProviderProtocol",
    "VerticalRLProviderProtocol",
    "EnrichmentStrategyProtocol",
    "VerticalEnrichmentProviderProtocol",
    "CapabilityProviderProtocol",
    "ChainProviderProtocol",
    "PersonaProviderProtocol",
    "VerticalPersonaProviderProtocol",
    "StageContract",
    "StageValidator",
    "StageValidationResult",
    "ValidationError",
    "validate_stage_contract",
    "StageContractMixin",
    # Promoted data types
    "SafetyPatternData",
    "MiddlewarePriority",
    "MiddlewareResult",
    "TaskTypeHintData",
    "ModeConfig",
    "ToolSelectionContext",
    "ToolSelectionResult",
]
