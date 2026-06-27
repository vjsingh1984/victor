"""Protocol definitions for vertical extensions.

This module provides Protocol definitions for all vertical extension points.
These protocols use @runtime_checkable to support isinstance() checks.

Includes both SDK-native protocols and promoted protocols from
victor.core.verticals.protocols for external vertical compatibility.

Usage (external verticals):
    from victor_contracts.verticals.protocols import (
        MiddlewareProtocol,
        SafetyExtensionProtocol,
        PromptContributorProtocol,
        ModeConfigProviderProtocol,
    )
"""

# SDK-native protocols
from victor_contracts.verticals.protocols.tools import (
    ToolDependency,
    ToolDependencyProviderProtocol,
    ToolProvider,
    ToolSelectionStrategy,
    TieredToolConfigProvider,
)
from victor_contracts.verticals.protocols.safety import (
    SafetyProvider,
    SafetyExtension,
    SafetyPattern,
)
from victor_contracts.verticals.protocols.prompts import (
    PromptProvider,
    PromptContributor,
    TaskTypeHint,
)
from victor_contracts.verticals.protocols.workflows import (
    WorkflowProvider,
    HandlerProvider,
)
from victor_contracts.verticals.protocols.teams import TeamProvider
from victor_contracts.verticals.protocols.middleware import MiddlewareProvider
from victor_contracts.verticals.protocols.modes import ModeConfigProvider
from victor_contracts.verticals.protocols.rl import RLProvider
from victor_contracts.verticals.protocols.enrichment import EnrichmentProvider
from victor_contracts.verticals.protocols.services import ServiceProvider
from victor_contracts.verticals.protocols.handlers import (
    HandlerProvider as InputHandlerProvider,
)
from victor_contracts.verticals.protocols.capabilities import CapabilityProvider

# Promoted protocols from victor.core.verticals.protocols
# These allow external verticals to import from SDK instead of victor.core.*
from victor_contracts.verticals.protocols.promoted import (
    # Tree-sitter analysis
    TreeSitterAnalysisProtocol,
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
from victor_contracts.verticals.protocols.promoted_types import (
    SafetyPatternData,
    MiddlewarePriority,
    MiddlewareResult,
    TaskTypeHintData,
    ModeConfig,
    ToolSelectionContext,
    ToolSelectionResult,
)

# Conversation management protocol
from victor_contracts.verticals.protocols.conversation import (
    ConversationManagerProtocol,
)

# Promoted storage protocols (from victor.storage.*)
from victor_contracts.verticals.protocols.storage import (
    GraphNodeData,
    GraphEdgeData,
    EmbeddingSearchResultData,
    EmbeddingConfigData,
    GraphStoreProtocol,
    VectorStoreProtocol,
    EmbeddingServiceProtocol,
    CCGBuilderProtocol,
)
from victor_contracts.verticals.protocols.memory import MemoryCoordinatorProtocol

# Promoted tool protocols (from victor.tools.* and victor.providers.*)
from victor_contracts.verticals.protocols.tools import (
    ToolResultData,
    MessageData,
    ToolRegistryProtocol,
    ProviderRegistryProtocol,
)

# Promoted config protocols (from victor.config.*)
from victor_contracts.verticals.protocols.config import (
    ProjectPathsData,
    SettingsProviderProtocol,
    ApiKeyProviderProtocol,
)

# Extended vertical protocols: MCP, sandbox, hooks, permissions, compaction, plugins
from victor_contracts.verticals.protocols.mcp import McpProvider, McpToolProvider
from victor_contracts.verticals.protocols.sandbox import SandboxProvider
from victor_contracts.verticals.protocols.hooks import HookProvider, HookConfigProvider
from victor_contracts.verticals.protocols.permissions import PermissionProvider
from victor_contracts.verticals.protocols.compaction import CompactionProvider
from victor_contracts.verticals.protocols.plugins import ExternalPluginProvider
from victor_contracts.verticals.protocols.tool_plugins import (
    ToolFactory,
    ToolFactoryAdapter,
    ToolFactoryPlugin,
    ToolPluginHelper,
)

__all__ = [
    # Tree-sitter analysis (promoted from framework)
    "TreeSitterAnalysisProtocol",
    # SDK-native protocols
    "ToolProvider",
    "ToolSelectionStrategy",
    "TieredToolConfigProvider",
    "ToolDependency",
    "ToolDependencyProviderProtocol",
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
    # Conversation management
    "ConversationManagerProtocol",
    # Extended vertical protocols
    "McpProvider",
    "McpToolProvider",
    "SandboxProvider",
    "HookProvider",
    "HookConfigProvider",
    "PermissionProvider",
    "CompactionProvider",
    "ExternalPluginProvider",
    "ToolFactory",
    "ToolFactoryAdapter",
    "ToolFactoryPlugin",
    "ToolPluginHelper",
    # Promoted storage protocols
    "GraphNodeData",
    "GraphEdgeData",
    "EmbeddingSearchResultData",
    "EmbeddingConfigData",
    "GraphStoreProtocol",
    "VectorStoreProtocol",
    "EmbeddingServiceProtocol",
    "CCGBuilderProtocol",
    "MemoryCoordinatorProtocol",
    # Promoted tool protocols
    "ToolResultData",
    "MessageData",
    "ToolRegistryProtocol",
    "ProviderRegistryProtocol",
    # Promoted config protocols
    "ProjectPathsData",
    "SettingsProviderProtocol",
    "ApiKeyProviderProtocol",
]
