# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core protocol interfaces for provider adaptation, grounding, and quality."""

# Provider Adapter
from victor.protocols.provider_adapter import (
    BaseProviderAdapter,
    ContinuationContext,
    IProviderAdapter,
    ProviderCapabilities,
    ToolCallFormat,
    get_provider_adapter,
    register_provider_adapter,
)

# Grounding
from victor.protocols.grounding import (
    AggregatedVerificationResult,
    ClaimVerificationResult,
    CompositeGroundingVerifier,
    ContentMatchStrategy,
    FileExistenceStrategy,
    GroundingClaim,
    GroundingClaimType,
    IGroundingStrategy,
    SymbolReferenceStrategy,
)

# Quality
from victor.protocols.quality import (
    BaseQualityAssessor,
    CompositeQualityAssessor,
    DimensionScore,
    IQualityAssessor,
    ProtocolQualityDimension,
    ProviderAwareQualityAssessor,
    QualityScore,
    SimpleQualityAssessor,
)

# Mode Awareness
from victor.protocols.mode_aware import (
    IModeController,
    ModeAwareMixin,
    ModeInfo,
    create_mode_aware_mixin,
)

# Mode Controller (Phase 1.1)
from victor.protocols.mode_controller import (
    ModeControllerProtocol,
    ExtendedModeControllerProtocol,
)

# Provider Lifecycle (Phase 1.2)
from victor.protocols.provider_lifecycle import ProviderLifecycleProtocol

# Capability Provider (Phase 1.4)
from victor.protocols.capability_provider import FileOperationsCapabilityProtocol

# Path Resolution
from victor.protocols.path_resolver import (
    IPathResolver,
    PathResolution,
    PathResolver,
    create_path_resolver,
)

# LSP Types - Enumerations
from victor.protocols.lsp_types import (
    DiagnosticSeverity,
    CompletionItemKind,
    SymbolKind,
    DiagnosticTag,
    # LSP Types - Position and Range
    Position,
    Range,
    Location,
    LocationLink,
    # LSP Types - Diagnostics
    DiagnosticRelatedInformation,
    Diagnostic,
    # LSP Types - Completions
    CompletionItem,
    # LSP Types - Hover
    Hover,
    # LSP Types - Symbols
    DocumentSymbol,
    SymbolInformation,
    # LSP Types - Text Edits
    TextEdit,
    TextDocumentIdentifier,
    VersionedTextDocumentIdentifier,
    TextDocumentEdit,
)

# Team Coordination
from victor.protocols.team import (
    IAgent,
    ITeamMember,
    ITeamCoordinator,
    IObservableCoordinator,
    IRLCoordinator,
    IMessageBusProvider,
    ISharedMemoryProvider,
    IEnhancedTeamCoordinator,
    ITeamCoordinator,
    ITeamMember,
)

# Search Protocols
from victor.protocols.search import (
    ISemanticSearch,
    IIndexable,
    ISemanticSearchWithIndexing,
)

# Agent Orchestrator Protocol (breaks circular dependencies)
from victor.protocols.agent import (
    IAgentOrchestrator,
    IAgentOrchestratorFactory,
)

# Focused Agent Protocols (ISP-compliant)
from victor.protocols.chat import ChatProtocol
from victor.protocols.config_agent import ConfigProtocol
from victor.protocols.provider import ProviderProtocol
from victor.protocols.state import StateProtocol
from victor.protocols.tools import ToolProtocol

# Tool Selector Protocol (unifies tool selection interfaces)
from victor.protocols.tool_selector import (
    IToolSelector,
    IConfigurableToolSelector,
    IToolSelectorFactory,
    ToolSelectionResult,
    ToolSelectionContext,
    ToolSelectionStrategy,
)

# Configuration Protocol (for ConfigCoordinator)
from victor.protocols.config import (
    IConfigProvider,
)

# Prompt Protocol (for PromptCoordinator)
from victor.protocols.prompt import (
    IPromptContributor,
    PromptContext,
)

# Context Protocol (for ContextCoordinator)
from victor.protocols.context import (
    ICompactionStrategy,
    CompactionResult,
    CompactionContext,
    ContextBudget,
)

# Analytics Protocol (for AnalyticsCoordinator)
from victor.protocols.analytics import (
    IAnalyticsExporter,
    ExportResult,
    AnalyticsEvent,
    AnalyticsQuery,
    AnalyticsResult,
)

# Cache Protocol (for distributed caching)
from victor.protocols.cache import (
    ICacheBackend,
    ICacheInvalidator,
    ICacheDependencyTracker,
    IIdempotentTool,
    IAdvancedCacheBackend,
    ICacheManager,
    CacheNamespace,
    CacheEntryMetadata,
    InvalidationResult,
    CacheStatistics,
    FileChangeType,
    FileChangeEvent,
    IFileWatcher,
    IDependencyExtractor,
)

# Provider Manager Protocol (for provider lifecycle management)
from victor.protocols.provider_manager import (
    IProviderManager,
    SwitchResult,
    HealthStatus,
)

# Search Router Protocol (for search operations)
from victor.protocols.search_router import (
    SearchType,
    SearchContext,
    SearchResult,
    ISearchRouter,
    ISearchBackend,
)

# Lifecycle Manager Protocol (for session lifecycle management)
from victor.protocols.lifecycle import (
    SessionMetadata,
    SessionConfig,
    CleanupResult,
    RecoveryResult,
    ILifecycleManager,
)

# UI Agent Protocol (for UI-orchestrator decoupling)
from victor.protocols.ui_agent import (
    UIAgentProtocol,
)

# Workflow Agent Protocol (for workflow-orchestrator decoupling)
from victor.protocols.workflow_agent import (
    WorkflowAgentProtocol,
)

# Agent Conversation Protocols (from victor.agent.protocols split)
from victor.protocols.agent_conversation import (
    ConversationControllerProtocol,
    ConversationStateMachineProtocol,
    MessageHistoryProtocol,
    StreamingToolChunk,
    StreamingToolAdapterProtocol,
    StreamingControllerProtocol,
    ContextCompactorProtocol,
    ConversationEmbeddingStoreProtocol,
    ReminderManagerProtocol,
)

# Agent Tools Protocols (from victor.agent.protocols split)
from victor.protocols.agent_tools import (
    ToolRegistryProtocol,
    ToolPipelineProtocol,
    ToolExecutorProtocol,
    ToolCacheProtocol,
    ToolOutputFormatterProtocol,
    ResponseSanitizerProtocol,
    ArgumentNormalizerProtocol,
    ProjectContextProtocol,
    ToolDependencyGraphProtocol,
    ToolPluginRegistryProtocol,
)

# Agent Providers Protocols (from victor.agent.protocols split)
from victor.protocols.agent_providers import (
    IProviderHealthMonitor,
    IProviderSwitcher,
    IToolAdapterCoordinator,
    IProviderEventEmitter,
    IProviderClassificationStrategy,
    ProviderRegistryProtocol,
)

# Agent Conversation Refined Protocols (from victor.agent.protocols split)
from victor.protocols.agent_conversation_refined import (
    IMessageStore,
    IContextOverflowHandler,
    ISessionManager,
    IEmbeddingManager,
)

# Session Repository Protocol (for UI-database decoupling)
from victor.protocols.session_repository import (
    SessionRepositoryProtocol,
)

# Agent Budget and Utility Protocols (from victor.agent.protocols split)
from victor.protocols.agent_budget import (
    IBudgetTracker,
    IMultiplierCalculator,
    IModeCompletionChecker,
    IToolCallClassifier,
    DebugLoggerProtocol,
    TaskTypeHinterProtocol,
    SafetyCheckerProtocol,
    AutoCommitterProtocol,
    MCPBridgeProtocol,
    SystemPromptBuilderProtocol,
    ParallelExecutorProtocol,
    ResponseCompleterProtocol,
    StreamingHandlerProtocol,
    UsageLoggerProtocol,
    StreamingMetricsCollectorProtocol,
    IntentClassifierProtocol,
    RLCoordinatorProtocol,
)

# Classification Protocol (breaks circular dependencies)
from victor.protocols.classification import (
    IClassificationResult,
    IKeywordMatch,
)

# ISP Compliance Protocols (Phase 1 SOLID Remediation)
from victor.protocols.capability import (
    CapabilityContainerProtocol,
    get_capability_registry,
)
from victor.protocols.workflow_provider import WorkflowProviderProtocol
from victor.protocols.tiered_config import TieredConfigProviderProtocol
from victor.protocols.extension_provider import ExtensionProviderProtocol

__all__ = [
    # Provider Adapter
    "IProviderAdapter",
    "ProviderCapabilities",
    "ToolCallFormat",
    "BaseProviderAdapter",
    "ContinuationContext",
    "get_provider_adapter",
    "register_provider_adapter",
    # Grounding
    "IGroundingStrategy",
    "GroundingClaimType",
    "GroundingClaim",
    "ClaimVerificationResult",
    "AggregatedVerificationResult",
    "FileExistenceStrategy",
    "SymbolReferenceStrategy",
    "ContentMatchStrategy",
    "CompositeGroundingVerifier",
    # Quality
    "IQualityAssessor",
    "ProtocolQualityDimension",
    "DimensionScore",
    "QualityScore",
    "BaseQualityAssessor",
    "SimpleQualityAssessor",
    "ProviderAwareQualityAssessor",
    "CompositeQualityAssessor",
    # Mode Awareness
    "IModeController",
    "ModeInfo",
    "ModeAwareMixin",
    "create_mode_aware_mixin",
    # Mode Controller (Phase 1.1)
    "ModeControllerProtocol",
    "ExtendedModeControllerProtocol",
    # Provider Lifecycle (Phase 1.2)
    "ProviderLifecycleProtocol",
    # Capability Provider (Phase 1.4)
    "FileOperationsCapabilityProtocol",
    # Path Resolution
    "IPathResolver",
    "PathResolution",
    "PathResolver",
    "create_path_resolver",
    # LSP Types - Enumerations
    "DiagnosticSeverity",
    "CompletionItemKind",
    "SymbolKind",
    "DiagnosticTag",
    # LSP Types - Position and Range
    "Position",
    "Range",
    "Location",
    "LocationLink",
    # LSP Types - Diagnostics
    "DiagnosticRelatedInformation",
    "Diagnostic",
    # LSP Types - Completions
    "CompletionItem",
    # LSP Types - Hover
    "Hover",
    # LSP Types - Symbols
    "DocumentSymbol",
    "SymbolInformation",
    # LSP Types - Text Edits
    "TextEdit",
    "TextDocumentIdentifier",
    "VersionedTextDocumentIdentifier",
    "TextDocumentEdit",
    # Team Coordination
    "IAgent",
    "ITeamMember",
    "ITeamCoordinator",
    "IObservableCoordinator",
    "IRLCoordinator",
    "IMessageBusProvider",
    "ISharedMemoryProvider",
    "IEnhancedTeamCoordinator",
    "ITeamCoordinator",
    "ITeamMember",
    # Search Protocols
    "ISemanticSearch",
    "IIndexable",
    "ISemanticSearchWithIndexing",
    # Agent Orchestrator Protocol
    "IAgentOrchestrator",
    "IAgentOrchestratorFactory",
    # Focused Agent Protocols (ISP-compliant)
    "ChatProtocol",
    "ProviderProtocol",
    "ToolProtocol",
    "StateProtocol",
    "ConfigProtocol",
    # Tool Selector Protocol
    "IToolSelector",
    "IConfigurableToolSelector",
    "IToolSelectorFactory",
    "ToolSelectionResult",
    "ToolSelectionContext",
    "ToolSelectionStrategy",
    # Configuration Protocol
    "IConfigProvider",
    # Prompt Protocol
    "IPromptContributor",
    "PromptContext",
    # Context Protocol
    "ICompactionStrategy",
    "CompactionResult",
    "CompactionContext",
    "ContextBudget",
    # Analytics Protocol
    "IAnalyticsExporter",
    "ExportResult",
    "AnalyticsEvent",
    "AnalyticsQuery",
    "AnalyticsResult",
    # Cache Protocol
    "ICacheBackend",
    "ICacheInvalidator",
    "ICacheDependencyTracker",
    "IIdempotentTool",
    "IAdvancedCacheBackend",
    "ICacheManager",
    "CacheNamespace",
    "CacheEntryMetadata",
    "InvalidationResult",
    "CacheStatistics",
    "FileChangeType",
    "FileChangeEvent",
    "IFileWatcher",
    "IDependencyExtractor",
    # Provider Manager Protocol
    "IProviderManager",
    "SwitchResult",
    "HealthStatus",
    # Search Router Protocol
    "SearchType",
    "SearchContext",
    "SearchResult",
    "ISearchRouter",
    "ISearchBackend",
    # Lifecycle Manager Protocol
    "ILifecycleManager",
    "SessionMetadata",
    "SessionConfig",
    "CleanupResult",
    "RecoveryResult",
    # UI Agent Protocol
    "UIAgentProtocol",
    # Workflow Agent Protocol
    "WorkflowAgentProtocol",
    # Agent Conversation Protocols
    "ConversationControllerProtocol",
    "ConversationStateMachineProtocol",
    "MessageHistoryProtocol",
    "StreamingToolChunk",
    "StreamingToolAdapterProtocol",
    "StreamingControllerProtocol",
    "ContextCompactorProtocol",
    "ConversationEmbeddingStoreProtocol",
    "ReminderManagerProtocol",
    # Agent Tools Protocols
    "ToolRegistryProtocol",
    "ToolPipelineProtocol",
    "ToolExecutorProtocol",
    "ToolCacheProtocol",
    "ToolOutputFormatterProtocol",
    "ResponseSanitizerProtocol",
    "ArgumentNormalizerProtocol",
    "ProjectContextProtocol",
    "ToolDependencyGraphProtocol",
    "ToolPluginRegistryProtocol",
    # Agent Providers Protocols
    "IProviderHealthMonitor",
    "IProviderSwitcher",
    "IToolAdapterCoordinator",
    "IProviderEventEmitter",
    "IProviderClassificationStrategy",
    "ProviderRegistryProtocol",
    # Agent Conversation Refined Protocols
    "IMessageStore",
    "IContextOverflowHandler",
    "ISessionManager",
    "IEmbeddingManager",
    # Session Repository Protocol
    "SessionRepositoryProtocol",
    # Agent Budget and Utility Protocols
    "IBudgetTracker",
    "IMultiplierCalculator",
    "IModeCompletionChecker",
    "IToolCallClassifier",
    "DebugLoggerProtocol",
    "TaskTypeHinterProtocol",
    "SafetyCheckerProtocol",
    "AutoCommitterProtocol",
    "MCPBridgeProtocol",
    "SystemPromptBuilderProtocol",
    "ParallelExecutorProtocol",
    "ResponseCompleterProtocol",
    "StreamingHandlerProtocol",
    "UsageLoggerProtocol",
    "StreamingMetricsCollectorProtocol",
    "IntentClassifierProtocol",
    "RLCoordinatorProtocol",
    # Classification Protocol
    "IClassificationResult",
    "IKeywordMatch",
    # ISP Compliance Protocols (Phase 1 SOLID Remediation)
    "CapabilityContainerProtocol",
    "get_capability_registry",
    "WorkflowProviderProtocol",
    "TieredConfigProviderProtocol",
    "ExtensionProviderProtocol",
]
