"""
Victor SDK - Protocol definitions for vertical development.

This package provides pure protocol/ABC definitions that external verticals
can depend on without pulling in the entire Victor framework.

Version: Synchronized with victor-ai
"""

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("victor-sdk")
except Exception:
    __version__ = "0.0.0"

# Core types
from victor_sdk.core.types import (
    CapabilityRequirement,
    CapabilityRequirementLike,
    CURRENT_DEFINITION_VERSION,
    MINIMUM_SUPPORTED_DEFINITION_VERSION,
    PromptMetadata,
    PromptTemplateDefinition,
    PromptTemplateLike,
    TeamDefinition,
    TeamDefinitionLike,
    TeamMemberDefinition,
    TeamMemberDefinitionLike,
    TeamMetadata,
    StageDefinitionLike,
    TaskTypeHintDefinition,
    TaskTypeHintLike,
    ToolRequirement,
    ToolRequirementLike,
    VerticalConfig,
    VerticalDefinition,
    is_supported_definition_version,
    WorkflowMetadata,
    StageDefinition,
    TieredToolConfig,
    ToolSet,
    normalize_capability_requirement,
    normalize_capability_requirements,
    normalize_prompt_metadata,
    normalize_prompt_template,
    normalize_prompt_templates,
    normalize_stage_definition,
    normalize_stage_definitions,
    normalize_task_type_hint,
    normalize_task_type_hints,
    normalize_team_definition,
    normalize_team_definitions,
    normalize_team_member_definition,
    normalize_team_member_definitions,
    normalize_team_metadata,
    normalize_tool_requirement,
    normalize_tool_requirements,
    validate_definition_version,
    normalize_workflow_metadata,
)
from victor_sdk.capabilities import (
    BaseCapabilityProvider,
    CapabilityConfigMergePolicy,
    CapabilityConfigScopePortProtocol,
    CapabilityConfigService,
    CapabilityEntry,
    CapabilityLoaderPortProtocol,
    CapabilityMetadata,
    CapabilityType,
    DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY,
    FileOperation,
    FileOperationsCapability,
    FileOperationType,
    OrchestratorCapability,
    PromptContribution,
    PromptContributionCapability,
    build_capability_loader,
    capability,
    load_capability_config,
    register_capability_entries,
    resolve_capability_config_scope_key,
    resolve_capability_config_service,
    store_capability_config,
    update_capability_config_section,
)
from victor_sdk.rl import (
    BaseRLConfig,
    DEFAULT_ACTIVE_LEARNERS,
    DEFAULT_PATIENCE_MAP,
    LearnerType,
)
from victor_sdk.conversation import (
    ConversationContext,
    ConversationCoordinator,
    ConversationStats,
    ConversationTurn,
    TurnType,
)
from victor_sdk.multi_agent import (
    CommunicationStyle,
    ExpertiseLevel,
    PersonaTemplate,
    PersonaTraits,
    TaskAssignmentStrategy,
    TeamMember,
    TeamSpec,
    TeamTemplate,
    TeamTopology,
)
from victor_sdk.safety import (
    SafetyAction,
    SafetyCategory,
    SafetyCheckResult,
    SafetyCoordinator,
    SafetyRule,
    SafetyStats,
)
from victor_sdk.workflows import (
    ComputeHandlerProtocol,
    ComputeHandlerRegistrar,
    ComputeNodeProtocol,
    ExecutorNodeStatus,
    NodeResult,
    WorkflowContextProtocol,
    register_compute_handlers,
)
from victor_sdk.workflow_runtime import (
    BaseYAMLWorkflowProvider,
    WorkflowBuilder,
    WorkflowDefinition,
    workflow,
)
from victor_sdk.core.plugins import PluginContext, VictorPlugin
from victor_sdk.registries import (
    PersonaRegistryProtocol,
    TeamRegistryProtocol,
    get_default_persona_registry,
    get_default_team_registry,
    set_default_persona_registry,
    set_default_team_registry,
)

# Core exceptions
from victor_sdk.core.exceptions import (
    VerticalException,
    VerticalConfigurationError,
    VerticalProtocolError,
)

# Vertical protocols
from victor_sdk.verticals import (
    ExtensionDependency,
    MiddlewarePriority,
    MiddlewareResult,
    SafetyPattern,
    StaticModeConfigProvider,
    TaskTypeHint,
    ToolDependencyConfig,
    ToolDependency,
    ToolDependencyLoadError,
    ToolDependencyLoader,
    ToolDependencyProviderProtocol,
    YAMLToolDependencyProvider,
    VerticalModeConfig,
    create_tool_dependency_provider,
    get_cached_provider,
    invalidate_provider_cache,
    load_tool_dependency_yaml,
    register_vertical,
)
from victor_sdk.verticals.mode_config import ModeDefinition
from victor_sdk.verticals.protocols.base import VerticalBase
from victor_sdk.verticals.protocols.capabilities import (
    CapabilityProvider,
    ChainProvider,
    PersonaProvider,
)

# Extension manifest
from victor_sdk.verticals.manifest import ExtensionManifest, ExtensionType
from victor_sdk.verticals.extensions import VerticalExtensions
from victor_sdk.verticals.protocols.mcp import McpProvider, McpToolProvider
from victor_sdk.verticals.protocols.sandbox import SandboxProvider
from victor_sdk.verticals.protocols.hooks import HookProvider, HookConfigProvider
from victor_sdk.verticals.protocols.permissions import PermissionProvider
from victor_sdk.verticals.protocols.compaction import CompactionProvider
from victor_sdk.verticals.protocols.plugins import ExternalPluginProvider
from victor_sdk.core.api_version import (
    CURRENT_API_VERSION,
    MIN_SUPPORTED_API_VERSION,
    is_compatible,
)

# Discovery and registration
from victor_sdk.discovery import (
    ProtocolRegistry,
    ProtocolMetadata,
    DiscoveryStats,
    collect_verticals_from_candidate,
    get_global_registry,
    reset_global_registry,
    discover_verticals,
    discover_protocols,
    get_discovery_summary,
    reload_discovery,
)
from victor_sdk.skills import SkillDefinition, SkillProvider

# Phase 4 promotions: types heavily used by external verticals
from victor_sdk.multi_agent import TeamFormation, TeamMemberSpec
from victor_sdk.rl import RLOutcome, RLRecommendation
from victor_sdk.safety import SafetyLevel

from victor_sdk.validation import (
    ValidationIssue,
    ValidationReport,
    validate_vertical_package,
)

# Stable SDK constants
from victor_sdk.constants import (
    CapabilityIds,
    CANONICAL_TO_ALIASES,
    TOOL_ALIASES,
    ToolNameEntry,
    ToolNames,
    get_all_capability_ids,
    get_aliases,
    get_all_canonical_names,
    get_canonical_name,
    get_name_mapping,
    is_known_capability_id,
    is_valid_tool_name,
)

__all__ = [
    # Version
    "__version__",
    # Core types
    "CapabilityRequirement",
    "CapabilityRequirementLike",
    "CURRENT_DEFINITION_VERSION",
    "MINIMUM_SUPPORTED_DEFINITION_VERSION",
    "PromptMetadata",
    "PromptTemplateDefinition",
    "PromptTemplateLike",
    "TeamDefinition",
    "TeamDefinitionLike",
    "TeamMemberDefinition",
    "TeamMemberDefinitionLike",
    "TeamMetadata",
    "StageDefinitionLike",
    "TaskTypeHintDefinition",
    "TaskTypeHintLike",
    "DEFAULT_ACTIVE_LEARNERS",
    "DEFAULT_PATIENCE_MAP",
    "ToolRequirement",
    "ToolRequirementLike",
    "VerticalConfig",
    "VerticalDefinition",
    "LearnerType",
    "WorkflowContextProtocol",
    "ConversationContext",
    "ConversationCoordinator",
    "ConversationStats",
    "ConversationTurn",
    "CommunicationStyle",
    "ExpertiseLevel",
    "is_supported_definition_version",
    "PersonaTemplate",
    "PersonaTraits",
    "TaskAssignmentStrategy",
    "TeamMember",
    "TeamSpec",
    "TeamTemplate",
    "TeamTopology",
    "WorkflowMetadata",
    "StageDefinition",
    "TurnType",
    "TieredToolConfig",
    "ToolSet",
    "normalize_capability_requirement",
    "normalize_capability_requirements",
    "normalize_prompt_metadata",
    "normalize_prompt_template",
    "normalize_prompt_templates",
    "normalize_stage_definition",
    "normalize_stage_definitions",
    "normalize_task_type_hint",
    "normalize_task_type_hints",
    "normalize_team_definition",
    "normalize_team_definitions",
    "normalize_team_member_definition",
    "normalize_team_member_definitions",
    "normalize_team_metadata",
    "normalize_tool_requirement",
    "normalize_tool_requirements",
    "validate_definition_version",
    "normalize_workflow_metadata",
    # Plugins
    "FileOperation",
    "FileOperationsCapability",
    "FileOperationType",
    "BaseCapabilityProvider",
    "BaseRLConfig",
    "BaseYAMLWorkflowProvider",
    "ComputeHandlerProtocol",
    "ComputeHandlerRegistrar",
    "ComputeNodeProtocol",
    "CapabilityConfigMergePolicy",
    "CapabilityConfigScopePortProtocol",
    "CapabilityConfigService",
    "CapabilityEntry",
    "CapabilityLoaderPortProtocol",
    "CapabilityMetadata",
    "CapabilityType",
    "DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY",
    "ExecutorNodeStatus",
    "SafetyAction",
    "SafetyCategory",
    "SafetyCheckResult",
    "SafetyCoordinator",
    "SafetyRule",
    "SafetyStats",
    "NodeResult",
    "OrchestratorCapability",
    "PromptContribution",
    "PromptContributionCapability",
    "build_capability_loader",
    "capability",
    "load_capability_config",
    "register_compute_handlers",
    "register_capability_entries",
    "resolve_capability_config_scope_key",
    "resolve_capability_config_service",
    "store_capability_config",
    "update_capability_config_section",
    "PersonaRegistryProtocol",
    "TeamRegistryProtocol",
    "get_default_persona_registry",
    "get_default_team_registry",
    "set_default_persona_registry",
    "set_default_team_registry",
    "WorkflowBuilder",
    "WorkflowDefinition",
    "workflow",
    "PluginContext",
    "VictorPlugin",
    # Exceptions
    "VerticalException",
    "VerticalConfigurationError",
    "VerticalProtocolError",
    # Base class
    "VerticalBase",
    "register_vertical",
    "ExtensionDependency",
    # Capability protocols
    "CapabilityProvider",
    "ChainProvider",
    "PersonaProvider",
    "ToolDependencyConfig",
    "ToolDependency",
    "ToolDependencyLoadError",
    "ToolDependencyLoader",
    "ToolDependencyProviderProtocol",
    "YAMLToolDependencyProvider",
    "create_tool_dependency_provider",
    "get_cached_provider",
    "invalidate_provider_cache",
    "load_tool_dependency_yaml",
    "MiddlewarePriority",
    "MiddlewareResult",
    "SafetyPattern",
    "StaticModeConfigProvider",
    "TaskTypeHint",
    # Extension manifest
    "ExtensionManifest",
    "ExtensionType",
    "CURRENT_API_VERSION",
    "MIN_SUPPORTED_API_VERSION",
    "is_compatible",
    # Discovery (Phase 4)
    "ProtocolRegistry",
    "ProtocolMetadata",
    "DiscoveryStats",
    "collect_verticals_from_candidate",
    "get_global_registry",
    "reset_global_registry",
    "discover_verticals",
    "discover_protocols",
    "get_discovery_summary",
    "reload_discovery",
    # Validation
    "ValidationIssue",
    "ValidationReport",
    "validate_vertical_package",
    # Stable constants
    "CapabilityIds",
    "ToolNames",
    "ToolNameEntry",
    "TOOL_ALIASES",
    "CANONICAL_TO_ALIASES",
    "get_all_capability_ids",
    "get_canonical_name",
    "get_aliases",
    "is_valid_tool_name",
    "get_all_canonical_names",
    "get_name_mapping",
    "is_known_capability_id",
    # Lazy extension container
    "VerticalExtensions",
    # Extended vertical protocols: MCP, sandbox, hooks, permissions, compaction, plugins
    "McpProvider",
    "McpToolProvider",
    "SandboxProvider",
    "HookProvider",
    "HookConfigProvider",
    "PermissionProvider",
    "CompactionProvider",
    "ExternalPluginProvider",
    "VerticalModeConfig",
    "ModeDefinition",
    # Skills
    "SkillDefinition",
    "SkillProvider",
    # Phase 4 promotions: types used by 6+ external verticals
    "TeamFormation",
    "TeamMemberSpec",
    "SafetyLevel",
    "RLOutcome",
    "RLRecommendation",
]
