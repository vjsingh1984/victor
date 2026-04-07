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
    FileOperation,
    FileOperationsCapability,
    FileOperationType,
    PromptContribution,
    PromptContributionCapability,
)
from victor_sdk.core.plugins import PluginContext, VictorPlugin

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
    TaskTypeHint,
    ToolDependency,
    ToolDependencyProviderProtocol,
    register_vertical,
)
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
    "ToolRequirement",
    "ToolRequirementLike",
    "VerticalConfig",
    "VerticalDefinition",
    "is_supported_definition_version",
    "WorkflowMetadata",
    "StageDefinition",
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
    "PromptContribution",
    "PromptContributionCapability",
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
    "ToolDependency",
    "ToolDependencyProviderProtocol",
    "MiddlewarePriority",
    "MiddlewareResult",
    "SafetyPattern",
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
]
