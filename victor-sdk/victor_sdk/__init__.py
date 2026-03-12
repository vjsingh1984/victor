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

# Core exceptions
from victor_sdk.core.exceptions import (
    VerticalException,
    VerticalConfigurationError,
    VerticalProtocolError,
)

# Vertical protocols
from victor_sdk.verticals.protocols.base import VerticalBase

# Discovery and registration
from victor_sdk.discovery import (
    ProtocolRegistry,
    ProtocolMetadata,
    DiscoveryStats,
    get_global_registry,
    reset_global_registry,
    discover_verticals,
    discover_protocols,
    get_discovery_summary,
    reload_discovery,
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
    # Exceptions
    "VerticalException",
    "VerticalConfigurationError",
    "VerticalProtocolError",
    # Base class
    "VerticalBase",
    # Discovery (Phase 4)
    "ProtocolRegistry",
    "ProtocolMetadata",
    "DiscoveryStats",
    "get_global_registry",
    "reset_global_registry",
    "discover_verticals",
    "discover_protocols",
    "get_discovery_summary",
    "reload_discovery",
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
]
