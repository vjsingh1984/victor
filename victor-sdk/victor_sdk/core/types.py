"""Core type definitions for Victor SDK.

These types define the data structures used across vertical configurations
and framework interactions without depending on any runtime implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TypeAlias, Union

from victor_sdk.core.exceptions import VerticalConfigurationError


class Tier(str, Enum):
    """Vertical capability tier for progressive enhancement."""

    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"


CURRENT_DEFINITION_VERSION = "1.0"
MINIMUM_SUPPORTED_DEFINITION_VERSION = "1.0"


def _parse_definition_version(version: str) -> tuple[int, int]:
    """Parse a definition version string into `(major, minor)` integers."""

    if not isinstance(version, str):
        raise VerticalConfigurationError(
            "Definition version must be a string.",
            details={"definition_version": version},
        )

    parts = version.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise VerticalConfigurationError(
            "Definition version must use '<major>.<minor>' numeric format.",
            details={"definition_version": version},
        )

    return int(parts[0]), int(parts[1])


def is_supported_definition_version(version: str) -> bool:
    """Return whether a definition version is supported by this SDK."""

    try:
        validate_definition_version(version)
    except VerticalConfigurationError:
        return False
    return True


def validate_definition_version(version: str) -> None:
    """Validate compatibility for a vertical definition schema version."""

    parsed_version = _parse_definition_version(version)
    minimum_version = _parse_definition_version(MINIMUM_SUPPORTED_DEFINITION_VERSION)
    current_version = _parse_definition_version(CURRENT_DEFINITION_VERSION)

    if parsed_version < minimum_version:
        raise VerticalConfigurationError(
            "Definition version is below the minimum supported schema version.",
            details={
                "definition_version": version,
                "minimum_supported": MINIMUM_SUPPORTED_DEFINITION_VERSION,
            },
        )

    if parsed_version > current_version:
        raise VerticalConfigurationError(
            "Definition version is newer than this SDK supports.",
            details={
                "definition_version": version,
                "current_supported": CURRENT_DEFINITION_VERSION,
            },
        )


@dataclass(frozen=True)
class StageDefinition:
    """Definition of a workflow stage with tool configuration.

    Attributes:
        name: Stage identifier (e.g., "planning", "execution")
        description: Human-readable stage description
        required_tools: Tools that MUST be available in this stage
        optional_tools: Tools that MAY be used if available
        allow_custom_tools: Whether user can add custom tools in this stage
        keywords: Optional routing keywords used by runtime stage detection
        next_stages: Optional valid stage transitions for runtime workflows
        min_confidence: Optional minimum confidence for runtime stage entry
    """

    name: str
    description: str = ""
    required_tools: List[str] = field(default_factory=list)
    optional_tools: List[str] = field(default_factory=list)
    allow_custom_tools: bool = True
    keywords: List[str] = field(default_factory=list)
    next_stages: Set[str] = field(default_factory=set)
    min_confidence: float = 0.5
    tools: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Normalize constructor inputs while preserving SDK immutability.

        ``tools`` is accepted as a compatibility alias for legacy runtime code.
        When explicit required/optional tool lists are absent, the legacy set is
        promoted into ``optional_tools`` to preserve prior semantics.
        """
        normalized_required = list(self.required_tools)
        normalized_optional = list(self.optional_tools)
        normalized_keywords = list(self.keywords)
        normalized_next_stages = set(self.next_stages)
        normalized_tools = set(self.tools)

        if normalized_tools and not normalized_required and not normalized_optional:
            normalized_optional = sorted(normalized_tools)

        normalized_tools.update(normalized_required)
        normalized_tools.update(normalized_optional)

        object.__setattr__(self, "required_tools", normalized_required)
        object.__setattr__(self, "optional_tools", normalized_optional)
        object.__setattr__(self, "keywords", normalized_keywords)
        object.__setattr__(self, "next_stages", normalized_next_stages)
        object.__setattr__(self, "tools", normalized_tools)

    def get_effective_tools(self, available_tools: List[str]) -> List[str]:
        """Get effective tool list for this stage.

        Args:
            available_tools: Tools currently available in the system

        Returns:
            List of tool names that should be used in this stage
        """
        effective = set(self.required_tools)

        # Add optional tools that are available
        effective.update(tool for tool in self.optional_tools if tool in available_tools)

        return sorted(effective)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of the stage definition."""

        return {
            "name": self.name,
            "description": self.description,
            "tools": sorted(self.tools),
            "required_tools": self.required_tools.copy(),
            "optional_tools": self.optional_tools.copy(),
            "allow_custom_tools": self.allow_custom_tools,
            "keywords": self.keywords.copy(),
            "next_stages": sorted(self.next_stages),
            "min_confidence": self.min_confidence,
        }


StageDefinitionLike: TypeAlias = Union[Dict[str, Any], StageDefinition]


def normalize_stage_definition(
    stage_name: str,
    stage: StageDefinitionLike,
) -> StageDefinition:
    """Normalize a stage declaration into a StageDefinition object."""

    if isinstance(stage, StageDefinition):
        return stage
    if isinstance(stage, dict):
        legacy_tools = list(stage.get("tools", []))
        return StageDefinition(
            name=stage.get("name", stage_name),
            description=stage.get("description", ""),
            required_tools=list(stage.get("required_tools", [])),
            optional_tools=list(stage.get("optional_tools", legacy_tools)),
            allow_custom_tools=bool(stage.get("allow_custom_tools", True)),
            keywords=list(stage.get("keywords", [])),
            next_stages=set(stage.get("next_stages", [])),
            min_confidence=float(stage.get("min_confidence", 0.5)),
        )
    if hasattr(stage, "name") and hasattr(stage, "description"):
        # Accept richer runtime stage objects during the migration to the
        # SDK definition contract without importing runtime-only types here.
        return StageDefinition(
            name=stage.name,
            description=stage.description,
            required_tools=list(getattr(stage, "required_tools", [])),
            optional_tools=sorted(
                getattr(
                    stage,
                    "optional_tools",
                    getattr(stage, "tools", []),
                )
            ),
            allow_custom_tools=bool(getattr(stage, "allow_custom_tools", True)),
            keywords=list(getattr(stage, "keywords", [])),
            next_stages=set(getattr(stage, "next_stages", [])),
            min_confidence=float(getattr(stage, "min_confidence", 0.5)),
        )
    raise TypeError(
        "Stage definitions must be dicts or StageDefinition objects, " f"got {type(stage)!r}"
    )


def normalize_stage_definitions(
    stages: Dict[str, StageDefinitionLike],
) -> Dict[str, StageDefinition]:
    """Normalize stage declarations to StageDefinition objects."""

    return {
        stage_name: normalize_stage_definition(stage_name, stage)
        for stage_name, stage in stages.items()
    }


@dataclass(frozen=True)
class TieredToolConfig:
    """Tool configuration with tier-based progressive enhancement.

    Attributes:
        basic_tools: Tools available at BASIC tier (minimal functionality)
        standard_tools: Additional tools at STANDARD tier
        advanced_tools: Additional tools at ADVANCED tier (full functionality)
        mandatory: Runtime-compatible always-on tool set
        vertical_core: Runtime-compatible vertical core tool set
        semantic_pool: Runtime-compatible semantic candidate tool set
        stage_tools: Optional runtime stage-specific tool mapping
        readonly_only_for_analysis: Whether analysis flows should hide write tools
    """

    basic_tools: List[str] = field(default_factory=list)
    standard_tools: List[str] = field(default_factory=list)
    advanced_tools: List[str] = field(default_factory=list)
    mandatory: Set[str] = field(default_factory=set)
    vertical_core: Set[str] = field(default_factory=set)
    semantic_pool: Set[str] = field(default_factory=set)
    stage_tools: Dict[str, Set[str]] = field(default_factory=dict)
    readonly_only_for_analysis: bool = True

    def __post_init__(self) -> None:
        """Normalize compatibility aliases between SDK and runtime tier shapes."""

        normalized_mandatory = set(self.mandatory) or set(self.basic_tools)
        normalized_vertical_core = set(self.vertical_core) or set(self.standard_tools)
        normalized_semantic_pool = set(self.semantic_pool) or set(self.advanced_tools)
        normalized_stage_tools = {
            stage_name: set(tool_names) for stage_name, tool_names in self.stage_tools.items()
        }

        object.__setattr__(self, "mandatory", normalized_mandatory)
        object.__setattr__(self, "vertical_core", normalized_vertical_core)
        object.__setattr__(self, "semantic_pool", normalized_semantic_pool)
        object.__setattr__(self, "stage_tools", normalized_stage_tools)

        if not self.basic_tools:
            object.__setattr__(self, "basic_tools", sorted(normalized_mandatory))
        if not self.standard_tools:
            object.__setattr__(self, "standard_tools", sorted(normalized_vertical_core))
        if not self.advanced_tools:
            object.__setattr__(self, "advanced_tools", sorted(normalized_semantic_pool))

    def get_base_tools(self) -> Set[str]:
        """Return tools that should always be available."""

        return set(self.mandatory) | set(self.vertical_core)

    def get_tools_for_stage(self, stage: str) -> Set[str]:
        """Return runtime-compatible base plus stage-specific tools."""

        return self.get_base_tools() | set(self.stage_tools.get(stage, set()))

    def get_effective_semantic_pool(self) -> Set[str]:
        """Return the runtime-compatible semantic candidate pool."""

        return set(self.semantic_pool)

    def get_tools_for_tier(self, tier: Union[Tier, str]) -> List[str]:
        """Get tools available at a specific tier.

        Args:
            tier: The capability tier

        Returns:
            List of tool names available at that tier
        """
        tier = Tier(tier) if isinstance(tier, str) else tier
        tools = []

        if tier == Tier.BASIC:
            tools = self.basic_tools.copy()
        elif tier == Tier.STANDARD:
            tools = self.basic_tools + self.standard_tools
        elif tier == Tier.ADVANCED:
            tools = self.basic_tools + self.standard_tools + self.advanced_tools

        return tools

    def get_max_tier_for_tools(self, available_tools: List[str]) -> Tier:
        """Determine max tier based on available tools.

        Args:
            available_tools: Tools that are currently available

        Returns:
            The highest tier that can be supported
        """
        available_set = set(available_tools)

        if available_set.issuperset(
            set(self.basic_tools + self.standard_tools + self.advanced_tools)
        ):
            return Tier.ADVANCED
        elif available_set.issuperset(set(self.basic_tools + self.standard_tools)):
            return Tier.STANDARD
        else:
            return Tier.BASIC


@dataclass(frozen=True)
class ToolSet:
    """A set of tools with metadata.

    Attributes:
        names: List of tool names
        description: Human-readable description of this tool set
        tier: Capability tier for this tool set
    """

    names: List[str]
    description: str = ""
    tier: Tier = Tier.STANDARD

    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is in this set."""
        return tool_name in self.names

    def __len__(self) -> int:
        """Return number of tools in this set."""
        return len(self.names)

    def __iter__(self):
        """Iterate over tool names."""
        return iter(self.names)


@dataclass(frozen=True)
class ToolRequirement:
    """Serializable requirement for a tool used by a vertical.

    Attributes:
        tool_name: Canonical tool identifier from the SDK tool registry
        required: Whether the tool is mandatory for the vertical definition
        purpose: Human-readable reason this tool is included
        metadata: Additional serializable requirement metadata
    """

    tool_name: str
    required: bool = True
    purpose: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_legacy_string(self) -> str:
        """Return the legacy string form used by older integrations."""

        return self.tool_name

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of the requirement."""

        payload: Dict[str, Any] = {
            "tool_name": self.tool_name,
            "required": self.required,
        }
        if self.purpose:
            payload["purpose"] = self.purpose
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


ToolRequirementLike: TypeAlias = Union[str, Dict[str, Any], ToolRequirement]


def normalize_tool_requirement(requirement: ToolRequirementLike) -> ToolRequirement:
    """Convert a legacy string or typed requirement into a requirement object."""

    if isinstance(requirement, ToolRequirement):
        return requirement
    if isinstance(requirement, str):
        return ToolRequirement(tool_name=requirement)
    if isinstance(requirement, dict):
        return ToolRequirement(
            tool_name=requirement.get("tool_name", ""),
            required=bool(requirement.get("required", True)),
            purpose=requirement.get("purpose", ""),
            metadata=dict(requirement.get("metadata", {})),
        )
    raise TypeError(
        "Tool requirements must be strings, dicts, or ToolRequirement objects, "
        f"got {type(requirement)!r}"
    )


def normalize_tool_requirements(
    requirements: List[ToolRequirementLike],
) -> List[ToolRequirement]:
    """Normalize a list of tool requirements to typed objects."""

    return [normalize_tool_requirement(requirement) for requirement in requirements]


@dataclass(frozen=True)
class PromptTemplateDefinition:
    """Serializable prompt template for a task type."""

    task_type: str
    template: str
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of the prompt template."""

        payload: Dict[str, Any] = {
            "task_type": self.task_type,
            "template": self.template,
        }
        if self.description:
            payload["description"] = self.description
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


PromptTemplateLike: TypeAlias = Union[str, Dict[str, Any], PromptTemplateDefinition]


def normalize_prompt_template(
    task_type: str,
    template: PromptTemplateLike,
) -> PromptTemplateDefinition:
    """Normalize a prompt template declaration."""

    if isinstance(template, PromptTemplateDefinition):
        return template
    if isinstance(template, str):
        return PromptTemplateDefinition(task_type=task_type, template=template)
    if isinstance(template, dict):
        return PromptTemplateDefinition(
            task_type=template.get("task_type", task_type),
            template=template.get("template", ""),
            description=template.get("description", ""),
            metadata=dict(template.get("metadata", {})),
        )
    raise TypeError(
        "Prompt templates must be strings, dicts, or PromptTemplateDefinition objects, "
        f"got {type(template)!r}"
    )


def normalize_prompt_templates(
    templates: Union[Dict[str, PromptTemplateLike], List[PromptTemplateDefinition]],
) -> List[PromptTemplateDefinition]:
    """Normalize prompt template declarations to serializable dataclasses."""

    if isinstance(templates, list):
        return [
            (
                template
                if isinstance(template, PromptTemplateDefinition)
                else normalize_prompt_template("", template)
            )
            for template in templates
        ]
    return [
        normalize_prompt_template(task_type, template) for task_type, template in templates.items()
    ]


@dataclass(frozen=True)
class TaskTypeHintDefinition:
    """Serializable task-type hint for prompt and planning guidance."""

    task_type: str
    hint: str
    tool_budget: int = 10
    priority_tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of the task hint."""

        payload: Dict[str, Any] = {
            "task_type": self.task_type,
            "hint": self.hint,
            "tool_budget": self.tool_budget,
            "priority_tools": self.priority_tools.copy(),
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


TaskTypeHintLike: TypeAlias = Union[str, Dict[str, Any], TaskTypeHintDefinition]


def normalize_task_type_hint(
    task_type: str,
    hint: TaskTypeHintLike,
) -> TaskTypeHintDefinition:
    """Normalize a task-type hint declaration."""

    if isinstance(hint, TaskTypeHintDefinition):
        return hint
    if isinstance(hint, str):
        return TaskTypeHintDefinition(task_type=task_type, hint=hint)
    if isinstance(hint, dict):
        metadata = {
            key: value
            for key, value in hint.items()
            if key not in {"task_type", "hint", "tool_budget", "priority_tools"}
        }
        return TaskTypeHintDefinition(
            task_type=hint.get("task_type", task_type),
            hint=hint.get("hint", ""),
            tool_budget=int(hint.get("tool_budget", 10)),
            priority_tools=list(hint.get("priority_tools", [])),
            metadata=metadata,
        )
    raise TypeError(
        "Task type hints must be strings, dicts, or TaskTypeHintDefinition objects, "
        f"got {type(hint)!r}"
    )


def normalize_task_type_hints(
    hints: Union[Dict[str, TaskTypeHintLike], List[TaskTypeHintDefinition]],
) -> List[TaskTypeHintDefinition]:
    """Normalize task-type hint declarations to serializable dataclasses."""

    if isinstance(hints, list):
        return [
            (
                hint
                if isinstance(hint, TaskTypeHintDefinition)
                else normalize_task_type_hint("", hint)
            )
            for hint in hints
        ]
    return [normalize_task_type_hint(task_type, hint) for task_type, hint in hints.items()]


@dataclass(frozen=True)
class PromptMetadata:
    """Serializable prompt metadata for a vertical definition."""

    templates: List[PromptTemplateDefinition] = field(default_factory=list)
    task_type_hints: List[TaskTypeHintDefinition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of the prompt metadata."""

        return {
            "templates": [template.to_dict() for template in self.templates],
            "task_type_hints": [hint.to_dict() for hint in self.task_type_hints],
            "metadata": dict(self.metadata),
        }


def normalize_prompt_metadata(
    metadata: Union[Dict[str, Any], PromptMetadata],
) -> PromptMetadata:
    """Normalize prompt metadata payloads to PromptMetadata."""

    if isinstance(metadata, PromptMetadata):
        return metadata
    if isinstance(metadata, dict):
        return PromptMetadata(
            templates=normalize_prompt_templates(metadata.get("templates", {})),
            task_type_hints=normalize_task_type_hints(metadata.get("task_type_hints", {})),
            metadata=dict(metadata.get("metadata", {})),
        )
    raise TypeError(
        "Prompt metadata must be a dict or PromptMetadata object, " f"got {type(metadata)!r}"
    )


@dataclass(frozen=True)
class WorkflowMetadata:
    """Serializable workflow metadata for a vertical definition."""

    initial_stage: Optional[str] = None
    workflow_spec: Dict[str, Any] = field(default_factory=dict)
    provider_hints: Dict[str, Any] = field(default_factory=dict)
    evaluation_criteria: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of the workflow metadata."""

        payload: Dict[str, Any] = {
            "workflow_spec": dict(self.workflow_spec),
            "provider_hints": dict(self.provider_hints),
            "evaluation_criteria": self.evaluation_criteria.copy(),
            "metadata": dict(self.metadata),
        }
        if self.initial_stage is not None:
            payload["initial_stage"] = self.initial_stage
        return payload


def normalize_workflow_metadata(
    metadata: Union[Dict[str, Any], WorkflowMetadata],
) -> WorkflowMetadata:
    """Normalize workflow metadata payloads to WorkflowMetadata."""

    if isinstance(metadata, WorkflowMetadata):
        return metadata
    if isinstance(metadata, dict):
        return WorkflowMetadata(
            initial_stage=metadata.get("initial_stage"),
            workflow_spec=dict(metadata.get("workflow_spec", {})),
            provider_hints=dict(metadata.get("provider_hints", {})),
            evaluation_criteria=list(metadata.get("evaluation_criteria", [])),
            metadata=dict(metadata.get("metadata", {})),
        )
    raise TypeError(
        "Workflow metadata must be a dict or WorkflowMetadata object, " f"got {type(metadata)!r}"
    )


@dataclass(frozen=True)
class TeamMemberDefinition:
    """Serializable declaration for a team member in a definition-layer team."""

    role: str
    goal: str
    name: Optional[str] = None
    tool_budget: Optional[int] = None
    allowed_tools: List[str] = field(default_factory=list)
    is_manager: bool = False
    priority: int = 0
    backstory: str = ""
    expertise: List[str] = field(default_factory=list)
    personality: str = ""
    max_delegation_depth: int = 0
    memory: bool = False
    memory_config: Dict[str, Any] = field(default_factory=dict)
    cache: bool = True
    verbose: bool = False
    max_iterations: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize mutable fields for direct dataclass construction."""

        object.__setattr__(self, "allowed_tools", list(self.allowed_tools))
        object.__setattr__(self, "expertise", list(self.expertise))
        object.__setattr__(self, "memory_config", dict(self.memory_config))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of the team member definition."""

        payload: Dict[str, Any] = {
            "role": self.role,
            "goal": self.goal,
            "is_manager": self.is_manager,
            "priority": self.priority,
            "backstory": self.backstory,
            "expertise": self.expertise.copy(),
            "personality": self.personality,
            "max_delegation_depth": self.max_delegation_depth,
            "memory": self.memory,
            "cache": self.cache,
            "verbose": self.verbose,
            "metadata": dict(self.metadata),
        }
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_budget is not None:
            payload["tool_budget"] = self.tool_budget
        if self.allowed_tools:
            payload["allowed_tools"] = self.allowed_tools.copy()
        if self.memory_config:
            payload["memory_config"] = dict(self.memory_config)
        if self.max_iterations is not None:
            payload["max_iterations"] = self.max_iterations
        return payload


TeamMemberDefinitionLike: TypeAlias = Union[Dict[str, Any], TeamMemberDefinition]


def normalize_team_member_definition(
    member: TeamMemberDefinitionLike,
) -> TeamMemberDefinition:
    """Normalize a team-member declaration to a TeamMemberDefinition."""

    if isinstance(member, TeamMemberDefinition):
        return member
    if isinstance(member, dict):
        return TeamMemberDefinition(
            role=member.get("role", ""),
            goal=member.get("goal", ""),
            name=member.get("name"),
            tool_budget=member.get("tool_budget"),
            allowed_tools=list(member.get("allowed_tools", [])),
            is_manager=bool(member.get("is_manager", False)),
            priority=int(member.get("priority", 0)),
            backstory=member.get("backstory", ""),
            expertise=list(member.get("expertise", [])),
            personality=member.get("personality", ""),
            max_delegation_depth=int(member.get("max_delegation_depth", 0)),
            memory=bool(member.get("memory", False)),
            memory_config=dict(member.get("memory_config", {})),
            cache=bool(member.get("cache", True)),
            verbose=bool(member.get("verbose", False)),
            max_iterations=member.get("max_iterations"),
            metadata=dict(member.get("metadata", {})),
        )
    raise TypeError(
        "Team members must be dicts or TeamMemberDefinition objects, " f"got {type(member)!r}"
    )


def normalize_team_member_definitions(
    members: List[TeamMemberDefinitionLike],
) -> List[TeamMemberDefinition]:
    """Normalize team-member declarations to TeamMemberDefinition objects."""

    return [normalize_team_member_definition(member) for member in members]


@dataclass(frozen=True)
class TeamDefinition:
    """Serializable declaration for a runtime-owned team configuration."""

    team_id: str
    name: str
    description: str = ""
    formation: str = "sequential"
    members: List[TeamMemberDefinition] = field(default_factory=list)
    total_tool_budget: int = 100
    max_iterations: int = 50
    tags: List[str] = field(default_factory=list)
    task_types: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize nested member declarations and formation strings."""

        object.__setattr__(self, "formation", _normalize_team_formation(self.formation))
        object.__setattr__(self, "members", normalize_team_member_definitions(list(self.members)))
        object.__setattr__(self, "tags", list(self.tags))
        object.__setattr__(self, "task_types", list(self.task_types))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of the team declaration."""

        return {
            "team_id": self.team_id,
            "name": self.name,
            "description": self.description,
            "formation": self.formation,
            "members": [member.to_dict() for member in self.members],
            "total_tool_budget": self.total_tool_budget,
            "max_iterations": self.max_iterations,
            "tags": self.tags.copy(),
            "task_types": self.task_types.copy(),
            "metadata": dict(self.metadata),
        }


TeamDefinitionLike: TypeAlias = Union[Dict[str, Any], TeamDefinition]


def _normalize_team_formation(formation: Any) -> str:
    """Normalize a team formation enum/string into the SDK string form."""

    if isinstance(formation, str):
        return formation.lower()
    if hasattr(formation, "value"):
        return str(formation.value).lower()
    if hasattr(formation, "name"):
        return str(formation.name).lower()
    raise TypeError(f"Unsupported team formation type: {type(formation)!r}")


def normalize_team_definition(
    team_id: str,
    team: TeamDefinitionLike,
) -> TeamDefinition:
    """Normalize a team declaration into a TeamDefinition object."""

    if isinstance(team, TeamDefinition):
        return team
    if isinstance(team, dict):
        resolved_team_id = team.get("team_id", team_id)
        return TeamDefinition(
            team_id=resolved_team_id,
            name=team.get("name", resolved_team_id),
            description=team.get("description", ""),
            formation=_normalize_team_formation(team.get("formation", "sequential")),
            members=normalize_team_member_definitions(list(team.get("members", []))),
            total_tool_budget=int(team.get("total_tool_budget", 100)),
            max_iterations=int(team.get("max_iterations", 50)),
            tags=list(team.get("tags", [])),
            task_types=list(team.get("task_types", [])),
            metadata=dict(team.get("metadata", {})),
        )
    raise TypeError(
        "Team definitions must be dicts or TeamDefinition objects, " f"got {type(team)!r}"
    )


def normalize_team_definitions(
    teams: Union[Dict[str, TeamDefinitionLike], List[TeamDefinitionLike]],
) -> List[TeamDefinition]:
    """Normalize team declarations into TeamDefinition objects."""

    if isinstance(teams, list):
        normalized_teams: List[TeamDefinition] = []
        for item in teams:
            if isinstance(item, TeamDefinition):
                normalized_teams.append(item)
                continue
            if not isinstance(item, dict):
                raise TypeError(
                    "Team definitions must be dicts or TeamDefinition objects, "
                    f"got {type(item)!r}"
                )
            normalized_teams.append(normalize_team_definition(str(item.get("team_id", "")), item))
        return normalized_teams

    return [normalize_team_definition(team_id, team) for team_id, team in teams.items()]


@dataclass(frozen=True)
class TeamMetadata:
    """Serializable team metadata for a vertical definition."""

    teams: List[TeamDefinition] = field(default_factory=list)
    default_team: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize nested team declarations for direct dataclass construction."""

        object.__setattr__(self, "teams", normalize_team_definitions(list(self.teams)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of the team metadata."""

        payload: Dict[str, Any] = {
            "teams": [team.to_dict() for team in self.teams],
            "metadata": dict(self.metadata),
        }
        if self.default_team is not None:
            payload["default_team"] = self.default_team
        return payload


def normalize_team_metadata(
    metadata: Union[Dict[str, Any], TeamMetadata],
) -> TeamMetadata:
    """Normalize team metadata payloads to TeamMetadata."""

    if isinstance(metadata, TeamMetadata):
        return metadata
    if isinstance(metadata, dict):
        return TeamMetadata(
            teams=normalize_team_definitions(metadata.get("teams", {})),
            default_team=metadata.get("default_team"),
            metadata=dict(metadata.get("metadata", {})),
        )
    raise TypeError(
        "Team metadata must be a dict or TeamMetadata object, " f"got {type(metadata)!r}"
    )


@dataclass(frozen=True)
class CapabilityRequirement:
    """Serializable requirement for a host/runtime capability.

    Attributes:
        capability_id: Stable capability identifier from `victor_sdk.constants`
        min_version: Optional minimum runtime capability version
        optional: Whether the capability is a preference instead of a hard requirement
        purpose: Human-readable reason this capability is needed
        metadata: Additional serializable requirement metadata
    """

    capability_id: str
    min_version: Optional[str] = None
    optional: bool = False
    purpose: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_legacy_string(self) -> str:
        """Return the legacy string form used by older integrations."""

        return self.capability_id

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of the requirement."""

        payload: Dict[str, Any] = {
            "capability_id": self.capability_id,
            "optional": self.optional,
        }
        if self.min_version is not None:
            payload["min_version"] = self.min_version
        if self.purpose:
            payload["purpose"] = self.purpose
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


CapabilityRequirementLike: TypeAlias = Union[str, Dict[str, Any], CapabilityRequirement]


def normalize_capability_requirement(
    requirement: CapabilityRequirementLike,
) -> CapabilityRequirement:
    """Convert a legacy string or typed requirement into a requirement object."""

    if isinstance(requirement, CapabilityRequirement):
        return requirement
    if isinstance(requirement, str):
        return CapabilityRequirement(capability_id=requirement)
    if isinstance(requirement, dict):
        return CapabilityRequirement(
            capability_id=requirement.get("capability_id", ""),
            min_version=requirement.get("min_version"),
            optional=bool(requirement.get("optional", False)),
            purpose=requirement.get("purpose", ""),
            metadata=dict(requirement.get("metadata", {})),
        )
    raise TypeError(
        "Capability requirements must be strings, dicts, or CapabilityRequirement objects, "
        f"got {type(requirement)!r}"
    )


def normalize_capability_requirements(
    requirements: List[CapabilityRequirementLike],
) -> List[CapabilityRequirement]:
    """Normalize a list of capability requirements to typed objects."""

    return [normalize_capability_requirement(requirement) for requirement in requirements]


@dataclass
class VerticalConfig:
    """Configuration for a vertical.

    This is the main configuration object that verticals provide to the framework.
    It contains all necessary information for the framework to create and configure
    an agent for this vertical.

    Attributes:
        name: Vertical identifier
        description: Human-readable description
        tools: Tool names or ToolSet for this vertical
        system_prompt: System prompt for the agent
        stages: Stage definitions for multi-stage workflows
        tier: Capability tier for this vertical
        metadata: Additional metadata as key-value pairs
        extensions: Extension configurations (protocols, capabilities, etc.)
    """

    name: str
    description: str
    tools: Union[List[str], ToolSet]
    system_prompt: str
    stages: Dict[str, StageDefinition] = field(default_factory=dict)
    tier: Tier = Tier.STANDARD
    metadata: Dict[str, Any] = field(default_factory=dict)
    extensions: Dict[str, Any] = field(default_factory=dict)

    def get_tool_names(self) -> List[str]:
        """Get list of tool names from config."""
        if isinstance(self.tools, ToolSet):
            return self.tools.names
        return self.tools

    def get_stage_names(self) -> List[str]:
        """Get list of stage names."""
        return list(self.stages.keys())

    def with_metadata(self, **kwargs) -> VerticalConfig:
        """Return a new config with additional metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return VerticalConfig(
            name=self.name,
            description=self.description,
            tools=self.tools,
            system_prompt=self.system_prompt,
            stages=self.stages,
            tier=self.tier,
            metadata=new_metadata,
            extensions=self.extensions,
        )

    def with_extension(self, key: str, value: Any) -> VerticalConfig:
        """Return a new config with an extension added."""
        new_extensions = {**self.extensions, key: value}
        return VerticalConfig(
            name=self.name,
            description=self.description,
            tools=self.tools,
            system_prompt=self.system_prompt,
            stages=self.stages,
            tier=self.tier,
            metadata=self.metadata,
            extensions=new_extensions,
        )


@dataclass(frozen=True)
class VerticalDefinition:
    """Serializable definition-layer contract for a vertical.

    This object is intentionally runtime-agnostic. It captures the declarative
    information a host runtime needs in order to construct a vertical-specific
    agent configuration.

    Attributes:
        name: Vertical identifier
        description: Human-readable description
        version: Version of the vertical definition/package
        definition_version: Schema version for this definition payload
        tools: Canonical tool identifiers required by the vertical
        capability_requirements: Host/runtime capabilities required by the vertical
        system_prompt: System prompt for the vertical
        stages: Declarative workflow stage definitions
        tier: Capability tier for the vertical
        metadata: Additional serializable metadata
        extensions: Additional serializable extension metadata
    """

    name: str
    description: str
    tools: List[str]
    system_prompt: str
    version: str = "1.0.0"
    definition_version: str = CURRENT_DEFINITION_VERSION
    framework_version_requirement: str = ">=1.0.0"
    tool_requirements: List[ToolRequirement] = field(default_factory=list)
    capability_requirements: List[CapabilityRequirement] = field(default_factory=list)
    prompt_metadata: PromptMetadata = field(default_factory=PromptMetadata)
    stages: Dict[str, StageDefinition] = field(default_factory=dict)
    team_metadata: TeamMetadata = field(default_factory=TeamMetadata)
    workflow_metadata: WorkflowMetadata = field(default_factory=WorkflowMetadata)
    tier: Tier = Tier.STANDARD
    metadata: Dict[str, Any] = field(default_factory=dict)
    extensions: Dict[str, Any] = field(default_factory=dict)
    skills: List[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize and validate constructor payloads."""

        try:
            normalized_tier = Tier(self.tier) if isinstance(self.tier, str) else self.tier
            normalized_tool_requirements = normalize_tool_requirements(list(self.tool_requirements))
            normalized_capability_requirements = normalize_capability_requirements(
                list(self.capability_requirements)
            )
            normalized_tools = list(self.tools)
            if not normalized_tools and normalized_tool_requirements:
                normalized_tools = [
                    requirement.tool_name for requirement in normalized_tool_requirements
                ]
            normalized_prompt_metadata = normalize_prompt_metadata(self.prompt_metadata)
            normalized_stages = normalize_stage_definitions(dict(self.stages))
            normalized_team_metadata = normalize_team_metadata(self.team_metadata)
            normalized_workflow_metadata = normalize_workflow_metadata(self.workflow_metadata)
            normalized_metadata = dict(self.metadata)
            normalized_extensions = dict(self.extensions)
            normalized_skills = list(self.skills)
        except VerticalConfigurationError:
            raise
        except Exception as exc:
            raise VerticalConfigurationError(
                "Vertical definition payload could not be normalized.",
                vertical_name=(self.name if isinstance(self.name, str) and self.name else None),
                details={"error": str(exc)},
            ) from exc

        object.__setattr__(self, "tier", normalized_tier)
        object.__setattr__(self, "tool_requirements", normalized_tool_requirements)
        object.__setattr__(self, "capability_requirements", normalized_capability_requirements)
        object.__setattr__(self, "tools", normalized_tools)
        object.__setattr__(self, "prompt_metadata", normalized_prompt_metadata)
        object.__setattr__(self, "stages", normalized_stages)
        object.__setattr__(self, "team_metadata", normalized_team_metadata)
        object.__setattr__(self, "workflow_metadata", normalized_workflow_metadata)
        object.__setattr__(self, "metadata", normalized_metadata)
        object.__setattr__(self, "extensions", normalized_extensions)
        object.__setattr__(self, "skills", normalized_skills)

        self.validate()

    def get_tool_names(self) -> List[str]:
        """Return the canonical tool identifiers for this definition."""

        if self.tools:
            return self.tools.copy()
        if self.tool_requirements:
            return [requirement.tool_name for requirement in self.tool_requirements]
        return []

    def validate(self) -> None:
        """Validate the definition contract and schema compatibility."""

        validate_definition_version(self.definition_version)

        if not isinstance(self.name, str) or not self.name.strip():
            raise VerticalConfigurationError("Vertical definition name must be a non-empty string.")

        if not isinstance(self.description, str):
            raise VerticalConfigurationError(
                "Vertical definition description must be a string.",
                vertical_name=self.name,
            )

        if not isinstance(self.system_prompt, str):
            raise VerticalConfigurationError(
                "Vertical definition system_prompt must be a string.",
                vertical_name=self.name,
            )

        invalid_tools = [
            tool_name
            for tool_name in self.tools
            if not isinstance(tool_name, str) or not tool_name.strip()
        ]
        if invalid_tools:
            raise VerticalConfigurationError(
                "Vertical definition tools must be non-empty strings.",
                vertical_name=self.name,
                details={"tools": invalid_tools},
            )

        if self.tool_requirements:
            requirement_tools = [requirement.tool_name for requirement in self.tool_requirements]
            if self.tools != requirement_tools:
                raise VerticalConfigurationError(
                    "Vertical definition tools must match tool requirement order.",
                    vertical_name=self.name,
                    details={
                        "tools": self.tools,
                        "tool_requirements": requirement_tools,
                    },
                )

        for stage_name, stage_definition in self.stages.items():
            if stage_definition.name != stage_name:
                raise VerticalConfigurationError(
                    "Stage definition name must match its mapping key.",
                    vertical_name=self.name,
                    details={
                        "stage_key": stage_name,
                        "stage_name": stage_definition.name,
                    },
                )

        initial_stage = self.workflow_metadata.initial_stage
        if initial_stage is not None and initial_stage not in self.stages:
            raise VerticalConfigurationError(
                "Workflow initial_stage must exist in the stage definitions.",
                vertical_name=self.name,
                details={"initial_stage": initial_stage},
            )

        stage_order = self.workflow_metadata.workflow_spec.get("stage_order")
        if stage_order is not None:
            if not isinstance(stage_order, list) or not all(
                isinstance(stage_name, str) for stage_name in stage_order
            ):
                raise VerticalConfigurationError(
                    "Workflow stage_order must be a list of stage-name strings.",
                    vertical_name=self.name,
                )

            missing_stages = [
                stage_name for stage_name in stage_order if stage_name not in self.stages
            ]
            if missing_stages:
                raise VerticalConfigurationError(
                    "Workflow stage_order references undefined stages.",
                    vertical_name=self.name,
                    details={"missing_stages": missing_stages},
                )

        template_task_types = [template.task_type for template in self.prompt_metadata.templates]
        if len(template_task_types) != len(set(template_task_types)):
            raise VerticalConfigurationError(
                "Prompt metadata template task types must be unique.",
                vertical_name=self.name,
            )

        hint_task_types = [hint.task_type for hint in self.prompt_metadata.task_type_hints]
        if len(hint_task_types) != len(set(hint_task_types)):
            raise VerticalConfigurationError(
                "Prompt metadata task-type hints must be unique.",
                vertical_name=self.name,
            )

        team_ids = [team.team_id for team in self.team_metadata.teams]
        if len(team_ids) != len(set(team_ids)):
            raise VerticalConfigurationError(
                "Team metadata team identifiers must be unique.",
                vertical_name=self.name,
            )

        for team in self.team_metadata.teams:
            if not isinstance(team.team_id, str) or not team.team_id.strip():
                raise VerticalConfigurationError(
                    "Team metadata team_id values must be non-empty strings.",
                    vertical_name=self.name,
                )
            if not isinstance(team.name, str) or not team.name.strip():
                raise VerticalConfigurationError(
                    "Team metadata team names must be non-empty strings.",
                    vertical_name=self.name,
                    details={"team_id": team.team_id},
                )
            if not team.members:
                raise VerticalConfigurationError(
                    "Team metadata declarations must include at least one member.",
                    vertical_name=self.name,
                    details={"team_id": team.team_id},
                )
            for member in team.members:
                if not isinstance(member.role, str) or not member.role.strip():
                    raise VerticalConfigurationError(
                        "Team metadata member roles must be non-empty strings.",
                        vertical_name=self.name,
                        details={"team_id": team.team_id},
                    )
                if not isinstance(member.goal, str) or not member.goal.strip():
                    raise VerticalConfigurationError(
                        "Team metadata member goals must be non-empty strings.",
                        vertical_name=self.name,
                        details={"team_id": team.team_id, "role": member.role},
                    )

        if (
            self.team_metadata.default_team is not None
            and self.team_metadata.default_team not in team_ids
        ):
            raise VerticalConfigurationError(
                "Team metadata default_team must reference a declared team.",
                vertical_name=self.name,
                details={"default_team": self.team_metadata.default_team},
            )

    def get_stage_names(self) -> List[str]:
        """Return the stage names present in this definition."""

        return list(self.stages.keys())

    def to_config(self) -> VerticalConfig:
        """Convert the definition into the legacy SDK config shape."""

        config_extensions = dict(self.extensions)
        if self.tool_requirements:
            config_extensions["tool_requirements"] = self.tool_requirements.copy()
        if self.capability_requirements:
            config_extensions["capability_requirements"] = self.capability_requirements.copy()
        if (
            self.prompt_metadata.templates
            or self.prompt_metadata.task_type_hints
            or self.prompt_metadata.metadata
        ):
            config_extensions["prompt_metadata"] = self.prompt_metadata.to_dict()
        if (
            self.workflow_metadata.initial_stage is not None
            or self.workflow_metadata.workflow_spec
            or self.workflow_metadata.provider_hints
            or self.workflow_metadata.evaluation_criteria
            or self.workflow_metadata.metadata
        ):
            config_extensions["workflow_metadata"] = self.workflow_metadata.to_dict()
        if (
            self.team_metadata.teams
            or self.team_metadata.default_team is not None
            or self.team_metadata.metadata
        ):
            config_extensions["team_metadata"] = self.team_metadata.to_dict()

        return VerticalConfig(
            name=self.name,
            description=self.description,
            tools=self.get_tool_names(),
            system_prompt=self.system_prompt,
            stages=self.stages.copy(),
            tier=self.tier,
            metadata={
                **self.metadata,
                "vertical_version": self.version,
                "definition_version": self.definition_version,
            },
            extensions=config_extensions,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of the definition."""

        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "definition_version": self.definition_version,
            "framework_version_requirement": self.framework_version_requirement,
            "tools": self.get_tool_names(),
            "tool_requirements": [requirement.to_dict() for requirement in self.tool_requirements],
            "capability_requirements": [
                requirement.to_dict() for requirement in self.capability_requirements
            ],
            "system_prompt": self.system_prompt,
            "prompt_metadata": self.prompt_metadata.to_dict(),
            "stages": {
                stage_name: stage_definition.to_dict()
                for stage_name, stage_definition in self.stages.items()
            },
            "team_metadata": self.team_metadata.to_dict(),
            "workflow_metadata": self.workflow_metadata.to_dict(),
            "tier": self.tier.value,
            "metadata": dict(self.metadata),
            "extensions": dict(self.extensions),
        }

    @classmethod
    def from_config(
        cls,
        config: VerticalConfig,
        *,
        version: str = "1.0.0",
        definition_version: str = CURRENT_DEFINITION_VERSION,
    ) -> "VerticalDefinition":
        """Create a definition from an existing SDK config object."""

        config_metadata = dict(config.metadata)
        config_version = config_metadata.pop("vertical_version", version)
        config_definition_version = config_metadata.pop("definition_version", definition_version)
        config_extensions = dict(config.extensions)
        tool_requirements = normalize_tool_requirements(
            config_extensions.pop("tool_requirements", config.get_tool_names())
        )
        capability_requirements = normalize_capability_requirements(
            config_extensions.pop("capability_requirements", [])
        )
        prompt_metadata = normalize_prompt_metadata(config_extensions.pop("prompt_metadata", {}))
        team_metadata = normalize_team_metadata(config_extensions.pop("team_metadata", {}))
        workflow_metadata = normalize_workflow_metadata(
            config_extensions.pop("workflow_metadata", {})
        )

        return cls(
            name=config.name,
            description=config.description,
            version=config_version,
            definition_version=config_definition_version,
            tools=config.get_tool_names(),
            tool_requirements=tool_requirements,
            capability_requirements=capability_requirements,
            system_prompt=config.system_prompt,
            prompt_metadata=prompt_metadata,
            stages=config.stages.copy(),
            team_metadata=team_metadata,
            workflow_metadata=workflow_metadata,
            tier=config.tier,
            metadata=config_metadata,
            extensions=config_extensions,
        )

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VerticalDefinition":
        """Create a definition from a serialized dictionary payload."""

        return cls(
            name=payload.get("name", ""),
            description=payload.get("description", ""),
            version=payload.get("version", "1.0.0"),
            definition_version=payload.get("definition_version", CURRENT_DEFINITION_VERSION),
            tools=list(payload.get("tools", [])),
            tool_requirements=payload.get("tool_requirements", []),
            capability_requirements=payload.get("capability_requirements", []),
            system_prompt=payload.get("system_prompt", ""),
            prompt_metadata=payload.get("prompt_metadata", {}),
            stages=payload.get("stages", {}),
            team_metadata=payload.get("team_metadata", {}),
            workflow_metadata=payload.get("workflow_metadata", {}),
            tier=payload.get("tier", Tier.STANDARD),
            metadata=dict(payload.get("metadata", {})),
            extensions=dict(payload.get("extensions", {})),
        )
