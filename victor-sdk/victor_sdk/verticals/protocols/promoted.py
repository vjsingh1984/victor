"""Promoted protocol definitions from victor.core.verticals.protocols.

These protocols were originally defined in victor.core.verticals.protocols.*
and are promoted here so external verticals can import from the SDK without
depending on the victor runtime package.

All protocols in this module are pure Protocol/ABC definitions with ZERO
runtime dependencies on the victor package.

Usage (external verticals):
    from victor_sdk.verticals.protocols import (
        MiddlewareProtocol,
        SafetyExtensionProtocol,
        PromptContributorProtocol,
        ModeConfigProviderProtocol,
    )
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable

from victor_sdk.verticals.protocols.promoted_types import (
    ModeConfig,
    MiddlewarePriority,
    MiddlewareResult,
    SafetyPatternData,
    StageValidationResult,
    TaskTypeHintData,
    ToolSelectionContext,
    ToolSelectionResult,
    ValidationError,
)


# =============================================================================
# Tool Selection Strategy Protocol
# =============================================================================


@runtime_checkable
class ToolSelectionStrategyProtocol(Protocol):
    """Protocol for vertical-specific tool selection strategies.

    Enables verticals to customize tool selection based on domain knowledge.
    """

    def select_tools(
        self,
        context: ToolSelectionContext,
    ) -> ToolSelectionResult:
        """Select tools based on vertical-specific strategy."""
        ...

    def get_task_tool_mapping(self) -> Dict[str, List[str]]:
        """Get mapping of task types to priority tools."""
        ...

    def get_priority(self) -> int:
        """Get priority for this strategy. Lower values processed first."""
        return 50


@runtime_checkable
class VerticalToolSelectionProviderProtocol(Protocol):
    """Protocol for verticals providing tool selection strategies."""

    @classmethod
    def get_tool_selection_strategy(cls) -> Optional[ToolSelectionStrategyProtocol]:
        """Get the tool selection strategy for this vertical."""
        ...


@runtime_checkable
class TieredToolConfigProviderProtocol(Protocol):
    """Protocol for verticals providing tiered tool configuration."""

    @classmethod
    def get_tiered_tool_config(cls) -> Optional[Any]:
        """Get the tiered tool configuration for this vertical."""
        ...

    @classmethod
    def get_tool_tier_name(cls) -> str:
        """Get the tier name for registry lookup."""
        ...


@runtime_checkable
class VerticalTieredToolProviderProtocol(Protocol):
    """Protocol for verticals providing tiered tool management."""

    @classmethod
    def get_tiered_tool_config(cls) -> Optional[Any]:
        """Get the tiered tool configuration for this vertical."""
        ...


# =============================================================================
# Safety Extension Protocol
# =============================================================================


@runtime_checkable
class SafetyExtensionProtocol(Protocol):
    r"""Protocol for vertical-specific safety patterns.

    Extends the framework's core safety checker with domain-specific
    dangerous operation patterns.
    """

    @abstractmethod
    def get_bash_patterns(self) -> List[SafetyPatternData]:
        """Get bash command patterns for this vertical."""
        ...

    def get_file_patterns(self) -> List[SafetyPatternData]:
        """Get file operation patterns for this vertical."""
        return []

    def get_tool_restrictions(self) -> Dict[str, List[str]]:
        """Get tool-specific argument restrictions."""
        return {}

    def get_category(self) -> str:
        """Get the category name for these patterns."""
        return "custom"


# =============================================================================
# Team Provider Protocols
# =============================================================================


@runtime_checkable
class TeamSpecProviderProtocol(Protocol):
    """Protocol for providing team specifications."""

    @abstractmethod
    def get_team_specs(self) -> Dict[str, Any]:
        """Get team specifications for this vertical."""
        ...

    def get_default_team(self) -> Optional[str]:
        """Get the default team name."""
        return None


@runtime_checkable
class VerticalTeamProviderProtocol(Protocol):
    """Protocol for verticals providing team specifications."""

    @classmethod
    def get_team_spec_provider(cls) -> Optional[TeamSpecProviderProtocol]:
        """Get the team specification provider for this vertical."""
        ...


# =============================================================================
# Middleware Protocol
# =============================================================================


@runtime_checkable
class MiddlewareProtocol(Protocol):
    """Protocol for tool execution middleware.

    Middleware can intercept and modify tool calls before and after execution.
    """

    @abstractmethod
    async def before_tool_call(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> MiddlewareResult:
        """Called before a tool is executed."""
        ...

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """Called after a tool is executed."""
        return None

    def get_priority(self) -> MiddlewarePriority:
        """Get the priority of this middleware."""
        return MiddlewarePriority.NORMAL

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get tools this middleware applies to. None means all tools."""
        return None


# =============================================================================
# Prompt Contributor Protocol
# =============================================================================


@runtime_checkable
class PromptContributorProtocol(Protocol):
    """Protocol for contributing to system prompts."""

    @abstractmethod
    def get_task_type_hints(self) -> Dict[str, TaskTypeHintData]:
        """Get task-type-specific prompt hints."""
        ...

    def get_system_prompt_section(self) -> str:
        """Get a section to append to the system prompt."""
        return ""

    def get_grounding_rules(self) -> str:
        """Get vertical-specific grounding rules."""
        return ""

    def get_priority(self) -> int:
        """Get priority for prompt section ordering. Lower values appear first."""
        return 50


# =============================================================================
# Mode Config Provider Protocol
# =============================================================================


@runtime_checkable
class ModeConfigProviderProtocol(Protocol):
    """Protocol for providing mode configurations."""

    @abstractmethod
    def get_mode_configs(self) -> Dict[str, ModeConfig]:
        """Get mode configurations for this vertical."""
        ...

    def get_default_mode(self) -> str:
        """Get the default mode name."""
        return "default"

    def get_default_tool_budget(self) -> int:
        """Get default tool budget when no mode is specified."""
        return 10


# =============================================================================
# Workflow Provider Protocols
# =============================================================================


@runtime_checkable
class WorkflowProviderProtocol(Protocol):
    """Protocol for providing vertical-specific workflows."""

    @abstractmethod
    def get_workflows(self) -> Dict[str, Any]:
        """Get workflow definitions for this vertical."""
        ...

    def get_auto_workflows(self) -> List[Tuple[str, str]]:
        """Get automatically triggered workflows."""
        return []


@runtime_checkable
class VerticalWorkflowProviderProtocol(Protocol):
    """Protocol for verticals providing workflow definitions."""

    @classmethod
    def get_workflow_provider(cls) -> Optional[WorkflowProviderProtocol]:
        """Get the workflow provider for this vertical."""
        ...


# =============================================================================
# Service Provider Protocol
# =============================================================================


@runtime_checkable
class ServiceProviderProtocol(Protocol):
    """Protocol for registering vertical-specific services with DI container."""

    @abstractmethod
    def register_services(
        self,
        container: Any,  # ServiceContainer
        settings: Any,  # Settings
    ) -> None:
        """Register vertical-specific services."""
        ...

    def get_required_services(self) -> List[type]:
        """Get list of required service types."""
        return []

    def get_optional_services(self) -> List[type]:
        """Get list of optional service types."""
        return []


# =============================================================================
# RL Provider Protocols
# =============================================================================


@runtime_checkable
class RLConfigProviderProtocol(Protocol):
    """Protocol for providing RL configuration."""

    @abstractmethod
    def get_rl_config(self) -> Dict[str, Any]:
        """Get RL configuration for this vertical."""
        ...

    def get_rl_hooks(self) -> Optional[Any]:
        """Get RL hooks for outcome recording."""
        return None


@runtime_checkable
class VerticalRLProviderProtocol(Protocol):
    """Protocol for verticals providing RL configuration."""

    @classmethod
    def get_rl_config_provider(cls) -> Optional[RLConfigProviderProtocol]:
        """Get the RL configuration provider for this vertical."""
        ...

    @classmethod
    def get_rl_hooks(cls) -> Optional[Any]:
        """Get RL hooks for outcome recording."""
        ...


# =============================================================================
# Enrichment Protocols
# =============================================================================


@runtime_checkable
class EnrichmentStrategyProtocol(Protocol):
    """Protocol for vertical-specific prompt enrichment strategies."""

    async def get_enrichments(
        self,
        prompt: str,
        context: Any,
    ) -> List[Any]:
        """Get enrichments for a prompt."""
        ...

    def get_priority(self) -> int:
        """Get priority for this strategy. Lower values processed first."""
        ...

    def get_token_allocation(self) -> float:
        """Get fraction of token budget this strategy can use."""
        ...


@runtime_checkable
class VerticalEnrichmentProviderProtocol(Protocol):
    """Protocol for verticals providing enrichment strategies."""

    @classmethod
    def get_enrichment_strategy(cls) -> Optional[EnrichmentStrategyProtocol]:
        """Get the enrichment strategy for this vertical."""
        ...


# =============================================================================
# Capability Provider Protocols
# =============================================================================


@runtime_checkable
class CapabilityProviderProtocol(Protocol):
    """Protocol for capability configuration providers."""

    def get_capabilities(self) -> Dict[str, Any]:
        """Get capability definitions for this vertical."""
        ...


@runtime_checkable
class ChainProviderProtocol(Protocol):
    """Protocol for chain configuration providers."""

    def get_chains(self) -> Dict[str, Any]:
        """Get chain definitions for this vertical."""
        ...


@runtime_checkable
class PersonaProviderProtocol(Protocol):
    """Protocol for persona configuration providers."""

    def get_personas(self) -> Dict[str, Any]:
        """Get persona definitions for this vertical."""
        ...


@runtime_checkable
class VerticalPersonaProviderProtocol(Protocol):
    """Enhanced protocol for vertical-specific persona providers."""

    @classmethod
    def get_persona_specs(cls) -> Dict[str, Any]:
        """Get typed PersonaSpec objects for this vertical."""
        ...

    @classmethod
    def get_default_persona(cls) -> Optional[str]:
        """Get the default persona name for this vertical."""
        ...

    @classmethod
    def get_persona_tags(cls) -> List[str]:
        """Get tags to apply to all personas from this vertical."""
        ...


# =============================================================================
# Stage Contract Protocol and Utilities
# =============================================================================


@runtime_checkable
class StageContract(Protocol):
    """Protocol defining the contract for stage definitions."""

    REQUIRED_STAGES: frozenset[str] = frozenset({"INITIAL", "COMPLETION"})
    TERMINAL_STAGES: frozenset[str] = frozenset({"COMPLETION"})
    RESERVED_PREFIXES: frozenset[str] = frozenset({"_", "SYSTEM", "INTERNAL"})


class StageValidator:
    """Validator for stage contract compliance."""

    def __init__(
        self,
        strict_mode: bool = False,
        allow_custom_stages: bool = True,
    ):
        self._strict_mode = strict_mode
        self._allow_custom_stages = allow_custom_stages

    def validate(
        self, stages: Dict[str, Any], stage_name: str = "default"
    ) -> StageValidationResult:
        """Validate stage definitions against contract."""
        result = StageValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            details={"stage_count": len(stages), "stage_name": stage_name},
        )
        self._validate_required_stages(stages, result)
        self._validate_stage_names(stages, result)
        for sname, stage_def in stages.items():
            self._validate_stage_attributes(sname, stage_def, result)
        self._validate_transitions(stages, result)
        if self._strict_mode and result.warnings:
            for warning in result.warnings:
                result.add_error(ValidationError.INVALID_TRANSITION, f"[STRICT] {warning}")
            result.warnings.clear()
        return result

    def _validate_required_stages(
        self, stages: Dict[str, Any], result: StageValidationResult
    ) -> None:
        missing = StageContract.REQUIRED_STAGES - set(stages.keys())
        if missing:
            result.add_error(
                ValidationError.MISSING_REQUIRED_STAGE,
                f"Missing required stages: {sorted(missing)}",
            )

    def _validate_stage_names(
        self, stages: Dict[str, Any], result: StageValidationResult
    ) -> None:
        for sname in stages.keys():
            if any(sname.startswith(prefix) for prefix in StageContract.RESERVED_PREFIXES):
                result.add_error(
                    ValidationError.INVALID_STAGE_NAME,
                    f"Stage '{sname}' uses reserved prefix",
                )
            if not sname.isupper():
                result.add_warning(
                    f"Stage '{sname}' should use UPPERCASE naming convention"
                )

    def _validate_stage_attributes(
        self, sname: str, stage_def: Any, result: StageValidationResult
    ) -> None:
        if not isinstance(stage_def, dict):
            result.add_error(
                ValidationError.INVALID_DESCRIPTION,
                f"Stage '{sname}' must be a dictionary",
            )
            return
        if "name" not in stage_def:
            result.add_error(
                ValidationError.INVALID_DESCRIPTION,
                f"Stage '{sname}' missing 'name' attribute",
            )
        if "description" not in stage_def:
            result.add_warning(f"Stage '{sname}' missing 'description' attribute")
        elif (
            not isinstance(stage_def.get("description"), str)
            or not stage_def.get("description").strip()
        ):
            result.add_error(
                ValidationError.INVALID_DESCRIPTION,
                f"Stage '{sname}' has invalid 'description' (must be non-empty string)",
            )
        keywords = stage_def.get("keywords", [])
        if not isinstance(keywords, list):
            result.add_error(
                ValidationError.INVALID_KEYWORDS,
                f"Stage '{sname}' 'keywords' must be a list",
            )
        else:
            for kw in keywords:
                if not isinstance(kw, str):
                    result.add_warning(
                        f"Stage '{sname}' has non-string keyword: {type(kw).__name__}"
                    )
        next_stages = stage_def.get("next_stages")
        if next_stages is None:
            result.add_warning(f"Stage '{sname}' missing 'next_stages' attribute")
        elif not isinstance(next_stages, (set, list)):
            result.add_error(
                ValidationError.MISSING_NEXT_STAGES,
                f"Stage '{sname}' 'next_stages' must be a set or list",
            )
        if sname in StageContract.TERMINAL_STAGES and next_stages:
            result.add_error(
                ValidationError.INVALID_TRANSITION,
                f"Terminal stage '{sname}' should not have next_stages",
            )

    def _validate_transitions(
        self, stages: Dict[str, Any], result: StageValidationResult
    ) -> None:
        graph: Dict[str, Set[str]] = {}
        for sname, stage_def in stages.items():
            next_stages = stage_def.get("next_stages", set())
            if isinstance(next_stages, list):
                next_stages = set(next_stages)
            graph[sname] = set(next_stages) if next_stages else set()
        self._check_circular_transitions(graph, result)
        for from_stage, to_stages in graph.items():
            for to_stage in to_stages:
                if to_stage not in stages:
                    result.add_error(
                        ValidationError.INVALID_TRANSITION,
                        f"Invalid transition from '{from_stage}' to non-existent stage '{to_stage}'",
                    )
        for sname in stages:
            if sname not in StageContract.TERMINAL_STAGES:
                if not graph.get(sname):
                    result.add_warning(
                        f"Non-terminal stage '{sname}' has no next_stages defined"
                    )

    def _check_circular_transitions(
        self, graph: Dict[str, Set[str]], result: StageValidationResult
    ) -> None:
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    result.add_error(
                        ValidationError.CIRCULAR_TRANSITION,
                        f"Circular transition detected: {' -> '.join(cycle)}",
                    )
            path.pop()
            rec_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node, [])


class StageContractMixin:
    """Mixin for classes that need stage contract validation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._stage_validator = StageValidator()

    def validate_stages(
        self, stages: Dict[str, Any], stage_name: str = "default"
    ) -> StageValidationResult:
        """Validate stage definitions."""
        return self._stage_validator.validate(stages, stage_name)

    def ensure_stage_contract(
        self, stages: Dict[str, Any], stage_name: str = "default"
    ) -> None:
        """Ensure stage definitions satisfy contract, raise if not."""
        result = self.validate_stages(stages, stage_name)
        if not result.is_valid:
            error_msgs = [msg for _, msg in result.errors]
            raise ValueError(
                f"Stage contract validation failed for '{stage_name}':\n"
                + "\n".join(f"  - {msg}" for msg in error_msgs)
            )


def validate_stage_contract(
    stages: Dict[str, Any], stage_name: str = "default", strict: bool = False
) -> StageValidationResult:
    """Convenience function to validate stage contract."""
    validator = StageValidator(strict_mode=strict)
    return validator.validate(stages, stage_name)


# =============================================================================
# ISP-Compliant Vertical Provider Protocols
# (promoted from victor.core.verticals.protocols.providers)
# =============================================================================

# These are already defined in the SDK's individual protocol files
# (middleware.py, safety.py, etc.) as MiddlewareProvider, SafetyProvider, etc.
# The ISP provider protocols from victor.core.verticals.protocols.providers
# match those existing SDK definitions, so we don't re-define them here.
# They are exported from the SDK's protocols/__init__.py.


__all__ = [
    # Tool Selection
    "ToolSelectionContext",
    "ToolSelectionResult",
    "ToolSelectionStrategyProtocol",
    "VerticalToolSelectionProviderProtocol",
    "TieredToolConfigProviderProtocol",
    "VerticalTieredToolProviderProtocol",
    # Safety
    "SafetyExtensionProtocol",
    "SafetyPatternData",
    # Team
    "TeamSpecProviderProtocol",
    "VerticalTeamProviderProtocol",
    # Middleware
    "MiddlewareProtocol",
    "MiddlewarePriority",
    "MiddlewareResult",
    # Prompt
    "PromptContributorProtocol",
    "TaskTypeHintData",
    # Mode
    "ModeConfig",
    "ModeConfigProviderProtocol",
    # Workflow
    "WorkflowProviderProtocol",
    "VerticalWorkflowProviderProtocol",
    # Service
    "ServiceProviderProtocol",
    # RL
    "RLConfigProviderProtocol",
    "VerticalRLProviderProtocol",
    # Enrichment
    "EnrichmentStrategyProtocol",
    "VerticalEnrichmentProviderProtocol",
    # Capability
    "CapabilityProviderProtocol",
    "ChainProviderProtocol",
    "PersonaProviderProtocol",
    "VerticalPersonaProviderProtocol",
    # Stage Contract
    "StageContract",
    "StageValidator",
    "StageValidationResult",
    "ValidationError",
    "validate_stage_contract",
    "StageContractMixin",
]
