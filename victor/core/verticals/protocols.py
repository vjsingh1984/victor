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

"""Vertical Extension Protocols.

This module defines protocols for vertical-framework integration, enabling
verticals to extend framework behavior without hardcoding domain-specific
logic in core components.

Design Philosophy:
- Framework is agnostic - no hardcoded coding/research/devops logic
- Verticals inject behavior through protocol implementations
- Each vertical can function independently
- New verticals can be added without modifying framework code

Protocol Categories:
1. Middleware: Pre/post tool execution processing
2. Safety: Vertical-specific danger patterns
3. Prompt: System prompt and task hints
4. Mode: Mode/budget configurations
5. Workflow: Vertical-specific workflows
6. Service: DI service registration

Usage:
    from victor.core.verticals.protocols import (
        MiddlewareProtocol,
        SafetyExtensionProtocol,
        PromptContributorProtocol,
    )

    class CodingMiddleware(MiddlewareProtocol):
        async def before_tool_call(self, tool_name, arguments):
            # Validate code before write
            ...
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.core.container import ServiceContainer
    from victor.config.settings import Settings

# Import SafetyPattern from safety.types for backward compatibility
# (it was originally defined here but moved to break circular imports)
from victor.security.safety.types import SafetyPattern

# Import tool types from core for backward compatibility
# (moved to core to avoid circular imports between core and verticals)
from victor.core.tool_types import ToolDependency, ToolDependencyProviderProtocol

# Import vertical types from core for backward compatibility
# (consolidated in core.vertical_types to break circular imports)
from victor.core.vertical_types import (
    TaskTypeHint,
    MiddlewarePriority,
    MiddlewareResult,
    TieredToolConfig,
    StageDefinition,
)


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class ModeConfig:
    """Configuration for an operational mode.

    Attributes:
        name: Mode name (e.g., "fast", "thorough")
        tool_budget: Tool call budget
        max_iterations: Maximum iterations
        temperature: Temperature setting
        description: Human-readable description
    """

    name: str
    tool_budget: int
    max_iterations: int
    temperature: float = 0.7
    description: str = ""


# NOTE: ToolDependency is now imported from victor.core.tool_types
# NOTE: TieredToolConfig is now imported from victor.core.vertical_types
# NOTE: StageDefinition is now imported from victor.core.vertical_types


# =============================================================================
# Middleware Protocol
# =============================================================================


@runtime_checkable
class MiddlewareProtocol(Protocol):
    """Protocol for tool execution middleware.

    Middleware can intercept and modify tool calls before and after execution.
    Use for validation, transformation, logging, or domain-specific processing.

    Example:
        class CodeValidationMiddleware(MiddlewareProtocol):
            async def before_tool_call(
                self, tool_name: str, arguments: Dict[str, Any]
            ) -> MiddlewareResult:
                if tool_name == "write_file" and "content" in arguments:
                    # Validate code syntax before writing
                    is_valid, error = self._validate_syntax(arguments["content"])
                    if not is_valid:
                        return MiddlewareResult(
                            proceed=False,
                            error_message=f"Syntax error: {error}"
                        )
                return MiddlewareResult()

            def get_priority(self) -> MiddlewarePriority:
                return MiddlewarePriority.HIGH
    """

    @abstractmethod
    async def before_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> MiddlewareResult:
        """Called before a tool is executed.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments passed to the tool

        Returns:
            MiddlewareResult indicating whether to proceed
        """
        ...

    async def after_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        success: bool,
    ) -> Optional[Any]:
        """Called after a tool is executed.

        Args:
            tool_name: Name of the tool that was called
            arguments: Arguments that were passed
            result: Result from the tool execution
            success: Whether the tool execution succeeded

        Returns:
            Modified result (or None to keep original)
        """
        return None

    def get_priority(self) -> MiddlewarePriority:
        """Get the priority of this middleware.

        Returns:
            Priority level for execution ordering
        """
        return MiddlewarePriority.NORMAL

    def get_applicable_tools(self) -> Optional[Set[str]]:
        """Get tools this middleware applies to.

        Returns:
            Set of tool names, or None for all tools
        """
        return None


# =============================================================================
# Safety Extension Protocol
# =============================================================================


@runtime_checkable
class SafetyExtensionProtocol(Protocol):
    r"""Protocol for vertical-specific safety patterns.

    Extends the framework's core safety checker with domain-specific
    dangerous operation patterns.

    Example:
        class GitSafetyExtension(SafetyExtensionProtocol):
            def get_bash_patterns(self) -> List[SafetyPattern]:
                return [
                    SafetyPattern(
                        pattern=r"git\s+reset\s+--hard",
                        description="Discard uncommitted changes",
                        risk_level="HIGH",
                        category="git",
                    ),
                    SafetyPattern(
                        pattern=r"git\s+push\s+.*--force",
                        description="Force push (may lose commits)",
                        risk_level="HIGH",
                        category="git",
                    ),
                ]
    """

    @abstractmethod
    def get_bash_patterns(self) -> List[SafetyPattern]:
        """Get bash command patterns for this vertical.

        Returns:
            List of safety patterns for dangerous bash commands
        """
        ...

    def get_file_patterns(self) -> List[SafetyPattern]:
        """Get file operation patterns for this vertical.

        Returns:
            List of safety patterns for dangerous file operations
        """
        return []

    def get_tool_restrictions(self) -> Dict[str, List[str]]:
        """Get tool-specific argument restrictions.

        Returns:
            Dict mapping tool names to list of restricted argument patterns
        """
        return {}

    def get_category(self) -> str:
        """Get the category name for these patterns.

        Returns:
            Category identifier (e.g., "coding", "devops")
        """
        return "custom"


# =============================================================================
# Prompt Contributor Protocol
# =============================================================================


@runtime_checkable
class PromptContributorProtocol(Protocol):
    """Protocol for contributing to system prompts.

    Verticals can contribute domain-specific task hints and system
    prompt sections without modifying framework code.

    Example:
        class CodingPromptContributor(PromptContributorProtocol):
            def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
                return {
                    "edit": TaskTypeHint(
                        task_type="edit",
                        hint="[EDIT] Read target file first, then modify.",
                        tool_budget=5,
                        priority_tools=["read_file", "edit_files"],
                    ),
                }

            def get_system_prompt_section(self) -> str:
                return "When modifying code, always run tests afterward."
    """

    @abstractmethod
    def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
        """Get task-type-specific prompt hints.

        Returns:
            Dict mapping task types to their hints
        """
        ...

    def get_system_prompt_section(self) -> str:
        """Get a section to append to the system prompt.

        Returns:
            Additional system prompt text (or empty string)
        """
        return ""

    def get_grounding_rules(self) -> str:
        """Get vertical-specific grounding rules.

        Returns:
            Grounding rules text (or empty string for default)
        """
        return ""

    def get_priority(self) -> int:
        """Get priority for prompt section ordering.

        Lower values appear first.

        Returns:
            Priority value (default 50)
        """
        return 50


# =============================================================================
# Mode Config Provider Protocol
# =============================================================================


@runtime_checkable
class ModeConfigProviderProtocol(Protocol):
    """Protocol for providing mode configurations.

    Verticals can define domain-specific operational modes with
    appropriate tool budgets and iteration limits.

    Example:
        class CodingModeProvider(ModeConfigProviderProtocol):
            def get_mode_configs(self) -> Dict[str, ModeConfig]:
                return {
                    "fast": ModeConfig(
                        name="fast",
                        tool_budget=5,
                        max_iterations=10,
                        description="Quick code changes",
                    ),
                    "thorough": ModeConfig(
                        name="thorough",
                        tool_budget=30,
                        max_iterations=60,
                        description="Deep code analysis",
                    ),
                }
    """

    @abstractmethod
    def get_mode_configs(self) -> Dict[str, ModeConfig]:
        """Get mode configurations for this vertical.

        Returns:
            Dict mapping mode names to configurations
        """
        ...

    def get_default_mode(self) -> str:
        """Get the default mode name.

        Returns:
            Name of the default mode
        """
        return "default"

    def get_default_tool_budget(self) -> int:
        """Get default tool budget when no mode is specified.

        Returns:
            Default tool call budget
        """
        return 10


# =============================================================================
# Tool Dependency Provider Protocol
# NOTE: ToolDependencyProviderProtocol is now imported from victor.core.tool_types
# =============================================================================


# =============================================================================
# Workflow Provider Protocol
# =============================================================================


@runtime_checkable
class WorkflowProviderProtocol(Protocol):
    """Protocol for providing vertical-specific workflows.

    Workflows are named sequences of operations that can be
    triggered by user commands or automatically detected.

    Note: Return type uses Any to support both workflow classes (Type)
    and WorkflowDefinition instances. Implementations typically return
    Dict[str, WorkflowDefinition] from WorkflowBuilder.

    Example:
        class CodingWorkflowProvider(WorkflowProviderProtocol):
            def get_workflows(self) -> Dict[str, Any]:
                return {
                    "feature": feature_implementation_workflow(),
                    "bugfix": bug_fix_workflow(),
                }
    """

    @abstractmethod
    def get_workflows(self) -> Dict[str, Any]:
        """Get workflow definitions for this vertical.

        Returns:
            Dict mapping workflow names to WorkflowDefinition instances
            or workflow classes (Type). Most implementations return
            WorkflowDefinition from WorkflowBuilder.
        """
        ...

    def get_auto_workflows(self) -> List[Tuple[str, str]]:
        """Get automatically triggered workflows.

        Returns:
            List of (pattern, workflow_name) tuples for auto-triggering
        """
        return []


# =============================================================================
# Service Provider Protocol
# =============================================================================


@runtime_checkable
class ServiceProviderProtocol(Protocol):
    """Protocol for registering vertical-specific services with DI container.

    Enables verticals to register their own services alongside
    framework services for consistent lifecycle management.

    Example:
        class CodingServiceProvider(ServiceProviderProtocol):
            def register_services(self, container: ServiceContainer) -> None:
                container.register(
                    CodeCorrectionMiddlewareProtocol,
                    lambda c: CodeCorrectionMiddleware(),
                    ServiceLifetime.SINGLETON,
                )

            def get_required_services(self) -> List[Type]:
                return [CodeCorrectionMiddlewareProtocol]
    """

    @abstractmethod
    def register_services(
        self,
        container: "ServiceContainer",
        settings: "Settings",
    ) -> None:
        """Register vertical-specific services.

        Args:
            container: DI container to register services in
            settings: Application settings
        """
        ...

    def get_required_services(self) -> List[Type]:
        """Get list of required service types.

        Used for validation that all dependencies are registered.

        Returns:
            List of protocol/interface types this vertical requires
        """
        return []

    def get_optional_services(self) -> List[Type]:
        """Get list of optional service types.

        These are used if available but not required.

        Returns:
            List of optional protocol/interface types
        """
        return []


# =============================================================================
# RL Config Provider Protocol
# =============================================================================


@runtime_checkable
class RLConfigProviderProtocol(Protocol):
    """Protocol for providing RL (Reinforcement Learning) configuration.

    Enables verticals to configure RL learners, task type mappings,
    and quality thresholds for adaptive behavior.

    Example:
        class CodingRLConfigProvider(RLConfigProviderProtocol):
            def get_rl_config(self) -> Dict[str, Any]:
                return {
                    "active_learners": ["tool_selection", "semantic_threshold"],
                    "quality_thresholds": {"code_review": 0.8, "bugfix": 0.85},
                }
    """

    @abstractmethod
    def get_rl_config(self) -> Dict[str, Any]:
        """Get RL configuration for this vertical.

        Returns:
            Dict with RL configuration including:
            - active_learners: List of learner types to enable
            - quality_thresholds: Task-specific quality thresholds
            - task_type_mappings: Map task types to learner configs
        """
        ...

    def get_rl_hooks(self) -> Optional[Any]:
        """Get RL hooks for outcome recording.

        Returns:
            RLHooks instance or None
        """
        return None


# =============================================================================
# Team Spec Provider Protocol
# =============================================================================


@runtime_checkable
class TeamSpecProviderProtocol(Protocol):
    """Protocol for providing team specifications.

    Enables verticals to define multi-agent team configurations
    for complex task execution.

    Example:
        class CodingTeamSpecProvider(TeamSpecProviderProtocol):
            def get_team_specs(self) -> Dict[str, Any]:
                return {
                    "code_review_team": TeamSpec(
                        name="code_review_team",
                        formation=TeamFormation.PIPELINE,
                        agents=[...],
                    ),
                }
    """

    @abstractmethod
    def get_team_specs(self) -> Dict[str, Any]:
        """Get team specifications for this vertical.

        Returns:
            Dict mapping team names to TeamSpec instances
        """
        ...

    def get_default_team(self) -> Optional[str]:
        """Get the default team name.

        Returns:
            Default team name or None
        """
        return None


# =============================================================================
# Chain Provider Protocol
# =============================================================================


@runtime_checkable
class ChainProviderProtocol(Protocol):
    """Protocol for chain configuration providers.

    Enables verticals to define chains of operations that can be executed
    in sequence, supporting DIP by providing a protocol interface rather
    than concrete implementations.

    Example:
        class CodingChainProvider(ChainProviderProtocol):
            def get_chains(self) -> Dict[str, Any]:
                return {
                    "refactor": RefactorChain(steps=[...]),
                    "test_and_fix": TestFixChain(steps=[...]),
                }
    """

    def get_chains(self) -> Dict[str, Any]:
        """Get chain definitions for this vertical.

        Returns:
            Dict mapping chain names to chain configurations or instances
        """
        ...


# =============================================================================
# Persona Provider Protocol
# =============================================================================


@runtime_checkable
class PersonaProviderProtocol(Protocol):
    """Protocol for persona configuration providers.

    Enables verticals to define different personas (behavioral profiles)
    that affect how the agent interacts and responds.

    Example:
        class CodingPersonaProvider(PersonaProviderProtocol):
            def get_personas(self) -> Dict[str, Any]:
                return {
                    "senior_dev": {"name": "Senior Developer", "style": "thorough"},
                    "junior_dev": {"name": "Junior Developer", "style": "verbose"},
                }
    """

    def get_personas(self) -> Dict[str, Any]:
        """Get persona definitions for this vertical.

        Returns:
            Dict mapping persona names to persona configurations
        """
        ...


# =============================================================================
# Capability Provider Protocol
# =============================================================================


@runtime_checkable
class CapabilityProviderProtocol(Protocol):
    """Protocol for capability configuration providers.

    Enables verticals to declare which capabilities they support,
    allowing runtime discovery and feature toggling.

    Example:
        class CodingCapabilityProvider(CapabilityProviderProtocol):
            def get_capabilities(self) -> Dict[str, Any]:
                return {
                    "code_review": True,
                    "refactoring": True,
                    "test_generation": True,
                    "max_file_size": 100000,
                }
    """

    def get_capabilities(self) -> Dict[str, Any]:
        """Get capability definitions for this vertical.

        Returns:
            Dict mapping capability names to their configurations
            (typically bool for feature flags, or values for limits)
        """
        ...


# =============================================================================
# Vertical Provider Protocols (for isinstance() checks in integration)
# =============================================================================


@runtime_checkable
class VerticalRLProviderProtocol(Protocol):
    """Protocol for verticals providing RL configuration.

    This protocol enables type-safe isinstance() checks instead of hasattr()
    when integrating vertical RL configuration with the framework.

    Example:
        class CodingVertical(VerticalBase, VerticalRLProviderProtocol):
            @classmethod
            def get_rl_config_provider(cls) -> Optional[RLConfigProviderProtocol]:
                return CodingRLConfigProvider()

            @classmethod
            def get_rl_hooks(cls) -> Optional[Any]:
                return CodingRLHooks()
    """

    @classmethod
    def get_rl_config_provider(cls) -> Optional[RLConfigProviderProtocol]:
        """Get the RL configuration provider for this vertical.

        Returns:
            RLConfigProviderProtocol implementation or None
        """
        ...

    @classmethod
    def get_rl_hooks(cls) -> Optional[Any]:
        """Get RL hooks for outcome recording.

        Returns:
            RLHooks instance or None
        """
        ...


@runtime_checkable
class VerticalTeamProviderProtocol(Protocol):
    """Protocol for verticals providing team specifications.

    This protocol enables type-safe isinstance() checks instead of hasattr()
    when integrating vertical team specs with the framework.

    Example:
        class CodingVertical(VerticalBase, VerticalTeamProviderProtocol):
            @classmethod
            def get_team_spec_provider(cls) -> Optional[TeamSpecProviderProtocol]:
                return CodingTeamSpecProvider()
    """

    @classmethod
    def get_team_spec_provider(cls) -> Optional[TeamSpecProviderProtocol]:
        """Get the team specification provider for this vertical.

        Returns:
            TeamSpecProviderProtocol implementation or None
        """
        ...


@runtime_checkable
class VerticalWorkflowProviderProtocol(Protocol):
    """Protocol for verticals providing workflow definitions.

    This protocol enables type-safe isinstance() checks instead of hasattr()
    when integrating vertical workflows with the framework.

    Example:
        class CodingVertical(VerticalBase, VerticalWorkflowProviderProtocol):
            @classmethod
            def get_workflow_provider(cls) -> Optional[WorkflowProviderProtocol]:
                return CodingWorkflowProvider()
    """

    @classmethod
    def get_workflow_provider(cls) -> Optional[WorkflowProviderProtocol]:
        """Get the workflow provider for this vertical.

        Returns:
            WorkflowProviderProtocol implementation or None
        """
        ...


# =============================================================================
# Enrichment Strategy Protocol
# =============================================================================


@runtime_checkable
class EnrichmentStrategyProtocol(Protocol):
    """Protocol for vertical-specific prompt enrichment strategies.

    Enables auto prompt optimization where prompts are enriched
    with relevant context from vertical-specific sources:
    - Coding: Knowledge graph symbols, related code snippets
    - Research: Web search results, source citations
    - DevOps: Infrastructure context, command patterns
    - Data Analysis: Schema context, query patterns

    Example:
        class CodingEnrichmentStrategy:
            async def get_enrichments(
                self,
                prompt: str,
                context: "EnrichmentContext",
            ) -> List["ContextEnrichment"]:
                # Query knowledge graph for relevant symbols
                symbols = await self.graph.search(prompt)
                return [
                    ContextEnrichment(
                        type=EnrichmentType.KNOWLEDGE_GRAPH,
                        content=format_symbols(symbols),
                        priority=EnrichmentPriority.HIGH,
                    )
                ]

            def get_priority(self) -> int:
                return 50

            def get_token_allocation(self) -> float:
                return 0.4  # Use up to 40% of token budget
    """

    async def get_enrichments(
        self,
        prompt: str,
        context: Any,  # EnrichmentContext from victor.framework.enrichment
    ) -> List[Any]:  # List[ContextEnrichment]
        """Get enrichments for a prompt.

        Args:
            prompt: The prompt to enrich
            context: EnrichmentContext with task metadata

        Returns:
            List of ContextEnrichment objects to apply
        """
        ...

    def get_priority(self) -> int:
        """Get priority for this strategy.

        Lower values are processed first.

        Returns:
            Priority value (default 50)
        """
        ...

    def get_token_allocation(self) -> float:
        """Get fraction of token budget this strategy can use.

        Returns:
            Float between 0.0 and 1.0 (e.g., 0.4 for 40%)
        """
        ...


@runtime_checkable
class VerticalEnrichmentProviderProtocol(Protocol):
    """Protocol for verticals providing enrichment strategies.

    This protocol enables type-safe isinstance() checks when integrating
    vertical prompt enrichment with the framework.

    Example:
        class CodingVertical(VerticalBase, VerticalEnrichmentProviderProtocol):
            @classmethod
            def get_enrichment_strategy(cls) -> Optional[EnrichmentStrategyProtocol]:
                return CodingEnrichmentStrategy()
    """

    @classmethod
    def get_enrichment_strategy(cls) -> Optional[EnrichmentStrategyProtocol]:
        """Get the enrichment strategy for this vertical.

        Returns:
            EnrichmentStrategyProtocol implementation or None
        """
        ...


# =============================================================================
# Composite Vertical Extension
# =============================================================================


@dataclass
class VerticalExtensions:
    """Container for all vertical extension implementations.

    Aggregates all extension protocols for a vertical, making it easy
    to pass vertical capabilities to framework components.

    Attributes:
        middleware: List of middleware implementations
        safety_extensions: List of safety extensions
        prompt_contributors: List of prompt contributors
        mode_config_provider: Mode configuration provider
        tool_dependency_provider: Tool dependency provider
        workflow_provider: Workflow provider
        service_provider: Service provider
        enrichment_strategy: Prompt enrichment strategy for DSPy-like optimization
        tool_selection_strategy: Strategy for vertical-specific tool selection
    """

    middleware: List[MiddlewareProtocol] = field(default_factory=list)
    safety_extensions: List[SafetyExtensionProtocol] = field(default_factory=list)
    prompt_contributors: List[PromptContributorProtocol] = field(default_factory=list)
    mode_config_provider: Optional[ModeConfigProviderProtocol] = None
    tool_dependency_provider: Optional[ToolDependencyProviderProtocol] = None
    workflow_provider: Optional[WorkflowProviderProtocol] = None
    service_provider: Optional[ServiceProviderProtocol] = None
    rl_config_provider: Optional[RLConfigProviderProtocol] = None
    team_spec_provider: Optional[TeamSpecProviderProtocol] = None
    enrichment_strategy: Optional[EnrichmentStrategyProtocol] = None
    tool_selection_strategy: Optional[ToolSelectionStrategyProtocol] = None

    def get_all_task_hints(self) -> Dict[str, TaskTypeHint]:
        """Merge task hints from all contributors.

        Later contributors override earlier ones for same task type.

        Returns:
            Merged dict of task type hints
        """
        merged = {}
        for contributor in sorted(self.prompt_contributors, key=lambda c: c.get_priority()):
            merged.update(contributor.get_task_type_hints())
        return merged

    def get_all_safety_patterns(self) -> List[SafetyPattern]:
        """Collect safety patterns from all extensions.

        Returns:
            Combined list of safety patterns
        """
        patterns = []
        for ext in self.safety_extensions:
            patterns.extend(ext.get_bash_patterns())
            patterns.extend(ext.get_file_patterns())
        return patterns

    def get_all_mode_configs(self) -> Dict[str, ModeConfig]:
        """Get mode configs from provider.

        Returns:
            Dict of mode configurations
        """
        if self.mode_config_provider:
            return self.mode_config_provider.get_mode_configs()
        return {}


# =============================================================================
# Tool Selection Strategy Protocol (SOLID: OCP, DIP)
# =============================================================================


@dataclass
class ToolSelectionContext:
    """Context for tool selection decisions.

    Provides all information needed for vertical-specific tool selection.

    Attributes:
        task_type: Detected task type (e.g., "edit", "debug", "refactor")
        user_message: The user's message/query
        conversation_stage: Current conversation stage
        available_tools: Set of currently available tool names
        recent_tools: List of recently used tools (for context)
        metadata: Additional context metadata
    """

    task_type: str
    user_message: str
    conversation_stage: str = "exploration"
    available_tools: Set[str] = field(default_factory=set)
    recent_tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSelectionResult:
    """Result of vertical-specific tool selection.

    Attributes:
        priority_tools: Tools to prioritize (ordered by priority)
        excluded_tools: Tools to exclude from selection
        tool_weights: Custom weights for tool scoring (0.0-1.0)
        budget_override: Optional budget override for this selection
        reasoning: Optional explanation for selection decisions
    """

    priority_tools: List[str] = field(default_factory=list)
    excluded_tools: Set[str] = field(default_factory=set)
    tool_weights: Dict[str, float] = field(default_factory=dict)
    budget_override: Optional[int] = None
    reasoning: Optional[str] = None


@runtime_checkable
class ToolSelectionStrategyProtocol(Protocol):
    """Protocol for vertical-specific tool selection strategies.

    Enables verticals to customize tool selection based on domain knowledge.
    This follows the Strategy Pattern (OCP) and Dependency Inversion (DIP).

    The strategy is consulted during tool selection to:
    1. Prioritize domain-relevant tools
    2. Exclude inappropriate tools for the task
    3. Adjust tool weights for semantic scoring
    4. Override tool budgets based on task complexity

    Example:
        class CodingToolSelectionStrategy(ToolSelectionStrategyProtocol):
            def select_tools(
                self,
                context: ToolSelectionContext,
            ) -> ToolSelectionResult:
                if context.task_type == "refactor":
                    return ToolSelectionResult(
                        priority_tools=["read", "edit", "search"],
                        tool_weights={"edit": 0.9, "write": 0.7},
                        budget_override=15,
                        reasoning="Refactoring requires read-edit cycles",
                    )
                return ToolSelectionResult()

            def get_task_tool_mapping(self) -> Dict[str, List[str]]:
                return {
                    "edit": ["read", "edit", "search"],
                    "debug": ["read", "shell", "search"],
                    "test": ["shell", "read", "write"],
                }
    """

    def select_tools(
        self,
        context: ToolSelectionContext,
    ) -> ToolSelectionResult:
        """Select tools based on vertical-specific strategy.

        Args:
            context: Selection context with task and conversation info

        Returns:
            ToolSelectionResult with prioritized/excluded tools and weights
        """
        ...

    def get_task_tool_mapping(self) -> Dict[str, List[str]]:
        """Get mapping of task types to priority tools.

        Returns:
            Dict mapping task type names to ordered list of priority tools
        """
        ...

    def get_priority(self) -> int:
        """Get priority for this strategy.

        Lower values are processed first when multiple strategies exist.

        Returns:
            Priority value (default 50)
        """
        return 50


@runtime_checkable
class VerticalToolSelectionProviderProtocol(Protocol):
    """Protocol for verticals providing tool selection strategies.

    This protocol enables type-safe isinstance() checks when integrating
    vertical tool selection with the framework.

    Example:
        class CodingVertical(VerticalBase, VerticalToolSelectionProviderProtocol):
            @classmethod
            def get_tool_selection_strategy(cls) -> Optional[ToolSelectionStrategyProtocol]:
                return CodingToolSelectionStrategy()
    """

    @classmethod
    def get_tool_selection_strategy(cls) -> Optional[ToolSelectionStrategyProtocol]:
        """Get the tool selection strategy for this vertical.

        Returns:
            ToolSelectionStrategyProtocol implementation or None
        """
        ...


__all__ = [
    # Enums
    "MiddlewarePriority",
    # Data types (re-exported from victor.core.vertical_types)
    "StageDefinition",
    "MiddlewareResult",
    "SafetyPattern",
    "TaskTypeHint",
    "ModeConfig",
    "ToolDependency",
    "TieredToolConfig",
    # Tool Selection Data Types
    "ToolSelectionContext",
    "ToolSelectionResult",
    # Protocols
    "MiddlewareProtocol",
    "SafetyExtensionProtocol",
    "PromptContributorProtocol",
    "ModeConfigProviderProtocol",
    "ToolDependencyProviderProtocol",
    "WorkflowProviderProtocol",
    "ServiceProviderProtocol",
    "RLConfigProviderProtocol",
    "TeamSpecProviderProtocol",
    "ChainProviderProtocol",
    "PersonaProviderProtocol",
    "CapabilityProviderProtocol",
    "EnrichmentStrategyProtocol",
    "ToolSelectionStrategyProtocol",
    # Vertical Provider Protocols (for isinstance() checks)
    "VerticalRLProviderProtocol",
    "VerticalTeamProviderProtocol",
    "VerticalWorkflowProviderProtocol",
    "VerticalEnrichmentProviderProtocol",
    "VerticalToolSelectionProviderProtocol",
    # Composite
    "VerticalExtensions",
]
