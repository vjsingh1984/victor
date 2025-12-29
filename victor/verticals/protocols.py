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
    from victor.verticals.protocols import (
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
from victor.safety.types import SafetyPattern

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

    Example:
        class CodingWorkflowProvider(WorkflowProviderProtocol):
            def get_workflows(self) -> Dict[str, Type]:
                from victor.verticals.coding.workflows import NewFeatureWorkflow
                return {"new_feature": NewFeatureWorkflow}
    """

    @abstractmethod
    def get_workflows(self) -> Dict[str, Type]:
        """Get workflow classes for this vertical.

        Returns:
            Dict mapping workflow names to workflow classes
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
    # Composite
    "VerticalExtensions",
]
