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

"""ISP-Compliant Vertical Provider Protocols.

This module provides segregated Protocol classes that group the 26+ hooks in
VerticalBase into focused, single-responsibility interfaces following the
Interface Segregation Principle (ISP).

Instead of forcing verticals to implement all possible methods (many with
empty defaults), verticals can implement only the provider protocols that
are relevant to their functionality. The framework uses isinstance() checks
to determine which capabilities a vertical supports.

Protocol Categories:
    - MiddlewareProvider: Middleware for tool execution
    - SafetyProvider: Safety patterns and extensions
    - WorkflowProvider: Workflow definitions and management
    - TeamProvider: Multi-agent team specifications
    - RLProvider: Reinforcement learning configuration
    - EnrichmentProvider: Prompt enrichment strategies
    - ToolProvider: Tool sets and tool graphs
    - HandlerProvider: Compute handlers for workflows
    - CapabilityProvider: Capability declarations

Usage:
    from victor.core.verticals.protocols.providers import (
        MiddlewareProvider,
        SafetyProvider,
        ToolProvider,
    )

    # Type-safe isinstance checks
    if isinstance(vertical, MiddlewareProvider):
        middleware = vertical.get_middleware()

    if isinstance(vertical, SafetyProvider):
        safety = vertical.get_safety_extension()

Example Implementation:
    from victor.core.verticals import VerticalBase
    from victor.core.verticals.protocols.providers import (
        MiddlewareProvider,
        SafetyProvider,
    )

    class SecurityVertical(VerticalBase):
        '''Security-focused vertical implementing specific provider protocols.'''
        name = "security"

        @classmethod
        def get_tools(cls) -> List[str]:
            return ["read", "security_scan", "audit"]

        @classmethod
        def get_system_prompt(cls) -> str:
            return "You are a security expert..."

        # Only implement the providers relevant to this vertical
        @classmethod
        def get_middleware(cls) -> List[Any]:
            from mypackage.middleware import SecurityMiddleware
            return [SecurityMiddleware()]

        @classmethod
        def get_safety_extension(cls) -> Optional[Any]:
            from mypackage.safety import SecuritySafetyExtension
            return SecuritySafetyExtension()

    # Framework code can check capabilities:
    vertical = SecurityVertical
    if isinstance(vertical, MiddlewareProvider):
        print("Has middleware support")
    if isinstance(vertical, SafetyProvider):
        print("Has safety extensions")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

# =============================================================================
# Middleware Provider Protocol
# =============================================================================


@runtime_checkable
class MiddlewareProvider(Protocol):
    """Protocol for verticals that provide middleware.

    Middleware intercepts and modifies tool calls before and after execution.
    Verticals implementing this protocol can provide domain-specific middleware
    for validation, transformation, logging, or processing.

    Example:
        class CodingVertical(VerticalBase, MiddlewareProvider):
            @classmethod
            def get_middleware(cls) -> List[Any]:
                from victor_coding.middleware import CodeValidationMiddleware
                return [CodeValidationMiddleware()]
    """

    @classmethod
    def get_middleware(cls) -> List[Any]:
        """Get middleware implementations for this vertical.

        Returns:
            List of middleware implementations (MiddlewareProtocol instances)
        """
        ...


# =============================================================================
# Safety Provider Protocol
# =============================================================================


@runtime_checkable
class SafetyProvider(Protocol):
    """Protocol for verticals that provide safety extensions.

    Safety extensions define dangerous operation patterns specific to the
    vertical's domain. These patterns are used to warn users or block
    potentially harmful operations.

    Example:
        class DevOpsVertical(VerticalBase, SafetyProvider):
            @classmethod
            def get_safety_extension(cls) -> Optional[Any]:
                from victor_devops.safety import DevOpsSafetyExtension
                return DevOpsSafetyExtension()
    """

    @classmethod
    def get_safety_extension(cls) -> Optional[Any]:
        """Get safety extension for this vertical.

        Returns:
            Safety extension (SafetyExtensionProtocol) or None
        """
        ...


# =============================================================================
# Workflow Provider Protocol
# =============================================================================


@runtime_checkable
class WorkflowProvider(Protocol):
    """Protocol for verticals that provide workflows.

    Workflows are named sequences of operations that can be triggered
    by user commands or automatically detected based on context.

    Example:
        class CodingVertical(VerticalBase, WorkflowProvider):
            @classmethod
            def get_workflow_provider(cls) -> Optional[Any]:
                from victor_coding.workflows import CodingWorkflowProvider
                return CodingWorkflowProvider()

            @classmethod
            def get_workflows(cls) -> Dict[str, Any]:
                provider = cls.get_workflow_provider()
                return provider.get_workflows() if provider else {}
    """

    @classmethod
    def get_workflow_provider(cls) -> Optional[Any]:
        """Get workflow provider for this vertical.

        Returns:
            Workflow provider (WorkflowProviderProtocol) or None
        """
        ...

    @classmethod
    def get_workflows(cls) -> Dict[str, Any]:
        """Get workflow definitions for this vertical.

        Returns:
            Dict mapping workflow names to workflow definitions
        """
        ...


# =============================================================================
# Team Provider Protocol
# =============================================================================


@runtime_checkable
class TeamProvider(Protocol):
    """Protocol for verticals that provide team specs.

    Team specifications define multi-agent team configurations for
    complex task execution with multiple specialized agents.

    Example:
        class ResearchVertical(VerticalBase, TeamProvider):
            @classmethod
            def get_team_spec_provider(cls) -> Optional[Any]:
                from victor_research.teams import ResearchTeamProvider
                return ResearchTeamProvider()

            @classmethod
            def get_team_specs(cls) -> Dict[str, Any]:
                provider = cls.get_team_spec_provider()
                return provider.get_team_specs() if provider else {}
    """

    @classmethod
    def get_team_spec_provider(cls) -> Optional[Any]:
        """Get team specification provider for this vertical.

        Returns:
            Team spec provider (TeamSpecProviderProtocol) or None
        """
        ...

    @classmethod
    def get_team_specs(cls) -> Dict[str, Any]:
        """Get team specifications for this vertical.

        Returns:
            Dict mapping team names to TeamSpec instances
        """
        ...


# =============================================================================
# RL Provider Protocol
# =============================================================================


@runtime_checkable
class RLProvider(Protocol):
    """Protocol for verticals that provide RL configuration.

    RL (Reinforcement Learning) configuration enables adaptive behavior
    through learner configurations, task type mappings, and quality thresholds.

    Example:
        class CodingVertical(VerticalBase, RLProvider):
            @classmethod
            def get_rl_config_provider(cls) -> Optional[Any]:
                from victor_coding.rl import CodingRLProvider
                return CodingRLProvider()

            @classmethod
            def get_rl_hooks(cls) -> List[Any]:
                provider = cls.get_rl_config_provider()
                return [provider.get_rl_hooks()] if provider else []
    """

    @classmethod
    def get_rl_config_provider(cls) -> Optional[Any]:
        """Get RL configuration provider for this vertical.

        Returns:
            RL config provider (RLConfigProviderProtocol) or None
        """
        ...

    @classmethod
    def get_rl_hooks(cls) -> List[Any]:
        """Get RL hooks for outcome recording.

        Returns:
            List of RLHooks instances
        """
        ...


# =============================================================================
# Enrichment Provider Protocol
# =============================================================================


@runtime_checkable
class EnrichmentProvider(Protocol):
    """Protocol for verticals that provide enrichment strategies.

    Enrichment strategies provide vertical-specific context for prompt
    optimization (DSPy-like auto prompt enhancement). Examples include:
    - Coding: Knowledge graph symbols, related code snippets
    - Research: Web search results, source citations
    - DevOps: Infrastructure context, command patterns
    - Data Analysis: Schema context, query patterns

    Example:
        class CodingVertical(VerticalBase, EnrichmentProvider):
            @classmethod
            def get_enrichment_strategy(cls) -> Optional[Any]:
                from victor_coding.enrichment import CodingEnrichmentStrategy
                return CodingEnrichmentStrategy()
    """

    @classmethod
    def get_enrichment_strategy(cls) -> Optional[Any]:
        """Get vertical-specific enrichment strategy.

        Returns:
            EnrichmentStrategyProtocol implementation or None
        """
        ...


# =============================================================================
# Tool Provider Protocol
# =============================================================================


@runtime_checkable
class ToolProvider(Protocol):
    """Protocol for verticals that provide tools.

    Defines the tools available to the vertical and optional tool execution
    graphs for defining tool dependencies and sequences.

    Example:
        class CodingVertical(VerticalBase, ToolProvider):
            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read", "write", "edit", "grep", "run_tests"]

            @classmethod
            def get_tool_graph(cls) -> Optional[Any]:
                from victor_coding.tools import CodingToolGraph
                return CodingToolGraph()
    """

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get the list of tool names for this vertical.

        Returns:
            List of tool names to enable
        """
        ...

    @classmethod
    def get_tool_graph(cls) -> Optional[Any]:
        """Get tool execution graph for this vertical.

        Returns:
            ToolExecutionGraph instance or None
        """
        ...


# =============================================================================
# Handler Provider Protocol
# =============================================================================


@runtime_checkable
class HandlerProvider(Protocol):
    """Protocol for verticals that provide compute handlers.

    Compute handlers are registered with the HandlerRegistry for
    workflow execution. They process specific node types in YAML workflows.

    Example:
        class DataAnalysisVertical(VerticalBase, HandlerProvider):
            @classmethod
            def get_handlers(cls) -> Dict[str, Any]:
                from victor_dataanalysis.handlers import HANDLERS
                return HANDLERS
    """

    @classmethod
    def get_handlers(cls) -> Dict[str, Any]:
        """Get compute handlers for workflow execution.

        Returns:
            Dict mapping handler name to handler instance
        """
        ...


# =============================================================================
# Capability Provider Protocol
# =============================================================================


@runtime_checkable
class CapabilityProvider(Protocol):
    """Protocol for verticals that provide capabilities.

    Capability providers declare which features the vertical supports,
    enabling runtime discovery and feature toggling.

    Example:
        class CodingVertical(VerticalBase, CapabilityProvider):
            @classmethod
            def get_capability_provider(cls) -> Optional[Any]:
                from victor_coding.capabilities import CodingCapabilityProvider
                return CodingCapabilityProvider()
    """

    @classmethod
    def get_capability_provider(cls) -> Optional[Any]:
        """Get capability provider for this vertical.

        Returns:
            CapabilityProviderProtocol implementation or None
        """
        ...


# =============================================================================
# Mode Config Provider Protocol
# =============================================================================


@runtime_checkable
class ModeConfigProvider(Protocol):
    """Protocol for verticals that provide mode configurations.

    Mode configurations define operational modes like 'fast', 'thorough',
    or 'explore' with their respective tool budgets, iterations, and
    temperature settings.

    Example:
        class CodingVertical(VerticalBase, ModeConfigProvider):
            @classmethod
            def get_mode_config_provider(cls) -> Optional[Any]:
                from victor_coding.modes import CodingModeProvider
                return CodingModeProvider()

            @classmethod
            def get_mode_config(cls) -> Dict[str, Any]:
                return {
                    "fast": {"tool_budget": 10, "max_iterations": 20},
                    "thorough": {"tool_budget": 50, "max_iterations": 50},
                }
    """

    @classmethod
    def get_mode_config_provider(cls) -> Optional[Any]:
        """Get mode configuration provider for this vertical.

        Returns:
            Mode config provider (ModeConfigProviderProtocol) or None
        """
        ...

    @classmethod
    def get_mode_config(cls) -> Dict[str, Any]:
        """Get mode configurations for this vertical.

        Returns:
            Dictionary mapping mode names to configuration dicts
        """
        ...


# =============================================================================
# Prompt Contributor Provider Protocol
# =============================================================================


@runtime_checkable
class PromptContributorProvider(Protocol):
    """Protocol for verticals that provide prompt contributors.

    Prompt contributors add vertical-specific hints, sections, and
    task type mappings to prompts.

    Example:
        class ResearchVertical(VerticalBase, PromptContributorProvider):
            @classmethod
            def get_prompt_contributor(cls) -> Optional[Any]:
                from victor_research.prompts import ResearchPromptContributor
                return ResearchPromptContributor()

            @classmethod
            def get_task_type_hints(cls) -> Dict[str, Any]:
                return {
                    "search": {"hint": "Use web search tools", "priority_tools": ["web"]},
                    "synthesize": {"hint": "Combine sources", "priority_tools": ["write"]},
                }
    """

    @classmethod
    def get_prompt_contributor(cls) -> Optional[Any]:
        """Get prompt contributor for this vertical.

        Returns:
            Prompt contributor (PromptContributorProtocol) or None
        """
        ...

    @classmethod
    def get_task_type_hints(cls) -> Dict[str, Any]:
        """Get task-type-specific prompt hints.

        Returns:
            Dictionary mapping task types to hint configurations
        """
        ...


# =============================================================================
# Tool Dependency Provider Protocol
# =============================================================================


@runtime_checkable
class ToolDependencyProvider(Protocol):
    """Protocol for verticals that provide tool dependencies.

    Tool dependency providers define relationships between tools,
    enabling intelligent tool sequencing and pre-requisite handling.

    Example:
        class CodingVertical(VerticalBase, ToolDependencyProvider):
            @classmethod
            def get_tool_dependency_provider(cls) -> Optional[Any]:
                from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider
                return create_vertical_tool_dependency_provider("coding")
    """

    @classmethod
    def get_tool_dependency_provider(cls) -> Optional[Any]:
        """Get tool dependency provider for this vertical.

        Returns:
            Tool dependency provider (ToolDependencyProviderProtocol) or None
        """
        ...


# =============================================================================
# Tiered Tool Config Provider Protocol
# =============================================================================


@runtime_checkable
class TieredToolConfigProvider(Protocol):
    """Protocol for verticals that provide tiered tool configuration.

    Tiered tool configuration enables context-efficient tool management with
    three tiers:
    1. Mandatory: Always included tools (e.g., read, ls)
    2. Vertical Core: Always included for this vertical
    3. Semantic Pool: Selected based on query similarity

    Example:
        class ResearchVertical(VerticalBase, TieredToolConfigProvider):
            @classmethod
            def get_tiered_tool_config(cls) -> Optional[Any]:
                from victor.core.vertical_types import TieredToolConfig
                return TieredToolConfig(
                    mandatory={"read", "ls"},
                    vertical_core={"web_search", "web_fetch"},
                    semantic_pool={"write", "edit"},
                )
    """

    @classmethod
    def get_tiered_tool_config(cls) -> Optional[Any]:
        """Get tiered tool configuration for this vertical.

        Returns:
            TieredToolConfig instance or None
        """
        ...


# =============================================================================
# Service Provider Protocol
# =============================================================================


@runtime_checkable
class ServiceProvider(Protocol):
    """Protocol for verticals that provide DI services.

    Service providers register vertical-specific services with the
    dependency injection container.

    Example:
        class CodingVertical(VerticalBase, ServiceProvider):
            @classmethod
            def get_service_provider(cls) -> Optional[Any]:
                from victor_coding.services import CodingServiceProvider
                return CodingServiceProvider()
    """

    @classmethod
    def get_service_provider(cls) -> Optional[Any]:
        """Get service provider for this vertical.

        Returns:
            Service provider (ServiceProviderProtocol) or None
        """
        ...


__all__ = [
    # Core provider protocols
    "MiddlewareProvider",
    "SafetyProvider",
    "WorkflowProvider",
    "TeamProvider",
    "RLProvider",
    "EnrichmentProvider",
    "ToolProvider",
    "HandlerProvider",
    "CapabilityProvider",
    # Additional provider protocols
    "ModeConfigProvider",
    "PromptContributorProvider",
    "ToolDependencyProvider",
    "TieredToolConfigProvider",
    "ServiceProvider",
]
