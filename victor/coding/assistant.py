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

"""CodingAssistant - Victor's primary vertical for software development.

This module defines the CodingAssistant vertical with full integration
of coding-specific extensions, middleware, and configurations.

The CodingAssistant provides:
- 45+ tools optimized for coding tasks
- Stage-aware tool selection for workflow optimization
- Code validation and correction middleware
- Git operation safety checks
- Task-type-specific prompt hints
- Mode configurations for different coding scenarios
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type

if TYPE_CHECKING:
    from victor.framework.prompt_builder import PromptBuilder

from victor.core.verticals.base import VerticalBase, VerticalConfig
from victor.core.vertical_types import StageDefinition
from victor.core.verticals.defaults.tool_defaults import (
    COMMON_REQUIRED_TOOLS,
    merge_required_tools,
)
from victor.core.verticals.protocols import (
    MiddlewareProtocol,
    SafetyExtensionProtocol,
    PromptContributorProtocol,
    ModeConfigProviderProtocol,
    ToolDependencyProviderProtocol,
    WorkflowProviderProtocol,
    ServiceProviderProtocol,
    TieredToolConfig,
    VerticalExtensions,
)

# Import ISP-compliant provider protocols
from victor.core.verticals.protocols.providers import (
    CapabilityProvider,
    HandlerProvider,
    MiddlewareProvider,
    ModeConfigProvider,
    PromptContributorProvider,
    ServiceProvider,
    TieredToolConfigProvider,
    ToolDependencyProvider,
    ToolProvider,
)

# Phase 2.1: Protocol auto-registration decorator
from victor.core.verticals.protocol_decorators import register_protocols

# Phase 3: Import framework capabilities
from victor.framework.capabilities import (
    FileOperationsCapability,
    PromptContributionCapability,
)


@register_protocols
class CodingAssistant(VerticalBase):
    """Software development assistant vertical.

    This is Victor's default configuration, optimized for:
    - Code exploration and understanding
    - Bug fixing and refactoring
    - Feature implementation
    - Testing and verification
    - Git operations and version control

    The CodingAssistant provides full integration with the framework
    through extension protocols, enabling:
    - Code correction middleware for validation
    - Git safety checks for dangerous operations
    - Task-type-specific prompt hints
    - Mode configurations for different scenarios
    - Tool dependency graph for intelligent selection

    ISP Compliance:
        This vertical explicitly declares which protocols it implements through
        protocol registration, rather than inheriting from all possible protocol
        interfaces. This follows the Interface Segregation Principle (ISP) by
        implementing only needed protocols.

        Implemented Protocols:
        - ToolProvider: Provides 45+ tools optimized for coding tasks
        - PromptContributorProvider: Provides coding-specific task hints
        - MiddlewareProvider: Provides code correction and git safety middleware
        - ToolDependencyProvider: Provides tool dependency patterns
        - HandlerProvider: Provides workflow compute handlers
        - CapabilityProvider: Provides coding capability configurations
        - ModeConfigProvider: Provides mode configurations (build, plan, explore)
        - ServiceProvider: Provides coding-specific DI services
        - TieredToolConfigProvider: Provides tiered tool configuration

        Note: WorkflowProvider is NOT registered as this vertical uses auto-generated
        workflow provider getter without implementing the required get_workflows() method.

    Example:
        from victor.coding import CodingAssistant

        # Get vertical configuration
        config = CodingAssistant.get_config()

        # Get extensions for framework integration
        extensions = CodingAssistant.get_extensions()

        # Create agent with this vertical
        agent = await Agent.create(
            tools=config.tools,
            vertical=CodingAssistant,
        )
    """

    # Override class variables from base
    name: ClassVar[str] = "coding"
    description: ClassVar[str] = (
        "Software development assistant for code exploration, writing, and refactoring"
    )
    version: ClassVar[str] = "2.0.0"

    # =========================================================================
    # Phase 3: Framework Capabilities
    # =========================================================================
    # Framework prompt contributions (common hints like read_first, verify_changes)
    _prompt_contrib = PromptContributionCapability()

    # Framework file operations (read, write, edit, grep)
    _file_ops = FileOperationsCapability()

    # =========================================================================
    # Extension Caching
    # =========================================================================
    # Individual extension caching is provided by VerticalBase._get_cached_extension()
    # Composite extensions caching is provided by VerticalBase.get_extensions()
    # Use clear_config_cache() to invalidate all caches.

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get tools optimized for software development.

        Uses canonical tool names from victor.tools.tool_names.

        Extends COMMON_REQUIRED_TOOLS with coding-specific tools using
        merge_required_tools() to eliminate code duplication and maintain
        consistency across verticals.

        Returns:
            List of tool names including filesystem, git, shell, and code tools.
        """
        from victor.tools.tool_names import ToolNames

        # Coding-specific tools beyond common required tools
        coding_tools = [
            # Core filesystem (beyond framework basics)
            ToolNames.LS,  # list_directory -> ls
            ToolNames.OVERVIEW,  # get_project_overview -> overview
            # Search
            ToolNames.GREP,  # grep (framework capability)
            ToolNames.CODE_SEARCH,  # semantic_code_search -> code_search
            ToolNames.PLAN,  # plan_files -> plan
            # Git (unified git tool handles all operations)
            ToolNames.GIT,  # Git operations
            # Shell
            ToolNames.SHELL,  # execute_bash -> shell
            # Code intelligence
            ToolNames.LSP,  # lsp operations
            ToolNames.SYMBOL,  # find_symbol -> symbol
            ToolNames.REFS,  # find_references -> refs
            # Refactoring
            ToolNames.RENAME,  # refactor_rename_symbol -> rename
            ToolNames.EXTRACT,  # refactor_extract_function -> extract
            # Testing
            ToolNames.TEST,  # run_tests -> test
            # Docker
            ToolNames.DOCKER,  # docker operations
            # Web (for documentation)
            ToolNames.WEB_SEARCH,  # web_search
            ToolNames.WEB_FETCH,  # web_fetch
        ]

        # Merge common required tools with coding-specific tools
        return merge_required_tools(COMMON_REQUIRED_TOOLS, coding_tools)

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get coding-focused system prompt.

        Returns:
            System prompt optimized for software development.
        """
        from victor.coding.coding_prompt_template import CodingPromptTemplate

        return CodingPromptTemplate().build()

    # =========================================================================
    # PromptBuilder Support (Phase 7)
    # =========================================================================

    @classmethod
    def _get_vertical_prompt(cls) -> str:
        """Get coding-specific prompt content for PromptBuilder.

        Returns:
            Coding-specific vertical prompt content
        """
        from victor.coding.coding_prompt_template import CodingPromptTemplate

        return CodingPromptTemplate().get_vertical_prompt()

    @classmethod
    def get_prompt_builder(cls) -> "PromptBuilder":
        """Get configured PromptBuilder for coding vertical.

        Returns:
            PromptBuilder with coding-specific configuration

        Note:
            Uses CodingPromptTemplate for consistent prompt structure
            following the Template Method pattern.
        """
        from victor.coding.coding_prompt_template import CodingPromptTemplate

        # Use template for consistent structure
        template = CodingPromptTemplate()
        builder = template.get_prompt_builder()

        return builder

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Get coding-specific stage definitions.

        Uses canonical tool names from victor.tools.tool_names.

        Returns:
            Stage definitions optimized for software development workflow.
        """
        from victor.tools.tool_names import ToolNames

        return {
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Understanding the coding request",
                tools={ToolNames.READ, ToolNames.LS, ToolNames.OVERVIEW, ToolNames.GREP},
                keywords=["what", "how", "explain", "where", "show me"],
                next_stages={"PLANNING", "READING"},
            ),
            "PLANNING": StageDefinition(
                name="PLANNING",
                description="Planning the implementation approach",
                tools={ToolNames.GREP, ToolNames.PLAN, ToolNames.OVERVIEW, ToolNames.READ},
                keywords=["plan", "approach", "design", "architecture", "strategy"],
                next_stages={"READING", "EXECUTION"},
            ),
            "READING": StageDefinition(
                name="READING",
                description="Reading code and gathering context",
                tools={
                    ToolNames.READ,
                    ToolNames.CODE_SEARCH,
                    ToolNames.GREP,
                    ToolNames.LSP,
                    ToolNames.SYMBOL,
                    ToolNames.REFS,
                },
                keywords=["read", "show", "find", "look", "check", "search"],
                next_stages={"ANALYSIS", "EXECUTION"},
            ),
            "ANALYSIS": StageDefinition(
                name="ANALYSIS",
                description="Analyzing code structure and dependencies",
                tools={ToolNames.LSP, ToolNames.SYMBOL, ToolNames.REFS, ToolNames.OVERVIEW},
                keywords=["analyze", "review", "understand", "why", "how does"],
                next_stages={"EXECUTION", "PLANNING"},
            ),
            "EXECUTION": StageDefinition(
                name="EXECUTION",
                description="Implementing changes",
                tools={
                    ToolNames.WRITE,
                    ToolNames.EDIT,
                    ToolNames.SHELL,
                    ToolNames.GIT,
                    ToolNames.RENAME,
                },
                keywords=[
                    "change",
                    "modify",
                    "create",
                    "add",
                    "remove",
                    "fix",
                    "implement",
                    "write",
                    "update",
                    "refactor",
                ],
                next_stages={"VERIFICATION", "COMPLETION"},
            ),
            "VERIFICATION": StageDefinition(
                name="VERIFICATION",
                description="Testing and validating changes",
                tools={ToolNames.SHELL, ToolNames.TEST, ToolNames.GIT, ToolNames.READ},
                keywords=["test", "verify", "check", "validate", "run", "build"],
                next_stages={"COMPLETION", "EXECUTION"},
            ),
            "COMPLETION": StageDefinition(
                name="COMPLETION",
                description="Committing and summarizing",
                tools={ToolNames.GIT},
                keywords=["done", "finish", "complete", "commit", "summarize"],
                next_stages=set(),
            ),
        }

    @classmethod
    def customize_config(cls, config: VerticalConfig) -> VerticalConfig:
        """Add coding-specific configuration.

        Args:
            config: Base configuration.

        Returns:
            Customized configuration.
        """
        config.metadata["supports_lsp"] = True
        config.metadata["supports_git"] = True
        config.metadata["max_file_size"] = 1_000_000  # 1MB
        config.metadata["supported_languages"] = [
            "python",
            "typescript",
            "javascript",
            "rust",
            "go",
            "java",
            "c",
            "cpp",
        ]
        return config

    # =========================================================================
    # Extension Protocol Methods
    # =========================================================================
    # Most extension getters are auto-generated by VerticalExtensionLoaderMeta
    # to eliminate ~800 lines of duplication. Only override for custom logic.

    @classmethod
    def get_middleware(cls) -> List[MiddlewareProtocol]:
        """Get coding-specific middleware (cached).

        Custom implementation for Coding vertical with CodeCorrectionMiddleware
        and GitSafetyMiddleware. Auto-generated getter would return empty list.

        Returns:
            List of middleware implementations
        """

        def _create_middleware() -> List[MiddlewareProtocol]:
            from victor.coding.middleware import (
                CodeCorrectionMiddleware,
                GitSafetyMiddleware,
            )

            return [
                CodeCorrectionMiddleware(enabled=True, auto_fix=True),
                GitSafetyMiddleware(block_dangerous=False, warn_on_risky=True),
            ]

        result = cls._get_cached_extension("middleware", _create_middleware)
        return result

    @classmethod
    def get_tool_dependency_provider(cls) -> Optional[ToolDependencyProviderProtocol]:
        """Get coding-specific tool dependency provider (cached).

        Custom implementation using create_vertical_tool_dependency_provider.
        Auto-generated getter would try to import from victor.coding.tool_dependencies.

        Returns:
            Tool dependency provider
        """

        def _create() -> Optional[ToolDependencyProviderProtocol]:
            from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider

            return create_vertical_tool_dependency_provider("coding")

        result = cls._get_cached_extension("tool_dependency_provider", _create)
        return result

    @classmethod
    def get_composed_chains(cls) -> Dict[str, Any]:
        """Get pre-built LCEL-composed tool chains (cached).

        Custom implementation for Coding vertical with CODING_CHAINS constant.
        Auto-generated getter would try to import COMPOSED_CHAINS constant.

        Provides LCEL composition chains for common coding tasks:
        - explore_file: Read file and analyze symbols
        - analyze_function: Get function details with references
        - safe_edit: Edit with verification
        - git_status: Parallel git state collection
        - search_with_context: Code search with result context
        - lint: Language-aware linting
        - test_discovery: Find test files
        - review_analysis: Parallel review data collection

        Returns:
            Dict mapping chain names to Runnable instances
        """

        def _create() -> Dict[str, Any]:
            from victor.coding.composed_chains import CODING_CHAINS

            return CODING_CHAINS

        result = cls._get_cached_extension("composed_chains", _create)
        return result

    @classmethod
    def get_personas(cls) -> Dict[str, Any]:
        """Get persona definitions for team members (cached).

        Custom implementation for Coding vertical with CODING_PERSONAS constant.
        Auto-generated getter would try to import PERSONAS constant.

        Provides rich persona definitions with:
        - Expertise categories
        - Communication styles
        - Decision-making preferences
        - Behavioral traits

        Available personas:
        - code_archaeologist: Deep code analysis expert
        - security_auditor: Security-focused reviewer
        - architect: Solution designer
        - refactoring_strategist: Safe refactoring planner
        - craftsman: Clean code implementer
        - debugger: Bug hunting specialist
        - quality_guardian: Code review expert
        - test_specialist: Testing expert

        Returns:
            Dict mapping persona names to CodingPersona instances
        """

        def _create() -> Dict[str, Any]:
            from victor.coding.teams import CODING_PERSONAS

            return CODING_PERSONAS

        result = cls._get_cached_extension("personas", _create)
        return result

    @classmethod
    def get_handlers(cls) -> Dict[str, Any]:
        """Get compute handlers for workflow execution.

        Provides coding-specific handlers for workflow nodes:
        - code_validation: Validates code changes (lint, type check)
        - test_runner: Runs tests with timeout and reporting
        - code_analyzer: Deep code analysis for context gathering

        Returns:
            Dict mapping handler name to handler instance
        """
        from victor.framework.handler_registry import HandlerRegistry

        registry = HandlerRegistry.get_instance()

        # Auto-discover handlers if not already registered
        coding_handlers = registry.list_by_vertical("coding")
        if not coding_handlers:
            registry.discover_from_vertical("coding")

        handlers = {}
        for handler_name in registry.list_by_vertical("coding"):
            entry = registry.get_entry(handler_name)
            if entry:
                handlers[handler_name] = entry.handler
        return handlers

    @classmethod
    def get_capability_configs(cls) -> Dict[str, Any]:
        """Get coding capability configurations for centralized storage.

        Returns coding capability configurations for VerticalContext storage.
        This replaces direct orchestrator attribute assignment patterns like:
        - orchestrator.code_style = {...}
        - orchestrator.test_config = {...}
        - orchestrator.lsp_config = {...}

        Returns:
            Dict with coding capability configurations
        """
        from victor.coding.capabilities import get_capability_configs

        return get_capability_configs()

    # NOTE: The following getters are auto-generated by VerticalExtensionLoaderMeta:
    # - get_safety_extension()
    # - get_prompt_contributor()
    # - get_mode_config_provider()
    # - get_workflow_provider()
    # - get_service_provider()
    # - get_tiered_tools()
    # - get_rl_config_provider()
    # - get_rl_hooks()
    # - get_team_spec_provider()
    # - get_capability_provider()
    #
    # get_extensions() is inherited from VerticalBase with full caching support.
    # To clear all caches, use cls.clear_config_cache().


__all__ = ["CodingAssistant"]


# Protocol registration is now handled by @register_protocols decorator
# which auto-detects implemented protocols:
# - ToolProvider (get_tools)
# - PromptContributorProvider (get_prompt_contributor)
# - MiddlewareProvider (get_middleware)
# - ToolDependencyProvider (get_tool_dependency_provider)
# - HandlerProvider (get_handlers)
# - CapabilityProvider (get_capability_configs)
# - ModeConfigProvider (get_mode_config_provider)
# - ServiceProvider (get_service_provider)
# - TieredToolConfigProvider (get_tiered_tools)
#
# ISP Compliance Note:
# This vertical implements only the protocols it needs. The @register_protocols
# decorator auto-detects and registers these protocols at class decoration time.
