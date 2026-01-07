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

from typing import Any, Dict, List, Optional, Type

from victor.core.verticals.base import StageDefinition, VerticalBase, VerticalConfig
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

    name = "coding"
    description = "Software development assistant for code exploration, writing, and refactoring"
    version = "2.0.0"  # Extension support

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

        Returns:
            List of tool names including filesystem, git, shell, and code tools.
        """
        from victor.tools.tool_names import ToolNames

        return [
            # Core filesystem
            ToolNames.READ,  # read_file -> read
            ToolNames.WRITE,  # write_file -> write
            ToolNames.EDIT,  # edit_files -> edit
            ToolNames.LS,  # list_directory -> ls
            ToolNames.OVERVIEW,  # get_project_overview -> overview
            # Search
            ToolNames.CODE_SEARCH,  # semantic_code_search -> code_search
            ToolNames.GREP,  # code_search (keyword) -> grep
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

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get coding-focused system prompt.

        Returns:
            System prompt optimized for software development.
        """
        return """You are Victor, an expert software development assistant.

Your capabilities:
- Deep code understanding through semantic search and LSP integration
- Safe file operations with automatic backup and undo
- Git operations for version control
- Test execution and validation
- Multi-language support (Python, TypeScript, Rust, Go, and more)

Guidelines:
1. **Understand before modifying**: Always read and understand code before making changes
2. **Incremental changes**: Make small, focused changes rather than large rewrites
3. **Verify changes**: Run tests or validation after modifications
4. **Explain reasoning**: Briefly explain your approach when making non-trivial changes
5. **Preserve style**: Match existing code style and patterns
6. **Handle errors gracefully**: If something fails, diagnose and recover

When exploring code:
- Use semantic_code_search for conceptual queries ("authentication logic")
- Use code_search for exact patterns ("def authenticate")
- Use overview to understand file structure

When modifying code:
- Use edit for surgical changes to existing code
- Use write only for new files or complete rewrites
- Always verify changes compile/pass tests when possible

You have access to 45+ tools. Use them efficiently to accomplish tasks."""

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

    @classmethod
    def get_middleware(cls) -> List[MiddlewareProtocol]:
        """Get coding-specific middleware (cached).

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

        return cls._get_cached_extension("middleware", _create_middleware)

    @classmethod
    def get_safety_extension(cls) -> Optional[SafetyExtensionProtocol]:
        """Get coding-specific safety extension (cached).

        Returns:
            Safety extension for git/refactoring patterns
        """
        return cls._get_extension_factory("safety_extension", "victor.coding.safety")

    @classmethod
    def get_prompt_contributor(cls) -> Optional[PromptContributorProtocol]:
        """Get coding-specific prompt contributor (cached).

        Returns:
            Prompt contributor with task type hints
        """
        return cls._get_extension_factory("prompt_contributor", "victor.coding.prompts")

    @classmethod
    def get_mode_config_provider(cls) -> Optional[ModeConfigProviderProtocol]:
        """Get coding-specific mode configuration provider (cached).

        Returns:
            Mode configuration provider
        """
        return cls._get_extension_factory("mode_config_provider", "victor.coding.mode_config")

    @classmethod
    def get_tool_dependency_provider(cls) -> Optional[ToolDependencyProviderProtocol]:
        """Get coding-specific tool dependency provider (cached).

        Returns:
            Tool dependency provider
        """
        return cls._get_extension_factory(
            "tool_dependency_provider", "victor.coding.tool_dependencies"
        )

    @classmethod
    def get_workflow_provider(cls) -> Optional[WorkflowProviderProtocol]:
        """Get coding-specific workflow provider (cached).

        Provides workflows for common coding tasks:
        - feature_implementation: Full feature development
        - quick_feature: Fast feature implementation
        - bug_fix: Systematic bug fixing
        - quick_fix: Fast bug fix
        - code_review: Comprehensive review
        - quick_review: Fast review
        - pr_review: Pull request review

        Returns:
            CodingWorkflowProvider instance
        """
        return cls._get_extension_factory("workflow_provider", "victor.coding.workflows")

    @classmethod
    def get_service_provider(cls) -> Optional[ServiceProviderProtocol]:
        """Get coding-specific service provider (cached).

        Returns:
            Service provider for DI registration
        """
        return cls._get_extension_factory("service_provider", "victor.coding.service_provider")

    @classmethod
    def get_tiered_tools(cls) -> Optional[TieredToolConfig]:
        """Get tiered tool configuration for coding."""
        from victor.core.vertical_types import TieredToolTemplate

        return TieredToolTemplate.for_vertical(cls.name)

    @classmethod
    def get_rl_config_provider(cls) -> Optional[Any]:
        """Get RL configuration provider for Coding vertical (cached).

        Provides configuration for reinforcement learning integration,
        including active learners, task type mappings, and quality thresholds.

        Returns:
            CodingRLConfig instance (implements RLConfigProviderProtocol)
        """
        return cls._get_extension_factory(
            "rl_config_provider", "victor.coding.rl", "CodingRLConfig"
        )

    @classmethod
    def get_rl_hooks(cls) -> Optional[Any]:
        """Get RL hooks for coding vertical (cached).

        Provides hooks for recording RL outcomes and getting recommendations.

        Returns:
            CodingRLHooks instance
        """
        return cls._get_extension_factory("rl_hooks", "victor.coding.rl")

    @classmethod
    def get_team_spec_provider(cls) -> Optional[Any]:
        """Get team specification provider for Coding tasks (cached).

        Provides pre-configured team specifications for:
        - feature_team: Feature implementation
        - bug_fix_team: Bug investigation and fix
        - refactoring_team: Safe refactoring
        - review_team: Comprehensive code review
        - testing_team: Test coverage improvement
        - documentation_team: Documentation generation

        Returns:
            CodingTeamSpecProvider instance (implements TeamSpecProviderProtocol)
        """
        return cls._get_extension_factory("team_spec_provider", "victor.coding.teams")

    @classmethod
    def get_capability_provider(cls) -> Any:
        """Get capability provider for dynamic capability loading (cached).

        Provides CodingCapabilityProvider for runtime configuration
        of coding-specific capabilities like:
        - git_safety: Git operation safety rules
        - code_style: Code formatting preferences
        - test_requirements: Test configuration
        - language_server: LSP settings
        - refactoring: Refactoring capabilities

        Returns:
            CodingCapabilityProvider instance
        """
        return cls._get_extension_factory("capability_provider", "victor.coding.capabilities")

    @classmethod
    def get_composed_chains(cls) -> Dict[str, Any]:
        """Get pre-built LCEL-composed tool chains (cached).

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

        return cls._get_cached_extension("composed_chains", _create)

    @classmethod
    def get_personas(cls) -> Dict[str, Any]:
        """Get persona definitions for team members (cached).

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

        return cls._get_cached_extension("personas", _create)

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
        from victor.coding.handlers import HANDLERS

        return HANDLERS

    # NOTE: get_extensions() is inherited from VerticalBase with full caching support.
    # Individual extension getters use _get_cached_extension() from VerticalBase.
    # To clear all caches, use cls.clear_config_cache().


__all__ = ["CodingAssistant"]
