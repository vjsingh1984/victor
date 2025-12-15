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

from victor.verticals.base import StageDefinition, VerticalBase, VerticalConfig
from victor.verticals.protocols import (
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
        from victor.verticals.coding import CodingAssistant

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
    version = "2.0.0"  # Bumped for extension support

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get tools optimized for software development.

        Returns:
            List of tool names including filesystem, git, shell, and code tools.
        """
        return [
            # Core filesystem
            "read",
            "write",
            "edit",
            "ls",
            "overview",
            # Search
            "search",
            "code_search",
            "semantic_code_search",
            "plan_files",
            # Git
            "git",
            "git_status",
            "git_diff",
            "git_log",
            "git_commit",
            "git_branch",
            # Shell
            "shell",
            "bash",
            # Code intelligence
            "lsp",
            "symbols",
            "references",
            "hover",
            # Refactoring
            "refactor",
            "rename_symbol",
            "extract_function",
            # Testing
            "test",
            "run_tests",
            "test_file",
            # Docker
            "docker",
            "docker_compose",
            # Web (for documentation)
            "web_search",
            "web_fetch",
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

        Returns:
            Stage definitions optimized for software development workflow.
        """
        return {
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Understanding the coding request",
                tools={"read", "ls", "overview", "search"},
                keywords=["what", "how", "explain", "where", "show me"],
                next_stages={"PLANNING", "READING"},
            ),
            "PLANNING": StageDefinition(
                name="PLANNING",
                description="Planning the implementation approach",
                tools={"search", "plan_files", "overview", "read"},
                keywords=["plan", "approach", "design", "architecture", "strategy"],
                next_stages={"READING", "EXECUTION"},
            ),
            "READING": StageDefinition(
                name="READING",
                description="Reading code and gathering context",
                tools={
                    "read",
                    "code_search",
                    "semantic_code_search",
                    "lsp",
                    "symbols",
                    "references",
                },
                keywords=["read", "show", "find", "look", "check", "search"],
                next_stages={"ANALYSIS", "EXECUTION"},
            ),
            "ANALYSIS": StageDefinition(
                name="ANALYSIS",
                description="Analyzing code structure and dependencies",
                tools={"lsp", "symbols", "references", "hover", "overview"},
                keywords=["analyze", "review", "understand", "why", "how does"],
                next_stages={"EXECUTION", "PLANNING"},
            ),
            "EXECUTION": StageDefinition(
                name="EXECUTION",
                description="Implementing changes",
                tools={
                    "write",
                    "edit",
                    "shell",
                    "git",
                    "refactor",
                    "rename_symbol",
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
                tools={"shell", "test", "run_tests", "git_diff", "read"},
                keywords=["test", "verify", "check", "validate", "run", "build"],
                next_stages={"COMPLETION", "EXECUTION"},
            ),
            "COMPLETION": StageDefinition(
                name="COMPLETION",
                description="Committing and summarizing",
                tools={"git_commit", "git_status", "git_diff"},
                keywords=["done", "finish", "complete", "commit", "summarize"],
                next_stages=set(),
            ),
        }

    @classmethod
    def get_provider_hints(cls) -> Dict[str, Any]:
        """Get provider hints for coding tasks.

        Returns:
            Provider preferences for code generation.
        """
        return {
            "preferred_providers": ["anthropic", "openai"],
            "preferred_models": [
                "claude-sonnet-4-20250514",
                "gpt-4-turbo",
                "claude-3-5-sonnet-20241022",
            ],
            "min_context_window": 100000,
            "requires_tool_calling": True,
            "prefers_extended_thinking": True,
        }

    @classmethod
    def get_evaluation_criteria(cls) -> List[str]:
        """Get evaluation criteria for coding tasks.

        Returns:
            Criteria for evaluating code quality.
        """
        return [
            "Code correctness and functionality",
            "Adherence to existing code style",
            "Test coverage for changes",
            "Minimal unnecessary changes",
            "Clear commit messages",
            "Error handling",
            "Performance considerations",
        ]

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
        """Get coding-specific middleware.

        Returns:
            List of middleware implementations
        """
        from victor.verticals.coding.middleware import (
            CodeCorrectionMiddleware,
            GitSafetyMiddleware,
        )

        return [
            CodeCorrectionMiddleware(enabled=True, auto_fix=True),
            GitSafetyMiddleware(block_dangerous=False, warn_on_risky=True),
        ]

    @classmethod
    def get_safety_extension(cls) -> Optional[SafetyExtensionProtocol]:
        """Get coding-specific safety extension.

        Returns:
            Safety extension for git/refactoring patterns
        """
        from victor.verticals.coding.safety import CodingSafetyExtension

        return CodingSafetyExtension()

    @classmethod
    def get_prompt_contributor(cls) -> Optional[PromptContributorProtocol]:
        """Get coding-specific prompt contributor.

        Returns:
            Prompt contributor with task type hints
        """
        from victor.verticals.coding.prompts import CodingPromptContributor

        return CodingPromptContributor()

    @classmethod
    def get_mode_config_provider(cls) -> Optional[ModeConfigProviderProtocol]:
        """Get coding-specific mode configuration provider.

        Returns:
            Mode configuration provider
        """
        from victor.verticals.coding.mode_config import CodingModeConfigProvider

        return CodingModeConfigProvider()

    @classmethod
    def get_tool_dependency_provider(cls) -> Optional[ToolDependencyProviderProtocol]:
        """Get coding-specific tool dependency provider.

        Returns:
            Tool dependency provider
        """
        from victor.verticals.coding.tool_dependencies import CodingToolDependencyProvider

        return CodingToolDependencyProvider()

    @classmethod
    def get_workflow_provider(cls) -> Optional[WorkflowProviderProtocol]:
        """Get coding-specific workflow provider.

        Returns:
            Workflow provider (or None if not implemented)
        """
        # TODO: Implement workflow provider
        return None

    @classmethod
    def get_service_provider(cls) -> Optional[ServiceProviderProtocol]:
        """Get coding-specific service provider.

        Returns:
            Service provider for DI registration
        """
        from victor.verticals.coding.service_provider import CodingServiceProvider

        return CodingServiceProvider()

    @classmethod
    def get_tiered_tools(cls) -> Optional[TieredToolConfig]:
        """Get tiered tool configuration for coding.

        Simplified configuration using consolidated tool metadata:
        - Mandatory: Core tools always included for any task
        - Vertical Core: Essential tools for coding tasks
        - semantic_pool: Derived from ToolMetadataRegistry.get_all_tool_names()
        - stage_tools: Derived from @tool(stages=[...]) decorator metadata

        Returns:
            TieredToolConfig for coding vertical
        """
        from victor.tools.tool_names import ToolNames

        return TieredToolConfig(
            # Tier 1: Mandatory - always included for any task
            mandatory={
                ToolNames.READ,      # Read files - essential
                ToolNames.LS,        # List directory - essential
                ToolNames.GREP,      # Code search - essential for finding code
            },
            # Tier 2: Vertical Core - essential for coding tasks
            vertical_core={
                ToolNames.EDIT,      # Edit files - core coding
                ToolNames.WRITE,     # Write files - core coding
                ToolNames.SHELL,     # Shell commands - core for build/test
                ToolNames.GIT,       # Git operations - core for version control
                ToolNames.SEARCH,    # Semantic search - core for code exploration
                ToolNames.OVERVIEW,  # Codebase overview - core for understanding
            },
            # semantic_pool and stage_tools are now derived from @tool decorator metadata
            # Use get_effective_semantic_pool() and get_tools_for_stage_from_registry()
            # For analysis queries, don't hide write tools - coding often needs them
            readonly_only_for_analysis=False,
        )

    @classmethod
    def get_extensions(cls) -> VerticalExtensions:
        """Get all coding vertical extensions.

        Convenience method that aggregates all extension implementations
        for framework integration.

        Returns:
            VerticalExtensions containing all coding extensions
        """
        safety = cls.get_safety_extension()
        prompt = cls.get_prompt_contributor()

        return VerticalExtensions(
            middleware=cls.get_middleware(),
            safety_extensions=[safety] if safety else [],
            prompt_contributors=[prompt] if prompt else [],
            mode_config_provider=cls.get_mode_config_provider(),
            tool_dependency_provider=cls.get_tool_dependency_provider(),
            workflow_provider=cls.get_workflow_provider(),
            service_provider=cls.get_service_provider(),
        )


__all__ = ["CodingAssistant"]
