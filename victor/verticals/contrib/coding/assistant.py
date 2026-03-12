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

from typing import Any, Dict, List

from victor_sdk import (
    CapabilityIds,
    CapabilityRequirement,
    StageDefinition,
    ToolNames,
    VerticalBase,
)


class CodingAssistant(VerticalBase):
    """Software development assistant vertical.

    This is Victor's default configuration, optimized for:
    - Code exploration and understanding
    - Bug fixing and refactoring
    - Feature implementation
    - Testing and verification
    - Git operations and version control

    The definition layer is SDK-only. Runtime middleware, prompt contributors,
    service providers, and other integrations are attached by the package root
    runtime wrapper and shared host-side loaders.

    Example:
        from victor.verticals.contrib.coding.assistant import CodingAssistant

        definition = CodingAssistant.get_definition()
        prompt = definition.system_prompt
    """

    name = "coding"
    description = "Software development assistant for code exploration, writing, and refactoring"
    version = "2.0.0"  # Extension support

    @classmethod
    def get_name(cls) -> str:
        """Return the stable identifier for this vertical."""

        return cls.name

    @classmethod
    def get_description(cls) -> str:
        """Return the human-readable vertical description."""

        return cls.description

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get tools optimized for software development.

        Uses SDK-owned canonical tool identifiers, including the shared file-operation
        tool group, so the definition layer does not need framework capability objects.

        Returns:
            List of tool names including filesystem, git, shell, and code tools.
        """
        tools = list(ToolNames.file_operations())

        # Add coding-specific tools
        tools.extend(
            [
                # Core filesystem (beyond framework basics)
                ToolNames.LS,  # list_directory -> ls
                ToolNames.OVERVIEW,  # get_project_overview -> overview
                # Search
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
        )

        return tools

    @classmethod
    def get_capability_requirements(cls) -> List[CapabilityRequirement]:
        """Declare runtime capabilities required by the coding definition layer.

        Returns:
            Structured SDK capability requirements used by the runtime adapter.
        """
        return [
            CapabilityRequirement(
                capability_id=CapabilityIds.FILE_OPS,
                purpose="Read, write, and edit repository files.",
            ),
            CapabilityRequirement(
                capability_id=CapabilityIds.GIT,
                optional=True,
                purpose="Inspect and update repository state when git tooling is available.",
            ),
            CapabilityRequirement(
                capability_id=CapabilityIds.LSP,
                optional=True,
                purpose="Enable symbol, reference, and language-intelligence workflows.",
            ),
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

        Uses SDK-owned canonical tool identifiers.

        Returns:
            Stage definitions optimized for software development workflow.
        """
        return {
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Understanding the coding request",
                optional_tools=[
                    ToolNames.READ,
                    ToolNames.LS,
                    ToolNames.OVERVIEW,
                    ToolNames.GREP,
                ],
                keywords=["what", "how", "explain", "where", "show me"],
                next_stages={"PLANNING", "READING"},
            ),
            "PLANNING": StageDefinition(
                name="PLANNING",
                description="Planning the implementation approach",
                optional_tools=[
                    ToolNames.GREP,
                    ToolNames.PLAN,
                    ToolNames.OVERVIEW,
                    ToolNames.READ,
                ],
                keywords=["plan", "approach", "design", "architecture", "strategy"],
                next_stages={"READING", "EXECUTION"},
            ),
            "READING": StageDefinition(
                name="READING",
                description="Reading code and gathering context",
                optional_tools=[
                    ToolNames.READ,
                    ToolNames.CODE_SEARCH,
                    ToolNames.GREP,
                    ToolNames.LSP,
                    ToolNames.SYMBOL,
                    ToolNames.REFS,
                ],
                keywords=["read", "show", "find", "look", "check", "search"],
                next_stages={"ANALYSIS", "EXECUTION"},
            ),
            "ANALYSIS": StageDefinition(
                name="ANALYSIS",
                description="Analyzing code structure and dependencies",
                optional_tools=[
                    ToolNames.LSP,
                    ToolNames.SYMBOL,
                    ToolNames.REFS,
                    ToolNames.OVERVIEW,
                ],
                keywords=["analyze", "review", "understand", "why", "how does"],
                next_stages={"EXECUTION", "PLANNING"},
            ),
            "EXECUTION": StageDefinition(
                name="EXECUTION",
                description="Implementing changes",
                optional_tools=[
                    ToolNames.WRITE,
                    ToolNames.EDIT,
                    ToolNames.SHELL,
                    ToolNames.GIT,
                    ToolNames.RENAME,
                ],
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
                optional_tools=[
                    ToolNames.SHELL,
                    ToolNames.TEST,
                    ToolNames.GIT,
                    ToolNames.READ,
                ],
                keywords=["test", "verify", "check", "validate", "run", "build"],
                next_stages={"COMPLETION", "EXECUTION"},
            ),
            "COMPLETION": StageDefinition(
                name="COMPLETION",
                description="Committing and summarizing",
                optional_tools=[ToolNames.GIT],
                keywords=["done", "finish", "complete", "commit", "summarize"],
                next_stages=set(),
            ),
        }

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Return coding-specific serializable metadata for the definition layer."""

        return {
            "supports_lsp": True,
            "supports_git": True,
            "max_file_size": 1_000_000,
            "supported_languages": [
                "python",
                "typescript",
                "javascript",
                "rust",
                "go",
                "java",
                "c",
                "cpp",
            ],
        }


__all__ = ["CodingAssistant"]
