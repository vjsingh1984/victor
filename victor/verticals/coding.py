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

This vertical captures the current Victor behavior as a template,
making it explicit and reusable.

Features:
- Full access to 45+ tools optimized for coding
- Stage-aware tool selection
- Code-focused system prompt
- Evaluation criteria for code quality
"""

from __future__ import annotations

from typing import Any, Dict, List

from victor.verticals.base import StageDefinition, VerticalBase, VerticalConfig


class CodingAssistant(VerticalBase):
    """Software development assistant vertical.

    This is Victor's default configuration, optimized for:
    - Code exploration and understanding
    - Bug fixing and refactoring
    - Feature implementation
    - Testing and verification
    - Git operations and version control

    Example:
        from victor.verticals import CodingAssistant

        config = CodingAssistant.get_config()
        agent = await Agent.create(tools=config.tools)
    """

    name = "coding"
    description = "Software development assistant for code exploration, writing, and refactoring"
    version = "1.0.0"

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
