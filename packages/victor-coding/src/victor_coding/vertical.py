"""
CodingVertical - Main entry point for the coding vertical.

This vertical provides code intelligence capabilities including:
- Tree-sitter based code analysis
- LSP integration
- 25 coding-specific tools (code search, refactoring, review, etc.)

Usage:
    from victor_coding import CodingVertical

    # Register with victor-core
    vertical = CodingVertical()
    extensions = vertical.get_extensions()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

# Try to import VerticalBase for proper inheritance
# Falls back to standalone class if victor-core not installed
try:
    from victor.verticals.base import VerticalBase

    _HAS_VERTICAL_BASE = True
except ImportError:
    _HAS_VERTICAL_BASE = False
    VerticalBase = object  # type: ignore

if TYPE_CHECKING:
    from victor.verticals.protocols import VerticalExtensions, ModeConfig
    from victor.tools.base import BaseTool


class CodingVertical(VerticalBase):  # type: ignore
    """Coding assistant vertical with full IDE capabilities.

    Provides code intelligence through Tree-sitter parsing, LSP integration,
    and 25 specialized coding tools.

    This class will be extended to inherit from VerticalBase during
    the Phase 3 migration when the actual coding modules are moved.
    """

    name = "coding"
    description = "AI-powered coding assistant with code intelligence"

    def __init__(self) -> None:
        """Initialize the coding vertical."""
        self._tools: Optional[List[Type]] = None
        logger.info("Initialized CodingVertical")

    @classmethod
    def get_name(cls) -> str:
        """Get the vertical name."""
        return cls.name

    @classmethod
    def get_description(cls) -> str:
        """Get the vertical description."""
        return cls.description

    def get_tools(self) -> List[str]:
        """Get list of coding-specific tool names.

        Returns:
            List of 25 coding tool names
        """
        return [
            "code_search",
            "semantic_code_search",
            "code_review",
            "refactor",
            "test_generator",
            "symbol_lookup",
            "ast_query",
            "dependency_analyzer",
            "complexity_analyzer",
            "dead_code_finder",
            "code_explain",
            "documentation_generator",
            "import_organizer",
            "type_inference",
            "security_scanner",
            "performance_analyzer",
            "code_formatter",
            "linter",
            "project_analyzer",
            "git_diff_analyzer",
            "commit_message_generator",
            "pr_reviewer",
            "code_completion",
            "snippet_search",
            "api_endpoint_finder",
        ]

    def get_system_prompt(self) -> str:
        """Get the coding-specific system prompt.

        Returns:
            System prompt string for coding context
        """
        return """You are an expert coding assistant with deep knowledge of:
- Software architecture and design patterns
- Code analysis and refactoring
- Testing strategies and test generation
- Security best practices
- Performance optimization

You have access to specialized tools for code analysis, search, and modification.
Use these tools effectively to understand and improve code quality.
"""

    def get_mode_configs(self) -> Dict[str, dict]:
        """Get coding-specific mode configurations.

        Returns:
            Dictionary of mode configurations
        """
        return {
            "code": {
                "name": "code",
                "description": "Standard coding mode",
                "tools": self.get_tools(),
            },
            "review": {
                "name": "review",
                "description": "Code review mode",
                "tools": [
                    "code_search",
                    "code_review",
                    "security_scanner",
                    "complexity_analyzer",
                ],
            },
            "refactor": {
                "name": "refactor",
                "description": "Refactoring mode",
                "tools": [
                    "refactor",
                    "code_search",
                    "ast_query",
                    "dependency_analyzer",
                ],
            },
        }


# Convenience alias
__all__ = ["CodingVertical"]
