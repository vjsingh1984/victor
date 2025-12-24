"""Tools - Available capabilities for agents.

This module provides ToolSet for configuring which tools are
available to an agent. Most users can use the preset methods
(default, minimal, full, airgapped) without understanding
individual tools.

Example:
    # Default tools (most common)
    agent = await Agent.create(tools=ToolSet.default())

    # Minimal for simple tasks
    agent = await Agent.create(tools=ToolSet.minimal())

    # Full access
    agent = await Agent.create(tools=ToolSet.full())

    # Custom selection
    agent = await Agent.create(tools=["filesystem", "git"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set, Union


class ToolCategory(str, Enum):
    """Built-in tool categories.

    Categories group related tools together for easy selection.
    This is the canonical source for tool categories - used by both
    tools.py (ToolSet) and tool_config.py (ToolConfigurator).
    """

    CORE = "core"
    """Essential tools: read, write, edit, shell, search."""

    FILESYSTEM = "filesystem"
    """File operations: list, glob, find, mkdir, rm, mv, cp."""

    GIT = "git"
    """Version control: status, diff, commit, branch, log."""

    SEARCH = "search"
    """Search tools: grep, glob, code_search, semantic_code_search."""

    WEB = "web"
    """Network tools: web_search, web_fetch, http_request."""

    DATABASE = "database"
    """Database tools: query, schema inspection."""

    DOCKER = "docker"
    """Container operations: run, build, compose."""

    TESTING = "testing"
    """Test execution: pytest, unittest, coverage."""

    REFACTORING = "refactoring"
    """Code refactoring: rename, extract, inline."""

    # Legacy alias for backward compatibility
    REFACTOR = "refactoring"
    """Alias for REFACTORING (deprecated, use REFACTORING)."""

    DOCUMENTATION = "documentation"
    """Documentation: docstring generation, README updates."""

    ANALYSIS = "analysis"
    """Code analysis: complexity, metrics, static analysis."""

    COMMUNICATION = "communication"
    """Communication tools: slack, teams, jira."""

    CUSTOM = "custom"
    """Custom/user-defined tools."""


# Mapping of categories to tool names (canonical source)
_CATEGORY_TOOLS = {
    ToolCategory.CORE: {"read", "write", "edit", "shell", "search", "code_search", "ls"},
    ToolCategory.FILESYSTEM: {
        "read",
        "write",
        "edit",
        "ls",
        "list_directory",
        "glob",
        "find_files",
        "file_info",
        "mkdir",
        "rm",
        "mv",
        "cp",
    },
    ToolCategory.GIT: {"git", "git_status", "git_diff", "git_commit", "git_branch", "git_log"},
    ToolCategory.SEARCH: {"grep", "glob", "code_search", "semantic_code_search", "search"},
    ToolCategory.WEB: {"web_search", "web_fetch", "http_request", "fetch_url"},
    ToolCategory.DATABASE: {"sql_query", "db_schema", "database"},
    ToolCategory.DOCKER: {"docker", "docker_run", "docker_build", "docker_compose"},
    ToolCategory.TESTING: {"run_tests", "pytest", "test_runner", "test"},
    ToolCategory.REFACTORING: {"refactor", "rename_symbol", "extract_function", "rename"},
    ToolCategory.DOCUMENTATION: {"generate_docs", "update_readme", "documentation", "docstring"},
    ToolCategory.ANALYSIS: {"analyze", "complexity"},
    ToolCategory.COMMUNICATION: {"slack", "teams", "jira"},
    ToolCategory.CUSTOM: set(),  # User-defined tools
}

# Alias for legacy REFACTOR
_CATEGORY_TOOLS[ToolCategory.REFACTOR] = _CATEGORY_TOOLS[ToolCategory.REFACTORING]


@dataclass
class ToolSet:
    """Configuration for which tools are available to an agent.

    ToolSets define what capabilities the agent has access to.
    Use the class methods for common configurations, or create
    custom sets by specifying tools and categories.

    Attributes:
        tools: Specific tool names to include
        categories: Tool categories to include
        exclude: Tool names to exclude

    Example:
        # Use presets for common cases
        tools = ToolSet.default()     # Core + filesystem + git
        tools = ToolSet.minimal()     # Just core tools
        tools = ToolSet.full()        # Everything
        tools = ToolSet.airgapped()   # No network tools

        # Custom selection
        tools = ToolSet.from_categories(["filesystem", "git"])
        tools = ToolSet.from_tools(["read", "write", "git"])

        # Modify existing set
        tools = ToolSet.default().include("docker").exclude_tools("shell")
    """

    tools: Set[str] = field(default_factory=set)
    categories: Set[str] = field(default_factory=set)
    exclude: Set[str] = field(default_factory=set)

    @classmethod
    def default(cls) -> "ToolSet":
        """Default tool set - core + filesystem + git.

        This is the recommended starting point for most use cases.
        Includes file operations, shell access, and git integration.

        Returns:
            ToolSet with sensible defaults
        """
        return cls(
            categories={
                ToolCategory.CORE.value,
                ToolCategory.FILESYSTEM.value,
                ToolCategory.GIT.value,
            }
        )

    @classmethod
    def minimal(cls) -> "ToolSet":
        """Minimal tool set - only core operations.

        Use this for simple tasks that don't need git or
        advanced file operations.

        Includes: read, write, edit, shell, search

        Returns:
            ToolSet with minimal tools
        """
        return cls(categories={ToolCategory.CORE.value})

    @classmethod
    def full(cls) -> "ToolSet":
        """Full tool set - all available tools.

        Use with caution - gives the agent access to everything.

        Returns:
            ToolSet with all categories
        """
        return cls(categories={c.value for c in ToolCategory})

    @classmethod
    def airgapped(cls) -> "ToolSet":
        """Tool set for air-gapped environments (no network).

        Includes core, filesystem, and git but excludes all
        network-dependent tools.

        Returns:
            ToolSet without network tools
        """
        return cls(
            categories={
                ToolCategory.CORE.value,
                ToolCategory.FILESYSTEM.value,
                ToolCategory.GIT.value,
            },
            exclude={"web_search", "web_fetch", "http_request", "fetch_url"},
        )

    @classmethod
    def coding(cls) -> "ToolSet":
        """Tool set optimized for coding tasks.

        Includes core, filesystem, git, testing, and refactoring.

        Returns:
            ToolSet for coding workflows
        """
        return cls(
            categories={
                ToolCategory.CORE.value,
                ToolCategory.FILESYSTEM.value,
                ToolCategory.GIT.value,
                ToolCategory.TESTING.value,
                ToolCategory.REFACTORING.value,
            }
        )

    @classmethod
    def research(cls) -> "ToolSet":
        """Tool set optimized for research tasks.

        Includes core, web, and filesystem tools.

        Returns:
            ToolSet for research workflows
        """
        return cls(
            categories={
                ToolCategory.CORE.value,
                ToolCategory.FILESYSTEM.value,
                ToolCategory.WEB.value,
            }
        )

    @classmethod
    def from_categories(cls, categories: List[str]) -> "ToolSet":
        """Create ToolSet from category names.

        Args:
            categories: List of category names (e.g., ["core", "git"])

        Returns:
            ToolSet with specified categories
        """
        return cls(categories=set(categories))

    @classmethod
    def from_tools(cls, tools: List[str]) -> "ToolSet":
        """Create ToolSet from specific tool names.

        Args:
            tools: List of tool names (e.g., ["read", "write", "git"])

        Returns:
            ToolSet with specified tools
        """
        return cls(tools=set(tools))

    def include(self, *tools: str) -> "ToolSet":
        """Add tools to the set.

        Args:
            *tools: Tool names to add

        Returns:
            New ToolSet with added tools
        """
        return ToolSet(
            tools=self.tools | set(tools),
            categories=self.categories,
            exclude=self.exclude - set(tools),
        )

    def include_category(self, category: str) -> "ToolSet":
        """Add a category to the set.

        Args:
            category: Category name to add

        Returns:
            New ToolSet with added category
        """
        return ToolSet(
            tools=self.tools,
            categories=self.categories | {category},
            exclude=self.exclude,
        )

    def exclude_tools(self, *tools: str) -> "ToolSet":
        """Exclude tools from the set.

        Args:
            *tools: Tool names to exclude

        Returns:
            New ToolSet with exclusions
        """
        return ToolSet(
            tools=self.tools - set(tools),
            categories=self.categories,
            exclude=self.exclude | set(tools),
        )

    def get_tool_names(self) -> Set[str]:
        """Get all tool names in this set.

        Resolves categories to their tool names and applies exclusions.

        Returns:
            Set of tool names
        """
        all_tools = set(self.tools)

        # Add tools from categories
        for category in self.categories:
            try:
                cat_enum = ToolCategory(category)
                all_tools.update(_CATEGORY_TOOLS.get(cat_enum, set()))
            except ValueError:
                # Unknown category, skip
                pass

        # Apply exclusions
        return all_tools - self.exclude

    def __contains__(self, tool: str) -> bool:
        """Check if a tool is in this set."""
        return tool in self.get_tool_names()


# Type alias for convenience
Tools = ToolSet

# Type for tools parameter that accepts multiple formats
ToolsInput = Union[ToolSet, List[str], None]
