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

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union


class ToolCategory(str, Enum):
    """Built-in tool categories.

    Categories group related tools together for easy selection.
    This is the canonical source for tool categories - used by both
    tools.py (ToolSet) and tool_config.py (ToolConfigurator).
    """

    CORE = "core"
    """Essential tools: read, write, edit, shell, search."""

    FILESYSTEM = "filesystem"
    """File operations: list[Any], glob, find, mkdir, rm, mv, cp."""

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


# =============================================================================
# Built-in Category Defaults (Loaded from YAML)
# =============================================================================

# Phase 7.5: Tool categories are now loaded from victor/config/tool_categories.yaml
# This provides OCP compliance - new categories can be added without code changes.
# The YAML loader falls back to hardcoded defaults if the file is unavailable.


def _load_builtin_category_tools() -> dict[ToolCategory, set[str]]:
    """Load category-to-tools mapping from YAML configuration.

    Returns:
        Dict mapping ToolCategory enum -> set of tool names
    """
    from victor.config.tool_categories import load_tool_categories, get_fallback_categories

    # Try loading from YAML
    yaml_categories = load_tool_categories()

    if yaml_categories:
        # Convert string keys to ToolCategory enum
        result = {}
        for category_name, tools in yaml_categories.items():
            try:
                category_enum = ToolCategory(category_name)
                result[category_enum] = tools
            except ValueError:
                # Custom category not in enum - skip for _BUILTIN_CATEGORY_TOOLS
                # These are handled by ToolCategoryRegistry
                pass

        # Alias for legacy REFACTOR
        if ToolCategory.REFACTORING in result:
            result[ToolCategory.REFACTOR] = result[ToolCategory.REFACTORING]

        return result

    # Fallback to hardcoded defaults
    fallback = get_fallback_categories()
    result = {}
    for category_name, tools in fallback.items():
        try:
            category_enum = ToolCategory(category_name)
            result[category_enum] = tools
        except ValueError:
            pass

    # Alias for legacy REFACTOR
    if ToolCategory.REFACTORING in result:
        result[ToolCategory.REFACTOR] = result[ToolCategory.REFACTORING]

    return result


# Lazy-loaded category tools (populated on first access)
_BUILTIN_CATEGORY_TOOLS: Optional[dict[ToolCategory, set[str]]] = None


def _get_builtin_category_tools() -> dict[ToolCategory, set[str]]:
    """Get the builtin category tools, loading from YAML if needed."""
    global _BUILTIN_CATEGORY_TOOLS
    if _BUILTIN_CATEGORY_TOOLS is None:
        _BUILTIN_CATEGORY_TOOLS = _load_builtin_category_tools()
    return _BUILTIN_CATEGORY_TOOLS


# =============================================================================
# Tool Category Registry (OCP Compliant)
# =============================================================================


class ToolCategoryRegistry:
    """Registry for tool categories supporting dynamic extension.

    This registry replaces static _CATEGORY_TOOLS mapping with a dynamic
    system that allows plugins and verticals to register new categories
    and extend existing ones without modifying core code (OCP compliance).

    The registry provides a three-tier lookup:
    1. ToolMetadataRegistry (decorator-driven, highest priority)
    2. Custom registrations (from plugins/verticals)
    3. Built-in defaults (fallback)

    Example:
        # Get singleton instance
        registry = ToolCategoryRegistry.get_instance()

        # Register a new vertical-specific category
        registry.register_category("rag", {"rag_search", "rag_query", "rag_ingest"})

        # Extend an existing category
        registry.extend_category("search", {"semantic_rag_search"})

        # Get tools for a category
        tools = registry.get_tools("search")  # Includes extensions
    """

    _instance: Optional["ToolCategoryRegistry"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize the registry with built-in defaults."""
        # Custom category registrations (from plugins/verticals)
        self._custom_categories: dict[str, set[str]] = {}
        # Extensions to existing categories
        self._category_extensions: dict[str, set[str]] = {}
        # Cache for merged results
        self._cache: dict[str, set[str]] = {}
        self._cache_valid = False
        # Thread safety lock (reentrant for nested calls)
        self._lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "ToolCategoryRegistry":
        """Get singleton instance of the registry (thread-safe)."""
        if cls._instance is None:
            with cls._instance_lock:
                # Double-checked locking pattern
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None

    def register_category(
        self,
        name: str,
        tools: set[str],
        description: Optional[str] = None,
    ) -> None:
        """Register a new custom category.

        Use this to add vertical-specific or plugin-specific categories
        that aren't part of the built-in ToolCategory enum.

        Args:
            name: Category name (lowercase recommended)
            tools: Set of tool names in this category
            description: Optional description of the category

        Raises:
            ValueError: If category name conflicts with built-in enum
        """
        # Check for conflict with built-in categories
        name_lower = name.lower()
        try:
            ToolCategory(name_lower)
            # If we get here, it's a built-in category - reject registration
            raise ValueError(
                f"Cannot register '{name}' as custom category - "
                f"conflicts with built-in ToolCategory.{name.upper()}"
            )
        except ValueError as e:
            # Re-raise our custom error (contains "conflicts")
            if "conflicts" in str(e):
                raise
            # Original ValueError from ToolCategory() - name is not a builtin, safe to proceed
            pass

        self._custom_categories[name_lower] = set(tools)
        self._invalidate_cache()

    def extend_category(self, name: str, tools: set[str]) -> None:
        """Extend an existing category with additional tools.

        Use this to add vertical-specific tools to built-in categories
        without modifying core code.

        Args:
            name: Category name (can be built-in or custom)
            tools: Set of tool names to add
        """
        name_lower = name.lower()
        if name_lower not in self._category_extensions:
            self._category_extensions[name_lower] = set()
        self._category_extensions[name_lower].update(tools)
        self._invalidate_cache()

    def get_tools(self, category: str) -> set[str]:
        """Get all tools in a category (merged from all sources, thread-safe).

        Lookup priority:
        1. ToolMetadataRegistry (decorator-driven)
        2. Custom registrations
        3. Built-in defaults
        4. Extensions (added to result)

        Args:
            category: Category name (built-in enum value or custom)

        Returns:
            Set of tool names in this category
        """
        category_lower = category.lower()

        # Check cache first (atomic check-and-get under lock)
        with self._lock:
            if self._cache_valid and category_lower in self._cache:
                return self._cache[category_lower].copy()

            # Build result under lock to ensure atomicity
            result: set[str] = set()

            # 1. Try ToolMetadataRegistry first (decorator-driven)
            try:
                from victor.tools.metadata_registry import ToolMetadataRegistry

                metadata_registry = ToolMetadataRegistry.get_instance()  # type: ignore[attr-defined]
                registry_tools = metadata_registry.get_tools_by_category(category_lower)
                if registry_tools:
                    result.update(registry_tools)
            except (ImportError, Exception):
                pass  # Registry not available, continue with fallbacks

            # 2. Try custom registrations
            if category_lower in self._custom_categories:
                result.update(self._custom_categories[category_lower])

            # 3. Try built-in defaults (if no results yet from above sources)
            if not result:
                try:
                    cat_enum = ToolCategory(category_lower)
                    result.update(_get_builtin_category_tools().get(cat_enum, set()))
                except ValueError:
                    pass  # Unknown category

            # 4. Add extensions
            if category_lower in self._category_extensions:
                result.update(self._category_extensions[category_lower])

            # Cache the result
            self._cache[category_lower] = result.copy()
            self._cache_valid = True
            return result.copy()

    def get_all_categories(self) -> set[str]:
        """Get all available category names.

        Includes built-in categories, custom registrations, and
        categories discovered from ToolMetadataRegistry.

        Returns:
            Set of all category names
        """
        categories: set[str] = set()

        # Built-in categories
        for cat in ToolCategory:
            categories.add(cat.value)

        # Custom categories
        categories.update(self._custom_categories.keys())

        # Categories from metadata registry
        try:
            from victor.tools.metadata_registry import ToolMetadataRegistry

            metadata_registry = ToolMetadataRegistry.get_instance()  # type: ignore[attr-defined]
            categories.update(metadata_registry.get_all_categories())
        except (ImportError, Exception):
            pass

        return categories

    def is_builtin_category(self, name: str) -> bool:
        """Check if a category is a built-in category.

        Args:
            name: Category name to check

        Returns:
            True if this is a built-in ToolCategory enum value
        """
        try:
            ToolCategory(name.lower())
            return True
        except ValueError:
            return False

    def _invalidate_cache(self) -> None:
        """Invalidate the category cache (thread-safe)."""
        with self._lock:
            self._cache.clear()
            self._cache_valid = False


def get_category_registry() -> ToolCategoryRegistry:
    """Get the global ToolCategoryRegistry instance.

    Convenience function for accessing the singleton.

    Returns:
        The ToolCategoryRegistry singleton
    """
    return ToolCategoryRegistry.get_instance()


# Legacy alias for backward compatibility
# DEPRECATED: Use get_category_registry().get_tools(category) instead
def _get_category_tools(category: ToolCategory) -> set[str]:
    """Get tools for a category (backward compatible).

    DEPRECATED: Use get_category_registry().get_tools() instead.
    """
    return get_category_registry().get_tools(category.value)


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

    tools: set[str] = field(default_factory=set)
    categories: set[str] = field(default_factory=set)
    exclude: set[str] = field(default_factory=set)
    _resolved_names_cache: Optional[set[str]] = field(default=None, repr=False, compare=False)

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
    def from_categories(cls, categories: list[str]) -> "ToolSet":
        """Create ToolSet from category names.

        Args:
            categories: List of category names (e.g., ["core", "git"])

        Returns:
            ToolSet with specified categories
        """
        return cls(categories=set(categories))

    @classmethod
    def from_tools(cls, tools: list[str]) -> "ToolSet":
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

    def get_tool_names(self) -> set[str]:
        """Get all tool names in this set.

        Resolves categories to their tool names and applies exclusions.
        Uses ToolCategoryRegistry for OCP-compliant category resolution.

        Returns:
            Set of tool names
        """
        all_tools = set(self.tools)

        # Add tools from categories using registry (OCP compliant)
        registry = get_category_registry()
        for category in self.categories:
            all_tools.update(registry.get_tools(category))

        # Apply exclusions
        return all_tools - self.exclude

    def _get_resolved_names(self) -> set[str]:
        """Get resolved tool names with caching.

        This method caches the result of get_tool_names() for O(1)
        membership checks after the first call.

        Returns:
            Cached set of resolved tool names
        """
        if self._resolved_names_cache is None:
            # Use object.__setattr__ to bypass frozen dataclass if needed
            resolved = self.get_tool_names()
            object.__setattr__(self, "_resolved_names_cache", resolved)
        return self._resolved_names_cache if self._resolved_names_cache is not None else set()

    def invalidate_cache(self) -> None:
        """Invalidate the resolved names cache.

        Call this after modifying the ToolSet to ensure the cache
        is rebuilt on next access.
        """
        object.__setattr__(self, "_resolved_names_cache", None)

    def __contains__(self, tool: str) -> bool:
        """Check if a tool is in this set (cached O(1) after first call)."""
        return tool in self._get_resolved_names()

# Type for tools parameter that accepts multiple formats
ToolsInput = Union[ToolSet, list[str], None]
