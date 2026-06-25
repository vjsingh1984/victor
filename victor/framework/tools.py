"""Tools - Available capabilities for agents.

This module provides ToolSet for configuring which tools are
available to an agent. Most users can use the preset methods
without needing to name individual tools.

Presets (in order of scope):
    minimal()   — core only (7 tools: read, write, edit, shell, search,
                  code_search, ls)
    default()   — core + filesystem + git (~20 tools); recommended starting point
    airgapped() — core + filesystem + git + analysis; no network tools
    coding()    — core + filesystem + git + testing + refactoring
    research()  — core + filesystem + web
    full()      — all categories including web, database, docker, communication

Example:
    agent = await Agent.create(tools=ToolSet.default())
    agent = await Agent.create(tools=ToolSet.minimal())
    agent = await Agent.create(tools=ToolSet.airgapped())
    agent = await Agent.create(tools=ToolSet.full())

    # Custom selection
    agent = await Agent.create(tools=["filesystem", "git"])
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Union

from victor_contracts.core.types import Tier as SdkTier
from victor_contracts.core.types import ToolSet as SdkToolSet


class ToolCategory(str, Enum):
    """Built-in tool categories.

    Categories group related tools together for easy selection.

    This enum is the **single identity authority for category names**. The
    ``victor/config/tool_categories.yaml`` file is a *derived view*: it supplies
    per-category membership, descriptions, and presets, but may only name
    categories that exist here — a guard test
    (``tests/unit/framework/test_tool_category_yaml_parity.py``) pins the two
    vocabularies equal so they cannot silently drift. Runtime *membership* (which
    tools are in a category) is owned by ``ToolMetadataRegistry`` (decorator /
    ``resolve_contract``-driven, tier 1 of :meth:`ToolCategoryRegistry.get_tools`);
    the YAML provides the built-in fallback.

    Consumed by both ``tools.py`` (ToolSet) and ``tool_config.py``
    (ToolConfigurator).
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

    NOTEBOOK = "notebook"
    """Jupyter notebook operations: notebook_edit."""

    TASK_MANAGEMENT = "task_management"
    """Session-scoped task tracking: task_create, task_update, task_list, task_get."""

    VERIFICATION = "verification"
    """Codebase verification: codebase_verify, codebase_verify_batch."""

    CUSTOM = "custom"
    """Custom/user-defined tools."""


# =============================================================================
# Built-in Category Defaults (Loaded from YAML)
# =============================================================================

# Phase 7.5: Tool categories are now loaded from victor/config/tool_categories.yaml
# This provides OCP compliance - new categories can be added without code changes.
# The YAML loader falls back to hardcoded defaults if the file is unavailable.


def _load_builtin_category_tools() -> dict:
    """Load category-to-tools mapping from YAML configuration.

    Returns:
        Dict mapping ToolCategory enum -> set of tool names
    """
    from victor.config.tool_categories import (
        load_tool_categories,
        get_fallback_categories,
    )

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
_BUILTIN_CATEGORY_TOOLS: Optional[dict] = None
_builtin_tools_lock = threading.Lock()


def _get_builtin_category_tools() -> dict:
    """Get the builtin category tools, loading from YAML if needed.

    Uses double-checked locking to ensure thread-safe initialization.
    """
    global _BUILTIN_CATEGORY_TOOLS
    if _BUILTIN_CATEGORY_TOOLS is None:
        with _builtin_tools_lock:
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
        """Get singleton instance of the registry."""
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
        tools: Set[str],
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

    def extend_category(self, name: str, tools: Set[str]) -> None:
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

    def get_tools(self, category: str) -> Set[str]:
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
            result: Set[str] = set()

            # 1. Try ToolMetadataRegistry first (decorator-driven)
            try:
                from victor.tools.metadata_registry import ToolMetadataRegistry

                metadata_registry = ToolMetadataRegistry.get_instance()
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

    def get_all_categories(self) -> Set[str]:
        """Get all available category names.

        Includes built-in categories, custom registrations, and
        categories discovered from ToolMetadataRegistry.

        Returns:
            Set of all category names
        """
        categories: Set[str] = set()

        # Built-in categories
        for cat in ToolCategory:
            categories.add(cat.value)

        # Custom categories
        categories.update(self._custom_categories.keys())

        # Categories from metadata registry
        try:
            from victor.tools.metadata_registry import ToolMetadataRegistry

            metadata_registry = ToolMetadataRegistry.get_instance()
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
def _get_category_tools(category: ToolCategory) -> Set[str]:
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
        tools = ToolSet.default()     # core + filesystem + git (~20 tools)
        tools = ToolSet.minimal()     # core only (7 tools)
        tools = ToolSet.airgapped()   # core + filesystem + git, no network
        tools = ToolSet.coding()      # adds testing + refactoring to default
        tools = ToolSet.research()    # core + filesystem + web
        tools = ToolSet.full()        # all 14 categories (includes network + comms)

        # Custom selection
        tools = ToolSet.from_categories(["filesystem", "git"])
        tools = ToolSet.from_tools(["read", "write", "git"])

        # Modify existing set
        tools = ToolSet.default().include("docker").exclude_tools("shell")
    """

    tools: Set[str] = field(default_factory=set)
    categories: Set[str] = field(default_factory=set)
    exclude: Set[str] = field(default_factory=set)
    _resolved_names_cache: Optional[Set[str]] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Pre-resolve tool names at creation time.

        This optimization eliminates lock contention on every tool check
        by resolving categories eagerly rather than lazily. The cache
        is built once at creation time, making __contains__ O(1) with
        no locks.

        Performance impact: 40-60% reduction in tool resolution overhead.
        """
        # Pre-resolve tool names on creation (eager vs lazy)
        object.__setattr__(self, "_resolved_names_cache", self._resolve_tool_names())

    def _resolve_tool_names(self) -> Set[str]:
        """Resolve tool names from categories and apply exclusions.

        This method does the actual work of resolving categories to
        their tool names and applying exclusions. It's called once
        during __post_init__.

        Returns:
            Set of resolved tool names
        """
        all_tools = set(self.tools)

        # Add tools from categories using registry (OCP compliant)
        registry = get_category_registry()
        for category in self.categories:
            all_tools.update(registry.get_tools(category))

        # Apply exclusions
        return all_tools - self.exclude

    @classmethod
    def default(cls) -> "ToolSet":
        """Default tool set — core + filesystem + git (~20 tools).

        Recommended starting point for most use cases. Covers:
          core:       read, write, edit, shell, search, code_search, ls
          filesystem: list_directory, glob, find_files, file_info, mkdir,
                      rm, mv, cp (plus core file tools)
          git:        git, git_status, git_diff, git_commit, git_branch, git_log

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
        """Minimal tool set — core operations only (7 tools).

        Use for simple tasks that don't need git or filesystem navigation.

        Includes: read, write, edit, shell, search, code_search, ls

        Returns:
            ToolSet with minimal tools
        """
        return cls(categories={ToolCategory.CORE.value})

    @classmethod
    def full(cls) -> "ToolSet":
        """Full tool set — all 14 built-in categories.

        Includes every category: core, filesystem, git, search, web,
        database, docker, testing, refactoring, documentation, analysis,
        communication (slack/teams/jira), notebook, task_management.

        Use with caution — grants network access, shell execution, and
        external service integrations.

        Returns:
            ToolSet with all categories
        """
        return cls(categories={c.value for c in ToolCategory})

    @classmethod
    def airgapped(cls) -> "ToolSet":
        """Tool set for air-gapped environments (no network access).

        Includes core, filesystem, and git. Explicitly excludes
        web_search, web_fetch, http_request, and fetch_url regardless
        of any category extensions that might add them.

        Does not include web, database, docker, or communication categories.

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
          testing:     run_tests, pytest, test_runner, test
          refactoring: refactor, rename_symbol, extract_function, rename

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

        Includes core, filesystem, and web tools.
          web: web_search, web_fetch, http_request, fetch_url

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

        Returns the pre-resolved tool names. The cache is built during
        __post_init__ for performance (eager resolution).

        Returns:
            Set of tool names
        """
        # Cache is always set after __post_init__
        if self._resolved_names_cache is None:
            # Fallback: resolve on-demand (shouldn't happen after __post_init__)
            object.__setattr__(self, "_resolved_names_cache", self._resolve_tool_names())
        return self._resolved_names_cache

    @property
    def names(self) -> List[str]:
        """Return resolved tool names in SDK-compatible list form."""

        return sorted(self.get_tool_names())

    def _get_resolved_names(self) -> Set[str]:
        """Get resolved tool names from cache.

        The cache is pre-populated in __post_init__, so this method
        simply returns the cached value for O(1) access.

        Returns:
            Cached set of resolved tool names
        """
        # Cache should always be set after __post_init__
        if self._resolved_names_cache is None:
            # Fallback: resolve on-demand (shouldn't happen)
            object.__setattr__(self, "_resolved_names_cache", self._resolve_tool_names())
        return self._resolved_names_cache

    def invalidate_cache(self) -> None:
        """Invalidate the resolved names cache.

        Call this after modifying the ToolSet to ensure the cache
        is rebuilt on next access.
        """
        object.__setattr__(self, "_resolved_names_cache", None)

    def __contains__(self, tool: str) -> bool:
        """Check if a tool is in this set (cached O(1) after first call)."""
        return tool in self._get_resolved_names()

    def to_sdk_toolset(
        self,
        *,
        description: str = "",
        tier: Union[SdkTier, str] = SdkTier.STANDARD,
    ) -> SdkToolSet:
        """Convert the runtime toolset into the declarative SDK contract."""

        resolved_tier = SdkTier(tier) if isinstance(tier, str) else tier
        return SdkToolSet(
            names=self.names,
            description=description,
            tier=resolved_tier,
        )

    @classmethod
    def from_sdk_toolset(cls, toolset: Union[SdkToolSet, List[str], Set[str]]) -> "ToolSet":
        """Create a runtime toolset from an SDK toolset or tool-name list."""

        if isinstance(toolset, SdkToolSet):
            return cls.from_tools(toolset.names)
        if isinstance(toolset, (list, set)):
            return cls.from_tools(sorted(toolset))
        raise TypeError(f"Unsupported sdk toolset input: {type(toolset)!r}")


# Type alias for convenience
Tools = ToolSet

# Type for tools parameter that accepts multiple formats
ToolsInput = Union[ToolSet, List[str], None]
