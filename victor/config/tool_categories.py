"""Tool categories configuration loader.

Phase 7.5: Externalizes tool category definitions to YAML for OCP compliance.

This module provides:
- load_tool_categories(): Loads categories from YAML file
- load_presets(): Loads preset configurations
- get_categories_file_path(): Returns path to tool_categories.yaml

The YAML file serves as the single source of truth for tool categories.
Python code no longer contains hardcoded category-to-tool mappings.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

# Cache for loaded configuration
_categories_cache: Optional[Dict[str, Set[str]]] = None
_presets_cache: Optional[Dict[str, Dict[str, Any]]] = None


def get_categories_file_path() -> Path:
    """Get path to tool_categories.yaml.

    Returns:
        Path to the YAML configuration file
    """
    return Path(__file__).parent / "tool_categories.yaml"


@lru_cache(maxsize=1)
def _load_yaml_config() -> Dict[str, Any]:
    """Load and cache the YAML configuration.

    Returns:
        Parsed YAML as dictionary
    """
    yaml_path = get_categories_file_path()

    if not yaml_path.exists():
        logger.warning(f"Tool categories file not found: {yaml_path}")
        return {"categories": {}, "presets": {}}

    try:
        import yaml

        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f) or {}
        return config
    except ImportError:
        logger.warning("PyYAML not installed, using fallback categories")
        return {"categories": {}, "presets": {}}
    except Exception as e:
        logger.error(f"Failed to load tool categories: {e}")
        return {"categories": {}, "presets": {}}


def load_tool_categories() -> Dict[str, Set[str]]:
    """Load tool categories from YAML configuration.

    Returns a dictionary mapping category names to sets of tool names.
    Falls back to hardcoded defaults if YAML loading fails.

    Returns:
        Dict mapping category name -> set of tool names

    Example:
        categories = load_tool_categories()
        core_tools = categories.get("core", set())
        # {'read', 'write', 'edit', 'shell', 'search', 'code_search', 'ls'}
    """
    global _categories_cache

    if _categories_cache is not None:
        return _categories_cache

    config = _load_yaml_config()
    categories_config = config.get("categories", {})

    result: Dict[str, Set[str]] = {}

    for category_name, category_data in categories_config.items():
        if isinstance(category_data, dict):
            tools = category_data.get("tools", [])
            result[category_name] = set(tools) if isinstance(tools, list) else set()
        elif isinstance(category_data, list):
            # Support direct list format
            result[category_name] = set(category_data)
        else:
            result[category_name] = set()

    _categories_cache = result
    return result


def load_presets() -> Dict[str, Dict[str, Any]]:
    """Load preset configurations from YAML.

    Presets define common tool configurations like 'default', 'minimal',
    'full', 'airgapped', etc.

    Returns:
        Dict mapping preset name -> preset configuration

    Example:
        presets = load_presets()
        default_preset = presets.get("default", {})
        # {'categories': ['core', 'filesystem', 'git']}
    """
    global _presets_cache

    if _presets_cache is not None:
        return _presets_cache

    config = _load_yaml_config()
    _presets_cache = config.get("presets", {})
    return _presets_cache


def get_preset_categories(preset_name: str) -> Set[str]:
    """Get category names for a preset.

    Args:
        preset_name: Name of the preset (e.g., 'default', 'minimal')

    Returns:
        Set of category names included in the preset

    Example:
        categories = get_preset_categories("default")
        # {'core', 'filesystem', 'git'}
    """
    presets = load_presets()
    preset = presets.get(preset_name, {})
    categories = preset.get("categories", [])
    return set(categories)


def get_preset_tools(preset_name: str) -> Set[str]:
    """Get all tool names for a preset.

    Resolves categories to their constituent tools and applies
    any exclusions defined in the preset.

    Args:
        preset_name: Name of the preset

    Returns:
        Set of tool names for the preset

    Example:
        tools = get_preset_tools("airgapped")
        # All tools from core, filesystem, git, analysis minus web tools
    """
    presets = load_presets()
    preset = presets.get(preset_name, {})
    categories = load_tool_categories()

    # Collect tools from included categories
    tools: Set[str] = set()
    for category_name in preset.get("categories", []):
        tools.update(categories.get(category_name, set()))

    # Apply exclusions
    exclude = preset.get("exclude", [])
    tools -= set(exclude)

    return tools


def get_category_description(category_name: str) -> Optional[str]:
    """Get description for a category.

    Args:
        category_name: Name of the category

    Returns:
        Description string or None
    """
    config = _load_yaml_config()
    categories = config.get("categories", {})
    category = categories.get(category_name, {})

    if isinstance(category, dict):
        return category.get("description")
    return None


def clear_cache() -> None:
    """Clear cached configuration.

    Call this after modifying the YAML file to reload on next access.
    """
    global _categories_cache, _presets_cache
    _categories_cache = None
    _presets_cache = None
    _load_yaml_config.cache_clear()


# =============================================================================
# Fallback defaults (used if YAML loading fails)
# =============================================================================

_FALLBACK_CATEGORIES: Dict[str, Set[str]] = {
    "core": {"read", "write", "edit", "shell", "search", "code_search", "ls"},
    "filesystem": {
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
    "git": {"git", "git_status", "git_diff", "git_commit", "git_branch", "git_log"},
    "search": {"grep", "glob", "code_search", "semantic_code_search", "search"},
    "web": {"web_search", "web_fetch", "http_request", "fetch_url"},
    "database": {"sql_query", "db_schema", "database"},
    "docker": {"docker", "docker_run", "docker_build", "docker_compose"},
    "testing": {"run_tests", "pytest", "test_runner", "test"},
    "refactoring": {"refactor", "rename_symbol", "extract_function", "rename"},
    "documentation": {"generate_docs", "update_readme", "documentation", "docstring"},
    "analysis": {"analyze", "complexity"},
    "communication": {"slack", "teams", "jira"},
    "custom": set(),
}


def get_fallback_categories() -> Dict[str, Set[str]]:
    """Get fallback categories when YAML loading fails.

    Returns:
        Dict mapping category name -> set of tool names
    """
    return _FALLBACK_CATEGORIES.copy()
