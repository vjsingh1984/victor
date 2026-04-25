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

"""Tool tier configuration for schema level assignment.

This module provides data-driven tool tier assignments based on actual
usage patterns. Tools are assigned to FULL, COMPACT, or STUB schema levels
based on their usage frequency and importance.

Usage:
    from victor.config.tool_tiers import get_tool_tier

    tier = get_tool_tier("read")  # Returns "FULL", "COMPACT", or "STUB"
"""

import logging
from pathlib import Path
from typing import Dict, Optional

from victor.core.yaml_utils import safe_load

logger = logging.getLogger(__name__)

# Default tier assignments (fallback if no config file)
# These are conservative defaults until actual usage data is collected
_DEFAULT_TIERS = {
    # Essential tools that are used in almost every session
    "read": "FULL",
    "write": "FULL",
    "edit": "FULL",
    "code_search": "FULL",
    "shell": "FULL",

    # High-frequency tools used in most sessions
    "git_status": "COMPACT",
    "git_diff": "COMPACT",
    "test": "COMPACT",
    "ls": "COMPACT",
    "find": "COMPACT",
    "web_search": "COMPACT",

    # All other tools default to STUB
    "*": "STUB",
}

_tier_cache: Optional[Dict[str, str]] = None


def _load_tiers_from_config() -> Dict[str, str]:
    """Load tool tiers from configuration file.

    Returns:
        Dict mapping tool name to tier (FULL/COMPACT/STUB)
    """
    config_path = Path(__file__).parent / "tool_tiers.yaml"

    if not config_path.exists():
        logger.debug(f"Tool tiers config not found at {config_path}, using defaults")
        return _DEFAULT_TIERS.copy()

    try:
        with open(config_path, "r") as f:
            config = safe_load(f)

        if not config or "tool_tiers" not in config:
            logger.debug("No tool_tiers in config, using defaults")
            return _DEFAULT_TIERS.copy()

        # Build tier mapping from FULL and COMPACT lists
        tiers = {}

        for tool_name in config["tool_tiers"].get("FULL", []):
            tiers[tool_name] = "FULL"

        for tool_name in config["tool_tiers"].get("COMPACT", []):
            tiers[tool_name] = "COMPACT"

        # STUB is the default (handled by wildcard)
        tiers["*"] = "STUB"

        logger.debug(f"Loaded {len(tiers)} tool tiers from {config_path}")
        return tiers

    except Exception as e:
        logger.warning(f"Failed to load tool tiers from {config_path}: {e}, using defaults")
        return _DEFAULT_TIERS.copy()


def get_tool_tier(tool_name: str) -> str:
    """Get schema tier for a specific tool.

    Args:
        tool_name: Name of the tool

    Returns:
        Tier level: "FULL", "COMPACT", or "STUB"

    Examples:
        >>> get_tool_tier("read")
        'FULL'
        >>> get_tool_tier("unknown_tool")
        'STUB'
    """
    global _tier_cache

    if _tier_cache is None:
        _tier_cache = _load_tiers_from_config()

    # Look up specific tool
    if tool_name in _tier_cache:
        return _tier_cache[tool_name]

    # Check for wildcard
    if "*" in _tier_cache:
        return _tier_cache["*"]

    # Default to STUB
    return "STUB"


def reload_tiers() -> None:
    """Reload tool tiers from configuration file.

    Useful for picking up changes without restarting the process.
    """
    global _tier_cache
    _tier_cache = None


def get_all_tiers() -> Dict[str, str]:
    """Get all tool tier assignments.

    Returns:
        Dict mapping tool name to tier
    """
    global _tier_cache

    if _tier_cache is None:
        _tier_cache = _load_tiers_from_config()

    return _tier_cache.copy()


def get_tier_summary() -> Dict[str, int]:
    """Get summary of tool tier assignments.

    Returns:
        Dict with counts for each tier
    """
    tiers = get_all_tiers()

    summary = {
        "FULL": 0,
        "COMPACT": 0,
        "STUB": 0,
    }

    for tool_name, tier in tiers.items():
        if tool_name == "*":
            continue
        summary[tier] = summary.get(tier, 0) + 1

    return summary


# =============================================================================
# Provider-Specific Tier Assignments
# =============================================================================

_provider_tier_cache: Optional[Dict[str, Dict[str, list]]] = None


def get_provider_category(context_window: int) -> str:
    """Get provider category based on context window size.

    Args:
        context_window: Context window in tokens

    Returns:
        Provider category: 'edge', 'standard', or 'large'

    Examples:
        >>> get_provider_category(8192)
        'edge'
        >>> get_provider_category(32768)
        'standard'
        >>> get_provider_category(200000)
        'large'
    """
    if context_window < 16384:
        return "edge"
    elif context_window < 131072:
        return "standard"
    else:
        return "large"


def _load_provider_tiers() -> Dict[str, Dict[str, list]]:
    """Load provider-specific tier configurations from YAML.

    Returns:
        Dictionary mapping provider categories to their tier assignments
    """
    config_path = Path(__file__).parent / "tool_tiers.yaml"

    if not config_path.exists():
        logger.debug(f"Provider tiers config not found at {config_path}")
        return {}

    try:
        with open(config_path, "r") as f:
            config = safe_load(f)

        if not config or "provider_tiers" not in config:
            logger.debug("No provider_tiers in config")
            return {}

        logger.debug(f"Loaded provider tiers from {config_path}")
        return config.get("provider_tiers", {})

    except Exception as e:
        logger.warning(f"Failed to load provider tiers from {config_path}: {e}")
        return {}


def get_provider_tool_tier(tool_name: str, provider_category: str) -> str:
    """Get tool tier for a specific provider category.

    Args:
        tool_name: Name of the tool
        provider_category: Provider category ('edge', 'standard', 'large')

    Returns:
        Tool tier: 'FULL', 'COMPACT', or 'STUB'

    Examples:
        >>> get_provider_tool_tier("read", "edge")
        'FULL'
        >>> get_provider_tool_tier("ls", "edge")
        'STUB'
        >>> get_provider_tool_tier("write", "standard")
        'COMPACT'
    """
    global _provider_tier_cache

    if _provider_tier_cache is None:
        _provider_tier_cache = _load_provider_tiers()

    # Check if provider category exists
    if provider_category not in _provider_tier_cache:
        logger.debug(f"Provider category '{provider_category}' not found, using global tiers")
        return get_tool_tier(tool_name)

    category_tiers = _provider_tier_cache[provider_category]

    # Check FULL list
    if tool_name in category_tiers.get("FULL", []):
        return "FULL"

    # Check COMPACT list
    if tool_name in category_tiers.get("COMPACT", []):
        return "COMPACT"

    # Check STUB wildcard
    if category_tiers.get("STUB") == "*":
        return "STUB"

    # Fallback to global tiers
    return get_tool_tier(tool_name)


def reload_provider_tiers() -> None:
    """Reload provider-specific tool tiers from configuration file.

    Useful for picking up changes without restarting the process.
    """
    global _provider_tier_cache
    _provider_tier_cache = None
