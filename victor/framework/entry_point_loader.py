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

"""Entry point loader for framework capabilities.

This module provides utilities for loading vertical-specific capabilities
via entry points, enabling the framework to discover and use features
from external vertical packages without hardcoded dependencies.

Design Principles:
    - Framework does not depend on specific verticals
    - Verticals register capabilities via entry points
    - Graceful fallback when verticals are not installed
    - Clear separation between framework and vertical code

Entry Point Groups:
    victor.tool_dependencies  - Tool dependency provider factories
    victor.safety_rules        - Safety rule registration functions
    victor.rl_configs          - RL configuration provider factories
    victor.escape_hatches     - Escape hatch registration functions
    victor.commands            - CLI command registration functions
"""

from __future__ import annotations

import functools
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, TypeVar

from importlib.metadata import entry_points

from victor.framework.config import SafetyEnforcer

logger = logging.getLogger(__name__)

T = TypeVar("T")

_ENTRY_POINT_LOADER_STATS: Dict[str, int] = {
    "safety_calls": 0,
    "safety_loaded": 0,
    "safety_failures": 0,
    "tool_dependency_calls": 0,
    "tool_dependency_entry_point_resolutions": 0,
    "tool_dependency_fallback_resolutions": 0,
    "tool_dependency_none_returns": 0,
    "tool_dependency_failures": 0,
    "rl_config_calls": 0,
    "rl_config_hits": 0,
    "rl_config_failures": 0,
    "escape_hatch_calls": 0,
    "escape_hatch_loaded": 0,
    "escape_hatch_failures": 0,
    "command_calls": 0,
    "command_loaded": 0,
    "command_failures": 0,
    "cache_clears": 0,
}
_ENTRY_POINT_LOADER_STATS_LOCK = threading.Lock()


def _increment_loader_stat(name: str) -> None:
    """Increment an entry-point loader telemetry counter."""
    with _ENTRY_POINT_LOADER_STATS_LOCK:
        _ENTRY_POINT_LOADER_STATS[name] = _ENTRY_POINT_LOADER_STATS.get(name, 0) + 1


def get_entry_point_loader_stats() -> Dict[str, int]:
    """Get entry-point loader telemetry counters and cache stats."""
    with _ENTRY_POINT_LOADER_STATS_LOCK:
        stats = dict(_ENTRY_POINT_LOADER_STATS)

    cache_info = _cached_entry_points.cache_info()
    stats.update(
        {
            "cache_hits": cache_info.hits,
            "cache_misses": cache_info.misses,
            "cache_maxsize": cache_info.maxsize or 0,
            "cache_currsize": cache_info.currsize,
        }
    )
    return stats


def reset_entry_point_loader_stats(clear_cache: bool = False) -> None:
    """Reset entry-point loader telemetry counters and optionally clear cache."""
    if clear_cache:
        _cached_entry_points.cache_clear()
    with _ENTRY_POINT_LOADER_STATS_LOCK:
        for key in _ENTRY_POINT_LOADER_STATS:
            _ENTRY_POINT_LOADER_STATS[key] = 0


def _get_normalize_fn() -> Callable[[str], str]:
    """Lazy import normalize_vertical_name to avoid circular imports."""
    from victor.core.verticals.import_resolver import normalize_vertical_name

    return normalize_vertical_name


def normalize_vertical_name(name: str) -> str:
    """Normalize a vertical name (lazy-imported to avoid circular imports)."""
    return _get_normalize_fn()(name)


def _normalize_vertical_names(vertical_names: Optional[List[str]]) -> Optional[set[str]]:
    """Normalize an optional list of vertical names for matching."""
    if vertical_names is None:
        return None
    fn = _get_normalize_fn()
    return {fn(name) for name in vertical_names}


@functools.lru_cache(maxsize=16)
def _cached_entry_points(group: str) -> tuple:
    """Cache entry_points() result per group. Returns tuple for hashability."""
    return tuple(entry_points(group=group))


def clear_entry_point_loader_cache() -> None:
    """Clear cached entry-point lookups for this module."""
    _cached_entry_points.cache_clear()
    _increment_loader_stat("cache_clears")


def load_safety_rules_from_entry_points(
    enforcer: SafetyEnforcer,
    vertical_names: Optional[List[str]] = None,
) -> int:
    """Load safety rules from all installed vertical packages via entry points.

    This function discovers all safety rule registration functions via the
    victor.safety_rules entry point group and calls them to register rules
    with the provided enforcer.

    Args:
        enforcer: SafetyEnforcer to register rules with
        vertical_names: Optional list of specific verticals to load.
            If None, loads all available verticals.

    Returns:
        Number of verticals that successfully registered safety rules

    Example:
        from victor.framework.config import SafetyEnforcer, SafetyConfig, SafetyLevel
        from victor.framework.entry_point_loader import load_safety_rules_from_entry_points

        enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
        count = load_safety_rules_from_entry_points(enforcer)
        print(f"Loaded safety rules from {count} verticals")

        # Load specific verticals only
        count = load_safety_rules_from_entry_points(enforcer, vertical_names=["coding", "devops"])
    """
    _increment_loader_stat("safety_calls")
    count = 0
    normalized_verticals = _normalize_vertical_names(vertical_names)
    try:
        eps = _cached_entry_points("victor.safety_rules")
        for ep in eps:
            # Filter by vertical name if specified
            if (
                normalized_verticals is not None
                and normalize_vertical_name(ep.name) not in normalized_verticals
            ):
                continue

            try:
                register_func = ep.load()
                register_func(enforcer)
                count += 1
                _increment_loader_stat("safety_loaded")
                logger.debug(f"Loaded safety rules from '{ep.name}' vertical")
            except Exception as e:
                _increment_loader_stat("safety_failures")
                logger.warning(f"Failed to load safety rules from '{ep.name}': {e}")
    except Exception as e:
        _increment_loader_stat("safety_failures")
        logger.debug(f"No safety rule entry points found: {e}")

    return count


def load_tool_dependency_provider_from_entry_points(
    vertical: str,
) -> Optional[Any]:
    """Load a tool dependency provider for a specific vertical.

    Resolution order:
    1. ``victor.tool_dependencies`` entry points (preferred)
    2. Core compatibility loader fallbacks (module factory / package resources)

    Args:
        vertical: Vertical name (e.g., "coding", "devops", "research")

    Returns:
        Tool dependency provider instance, or None if not found

    Example:
        from victor.framework.entry_point_loader import load_tool_dependency_provider_from_entry_points

        provider = load_tool_dependency_provider_from_entry_points("coding")
        if provider:
            deps = provider.get_dependencies()
            sequences = provider.get_tool_sequences()
    """
    _increment_loader_stat("tool_dependency_calls")
    normalized_vertical = normalize_vertical_name(vertical)

    try:
        eps = _cached_entry_points("victor.tool_dependencies")
        for ep in eps:
            if normalize_vertical_name(ep.name) == normalized_vertical:
                provider_factory = ep.load()
                _increment_loader_stat("tool_dependency_entry_point_resolutions")
                return provider_factory()
    except Exception as e:
        _increment_loader_stat("tool_dependency_failures")
        logger.debug(f"No tool dependency provider found for '{vertical}' via entry points: {e}")

    # Compatibility fallback for extracted verticals (core + external packages).
    try:
        from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider
        from victor.core.tool_types import EmptyToolDependencyProvider

        provider = create_vertical_tool_dependency_provider(normalized_vertical)
        if isinstance(provider, EmptyToolDependencyProvider):
            _increment_loader_stat("tool_dependency_none_returns")
            return None
        _increment_loader_stat("tool_dependency_fallback_resolutions")
        return provider
    except ValueError:
        _increment_loader_stat("tool_dependency_none_returns")
        logger.debug("Unknown vertical '%s' for tool dependency provider resolution", vertical)
    except Exception as e:
        _increment_loader_stat("tool_dependency_failures")
        logger.debug(
            "Fallback tool dependency provider resolution failed for '%s': %s",
            vertical,
            e,
        )

    _increment_loader_stat("tool_dependency_none_returns")
    return None


def load_rl_config_from_entry_points(vertical: str) -> Optional[Dict[str, Any]]:
    """Load an RL configuration for a specific vertical via entry points.

    Args:
        vertical: Vertical name (e.g., "coding", "devops")

    Returns:
        RL configuration dictionary, or None if not found

    Example:
        from victor.framework.entry_point_loader import load_rl_config_from_entry_points

        rl_config = load_rl_config_from_entry_points("coding")
        if rl_config:
            learning_rate = rl_config.get("learning_rate", 0.001)
    """
    _increment_loader_stat("rl_config_calls")
    normalized_vertical = normalize_vertical_name(vertical)

    try:
        eps = _cached_entry_points("victor.rl_configs")
        for ep in eps:
            if normalize_vertical_name(ep.name) == normalized_vertical:
                config_factory = ep.load()
                _increment_loader_stat("rl_config_hits")
                return config_factory()
    except Exception as e:
        _increment_loader_stat("rl_config_failures")
        logger.debug(f"No RL config found for '{vertical}': {e}")

    return None


def register_escape_hatches_from_entry_points(
    registry: Any,
    vertical_names: Optional[List[str]] = None,
) -> int:
    """Register escape hatches from all installed vertical packages via entry points.

    Args:
        registry: Escape hatch registry to register hatches with
        vertical_names: Optional list of specific verticals to load.
            If None, loads all available verticals.

    Returns:
        Number of verticals that successfully registered escape hatches

    Example:
        from victor.framework.escape_hatch_registry import EscapeHatchRegistry
        from victor.framework.entry_point_loader import register_escape_hatches_from_entry_points

        registry = EscapeHatchRegistry()
        count = register_escape_hatches_from_entry_points(registry)
        print(f"Registered escape hatches from {count} verticals")
    """
    _increment_loader_stat("escape_hatch_calls")
    count = 0
    normalized_verticals = _normalize_vertical_names(vertical_names)
    try:
        eps = _cached_entry_points("victor.escape_hatches")
        for ep in eps:
            # Filter by vertical name if specified
            if (
                normalized_verticals is not None
                and normalize_vertical_name(ep.name) not in normalized_verticals
            ):
                continue

            try:
                register_func = ep.load()
                register_func(registry)
                count += 1
                _increment_loader_stat("escape_hatch_loaded")
                logger.debug(f"Registered escape hatches from '{ep.name}' vertical")
            except Exception as e:
                _increment_loader_stat("escape_hatch_failures")
                logger.warning(f"Failed to register escape hatches from '{ep.name}': {e}")
    except Exception as e:
        _increment_loader_stat("escape_hatch_failures")
        logger.debug(f"No escape hatch entry points found: {e}")

    return count


def register_commands_from_entry_points(
    app: Any,
    vertical_names: Optional[List[str]] = None,
) -> int:
    """Register CLI commands from all installed vertical packages via entry points.

    Args:
        app: Typer application to register commands with
        vertical_names: Optional list of specific verticals to load.
            If None, loads all available verticals.

    Returns:
        Number of verticals that successfully registered commands

    Example:
        import typer
        from victor.framework.entry_point_loader import register_commands_from_entry_points

        app = typer.Typer()
        count = register_commands_from_entry_points(app)
        print(f"Registered commands from {count} verticals")
    """
    _increment_loader_stat("command_calls")
    count = 0
    normalized_verticals = _normalize_vertical_names(vertical_names)
    try:
        eps = _cached_entry_points("victor.commands")
        for ep in eps:
            # Filter by vertical name if specified
            if (
                normalized_verticals is not None
                and normalize_vertical_name(ep.name) not in normalized_verticals
            ):
                continue

            try:
                register_func = ep.load()
                register_func(app)
                count += 1
                _increment_loader_stat("command_loaded")
                logger.debug(f"Registered commands from '{ep.name}' vertical")
            except Exception as e:
                _increment_loader_stat("command_failures")
                logger.warning(f"Failed to register commands from '{ep.name}': {e}")
    except Exception as e:
        _increment_loader_stat("command_failures")
        logger.debug(f"No command entry points found: {e}")

    return count


def list_installed_verticals() -> List[str]:
    """List all installed vertical packages via entry points.

    Returns:
        List of vertical names that have registered entry points

    Example:
        from victor.framework.entry_point_loader import list_installed_verticals

        verticals = list_installed_verticals()
        print(f"Installed verticals: {', '.join(verticals)}")
    """
    verticals = set()
    try:
        # Check victor.verticals entry points
        eps = _cached_entry_points("victor.verticals")
        for ep in eps:
            verticals.add(ep.name)
    except Exception:
        pass

    return sorted(list(verticals))


__all__ = [
    "load_safety_rules_from_entry_points",
    "load_tool_dependency_provider_from_entry_points",
    "load_rl_config_from_entry_points",
    "register_escape_hatches_from_entry_points",
    "register_commands_from_entry_points",
    "list_installed_verticals",
    "clear_entry_point_loader_cache",
    "get_entry_point_loader_stats",
    "reset_entry_point_loader_stats",
]
