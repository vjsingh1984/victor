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

import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar

from importlib.metadata import entry_points

from victor.framework.config import SafetyEnforcer

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
    count = 0
    try:
        eps = entry_points(group="victor.safety_rules")
        for ep in eps:
            # Filter by vertical name if specified
            if vertical_names is not None and ep.name not in vertical_names:
                continue

            try:
                register_func = ep.load()
                register_func(enforcer)
                count += 1
                logger.debug(f"Loaded safety rules from '{ep.name}' vertical")
            except Exception as e:
                logger.warning(f"Failed to load safety rules from '{ep.name}': {e}")
    except Exception as e:
        logger.debug(f"No safety rule entry points found: {e}")

    return count


def load_tool_dependency_provider_from_entry_points(
    vertical: str,
) -> Optional[Any]:
    """Load a tool dependency provider for a specific vertical via entry points.

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
    try:
        eps = entry_points(group="victor.tool_dependencies")
        for ep in eps:
            if ep.name == vertical:
                provider_factory = ep.load()
                return provider_factory()
    except Exception as e:
        logger.debug(f"No tool dependency provider found for '{vertical}': {e}")

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
    try:
        eps = entry_points(group="victor.rl_configs")
        for ep in eps:
            if ep.name == vertical:
                config_factory = ep.load()
                return config_factory()
    except Exception as e:
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
    count = 0
    try:
        eps = entry_points(group="victor.escape_hatches")
        for ep in eps:
            # Filter by vertical name if specified
            if vertical_names is not None and ep.name not in vertical_names:
                continue

            try:
                register_func = ep.load()
                register_func(registry)
                count += 1
                logger.debug(f"Registered escape hatches from '{ep.name}' vertical")
            except Exception as e:
                logger.warning(f"Failed to register escape hatches from '{ep.name}': {e}")
    except Exception as e:
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
    count = 0
    try:
        eps = entry_points(group="victor.commands")
        for ep in eps:
            # Filter by vertical name if specified
            if vertical_names is not None and ep.name not in vertical_names:
                continue

            try:
                register_func = ep.load()
                register_func(app)
                count += 1
                logger.debug(f"Registered commands from '{ep.name}' vertical")
            except Exception as e:
                logger.warning(f"Failed to register commands from '{ep.name}': {e}")
    except Exception as e:
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
        eps = entry_points(group="victor.verticals")
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
]
