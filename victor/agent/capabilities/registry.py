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

"""Dynamic capability registry with entry point support.

This module provides a dynamic capability registry that loads capabilities from
entry points, enabling Open/Closed Principle (OCP) compliance. External packages
can register capabilities without modifying core code.

Design Pattern: Registry + Plugin System
- Entry point discovery: Load capabilities from installed packages
- Built-in fallback: Ensure core capabilities always available
- Runtime registration: Support dynamic capability registration

Usage:
    registry = DynamicCapabilityRegistry()
    method_name = registry.get_method_for_capability("enabled_tools")
    # Returns: "set_enabled_tools"
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

try:
    from importlib.metadata import entry_points
except ImportError:
    # Python < 3.8
    from importlib_metadata import entry_points

from victor.agent.capabilities.base import CapabilitySpec

logger = logging.getLogger(__name__)


class DynamicCapabilityRegistry:
    """Dynamic capability registry (OCP compliant).

    This registry loads capabilities from multiple sources:
    1. Entry points (plugins) - "victor.capabilities" group
    2. Built-in capabilities - Direct imports
    3. Runtime registration - Dynamic add/remove

    The registry provides a fallback mechanism: if entry points fail to load,
    built-in capabilities are still available. This ensures reliability in all
    environments.

    Attributes:
        _capabilities: Dict mapping capability names to specs
        _entry_points_loaded: Whether entry points were successfully loaded

    Example:
        registry = DynamicCapabilityRegistry()
        method = registry.get_method_for_capability("enabled_tools")
        # Returns: "set_enabled_tools"

        # Register custom capability at runtime
        from victor.agent.capabilities.base import CapabilitySpec
        registry.register_capability(
            CapabilitySpec(
                name="my_custom",
                method_name="set_my_custom",
                version="1.0",
                description="My custom capability"
            )
        )
    """

    def __init__(self) -> None:
        """Initialize the dynamic capability registry."""
        self._capabilities: Dict[str, CapabilitySpec] = {}
        self._entry_points_loaded = False

        # Load from all sources
        self._load_from_entry_points()
        self._load_builtins()

        logger.debug(
            f"DynamicCapabilityRegistry initialized with {len(self._capabilities)} capabilities"
        )

    def _load_from_entry_points(self) -> None:
        """Load capabilities from entry points.

        This method discovers capabilities registered via entry points in the
        "victor.capabilities" group. External packages can register capabilities
        by adding to their pyproject.toml:

            [project.entry-points."victor.capabilities"]
            my_capability = "my_package.capabilities:MyCapabilityClass"

        Note:
            Errors during entry point loading are logged but don't raise exceptions.
            This ensures the registry remains functional even if some plugins fail.
        """
        try:
            eps = import_entry_points()
            # Python 3.10+ uses select() method, older versions have different API
            if hasattr(eps, "select"):
                capability_eps = eps.select(group="victor.capabilities")
            else:
                # Python 3.9 and earlier
                capability_eps = eps.get("victor.capabilities", [])  # type: ignore[arg-type]

            for ep in capability_eps:
                try:
                    capability_class = ep.load()
                    spec = capability_class.get_spec()
                    self._register(spec)
                    logger.debug(f"Loaded capability from entry point: {spec.name}")
                except Exception as e:
                    logger.warning(f"Failed to load entry point {ep.name}: {e}")

            self._entry_points_loaded = True

        except Exception as e:
            # Entry points might not exist in all environments (e.g., tests)
            logger.debug(f"No entry points found or failed to load: {e}")
            self._entry_points_loaded = False

    def _load_builtins(self) -> None:
        """Load built-in capabilities.

        This method loads all built-in capabilities directly. These serve as
        a fallback if entry points fail to load, ensuring core functionality
        is always available.

        Note:
            Built-in capabilities are overridden by entry point capabilities
            with the same name. This allows plugins to extend built-in capabilities.
        """
        from victor.agent.capabilities.builtin import (
            EnabledToolsCapability,
            ToolDependenciesCapability,
            ToolSequencesCapability,
            TieredToolConfigCapability,
            VerticalMiddlewareCapability,
            VerticalSafetyPatternsCapability,
            VerticalContextCapability,
            RlHooksCapability,
            TeamSpecsCapability,
            ModeConfigsCapability,
            DefaultBudgetCapability,
            CustomPromptCapability,
            PromptSectionCapability,
            TaskTypeHintsCapability,
            SafetyPatternsCapability,
            EnrichmentStrategyCapability,
        )

        builtin_capabilities = [
            EnabledToolsCapability,
            ToolDependenciesCapability,
            ToolSequencesCapability,
            TieredToolConfigCapability,
            VerticalMiddlewareCapability,
            VerticalSafetyPatternsCapability,
            VerticalContextCapability,
            RlHooksCapability,
            TeamSpecsCapability,
            ModeConfigsCapability,
            DefaultBudgetCapability,
            CustomPromptCapability,
            PromptSectionCapability,
            TaskTypeHintsCapability,
            SafetyPatternsCapability,
            EnrichmentStrategyCapability,
        ]

        for cap_class in builtin_capabilities:
            try:
                spec = cap_class.get_spec()
                # Only register if not already loaded from entry points
                if spec.name not in self._capabilities:
                    self._register(spec)
                    logger.debug(f"Loaded built-in capability: {spec.name}")
            except Exception as e:
                logger.warning(f"Failed to load built-in capability {cap_class.__name__}: {e}")

    def _register(self, spec: CapabilitySpec) -> None:
        """Register a capability.

        Args:
            spec: Capability specification to register

        Note:
            If a capability with the same name already exists, it will be
            overridden. This allows entry point capabilities to override
            built-in ones.
        """
        self._capabilities[spec.name] = spec

    def get_method_for_capability(self, capability_name: str) -> str:
        """Get method name for a capability.

        This method provides backward compatibility with the old hard-coded
        CAPABILITY_METHOD_MAPPINGS dict. It looks up the capability and returns
        the associated method name.

        Args:
            capability_name: Name of the capability to look up

        Returns:
            Method name to call for this capability

        Note:
            If capability is not found, falls back to "set_{capability_name}"
            for backward compatibility.
        """
        if capability_name in self._capabilities:
            return self._capabilities[capability_name].method_name

        # Fallback for backward compatibility
        logger.debug(f"Capability '{capability_name}' not found, using fallback")
        return f"set_{capability_name}"

    def register_capability(self, spec: CapabilitySpec) -> None:
        """Register a new capability at runtime.

        This method enables dynamic capability registration (OCP). External code
        can register new capabilities without modifying core code.

        Args:
            spec: Capability specification to register

        Example:
            from victor.agent.capabilities.base import CapabilitySpec
            registry = DynamicCapabilityRegistry()
            registry.register_capability(
                CapabilitySpec(
                    name="my_custom",
                    method_name="set_my_custom",
                    version="1.0",
                    description="My custom capability"
                )
            )
        """
        self._register(spec)
        logger.info(f"Registered runtime capability: {spec.name}")

    def get_capability(self, name: str) -> Optional[CapabilitySpec]:
        """Get a capability by name.

        Args:
            name: Capability name

        Returns:
            CapabilitySpec or None if not found
        """
        return self._capabilities.get(name)

    def list_capabilities(self) -> Dict[str, CapabilitySpec]:
        """List all registered capabilities.

        Returns:
            Dict mapping capability names to their specs
        """
        return dict(self._capabilities)

    def has_capability(self, name: str) -> bool:
        """Check if a capability is registered.

        Args:
            name: Capability name

        Returns:
            True if capability is registered
        """
        return name in self._capabilities

    def unregister_capability(self, name: str) -> bool:
        """Unregister a capability.

        Args:
            name: Capability name to unregister

        Returns:
            True if capability was unregistered, False if not found

        Note:
            This is mainly useful for testing. In production, capabilities
            should typically remain registered once loaded.
        """
        if name in self._capabilities:
            del self._capabilities[name]
            logger.debug(f"Unregistered capability: {name}")
            return True
        return False

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics.

        Returns:
            Dict with registry statistics including:
            - total_capabilities: Total number of capabilities
            - entry_points_loaded: Whether entry points were loaded
        """
        return {
            "total_capabilities": len(self._capabilities),
            "entry_points_loaded": self._entry_points_loaded,
        }


__all__ = ["DynamicCapabilityRegistry"]
