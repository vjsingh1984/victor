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

"""Vertical Loader for dynamic vertical activation.

This module provides functionality for loading, activating, and managing
verticals at runtime. It integrates with the DI container to register
vertical-specific services.

Supports plugin discovery via entry points:
- victor.verticals: Entry point group for vertical plugins
- victor.tools: Entry point group for tool plugins

Usage:
    from victor.verticals.vertical_loader import VerticalLoader

    # Load and activate a vertical
    loader = VerticalLoader()
    vertical = loader.load("coding")

    # Get extensions for framework integration
    extensions = loader.get_extensions()

    # Register services with DI container
    loader.register_services(container, settings)

    # Discover installed plugins
    verticals = loader.discover_verticals()
    tools = loader.discover_tools()
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Dict, List, Optional, Type

from victor.verticals.base import VerticalBase, VerticalRegistry

# Python 3.10+ has entry_points with group parameter
if sys.version_info >= (3, 10):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points

if TYPE_CHECKING:
    from victor.core.container import ServiceContainer
    from victor.config.settings import Settings
    from victor.verticals.protocols import VerticalExtensions

logger = logging.getLogger(__name__)


# Built-in vertical mappings
BUILTIN_VERTICALS: Dict[str, str] = {
    "coding": "victor.verticals.coding.CodingAssistant",
    "research": "victor.verticals.research.ResearchAssistant",
    "devops": "victor.verticals.devops.DevOpsAssistant",
    "data_analysis": "victor.verticals.data_analysis.DataAnalysisAssistant",
}


class VerticalLoader:
    """Loader for dynamic vertical activation and management.

    Handles loading verticals by name, activating them, and integrating
    their extensions with the framework. Supports plugin discovery via
    Python entry points for verticals and tools.

    Entry Point Groups:
        - victor.verticals: Vertical plugins (e.g., coding, research)
        - victor.tools: Tool plugins (e.g., code_search, refactor)

    Attributes:
        _active_vertical: Currently active vertical class
        _extensions: Cached extensions from active vertical
        _discovered_verticals: Cache of discovered vertical entry points
        _discovered_tools: Cache of discovered tool entry points
    """

    def __init__(self) -> None:
        """Initialize the vertical loader."""
        self._active_vertical: Optional[Type[VerticalBase]] = None
        self._extensions: Optional["VerticalExtensions"] = None
        self._registered_services: bool = False
        self._discovered_verticals: Optional[Dict[str, Type[VerticalBase]]] = None
        self._discovered_tools: Optional[Dict[str, Type]] = None

    @property
    def active_vertical(self) -> Optional[Type[VerticalBase]]:
        """Get the currently active vertical."""
        return self._active_vertical

    @property
    def active_vertical_name(self) -> Optional[str]:
        """Get the name of the currently active vertical."""
        return self._active_vertical.name if self._active_vertical else None

    def load(self, name: str) -> Type[VerticalBase]:
        """Load and activate a vertical by name.

        Searches for verticals in this order:
        1. Global VerticalRegistry
        2. Built-in vertical mappings
        3. Entry point plugins (victor.verticals group)

        Args:
            name: Vertical name (e.g., "coding", "research")

        Returns:
            Loaded vertical class

        Raises:
            ValueError: If vertical not found
        """
        # First check the registry
        vertical = VerticalRegistry.get(name)

        if vertical is None:
            # Try to import from built-in mappings
            vertical = self._import_builtin(name)

        if vertical is None:
            # Try to load from entry points (plugins)
            vertical = self._import_from_entrypoint(name)

        if vertical is None:
            available = self._get_available_names()
            raise ValueError(f"Vertical '{name}' not found. Available: {', '.join(available)}")

        self._activate(vertical)
        return vertical

    def _import_builtin(self, name: str) -> Optional[Type[VerticalBase]]:
        """Import a built-in vertical by name.

        Args:
            name: Vertical name

        Returns:
            Vertical class or None
        """
        if name not in BUILTIN_VERTICALS:
            return None

        module_path = BUILTIN_VERTICALS[name]
        module_name, class_name = module_path.rsplit(".", 1)

        try:
            import importlib

            module = importlib.import_module(module_name)
            vertical_class = getattr(module, class_name)

            # Register it for future lookups
            VerticalRegistry.register(vertical_class)

            return vertical_class
        except (ImportError, AttributeError) as e:
            logger.warning("Failed to import vertical '%s': %s", name, e)
            return None

    def _import_from_entrypoint(self, name: str) -> Optional[Type[VerticalBase]]:
        """Import a vertical from entry points.

        Args:
            name: Vertical name

        Returns:
            Vertical class or None
        """
        discovered = self.discover_verticals()
        return discovered.get(name)

    def discover_verticals(self) -> Dict[str, Type[VerticalBase]]:
        """Discover verticals from installed packages via entry points.

        Scans the 'victor.verticals' entry point group for installed
        vertical plugins. Results are cached for performance.

        Returns:
            Dictionary mapping vertical names to their classes

        Example:
            # In victor-coding's pyproject.toml:
            # [project.entry-points."victor.verticals"]
            # coding = "victor_coding:CodingVertical"

            loader = VerticalLoader()
            verticals = loader.discover_verticals()
            # {'coding': <class 'victor_coding.CodingVertical'>}
        """
        if self._discovered_verticals is not None:
            return self._discovered_verticals

        self._discovered_verticals = {}

        try:
            eps = entry_points(group="victor.verticals")
            for ep in eps:
                try:
                    vertical_cls = ep.load()
                    if isinstance(vertical_cls, type) and issubclass(
                        vertical_cls, VerticalBase
                    ):
                        self._discovered_verticals[ep.name] = vertical_cls
                        # Also register in the global registry
                        VerticalRegistry.register(vertical_cls)
                        logger.debug("Discovered vertical plugin: %s", ep.name)
                    else:
                        logger.warning(
                            "Entry point '%s' is not a VerticalBase subclass",
                            ep.name,
                        )
                except Exception as e:
                    logger.warning("Failed to load vertical entry point '%s': %s", ep.name, e)
        except Exception as e:
            logger.warning("Failed to discover vertical entry points: %s", e)

        return self._discovered_verticals

    def discover_tools(self) -> Dict[str, Type]:
        """Discover tools from installed packages via entry points.

        Scans the 'victor.tools' entry point group for installed
        tool plugins. Results are cached for performance.

        Returns:
            Dictionary mapping tool names to their classes

        Example:
            # In victor-coding's pyproject.toml:
            # [project.entry-points."victor.tools"]
            # code_search = "victor_coding.tools:CodeSearchTool"

            loader = VerticalLoader()
            tools = loader.discover_tools()
            # {'code_search': <class 'victor_coding.tools.CodeSearchTool'>}
        """
        if self._discovered_tools is not None:
            return self._discovered_tools

        self._discovered_tools = {}

        try:
            eps = entry_points(group="victor.tools")
            for ep in eps:
                try:
                    tool_cls = ep.load()
                    self._discovered_tools[ep.name] = tool_cls
                    logger.debug("Discovered tool plugin: %s", ep.name)
                except Exception as e:
                    logger.warning("Failed to load tool entry point '%s': %s", ep.name, e)
        except Exception as e:
            logger.warning("Failed to discover tool entry points: %s", e)

        return self._discovered_tools

    def refresh_plugins(self) -> None:
        """Refresh the cached plugin discovery.

        Call this after installing new packages to re-scan entry points.
        """
        self._discovered_verticals = None
        self._discovered_tools = None
        logger.info("Plugin cache cleared, will re-discover on next access")

    def _activate(self, vertical: Type[VerticalBase]) -> None:
        """Activate a vertical.

        Args:
            vertical: Vertical class to activate
        """
        self._active_vertical = vertical
        self._extensions = None  # Clear cached extensions
        self._registered_services = False
        logger.info("Activated vertical: %s", vertical.name)

    def _get_available_names(self) -> List[str]:
        """Get list of available vertical names.

        Includes:
        - Registered verticals
        - Built-in verticals
        - Entry point plugin verticals

        Returns:
            List of vertical names
        """
        names = set(VerticalRegistry.list_names())
        names.update(BUILTIN_VERTICALS.keys())
        # Include entry point discovered verticals
        names.update(self.discover_verticals().keys())
        return sorted(names)

    def get_extensions(self) -> Optional["VerticalExtensions"]:
        """Get extensions from the active vertical.

        Returns:
            VerticalExtensions or None if no vertical active
        """
        if self._active_vertical is None:
            return None

        if self._extensions is None:
            self._extensions = self._active_vertical.get_extensions()

        return self._extensions

    def register_services(
        self,
        container: "ServiceContainer",
        settings: "Settings",
    ) -> None:
        """Register vertical-specific services with DI container.

        Args:
            container: DI container
            settings: Application settings
        """
        if self._registered_services:
            logger.debug("Vertical services already registered")
            return

        extensions = self.get_extensions()
        if extensions is None or extensions.service_provider is None:
            logger.debug("No service provider for active vertical")
            return

        try:
            extensions.service_provider.register_services(container, settings)
            self._registered_services = True
            logger.info(
                "Registered services for vertical: %s",
                self.active_vertical_name,
            )
        except Exception as e:
            logger.error("Failed to register vertical services: %s", e)

    def get_config(self):
        """Get configuration from active vertical.

        Returns:
            VerticalConfig or None
        """
        if self._active_vertical is None:
            return None
        return self._active_vertical.get_config()

    def get_tools(self) -> List[str]:
        """Get tools from active vertical.

        Returns:
            List of tool names
        """
        if self._active_vertical is None:
            return []
        return self._active_vertical.get_tools()

    def get_system_prompt(self) -> str:
        """Get system prompt from active vertical.

        Returns:
            System prompt string
        """
        if self._active_vertical is None:
            return ""
        return self._active_vertical.get_system_prompt()

    def reset(self) -> None:
        """Reset the loader, deactivating current vertical."""
        self._active_vertical = None
        self._extensions = None
        self._registered_services = False


# Global loader instance
_loader: Optional[VerticalLoader] = None


def get_vertical_loader() -> VerticalLoader:
    """Get the global vertical loader instance.

    Returns:
        Global VerticalLoader instance
    """
    global _loader
    if _loader is None:
        _loader = VerticalLoader()
    return _loader


def load_vertical(name: str) -> Type[VerticalBase]:
    """Load a vertical by name (convenience function).

    Args:
        name: Vertical name

    Returns:
        Loaded vertical class
    """
    return get_vertical_loader().load(name)


def get_active_vertical() -> Optional[Type[VerticalBase]]:
    """Get the currently active vertical (convenience function).

    Returns:
        Active vertical class or None
    """
    return get_vertical_loader().active_vertical


def get_vertical_extensions() -> Optional["VerticalExtensions"]:
    """Get extensions from active vertical (convenience function).

    Returns:
        VerticalExtensions or None
    """
    return get_vertical_loader().get_extensions()


def discover_vertical_plugins() -> Dict[str, Type[VerticalBase]]:
    """Discover vertical plugins from entry points (convenience function).

    Returns:
        Dictionary mapping vertical names to their classes
    """
    return get_vertical_loader().discover_verticals()


def discover_tool_plugins() -> Dict[str, Type]:
    """Discover tool plugins from entry points (convenience function).

    Returns:
        Dictionary mapping tool names to their classes
    """
    return get_vertical_loader().discover_tools()


__all__ = [
    "VerticalLoader",
    "get_vertical_loader",
    "load_vertical",
    "get_active_vertical",
    "get_vertical_extensions",
    "discover_vertical_plugins",
    "discover_tool_plugins",
    "BUILTIN_VERTICALS",
]
