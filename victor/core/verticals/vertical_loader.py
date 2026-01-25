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

Lazy Loading:
    Supports lazy loading via vertical_loading_mode configuration:
    - eager: Load all extensions immediately (default, backward compatible)
    - lazy: Load metadata only, defer heavy modules until first access
    - auto: Automatically choose based on environment (production=lazy, dev=eager)

Usage:
    from victor.core.verticals.vertical_loader import VerticalLoader

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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from victor.framework.module_loader import get_entry_point_cache
from victor.core.verticals.base import VerticalBase, VerticalRegistry

if TYPE_CHECKING:
    from victor.core.container import ServiceContainer
    from victor.config.settings import Settings
    from victor.core.verticals.protocols import VerticalExtensions

logger = logging.getLogger(__name__)


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
        self._lazy_mode: bool = False  # Track if lazy loading is enabled

    @property
    def active_vertical(self) -> Optional[Type[VerticalBase]]:
        """Get the currently active vertical."""
        return self._active_vertical

    @property
    def active_vertical_name(self) -> Optional[str]:
        """Get the name of the currently active vertical."""
        return self._active_vertical.name if self._active_vertical else None

    def configure_lazy_mode(self, settings: Optional["Settings"] = None) -> None:
        """Configure lazy loading mode based on settings.

        Args:
            settings: Application settings (if None, tries to load from globals)
        """
        if settings is None:
            try:
                from victor.config.settings import get_settings

                settings = get_settings()
            except Exception:
                logger.debug("Could not load settings, using eager mode")
                self._lazy_mode = False
                return

        mode = getattr(settings, "vertical_loading_mode", "eager")
        self._lazy_mode = mode in ("lazy", "on_demand")

        if self._lazy_mode:
            logger.info(f"Lazy loading enabled for verticals (mode: {mode})")
        else:
            logger.debug(f"Eager loading enabled for verticals (mode: {mode})")

    def load(self, name: str, lazy: Optional[bool] = None) -> Union[Type[VerticalBase], Any]:
        """Load and activate a vertical by name.

        Searches for verticals in this order:
        1. Global VerticalRegistry (includes built-ins registered on import)
        2. Entry point plugins (victor.verticals group)

        Args:
            name: Vertical name (e.g., "coding", "research")
            lazy: Force lazy/eager loading. If None, uses configured mode.

        Returns:
            Loaded vertical class (or LazyVerticalProxy if lazy=True)

        Raises:
            ValueError: If vertical not found
        """
        # Determine if lazy loading should be used
        use_lazy = lazy if lazy is not None else self._lazy_mode

        # Query VerticalRegistry (includes built-ins registered on import)
        vertical = VerticalRegistry.get(name)

        # Try entry points as fallback
        if vertical is None:
            vertical = self._import_from_entrypoint(name)

        # Error with available names
        if vertical is None:
            available = self._get_available_names()
            raise ValueError(f"Vertical '{name}' not found. Available: {', '.join(available)}")

        # Return lazy proxy if lazy mode enabled
        if use_lazy:
            from victor.core.verticals.lazy_proxy import LazyProxy, LazyProxyType

            def _load_vertical() -> Type[VerticalBase]:
                self._activate(vertical)
                return vertical

            proxy = LazyProxy[Type[VerticalBase]](
                vertical_name=name, loader=_load_vertical, proxy_type=LazyProxyType.LAZY
            )
            logger.debug(f"Created type-safe lazy proxy for vertical: {name}")
            return proxy

        # Eager mode: activate immediately
        self._activate(vertical)
        return vertical

    def _import_from_entrypoint(self, name: str) -> Optional[Type[VerticalBase]]:
        """Import a vertical from entry points.

        Args:
            name: Vertical name

        Returns:
            Vertical class or None
        """
        discovered = self.discover_verticals()
        return discovered.get(name)

    def discover_verticals(
        self,
        force_refresh: bool = False,
    ) -> Dict[str, Type[VerticalBase]]:
        """Discover verticals from installed packages via entry points.

        Scans the 'victor.verticals' entry point group for installed
        vertical plugins. Results are cached for performance using
        EntryPointCache for fast startup.

        Args:
            force_refresh: Force re-scan of entry points (bypass cache)

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
        if self._discovered_verticals is not None and not force_refresh:
            return self._discovered_verticals

        self._discovered_verticals = {}

        try:
            # Use cached entry points for fast startup
            cache = get_entry_point_cache()
            ep_entries = cache.get_entry_points(
                "victor.verticals",
                force_refresh=force_refresh,
            )

            self._load_vertical_entries(ep_entries)
        except Exception as e:
            logger.warning("Failed to discover vertical entry points: %s", e)

        return self._discovered_verticals

    async def discover_verticals_async(
        self,
        force_refresh: bool = False,
    ) -> Dict[str, Type[VerticalBase]]:
        """Discover verticals asynchronously (non-blocking).

        Async version of discover_verticals() that offloads entry point
        scanning to a thread pool to avoid blocking the event loop.

        Args:
            force_refresh: Force re-scan of entry points (bypass cache)

        Returns:
            Dictionary mapping vertical names to their classes
        """
        if self._discovered_verticals is not None and not force_refresh:
            return self._discovered_verticals

        self._discovered_verticals = {}

        try:
            # Use async entry point cache to avoid blocking
            cache = get_entry_point_cache()
            ep_entries = await cache.get_entry_points_async(
                "victor.verticals",
                force_refresh=force_refresh,
            )

            self._load_vertical_entries(ep_entries)
        except Exception as e:
            logger.warning("Failed to discover vertical entry points: %s", e)

        return self._discovered_verticals

    def _load_vertical_entries(self, ep_entries: Dict[str, str]) -> None:
        """Load vertical classes from entry point entries.

        Args:
            ep_entries: Dictionary of name -> module:attr strings
        """
        for name, value in ep_entries.items():
            try:
                # Parse "module:attr" format and load
                vertical_cls = self._load_entry_point(name, value)
                if isinstance(vertical_cls, type) and issubclass(vertical_cls, VerticalBase):
                    self._discovered_verticals[name] = vertical_cls
                    # Also register in the global registry
                    VerticalRegistry.register(vertical_cls)
                    logger.debug("Discovered vertical plugin: %s", name)
                else:
                    logger.warning(
                        "Entry point '%s' is not a VerticalBase subclass",
                        name,
                    )
            except Exception as e:
                logger.warning("Failed to load vertical entry point '%s': %s", name, e)

    def _load_entry_point(self, name: str, value: str) -> Type[Any]:  # noqa: ANN401
        """Load an entry point by its value string.

        Args:
            name: Entry point name
            value: Entry point value (module:attr format)

        Returns:
            Loaded class/object
        """
        import importlib

        if ":" in value:
            module_name, attr_name = value.split(":", 1)
        else:
            # Handle "module.Class" format
            module_name, attr_name = value.rsplit(".", 1)

        module = importlib.import_module(module_name)
        result = getattr(module, attr_name)
        return result  # type: ignore[return-value]

    def discover_tools(
        self,
        force_refresh: bool = False,
    ) -> Dict[str, Type]:
        """Discover tools from installed packages via entry points.

        Scans the 'victor.tools' entry point group for installed
        tool plugins. Results are cached for performance using
        EntryPointCache for fast startup.

        Args:
            force_refresh: Force re-scan of entry points (bypass cache)

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
        if self._discovered_tools is not None and not force_refresh:
            return self._discovered_tools

        self._discovered_tools = {}

        try:
            # Use cached entry points for fast startup
            cache = get_entry_point_cache()
            ep_entries = cache.get_entry_points(
                "victor.tools",
                force_refresh=force_refresh,
            )

            self._load_tool_entries(ep_entries)
        except Exception as e:
            logger.warning("Failed to discover tool entry points: %s", e)

        return self._discovered_tools

    async def discover_tools_async(
        self,
        force_refresh: bool = False,
    ) -> Dict[str, Type]:
        """Discover tools asynchronously (non-blocking).

        Async version of discover_tools() that offloads entry point
        scanning to a thread pool to avoid blocking the event loop.

        Args:
            force_refresh: Force re-scan of entry points (bypass cache)

        Returns:
            Dictionary mapping tool names to their classes
        """
        if self._discovered_tools is not None and not force_refresh:
            return self._discovered_tools

        self._discovered_tools = {}

        try:
            # Use async entry point cache to avoid blocking
            cache = get_entry_point_cache()
            ep_entries = await cache.get_entry_points_async(
                "victor.tools",
                force_refresh=force_refresh,
            )

            self._load_tool_entries(ep_entries)
        except Exception as e:
            logger.warning("Failed to discover tool entry points: %s", e)

        return self._discovered_tools

    def _load_tool_entries(self, ep_entries: Dict[str, str]) -> None:
        """Load tool classes from entry point entries.

        Args:
            ep_entries: Dictionary of name -> module:attr strings
        """
        for name, value in ep_entries.items():
            try:
                tool_cls = self._load_entry_point(name, value)
                self._discovered_tools[name] = tool_cls
                logger.debug("Discovered tool plugin: %s", name)
            except Exception as e:
                logger.warning("Failed to load tool entry point '%s': %s", name, e)

    def refresh_plugins(self) -> None:
        """Refresh the cached plugin discovery.

        Call this after installing new packages to re-scan entry points.
        Invalidates both the local cache and the global EntryPointCache.
        Also clears the extension cache for consistency.
        """
        self._discovered_verticals = None
        self._discovered_tools = None

        # Also invalidate the entry point cache
        cache = get_entry_point_cache()
        cache.invalidate("victor.verticals")
        cache.invalidate("victor.tools")

        # Clear extension cache for consistency (Phase 3.3 fix)
        from victor.core.verticals.extension_loader import VerticalExtensionLoader

        VerticalExtensionLoader.clear_extension_cache(clear_all=True)

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
        - Registered verticals (includes built-ins registered on import)
        - Entry point plugin verticals

        Returns:
            List of vertical names
        """
        names = set(VerticalRegistry.list_names())
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

    def get_config(self) -> Optional[Any]:
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


def load_vertical(name: str, lazy: Optional[bool] = None) -> Union[Type[VerticalBase], Any]:
    """Load a vertical by name (convenience function).

    Args:
        name: Vertical name
        lazy: Force lazy/eager loading. If None, uses configured mode.

    Returns:
        Loaded vertical class (or LazyVerticalProxy if lazy=True)
    """
    return get_vertical_loader().load(name, lazy=lazy)


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


async def discover_vertical_plugins_async() -> Dict[str, Type[VerticalBase]]:
    """Discover vertical plugins asynchronously (convenience function).

    Non-blocking version that offloads entry point scanning to thread pool.

    Returns:
        Dictionary mapping vertical names to their classes
    """
    return await get_vertical_loader().discover_verticals_async()


async def discover_tool_plugins_async() -> Dict[str, Type]:
    """Discover tool plugins asynchronously (convenience function).

    Non-blocking version that offloads entry point scanning to thread pool.

    Returns:
        Dictionary mapping tool names to their classes
    """
    return await get_vertical_loader().discover_tools_async()


__all__ = [
    "VerticalLoader",
    "get_vertical_loader",
    "load_vertical",
    "get_active_vertical",
    "get_vertical_extensions",
    "discover_vertical_plugins",
    "discover_tool_plugins",
    "discover_vertical_plugins_async",
    "discover_tool_plugins_async",
]
