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

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

from victor.core.events.emit_helper import emit_event_sync
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
        self._lock = threading.RLock()
        self._active_vertical: Optional[Type[VerticalBase]] = None
        self._extensions: Optional["VerticalExtensions"] = None
        self._registered_services: bool = False
        self._discovered_verticals: Optional[Dict[str, Type[VerticalBase]]] = None
        self._discovered_tools: Optional[Dict[str, Type]] = None
        # Discovery telemetry counters (for diagnostics and observability).
        self._vertical_discovery_calls: int = 0
        self._vertical_discovery_cache_hits: int = 0
        self._vertical_discovery_scans: int = 0
        self._vertical_last_discovery_ms: float = 0.0
        self._tool_discovery_calls: int = 0
        self._tool_discovery_cache_hits: int = 0
        self._tool_discovery_scans: int = 0
        self._tool_last_discovery_ms: float = 0.0
        self._plugin_refresh_count: int = 0
        self._plugin_refresh_last_ms: float = 0.0

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
        1. Global VerticalRegistry (includes built-ins registered on import)
        2. Entry point plugins (victor.verticals group)

        Args:
            name: Vertical name (e.g., "coding", "research")

        Returns:
            Loaded vertical class

        Raises:
            ValueError: If vertical not found
        """
        with self._lock:
            # Query VerticalRegistry (includes built-ins registered on import)
            vertical = VerticalRegistry.get(name)

            # Try entry points as fallback
            if vertical is None:
                vertical = self._import_from_entrypoint(name)

            # Error with available names
            if vertical is None:
                available = self._get_available_names()
                raise ValueError(f"Vertical '{name}' not found. Available: {', '.join(available)}")

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

    def _emit_observability_event(self, topic: str, data: Dict[str, Any]) -> None:
        """Emit loader observability event from sync contexts."""
        try:
            from victor.core.events import get_observability_bus

            bus = get_observability_bus()
            if bus:
                emit_event_sync(
                    bus,
                    topic,
                    data,
                    source="VerticalLoader",
                    use_background_loop=True,
                )
        except Exception as e:
            logger.debug("Failed to emit %s event: %s", topic, e)

    async def _emit_observability_event_async(self, topic: str, data: Dict[str, Any]) -> None:
        """Emit loader observability event from async contexts."""
        try:
            from victor.core.events import get_observability_bus

            bus = get_observability_bus()
            if bus:
                await bus.emit(
                    topic=topic,
                    data=data,
                    source="VerticalLoader",
                )
        except Exception as e:
            logger.debug("Failed to emit %s event (async): %s", topic, e)

    def _build_discovery_event_payload(
        self,
        *,
        kind: str,
        count: int,
        duration_ms: float,
        cache_hit: bool,
        force_refresh: bool,
    ) -> Dict[str, Any]:
        """Build discovery observability payload."""
        return {
            "kind": kind,
            "count": count,
            "duration_ms": duration_ms,
            "cache_hit": cache_hit,
            "force_refresh": force_refresh,
            "stats": self.get_discovery_stats(),
        }

    def discover_verticals(
        self,
        force_refresh: bool = False,
        emit_event: bool = True,
    ) -> Dict[str, Type[VerticalBase]]:
        """Discover verticals from installed packages via entry points.

        Scans the 'victor.verticals' entry point group for installed
        vertical plugins. Results are cached for performance using
        EntryPointCache for fast startup.

        Args:
            force_refresh: Force re-scan of entry points (bypass cache)
            emit_event: Internal flag to suppress sync event emission when
                async callers emit from the event-loop context.

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
        discovered, cache_hit, duration_ms = self._discover_verticals_internal(force_refresh)
        if emit_event:
            self._emit_observability_event(
                topic="vertical.plugins.discovered",
                data=self._build_discovery_event_payload(
                    kind="vertical",
                    count=len(discovered),
                    duration_ms=duration_ms,
                    cache_hit=cache_hit,
                    force_refresh=force_refresh,
                ),
            )
        return discovered

    def _discover_verticals_internal(
        self,
        force_refresh: bool = False,
    ) -> Tuple[Dict[str, Type[VerticalBase]], bool, float]:
        """Discover verticals and return result with cache/duration metadata."""
        with self._lock:
            self._vertical_discovery_calls += 1
            if self._discovered_verticals is not None and not force_refresh:
                self._vertical_discovery_cache_hits += 1
                return self._discovered_verticals, True, 0.0

            start = time.perf_counter()
            self._vertical_discovery_scans += 1
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
            finally:
                self._vertical_last_discovery_ms = max(
                    0.0,
                    (time.perf_counter() - start) * 1000.0,
                )

            return self._discovered_verticals, False, self._vertical_last_discovery_ms

    async def discover_verticals_async(
        self,
        force_refresh: bool = False,
    ) -> Dict[str, Type[VerticalBase]]:
        """Discover verticals asynchronously (non-blocking).

        Async version of discover_verticals() that offloads entry point
        scanning to a thread pool to avoid blocking the event loop.

        Args:
            force_refresh: Force re-scan of entry points (bypass cache)
            emit_event: Internal flag to suppress sync event emission when
                async callers emit from the event-loop context.

        Returns:
            Dictionary mapping vertical names to their classes
        """
        discovered, cache_hit, duration_ms = await asyncio.to_thread(
            self._discover_verticals_internal,
            force_refresh,
        )

        await self._emit_observability_event_async(
            topic="vertical.plugins.discovered",
            data=self._build_discovery_event_payload(
                kind="vertical",
                count=len(discovered),
                duration_ms=duration_ms,
                cache_hit=cache_hit,
                force_refresh=force_refresh,
            ),
        )
        return discovered

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

    def _load_entry_point(self, name: str, value: str) -> Type:
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
        return getattr(module, attr_name)

    def discover_tools(
        self,
        force_refresh: bool = False,
        emit_event: bool = True,
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
        discovered, cache_hit, duration_ms = self._discover_tools_internal(force_refresh)
        if emit_event:
            self._emit_observability_event(
                topic="vertical.plugins.discovered",
                data=self._build_discovery_event_payload(
                    kind="tools",
                    count=len(discovered),
                    duration_ms=duration_ms,
                    cache_hit=cache_hit,
                    force_refresh=force_refresh,
                ),
            )
        return discovered

    def _discover_tools_internal(
        self,
        force_refresh: bool = False,
    ) -> Tuple[Dict[str, Type], bool, float]:
        """Discover tools and return result with cache/duration metadata."""
        with self._lock:
            self._tool_discovery_calls += 1
            if self._discovered_tools is not None and not force_refresh:
                self._tool_discovery_cache_hits += 1
                return self._discovered_tools, True, 0.0

            start = time.perf_counter()
            self._tool_discovery_scans += 1
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
            finally:
                self._tool_last_discovery_ms = max(
                    0.0,
                    (time.perf_counter() - start) * 1000.0,
                )

            return self._discovered_tools, False, self._tool_last_discovery_ms

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
        discovered, cache_hit, duration_ms = await asyncio.to_thread(
            self._discover_tools_internal,
            force_refresh,
        )

        await self._emit_observability_event_async(
            topic="vertical.plugins.discovered",
            data=self._build_discovery_event_payload(
                kind="tools",
                count=len(discovered),
                duration_ms=duration_ms,
                cache_hit=cache_hit,
                force_refresh=force_refresh,
            ),
        )
        return discovered

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
        with self._lock:
            refresh_start = time.perf_counter()
            self._plugin_refresh_count += 1
            self._discovered_verticals = None
            self._discovered_tools = None
            # Reset loader-level extension/service state to avoid stale plugin config
            self._extensions = None
            self._registered_services = False

            # Also invalidate the entry point cache
            cache = get_entry_point_cache()
            cache.invalidate("victor.verticals")
            cache.invalidate("victor.tools")

            # Clear extension cache for consistency (Phase 3.3 fix)
            from victor.core.verticals.extension_loader import VerticalExtensionLoader

            VerticalExtensionLoader.clear_extension_cache(clear_all=True)

            # Clear framework integration cache to avoid stale vertical metadata.
            try:
                from victor.framework.vertical_service import (
                    clear_vertical_integration_pipeline_cache,
                )

                clear_vertical_integration_pipeline_cache()
            except Exception as e:
                logger.debug("Failed clearing framework vertical integration cache: %s", e)

            self._plugin_refresh_last_ms = max(
                0.0,
                (time.perf_counter() - refresh_start) * 1000.0,
            )
            self._emit_observability_event(
                topic="vertical.plugins.refreshed",
                data={
                    "refresh_count": self._plugin_refresh_count,
                    "duration_ms": self._plugin_refresh_last_ms,
                    "stats": self.get_discovery_stats(),
                },
            )
            logger.info("Plugin cache cleared, will re-discover on next access")

    def _activate(self, vertical: Type[VerticalBase]) -> None:
        """Activate a vertical.

        Args:
            vertical: Vertical class to activate
        """
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            if self._active_vertical is None:
                return None
            return self._active_vertical.get_config()

    def get_tools(self) -> List[str]:
        """Get tools from active vertical.

        Returns:
            List of tool names
        """
        with self._lock:
            if self._active_vertical is None:
                return []
            return self._active_vertical.get_tools()

    def get_system_prompt(self) -> str:
        """Get system prompt from active vertical.

        Returns:
            System prompt string
        """
        with self._lock:
            if self._active_vertical is None:
                return ""
            return self._active_vertical.get_system_prompt()

    def reset(self) -> None:
        """Reset the loader, deactivating current vertical."""
        with self._lock:
            self._active_vertical = None
            self._extensions = None
            self._registered_services = False

    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get vertical/tool discovery telemetry snapshot."""
        with self._lock:
            return {
                "vertical": {
                    "calls": self._vertical_discovery_calls,
                    "cache_hits": self._vertical_discovery_cache_hits,
                    "scans": self._vertical_discovery_scans,
                    "last_discovery_ms": self._vertical_last_discovery_ms,
                },
                "tools": {
                    "calls": self._tool_discovery_calls,
                    "cache_hits": self._tool_discovery_cache_hits,
                    "scans": self._tool_discovery_scans,
                    "last_discovery_ms": self._tool_last_discovery_ms,
                },
                "refresh": {
                    "count": self._plugin_refresh_count,
                    "last_refresh_ms": self._plugin_refresh_last_ms,
                },
            }


# Global loader instance
_loader: Optional[VerticalLoader] = None
_loader_lock = threading.Lock()


def get_vertical_loader() -> VerticalLoader:
    """Get the global vertical loader instance.

    Returns:
        Global VerticalLoader instance
    """
    global _loader
    if _loader is None:
        with _loader_lock:
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
