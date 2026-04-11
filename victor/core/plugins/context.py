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

"""Host implementation of PluginContext for Victor framework.

This module provides the bridge between the SDK-defined PluginContext
and the framework's internal registries and containers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

import typer
from victor_sdk import PluginContext

logger = logging.getLogger(__name__)


class HostPluginContext(PluginContext):
    """Host implementation of PluginContext.

    Wraps the framework's internal registries to provide a clean interface
    for plugins.
    """

    def __init__(self, container: Optional[Any] = None) -> None:
        """Initialize context.

        Args:
            container: The host service container.
        """
        self._container = container
        self._commands: Dict[str, typer.Typer] = {}

    @property
    def commands(self) -> Dict[str, typer.Typer]:
        """Return registered commands."""
        return self._commands

    def register_tool(self, tool_instance: Any) -> None:
        """Register a tool with the framework's ToolRegistry."""
        try:
            registry = None

            if not self._container:
                from victor.core.container import get_container

                self._container = get_container()

            if self._container is not None:
                try:
                    from victor.agent.protocols import ToolRegistryProtocol

                    if hasattr(self._container, "get_optional"):
                        registry = self._container.get_optional(ToolRegistryProtocol)
                    elif hasattr(
                        self._container, "is_registered"
                    ) and self._container.is_registered(ToolRegistryProtocol):
                        registry = self._container.get(ToolRegistryProtocol)
                except Exception as e:
                    logger.debug(
                        "Plugin failed to resolve live ToolRegistry from container: %s",
                        e,
                    )

            if registry is None:
                from victor.tools.registry import ToolRegistry

                registry = ToolRegistry()

            if hasattr(registry, "register_tool"):
                registry.register_tool(tool_instance)
            else:
                registry.register(tool_instance)
            logger.debug(f"Plugin registered tool: {getattr(tool_instance, 'name', tool_instance)}")
        except Exception as e:
            logger.error(f"Plugin failed to register tool: {e}")

    def register_vertical(self, vertical_class: Type[Any]) -> None:
        """Register a vertical and auto-extract its capabilities.

        In addition to placing the class in VerticalRegistry, this method
        checks if the vertical declares capabilities via
        get_capability_registrations() and registers them with the
        CapabilityRegistry. This makes the plugin system the single
        authority for capability registration.
        """
        from victor.core.verticals.base import VerticalRegistry

        try:
            VerticalRegistry.register(vertical_class)
            logger.debug(
                "Plugin registered vertical: %s",
                getattr(vertical_class, "name", vertical_class),
            )
        except Exception as e:
            logger.error("Plugin failed to register vertical: %s", e)
            return

        # Auto-extract capabilities from the vertical
        self._auto_register_vertical_capabilities(vertical_class)

    def _auto_register_vertical_capabilities(self, vertical_class: Type[Any]) -> None:
        """Discover and register capabilities declared by the vertical.

        Strategy 1: Check for get_capability_registrations() class method
        that returns (protocol_type, provider) pairs.

        Strategy 2: Scan entry points in victor.sdk.capabilities group
        for entries matching the vertical name.
        """
        vertical_name = getattr(vertical_class, "name", "")
        if not vertical_name and hasattr(vertical_class, "get_name"):
            try:
                vertical_name = vertical_class.get_name()
            except Exception:
                pass

        # Strategy 1: Explicit capability registrations
        cap_method = getattr(vertical_class, "get_capability_registrations", None)
        if cap_method and callable(cap_method):
            try:
                registrations = cap_method()
                for protocol_type, provider in registrations:
                    self.register_capability(protocol_type, provider)
                if registrations:
                    logger.debug(
                        "Auto-registered %d capabilities from %s.get_capability_registrations()",
                        len(registrations),
                        vertical_name,
                    )
            except Exception as e:
                logger.debug(
                    "Capability extraction failed for %s: %s", vertical_name, e
                )

        # Strategy 2: Entry point scanning (deferred — runs if vertical has EP registrations)
        if vertical_name:
            try:
                from victor.framework.entry_point_registry import (
                    get_entry_point_registry,
                )

                ep_registry = get_entry_point_registry()
                for group in ("victor.capabilities", "victor.sdk.capabilities"):
                    group_obj = ep_registry.get_group(group)
                    if not group_obj:
                        continue
                    for ep_name, entry in group_obj.entry_points.items():
                        if ep_name.startswith(f"{vertical_name}-"):
                            try:
                                ep, loaded = entry[0], entry[1]
                                if not loaded:
                                    register_func = ep.load()
                                else:
                                    register_func = entry[2] if len(entry) > 2 else ep.load()
                                from victor.core.capability_registry import (
                                    CapabilityRegistry,
                                )

                                register_func(CapabilityRegistry.get_instance())
                                logger.debug(
                                    "Vertical %s registered EP capability: %s",
                                    vertical_name,
                                    ep_name,
                                )
                            except Exception as e:
                                logger.debug(
                                    "Skipped EP capability %s:%s: %s",
                                    group,
                                    ep_name,
                                    e,
                                )
            except Exception as e:
                logger.debug(
                    "Entry point discovery failed for %s: %s", vertical_name, e
                )

    def register_capability(
        self,
        protocol_type: Type[Any],
        provider: Any,
        *,
        lazy: bool = False,
    ) -> None:
        """Register a capability provider with CapabilityRegistry.

        This is the canonical way for plugins to register capabilities.
        Enhanced providers won't be downgraded by subsequent STUB registrations.
        """
        from victor.core.capability_registry import CapabilityRegistry, CapabilityStatus

        registry = CapabilityRegistry.get_instance()
        if lazy and callable(provider):
            proxy = _LazyCapabilityProxy(provider)
            registry.register(protocol_type, proxy, CapabilityStatus.ENHANCED)
        else:
            registry.register(protocol_type, provider, CapabilityStatus.ENHANCED)
        logger.debug(
            "Plugin registered capability: %s (lazy=%s)",
            getattr(protocol_type, "__name__", protocol_type),
            lazy,
        )

    def register_chunker(self, chunker_instance: Any) -> None:
        """Register a chunker with the framework's ChunkingRegistry."""
        from victor.core.chunking.registry import get_chunking_registry

        try:
            registry = get_chunking_registry()
            registry.register(chunker_instance)
            logger.debug(
                f"Plugin registered chunker: {getattr(chunker_instance, 'name', chunker_instance)}"
            )
        except Exception as e:
            logger.error(f"Plugin failed to register chunker: {e}")

    def register_command(self, name: str, app: typer.Typer) -> None:
        """Register a command app."""
        self._commands[name] = app
        logger.debug(f"Plugin registered command: {name}")

    def register_workflow_node_executor(
        self,
        node_type: str,
        executor_factory: Any,
        *,
        replace: bool = False,
    ) -> None:
        """Register a custom workflow node executor."""
        from victor.workflows.executors.registry import register_workflow_node_executor

        register_workflow_node_executor(
            node_type,
            executor_factory,
            replace=replace,
        )

        if self._container is not None:
            try:
                from victor.workflows.compiler_protocols import (
                    NodeExecutorFactoryProtocol,
                )

                if hasattr(self._container, "is_registered") and self._container.is_registered(
                    NodeExecutorFactoryProtocol
                ):
                    factory = self._container.get_optional(NodeExecutorFactoryProtocol)
                    if factory is not None:
                        existing = getattr(factory, "_executor_types", {}).get(node_type)
                        if existing is not executor_factory:
                            factory.register_executor_type(
                                node_type,
                                executor_factory,
                                replace=replace,
                            )
            except Exception as e:
                logger.debug(f"Plugin deferred live workflow executor registration: {e}")

        logger.debug("Plugin registered workflow node executor: %s", node_type)

    def get_service(self, service_type: Type[Any]) -> Optional[Any]:
        """Retrieve a service from the container."""
        if not self._container:
            from victor.core.container import get_container

            self._container = get_container()

        try:
            return self._container.get(service_type)
        except Exception:
            return None

    def get_settings(self) -> Any:
        """Retrieve the application settings from the container."""
        from victor.config.settings import Settings

        if not self._container:
            from victor.core.container import get_container

            self._container = get_container()

        try:
            return self._container.get(Settings)
        except Exception:
            # Fallback: load default settings
            return Settings()


class _LazyCapabilityProxy:
    """Proxy that defers provider instantiation until first attribute access.

    Used by register_capability(lazy=True) to avoid importing heavy
    dependencies at plugin registration time.
    """

    def __init__(self, factory):
        object.__setattr__(self, "_factory", factory)
        object.__setattr__(self, "_instance", None)
        object.__setattr__(self, "_resolved", False)

    def __getattr__(self, name):
        if not object.__getattribute__(self, "_resolved"):
            factory = object.__getattribute__(self, "_factory")
            instance = factory()
            object.__setattr__(self, "_instance", instance)
            object.__setattr__(self, "_resolved", True)
        return getattr(object.__getattribute__(self, "_instance"), name)
