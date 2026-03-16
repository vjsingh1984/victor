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
        from victor.tools.registry import ToolRegistry

        try:
            registry = ToolRegistry()
            registry.register_tool(tool_instance)
            logger.debug(f"Plugin registered tool: {getattr(tool_instance, 'name', tool_instance)}")
        except Exception as e:
            logger.error(f"Plugin failed to register tool: {e}")

    def register_vertical(self, vertical_class: Type[Any]) -> None:
        """Register a vertical with the framework's VerticalRegistry."""
        from victor.core.verticals.base import VerticalRegistry

        try:
            VerticalRegistry.register(vertical_class)
            logger.debug(f"Plugin registered vertical: {getattr(vertical_class, 'name', vertical_class)}")
        except Exception as e:
            logger.error(f"Plugin failed to register vertical: {e}")

    def register_chunker(self, chunker_instance: Any) -> None:
        """Register a chunker with the framework's ChunkingRegistry."""
        from victor.core.chunking.registry import get_chunking_registry

        try:
            registry = get_chunking_registry()
            registry.register(chunker_instance)
            logger.debug(f"Plugin registered chunker: {getattr(chunker_instance, 'name', chunker_instance)}")
        except Exception as e:
            logger.error(f"Plugin failed to register chunker: {e}")

    def register_command(self, name: str, app: typer.Typer) -> None:
        """Register a command app."""
        self._commands[name] = app
        logger.debug(f"Plugin registered command: {name}")

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
