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

"""Plugin protocols for Victor SDK.

This module defines the interfaces that external plugins and verticals
implement to register themselves with the Victor framework.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, Type, runtime_checkable

if TYPE_CHECKING:
    import typer


@runtime_checkable
class PluginContext(Protocol):
    """Context provided to plugins during registration.

    This interface allows plugins to register tools, verticals, chunkers,
    and CLI commands without depending on framework internal registries.
    """

    def register_tool(self, tool_instance: Any) -> None:
        """Register a tool instance with the framework.

        Args:
            tool_instance: The tool to register (e.g., a function with @tool).
        """
        ...

    def register_vertical(self, vertical_class: Type[Any]) -> None:
        """Register a vertical class with the framework.

        Args:
            vertical_class: The class implementing VerticalBase.
        """
        ...

    def register_chunker(self, chunker_instance: Any) -> None:
        """Register a specialized chunker for document processing.

        Args:
            chunker_instance: The chunker implementation.
        """
        ...

    def register_command(self, name: str, app: "typer.Typer") -> None:
        """Register a Typer CLI application as a sub-command.

        Args:
            name: The sub-command name.
            app: The Typer application to mount.
        """
        ...

    def register_workflow_node_executor(
        self,
        node_type: str,
        executor_factory: Any,
        *,
        replace: bool = False,
    ) -> None:
        """Register a custom workflow node executor type with the host.

        Args:
            node_type: Custom workflow node type identifier.
            executor_factory: Executor class or factory callable for the node type.
            replace: Whether to replace an existing registration.
        """
        ...

    def get_service(self, service_type: Type[Any]) -> Optional[Any]:
        """Retrieve a service from the host container.

        Args:
            service_type: The type or interface of the service.

        Returns:
            The service instance or None if not found.
        """
        ...

    def get_settings(self) -> Any:
        """Retrieve the application settings.

        Returns the framework Settings object. Plugins should use this
        instead of importing victor.config.settings directly.

        Returns:
            The Settings instance.
        """
        ...


@runtime_checkable
class VictorPlugin(Protocol):
    """Interface for Victor plugins.

    Plugins are discovered via the 'victor.plugins' entry point.
    """

    @property
    def name(self) -> str:
        """Return the stable identifier for this plugin (e.g., 'coding')."""
        ...

    def register(self, context: PluginContext) -> None:
        """Register the plugin's components with the host framework.

        Args:
            context: The registration context provided by the host.
        """
        ...

    def get_cli_app(self) -> Optional["typer.Typer"]:
        """Return a Typer application to be registered as a sub-command.

        .. deprecated::
            Use context.register_command() inside register() instead.
        """
        ...

    def on_activate(self) -> None:
        """Called when this plugin's vertical is activated.

        Optional lifecycle hook. Implement to perform setup when the
        vertical associated with this plugin becomes active.
        """
        ...

    def on_deactivate(self) -> None:
        """Called when this plugin's vertical is being deactivated.

        Optional lifecycle hook. Implement to perform cleanup when
        switching away from this plugin's vertical.
        """
        ...

    def health_check(self) -> Dict[str, Any]:
        """Return health status for this plugin.

        Optional lifecycle hook. Returns a dictionary with at minimum
        a 'healthy' boolean key.

        Returns:
            Dict with 'healthy' key and optional detail keys.
        """
        ...
