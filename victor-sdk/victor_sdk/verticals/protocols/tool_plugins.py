"""Tool plugin protocols for dynamic tool registration.

This module provides protocols and helper classes for registering tools
dynamically via plugins, complementing the existing VictorPlugin interface.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class ToolFactory(Protocol):
    """Protocol for dynamically creating tool instances.

    A ToolFactory is a callable that creates a tool instance on demand.
    This is useful for tools that require runtime configuration or
    lazy initialization.
    """

    def __call__(self) -> Any:
        """Create and return a tool instance.

        Returns:
            A tool instance (e.g., a function decorated with @tool)
        """
        ...

    @property
    def name(self) -> str:
        """Return the tool name for registration."""
        ...


@runtime_checkable
class ToolFactoryPlugin(Protocol):
    """Protocol for plugins that provide tool factories.

    Unlike VictorPlugin which receives a PluginContext, a ToolFactoryPlugin
    directly provides tool factories for registration.
    """

    def get_tool_factories(self) -> Dict[str, ToolFactory]:
        """Return dictionary of tool name to factory.

        Returns:
            Dict mapping tool names to ToolFactory callables
        """
        ...

    def get_tool_instances(self) -> Dict[str, Any]:
        """Return dictionary of tool name to tool instance.

        Returns:
            Dict mapping tool names to tool instances
        """
        ...


class ToolFactoryAdapter:
    """Adapter that converts a ToolFactory into a VictorPlugin.

    This allows tool factories to be registered via the standard
    victor.plugins entry point mechanism.
    """

    def __init__(
        self,
        factories: Optional[Dict[str, ToolFactory]] = None,
        instances: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the adapter with factories and/or instances.

        Args:
            factories: Dict of tool name to ToolFactory
            instances: Dict of tool name to tool instance
        """
        self._factories = factories or {}
        self._instances = instances or {}

    def register(self, context: Any) -> None:
        """Register tools with the provided context.

        Args:
            context: PluginContext with register_tool method
        """
        # Register all instances
        for tool_name, tool_instance in self._instances.items():
            context.register_tool(tool_instance)

        # Register all factories (create instances)
        for tool_name, factory in self._factories.items():
            tool_instance = factory()
            context.register_tool(tool_instance)


class ToolPluginHelper:
    """Helper class for creating tool plugins.

    Provides convenience methods for building tool plugins
    from tool instances or factories.
    """

    @staticmethod
    def from_instances(tools: Dict[str, Any]) -> ToolFactoryAdapter:
        """Create a plugin adapter from tool instances.

        Args:
            tools: Dict mapping tool names to tool instances

        Returns:
            ToolFactoryAdapter that registers the tools
        """
        return ToolFactoryAdapter(instances=tools)

    @staticmethod
    def from_factories(factories: Dict[str, ToolFactory]) -> ToolFactoryAdapter:
        """Create a plugin adapter from tool factories.

        Args:
            factories: Dict mapping tool names to ToolFactory callables

        Returns:
            ToolFactoryAdapter that creates and registers tools
        """
        return ToolFactoryAdapter(factories=factories)

    @staticmethod
    def from_module(
        module: Any,
        tool_attribute: str = "tool",
    ) -> ToolFactoryAdapter:
        """Create a plugin adapter from a module.

        Scans module for objects with the specified tool_attribute
        and creates a plugin adapter.

        Args:
            module: Python module to scan
            tool_attribute: Attribute name that identifies tools (default: "tool")

        Returns:
            ToolFactoryAdapter with discovered tools
        """
        tools = {}
        for attr_name in dir(module):
            if not attr_name.startswith("_"):
                attr = getattr(module, attr_name, None)
                if attr is not None and hasattr(attr, tool_attribute):
                    tools[attr_name] = attr
        return ToolFactoryAdapter(instances=tools)


__all__ = [
    "ToolFactory",
    "ToolFactoryPlugin",
    "ToolFactoryAdapter",
    "ToolPluginHelper",
]
