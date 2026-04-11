"""Mock test fixtures for external vertical development.

Provides MockPluginContext — an in-memory implementation of PluginContext
that allows testing VictorPlugin.register() without installing victor-ai.

Usage::

    from victor_sdk.testing import MockPluginContext

    def test_my_plugin_registers_tools():
        ctx = MockPluginContext()
        my_plugin.register(ctx)
        assert "my_tool" in ctx.registered_tools
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type


class MockPluginContext:
    """In-memory PluginContext for testing verticals without victor-ai.

    Satisfies the ``PluginContext`` protocol from ``victor_sdk.core.plugins``.
    All registration calls store items in public attributes for assertion.
    """

    def __init__(self) -> None:
        self.registered_tools: List[Any] = []
        self.registered_verticals: List[Type[Any]] = []
        self.registered_commands: Dict[str, Any] = {}
        self.registered_chunkers: List[Any] = []
        self.registered_workflow_node_executors: Dict[str, Any] = {}
        self._services: Dict[Type[Any], Any] = {}
        self._settings: Dict[str, Any] = {
            "default_provider": "mock",
            "default_model": "mock-model",
            "debug": False,
        }

    def register_tool(self, tool_instance: Any) -> None:
        """Register a tool instance."""
        self.registered_tools.append(tool_instance)

    def register_vertical(self, vertical_class: Type[Any]) -> None:
        """Register a vertical class."""
        self.registered_verticals.append(vertical_class)

    def register_chunker(self, chunker_instance: Any) -> None:
        """Register a chunker."""
        self.registered_chunkers.append(chunker_instance)

    def register_command(self, name: str, app: Any) -> None:
        """Register a CLI sub-command."""
        self.registered_commands[name] = app

    def register_workflow_node_executor(
        self,
        node_type: str,
        executor_factory: Any,
        *,
        replace: bool = False,
    ) -> None:
        """Register a custom workflow node executor."""
        self.registered_workflow_node_executors[node_type] = executor_factory

    def get_service(self, service_type: Type[Any]) -> Optional[Any]:
        """Retrieve a registered service, or None."""
        return self._services.get(service_type)

    def get_settings(self) -> Any:
        """Return mock settings dict."""
        return self._settings

    def register_capability(
        self,
        protocol_type: Type[Any],
        provider: Any,
        *,
        lazy: bool = False,
    ) -> None:
        """Register a capability provider (stores in _capabilities for testing)."""
        if not hasattr(self, "_capabilities"):
            self._capabilities: Dict[Type[Any], Any] = {}
        self._capabilities[protocol_type] = provider

    # --- Test helpers (not part of PluginContext protocol) ---

    def set_service(self, service_type: Type[Any], instance: Any) -> None:
        """Pre-register a service for testing."""
        self._services[service_type] = instance
