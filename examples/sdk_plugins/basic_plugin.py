#!/usr/bin/env python3
"""Basic Victor SDK plugin example.

This example demonstrates the simplest plugin implementation using
the new VictorPlugin protocol from victor_sdk.core.plugins.

Key concepts:
- VictorPlugin protocol
- PluginContext for tool registration
- Entry point configuration
"""

from typing import Any, Dict

from victor_sdk.core.plugins import VictorPlugin, PluginContext
from victor.tools.base import BaseTool, ToolMetadata, ToolResult
from victor.tools.enums import AccessMode, CostTier, DangerLevel


# =============================================================================
# Example Tool
# =============================================================================

class HelloWorldTool(BaseTool):
    """A simple hello world tool for demonstration."""

    def __init__(self):
        """Initialize the tool."""
        super().__init__(
            name="hello_world",
            description="Greets the user with a hello world message",
            metadata=ToolMetadata(
                name="hello_world",
                description="Greets the user with a hello world message",
                cost_tier=CostTier.FREE,
                danger_level=DangerLevel.LOW,
                access_mode=AccessMode.READ,
            ),
        )

    @property
    def name(self) -> str:
        return "hello_world"

    @property
    def description(self) -> str:
        return "Greets the user with a hello world message"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name to greet",
                },
            },
            "required": [],
        }

    async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
        """Execute the hello world tool."""
        name = kwargs.get("name", "World")
        message = f"Hello, {name}!"

        return ToolResult(
            success=True,
            output={"message": message},
            error=None,
        )


# =============================================================================
# Plugin Implementation
# =============================================================================

class BasicPlugin(VictorPlugin):
    """Basic plugin demonstrating VictorPlugin protocol.

    This is the simplest possible plugin implementation:
    1. Implement VictorPlugin protocol
    2. Provide name property
    3. Implement register() method
    4. Use PluginContext to register tools
    """

    @property
    def name(self) -> str:
        """Return stable plugin identifier."""
        return "basic_plugin"

    def register(self, context: PluginContext) -> None:
        """
        Register plugin components with the host framework.

        This method is called during plugin discovery. Use the context
        to register tools, verticals, CLI commands, etc.

        Args:
            context: PluginContext for registering components
        """
        # Register our tool directly
        context.register_tool(HelloWorldTool())

        print(f"[{self.name}] Registered hello_world tool")


# =============================================================================
# Entry Point
# =============================================================================

def plugin() -> VictorPlugin:
    """
    Entry point for plugin discovery.

    This function is called by the victor.plugins entry point.

    Returns:
        VictorPlugin instance
    """
    return BasicPlugin()


# =============================================================================
# Demo / Testing
# =============================================================================

async def demo():
    """Demonstrate the plugin."""
    from victor.tools.registry import ToolRegistry

    print("=" * 60)
    print("Basic SDK Plugin Demo")
    print("=" * 60)

    # Create registry and register plugin
    registry = ToolRegistry()
    plugin_instance = plugin()

    print(f"\nPlugin Name: {plugin_instance.name}")
    print("\nRegistering plugin...")
    registry.register_plugin(plugin_instance)

    print("\nAvailable Tools:")
    for tool_name in registry.list_tools():
        print(f"  - {tool_name}")

    # Execute the tool
    print("\nExecuting hello_world tool:")
    tool = registry.get("hello_world")
    if tool:
        result = await tool.execute({}, name="Victor Developer")
        if result.success:
            print(f"  Output: {result.output['message']}")
        else:
            print(f"  Error: {result.error}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import asyncio

    print("""
╔══════════════════════════════════════════════════════════════╗
║            Victor SDK Plugin System - Basic Example          ║
╠══════════════════════════════════════════════════════════════╣
║  This example demonstrates:                                  ║
║  1. VictorPlugin protocol implementation                     ║
║  2. PluginContext for tool registration                      ║
║  3. Entry point function                                     ║
║  4. Basic tool execution                                     ║
╚══════════════════════════════════════════════════════════════╝
""")

    asyncio.run(demo())
