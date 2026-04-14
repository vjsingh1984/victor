#!/usr/bin/env python3
"""Complete Victor SDK plugin with all features.

This example demonstrates a production-ready plugin with:
- Multiple tool registration patterns
- Plugin lifecycle hooks (on_activate, on_deactivate)
- Health check implementation
- Configuration handling
- Async lifecycle methods
- Error handling

Key concepts:
- VictorPlugin protocol (complete implementation)
- PluginContext usage
- Lifecycle hooks
- Health checks
- ToolPluginHelper patterns
"""

import asyncio
from typing import Any, Dict, Optional

from victor_sdk.core.plugins import VictorPlugin, PluginContext
from victor_sdk.verticals.protocols import ToolPluginHelper
from victor.tools.base import BaseTool, ToolMetadata, ToolResult
from victor.tools.enums import AccessMode, CostTier, DangerLevel


# =============================================================================
# Example Tools
# =============================================================================

class ConfigurableTool(BaseTool):
    """Tool that uses plugin configuration."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration."""
        super().__init__(
            name="configurable_tool",
            description="Tool with plugin-based configuration",
            metadata=ToolMetadata(
                name="configurable_tool",
                description="Tool with plugin-based configuration",
                cost_tier=CostTier.LOW,
                danger_level=DangerLevel.LOW,
                access_mode=AccessMode.READ,
            ),
        )
        self._config = config

    @property
    def name(self) -> str:
        return "configurable_tool"

    @property
    def description(self) -> str:
        return "Tool with plugin-based configuration"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Input to process",
                },
            },
            "required": ["input"],
        }

    async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
        """Execute with configuration."""
        input_val = kwargs.get("input", "")

        # Use plugin configuration
        prefix = self._config.get("output_prefix", "Result:")

        return ToolResult(
            success=True,
            output={"result": f"{prefix} {input_val}"},
            error=None,
        )


class StatefulTool(BaseTool):
    """Tool that maintains state across calls."""

    def __init__(self):
        """Initialize stateful tool."""
        super().__init__(
            name="stateful_tool",
            description="Tool that maintains call state",
            metadata=ToolMetadata(
                name="stateful_tool",
                description="Tool that maintains call state",
                cost_tier=CostTier.FREE,
                danger_level=DangerLevel.LOW,
                access_mode=AccessMode.READ,
            ),
        )
        self._call_count = 0
        self._history = []

    @property
    def name(self) -> str:
        return "stateful_tool"

    @property
    def description(self) -> str:
        return "Tool that maintains call state"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform",
                },
            },
            "required": ["action"],
        }

    async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
        """Execute with state tracking."""
        action = kwargs.get("action", "")
        self._call_count += 1
        self._history.append(action)

        return ToolResult(
            success=True,
            output={
                "action": action,
                "call_count": self._call_count,
                "history": self._history.copy(),
            },
            error=None,
        )


# =============================================================================
# Plugin Implementation
# =============================================================================

class CompletePlugin(VictorPlugin):
    """
    Complete production-ready plugin implementation.

    Features demonstrated:
    1. Multiple tool registration patterns
    2. Plugin lifecycle hooks (sync and async)
    3. Health check implementation
    4. Configuration handling
    5. State management
    6. Error handling
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize plugin with optional configuration.

        Args:
            config: Plugin configuration dictionary
        """
        self._config = config or {}
        self._activated = False
        self._activation_count = 0
        self._stateful_tool: Optional[StatefulTool] = None

    @property
    def name(self) -> str:
        """Return stable plugin identifier."""
        return "complete_plugin"

    def register(self, context: PluginContext) -> None:
        """
        Register plugin components with the host framework.

        This method demonstrates multiple registration patterns:
        1. Direct tool registration
        2. ToolPluginHelper.from_instances()
        3. Conditional registration based on config
        """
        from victor_sdk.verticals.protocols import ToolPluginHelper

        print(f"\n[{self.name}] Registering plugin components...")

        # --------------------------------------------------------------------
        # Pattern 1: Direct tool registration
        # --------------------------------------------------------------------
        self._stateful_tool = StatefulTool()
        context.register_tool(self._stateful_tool)
        print(f"  [{self.name}] Registered stateful_tool")

        # --------------------------------------------------------------------
        # Pattern 2: ToolPluginHelper.from_instances()
        # --------------------------------------------------------------------
        configurable = ConfigurableTool(self._config)
        helper = ToolPluginHelper.from_instances({
            "configurable": configurable,
        })
        helper.register(context)
        print(f"  [{self.name}] Registered configurable_tool")

        # --------------------------------------------------------------------
        # Pattern 3: Conditional registration
        # --------------------------------------------------------------------
        if self._config.get("enable_experimental", False):
            experimental_tool = StatefulTool()
            experimental_tool._name = "experimental_tool"
            context.register_tool(experimental_tool)
            print(f"  [{self.name}] Registered experimental_tool (experimental)")

    def get_cli_app(self) -> Optional[Any]:
        """
        Return Typer app for CLI commands (optional).

        Note: Deprecated - use context.register_command() in register() instead.
        """
        return None

    # ========================================================================
    # Lifecycle Hooks (Sync)
    # ========================================================================

    def on_activate(self) -> None:
        """
        Called when plugin's vertical is activated (sync variant).

        Use this for:
        - Initializing resources
        - Loading configuration
        - Setup that doesn't require I/O
        """
        self._activated = True
        self._activation_count += 1

        print(f"\n[{self.name}] Activated (activation #{self._activation_count})")
        print(f"  [{self.name}] Config: {self._config}")
        print(f"  [{self.name}] Resources initialized")

    def on_deactivate(self) -> None:
        """
        Called when plugin's vertical is deactivated (sync variant).

        Use this for:
        - Releasing resources
        - Saving state
        - Cleanup that doesn't require I/O
        """
        self._activated = False

        print(f"\n[{self.name}] Deactivated")
        print(f"  [{self.name}] Total activations: {self._activation_count}")

        if self._stateful_tool:
            print(f"  [{self.name}] Tool call count: {self._stateful_tool._call_count}")

    # ========================================================================
    # Lifecycle Hooks (Async)
    # ========================================================================

    async def on_activate_async(self) -> None:
        """
        Called when plugin's vertical is activated (async variant).

        When implemented, this is called instead of on_activate()
        in async contexts. Use this for I/O operations.

        Use this for:
        - Async database connections
        - Async HTTP client initialization
        - Async resource loading
        """
        # Simulate async initialization
        await asyncio.sleep(0.01)

        self._activated = True
        self._activation_count += 1

        print(f"\n[{self.name}] Activated (async, activation #{self._activation_count})")
        print(f"  [{self.name}] Async resources initialized")

    async def on_deactivate_async(self) -> None:
        """
        Called when plugin's vertical is deactivated (async variant).

        When implemented, this is called instead of on_deactivate()
        in async contexts. Use this for I/O operations.

        Use this for:
        - Async connection cleanup
        - Async state saving
        - Async resource release
        """
        # Simulate async cleanup
        await asyncio.sleep(0.01)

        self._activated = False

        print(f"\n[{self.name}] Deactivated (async)")
        print(f"  [{self.name}] Async resources released")

    # ========================================================================
    # Health Check
    # ========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Return plugin health status.

        Returns:
            Dict with 'healthy' key and optional detail keys

        Example:
            {
                "healthy": True,
                "version": "1.0.0",
                "activated": True,
                "tools_registered": 2,
            }
        """
        return {
            "healthy": True,
            "version": "1.0.0",
            "activated": self._activated,
            "activation_count": self._activation_count,
            "config": self._config,
            "tools_registered": 2,
        }


# =============================================================================
# Entry Point
# =============================================================================

def plugin(config: Optional[Dict[str, Any]] = None) -> VictorPlugin:
    """
    Entry point for plugin discovery.

    Args:
        config: Optional plugin configuration

    Returns:
        VictorPlugin instance
    """
    return CompletePlugin(config)


# =============================================================================
# Demo / Testing
# =============================================================================

async def demo():
    """Demonstrate complete plugin with all features."""
    from victor.tools.registry import ToolRegistry

    print("=" * 70)
    print("Complete SDK Plugin Demo - All Features")
    print("=" * 70)

    # Create plugin with configuration
    config = {
        "output_prefix": "[PROCESSED]",
        "enable_experimental": False,
    }

    plugin_instance = plugin(config)

    print(f"\nPlugin Name: {plugin_instance.name}")
    print(f"Plugin Config: {config}")

    # Test lifecycle hooks
    print("\n" + "-" * 70)
    print("Testing Lifecycle Hooks")
    print("-" * 70)

    print("\n1. Calling on_activate()...")
    plugin_instance.on_activate()

    print("\n2. Checking health...")
    health = plugin_instance.health_check()
    print(f"   Healthy: {health['healthy']}")
    print(f"   Activated: {health['activated']}")
    print(f"   Tools: {health['tools_registered']}")

    # Register with ToolRegistry
    print("\n" + "-" * 70)
    print("Registering with ToolRegistry")
    print("-" * 70)

    registry = ToolRegistry()
    registry.register_plugin(plugin_instance)

    print("\nAvailable Tools:")
    for tool_name in registry.list_tools():
        print(f"  - {tool_name}")

    # Execute tools
    print("\n" + "-" * 70)
    print("Executing Tools")
    print("-" * 70)

    # Test stateful tool
    print("\n1. Stateful Tool (maintains call count):")
    stateful = registry.get("stateful_tool")
    if stateful:
        for i in range(3):
            result = await stateful.execute({}, action=f"action_{i+1}")
            print(f"   Call {i+1}: {result.output}")

    # Test configurable tool
    print("\n2. Configurable Tool (uses plugin config):")
    configurable = registry.get("configurable_tool")
    if configurable:
        result = await configurable.execute({}, input="test input")
        print(f"   Result: {result.output['result']}")

    # Test async lifecycle
    print("\n" + "-" * 70)
    print("Testing Async Lifecycle Hooks")
    print("-" * 70)

    print("\n3. Calling on_activate_async()...")
    await plugin_instance.on_activate_async()

    print("\n4. Calling on_deactivate_async()...")
    await plugin_instance.on_deactivate_async()

    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          Victor SDK Plugin - Complete Example                     ║
╠══════════════════════════════════════════════════════════════════╣
║  This example demonstrates:                                       ║
║  1. VictorPlugin protocol (complete implementation)               ║
║  2. Multiple tool registration patterns                           ║
║  3. Plugin lifecycle hooks (on_activate, on_deactivate)           ║
║  4. Async lifecycle methods (on_activate_async, etc.)             ║
║  5. Health check implementation                                   ║
║  6. Configuration handling                                        ║
║  7. State management                                              ║
╚══════════════════════════════════════════════════════════════════╝
""")

    asyncio.run(demo())
