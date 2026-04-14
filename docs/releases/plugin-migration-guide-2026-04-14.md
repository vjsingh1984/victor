# Victor Plugin Migration Guide

**Version**: 0.7.0
**Date**: 2026-04-14
**Audience**: External Vertical Developers

---

## Overview

This guide helps you migrate your external Victor vertical to use the **new SDK-based plugin system**. The new system provides:

- ✅ **Better separation** - SDK protocols in `victor_sdk`, not core framework
- ✅ **Lazy loading** - Tools created only when needed (via `ToolFactory`)
- ✅ **Lifecycle hooks** - `on_activate()`, `on_deactivate()`, health checks
- ✅ **Type safety** - Protocol-based with `@runtime_checkable`
- ✅ **Auto-discovery** - Entry point based plugin loading

---

## Quick Start

### Before (Old System - Deprecated)

```python
# OLD: victor.tools.plugin.ToolPlugin (DEPRECATED)
from victor.tools.plugin import ToolPlugin

class MyVerticalPlugin(ToolPlugin):
    name = "my_vertical"
    version = "1.0.0"

    def get_tools(self):
        return [Tool1(), Tool2()]
```

### After (New System - Recommended)

```python
# NEW: victor_sdk.core.plugins.VictorPlugin (RECOMMENDED)
from victor_sdk.core.plugins import VictorPlugin, PluginContext

class MyVerticalPlugin(VictorPlugin):
    @property
    def name(self) -> str:
        return "my_vertical"

    def register(self, context: PluginContext) -> None:
        # Register your tools
        context.register_tool(Tool1())
        context.register_tool(Tool2())
```

---

## Migration Steps

### Step 1: Update Imports

**Old imports**:
```python
from victor.tools.plugin import ToolPlugin
from victor.tools.base import BaseTool
```

**New imports**:
```python
from victor_sdk.core.plugins import VictorPlugin, PluginContext
from victor.tools.base import BaseTool
```

### Step 2: Implement VictorPlugin Protocol

**Old pattern**:
```python
class MyPlugin(ToolPlugin):
    name = "my_plugin"
    version = "1.0.0"
    description = "My plugin"

    def get_tools(self):
        return [MyTool()]
```

**New pattern**:
```python
class MyPlugin(VictorPlugin):
    @property
    def name(self) -> str:
        return "my_plugin"

    def register(self, context: PluginContext) -> None:
        context.register_tool(MyTool())
```

### Step 3: Update Entry Points

**Old entry point** (deprecated):
```toml
[project.entry-points."victor.plugins"]
my_vertical = "my_vertical.plugin:MyVerticalPlugin
```

**New entry point** (same, but plugin class changed):
```toml
[project.entry-points."victor.plugins"]
my_vertical = "my_vertical.plugin:plugin"
```

Add entry point function:
```python
def plugin() -> VictorPlugin:
    """Entry point for plugin discovery."""
    return MyPlugin()
```

---

## Common Migration Patterns

### Pattern 1: Simple Tool List

**Before**:
```python
class SimplePlugin(ToolPlugin):
    name = "simple"

    def get_tools(self):
        return [
            SearchTool(),
            EditTool(),
            RefactorTool(),
        ]
```

**After**:
```python
class SimplePlugin(VictorPlugin):
    @property
    def name(self) -> str:
        return "simple"

    def register(self, context: PluginContext) -> None:
        # Option A: Direct registration
        context.register_tool(SearchTool())
        context.register_tool(EditTool())
        context.register_tool(RefactorTool())

        # Option B: Use ToolPluginHelper
        from victor_sdk.verticals.protocols import ToolPluginHelper

        tools = {
            "search": SearchTool(),
            "edit": EditTool(),
            "refactor": RefactorTool(),
        }
        helper = ToolPluginHelper.from_instances(tools)
        helper.register(context)
```

### Pattern 2: Conditional Tool Registration

**Before**:
```python
class ConditionalPlugin(ToolPlugin):
    name = "conditional"

    def __init__(self, config=None):
        self.config = config or {}

    def get_tools(self):
        tools = [BasicTool()]

        if self.config.get("enable_advanced"):
            tools.append(AdvancedTool())

        return tools
```

**After**:
```python
class ConditionalPlugin(VictorPlugin):
    def __init__(self, config=None):
        self.config = config or {}

    @property
    def name(self) -> str:
        return "conditional"

    def register(self, context: PluginContext) -> None:
        # Always register basic tools
        context.register_tool(BasicTool())

        # Conditionally register advanced tools
        if self.config.get("enable_advanced"):
            context.register_tool(AdvancedTool())
```

### Pattern 3: Expensive Tool Initialization

**Before**:
```python
class ExpensivePlugin(ToolPlugin):
    name = "expensive"

    def get_tools(self):
        # Tools created immediately, even if not used
        return [
            HeavyModelTool(),  # Expensive!
            DatabaseTool(),    # Expensive!
        ]
```

**After** (with lazy loading):
```python
from victor_sdk.verticals.protocols import ToolFactory, ToolPluginHelper

class ModelToolFactory(ToolFactory):
    """Factory for lazy tool creation."""

    def __init__(self, model_path):
        self._model_path = model_path

    def __call__(self):
        # Tool created only when needed
        return HeavyModelTool(model_path=self._model_path)

    @property
    def name(self) -> str:
        return "heavy_model"

class ExpensivePlugin(VictorPlugin):
    @property
    def name(self) -> str:
        return "expensive"

    def register(self, context: PluginContext) -> None:
        # Use factories for expensive tools
        factories = {
            "heavy_model": ModelToolFactory("/models/heavy.pkl"),
            "database": DatabaseToolFactory(),
        }
        helper = ToolPluginHelper.from_factories(factories)
        helper.register(context)
```

### Pattern 4: Plugin with Configuration

**Before**:
```python
class ConfigurablePlugin(ToolPlugin):
    name = "configurable"

    def __init__(self, config=None):
        super().__init__(config)
        self.settings = config or {}

    def get_tools(self):
        return [ToolWithConfig(self.settings)]
```

**After**:
```python
class ConfigurablePlugin(VictorPlugin):
    def __init__(self, config=None):
        self.config = config or {}

    @property
    def name(self) -> str:
        return "configurable"

    def register(self, context: PluginContext) -> None:
        # Pass config to tools
        tool = ToolWithConfig(self.config)
        context.register_tool(tool)
```

---

## Advanced Features

### Lifecycle Hooks

The new system provides lifecycle hooks for setup/teardown:

```python
class MyPlugin(VictorPlugin):
    @property
    def name(self) -> str:
        return "my_plugin"

    def register(self, context: PluginContext) -> None:
        context.register_tool(MyTool())

    def on_activate(self) -> None:
        """Called when plugin's vertical is activated."""
        print(f"{self.name} activated!")
        # Initialize resources
        # Load configuration
        # Establish connections

    def on_deactivate(self) -> None:
        """Called when plugin's vertical is deactivated."""
        print(f"{self.name} deactivated!")
        # Release resources
        # Save state
        # Close connections

    async def on_activate_async(self) -> None:
        """Async variant for I/O operations."""
        # Async database connections
        # Async HTTP client initialization
        await asyncio.sleep(0)

    async def on_deactivate_async(self) -> None:
        """Async variant for I/O cleanup."""
        # Async connection cleanup
        # Async state saving
        await asyncio.sleep(0)
```

### Health Checks

Implement health checks for monitoring:

```python
class MyPlugin(VictorPlugin):
    def __init__(self):
        self._healthy = True
        self._error_count = 0

    @property
    def name(self) -> str:
        return "my_plugin"

    def register(self, context: PluginContext) -> None:
        context.register_tool(MyTool())

    def health_check(self) -> Dict[str, Any]:
        """Return plugin health status."""
        return {
            "healthy": self._healthy,
            "version": "1.0.0",
            "error_count": self._error_count,
            "last_check": datetime.now().isoformat(),
        }
```

---

## Complete Example

### Full Migration Example

**File**: `my_vertical/plugin.py`

```python
"""My Vertical Plugin for Victor."""

from typing import Any, Dict, Optional
from victor_sdk.core.plugins import VictorPlugin, PluginContext
from victor_sdk.verticals.protocols import ToolPluginHelper
from victor.tools.base import BaseTool


class MyVerticalPlugin(VictorPlugin):
    """My vertical plugin using new SDK system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._activated = False

    @property
    def name(self) -> str:
        return "my_vertical"

    def register(self, context: PluginContext) -> None:
        """Register tools with Victor."""
        # Register simple tools directly
        from my_vertical.tools import SearchTool, EditTool

        context.register_tool(SearchTool())
        context.register_tool(EditTool())

        # Use helper for multiple tools
        tools = {
            "refactor": RefactorTool(),
            "analyze": AnalyzeTool(),
        }
        helper = ToolPluginHelper.from_instances(tools)
        helper.register(context)

    def on_activate(self) -> None:
        """Setup when vertical is activated."""
        self._activated = True
        print(f"[{self.name}] Activated with config: {self.config}")

    def on_deactivate(self) -> None:
        """Cleanup when vertical is deactivated."""
        self._activated = False
        print(f"[{self.name}] Deactivated")

    def health_check(self) -> Dict[str, Any]:
        """Return health status."""
        return {
            "healthy": True,
            "activated": self._activated,
            "version": "1.0.0",
        }


def plugin() -> VictorPlugin:
    """Entry point for plugin discovery."""
    return MyVerticalPlugin()
```

**File**: `pyproject.toml`

```toml
[project.entry-points."victor.plugins"]
my_vertical = "my_vertical.plugin:plugin"
```

---

## Testing Your Plugin

### Unit Tests

```python
import pytest
from victor.tools.registry import ToolRegistry
from my_vertical.plugin import plugin

def test_plugin_registration():
    """Test plugin registers tools correctly."""
    registry = ToolRegistry()
    plugin_instance = plugin()

    # Register plugin
    registry.register_plugin(plugin_instance)

    # Verify tools registered
    assert "search" in registry
    assert "edit" in registry
    assert "refactor" in registry

def test_plugin_lifecycle():
    """Test plugin lifecycle hooks."""
    plugin_instance = plugin()

    # Test activation
    plugin_instance.on_activate()
    assert plugin_instance._activated is True

    # Test health
    health = plugin_instance.health_check()
    assert health["healthy"] is True
    assert health["activated"] is True

    # Test deactivation
    plugin_instance.on_deactivate()
    assert plugin_instance._activated is False
```

### Manual Testing

```python
# test_plugin.py
import asyncio
from victor.tools.registry import ToolRegistry
from my_vertical.plugin import plugin

async def main():
    registry = ToolRegistry()
    plugin_instance = plugin()

    # Register
    registry.register_plugin(plugin_instance)

    # Activate
    plugin_instance.on_activate()

    # Use tools
    tool = registry.get("search")
    result = await tool.execute({}, query="test")
    print(f"Result: {result}")

    # Health check
    health = plugin_instance.health_check()
    print(f"Health: {health}")

    # Deactivate
    plugin_instance.on_deactivate()

asyncio.run(main())
```

---

## Troubleshooting

### Issue: Plugin Not Discovered

**Symptoms**: Plugin not loading via entry points

**Solutions**:
1. Verify entry point in `pyproject.toml`:
   ```toml
   [project.entry-points."victor.plugins"]
   my_vertical = "my_vertical.plugin:plugin"
   ```

2. Check entry point function exists:
   ```python
   def plugin() -> VictorPlugin:
       return MyPlugin()
   ```

3. Reinstall package:
   ```bash
   pip install -e .
   ```

### Issue: Tools Not Registering

**Symptoms**: Plugin loads but tools not available

**Solutions**:
1. Verify `register()` method called
2. Check `context.register_tool()` called for each tool
3. Add logging to `register()` method
4. Check tool name conflicts

### Issue: Import Errors

**Symptoms**: Can't import from `victor_sdk`

**Solutions**:
1. Verify `victor-sdk` installed:
   ```bash
   pip show victor-sdk
   ```

2. Upgrade to latest:
   ```bash
   pip install --upgrade victor-sdk
   ```

3. Check Python version (requires 3.10+)

---

## Best Practices

### 1. Use Lazy Loading for Expensive Tools

```python
# GOOD: Lazy loading with ToolFactory
factories = {
    "heavy_model": HeavyModelFactory(),
}
helper = ToolPluginHelper.from_factories(factories)

# AVOID: Eager loading
tools = {
    "heavy_model": HeavyModel(),  # Created immediately!
}
```

### 2. Implement Lifecycle Hooks

```python
# GOOD: Proper resource management
def on_activate(self):
    self.connection = Database.connect()

def on_deactivate(self):
    self.connection.close()

# AVOID: Resource leaks
# (no cleanup)
```

### 3. Provide Health Checks

```python
# GOOD: Health monitoring
def health_check(self):
    return {
        "healthy": self.connection.is_connected(),
        "version": "1.0.0",
    }

# AVOID: No health information
# (no health_check method)
```

### 4. Handle Configuration

```python
# GOOD: Configuration support
def __init__(self, config=None):
    self.config = config or {}
    self.validate_config()

# AVOID: Hard-coded values
# (no configuration support)
```

---

## Additional Resources

- **SDK Reference**: `victor-sdk/victor_sdk/core/plugins.py`
- **ToolFactory Protocol**: `victor-sdk/victor_sdk/verticals/protocols/tool_plugins.py`
- **Examples**: `examples/sdk_plugins/`
- **Release Notes**: `docs/releases/tool-plugin-implementation-complete-2026-04-14.md`

---

## Support

For questions or issues:
- GitHub Issues: https://github.com/anthropics/victor-ai/issues
- Documentation: https://docs.victor.ai
- SDK Documentation: https://docs.victor.ai/sdk/

---

**Happy Plugin Development! 🚀**
