# Victor Contract Plugin System Examples

This directory contains examples demonstrating the contract-based plugin system
introduced in Victor 0.7.0.

## Overview

The new plugin system provides:

1. **VictorPlugin Protocol** - From `victor_contracts.core.plugins`
2. **PluginContext** - For registering tools, verticals, commands
3. **ToolPluginHelper** - Convenience methods from `victor_contracts.verticals.protocols`
4. **ToolFactory Protocol** - For lazy tool creation
5. **Lifecycle Hooks** - on_activate, on_deactivate, health_check

## Examples

### 1. Basic Plugin (`basic_plugin.py`)

Shows the simplest plugin with direct tool registration.

```python
from victor_contracts.core.plugins import VictorPlugin, PluginContext

class MyPlugin(VictorPlugin):
    @property
    def name(self) -> str:
        return "my_plugin"

    def register(self, context: PluginContext) -> None:
        context.register_tool(MyTool())
```

### 2. ToolFactory Plugin (`factory_plugin.py`)

Demonstrates lazy tool creation using ToolFactory protocol.

Use this for:
- Expensive-to-initialize tools
- Tools requiring runtime configuration
- Tools with heavy resource requirements

### 3. Module Discovery Plugin (`discovery_plugin.py`)

Shows automatic tool discovery from modules.

Use this for:
- Large tool collections
- Auto-registration patterns
- Dynamic tool loading

### 4. Complete Plugin (`complete_plugin.py`)

Full-featured plugin with:
- Multiple registration patterns
- Lifecycle hooks
- Health checks
- Configuration
- Error handling

## Usage

### As Entry Point

Add to `pyproject.toml`:

```toml
[project.entry-points."victor.plugins"]
my_plugin = "examples.sdk_plugins.basic_plugin:plugin"
```

### Programmatic

```python
from victor.tools.registry import ToolRegistry
from examples.sdk_plugins.basic_plugin import plugin

registry = ToolRegistry()
registry.register_plugin(plugin())
```

## Migration from Old System

Old system (`victor.tools.plugin.ToolPlugin`):
```python
# OLD - deprecated
from victor.tools.plugin import ToolPlugin

class OldPlugin(ToolPlugin):
    def get_tools(self):
        return [MyTool()]
```

New system (`victor_contracts.core.plugins.VictorPlugin`):
```python
# NEW - recommended
from victor_contracts.core.plugins import VictorPlugin, PluginContext

class NewPlugin(VictorPlugin):
    @property
    def name(self) -> str:
        return "new_plugin"

    def register(self, context: PluginContext) -> None:
        context.register_tool(MyTool())
```

## Testing

Run examples:

```bash
# Basic plugin
python examples/sdk_plugins/basic_plugin.py

# Factory plugin
python examples/sdk_plugins/factory_plugin.py

# Complete plugin
python examples/sdk_plugins/complete_plugin.py
```

## Documentation

- [Contract Plugin Protocols](../../victor-sdk/victor_contracts/__init__.py)
- [VictorPlugin Reference](../../victor-sdk/victor_contracts/__init__.py)
- [Migration Guide](../../docs/releases/tool-plugin-implementation-complete-2026-04-14.md)

## Support

For questions or issues:
- GitHub Issues: https://github.com/anthropics/victor-ai/issues
- Documentation: https://docs.victor.ai
