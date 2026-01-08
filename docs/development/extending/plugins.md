# Victor Plugin Development Guide

This guide covers how to create custom tool plugins for Victor, allowing you to extend its capabilities with domain-specific tools.

## Overview

Victor's plugin system allows you to:

- Create custom tools packaged as plugins
- Load plugins dynamically at runtime
- Share tools across the community
- Hot-reload plugins during development

## Quick Start

### 1. Create a Simple Plugin

```python
from victor.tools.plugin import ToolPlugin
from victor.tools.base import BaseTool, ToolResult, CostTier

class MyTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "My custom tool that does something useful"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input value"}
            },
            "required": ["input"]
        }

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.FREE

    async def execute(self, context: dict, **kwargs) -> ToolResult:
        input_val = kwargs.get("input", "")
        return ToolResult(
            success=True,
            output=f"Processed: {input_val}",
            error=None
        )


class Plugin(ToolPlugin):
    name = "my_plugin"
    version = "1.0.0"
    description = "My custom plugin"

    def get_tools(self):
        return [MyTool()]
```

### 2. Load the Plugin

```python
from victor.tools.plugin_manager import ToolPluginManager
from victor.tools.base import ToolRegistry

# Create manager and registry
manager = ToolPluginManager()
registry = ToolRegistry()

# Load plugin
manager.load(MyPlugin())

# Register tools with registry
manager.register_tools(registry)

# Execute tool
result = await registry.execute("my_tool", {}, input="Hello")
```

## Plugin Structure

### Directory Layout

```
my_plugin/
├── __init__.py
├── plugin.py        # Contains Plugin class
├── tools/
│   ├── __init__.py
│   ├── tool_one.py
│   └── tool_two.py
└── config.yaml      # Optional configuration
```

### Plugin Class

Every plugin must define a class extending `ToolPlugin`:

```python
from victor.tools.plugin import ToolPlugin
from victor.tools.base import BaseTool
from typing import List, Dict, Any

class Plugin(ToolPlugin):
    # Required metadata
    name = "my_plugin"
    version = "1.0.0"
    description = "Description of what the plugin does"

    # Optional metadata
    author = "Your Name"
    homepage = "https://github.com/you/my-plugin"
    dependencies = ["requests>=2.28.0"]

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Initialize plugin state
        self.db = None

    def get_tools(self) -> List[BaseTool]:
        """Return list of tools provided by this plugin."""
        return [
            MyToolOne(),
            MyToolTwo(),
        ]

    def initialize(self) -> None:
        """Called when plugin is loaded. Setup resources here."""
        self.db = connect_to_database(self.config.get("db_url"))

    def cleanup(self) -> None:
        """Called when plugin is unloaded. Cleanup resources here."""
        if self.db:
            self.db.close()

    def validate_config(self) -> List[str]:
        """Validate configuration. Return list of errors."""
        errors = []
        if not self.config.get("api_key"):
            errors.append("api_key is required")
        return errors

    def on_tool_registered(self, tool: BaseTool) -> None:
        """Called when a tool from this plugin is registered."""
        print(f"Registered: {tool.name}")

    def on_tool_executed(self, tool_name: str, success: bool, result: Any) -> None:
        """Called after a tool from this plugin executes."""
        self.metrics[tool_name] = self.metrics.get(tool_name, 0) + 1
```

## Tool Development

### Basic Tool

```python
from victor.tools.base import BaseTool, ToolResult, CostTier
from typing import Dict, Any

class MyTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "A detailed description of what this tool does"

    @property
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for parameters."""
        return {
            "type": "object",
            "properties": {
                "required_param": {
                    "type": "string",
                    "description": "A required parameter"
                },
                "optional_param": {
                    "type": "integer",
                    "description": "An optional parameter",
                    "default": 10
                }
            },
            "required": ["required_param"]
        }

    @property
    def cost_tier(self) -> CostTier:
        """Cost tier affects tool selection priority."""
        return CostTier.LOW

    async def execute(self, context: Dict[str, Any], **kwargs) -> ToolResult:
        """Execute the tool."""
        try:
            # Get parameters
            required = kwargs["required_param"]
            optional = kwargs.get("optional_param", 10)

            # Do work
            result = process(required, optional)

            return ToolResult(
                success=True,
                output=result,
                error=None
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )
```

### Cost Tiers

Tools should specify their cost tier for intelligent selection:

```python
from victor.tools.base import CostTier

class CostTier(Enum):
    FREE = "free"      # Local operations (filesystem, bash)
    LOW = "low"        # Compute-only (code review, parsing)
    MEDIUM = "medium"  # External API calls (web search)
    HIGH = "high"      # Resource-intensive (batch 100+ files)
```

### Using Context

Tools receive a `context` dictionary with shared resources:

```python
async def execute(self, context: Dict[str, Any], **kwargs) -> ToolResult:
    # Available context items:
    # - context.get("code_manager"): Code management utilities
    # - context.get("settings"): Application settings
    # - context.get("provider"): Current LLM provider

    settings = context.get("settings")
    if settings.airgapped_mode:
        return ToolResult(success=False, error="Not available offline")

    # ... tool logic
```

## Plugin Manager

### Loading Plugins

```python
from victor.tools.plugin_manager import ToolPluginManager
from pathlib import Path

# Create manager with plugin directories
manager = ToolPluginManager(
    plugin_dirs=[
        Path("~/.victor/plugins"),
        Path("./plugins")
    ],
    config={
        "my_plugin": {"api_key": "secret"},
        "other_plugin": {"timeout": 30}
    }
)

# Auto-discover and load all plugins
manager.discover_and_load()

# Or load specific plugin
from my_plugin import Plugin
manager.load(Plugin(config={"api_key": "secret"}))
```

### Plugin from Python Package

```python
# Load from installed package
plugin = manager.load_plugin_from_package("victor_database_tools")
```

### Managing Plugins

```python
# List loaded plugins
for name, info in manager.list_plugins().items():
    print(f"{name} v{info['version']}: {info['description']}")
    print(f"  Tools: {info['tools']}")

# Unload plugin
manager.unload("my_plugin")

# Reload plugin (hot reload for development)
manager.reload_plugin("my_plugin")

# Disable plugin (prevents loading)
manager.disable_plugin("my_plugin")

# Enable disabled plugin
manager.enable_plugin("my_plugin")
```

### Registering with Tool Registry

```python
from victor.tools.base import ToolRegistry

registry = ToolRegistry()

# Register all plugin tools
count = manager.register_tools(registry)
print(f"Registered {count} tools")

# Get all tools
tools = manager.get_all_tools()
```

## Function-Based Plugins

For simple plugins, you can use decorated functions:

```python
from victor.tools.decorators import tool
from victor.tools.plugin import FunctionToolPlugin

@tool
async def hello_world(name: str = "World") -> str:
    """Say hello to someone.

    Args:
        name: Person to greet

    Returns:
        Greeting message
    """
    return f"Hello, {name}!"

@tool
async def add_numbers(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of the numbers
    """
    return a + b

# Create plugin from functions
plugin = FunctionToolPlugin(
    name="math_tools",
    version="1.0.0",
    tool_functions=[hello_world, add_numbers],
    description="Simple math utilities"
)
```

## Configuration

### Plugin Configuration

Plugins receive configuration via their constructor:

```python
class Plugin(ToolPlugin):
    def __init__(self, config=None):
        super().__init__(config)

    def initialize(self):
        api_key = self.config.get("api_key")
        timeout = self.config.get("timeout", 30)
```

### Configuration File

```yaml
# ~/.victor/plugins.yaml
plugins:
  my_plugin:
    api_key: "${MY_API_KEY}"
    timeout: 60

  database_plugin:
    connection_string: "postgresql://localhost/mydb"
    pool_size: 5
```

### Loading Configuration

```python
import yaml
from pathlib import Path

config_path = Path("~/.victor/plugins.yaml").expanduser()
config = yaml.safe_load(config_path.read_text())

manager = ToolPluginManager(
    plugin_dirs=[Path("~/.victor/plugins")],
    config=config.get("plugins", {})
)
```

## Best Practices

### 1. Error Handling

Always return `ToolResult` with appropriate error messages:

```python
async def execute(self, context, **kwargs) -> ToolResult:
    try:
        result = do_work()
        return ToolResult(success=True, output=result)
    except ValidationError as e:
        return ToolResult(success=False, error=f"Invalid input: {e}")
    except Exception as e:
        logger.exception("Tool failed")
        return ToolResult(success=False, error=str(e))
```

### 2. Validate Parameters

Check parameters before processing:

```python
async def execute(self, context, **kwargs) -> ToolResult:
    path = kwargs.get("path")
    if not path:
        return ToolResult(success=False, error="path is required")

    if not Path(path).exists():
        return ToolResult(success=False, error=f"File not found: {path}")
```

### 3. Use Appropriate Cost Tiers

Be honest about tool costs:

```python
@property
def cost_tier(self) -> CostTier:
    if self.uses_external_api:
        return CostTier.MEDIUM
    return CostTier.FREE
```

### 4. Respect Air-Gapped Mode

Check settings before network operations:

```python
async def execute(self, context, **kwargs) -> ToolResult:
    settings = context.get("settings")
    if settings and settings.airgapped_mode:
        return ToolResult(
            success=False,
            error="This tool requires network access"
        )
```

### 5. Cleanup Resources

Always implement cleanup:

```python
def cleanup(self) -> None:
    if self.db_connection:
        self.db_connection.close()
    if self.temp_dir:
        shutil.rmtree(self.temp_dir, ignore_errors=True)
```

### 6. Add Metadata

Include helpful information:

```python
class Plugin(ToolPlugin):
    name = "my_plugin"
    version = "1.0.0"
    description = "Provides database tools for PostgreSQL"
    author = "Your Name <you@example.com>"
    homepage = "https://github.com/you/victor-db-plugin"
    dependencies = ["psycopg2>=2.9.0"]
```

## Example: Complete Plugin

See `examples/example_plugin.py` for a complete working example including:

- Multiple tools (WeatherTool, TemperatureConverterTool)
- Plugin lifecycle hooks
- Configuration validation
- Tool execution tracking
- Demo code

Run it with:
```bash
python examples/example_plugin.py
```

## API Reference

### ToolPlugin

| Method | Description |
|--------|-------------|
| `get_tools()` | Return list of tools (required) |
| `initialize()` | Setup resources on load |
| `cleanup()` | Cleanup on unload |
| `validate_config()` | Validate configuration |
| `on_tool_registered(tool)` | Called when tool registered |
| `on_tool_executed(name, success, result)` | Called after execution |
| `get_metadata()` | Get plugin metadata |

### ToolPluginManager

| Method | Description |
|--------|-------------|
| `load(plugin)` | Load plugin instance |
| `unload(name)` | Unload plugin by name |
| `discover_and_load()` | Auto-discover plugins |
| `load_plugin_from_path(path)` | Load from directory |
| `load_plugin_from_package(name)` | Load from package |
| `register_tools(registry)` | Register with ToolRegistry |
| `get_all_tools()` | Get all plugin tools |
| `list_plugins()` | List loaded plugins |
| `reload_plugin(name)` | Hot reload plugin |
| `disable_plugin(name)` | Disable plugin |
| `enable_plugin(name)` | Enable plugin |
| `cleanup_all()` | Unload all plugins |

### BaseTool

| Property/Method | Description |
|-----------------|-------------|
| `name` | Tool name (required) |
| `description` | Tool description (required) |
| `parameters` | JSON Schema dict (required) |
| `cost_tier` | CostTier enum (optional) |
| `execute(context, **kwargs)` | Execute tool (required) |

### ToolResult

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Whether execution succeeded |
| `output` | Any | Tool output data |
| `error` | str | Error message if failed |
| `metadata` | dict | Optional metadata |
