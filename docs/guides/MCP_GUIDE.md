# Victor MCP Integration Guide

This guide covers Victor's Model Context Protocol (MCP) integration, including how to use Victor as an MCP server and connect to external MCP servers.

## Overview

MCP (Model Context Protocol) is an open protocol by Anthropic that standardizes how AI applications connect to data sources and tools. Victor provides both:

1. **MCP Server**: Exposes Victor's tools to external clients (Claude Desktop, VS Code)
2. **MCP Client**: Connects to external MCP servers for additional tools
3. **MCP Registry**: Auto-discovery and management of multiple MCP servers

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Victor Agent                              │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ MCP Server   │    │ Tool         │    │ MCP Client   │  │
│  │ (Expose      │◄──►│ Registry     │◄──►│ (Use ext.    │  │
│  │  tools)      │    │ (32+ tools)  │    │  tools)      │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                                       │           │
└─────────┼───────────────────────────────────────┼───────────┘
          │                                       │
          ▼                                       ▼
   Claude Desktop                          External MCP
   VS Code                                 Servers
```

## Using Victor as an MCP Server

### Starting the MCP Server

```bash
# Start Victor's MCP server on stdio
python -m victor.mcp.server
```

### Claude Desktop Configuration

Add to `~/.config/claude-desktop/config.json`:

```json
{
  "mcpServers": {
    "victor": {
      "command": "python",
      "args": ["-m", "victor.mcp.server"]
    }
  }
}
```

### Programmatic Server Usage

```python
from victor.mcp import MCPServer
from victor.tools.base import ToolRegistry

# Create registry with tools
registry = ToolRegistry()
# ... register tools ...

# Create and run server
server = MCPServer(
    name="My Victor Server",
    version="1.0.0",
    tool_registry=registry
)

# Start stdio server
await server.start_stdio_server()
```

### Exposing from Orchestrator

```python
from victor.mcp import create_mcp_server_from_orchestrator

# Create server from existing orchestrator
server = create_mcp_server_from_orchestrator(
    orchestrator,
    name="Victor MCP Server"
)

# Get server info
print(server.get_server_info())
# {'name': 'Victor MCP Server', 'tools_count': 32, ...}
```

## Connecting to External MCP Servers

### Basic Client Usage

```python
from victor.mcp import MCPClient

async with MCPClient() as client:
    # Connect to server
    await client.connect(["python", "external_server.py"])

    # List available tools
    print(client.tools)

    # Call a tool
    result = await client.call_tool("read_file", path="/etc/hosts")
    print(result)
```

### Client with Health Monitoring

```python
from victor.mcp import MCPClient

client = MCPClient(
    name="My Client",
    health_check_interval=30,  # Check every 30 seconds
    auto_reconnect=True,       # Reconnect on failure
    max_reconnect_attempts=3,  # Max retries before giving up
    reconnect_delay=5          # Seconds between retries
)

# Register event callbacks
client.on_connect(lambda: print("Connected!"))
client.on_disconnect(lambda reason: print(f"Disconnected: {reason}"))
client.on_health_change(lambda healthy: print(f"Health: {healthy}"))

await client.connect(["python", "server.py"])
```

## MCP Registry (Auto-Discovery)

The MCP Registry provides centralized management of multiple MCP servers.

### Basic Registry Usage

```python
from victor.mcp import MCPRegistry, MCPServerConfig

# Create registry
registry = MCPRegistry(
    health_check_enabled=True,
    default_health_interval=30
)

# Register servers
registry.register_server(MCPServerConfig(
    name="filesystem",
    command=["python", "-m", "mcp_filesystem"],
    description="Filesystem operations",
    auto_connect=True,
    tags=["storage", "files"]
))

registry.register_server(MCPServerConfig(
    name="database",
    command=["python", "-m", "mcp_postgres"],
    description="PostgreSQL operations",
    auto_connect=True,
    health_check_interval=60,
    tags=["storage", "sql"]
))

# Start registry (connects to all auto_connect servers)
await registry.start()

# Call tools across any server
result = await registry.call_tool("read_file", path="/etc/hosts")

# Get all tools from all servers
all_tools = registry.get_all_tools()

# Get tools by tag
storage_tools = registry.get_tools_by_tag("storage")
```

### Loading from Configuration File

```yaml
# ~/.victor/mcp_servers.yaml
health_check_enabled: true
default_health_interval: 30

servers:
  - name: filesystem
    command: ["python", "-m", "mcp_filesystem"]
    description: Filesystem operations
    auto_connect: true
    tags: [storage, files]

  - name: database
    command: ["python", "-m", "mcp_postgres", "--host", "localhost"]
    description: PostgreSQL operations
    health_check_interval: 60
    tags: [storage, sql]
    env:
      PGPASSWORD: "${DB_PASSWORD}"
```

```python
from pathlib import Path
from victor.mcp import MCPRegistry

registry = MCPRegistry.from_config(Path("~/.victor/mcp_servers.yaml"))
await registry.start()
```

### Registry Event Handling

```python
# Register event callback
def on_event(event_type: str, server_name: str, data: Any):
    print(f"[{event_type}] {server_name}: {data}")

registry.on_event(on_event)

# Events: "connecting", "connected", "failed", "disconnected"
```

### Server Status and Management

```python
# Get status of specific server
status = registry.get_server_status("filesystem")
# {
#     "name": "filesystem",
#     "status": "CONNECTED",
#     "tools_count": 5,
#     "resources_count": 0,
#     "last_health_check": 1732934400.0,
#     "consecutive_failures": 0,
#     "tags": ["storage", "files"]
# }

# Get overall registry status
registry_status = registry.get_registry_status()
# {
#     "total_servers": 2,
#     "connected_servers": 2,
#     "total_tools": 12,
#     "health_monitoring": True
# }

# Reset a failed server for reconnection
registry.reset_server("database")

# Manually disconnect/reconnect
await registry.disconnect("filesystem")
await registry.connect("filesystem")
```

## Configuration

### Settings

Add to your `~/.victor/settings.yaml`:

```yaml
# Enable MCP tools from external servers
use_mcp_tools: true

# Command to start MCP server
mcp_command: "python -m my_mcp_server"

# Prefix for MCP tools (avoids naming conflicts)
mcp_prefix: "mcp"
```

### Environment Variables

```bash
# MCP server discovery paths
export VICTOR_MCP_CONFIG="~/.victor/mcp_servers.yaml"

# Default health check interval
export VICTOR_MCP_HEALTH_INTERVAL=30
```

## Tool Prefixing

When using external MCP tools, they are automatically prefixed to avoid conflicts:

```python
# External tool "read_file" becomes "mcp_read_file"
result = await registry.call_tool("mcp_read_file", path="/etc/hosts")
```

## Error Handling

```python
from victor.mcp import MCPToolCallResult

result = await client.call_tool("some_tool", arg="value")

if result.success:
    print(f"Result: {result.result}")
else:
    print(f"Error: {result.error}")
```

## Best Practices

1. **Use the Registry** for managing multiple servers with auto-reconnection
2. **Enable health checks** to detect and recover from server failures
3. **Use tags** to organize and filter tools by capability
4. **Set appropriate timeouts** based on tool complexity
5. **Use configuration files** for production deployments
6. **Implement event callbacks** for monitoring and alerting

## Troubleshooting

### Server Not Connecting

```python
# Check server status
status = registry.get_server_status("my_server")
print(f"Status: {status['status']}")
print(f"Error: {status['error']}")
```

### Tools Not Appearing

```python
# Refresh tools from server
await client.refresh_tools()
print(f"Tools: {[t.name for t in client.tools]}")
```

### Health Check Failures

```python
# Reset consecutive failure counter
registry.reset_server("my_server")

# Or manually reconnect
await registry.disconnect("my_server")
await registry.connect("my_server")
```

## API Reference

### MCPServer

| Method | Description |
|--------|-------------|
| `register_resource(resource)` | Register a resource |
| `handle_message(message)` | Handle incoming MCP message |
| `start_stdio_server()` | Start stdio server |
| `get_server_info()` | Get server information |
| `get_tool_definitions()` | Get tool definitions |

### MCPClient

| Method | Description |
|--------|-------------|
| `connect(command)` | Connect to MCP server |
| `disconnect(reason)` | Disconnect from server |
| `call_tool(name, **args)` | Call a tool |
| `ping()` | Health check |
| `refresh_tools()` | Refresh tool list |
| `on_connect(callback)` | Register connect callback |
| `on_disconnect(callback)` | Register disconnect callback |
| `on_health_change(callback)` | Register health callback |

### MCPRegistry

| Method | Description |
|--------|-------------|
| `register_server(config)` | Register server config |
| `unregister_server(name)` | Remove server |
| `connect(name)` | Connect to specific server |
| `disconnect(name)` | Disconnect from server |
| `connect_all()` | Connect to all auto_connect servers |
| `start()` | Start registry with health monitoring |
| `stop()` | Stop registry and disconnect all |
| `call_tool(name, **args)` | Call tool across any server |
| `get_all_tools()` | Get tools from all servers |
| `get_tools_by_tag(tag)` | Filter tools by server tag |
| `get_registry_status()` | Get overall status |
| `reset_server(name)` | Reset failed server |
| `on_event(callback)` | Register event callback |
