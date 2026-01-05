# Victor MCP Server - Docker

This directory contains Docker configuration for running Victor as an MCP (Model Context Protocol) server.

## Quick Start

### Build Image

```bash
# From project root
docker build -t vjsingh1984/victor-ai:mcp-server -f docker/mcp-server/Dockerfile .
```

### Run Container

```bash
# Basic usage (stdio mode for Claude Desktop)
docker run -it --rm -v $(pwd):/workspace vjsingh1984/victor-ai:mcp-server

# With specific tools
docker run -it --rm \
  -e VICTOR_MCP_TOOLS=read,write,ls,git \
  -v $(pwd):/workspace \
  vjsingh1984/victor-ai:mcp-server

# With specific vertical
docker run -it --rm \
  -e VICTOR_MCP_VERTICALS=coding \
  -v $(pwd):/workspace \
  vjsingh1984/victor-ai:mcp-server
```

### Docker Compose

```bash
# Start default MCP server
docker-compose up -d

# Start coding-focused server
docker-compose --profile coding up -d

# Start safe (read-only) server
docker-compose --profile safe up -d

# Start DevOps server (with Docker socket)
docker-compose --profile devops up -d

# Start full-featured server
docker-compose --profile full up -d
```

## Claude Desktop Integration

Add to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "victor-docker": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "/Users/YOUR_USERNAME:/workspace",
        "vjsingh1984/victor-ai:mcp-server"
      ],
      "env": {}
    }
  }
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VICTOR_MCP_TOOLS` | Comma-separated list of tools to expose | All tools |
| `VICTOR_MCP_VERTICALS` | Comma-separated list of verticals | All verticals |
| `VICTOR_MCP_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | WARNING |
| `VICTOR_SANDBOX_MODE` | Enable sandbox mode | false |
| `VICTOR_AIRGAPPED_MODE` | Disable network tools | false |

## Volumes

- `/workspace` - Working directory for file operations (mount your project here)
- `/home/victor/.victor` - Victor configuration and cache

## Services (docker-compose)

| Service | Description | Profile |
|---------|-------------|---------|
| `victor-mcp` | Default MCP server with all tools | default |
| `victor-mcp-coding` | Coding vertical only | coding |
| `victor-mcp-safe` | Read-only tools, sandboxed | safe |
| `victor-mcp-devops` | DevOps vertical with Docker access | devops |
| `victor-mcp-full` | All features with Docker access | full |

## Security Notes

1. **Mount workspace read-only** for untrusted environments:
   ```bash
   docker run -it --rm -v $(pwd):/workspace:ro vjsingh1984/victor-ai:mcp-server
   ```

2. **Docker socket access** is required for Docker tools:
   ```bash
   docker run -it --rm \
     -v /var/run/docker.sock:/var/run/docker.sock \
     -v $(pwd):/workspace \
     vjsingh1984/victor-ai:mcp-server
   ```

3. **Sandbox mode** adds additional restrictions:
   ```bash
   docker run -it --rm \
     -e VICTOR_SANDBOX_MODE=true \
     -v $(pwd):/workspace \
     vjsingh1984/victor-ai:mcp-server
   ```

## See Also

- [Victor MCP Server Guide](../../docs/guides/VICTOR_AS_MCP_SERVER.md)
- [MCP Integration Guide](../../docs/guides/MCP_INTEGRATION.md)
