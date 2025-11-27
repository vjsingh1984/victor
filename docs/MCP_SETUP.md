# MCP Integration Guide

Victor includes an MCP client/server implementation but it is **opt‑in**. Use this guide to enable MCP tools and expose them to any provider that supports tool calling (including Ollama models like qwen3-coder:30b).

## What’s Included
- MCP client (`victor/mcp/client.py`) that can connect to an external MCP server (stdio).
- MCP server utilities (`victor/mcp/server.py`) to expose Victor tools as MCP endpoints (not auto-started).
- A bridge (`victor/tools/mcp_bridge_tool.py`) that wraps MCP tools into Victor’s ToolRegistry when enabled.

## Quick Enablement
1) Add MCP settings to your profile (`~/.victor/profiles.yaml`):
   ```yaml
   use_mcp_tools: true
   mcp_command: "python mcp_server.py"   # replace with your MCP server command
   mcp_prefix: "mcp"                    # optional prefix to avoid name collisions
   ```
2) Restart the backend. Victor will:
   - Launch the MCP server specified by `mcp_command` (stdio).
   - Discover MCP tools and register them with names prefixed by `mcp_...`.
   - Enforce existing tool budget and selection logic on these tools.

## Notes & Best Practices
- You must provide an MCP server command; Victor does not start an embedded MCP server by default.
- Keep the MCP tool set small or use a prefix to avoid polluting tool selection.
- MCP tools are subject to the same action budget and evidence guardrails as other tools.
- If you rely on semantic tool selection, ensure MCP tool descriptions are meaningful; they are passed through as tool metadata.

## Example MCP Server Command
- Python example (adjust path): `mcp_command: "python path/to/mcp_server.py"`
- Node example: `mcp_command: "node mcp-server.js"`

## Troubleshooting
- If no MCP tools appear: verify `use_mcp_tools: true`, the `mcp_command` is valid, and the server supports MCP `LIST_TOOLS`.
- If tool names collide: change `mcp_prefix` to a unique namespace.
