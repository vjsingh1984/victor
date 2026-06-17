"""Model Context Protocol (MCP) server configuration."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class McpSettings(BaseModel):
    """Model Context Protocol (MCP) server configuration.

    MCP enables connecting to external tool servers using the JSON-RPC 2.0
    protocol. Servers provide tools that are namespaced as mcp__{server}__{tool}.

    Config file format (mcp.yaml or settings.json mcpServers section):
        servers:
          server-name:
            type: stdio|sse|http
            command: "npx"          # for stdio
            args: ["-y", "server"]
            env: {KEY: value}
            url: "https://..."      # for sse/http
    """

    mcp_enabled: bool = True
    mcp_servers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    mcp_auto_discover: bool = True
    mcp_tool_timeout_ms: int = 30000
    mcp_max_retries: int = 2
