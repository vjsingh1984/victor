"""Model Context Protocol (MCP) integration for Victor.

MCP is an open protocol that standardizes how AI applications connect to
data sources and tools. This module provides both client and server
implementations for Victor.

Components:
- MCP Server: Expose Victor's tools as MCP resources
- MCP Client: Connect to external MCP servers
- Protocol: Standard message formats and communication
"""

from victor.mcp.server import MCPServer
from victor.mcp.client import MCPClient
from victor.mcp.protocol import MCPMessage, MCPTool, MCPResource

__all__ = [
    "MCPServer",
    "MCPClient",
    "MCPMessage",
    "MCPTool",
    "MCPResource",
]
