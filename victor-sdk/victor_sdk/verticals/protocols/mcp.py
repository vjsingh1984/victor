"""MCP (Model Context Protocol) protocol definitions.

These protocols enable external verticals to declare MCP server requirements
and provide custom MCP tool configurations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class McpProvider(Protocol):
    """Protocol for providing MCP server configurations.

    Verticals implement this to declare which MCP servers they need
    and how tools from those servers should be configured.
    """

    def get_mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """Return MCP server configurations for this vertical.

        Returns:
            Dictionary mapping server names to their config dicts.
            Each config should include 'type' (stdio/sse/http),
            'command'/'url', and optional 'args'/'env'/'headers'.
        """
        ...

    def get_mcp_tool_filters(self) -> Optional[Dict[str, List[str]]]:
        """Return tool filters for MCP servers.

        Returns:
            Optional dict mapping server names to lists of allowed
            tool names. None means all tools are allowed.
        """
        ...


@runtime_checkable
class McpToolProvider(Protocol):
    """Protocol for providing MCP-specific tool configurations.

    Verticals implement this to customize how MCP tools are
    presented to the LLM (e.g., custom descriptions, permission
    overrides).
    """

    def get_mcp_tool_overrides(self) -> Dict[str, Dict[str, Any]]:
        """Return per-tool configuration overrides for MCP tools.

        Returns:
            Dict mapping qualified tool names (mcp__server__tool)
            to override dicts with optional keys:
            - 'description': Custom tool description
            - 'required_permission': Permission level override
            - 'enabled': Whether tool is active (default True)
        """
        ...
