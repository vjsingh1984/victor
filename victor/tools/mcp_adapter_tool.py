# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""MCP tool adapter — projects MCP tools as first-class Victor tools.

Adapter Pattern: MCPAdapterTool wraps an MCPTool + MCPRegistry into
the BaseTool interface. The LLM sees native tool names (e.g., "github_search")
with proper JSON Schema parameters — no bridge indirection.

Factory Pattern: MCPToolProjector creates MCPAdapterTool instances from
all connected MCP servers, handling name collisions and optional prefixing.

Usage:
    from victor.tools.mcp_adapter_tool import MCPToolProjector

    tools = MCPToolProjector.project(mcp_registry)
    for tool in tools:
        tool_registry.register(tool)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from victor.tools.base import BaseTool, CostTier, ToolResult

if TYPE_CHECKING:
    from victor.integrations.mcp.protocol import MCPParameter, MCPTool
    from victor.integrations.mcp.registry import MCPRegistry

logger = logging.getLogger(__name__)


def _mcp_param_to_json_schema(param: "MCPParameter") -> Dict[str, Any]:
    """Convert a single MCPParameter to JSON Schema property."""
    schema: Dict[str, Any] = {
        "type": param.type.value,
        "description": param.description,
    }
    if param.default is not None:
        schema["default"] = param.default
    return schema


def _mcp_params_to_json_schema(params: List["MCPParameter"]) -> Dict[str, Any]:
    """Convert MCPParameter list to JSON Schema parameters dict."""
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param in params:
        properties[param.name] = _mcp_param_to_json_schema(param)
        if param.required:
            required.append(param.name)

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


class MCPAdapterTool(BaseTool):
    """Adapts an MCP tool to Victor's BaseTool interface.

    Makes MCP tools indistinguishable from native tools in the agent's
    toolset. The LLM sees proper name, description, and JSON Schema
    parameters. Execution routes through MCPRegistry.call_tool().
    """

    def __init__(
        self,
        mcp_tool: "MCPTool",
        mcp_registry: "MCPRegistry",
        server_name: str,
        name_prefix: str = "",
    ):
        self._mcp_tool = mcp_tool
        self._registry = mcp_registry
        self._server_name = server_name
        self._name_prefix = name_prefix
        self._json_schema = _mcp_params_to_json_schema(mcp_tool.parameters)

    @property
    def name(self) -> str:
        if self._name_prefix:
            return f"{self._name_prefix}_{self._mcp_tool.name}"
        return self._mcp_tool.name

    @property
    def description(self) -> str:
        desc = self._mcp_tool.description or f"MCP tool: {self._mcp_tool.name}"
        return f"{desc} (via MCP server: {self._server_name})"

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._json_schema

    @property
    def cost_tier(self) -> CostTier:
        return CostTier.MEDIUM  # MCP tools involve IPC

    @property
    def is_idempotent(self) -> bool:
        return False  # Cannot guarantee for external MCP tools

    @property
    def default_schema_level(self) -> str:
        """MCP tools default to STUB schema for token efficiency."""
        return "stub"

    @property
    def mcp_server_name(self) -> str:
        """The MCP server providing this tool."""
        return self._server_name

    @property
    def mcp_tool_name(self) -> str:
        """The original MCP tool name (without prefix)."""
        return self._mcp_tool.name

    async def execute(self, _exec_ctx: Dict[str, Any], **kwargs: Any) -> ToolResult:
        """Execute by routing through MCPRegistry.call_tool()."""
        try:
            result = await self._registry.call_tool(self._mcp_tool.name, **kwargs)
            return ToolResult(
                success=result.success,
                output=result.result if result.success else result.error,
                error=result.error if not result.success else None,
                metadata={
                    "mcp_server": self._server_name,
                    "mcp_tool": self._mcp_tool.name,
                },
            )
        except Exception as e:
            logger.warning("MCP tool %s execution failed: %s", self.name, e)
            return ToolResult(
                success=False,
                output="",
                error=f"MCP tool execution failed: {e}",
                metadata={"mcp_server": self._server_name},
            )


class MCPToolProjector:
    """Factory that projects MCPRegistry tools into BaseTool instances.

    Iterates all connected MCP servers, creates an MCPAdapterTool for each
    tool, and handles name collisions by prefixing with server name.
    """

    @staticmethod
    def project(
        registry: "MCPRegistry",
        prefix: str = "",
        conflict_strategy: str = "prefix_server",
    ) -> List[MCPAdapterTool]:
        """Create adapter tools for all tools in connected MCP servers.

        Args:
            registry: MCPRegistry with connected servers
            prefix: Optional prefix for all tool names (e.g., "mcp")
            conflict_strategy: How to handle name collisions:
                "prefix_server" — prepend server name (default)
                "skip" — skip duplicates, keep first

        Returns:
            List of MCPAdapterTool instances ready for registration
        """
        tools: List[MCPAdapterTool] = []
        seen_names: Dict[str, str] = {}  # name -> server_name

        for server_name, entry in registry._servers.items():
            if not entry.tools_cache:
                continue

            for mcp_tool in entry.tools_cache:
                adapter = MCPAdapterTool(
                    mcp_tool=mcp_tool,
                    mcp_registry=registry,
                    server_name=server_name,
                    name_prefix=prefix,
                )
                tool_name = adapter.name

                if tool_name in seen_names:
                    if conflict_strategy == "skip":
                        logger.debug(
                            "Skipping duplicate MCP tool %s from %s (already from %s)",
                            tool_name, server_name, seen_names[tool_name],
                        )
                        continue
                    # prefix_server: create with server-prefixed name
                    adapter = MCPAdapterTool(
                        mcp_tool=mcp_tool,
                        mcp_registry=registry,
                        server_name=server_name,
                        name_prefix=f"{prefix}_{server_name}" if prefix else server_name,
                    )
                    tool_name = adapter.name
                    logger.info(
                        "MCP tool name collision: %s → %s (from %s)",
                        mcp_tool.name, tool_name, server_name,
                    )

                seen_names[tool_name] = server_name
                tools.append(adapter)

        logger.info(
            "Projected %d MCP tools from %d servers",
            len(tools),
            len({t.mcp_server_name for t in tools}),
        )
        return tools


__all__ = [
    "MCPAdapterTool",
    "MCPToolProjector",
]
