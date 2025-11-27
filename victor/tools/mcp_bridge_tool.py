import asyncio
from typing import Any, Dict, List, Optional

from victor.mcp.client import MCPClient
from victor.tools.decorators import tool


_mcp_client: Optional[MCPClient] = None
_mcp_prefix = "mcp"


def configure_mcp_client(client: MCPClient, prefix: str = "mcp") -> None:
    global _mcp_client, _mcp_prefix
    _mcp_client = client
    _mcp_prefix = prefix.strip() or "mcp"


def _prefixed(name: str) -> str:
    return f"{_mcp_prefix}_{name}"


def get_mcp_tool_definitions() -> List[Dict[str, Any]]:
    """Return MCP tools as Victor tool definitions with a name prefix."""
    if not _mcp_client or not _mcp_client.tools:
        return []
    defs = []
    for t in _mcp_client.tools:
        defs.append(
            {
                "name": _prefixed(t.name),
                "description": t.description or f"MCP tool {t.name}",
                "parameters": {
                    "type": "object",
                    "properties": t.parameters or {},
                },
            }
        )
    return defs


@tool
async def mcp_call(name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Call an MCP tool by name (prefixed with the MCP namespace).
    """
    if not _mcp_client:
        return {"success": False, "error": "MCP client not configured"}
    if not _mcp_client.initialized:
        return {"success": False, "error": "MCP client not initialized"}

    raw_name = name
    if name.startswith(f"{_mcp_prefix}_"):
        raw_name = name[len(_mcp_prefix) + 1 :]

    try:
        result = await _mcp_client.call_tool(raw_name, **(arguments or {}))
        return {
            "success": result.success,
            "output": result.result if result.success else result.error,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
