from typing import Any, Dict, List, Optional

from victor.mcp.client import MCPClient
from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool

# Constants
_DEFAULT_MCP_PREFIX = "mcp"


def _get_mcp_client(context: Optional[Dict[str, Any]] = None) -> Optional[MCPClient]:
    """Get MCP client from execution context.

    Args:
        context: Tool execution context

    Returns:
        MCPClient if available in context, None otherwise
    """
    if context:
        return context.get("mcp_client")
    return None


def _get_mcp_prefix(context: Optional[Dict[str, Any]] = None) -> str:
    """Get MCP prefix from execution context.

    Args:
        context: Tool execution context

    Returns:
        MCP prefix string, defaults to 'mcp'
    """
    if context:
        return context.get("mcp_prefix", _DEFAULT_MCP_PREFIX)
    return _DEFAULT_MCP_PREFIX


def _prefixed(name: str, context: Optional[Dict[str, Any]] = None) -> str:
    prefix = _get_mcp_prefix(context)
    return f"{prefix}_{name}"


def configure_mcp_client(client: MCPClient, prefix: str = "mcp") -> None:
    """Configure MCP client globally.

    DEPRECATED: Use context-based injection instead.
    This function is kept for backward compatibility with orchestrator imports.
    """
    import warnings

    warnings.warn(
        "configure_mcp_client() is deprecated. Use context-based injection instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # No-op for backward compatibility - orchestrator should use context injection


def get_mcp_tool_definitions(context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Return MCP tools as Victor tool definitions with a name prefix."""
    mcp_client = _get_mcp_client(context)
    if not mcp_client or not mcp_client.tools:
        return []
    defs = []
    for t in mcp_client.tools:
        defs.append(
            {
                "name": _prefixed(t.name, context),
                "description": t.description or f"MCP tool {t.name}",
                "parameters": {
                    "type": "object",
                    "properties": t.parameters or {},
                },
            }
        )
    return defs


@tool(
    category="mcp",
    priority=Priority.MEDIUM,  # Task-specific MCP bridge
    access_mode=AccessMode.MIXED,  # Depends on MCP tool being called
    danger_level=DangerLevel.MEDIUM,  # External tool execution
    keywords=["mcp", "model context protocol", "external tool"],
)
async def mcp(
    name: str,
    arguments: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Call an MCP tool by name (prefixed with the MCP namespace).
    """
    mcp_client = _get_mcp_client(context)
    if not mcp_client:
        return {
            "success": False,
            "error": "MCP client not configured. Provide mcp_client in context.",
        }
    if not mcp_client.initialized:
        return {"success": False, "error": "MCP client not initialized"}

    prefix = _get_mcp_prefix(context)
    raw_name = name
    if name.startswith(f"{prefix}_"):
        raw_name = name[len(prefix) + 1 :]

    try:
        result = await mcp_client.call_tool(raw_name, **(arguments or {}))
        return {
            "success": result.success,
            "output": result.result if result.success else result.error,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
