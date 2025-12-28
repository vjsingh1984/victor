"""Model Context Protocol (MCP) message formats and types.

Implements the MCP specification for tool and resource definitions.
Based on the Model Context Protocol by Anthropic.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MCPMessageType(str, Enum):
    """MCP message types."""

    # Client -> Server
    INITIALIZE = "initialize"
    LIST_TOOLS = "tools/list"
    CALL_TOOL = "tools/call"
    LIST_RESOURCES = "resources/list"
    READ_RESOURCE = "resources/read"

    # Server -> Client
    INITIALIZED = "initialized"
    TOOLS_LIST = "tools/list_result"
    TOOL_RESULT = "tools/call_result"
    RESOURCES_LIST = "resources/list_result"
    RESOURCE_CONTENT = "resources/read_result"

    # Bidirectional
    PING = "ping"
    PONG = "pong"
    ERROR = "error"


class MCPParameterType(str, Enum):
    """MCP parameter types."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"


class MCPParameter(BaseModel):
    """MCP tool parameter definition."""

    name: str = Field(description="Parameter name")
    type: MCPParameterType = Field(description="Parameter type")
    description: str = Field(description="Parameter description")
    required: bool = Field(default=False, description="Whether parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value")


class MCPTool(BaseModel):
    """MCP tool definition."""

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    parameters: List[MCPParameter] = Field(
        default_factory=list, description="Tool parameters"
    )
    version: str = Field(default="1.0.0", description="Tool version")


class MCPResource(BaseModel):
    """MCP resource definition."""

    uri: str = Field(description="Resource URI (e.g., file://path/to/file)")
    name: str = Field(description="Resource name")
    description: str = Field(description="Resource description")
    mime_type: Optional[str] = Field(default=None, description="MIME type")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class MCPMessage(BaseModel):
    """MCP protocol message."""

    jsonrpc: str = Field(default="2.0", description="JSON-RPC version")
    id: Optional[str] = Field(default=None, description="Message ID")
    method: Optional[MCPMessageType] = Field(default=None, description="Method name")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Method parameters")
    result: Optional[Any] = Field(default=None, description="Result (for responses)")
    error: Optional[Dict[str, Any]] = Field(default=None, description="Error (for error responses)")


class MCPCapabilities(BaseModel):
    """MCP server/client capabilities."""

    tools: bool = Field(default=True, description="Supports tools")
    resources: bool = Field(default=True, description="Supports resources")
    prompts: bool = Field(default=False, description="Supports prompts")
    sampling: bool = Field(default=False, description="Supports sampling")


class MCPServerInfo(BaseModel):
    """MCP server information."""

    name: str = Field(description="Server name")
    version: str = Field(description="Server version")
    capabilities: MCPCapabilities = Field(default_factory=MCPCapabilities)


class MCPClientInfo(BaseModel):
    """MCP client information."""

    name: str = Field(description="Client name")
    version: str = Field(description="Client version")
    capabilities: MCPCapabilities = Field(default_factory=MCPCapabilities)


class MCPToolCallResult(BaseModel):
    """Result of an MCP tool call."""

    tool_name: str = Field(description="Name of the tool that was called")
    success: bool = Field(description="Whether the call succeeded")
    result: Any = Field(default=None, description="Tool result")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class MCPResourceContent(BaseModel):
    """Content of an MCP resource."""

    uri: str = Field(description="Resource URI")
    mime_type: Optional[str] = Field(default=None, description="Content MIME type")
    content: str = Field(description="Resource content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
