"""MCP Server implementation for Victor.

Exposes Victor's tools and resources through the Model Context Protocol,
allowing other MCP clients to use Victor's capabilities.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional

from victor.mcp.protocol import (
    MCPCapabilities,
    MCPMessage,
    MCPMessageType,
    MCPParameter,
    MCPParameterType,
    MCPResource,
    MCPResourceContent,
    MCPServerInfo,
    MCPTool,
    MCPToolCallResult,
)
from victor.tools.base import BaseTool, ToolRegistry


class MCPServer:
    """MCP server that exposes Victor's tools and resources.

    This server implements the Model Context Protocol, allowing external
    clients to discover and use Victor's tools through a standardized interface.
    """

    def __init__(
        self,
        name: str = "Victor MCP Server",
        version: str = "1.0.0",
        tool_registry: Optional[ToolRegistry] = None,
    ):
        """Initialize MCP server.

        Args:
            name: Server name
            version: Server version
            tool_registry: Victor's tool registry
        """
        self.name = name
        self.version = version
        self.tool_registry = tool_registry or ToolRegistry()

        self.info = MCPServerInfo(
            name=name,
            version=version,
            capabilities=MCPCapabilities(
                tools=True,
                resources=True,
                prompts=False,
                sampling=False,
            ),
        )

        self.initialized = False
        self.resources: List[MCPResource] = []

    def register_resource(self, resource: MCPResource) -> None:
        """Register a resource with the MCP server.

        Args:
            resource: Resource to register
        """
        self.resources.append(resource)

    def _tool_to_mcp(self, tool: BaseTool) -> MCPTool:
        """Convert Victor tool to MCP tool definition.

        Args:
            tool: Victor tool

        Returns:
            MCP tool definition
        """
        mcp_params = []

        # Handle both formats: List[ToolParameter] and Dict (JSON Schema)
        tool_params = tool.parameters

        if isinstance(tool_params, list):
            # New format: List[ToolParameter]
            for param in tool_params:
                type_map = {
                    "string": MCPParameterType.STRING,
                    "number": MCPParameterType.NUMBER,
                    "integer": MCPParameterType.NUMBER,
                    "boolean": MCPParameterType.BOOLEAN,
                    "object": MCPParameterType.OBJECT,
                    "array": MCPParameterType.ARRAY,
                }

                mcp_type = type_map.get(param.type.lower(), MCPParameterType.STRING)

                mcp_params.append(
                    MCPParameter(
                        name=param.name,
                        type=mcp_type,
                        description=param.description,
                        required=param.required,
                    )
                )

        elif isinstance(tool_params, dict):
            # Old format: JSON Schema dict
            properties = tool_params.get("properties", {})
            required_list = tool_params.get("required", [])

            for param_name, param_def in properties.items():
                type_map = {
                    "string": MCPParameterType.STRING,
                    "number": MCPParameterType.NUMBER,
                    "integer": MCPParameterType.NUMBER,
                    "boolean": MCPParameterType.BOOLEAN,
                    "object": MCPParameterType.OBJECT,
                    "array": MCPParameterType.ARRAY,
                }

                param_type = param_def.get("type", "string")
                mcp_type = type_map.get(param_type.lower(), MCPParameterType.STRING)

                mcp_params.append(
                    MCPParameter(
                        name=param_name,
                        type=mcp_type,
                        description=param_def.get("description", ""),
                        required=param_name in required_list,
                    )
                )

        return MCPTool(
            name=tool.name,
            description=tool.description,
            parameters=mcp_params,
        )

    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP message.

        Args:
            message: MCP message dictionary

        Returns:
            Response message dictionary
        """
        try:
            mcp_msg = MCPMessage(**message)
            msg_id = mcp_msg.id or str(uuid.uuid4())

            if mcp_msg.method == MCPMessageType.INITIALIZE:
                return await self._handle_initialize(msg_id, mcp_msg.params or {})

            elif mcp_msg.method == MCPMessageType.LIST_TOOLS:
                return await self._handle_list_tools(msg_id)

            elif mcp_msg.method == MCPMessageType.CALL_TOOL:
                return await self._handle_call_tool(msg_id, mcp_msg.params or {})

            elif mcp_msg.method == MCPMessageType.LIST_RESOURCES:
                return await self._handle_list_resources(msg_id)

            elif mcp_msg.method == MCPMessageType.READ_RESOURCE:
                return await self._handle_read_resource(msg_id, mcp_msg.params or {})

            elif mcp_msg.method == MCPMessageType.PING:
                return self._create_response(msg_id, {"pong": True})

            else:
                return self._create_error(
                    msg_id, -32601, f"Method not found: {mcp_msg.method}"
                )

        except Exception as e:
            return self._create_error(None, -32700, f"Parse error: {str(e)}")

    async def _handle_initialize(
        self, msg_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle initialize request.

        Args:
            msg_id: Message ID
            params: Request parameters

        Returns:
            Initialize response
        """
        self.initialized = True

        return self._create_response(
            msg_id,
            {
                "serverInfo": self.info.model_dump(),
                "capabilities": self.info.capabilities.model_dump(),
            },
        )

    async def _handle_list_tools(self, msg_id: str) -> Dict[str, Any]:
        """Handle list tools request.

        Args:
            msg_id: Message ID

        Returns:
            Tools list response
        """
        if not self.initialized:
            return self._create_error(msg_id, -32002, "Server not initialized")

        tools = []
        for tool in self.tool_registry.list_tools():
            mcp_tool = self._tool_to_mcp(tool)
            tools.append(mcp_tool.model_dump())

        return self._create_response(msg_id, {"tools": tools})

    async def _handle_call_tool(
        self, msg_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle call tool request.

        Args:
            msg_id: Message ID
            params: Tool call parameters

        Returns:
            Tool call response
        """
        if not self.initialized:
            return self._create_error(msg_id, -32002, "Server not initialized")

        tool_name = params.get("name")
        tool_args = params.get("arguments", {})

        if not tool_name:
            return self._create_error(msg_id, -32602, "Missing tool name")

        try:
            result = await self.tool_registry.execute(tool_name, **tool_args)

            tool_result = MCPToolCallResult(
                tool_name=tool_name,
                success=result.success,
                result=result.output if result.success else None,
                error=result.error if not result.success else None,
            )

            return self._create_response(msg_id, tool_result.model_dump())

        except Exception as e:
            return self._create_error(msg_id, -32603, f"Tool execution error: {str(e)}")

    async def _handle_list_resources(self, msg_id: str) -> Dict[str, Any]:
        """Handle list resources request.

        Args:
            msg_id: Message ID

        Returns:
            Resources list response
        """
        if not self.initialized:
            return self._create_error(msg_id, -32002, "Server not initialized")

        resources = [r.model_dump() for r in self.resources]
        return self._create_response(msg_id, {"resources": resources})

    async def _handle_read_resource(
        self, msg_id: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle read resource request.

        Args:
            msg_id: Message ID
            params: Request parameters

        Returns:
            Resource content response
        """
        if not self.initialized:
            return self._create_error(msg_id, -32002, "Server not initialized")

        uri = params.get("uri")
        if not uri:
            return self._create_error(msg_id, -32602, "Missing resource URI")

        # Find resource
        resource = next((r for r in self.resources if r.uri == uri), None)
        if not resource:
            return self._create_error(msg_id, -32001, f"Resource not found: {uri}")

        # For file:// URIs, read the file
        if uri.startswith("file://"):
            try:
                from pathlib import Path

                file_path = uri.replace("file://", "")
                content = Path(file_path).read_text()

                resource_content = MCPResourceContent(
                    uri=uri,
                    mime_type=resource.mime_type or "text/plain",
                    content=content,
                    metadata=resource.metadata,
                )

                return self._create_response(msg_id, resource_content.model_dump())

            except Exception as e:
                return self._create_error(
                    msg_id, -32603, f"Error reading resource: {str(e)}"
                )

        return self._create_error(msg_id, -32001, "Resource type not supported")

    def _create_response(self, msg_id: str, result: Any) -> Dict[str, Any]:
        """Create success response.

        Args:
            msg_id: Message ID
            result: Response result

        Returns:
            Response message
        """
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _create_error(
        self, msg_id: Optional[str], code: int, message: str
    ) -> Dict[str, Any]:
        """Create error response.

        Args:
            msg_id: Message ID
            code: Error code
            message: Error message

        Returns:
            Error response message
        """
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": code, "message": message},
        }

    async def start_stdio_server(self) -> None:
        """Start MCP server on stdio (for MCP clients).

        This allows the server to communicate via stdin/stdout,
        which is the standard MCP transport mechanism.
        """
        import sys

        print("MCP Server started on stdio", file=sys.stderr)
        print(f"Server: {self.name} v{self.version}", file=sys.stderr)
        print("Waiting for messages...", file=sys.stderr)

        while True:
            try:
                # Read JSON-RPC message from stdin
                line = sys.stdin.readline()
                if not line:
                    break

                message = json.loads(line)
                response = await self.handle_message(message)

                # Write response to stdout
                print(json.dumps(response), flush=True)

            except json.JSONDecodeError as e:
                error_response = self._create_error(
                    None, -32700, f"Parse error: {str(e)}"
                )
                print(json.dumps(error_response), flush=True)

            except Exception as e:
                error_response = self._create_error(
                    None, -32603, f"Internal error: {str(e)}"
                )
                print(json.dumps(error_response), flush=True)

    def get_server_info(self) -> Dict[str, Any]:
        """Get server information.

        Returns:
            Server info dictionary
        """
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": self.info.capabilities.model_dump(),
            "tools_count": len(self.tool_registry.list_tools()),
            "resources_count": len(self.resources),
        }
