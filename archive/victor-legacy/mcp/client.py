"""MCP Client implementation for Victor.

Connects to external MCP servers to access their tools and resources,
extending Victor's capabilities with third-party integrations.
"""

import asyncio
import json
import subprocess
import uuid
from typing import Any, Dict, List, Optional

from victor.mcp.protocol import (
    MCPClientInfo,
    MCPMessage,
    MCPMessageType,
    MCPResource,
    MCPServerInfo,
    MCPTool,
    MCPToolCallResult,
)


class MCPClient:
    """MCP client that connects to external MCP servers.

    This client can discover and use tools from MCP-compatible servers,
    allowing Victor to integrate with external tools and data sources.
    """

    def __init__(
        self,
        name: str = "Victor MCP Client",
        version: str = "1.0.0",
    ):
        """Initialize MCP client.

        Args:
            name: Client name
            version: Client version
        """
        self.name = name
        self.version = version
        self.client_info = MCPClientInfo(name=name, version=version)

        self.server_info: Optional[MCPServerInfo] = None
        self.tools: List[MCPTool] = []
        self.resources: List[MCPResource] = []

        self.process: Optional[subprocess.Popen] = None
        self.initialized = False

    async def connect(self, command: List[str]) -> bool:
        """Connect to MCP server via stdio.

        Args:
            command: Command to start MCP server (e.g., ["python", "server.py"])

        Returns:
            True if connection successful
        """
        try:
            # Start server process
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            # Initialize connection
            return await self.initialize()

        except Exception as e:
            print(f"Error connecting to MCP server: {e}")
            return False

    async def initialize(self) -> bool:
        """Initialize MCP connection.

        Returns:
            True if initialization successful
        """
        if not self.process:
            return False

        # Send initialize message
        response = await self._send_request(
            MCPMessageType.INITIALIZE,
            {
                "clientInfo": self.client_info.model_dump(),
                "capabilities": self.client_info.capabilities.model_dump(),
            },
        )

        if response and "result" in response:
            result = response["result"]
            self.server_info = MCPServerInfo(**result.get("serverInfo", {}))
            self.initialized = True

            # Fetch available tools and resources
            await self.refresh_tools()
            await self.refresh_resources()

            return True

        return False

    async def refresh_tools(self) -> List[MCPTool]:
        """Refresh list of available tools from server.

        Returns:
            List of available tools
        """
        if not self.initialized:
            return []

        response = await self._send_request(MCPMessageType.LIST_TOOLS, {})

        if response and "result" in response:
            tools_data = response["result"].get("tools", [])
            self.tools = [MCPTool(**t) for t in tools_data]
            return self.tools

        return []

    async def refresh_resources(self) -> List[MCPResource]:
        """Refresh list of available resources from server.

        Returns:
            List of available resources
        """
        if not self.initialized:
            return []

        response = await self._send_request(MCPMessageType.LIST_RESOURCES, {})

        if response and "result" in response:
            resources_data = response["result"].get("resources", [])
            self.resources = [MCPResource(**r) for r in resources_data]
            return self.resources

        return []

    async def call_tool(self, tool_name: str, **arguments: Any) -> MCPToolCallResult:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of tool to call
            **arguments: Tool arguments

        Returns:
            Tool call result
        """
        if not self.initialized:
            return MCPToolCallResult(
                tool_name=tool_name,
                success=False,
                error="Client not initialized",
            )

        response = await self._send_request(
            MCPMessageType.CALL_TOOL, {"name": tool_name, "arguments": arguments}
        )

        if response and "result" in response:
            return MCPToolCallResult(**response["result"])
        elif response and "error" in response:
            error = response["error"]
            return MCPToolCallResult(
                tool_name=tool_name,
                success=False,
                error=error.get("message", "Unknown error"),
            )

        return MCPToolCallResult(
            tool_name=tool_name, success=False, error="No response from server"
        )

    async def read_resource(self, uri: str) -> Optional[str]:
        """Read content from a resource.

        Args:
            uri: Resource URI

        Returns:
            Resource content if successful
        """
        if not self.initialized:
            return None

        response = await self._send_request(MCPMessageType.READ_RESOURCE, {"uri": uri})

        if response and "result" in response:
            return response["result"].get("content")

        return None

    async def _send_request(
        self, method: MCPMessageType, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Send request to MCP server.

        Args:
            method: Request method
            params: Request parameters

        Returns:
            Response dictionary or None
        """
        if not self.process or not self.process.stdin or not self.process.stdout:
            return None

        msg_id = str(uuid.uuid4())
        message = MCPMessage(id=msg_id, method=method, params=params)

        try:
            # Send request
            request_json = message.model_dump_json(exclude_none=True)
            self.process.stdin.write(request_json + "\n")
            self.process.stdin.flush()

            # Read response
            response_line = self.process.stdout.readline()
            if not response_line:
                return None

            response = json.loads(response_line)

            # Verify response ID matches
            if response.get("id") != msg_id:
                print(f"Warning: Response ID mismatch: {response.get('id')} != {msg_id}")

            return response

        except Exception as e:
            print(f"Error sending MCP request: {e}")
            return None

    async def ping(self) -> bool:
        """Ping the MCP server.

        Returns:
            True if server responds
        """
        response = await self._send_request(MCPMessageType.PING, {})
        return response is not None and "result" in response

    def disconnect(self) -> None:
        """Disconnect from MCP server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

            self.process = None
            self.initialized = False

    def get_tool_by_name(self, name: str) -> Optional[MCPTool]:
        """Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool definition or None
        """
        return next((t for t in self.tools if t.name == name), None)

    def get_resource_by_uri(self, uri: str) -> Optional[MCPResource]:
        """Get resource by URI.

        Args:
            uri: Resource URI

        Returns:
            Resource definition or None
        """
        return next((r for r in self.resources if r.uri == uri), None)

    def get_status(self) -> Dict[str, Any]:
        """Get client status.

        Returns:
            Status dictionary
        """
        return {
            "connected": self.initialized,
            "server": self.server_info.model_dump() if self.server_info else None,
            "tools_count": len(self.tools),
            "resources_count": len(self.resources),
        }

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.disconnect()
