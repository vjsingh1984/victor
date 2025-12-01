# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MCP Client implementation for Victor.

Connects to external MCP servers to access their tools and resources,
extending Victor's capabilities with third-party integrations.

Features:
- Stdio-based communication with MCP servers
- Health monitoring with configurable intervals
- Automatic reconnection on connection loss
- Tool and resource caching
"""

import asyncio
import json
import logging
import subprocess
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from victor.config.timeouts import McpTimeouts

logger = logging.getLogger(__name__)

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

    Features:
    - Health monitoring with configurable intervals
    - Automatic reconnection on connection loss
    - Event callbacks for connection state changes
    - Tool and resource caching
    """

    def __init__(
        self,
        name: str = "Victor MCP Client",
        version: str = "1.0.0",
        health_check_interval: int = 30,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 3,
        reconnect_delay: int = 5,
    ):
        """Initialize MCP client.

        Args:
            name: Client name
            version: Client version
            health_check_interval: Seconds between health checks (0 to disable)
            auto_reconnect: Whether to automatically reconnect on failure
            max_reconnect_attempts: Maximum reconnection attempts before giving up
            reconnect_delay: Seconds to wait between reconnection attempts
        """
        self.name = name
        self.version = version
        self.client_info = MCPClientInfo(name=name, version=version)

        self.server_info: Optional[MCPServerInfo] = None
        self.tools: List[MCPTool] = []
        self.resources: List[MCPResource] = []

        self.process: Optional[subprocess.Popen] = None
        self.initialized = False

        # Health monitoring configuration
        self._health_check_interval = health_check_interval
        self._auto_reconnect = auto_reconnect
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_delay = reconnect_delay

        # Connection state
        self._command: Optional[List[str]] = None
        self._last_health_check: float = 0.0
        self._consecutive_failures: int = 0
        self._health_task: Optional[asyncio.Task] = None
        self._running = False

        # Event callbacks
        self._on_connect_callbacks: List[Callable[[], None]] = []
        self._on_disconnect_callbacks: List[Callable[[Optional[str]], None]] = []
        self._on_health_change_callbacks: List[Callable[[bool], None]] = []

    async def connect(self, command: List[str]) -> bool:
        """Connect to MCP server via stdio.

        Args:
            command: Command to start MCP server (e.g., ["python", "server.py"])

        Returns:
            True if connection successful
        """
        self._command = command  # Store for reconnection

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
            success = await self.initialize()

            if success:
                self._consecutive_failures = 0
                self._last_health_check = time.time()
                self._running = True

                # Emit connect event
                for callback in self._on_connect_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Connect callback error: {e}")

                # Start health monitoring if enabled
                if self._health_check_interval > 0:
                    self._health_task = asyncio.create_task(self._health_monitor_loop())

                return True

            return False

        except Exception as e:
            logger.error(f"Error connecting to MCP server: {e}")
            self._consecutive_failures += 1
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
            # Send request - use asyncio to avoid blocking
            request_json = message.model_dump_json(exclude_none=True)
            (request_json + "\n").encode()

            # Write asynchronously using run_in_executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: (self.process.stdin.write(request_json + "\n"), self.process.stdin.flush()),
            )

            # Read response with timeout to avoid indefinite blocking
            try:
                response_line = await asyncio.wait_for(
                    loop.run_in_executor(None, self.process.stdout.readline),
                    timeout=McpTimeouts.RESPONSE,
                )
            except asyncio.TimeoutError:
                logger.warning(f"MCP request timeout for method: {method}")
                return None

            if not response_line:
                return None

            response = json.loads(response_line)

            # Verify response ID matches
            if response.get("id") != msg_id:
                logger.warning(f"Response ID mismatch: {response.get('id')} != {msg_id}")

            return response

        except Exception as e:
            logger.error(f"Error sending MCP request: {e}")
            return None

    async def ping(self) -> bool:
        """Ping the MCP server.

        Returns:
            True if server responds
        """
        try:
            response = await self._send_request(MCPMessageType.PING, {})
            healthy = response is not None and "result" in response

            if healthy:
                self._last_health_check = time.time()
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1

            return healthy
        except Exception as e:
            logger.warning(f"Ping failed: {e}")
            self._consecutive_failures += 1
            return False

    def disconnect(self, reason: Optional[str] = None) -> None:
        """Disconnect from MCP server.

        Args:
            reason: Optional reason for disconnection
        """
        self._running = False

        # Cancel health monitoring
        if self._health_task:
            self._health_task.cancel()
            self._health_task = None

        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=McpTimeouts.TERMINATE)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

            self.process = None
            self.initialized = False

            # Emit disconnect event
            for callback in self._on_disconnect_callbacks:
                try:
                    callback(reason)
                except Exception as e:
                    logger.error(f"Disconnect callback error: {e}")

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        last_healthy = True

        while self._running:
            await asyncio.sleep(self._health_check_interval)

            if not self._running:
                break

            healthy = await self.ping()

            # Emit health change event if state changed
            if healthy != last_healthy:
                for callback in self._on_health_change_callbacks:
                    try:
                        callback(healthy)
                    except Exception as e:
                        logger.error(f"Health change callback error: {e}")
                last_healthy = healthy

            # Attempt reconnection if unhealthy
            if not healthy and self._auto_reconnect:
                await self._try_reconnect()

    async def _try_reconnect(self) -> bool:
        """Attempt to reconnect to the server.

        Returns:
            True if reconnection successful
        """
        if not self._command:
            logger.error("Cannot reconnect: no command stored")
            return False

        if self._consecutive_failures >= self._max_reconnect_attempts:
            logger.error(
                f"Max reconnect attempts ({self._max_reconnect_attempts}) exceeded, " "giving up"
            )
            self.disconnect("max_retries_exceeded")
            return False

        logger.info(
            f"Attempting reconnection "
            f"(attempt {self._consecutive_failures + 1}/{self._max_reconnect_attempts})"
        )

        # Clean up current connection
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=McpTimeouts.KILL)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            self.initialized = False

        await asyncio.sleep(self._reconnect_delay)

        # Reconnect
        try:
            self.process = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            success = await self.initialize()
            if success:
                self._consecutive_failures = 0
                logger.info("Reconnection successful")

                # Emit connect event
                for callback in self._on_connect_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Connect callback error: {e}")

                return True

            return False

        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            self._consecutive_failures += 1
            return False

    def reset_connection(self) -> None:
        """Reset connection state to allow fresh reconnection attempts."""
        self._consecutive_failures = 0

    # Event callback registration

    def on_connect(self, callback: Callable[[], None]) -> None:
        """Register callback for connection events.

        Args:
            callback: Callback function (no arguments)
        """
        self._on_connect_callbacks.append(callback)

    def on_disconnect(self, callback: Callable[[Optional[str]], None]) -> None:
        """Register callback for disconnection events.

        Args:
            callback: Callback function (reason: Optional[str])
        """
        self._on_disconnect_callbacks.append(callback)

    def on_health_change(self, callback: Callable[[bool], None]) -> None:
        """Register callback for health state changes.

        Args:
            callback: Callback function (is_healthy: bool)
        """
        self._on_health_change_callbacks.append(callback)

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
            "health": {
                "last_check": self._last_health_check,
                "consecutive_failures": self._consecutive_failures,
                "health_monitoring": self._health_check_interval > 0,
                "auto_reconnect": self._auto_reconnect,
            },
        }

    @property
    def is_healthy(self) -> bool:
        """Check if client connection is healthy.

        Returns:
            True if connected and no recent failures
        """
        return self.initialized and self._consecutive_failures == 0

    @property
    def last_health_check(self) -> float:
        """Get timestamp of last health check.

        Returns:
            Unix timestamp of last health check
        """
        return self._last_health_check

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.disconnect()
