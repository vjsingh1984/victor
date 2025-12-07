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

"""Tests for MCP server module."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from victor.mcp.server import (
    MCPServer,
    create_mcp_server_from_orchestrator,
)
from victor.mcp.protocol import (
    MCPResource,
    MCPMessageType,
)
from victor.tools.base import ToolRegistry, BaseTool, ToolResult


class MockTool(BaseTool):
    """Mock tool for testing."""

    name = "mock_tool"
    description = "A mock tool for testing"
    parameters = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Test message"},
            "count": {"type": "integer", "description": "Count value"},
        },
        "required": ["message"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output=f"Executed with: {kwargs}")


class MockToolWithListParams(BaseTool):
    """Mock tool with list-style parameters."""

    name = "list_params_tool"
    description = "Tool with list parameters"

    def __init__(self):
        from victor.tools.base import ToolParameter

        self._parameters = [
            ToolParameter(name="path", type="string", description="File path", required=True),
            ToolParameter(
                name="recursive", type="boolean", description="Recursive", required=False
            ),
        ]

    @property
    def parameters(self):
        return self._parameters

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="Done")


class TestMCPServerInit:
    """Tests for MCPServer initialization."""

    def test_init_defaults(self):
        """Test initialization with default values."""
        server = MCPServer()
        assert server.name == "Victor MCP Server"
        assert server.version == "1.0.0"
        assert server.initialized is False
        assert server.resources == []

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        registry = ToolRegistry()
        server = MCPServer(
            name="Custom Server",
            version="2.0.0",
            tool_registry=registry,
        )
        assert server.name == "Custom Server"
        assert server.version == "2.0.0"
        assert server.tool_registry is registry

    def test_server_info(self):
        """Test server info is set correctly."""
        server = MCPServer()
        assert server.info.name == "Victor MCP Server"
        assert server.info.capabilities.tools is True
        assert server.info.capabilities.resources is True
        assert server.info.capabilities.prompts is False


class TestMCPServerRegisterResource:
    """Tests for resource registration."""

    def test_register_resource(self):
        """Test registering a resource."""
        server = MCPServer()
        resource = MCPResource(
            uri="file:///test.txt",
            name="Test File",
            description="A test file",
        )
        server.register_resource(resource)
        assert len(server.resources) == 1
        assert server.resources[0].uri == "file:///test.txt"

    def test_register_multiple_resources(self):
        """Test registering multiple resources."""
        server = MCPServer()
        server.register_resource(MCPResource(uri="file:///a.txt", name="A", description="File A"))
        server.register_resource(MCPResource(uri="file:///b.txt", name="B", description="File B"))
        assert len(server.resources) == 2


class TestMCPServerToolConversion:
    """Tests for tool-to-MCP conversion."""

    def test_tool_to_mcp_dict_params(self):
        """Test converting tool with dict parameters."""
        server = MCPServer()
        tool = MockTool()
        mcp_tool = server._tool_to_mcp(tool)

        assert mcp_tool.name == "mock_tool"
        assert mcp_tool.description == "A mock tool for testing"
        assert len(mcp_tool.parameters) == 2

    def test_tool_to_mcp_list_params(self):
        """Test converting tool with list parameters."""
        server = MCPServer()
        tool = MockToolWithListParams()
        mcp_tool = server._tool_to_mcp(tool)

        assert mcp_tool.name == "list_params_tool"
        assert len(mcp_tool.parameters) == 2


class TestMCPServerHandleMessage:
    """Tests for message handling."""

    @pytest.mark.asyncio
    async def test_handle_initialize(self):
        """Test handling initialize message."""
        server = MCPServer()

        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "1",
                "method": MCPMessageType.INITIALIZE.value,
                "params": {},
            }
        )

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "1"
        assert "result" in response
        assert server.initialized is True

    @pytest.mark.asyncio
    async def test_handle_ping(self):
        """Test handling ping message."""
        server = MCPServer()

        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "2",
                "method": MCPMessageType.PING.value,
            }
        )

        assert response["result"]["pong"] is True

    @pytest.mark.asyncio
    async def test_handle_list_tools_not_initialized(self):
        """Test list_tools when not initialized."""
        server = MCPServer()

        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "3",
                "method": MCPMessageType.LIST_TOOLS.value,
            }
        )

        assert "error" in response
        assert response["error"]["code"] == -32002

    @pytest.mark.asyncio
    async def test_handle_list_tools_initialized(self):
        """Test list_tools when initialized."""
        registry = ToolRegistry()
        registry.register(MockTool())
        server = MCPServer(tool_registry=registry)
        server.initialized = True

        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "4",
                "method": MCPMessageType.LIST_TOOLS.value,
            }
        )

        assert "result" in response
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) == 1

    @pytest.mark.asyncio
    async def test_handle_call_tool_success(self):
        """Test calling a tool successfully."""
        registry = ToolRegistry()
        registry.register(MockTool())
        server = MCPServer(tool_registry=registry)
        server.initialized = True

        # Mock the registry.execute method since it requires context
        from victor.tools.base import ToolResult

        with patch.object(
            registry, "execute", return_value=ToolResult(success=True, output="Done")
        ):
            response = await server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": "5",
                    "method": MCPMessageType.CALL_TOOL.value,
                    "params": {
                        "name": "mock_tool",
                        "arguments": {"message": "test"},
                    },
                }
            )

            assert "result" in response
            assert response["result"]["success"] is True

    @pytest.mark.asyncio
    async def test_handle_call_tool_missing_name(self):
        """Test calling tool without name."""
        server = MCPServer()
        server.initialized = True

        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "6",
                "method": MCPMessageType.CALL_TOOL.value,
                "params": {},
            }
        )

        assert "error" in response
        assert response["error"]["code"] == -32602

    @pytest.mark.asyncio
    async def test_handle_list_resources(self):
        """Test listing resources."""
        server = MCPServer()
        server.initialized = True
        server.register_resource(
            MCPResource(uri="file:///test.txt", name="Test", description="Desc")
        )

        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "7",
                "method": MCPMessageType.LIST_RESOURCES.value,
            }
        )

        assert "result" in response
        assert len(response["result"]["resources"]) == 1

    @pytest.mark.asyncio
    async def test_handle_read_resource_success(self):
        """Test reading a file resource."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, World!")
            temp_path = f.name

        try:
            server = MCPServer()
            server.initialized = True
            server.register_resource(
                MCPResource(
                    uri=f"file://{temp_path}",
                    name="Test File",
                    description="Test file description",
                )
            )

            response = await server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": "8",
                    "method": MCPMessageType.READ_RESOURCE.value,
                    "params": {"uri": f"file://{temp_path}"},
                }
            )

            assert "result" in response
            assert response["result"]["content"] == "Hello, World!"
        finally:
            Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_handle_read_resource_not_found(self):
        """Test reading non-existent resource."""
        server = MCPServer()
        server.initialized = True

        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "9",
                "method": MCPMessageType.READ_RESOURCE.value,
                "params": {"uri": "file:///nonexistent.txt"},
            }
        )

        assert "error" in response
        assert response["error"]["code"] == -32001

    @pytest.mark.asyncio
    async def test_handle_read_resource_missing_uri(self):
        """Test read_resource without URI."""
        server = MCPServer()
        server.initialized = True

        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "10",
                "method": MCPMessageType.READ_RESOURCE.value,
                "params": {},
            }
        )

        assert "error" in response
        assert response["error"]["code"] == -32602

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self):
        """Test handling unknown method."""
        server = MCPServer()

        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "11",
                "method": "unknown_method",
            }
        )

        # Unknown method triggers parse error due to MCPMessage validation
        assert "error" in response
        # Can be parse error (-32700) or method not found (-32601)
        assert response["error"]["code"] in [-32700, -32601]

    @pytest.mark.asyncio
    async def test_handle_parse_error(self):
        """Test handling parse errors or invalid messages."""
        server = MCPServer()

        response = await server.handle_message(
            {
                "invalid": "message",
            }
        )

        assert "error" in response
        # Can be parse error (-32700) or method not found (-32601)
        assert response["error"]["code"] in [-32700, -32601]


class TestMCPServerHelpers:
    """Tests for helper methods."""

    def test_create_response(self):
        """Test creating success response."""
        server = MCPServer()
        response = server._create_response("123", {"data": "value"})

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "123"
        assert response["result"]["data"] == "value"

    def test_create_error(self):
        """Test creating error response."""
        server = MCPServer()
        response = server._create_error("456", -32600, "Invalid request")

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "456"
        assert response["error"]["code"] == -32600
        assert response["error"]["message"] == "Invalid request"

    def test_get_server_info(self):
        """Test getting server info."""
        registry = ToolRegistry()
        registry.register(MockTool())
        server = MCPServer(tool_registry=registry)
        server.register_resource(MCPResource(uri="file:///a.txt", name="A", description="File A"))

        info = server.get_server_info()

        assert info["name"] == "Victor MCP Server"
        assert info["version"] == "1.0.0"
        assert info["tools_count"] == 1
        assert info["resources_count"] == 1

    def test_get_tool_definitions(self):
        """Test getting tool definitions."""
        registry = ToolRegistry()
        registry.register(MockTool())
        server = MCPServer(tool_registry=registry)

        tools = server.get_tool_definitions()

        assert len(tools) == 1
        assert tools[0]["name"] == "mock_tool"


class TestMCPServerCallToolErrors:
    """Tests for tool call error handling."""

    @pytest.mark.asyncio
    async def test_call_tool_not_initialized(self):
        """Test calling tool when server not initialized."""
        server = MCPServer()

        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "1",
                "method": MCPMessageType.CALL_TOOL.value,
                "params": {"name": "mock_tool"},
            }
        )

        assert "error" in response
        assert response["error"]["code"] == -32002

    @pytest.mark.asyncio
    async def test_call_tool_execution_error(self):
        """Test tool execution error."""
        registry = ToolRegistry()
        # Register mock that raises error
        mock_tool = MockTool()
        registry.register(mock_tool)
        server = MCPServer(tool_registry=registry)
        server.initialized = True

        # Mock execute to raise error
        with patch.object(registry, "execute", side_effect=Exception("Execution failed")):
            response = await server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": "2",
                    "method": MCPMessageType.CALL_TOOL.value,
                    "params": {"name": "mock_tool", "arguments": {}},
                }
            )

            assert "error" in response
            assert response["error"]["code"] == -32603


class TestMCPServerReadResource:
    """Tests for resource reading edge cases."""

    @pytest.mark.asyncio
    async def test_read_resource_not_initialized(self):
        """Test reading resource when not initialized."""
        server = MCPServer()

        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "1",
                "method": MCPMessageType.READ_RESOURCE.value,
                "params": {"uri": "file:///test.txt"},
            }
        )

        assert "error" in response
        assert response["error"]["code"] == -32002

    @pytest.mark.asyncio
    async def test_read_resource_non_file_uri(self):
        """Test reading resource with non-file URI."""
        server = MCPServer()
        server.initialized = True
        server.register_resource(
            MCPResource(
                uri="http://example.com/test",
                name="HTTP Resource",
                description="An HTTP resource",
            )
        )

        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "2",
                "method": MCPMessageType.READ_RESOURCE.value,
                "params": {"uri": "http://example.com/test"},
            }
        )

        assert "error" in response
        assert "not supported" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_read_resource_file_error(self):
        """Test reading resource when file read fails."""
        server = MCPServer()
        server.initialized = True
        # Register resource pointing to non-existent file
        server.register_resource(
            MCPResource(
                uri="file:///nonexistent/path/file.txt",
                name="Bad File",
                description="Non-existent file",
            )
        )

        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "3",
                "method": MCPMessageType.READ_RESOURCE.value,
                "params": {"uri": "file:///nonexistent/path/file.txt"},
            }
        )

        assert "error" in response
        assert response["error"]["code"] == -32603


class TestCreateMCPServerFromOrchestrator:
    """Tests for creating server from orchestrator."""

    def test_create_from_orchestrator(self):
        """Test creating server from orchestrator."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.tools = ToolRegistry()

        server = create_mcp_server_from_orchestrator(
            orchestrator=mock_orchestrator,
            name="Test Server",
        )

        assert server.name == "Test Server"
        assert server.tool_registry is mock_orchestrator.tools


class TestMCPServerCreateWithDefaultTools:
    """Tests for create_with_default_tools class method."""

    def test_create_with_default_tools(self):
        """Test creating server with default tools."""
        server = MCPServer.create_with_default_tools(name="Default Tools Server")

        assert server.name == "Default Tools Server"
        assert server.tool_registry is not None


class TestMCPServerMessageWithoutId:
    """Tests for messages without ID."""

    @pytest.mark.asyncio
    async def test_handle_message_generates_id(self):
        """Test that message ID is generated if not provided."""
        server = MCPServer()

        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "method": MCPMessageType.INITIALIZE.value,
                "params": {},
            }
        )

        # Should still work and return a response
        assert "result" in response
        assert response["id"] is not None


class TestMCPServerListResourcesNotInitialized:
    """Tests for list_resources when not initialized."""

    @pytest.mark.asyncio
    async def test_list_resources_not_initialized(self):
        """Test listing resources when not initialized."""
        server = MCPServer()

        response = await server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "1",
                "method": MCPMessageType.LIST_RESOURCES.value,
            }
        )

        assert "error" in response
        assert response["error"]["code"] == -32002
