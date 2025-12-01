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

"""Tests for mcp_bridge_tool module."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from victor.tools.mcp_bridge_tool import (
    configure_mcp_client,
    get_mcp_tool_definitions,
    mcp_call,
    _prefixed,
)


class TestConfigureMCPClient:
    """Tests for configure_mcp_client function."""

    def test_configure_with_default_prefix(self):
        """Test configuring MCP client with default prefix."""
        mock_client = MagicMock()
        configure_mcp_client(mock_client)
        # No exception raised

    def test_configure_with_custom_prefix(self):
        """Test configuring MCP client with custom prefix."""
        mock_client = MagicMock()
        configure_mcp_client(mock_client, prefix="custom")
        # No exception raised

    def test_configure_with_empty_prefix(self):
        """Test configuring MCP client with empty prefix defaults to mcp."""
        mock_client = MagicMock()
        configure_mcp_client(mock_client, prefix="")
        # Should default to "mcp"


class TestPrefixed:
    """Tests for _prefixed function."""

    def test_prefixed(self):
        """Test prefixing a name."""
        configure_mcp_client(MagicMock(), prefix="test")
        assert _prefixed("tool") == "test_tool"


class TestGetMCPToolDefinitions:
    """Tests for get_mcp_tool_definitions function."""

    def test_no_client(self):
        """Test when no client is configured."""
        with patch("victor.tools.mcp_bridge_tool._mcp_client", None):
            defs = get_mcp_tool_definitions()
            assert defs == []

    def test_client_with_no_tools(self):
        """Test when client has no tools."""
        mock_client = MagicMock()
        mock_client.tools = None
        with patch("victor.tools.mcp_bridge_tool._mcp_client", mock_client):
            defs = get_mcp_tool_definitions()
            assert defs == []

    def test_client_with_tools(self):
        """Test when client has tools."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.parameters = {"param1": {"type": "string"}}

        mock_client = MagicMock()
        mock_client.tools = [mock_tool]

        with patch("victor.tools.mcp_bridge_tool._mcp_client", mock_client):
            with patch("victor.tools.mcp_bridge_tool._mcp_prefix", "mcp"):
                defs = get_mcp_tool_definitions()
                assert len(defs) == 1
                assert defs[0]["name"] == "mcp_test_tool"


class TestMCPCall:
    """Tests for mcp_call function."""

    @pytest.mark.asyncio
    async def test_mcp_call_no_client(self):
        """Test mcp_call when no client is configured."""
        with patch("victor.tools.mcp_bridge_tool._mcp_client", None):
            result = await mcp_call(name="test_tool")
            assert result["success"] is False
            assert "not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_mcp_call_client_not_initialized(self):
        """Test mcp_call when client is not initialized."""
        mock_client = MagicMock()
        mock_client.initialized = False
        with patch("victor.tools.mcp_bridge_tool._mcp_client", mock_client):
            result = await mcp_call(name="test_tool")
            assert result["success"] is False
            assert "not initialized" in result["error"]

    @pytest.mark.asyncio
    async def test_mcp_call_success(self):
        """Test successful mcp_call."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.result = {"data": "test"}

        mock_client = MagicMock()
        mock_client.initialized = True
        mock_client.call_tool = AsyncMock(return_value=mock_result)

        with patch("victor.tools.mcp_bridge_tool._mcp_client", mock_client):
            with patch("victor.tools.mcp_bridge_tool._mcp_prefix", "mcp"):
                result = await mcp_call(name="mcp_test_tool", arguments={"arg": "value"})
                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_mcp_call_exception(self):
        """Test mcp_call when exception occurs."""
        mock_client = MagicMock()
        mock_client.initialized = True
        mock_client.call_tool = AsyncMock(side_effect=Exception("Test error"))

        with patch("victor.tools.mcp_bridge_tool._mcp_client", mock_client):
            result = await mcp_call(name="test_tool")
            assert result["success"] is False
            assert "Test error" in result["error"]
