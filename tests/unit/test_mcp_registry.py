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

"""Tests for MCP registry module."""


from victor.mcp.registry import (
    ServerStatus,
    MCPServerConfig,
    ServerEntry,
    MCPRegistry,
)


class TestServerStatus:
    """Tests for ServerStatus enum."""

    def test_all_statuses_defined(self):
        """Test all server statuses are defined."""
        assert ServerStatus.DISCONNECTED is not None
        assert ServerStatus.CONNECTING is not None
        assert ServerStatus.CONNECTED is not None
        assert ServerStatus.UNHEALTHY is not None
        assert ServerStatus.FAILED is not None


class TestMCPServerConfig:
    """Tests for MCPServerConfig model."""

    def test_minimal_config(self):
        """Test creating config with minimal fields."""
        config = MCPServerConfig(
            name="test_server",
            command=["python", "-m", "mcp_test"],
        )
        assert config.name == "test_server"
        assert config.command == ["python", "-m", "mcp_test"]
        assert config.auto_connect is True
        assert config.enabled is True

    def test_full_config(self):
        """Test creating config with all fields."""
        config = MCPServerConfig(
            name="full_server",
            command=["node", "server.js"],
            description="Test server",
            auto_connect=False,
            health_check_interval=60,
            max_retries=5,
            retry_delay=10,
            enabled=True,
            tags=["test", "local"],
            env={"API_KEY": "test"},
        )
        assert config.name == "full_server"
        assert config.description == "Test server"
        assert config.auto_connect is False
        assert config.health_check_interval == 60
        assert config.max_retries == 5
        assert config.tags == ["test", "local"]
        assert config.env["API_KEY"] == "test"


class TestServerEntry:
    """Tests for ServerEntry dataclass."""

    def test_default_entry(self):
        """Test default server entry values."""
        config = MCPServerConfig(name="test", command=["echo"])
        entry = ServerEntry(config=config)

        assert entry.config == config
        assert entry.client is None
        assert entry.status == ServerStatus.DISCONNECTED
        assert entry.last_health_check == 0.0
        assert entry.consecutive_failures == 0
        assert entry.error_message is None
        assert entry.tools_cache == []
        assert entry.resources_cache == []


class TestMCPRegistryInit:
    """Tests for MCPRegistry initialization."""

    def test_default_init(self):
        """Test default initialization."""
        registry = MCPRegistry()

        assert registry._servers == {}
        assert registry._tool_to_server == {}
        assert registry._health_check_enabled is True
        assert registry._default_health_interval == 30
        assert registry._running is False

    def test_custom_init(self):
        """Test custom initialization."""
        registry = MCPRegistry(
            health_check_enabled=False,
            default_health_interval=60,
        )

        assert registry._health_check_enabled is False
        assert registry._default_health_interval == 60


class TestMCPRegistryRegister:
    """Tests for MCPRegistry.register_server method."""

    def test_register_server(self):
        """Test registering a server."""
        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["echo"])

        registry.register_server(config)

        assert "test" in registry._servers
        assert registry._servers["test"].config == config
        assert registry._servers["test"].status == ServerStatus.DISCONNECTED

    def test_register_duplicate_server(self):
        """Test registering duplicate server updates config."""
        registry = MCPRegistry()
        config1 = MCPServerConfig(name="test", command=["echo"])
        config2 = MCPServerConfig(name="test", command=["cat"])

        registry.register_server(config1)
        registry.register_server(config2)

        # Should have updated config
        assert registry._servers["test"].config.command == ["cat"]


class TestMCPRegistryUnregister:
    """Tests for MCPRegistry.unregister_server method."""

    def test_unregister_server(self):
        """Test unregistering a server."""
        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["echo"])
        registry.register_server(config)

        result = registry.unregister_server("test")

        assert result is True
        assert "test" not in registry._servers

    def test_unregister_nonexistent(self):
        """Test unregistering non-existent server."""
        registry = MCPRegistry()

        result = registry.unregister_server("nonexistent")

        assert result is False


class TestMCPRegistryServerAccess:
    """Tests for MCPRegistry server access methods."""

    def test_get_server_entry_directly(self):
        """Test getting a server entry from internal dict."""
        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["echo"])
        registry.register_server(config)

        entry = registry._servers.get("test")

        assert entry is not None
        assert entry.config.name == "test"

    def test_get_nonexistent_server_entry(self):
        """Test getting non-existent server returns None."""
        registry = MCPRegistry()

        entry = registry._servers.get("nonexistent")

        assert entry is None

    def test_list_servers(self):
        """Test listing all server names."""
        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="server1", command=["echo"]))
        registry.register_server(MCPServerConfig(name="server2", command=["cat"]))

        servers = registry.list_servers()

        assert len(servers) == 2
        assert "server1" in servers
        assert "server2" in servers

    def test_list_servers_empty(self):
        """Test listing servers when empty."""
        registry = MCPRegistry()

        servers = registry.list_servers()

        assert servers == []


class TestMCPRegistryServerStatus:
    """Tests for MCPRegistry server status methods."""

    def test_get_server_status(self):
        """Test getting server status dict."""
        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["echo"])
        registry.register_server(config)

        status = registry.get_server_status("test")

        assert status is not None
        assert status["name"] == "test"
        assert status["status"] == "DISCONNECTED"

    def test_get_status_nonexistent(self):
        """Test getting status of non-existent server."""
        registry = MCPRegistry()

        status = registry.get_server_status("nonexistent")

        assert status is None

    def test_get_registry_status(self):
        """Test getting overall registry status."""
        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="server1", command=["echo"]))
        registry.register_server(MCPServerConfig(name="server2", command=["cat"]))

        # Manually set one as connected
        registry._servers["server1"].status = ServerStatus.CONNECTED

        status = registry.get_registry_status()

        assert status["total_servers"] == 2
        assert status["connected_servers"] == 1
        assert "server1" in status["servers"]
        assert "server2" in status["servers"]


class TestMCPRegistryFiltering:
    """Tests for MCPRegistry filtering methods."""

    def test_get_tools_by_tag(self):
        """Test getting tools by server tag."""
        registry = MCPRegistry()
        registry.register_server(
            MCPServerConfig(name="server1", command=["echo"], tags=["file", "local"])
        )
        registry.register_server(MCPServerConfig(name="server2", command=["cat"], tags=["network"]))

        # Without connected servers, tools will be empty
        tools = registry.get_tools_by_tag("file")
        assert tools == []

    def test_get_tools_by_tag_no_match(self):
        """Test getting tools by non-existent tag."""
        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="server1", command=["echo"], tags=["local"]))

        tools = registry.get_tools_by_tag("remote")

        assert tools == []

    def test_get_tools_by_server(self):
        """Test getting tools from a specific server."""
        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="server1", command=["echo"]))

        # Without connected servers, tools_cache is empty
        tools = registry.get_tools_by_server("server1")
        assert tools == []

    def test_get_tools_by_nonexistent_server(self):
        """Test getting tools from non-existent server."""
        registry = MCPRegistry()

        tools = registry.get_tools_by_server("nonexistent")
        assert tools == []


class TestMCPRegistryEvents:
    """Tests for MCPRegistry event callbacks."""

    def test_on_event_callback(self):
        """Test registering event callback."""
        registry = MCPRegistry()
        events = []

        def callback(event, server, data):
            events.append((event, server, data))

        registry.on_event(callback)

        assert callback in registry._event_callbacks

    def test_emit_event(self):
        """Test emitting events."""
        registry = MCPRegistry()
        events = []

        def callback(event, server, data):
            events.append((event, server, data))

        registry.on_event(callback)
        registry._emit_event("test_event", "server1", {"key": "value"})

        assert len(events) == 1
        assert events[0] == ("test_event", "server1", {"key": "value"})

    def test_emit_event_callback_error_handled(self):
        """Test that callback errors are handled gracefully."""
        registry = MCPRegistry()

        def bad_callback(event, server, data):
            raise ValueError("Callback error")

        registry.on_event(bad_callback)

        # Should not raise despite callback error
        registry._emit_event("test_event", "server1", {})


class TestMCPRegistryToolMapping:
    """Tests for MCPRegistry tool-to-server mapping."""

    def test_register_tool_mapping(self):
        """Test registering tool-to-server mapping."""
        registry = MCPRegistry()
        config = MCPServerConfig(name="server1", command=["echo"])
        registry.register_server(config)

        registry._tool_to_server["read_file"] = "server1"

        assert registry._tool_to_server["read_file"] == "server1"

    def test_get_tool_from_mapping(self):
        """Test getting tool info through get_tool method."""
        registry = MCPRegistry()
        config = MCPServerConfig(name="server1", command=["echo"])
        registry.register_server(config)

        # get_tool returns None when tool not in tool_to_server
        tool = registry.get_tool("read_file")
        assert tool is None

    def test_tool_to_server_mapping(self):
        """Test tool_to_server mapping directly."""
        registry = MCPRegistry()
        registry._tool_to_server["test_tool"] = "server1"

        assert registry._tool_to_server.get("test_tool") == "server1"
        assert registry._tool_to_server.get("unknown") is None

    def test_get_all_tools_empty(self):
        """Test getting all tools when none connected."""
        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="server1", command=["echo"]))

        tools = registry.get_all_tools()
        assert tools == []

    def test_get_all_resources_empty(self):
        """Test getting all resources when none connected."""
        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="server1", command=["echo"]))

        resources = registry.get_all_resources()
        assert resources == []


class TestMCPRegistrySummary:
    """Tests for MCPRegistry summary/status methods."""

    def test_get_registry_status_full(self):
        """Test getting full registry status."""
        registry = MCPRegistry()
        registry.register_server(MCPServerConfig(name="server1", command=["echo"], enabled=True))
        registry.register_server(MCPServerConfig(name="server2", command=["cat"], enabled=False))
        registry._servers["server1"].status = ServerStatus.CONNECTED

        status = registry.get_registry_status()

        assert status["total_servers"] == 2
        assert status["connected_servers"] == 1
        assert status["total_tools"] == 0  # No tools cached
        assert status["total_resources"] == 0
        assert status["health_monitoring"] is True
        assert status["running"] is False

    def test_get_registry_status_empty(self):
        """Test getting status when empty."""
        registry = MCPRegistry()

        status = registry.get_registry_status()

        assert status["total_servers"] == 0
        assert status["connected_servers"] == 0

    def test_reset_server(self):
        """Test resetting a failed server."""
        registry = MCPRegistry()
        config = MCPServerConfig(name="test", command=["echo"])
        registry.register_server(config)

        # Simulate failure
        registry._servers["test"].status = ServerStatus.FAILED
        registry._servers["test"].consecutive_failures = 5
        registry._servers["test"].error_message = "Connection refused"

        result = registry.reset_server("test")

        assert result is True
        assert registry._servers["test"].status == ServerStatus.DISCONNECTED
        assert registry._servers["test"].consecutive_failures == 0
        assert registry._servers["test"].error_message is None

    def test_reset_nonexistent_server(self):
        """Test resetting non-existent server."""
        registry = MCPRegistry()

        result = registry.reset_server("nonexistent")

        assert result is False
