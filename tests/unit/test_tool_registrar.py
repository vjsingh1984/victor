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

"""Tests for ToolRegistrar component.

Tests cover:
- Configuration dataclasses
- Dynamic tool discovery
- Plugin initialization
- MCP integration
- Tool dependency graph
- Goal inference
- Statistics collection
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import asdict

from victor.agent.tool_registrar import (
    ToolRegistrar,
    ToolRegistrarConfig,
    RegistrationStats,
)


class TestToolRegistrarConfig:
    """Tests for ToolRegistrarConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ToolRegistrarConfig()

        assert config.enable_plugins is True
        assert config.enable_mcp is False
        assert config.enable_tool_graph is True
        assert config.airgapped_mode is False
        assert config.plugin_dirs == []
        assert config.disabled_plugins == set()
        assert config.plugin_packages == []
        assert config.max_workers == 4
        assert config.max_complexity == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ToolRegistrarConfig(
            enable_plugins=False,
            enable_mcp=True,
            airgapped_mode=True,
            plugin_dirs=["/custom/path"],
            disabled_plugins={"plugin1", "plugin2"},
            max_workers=8,
        )

        assert config.enable_plugins is False
        assert config.enable_mcp is True
        assert config.airgapped_mode is True
        assert "/custom/path" in config.plugin_dirs
        assert "plugin1" in config.disabled_plugins
        assert config.max_workers == 8


class TestRegistrationStats:
    """Tests for RegistrationStats dataclass."""

    def test_default_values(self):
        """Test default statistics values."""
        stats = RegistrationStats()

        assert stats.dynamic_tools == 0
        assert stats.plugin_tools == 0
        assert stats.mcp_tools == 0
        assert stats.dependency_graph_tools == 0
        assert stats.total_tools == 0
        assert stats.plugins_loaded == 0
        assert stats.mcp_servers_connected == 0

    def test_custom_values(self):
        """Test custom statistics values."""
        stats = RegistrationStats(
            dynamic_tools=50,
            plugin_tools=10,
            mcp_tools=5,
            total_tools=65,
            plugins_loaded=3,
        )

        assert stats.dynamic_tools == 50
        assert stats.plugin_tools == 10
        assert stats.total_tools == 65
        assert stats.plugins_loaded == 3

    def test_as_dict(self):
        """Test converting stats to dictionary."""
        stats = RegistrationStats(dynamic_tools=10, total_tools=10)
        d = asdict(stats)

        assert d["dynamic_tools"] == 10
        assert d["total_tools"] == 10
        assert "plugin_tools" in d


class TestToolRegistrarInit:
    """Tests for ToolRegistrar initialization."""

    @pytest.fixture
    def mock_tools(self):
        """Create mock tool registry."""
        tools = MagicMock()
        tools.list_tools.return_value = []
        tools.get.return_value = None
        return tools

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.load_tool_config.return_value = {}
        return settings

    def test_initialization(self, mock_tools, mock_settings):
        """Test basic initialization."""
        registrar = ToolRegistrar(
            tools=mock_tools,
            settings=mock_settings,
        )

        assert registrar.tools is mock_tools
        assert registrar.settings is mock_settings
        assert registrar.provider is None
        assert registrar.model is None
        assert registrar.config is not None

    def test_initialization_with_config(self, mock_tools, mock_settings):
        """Test initialization with custom config."""
        config = ToolRegistrarConfig(enable_mcp=True, max_workers=16)

        registrar = ToolRegistrar(
            tools=mock_tools,
            settings=mock_settings,
            config=config,
        )

        assert registrar.config.enable_mcp is True
        assert registrar.config.max_workers == 16

    def test_initialization_with_provider(self, mock_tools, mock_settings):
        """Test initialization with provider."""
        mock_provider = MagicMock()

        registrar = ToolRegistrar(
            tools=mock_tools,
            settings=mock_settings,
            provider=mock_provider,
            model="test-model",
        )

        assert registrar.provider is mock_provider
        assert registrar.model == "test-model"


class TestToolRegistrarBackgroundTasks:
    """Tests for background task management."""

    @pytest.fixture
    def registrar(self):
        """Create registrar with mocks."""
        tools = MagicMock()
        tools.list_tools.return_value = []
        settings = MagicMock()
        settings.load_tool_config.return_value = {}
        return ToolRegistrar(tools=tools, settings=settings)

    def test_set_background_task_callback(self, registrar):
        """Test setting background task callback."""
        callback = MagicMock(return_value=MagicMock())
        registrar.set_background_task_callback(callback)

        assert registrar._create_background_task is callback

    def test_create_task_with_callback(self, registrar):
        """Test creating task with callback."""
        mock_task = MagicMock()
        callback = MagicMock(return_value=mock_task)
        registrar.set_background_task_callback(callback)

        async def dummy_coro():
            pass

        coro = dummy_coro()
        result = registrar._create_task(coro, "test_task")

        callback.assert_called_once_with(coro, "test_task")
        assert result is mock_task


class TestDynamicToolRegistration:
    """Tests for dynamic tool discovery and registration."""

    @pytest.fixture
    def registrar(self):
        """Create registrar with mocks."""
        tools = MagicMock()
        tools.list_tools.return_value = []
        tools.register.return_value = None
        settings = MagicMock()
        settings.load_tool_config.return_value = {}
        return ToolRegistrar(tools=tools, settings=settings)

    def test_register_dynamic_tools_runs(self, registrar):
        """Test dynamic tool registration runs without error."""
        # Simply verify the method can be called - actual tool discovery
        # depends on filesystem state
        count = registrar._register_dynamic_tools()

        # Should return a count >= 0
        assert count >= 0

    def test_excluded_files(self, registrar):
        """Test that excluded files are not loaded."""
        # The excluded files are checked inside _register_dynamic_tools
        excluded = {"__init__.py", "base.py", "decorators.py", "semantic_selector.py"}

        with patch("os.listdir") as mock_listdir:
            mock_listdir.return_value = list(excluded)

            with patch("importlib.import_module") as mock_import:
                registrar._register_dynamic_tools()

                # Should not have imported any of the excluded files
                assert mock_import.call_count == 0


class TestToolConfiguration:
    """Tests for tool configuration loading."""

    @pytest.fixture
    def registrar(self):
        """Create registrar with mocks."""
        tools = MagicMock()
        tools.list_tools.return_value = []
        tools.disable_tool.return_value = None
        settings = MagicMock()
        return ToolRegistrar(tools=tools, settings=settings)

    def test_load_empty_config(self, registrar):
        """Test loading empty configuration."""
        registrar.settings.load_tool_config.return_value = {}

        registrar._load_tool_configurations()

        # Should not disable any tools
        registrar.tools.disable_tool.assert_not_called()

    def test_load_disabled_tools(self, registrar):
        """Test loading disabled tools configuration."""
        registrar.settings.load_tool_config.return_value = {
            "disabled": ["tool1", "tool2"],
        }

        registrar._load_tool_configurations()

        assert registrar.tools.disable_tool.call_count == 2

    def test_load_enabled_tools(self, registrar):
        """Test loading enabled tools configuration."""
        mock_tool = MagicMock()
        mock_tool.name = "allowed_tool"
        registrar.tools.list_tools.return_value = [mock_tool]

        registrar.settings.load_tool_config.return_value = {
            "enabled": ["other_tool"],  # allowed_tool not in list
        }

        registrar._load_tool_configurations()

        # Should disable tools not in enabled list
        registrar.tools.disable_tool.assert_called_with("allowed_tool")

    def test_load_per_tool_config(self, registrar):
        """Test loading per-tool configuration."""
        registrar.settings.load_tool_config.return_value = {
            "my_tool": {"enabled": False},
        }

        registrar._load_tool_configurations()

        registrar.tools.disable_tool.assert_called_with("my_tool")

    def test_config_load_error(self, registrar):
        """Test handling configuration load errors."""
        registrar.settings.load_tool_config.side_effect = Exception("Config error")

        # Should not raise
        registrar._load_tool_configurations()


class TestPluginInitialization:
    """Tests for plugin system initialization."""

    @pytest.fixture
    def registrar(self):
        """Create registrar with mocks."""
        tools = MagicMock()
        tools.list_tools.return_value = []
        settings = MagicMock()
        settings.load_tool_config.return_value = {}
        settings.plugin_config = {}
        return ToolRegistrar(tools=tools, settings=settings)

    @patch("victor.tools.plugin_registry.ToolPluginRegistry")
    @patch("victor.config.settings.get_project_paths")
    def test_initialize_plugins(self, mock_paths, mock_registry_class, registrar):
        """Test plugin initialization."""
        mock_paths.return_value = MagicMock(global_plugins_dir="/plugins")

        mock_registry = MagicMock()
        mock_registry.discover_and_load.return_value = 2
        mock_registry.loaded_plugins = {"p1": MagicMock(), "p2": MagicMock()}
        mock_registry.register_tools.return_value = 5
        mock_registry_class.return_value = mock_registry

        count = registrar._initialize_plugins()

        assert count == 5
        assert registrar.plugin_manager is mock_registry

    @patch("victor.tools.plugin_registry.ToolPluginRegistry")
    @patch("victor.config.settings.get_project_paths")
    def test_plugin_dirs_from_config(self, mock_paths, mock_registry_class, registrar):
        """Test custom plugin directories."""
        mock_paths.return_value = MagicMock(global_plugins_dir="/plugins")
        registrar.config.plugin_dirs = ["/custom/plugins"]

        mock_registry = MagicMock()
        mock_registry.discover_and_load.return_value = 0
        mock_registry.loaded_plugins = {}
        mock_registry_class.return_value = mock_registry

        registrar._initialize_plugins()

        call_args = mock_registry_class.call_args
        plugin_dirs = call_args.kwargs.get("plugin_dirs", [])
        assert "/custom/plugins" in plugin_dirs

    @patch("victor.tools.plugin_registry.ToolPluginRegistry")
    @patch("victor.config.settings.get_project_paths")
    def test_disabled_plugins(self, mock_paths, mock_registry_class, registrar):
        """Test disabling specific plugins."""
        mock_paths.return_value = MagicMock(global_plugins_dir="/plugins")
        registrar.config.disabled_plugins = {"bad_plugin"}

        mock_registry = MagicMock()
        mock_registry.discover_and_load.return_value = 0
        mock_registry.loaded_plugins = {}
        mock_registry_class.return_value = mock_registry

        registrar._initialize_plugins()

        mock_registry.disable_plugin.assert_called_with("bad_plugin")

    @patch("victor.tools.plugin_registry.ToolPluginRegistry", side_effect=ImportError("No plugins"))
    def test_plugin_init_error(self, mock_registry_class, registrar):
        """Test handling plugin initialization errors."""
        count = registrar._initialize_plugins()

        assert count == 0
        assert registrar.plugin_manager is None


class TestMCPIntegration:
    """Tests for MCP integration."""

    @pytest.fixture
    def registrar(self):
        """Create registrar with mocks."""
        tools = MagicMock()
        tools.list_tools.return_value = []
        tools.register_dict.return_value = None
        settings = MagicMock()
        settings.load_tool_config.return_value = {}
        settings.mcp_command = None
        settings.use_mcp_tools = False
        return ToolRegistrar(
            tools=tools, settings=settings, config=ToolRegistrarConfig(enable_mcp=True)
        )

    @patch("victor.mcp.registry.MCPRegistry")
    def test_setup_mcp_registry(self, mock_registry_class, registrar):
        """Test MCP registry setup."""
        mock_registry = MagicMock()
        mock_registry.list_servers.return_value = []
        mock_registry_class.discover_servers.return_value = mock_registry

        registrar._setup_mcp_integration()

        mock_registry_class.discover_servers.assert_called_once()
        assert registrar.mcp_registry is mock_registry

    @patch("victor.mcp.registry.MCPRegistry")
    def test_setup_mcp_with_command(self, mock_registry_class, registrar):
        """Test MCP setup with command from settings."""
        registrar.settings.mcp_command = "python mcp_server.py"

        mock_registry = MagicMock()
        mock_registry.list_servers.return_value = []
        mock_registry_class.discover_servers.return_value = mock_registry

        registrar._setup_mcp_integration()

        # Should register server from command
        mock_registry.register_server.assert_called_once()

    @patch("victor.tools.mcp_bridge_tool.get_mcp_tool_definitions")
    @patch("victor.mcp.registry.MCPRegistry")
    def test_register_mcp_tools(self, mock_registry_class, mock_get_tools, registrar):
        """Test registering MCP tool definitions."""
        mock_registry = MagicMock()
        mock_registry.list_servers.return_value = []
        mock_registry_class.discover_servers.return_value = mock_registry

        mock_get_tools.return_value = [
            {"name": "mcp_tool1"},
            {"name": "mcp_tool2"},
        ]

        count = registrar._setup_mcp_integration()

        assert count == 2
        assert registrar.tools.register_dict.call_count == 2


class TestToolDependencyGraph:
    """Tests for tool dependency graph."""

    @pytest.fixture
    def registrar(self):
        """Create registrar with tool graph."""
        tools = MagicMock()
        tools.list_tools.return_value = []
        settings = MagicMock()
        settings.load_tool_config.return_value = {}

        tool_graph = MagicMock()
        tool_graph.add_tool.return_value = None

        return ToolRegistrar(tools=tools, settings=settings, tool_graph=tool_graph)

    def test_register_tool_dependencies(self, registrar):
        """Test registering tool dependencies."""
        count = registrar._register_tool_dependencies()

        # Should register 8 tools with dependencies
        assert count == 8
        assert registrar.tool_graph.add_tool.call_count == 8

    def test_no_graph(self):
        """Test with no tool graph."""
        tools = MagicMock()
        settings = MagicMock()
        registrar = ToolRegistrar(tools=tools, settings=settings, tool_graph=None)

        count = registrar._register_tool_dependencies()

        assert count == 0


class TestToolPlanning:
    """Tests for tool planning functionality."""

    @pytest.fixture
    def registrar(self):
        """Create registrar with tool graph."""
        tools = MagicMock()
        tool = MagicMock()
        tool.name = "code_search"
        tool.description = "Search code"
        tool.parameters = {}
        tools.get.return_value = tool
        tools.is_tool_enabled.return_value = True

        settings = MagicMock()

        tool_graph = MagicMock()
        tool_graph.plan.return_value = ["code_search", "read_file"]

        return ToolRegistrar(tools=tools, settings=settings, tool_graph=tool_graph)

    def test_plan_tools(self, registrar):
        """Test planning tools for goals."""
        result = registrar.plan_tools(goals=["summary"])

        assert len(result) >= 1
        registrar.tool_graph.plan.assert_called_once()

    def test_plan_tools_empty_goals(self, registrar):
        """Test planning with empty goals."""
        result = registrar.plan_tools(goals=[])

        assert result == []

    def test_plan_tools_no_graph(self):
        """Test planning without tool graph."""
        tools = MagicMock()
        settings = MagicMock()
        registrar = ToolRegistrar(tools=tools, settings=settings, tool_graph=None)

        result = registrar.plan_tools(goals=["summary"])

        assert result == []


class TestGoalInference:
    """Tests for goal inference from messages."""

    @pytest.fixture
    def registrar(self):
        """Create registrar."""
        tools = MagicMock()
        settings = MagicMock()
        return ToolRegistrar(tools=tools, settings=settings)

    def test_infer_summary_goals(self, registrar):
        """Test inferring summary goals."""
        test_cases = [
            "Summarize this code",
            "Give me an overview",
            "Analyze the project",
        ]

        for message in test_cases:
            goals = registrar.infer_goals_from_message(message)
            assert "summary" in goals, f"Failed for: {message}"

    def test_infer_documentation_goals(self, registrar):
        """Test inferring documentation goals."""
        test_cases = [
            "Generate documentation",
            "Create a readme",
            "Write docs for this",
        ]

        for message in test_cases:
            goals = registrar.infer_goals_from_message(message)
            assert "documentation" in goals, f"Failed for: {message}"

    def test_infer_security_goals(self, registrar):
        """Test inferring security goals."""
        test_cases = [
            "Scan for vulnerabilities",
            "Check for secrets",
            "Security audit",
        ]

        for message in test_cases:
            goals = registrar.infer_goals_from_message(message)
            assert "security_report" in goals, f"Failed for: {message}"

    def test_infer_metrics_goals(self, registrar):
        """Test inferring metrics goals."""
        test_cases = [
            "Calculate complexity",
            "Show me the metrics",
            "Technical debt analysis",
        ]

        for message in test_cases:
            goals = registrar.infer_goals_from_message(message)
            assert "metrics_report" in goals, f"Failed for: {message}"

    def test_no_goals(self, registrar):
        """Test message with no matching goals."""
        goals = registrar.infer_goals_from_message("Hello, how are you?")

        assert goals == []


class TestRegistrationStatistics:
    """Tests for registration statistics."""

    @pytest.fixture
    def registrar(self):
        """Create registrar."""
        tools = MagicMock()
        tools.list_tools.return_value = []
        settings = MagicMock()
        return ToolRegistrar(tools=tools, settings=settings)

    def test_get_stats(self, registrar):
        """Test getting registration stats."""
        stats = registrar.get_stats()

        assert isinstance(stats, RegistrationStats)

    def test_get_plugin_info_no_manager(self, registrar):
        """Test getting plugin info without manager."""
        info = registrar.get_plugin_info()

        assert info == {"plugins": [], "total": 0}

    def test_get_plugin_info_with_manager(self, registrar):
        """Test getting plugin info with manager."""
        mock_plugin = MagicMock()
        mock_plugin.name = "test_plugin"
        mock_plugin.version = "1.0.0"
        mock_plugin.get_tools.return_value = [MagicMock(), MagicMock()]

        registrar.plugin_manager = MagicMock()
        registrar.plugin_manager.loaded_plugins = {"test_plugin": mock_plugin}

        info = registrar.get_plugin_info()

        assert info["total"] == 1
        assert len(info["plugins"]) == 1
        assert info["plugins"][0]["name"] == "test_plugin"

    def test_get_mcp_info_no_registry(self, registrar):
        """Test getting MCP info without registry."""
        info = registrar.get_mcp_info()

        assert info == {"servers": [], "connected": 0, "total": 0}

    def test_get_mcp_info_with_registry(self, registrar):
        """Test getting MCP info with registry."""
        mock_server = MagicMock()
        mock_server.name = "test_server"
        mock_server.description = "Test MCP server"

        registrar.mcp_registry = MagicMock()
        registrar.mcp_registry.list_servers.return_value = [mock_server]
        registrar._stats.mcp_servers_connected = 1

        info = registrar.get_mcp_info()

        assert info["total"] == 1
        assert info["connected"] == 1


class TestShutdown:
    """Tests for shutdown functionality."""

    @pytest.fixture
    def registrar(self):
        """Create registrar."""
        tools = MagicMock()
        settings = MagicMock()
        return ToolRegistrar(tools=tools, settings=settings)

    @pytest.mark.asyncio
    async def test_shutdown_cancels_tasks(self, registrar):
        """Test that shutdown cancels pending tasks."""
        # Create mock task
        mock_task = MagicMock()
        mock_task.done.return_value = False
        registrar._mcp_tasks.append(mock_task)

        await registrar.shutdown()

        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_with_mcp_registry(self, registrar):
        """Test shutdown with MCP registry."""
        registrar.mcp_registry = MagicMock()
        registrar.mcp_registry.shutdown = AsyncMock()

        await registrar.shutdown()

        registrar.mcp_registry.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_errors(self, registrar):
        """Test shutdown handles errors gracefully."""
        registrar.mcp_registry = MagicMock()
        registrar.mcp_registry.shutdown = AsyncMock(side_effect=Exception("Shutdown error"))

        # Should not raise
        await registrar.shutdown()


class TestInitializeMethod:
    """Tests for the main initialize method."""

    @pytest.fixture
    def registrar(self):
        """Create registrar with all mocks."""
        tools = MagicMock()
        tools.list_tools.return_value = [MagicMock() for _ in range(10)]
        settings = MagicMock()
        settings.load_tool_config.return_value = {}
        settings.use_mcp_tools = False

        config = ToolRegistrarConfig(enable_plugins=False, enable_mcp=False)
        return ToolRegistrar(tools=tools, settings=settings, config=config)

    @pytest.mark.asyncio
    async def test_initialize_basic(self, registrar):
        """Test basic initialization."""
        with patch.object(registrar, "_setup_providers"):
            with patch.object(registrar, "_register_dynamic_tools", return_value=10):
                with patch.object(registrar, "_load_tool_configurations"):
                    stats = await registrar.initialize()

        assert stats.dynamic_tools == 10
        assert stats.total_tools == 10

    @pytest.mark.asyncio
    async def test_initialize_with_plugins(self, registrar):
        """Test initialization with plugins enabled."""
        registrar.config.enable_plugins = True

        with patch.object(registrar, "_setup_providers"):
            with patch.object(registrar, "_register_dynamic_tools", return_value=10):
                with patch.object(registrar, "_load_tool_configurations"):
                    with patch.object(registrar, "_initialize_plugins", return_value=5):
                        stats = await registrar.initialize()

        assert stats.dynamic_tools == 10
        assert stats.plugin_tools == 5

    @pytest.mark.asyncio
    async def test_initialize_with_tool_graph(self, registrar):
        """Test initialization with tool graph."""
        registrar.config.enable_tool_graph = True
        registrar.tool_graph = MagicMock()
        registrar.tool_graph.add_tool.return_value = None

        with patch.object(registrar, "_setup_providers"):
            with patch.object(registrar, "_register_dynamic_tools", return_value=10):
                with patch.object(registrar, "_load_tool_configurations"):
                    stats = await registrar.initialize()

        assert stats.dependency_graph_tools == 8
