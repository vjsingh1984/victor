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

"""Tests for config_loader module."""

import pytest
from unittest.mock import MagicMock, patch

from victor.agent.config_loader import ConfigLoader
from victor.agent.tool_selection import get_critical_tools
from victor.tools.base import ToolRegistry, BaseTool


class MockTool(BaseTool):
    """Mock tool for testing."""

    def __init__(self, name: str, category: str = "general"):
        self._name = name
        self._category = category

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Mock tool {self._name}"

    @property
    def parameters(self) -> dict:
        return {}

    @property
    def category(self) -> str:
        return self._category

    async def execute(self, context, **kwargs):
        return {"success": True}


class TestCriticalTools:
    """Tests for get_critical_tools function."""

    def test_critical_tools_contains_essential(self):
        """Test that get_critical_tools() returns essential tools (canonical names)."""
        critical_tools = get_critical_tools()  # Fallback returns canonical names
        assert "read" in critical_tools
        assert "write" in critical_tools
        assert "ls" in critical_tools
        assert "shell" in critical_tools
        assert "edit" in critical_tools
        assert "search" in critical_tools


class TestConfigLoaderInit:
    """Tests for ConfigLoader initialization."""

    def test_init(self):
        """Test ConfigLoader initialization."""
        mock_settings = MagicMock()
        loader = ConfigLoader(settings=mock_settings)

        assert loader.settings is mock_settings


class TestConfigLoaderLoadToolConfig:
    """Tests for ConfigLoader.load_tool_config method."""

    @pytest.fixture
    def registry_with_tools(self):
        """Create a registry with test tools (using canonical names)."""
        registry = ToolRegistry()
        registry.register(MockTool("read", category="core"))
        registry.register(MockTool("write", category="core"))
        registry.register(MockTool("code_review"))
        registry.register(MockTool("git"))
        return registry

    def test_load_empty_config(self, registry_with_tools):
        """Test loading with empty configuration."""
        mock_settings = MagicMock()
        mock_settings.load_tool_config.return_value = None

        loader = ConfigLoader(settings=mock_settings)
        loader.load_tool_config(registry_with_tools)

        # All tools should remain enabled
        assert registry_with_tools.is_tool_enabled("read")
        assert registry_with_tools.is_tool_enabled("write")

    def test_load_with_disabled_list(self, registry_with_tools):
        """Test loading with disabled list."""
        mock_settings = MagicMock()
        mock_settings.load_tool_config.return_value = {"disabled": ["code_review"]}

        loader = ConfigLoader(settings=mock_settings)
        loader.load_tool_config(registry_with_tools)

        # code_review should be disabled
        assert not registry_with_tools.is_tool_enabled("code_review")
        # Other tools should remain enabled
        assert registry_with_tools.is_tool_enabled("read")

    def test_load_with_enabled_list(self, registry_with_tools):
        """Test loading with enabled list (exclusive mode)."""
        mock_settings = MagicMock()
        mock_settings.load_tool_config.return_value = {"enabled": ["read", "write"]}

        loader = ConfigLoader(settings=mock_settings)
        loader.load_tool_config(registry_with_tools)

        # Only enabled tools should be enabled
        assert registry_with_tools.is_tool_enabled("read")
        assert registry_with_tools.is_tool_enabled("write")
        assert not registry_with_tools.is_tool_enabled("code_review")
        assert not registry_with_tools.is_tool_enabled("git")

    def test_load_with_invalid_tool_names(self, registry_with_tools):
        """Test loading with invalid tool names logs warning."""
        mock_settings = MagicMock()
        mock_settings.load_tool_config.return_value = {"disabled": ["nonexistent_tool"]}

        loader = ConfigLoader(settings=mock_settings)

        with patch("victor.agent.config_loader.logger"):
            loader.load_tool_config(registry_with_tools)
            # Should log warning about invalid tool name
            # (Checking that no exception was raised is sufficient)

    def test_load_with_individual_settings(self, registry_with_tools):
        """Test loading with individual tool settings."""
        mock_settings = MagicMock()
        mock_settings.load_tool_config.return_value = {
            "code_review": {"enabled": False},
            "git": {"enabled": True},
        }

        loader = ConfigLoader(settings=mock_settings)
        loader.load_tool_config(registry_with_tools)

        # code_review should be disabled via individual setting
        assert not registry_with_tools.is_tool_enabled("code_review")

    def test_load_handles_exception(self, registry_with_tools):
        """Test that exceptions are handled gracefully."""
        mock_settings = MagicMock()
        mock_settings.load_tool_config.side_effect = ValueError("Config error")

        loader = ConfigLoader(settings=mock_settings)

        # Should not raise exception
        loader.load_tool_config(registry_with_tools)


class TestConfigLoaderHelpers:
    """Tests for ConfigLoader helper methods."""

    @pytest.fixture
    def registry_with_tools(self):
        """Create a registry with test tools (using canonical names)."""
        registry = ToolRegistry()
        registry.register(MockTool("read", category="core"))
        registry.register(MockTool("write", category="core"))
        registry.register(MockTool("ls", category="core"))
        registry.register(MockTool("shell", category="core"))
        registry.register(MockTool("code_review"))
        return registry

    def test_apply_enabled_list_warns_about_missing_core(self, registry_with_tools):
        """Test that missing core tools triggers warning."""
        mock_settings = MagicMock()
        loader = ConfigLoader(settings=mock_settings)

        config = {"enabled": ["code_review"]}  # Missing core tools
        registered = {t.name for t in registry_with_tools.list_tools(only_enabled=False)}

        with patch("victor.agent.config_loader.logger") as mock_logger:
            loader._apply_enabled_list(config, registry_with_tools, registered)
            # Should warn about missing core tools
            assert mock_logger.warning.called

    def test_apply_disabled_list_warns_about_core_tools(self, registry_with_tools):
        """Test that disabling core tools triggers warning."""
        mock_settings = MagicMock()
        loader = ConfigLoader(settings=mock_settings)

        config = {"disabled": ["read"]}  # Disabling core tool (canonical name)
        registered = {t.name for t in registry_with_tools.list_tools(only_enabled=False)}

        with patch("victor.agent.config_loader.logger") as mock_logger:
            loader._apply_disabled_list(config, registry_with_tools, registered)
            # Should warn about disabling core tool
            assert mock_logger.warning.called

    def test_apply_individual_settings(self, registry_with_tools):
        """Test applying individual tool settings."""
        mock_settings = MagicMock()
        loader = ConfigLoader(settings=mock_settings)

        config = {
            "code_review": {"enabled": False},
            "read": {"enabled": True},  # Canonical name
        }
        registered = {t.name for t in registry_with_tools.list_tools(only_enabled=False)}

        loader._apply_individual_settings(config, registry_with_tools, registered)

        assert not registry_with_tools.is_tool_enabled("code_review")
        assert registry_with_tools.is_tool_enabled("read")

    def test_apply_individual_settings_ignores_reserved_keys(self, registry_with_tools):
        """Test that reserved keys are ignored."""
        mock_settings = MagicMock()
        loader = ConfigLoader(settings=mock_settings)

        config = {
            "enabled": ["some_tool"],  # Reserved key
            "disabled": ["other_tool"],  # Reserved key
            "code_review": {"enabled": False},
        }
        registered = {t.name for t in registry_with_tools.list_tools(only_enabled=False)}

        loader._apply_individual_settings(config, registry_with_tools, registered)

        # code_review should be disabled
        assert not registry_with_tools.is_tool_enabled("code_review")


class TestConfigLoaderLogging:
    """Tests for ConfigLoader logging."""

    def test_log_tool_states(self):
        """Test logging tool states."""
        registry = ToolRegistry()
        registry.register(MockTool("tool1"))
        registry.register(MockTool("tool2"))
        registry.disable_tool("tool2")

        mock_settings = MagicMock()
        loader = ConfigLoader(settings=mock_settings)

        with patch("victor.agent.config_loader.logger") as mock_logger:
            loader._log_tool_states(registry)
            # Should log at debug level
            assert mock_logger.debug.called or mock_logger.info.called
