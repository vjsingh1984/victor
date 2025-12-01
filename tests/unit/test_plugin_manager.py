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

"""Tests for plugin_manager module."""

import pytest
import tempfile
from pathlib import Path

from victor.tools.plugin_manager import ToolPluginManager, PluginLoadError


class TestToolPluginManager:
    """Tests for ToolPluginManager class."""

    def test_init_no_dirs(self):
        """Test initialization without directories."""
        manager = ToolPluginManager()
        assert manager.plugin_dirs == []
        assert manager.loaded_plugins == {}
        assert manager.config == {}

    def test_init_with_nonexistent_dir(self):
        """Test initialization with nonexistent directory."""
        manager = ToolPluginManager(plugin_dirs=[Path("/nonexistent/path")])
        assert len(manager.plugin_dirs) == 0

    def test_init_with_existing_dir(self):
        """Test initialization with existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ToolPluginManager(plugin_dirs=[Path(tmpdir)])
            assert len(manager.plugin_dirs) == 1

    def test_init_with_config(self):
        """Test initialization with config."""
        config = {"test_plugin": {"api_key": "test_key"}}
        manager = ToolPluginManager(config=config)
        assert manager.config == config

    def test_config_access(self):
        """Test accessing plugin config via config attribute."""
        config = {"my_plugin": {"setting": "value"}}
        manager = ToolPluginManager(config=config)
        assert manager.config.get("my_plugin") == {"setting": "value"}
        assert manager.config.get("nonexistent", {}) == {}

    def test_get_all_tools_empty(self):
        """Test getting tools when no plugins loaded."""
        manager = ToolPluginManager()
        tools = manager.get_all_tools()
        assert tools == []

    def test_list_plugins_empty(self):
        """Test listing plugins when none loaded."""
        manager = ToolPluginManager()
        result = manager.list_plugins()
        # Returns empty dict when no plugins loaded
        assert result == {}

    def test_discover_plugins_empty_dirs(self):
        """Test discovering plugins in empty directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ToolPluginManager(plugin_dirs=[Path(tmpdir)])
            discovered = manager.discover_plugins()
            assert discovered == []

    def test_context_manager(self):
        """Test context manager support."""
        with ToolPluginManager() as manager:
            assert isinstance(manager, ToolPluginManager)


class TestPluginLoadError:
    """Tests for PluginLoadError exception."""

    def test_plugin_load_error(self):
        """Test PluginLoadError can be raised."""
        with pytest.raises(PluginLoadError):
            raise PluginLoadError("Test error")

    def test_plugin_load_error_message(self):
        """Test PluginLoadError message."""
        try:
            raise PluginLoadError("Plugin failed to load")
        except PluginLoadError as e:
            assert "Plugin failed to load" in str(e)
