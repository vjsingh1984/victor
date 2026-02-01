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

"""Tests for ConfigProtocol.

Tests the ConfigProtocol interface and conformance.
"""

from typing import Any

from victor.protocols.config_agent import ConfigProtocol


class MockSettings:
    """Mock settings for testing."""

    def __init__(self):
        self.debug_mode = True
        self.verbose = False


class MockAgentMode:
    """Mock agent mode for testing."""

    def __init__(self, name: str):
        self.name = name


class MockConfigImplementation:
    """Mock implementation of ConfigProtocol for testing."""

    def __init__(self):
        self._settings = MockSettings()
        self._tool_budget = 100
        self._mode = MockAgentMode("BUILD")

    @property
    def settings(self) -> Any:
        """Get configuration settings."""
        return self._settings

    @property
    def tool_budget(self) -> int:
        """Get the tool budget for this session."""
        return self._tool_budget

    @property
    def mode(self) -> Any:
        """Get the current agent mode."""
        return self._mode


class TestConfigProtocol:
    """Test suite for ConfigProtocol."""

    def test_settings_property(self):
        """Test that settings property works correctly."""
        impl = MockConfigImplementation()
        settings = impl.settings
        assert isinstance(settings, MockSettings)
        assert settings.debug_mode is True
        assert settings.verbose is False

    def test_tool_budget_property(self):
        """Test that tool_budget property works correctly."""
        impl = MockConfigImplementation()
        assert impl.tool_budget == 100

    def test_mode_property(self):
        """Test that mode property works correctly."""
        impl = MockConfigImplementation()
        mode = impl.mode
        assert isinstance(mode, MockAgentMode)
        assert mode.name == "BUILD"

    def test_protocol_conformance(self):
        """Test that mock implements ConfigProtocol."""
        impl = MockConfigImplementation()
        # This should not raise an error
        assert isinstance(impl, ConfigProtocol)

    def test_settings_different_values(self):
        """Test with different settings values."""
        impl = MockConfigImplementation()
        impl._settings.debug_mode = False
        impl._settings.verbose = True
        assert impl.settings.debug_mode is False
        assert impl.settings.verbose is True

    def test_tool_budget_update(self):
        """Test updating tool_budget."""
        impl = MockConfigImplementation()
        impl._tool_budget = 200
        assert impl.tool_budget == 200

    def test_mode_update(self):
        """Test updating mode."""
        impl = MockConfigImplementation()
        impl._mode = MockAgentMode("PLAN")
        assert impl.mode.name == "PLAN"

    def test_settings_is_property(self):
        """Test that settings is a property."""
        impl = MockConfigImplementation()
        # Should be accessible as a property
        assert hasattr(impl, "settings")
        # Property should be callable without parentheses
        settings = impl.settings
        assert settings is not None


class TestConfigProtocolTypeChecking:
    """Test type checking and protocol compliance."""

    def test_config_protocol_is_protocol(self):
        """Test that ConfigProtocol is a Protocol."""
        from typing import Protocol

        assert issubclass(ConfigProtocol, Protocol)

    def test_config_protocol_has_settings_property(self):
        """Test that ConfigProtocol defines settings property."""
        assert hasattr(ConfigProtocol, "__annotations__")
        # Check that settings is in the protocol
        assert "settings" in dir(ConfigProtocol)

    def test_config_protocol_has_tool_budget_property(self):
        """Test that ConfigProtocol defines tool_budget property."""
        assert hasattr(ConfigProtocol, "__annotations__")
        assert "tool_budget" in dir(ConfigProtocol)

    def test_config_protocol_has_mode_property(self):
        """Test that ConfigProtocol defines mode property."""
        assert hasattr(ConfigProtocol, "__annotations__")
        assert "mode" in dir(ConfigProtocol)
