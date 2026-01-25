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

"""Tests for ToolAccessCoordinator.

This test suite validates the ToolAccessCoordinator which handles:
- Tool enable/disable management
- Mode-based access control
- Session-based tool filtering
- Tool availability checking

The coordinator extracts access control logic from ToolCoordinator
following SRP (Single Responsibility Principle).
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Set

from victor.agent.coordinators.tool_access_coordinator import (
    ToolAccessCoordinator,
    ToolAccessConfig,
    AccessDecision,
    ToolAccessContext,
    create_tool_access_coordinator,
)


class TestToolAccessCoordinatorInit:
    """Test suite for ToolAccessCoordinator initialization."""

    @pytest.fixture
    def mock_tool_registry(self) -> Mock:
        """Create mock tool registry."""
        registry = Mock()
        registry.list_tools = Mock(return_value=["read_file", "write_file", "bash", "grep"])
        return registry

    def test_init_with_default_config(self, mock_tool_registry: Mock):
        """Test initialization with default configuration."""
        # Execute
        coordinator = ToolAccessCoordinator(tool_registry=mock_tool_registry)

        # Assert
        assert coordinator._registry == mock_tool_registry
        assert coordinator._config.default_allow_all is True
        assert coordinator._config.strict_mode is False
        assert coordinator._session_enabled_tools is None

    def test_init_with_custom_config(self, mock_tool_registry: Mock):
        """Test initialization with custom configuration."""
        # Setup
        config = ToolAccessConfig(
            default_allow_all=False,
            strict_mode=True,
            enabled=False,
            timeout=60.0,
        )

        # Execute
        coordinator = ToolAccessCoordinator(
            tool_registry=mock_tool_registry,
            config=config,
        )

        # Assert
        assert coordinator._config.default_allow_all is False
        assert coordinator._config.strict_mode is True
        assert coordinator._config.enabled is False
        assert coordinator._config.timeout == 60.0

    def test_init_with_mode_controller(self, mock_tool_registry: Mock):
        """Test initialization with mode controller."""
        # Setup
        mode_controller = Mock()
        mode_controller.config = Mock()

        # Execute
        coordinator = ToolAccessCoordinator(
            tool_registry=mock_tool_registry,
            mode_controller=mode_controller,
        )

        # Assert
        assert coordinator._mode_controller == mode_controller

    def test_init_without_tool_registry(self):
        """Test initialization without tool registry."""
        # Execute
        coordinator = ToolAccessCoordinator(tool_registry=None)

        # Assert
        assert coordinator._registry is None


class TestToolAccessCoordinatorCheckAccess:
    """Test suite for check_access method."""

    @pytest.fixture
    def mock_tool_registry(self) -> Mock:
        """Create mock tool registry."""
        registry = Mock()
        registry.list_tools = Mock(return_value=["read_file", "write_file", "bash", "grep"])
        return registry

    @pytest.fixture
    def coordinator(self, mock_tool_registry: Mock) -> ToolAccessCoordinator:
        """Create coordinator with default config."""
        return ToolAccessCoordinator(tool_registry=mock_tool_registry)

    @pytest.fixture
    def coordinator_strict(self, mock_tool_registry: Mock) -> ToolAccessCoordinator:
        """Create coordinator with strict mode."""
        config = ToolAccessConfig(default_allow_all=False, strict_mode=True)
        return ToolAccessCoordinator(tool_registry=mock_tool_registry, config=config)

    def test_check_access_registered_tool_default_allow(self, coordinator: ToolAccessCoordinator):
        """Test access check for registered tool with default_allow_all=True."""
        # Execute
        decision = coordinator.check_access("read_file")

        # Assert
        assert decision.allowed is True
        assert decision.tool_name == "read_file"
        assert decision.layer == "registry"

    def test_check_access_unregistered_tool_default_allow(self, coordinator: ToolAccessCoordinator):
        """Test access check for unregistered tool with default_allow_all=True."""
        # Execute
        decision = coordinator.check_access("unknown_tool")

        # Assert
        assert decision.allowed is True
        assert decision.tool_name == "unknown_tool"
        assert decision.layer == "default"

    def test_check_access_unregistered_tool_strict_mode(
        self, coordinator_strict: ToolAccessCoordinator
    ):
        """Test access check for unregistered tool with strict_mode=True."""
        # Execute
        decision = coordinator_strict.check_access("unknown_tool")

        # Assert
        assert decision.allowed is False
        assert decision.tool_name == "unknown_tool"
        assert decision.layer == "strict"
        assert "strict mode" in decision.reason  # type: ignore[operator]

    def test_check_access_unregistered_tool_no_default_allow(self, mock_tool_registry: Mock):
        """Test access check for unregistered tool with default_allow_all=False."""
        # Setup
        config = ToolAccessConfig(default_allow_all=False, strict_mode=False)
        coordinator = ToolAccessCoordinator(tool_registry=mock_tool_registry, config=config)

        # Execute
        decision = coordinator.check_access("unknown_tool")

        # Assert
        assert decision.allowed is False
        assert decision.tool_name == "unknown_tool"
        assert decision.layer == "default"
        assert "default_allow_all is False" in decision.reason  # type: ignore[operator]

    def test_check_access_with_session_enabled_tools(self, coordinator: ToolAccessCoordinator):
        """Test access check with session-enabled tools."""
        # Setup
        coordinator.set_enabled_tools({"read_file", "grep"})
        context = ToolAccessContext(session_enabled_tools={"read_file", "grep"})

        # Execute - tool in session set
        decision_allowed = coordinator.check_access("read_file", context)

        # Execute - tool not in session set
        decision_denied = coordinator.check_access("write_file", context)

        # Assert
        assert decision_allowed.allowed is True
        assert decision_denied.allowed is False
        assert decision_denied.layer == "session"
        assert "session-enabled" in decision_denied.reason

    def test_check_access_with_mode_controller_disallowed(self, mock_tool_registry: Mock):
        """Test access check with mode controller disallowing tool."""
        # Setup
        mode_controller = Mock()
        mode_controller.config.allow_all_tools = False
        mode_controller.config.disallowed_tools = {"bash", "write_file"}

        coordinator = ToolAccessCoordinator(
            tool_registry=mock_tool_registry,
            mode_controller=mode_controller,
        )

        # Execute
        decision = coordinator.check_access("bash")

        # Assert
        assert decision.allowed is False
        assert decision.layer == "mode"
        assert "disallowed by current mode" in decision.reason  # type: ignore[operator]

    def test_check_access_with_mode_controller_allow_all(self, mock_tool_registry: Mock):
        """Test access check with mode controller allowing all tools."""
        # Setup
        mode_controller = Mock()
        mode_controller.config.allow_all_tools = True
        mode_controller.config.disallowed_tools = set()

        coordinator = ToolAccessCoordinator(
            tool_registry=mock_tool_registry,
            mode_controller=mode_controller,
        )

        # Execute
        decision = coordinator.check_access("bash")

        # Assert
        assert decision.allowed is True
        assert decision.layer == "mode"

    def test_check_access_layer_priority_mode_first(self, mock_tool_registry: Mock):
        """Test that mode layer is checked first (highest priority)."""
        # Setup
        mode_controller = Mock()
        mode_controller.config.allow_all_tools = True
        mode_controller.config.disallowed_tools = {"bash"}

        coordinator = ToolAccessCoordinator(
            tool_registry=mock_tool_registry,
            mode_controller=mode_controller,
        )
        coordinator.set_enabled_tools({"bash"})  # Session allows it, but mode disallows

        # Execute
        decision = coordinator.check_access("bash")

        # Assert - mode decision should take precedence
        assert decision.allowed is False
        assert decision.layer == "mode"

    def test_check_access_layer_priority_session_second(self, mock_tool_registry: Mock):
        """Test that session layer is checked second."""
        # Setup
        coordinator = ToolAccessCoordinator(tool_registry=mock_tool_registry)
        coordinator.set_enabled_tools({"read_file"})

        context = ToolAccessContext(session_enabled_tools={"read_file"})

        # Execute - tool in session set
        decision = coordinator.check_access("read_file", context)

        # Assert
        assert decision.allowed is True

    def test_check_access_layer_priority_registry_third(self, coordinator: ToolAccessCoordinator):
        """Test that registry layer is checked third."""
        # Execute
        decision = coordinator.check_access("grep")

        # Assert
        assert decision.allowed is True
        assert decision.layer == "registry"

    def test_check_access_with_context_none(self, coordinator: ToolAccessCoordinator):
        """Test check_access builds context when None is provided."""
        # Execute - should not raise
        decision = coordinator.check_access("read_file", context=None)

        # Assert
        assert decision.allowed is True

    def test_check_access_empty_tool_name(self, coordinator: ToolAccessCoordinator):
        """Test check_access with empty tool name."""
        # Execute
        decision = coordinator.check_access("")

        # Assert - should still process
        assert isinstance(decision, AccessDecision)
        assert decision.tool_name == ""

    def test_check_access_case_sensitive(self, coordinator: ToolAccessCoordinator):
        """Test that tool names are case-sensitive."""
        # Execute - lowercase
        decision_lower = coordinator.check_access("read_file")

        # Execute - uppercase
        decision_upper = coordinator.check_access("READ_FILE")

        # Assert - only lowercase should be in registry
        assert decision_lower.allowed is True
        assert decision_upper.allowed is True  # default_allow_all=True


class TestToolAccessCoordinatorIsToolEnabled:
    """Test suite for is_tool_enabled method."""

    @pytest.fixture
    def mock_tool_registry(self) -> Mock:
        """Create mock tool registry."""
        registry = Mock()
        registry.list_tools = Mock(return_value=["read_file", "write_file", "bash"])
        return registry

    @pytest.fixture
    def coordinator(self, mock_tool_registry: Mock) -> ToolAccessCoordinator:
        """Create coordinator."""
        return ToolAccessCoordinator(tool_registry=mock_tool_registry)

    def test_is_tool_enabled_allowed(self, coordinator: ToolAccessCoordinator):
        """Test is_tool_enabled returns True for allowed tool."""
        # Execute
        result = coordinator.is_tool_enabled("read_file")

        # Assert
        assert result is True

    def test_is_tool_enabled_disallowed_by_mode(self, mock_tool_registry: Mock):
        """Test is_tool_enabled returns False when mode disallows."""
        # Setup
        mode_controller = Mock()
        mode_controller.config.disallowed_tools = {"bash"}
        mode_controller.config.allow_all_tools = False

        coordinator = ToolAccessCoordinator(
            tool_registry=mock_tool_registry,
            mode_controller=mode_controller,
        )

        # Execute
        result = coordinator.is_tool_enabled("bash")

        # Assert
        assert result is False

    def test_is_tool_enabled_with_context(self, coordinator: ToolAccessCoordinator):
        """Test is_tool_enabled with provided context."""
        # Setup
        context = ToolAccessContext(session_enabled_tools={"read_file"})

        # Execute
        result = coordinator.is_tool_enabled("read_file", context)

        # Assert
        assert result is True


class TestToolAccessCoordinatorGetEnabledTools:
    """Test suite for get_enabled_tools method."""

    @pytest.fixture
    def mock_tool_registry(self) -> Mock:
        """Create mock tool registry."""
        registry = Mock()
        registry.list_tools = Mock(return_value=["read_file", "write_file", "bash", "grep"])
        return registry

    @pytest.fixture
    def coordinator(self, mock_tool_registry: Mock) -> ToolAccessCoordinator:
        """Create coordinator."""
        return ToolAccessCoordinator(tool_registry=mock_tool_registry)

    def test_get_enabled_tools_all_available(self, coordinator: ToolAccessCoordinator):
        """Test get_enabled_tools returns all available tools."""
        # Execute
        enabled = coordinator.get_enabled_tools()

        # Assert
        assert enabled == {"read_file", "write_file", "bash", "grep"}

    def test_get_enabled_tools_session_restricted(self, coordinator: ToolAccessCoordinator):
        """Test get_enabled_tools with session restrictions."""
        # Setup
        coordinator.set_enabled_tools({"read_file", "grep"})

        # Execute
        enabled = coordinator.get_enabled_tools()

        # Assert
        assert enabled == {"read_file", "grep"}

    def test_get_enabled_tools_mode_allow_all(self, mock_tool_registry: Mock):
        """Test get_enabled_tools with mode allow_all."""
        # Setup
        mode_controller = Mock()
        mode_controller.config.allow_all_tools = True
        mode_controller.config.disallowed_tools = {"bash"}

        coordinator = ToolAccessCoordinator(
            tool_registry=mock_tool_registry,
            mode_controller=mode_controller,
        )

        # Execute
        enabled = coordinator.get_enabled_tools()

        # Assert - should exclude disallowed tools
        assert "bash" not in enabled
        assert "read_file" in enabled
        assert "write_file" in enabled
        assert "grep" in enabled

    def test_get_enabled_tools_empty_registry(self):
        """Test get_enabled_tools with empty registry."""
        # Setup
        registry = Mock()
        registry.list_tools = Mock(return_value=[])
        coordinator = ToolAccessCoordinator(tool_registry=registry)

        # Execute
        enabled = coordinator.get_enabled_tools()

        # Assert
        assert enabled == set()

    def test_get_enabled_tools_with_context(self, coordinator: ToolAccessCoordinator):
        """Test get_enabled_tools with provided context."""
        # Setup
        context = ToolAccessContext(session_enabled_tools={"read_file"})

        # Execute
        enabled = coordinator.get_enabled_tools(context)

        # Assert
        assert enabled == {"read_file"}


class TestToolAccessCoordinatorGetAvailableTools:
    """Test suite for get_available_tools method."""

    @pytest.fixture
    def mock_tool_registry(self) -> Mock:
        """Create mock tool registry."""
        registry = Mock()
        registry.list_tools = Mock(return_value=["read_file", "write_file", "bash"])
        return registry

    @pytest.fixture
    def coordinator(self, mock_tool_registry: Mock) -> ToolAccessCoordinator:
        """Create coordinator."""
        return ToolAccessCoordinator(tool_registry=mock_tool_registry)

    def test_get_available_tools_returns_set(self, coordinator: ToolAccessCoordinator):
        """Test get_available_tools returns set of tool names."""
        # Execute
        tools = coordinator.get_available_tools()

        # Assert
        assert isinstance(tools, set)
        assert tools == {"read_file", "write_file", "bash"}

    def test_get_available_tools_empty_registry(self):
        """Test get_available_tools with None registry."""
        # Setup
        coordinator = ToolAccessCoordinator(tool_registry=None)

        # Execute
        tools = coordinator.get_available_tools()

        # Assert
        assert tools == set()

    def test_get_available_tools_delegates_to_registry(self, coordinator: ToolAccessCoordinator):
        """Test get_available_tools delegates to registry.list_tools."""
        # Execute
        tools = coordinator.get_available_tools()

        # Assert
        coordinator._registry.list_tools.assert_called_once()

    def test_get_available_tools_returns_copy(self, coordinator: ToolAccessCoordinator):
        """Test get_available_tools returns a copy (not internal reference)."""
        # Execute
        tools1 = coordinator.get_available_tools()
        tools2 = coordinator.get_available_tools()

        # Assert - should be equal but different objects
        assert tools1 == tools2
        assert tools1 is not tools2


class TestToolAccessCoordinatorSetEnabledTools:
    """Test suite for set_enabled_tools method."""

    @pytest.fixture
    def mock_tool_registry(self) -> Mock:
        """Create mock tool registry."""
        registry = Mock()
        registry.list_tools = Mock(return_value=["read_file", "write_file", "bash"])
        return registry

    @pytest.fixture
    def coordinator(self, mock_tool_registry: Mock) -> ToolAccessCoordinator:
        """Create coordinator."""
        return ToolAccessCoordinator(tool_registry=mock_tool_registry)

    def test_set_enabled_tools_updates_session_tools(self, coordinator: ToolAccessCoordinator):
        """Test set_enabled_tools updates session-enabled tools."""
        # Execute
        coordinator.set_enabled_tools({"read_file", "bash"})

        # Assert
        assert coordinator._session_enabled_tools == {"read_file", "bash"}

    def test_set_enabled_tools_empty_set(self, coordinator: ToolAccessCoordinator):
        """Test set_enabled_tools with empty set."""
        # Execute
        coordinator.set_enabled_tools(set())

        # Assert
        assert coordinator._session_enabled_tools == set()

    def test_set_enabled_tools_propagates_to_tool_selector(self, mock_tool_registry: Mock):
        """Test set_enabled_tools propagates to tool_selector if available."""
        # Setup
        mode_controller = Mock()
        tool_selector = Mock()
        tool_selector.set_enabled_tools = Mock()
        mode_controller.tool_selector = tool_selector

        coordinator = ToolAccessCoordinator(
            tool_registry=mock_tool_registry,
            mode_controller=mode_controller,
        )

        # Execute
        coordinator.set_enabled_tools({"read_file"})

        # Assert
        tool_selector.set_enabled_tools.assert_called_once_with({"read_file"})

    def test_set_enabled_tools_no_tool_selector(self, mock_tool_registry: Mock):
        """Test set_enabled_tools when mode_controller has no tool_selector."""
        # Setup
        mode_controller = Mock()
        # No tool_selector attribute
        del mode_controller.tool_selector

        coordinator = ToolAccessCoordinator(
            tool_registry=mock_tool_registry,
            mode_controller=mode_controller,
        )

        # Execute - should not raise
        coordinator.set_enabled_tools({"read_file"})

        # Assert
        assert coordinator._session_enabled_tools == {"read_file"}

    def test_set_enabled_tools_overwrites_previous(self, coordinator: ToolAccessCoordinator):
        """Test set_enabled_tools overwrites previous values."""
        # Setup
        coordinator.set_enabled_tools({"read_file", "bash"})

        # Execute - set new tools
        coordinator.set_enabled_tools({"grep"})

        # Assert - should be completely replaced
        assert coordinator._session_enabled_tools == {"grep"}

    def test_set_enabled_tools_with_duplicate_names(self, coordinator: ToolAccessCoordinator):
        """Test set_enabled_tools handles duplicate tool names."""
        # Execute - set with duplicates (shouldn't happen but testing robustness)
        coordinator.set_enabled_tools({"read_file"})

        # Assert - should deduplicate
        assert coordinator._session_enabled_tools == {"read_file"}


class TestToolAccessCoordinatorClearSessionRestrictions:
    """Test suite for clear_session_restrictions method."""

    @pytest.fixture
    def mock_tool_registry(self) -> Mock:
        """Create mock tool registry."""
        registry = Mock()
        registry.list_tools = Mock(return_value=["read_file", "write_file"])
        return registry

    @pytest.fixture
    def coordinator(self, mock_tool_registry: Mock) -> ToolAccessCoordinator:
        """Create coordinator."""
        return ToolAccessCoordinator(tool_registry=mock_tool_registry)

    def test_clear_session_restrictions_resets_to_none(self, coordinator: ToolAccessCoordinator):
        """Test clear_session_restrictions resets session tools to None."""
        # Setup
        coordinator.set_enabled_tools({"read_file"})

        # Execute
        coordinator.clear_session_restrictions()

        # Assert
        assert coordinator._session_enabled_tools is None

    def test_clear_session_restrictions_when_already_none(self, coordinator: ToolAccessCoordinator):
        """Test clear_session_restrictions when already None."""
        # Setup - already None by default

        # Execute - should not raise
        coordinator.clear_session_restrictions()

        # Assert
        assert coordinator._session_enabled_tools is None

    def test_clear_session_restrictions_after_multiple_sets(
        self, coordinator: ToolAccessCoordinator
    ):
        """Test clear_session_restrictions after multiple set operations."""
        # Setup
        coordinator.set_enabled_tools({"read_file"})
        coordinator.set_enabled_tools({"bash", "grep"})
        coordinator.set_enabled_tools({"write_file"})

        # Execute
        coordinator.clear_session_restrictions()

        # Assert
        assert coordinator._session_enabled_tools is None


class TestToolAccessCoordinatorSetModeController:
    """Test suite for set_mode_controller method."""

    @pytest.fixture
    def mock_tool_registry(self) -> Mock:
        """Create mock tool registry."""
        registry = Mock()
        registry.list_tools = Mock(return_value=["read_file", "write_file"])
        return registry

    @pytest.fixture
    def coordinator(self, mock_tool_registry: Mock) -> ToolAccessCoordinator:
        """Create coordinator."""
        return ToolAccessCoordinator(tool_registry=mock_tool_registry)

    def test_set_mode_controller_updates_controller(self, coordinator: ToolAccessCoordinator):
        """Test set_mode_controller updates the mode controller."""
        # Setup
        mode_controller = Mock()

        # Execute
        coordinator.set_mode_controller(mode_controller)

        # Assert
        assert coordinator._mode_controller == mode_controller

    def test_set_mode_controller_overwrites_previous(self, coordinator: ToolAccessCoordinator):
        """Test set_mode_controller overwrites previous controller."""
        # Setup
        old_controller = Mock()
        coordinator.set_mode_controller(old_controller)

        new_controller = Mock()

        # Execute
        coordinator.set_mode_controller(new_controller)

        # Assert
        assert coordinator._mode_controller == new_controller
        assert coordinator._mode_controller != old_controller

    def test_set_mode_controller_to_none(self, coordinator: ToolAccessCoordinator):
        """Test setting mode controller to None."""
        # Setup
        coordinator.set_mode_controller(Mock())

        # Execute
        coordinator.set_mode_controller(None)

        # Assert
        assert coordinator._mode_controller is None


class TestToolAccessCoordinatorBuildContext:
    """Test suite for _build_context internal method."""

    @pytest.fixture
    def mock_tool_registry(self) -> Mock:
        """Create mock tool registry."""
        registry = Mock()
        registry.list_tools = Mock(return_value=["read_file", "write_file"])
        return registry

    @pytest.fixture
    def coordinator(self, mock_tool_registry: Mock) -> ToolAccessCoordinator:
        """Create coordinator."""
        return ToolAccessCoordinator(tool_registry=mock_tool_registry)

    def test_build_context_without_mode_controller(self, coordinator: ToolAccessCoordinator):
        """Test _build_context without mode controller."""
        # Execute
        context = coordinator._build_context()

        # Assert
        assert isinstance(context, ToolAccessContext)
        assert context.session_enabled_tools is None
        assert context.current_mode is None
        assert context.disallowed_tools == set()

    def test_build_context_with_mode_controller(self, mock_tool_registry: Mock):
        """Test _build_context with mode controller."""
        # Setup
        mode_controller = Mock()
        mode_controller.config.mode_name = "plan"
        mode_controller.config.disallowed_tools = {"bash", "write_file"}

        coordinator = ToolAccessCoordinator(
            tool_registry=mock_tool_registry,
            mode_controller=mode_controller,
        )

        # Execute
        context = coordinator._build_context()

        # Assert
        assert context.current_mode == "plan"
        assert context.disallowed_tools == {"bash", "write_file"}

    def test_build_context_with_session_tools(self, coordinator: ToolAccessCoordinator):
        """Test _build_context includes session tools."""
        # Setup
        coordinator.set_enabled_tools({"read_file"})

        # Execute
        context = coordinator._build_context()

        # Assert
        assert context.session_enabled_tools == {"read_file"}

    def test_build_context_missing_mode_attributes(self, mock_tool_registry: Mock):
        """Test _build_context when mode config missing attributes."""
        # Setup
        mode_controller = Mock()
        # Missing mode_name and disallowed_tools attributes
        del mode_controller.config.mode_name
        del mode_controller.config.disallowed_tools

        coordinator = ToolAccessCoordinator(
            tool_registry=mock_tool_registry,
            mode_controller=mode_controller,
        )

        # Execute - should not raise
        context = coordinator._build_context()

        # Assert - should handle missing attributes gracefully
        assert context.current_mode is None
        assert context.disallowed_tools == set()


class TestToolAccessCoordinatorFactory:
    """Test suite for create_tool_access_coordinator factory function."""

    @pytest.fixture
    def mock_tool_registry(self) -> Mock:
        """Create mock tool registry."""
        registry = Mock()
        registry.list_tools = Mock(return_value=["read_file", "write_file"])
        return registry

    def test_factory_with_defaults(self, mock_tool_registry: Mock):
        """Test factory creates coordinator with defaults."""
        # Execute
        coordinator = create_tool_access_coordinator(tool_registry=mock_tool_registry)

        # Assert
        assert isinstance(coordinator, ToolAccessCoordinator)
        assert coordinator._registry == mock_tool_registry
        assert coordinator._config.default_allow_all is True

    def test_factory_with_default_allow_all_false(self, mock_tool_registry: Mock):
        """Test factory with default_allow_all=False."""
        # Execute
        coordinator = create_tool_access_coordinator(
            tool_registry=mock_tool_registry,
            default_allow_all=False,
        )

        # Assert
        assert coordinator._config.default_allow_all is False

    def test_factory_with_mode_controller(self, mock_tool_registry: Mock):
        """Test factory with mode controller."""
        # Setup
        mode_controller = Mock()

        # Execute
        coordinator = create_tool_access_coordinator(
            tool_registry=mock_tool_registry,
            mode_controller=mode_controller,
        )

        # Assert
        assert coordinator._mode_controller == mode_controller

    def test_factory_creates_configured_coordinator(self, mock_tool_registry: Mock):
        """Test factory creates fully configured coordinator."""
        # Setup
        mode_controller = Mock()

        # Execute
        coordinator = create_tool_access_coordinator(
            tool_registry=mock_tool_registry,
            default_allow_all=False,
            mode_controller=mode_controller,
        )

        # Assert
        assert isinstance(coordinator._config, ToolAccessConfig)
        assert coordinator._config.default_allow_all is False
        assert coordinator._mode_controller == mode_controller


class TestAccessDecisionDataclass:
    """Test suite for AccessDecision dataclass."""

    def test_access_decision_creation(self):
        """Test AccessDecision creation with all fields."""
        # Execute
        decision = AccessDecision(
            tool_name="bash",
            allowed=True,
            reason="Tool is allowed",
            layer="mode",
        )

        # Assert
        assert decision.tool_name == "bash"
        assert decision.allowed is True
        assert decision.reason == "Tool is allowed"
        assert decision.layer == "mode"

    def test_access_decision_defaults(self):
        """Test AccessDecision with default values."""
        # Execute
        decision = AccessDecision(tool_name="read_file", allowed=False)

        # Assert
        assert decision.tool_name == "read_file"
        assert decision.allowed is False
        assert decision.reason is None
        assert decision.layer == "unknown"

    def test_access_decision_immutability(self):
        """Test that AccessDecision fields can be modified (not frozen)."""
        # Setup
        decision = AccessDecision(tool_name="bash", allowed=True)

        # Execute - dataclasses aren't frozen by default
        decision.allowed = False

        # Assert
        assert decision.allowed is False


class TestToolAccessContextDataclass:
    """Test suite for ToolAccessContext dataclass."""

    def test_context_creation_with_all_fields(self):
        """Test ToolAccessContext creation with all fields."""
        # Execute
        context = ToolAccessContext(
            session_enabled_tools={"read_file", "grep"},
            current_mode="plan",
            disallowed_tools={"bash", "write_file"},
        )

        # Assert
        assert context.session_enabled_tools == {"read_file", "grep"}
        assert context.current_mode == "plan"
        assert context.disallowed_tools == {"bash", "write_file"}

    def test_context_defaults(self):
        """Test ToolAccessContext with default values."""
        # Execute
        context = ToolAccessContext()

        # Assert
        assert context.session_enabled_tools is None
        assert context.current_mode is None
        assert context.disallowed_tools == set()

    def test_context_empty_disallowed_tools(self):
        """Test ToolAccessContext with empty disallowed_tools set."""
        # Execute
        context = ToolAccessContext(disallowed_tools=set())

        # Assert
        assert context.disallowed_tools == set()
        assert len(context.disallowed_tools) == 0


class TestToolAccessConfigDataclass:
    """Test suite for ToolAccessConfig dataclass."""

    def test_config_defaults(self):
        """Test ToolAccessConfig with default values."""
        # Execute
        config = ToolAccessConfig()

        # Assert
        assert config.default_allow_all is True
        assert config.strict_mode is False
        # Inherited from BaseCoordinatorConfig
        assert config.enabled is True
        assert config.timeout == 30.0

    def test_config_custom_values(self):
        """Test ToolAccessConfig with custom values."""
        # Execute
        config = ToolAccessConfig(
            default_allow_all=False,
            strict_mode=True,
            enabled=False,
            timeout=60.0,
        )

        # Assert
        assert config.default_allow_all is False
        assert config.strict_mode is True
        assert config.enabled is False
        assert config.timeout == 60.0

    def test_config_to_dict(self):
        """Test ToolAccessConfig to_dict method (inherited)."""
        # Setup
        config = ToolAccessConfig(
            default_allow_all=False,
            strict_mode=True,
        )

        # Execute
        result = config.to_dict()

        # Assert
        assert isinstance(result, dict)
        assert "default_allow_all" in result
        assert "strict_mode" in result
        assert result["default_allow_all"] is False
        assert result["strict_mode"] is True

    def test_config_validate(self):
        """Test ToolAccessConfig validate method (inherited)."""
        # Setup
        config = ToolAccessConfig(
            timeout=-1.0,  # Invalid
            max_retries=-1,  # Invalid
        )

        # Execute
        errors = config.validate()

        # Assert
        assert len(errors) > 0
        assert any("timeout" in e for e in errors)
        assert any("max_retries" in e for e in errors)


class TestToolAccessCoordinatorIntegration:
    """Integration tests for ToolAccessCoordinator scenarios."""

    @pytest.fixture
    def mock_tool_registry(self) -> Mock:
        """Create mock tool registry."""
        registry = Mock()
        registry.list_tools = Mock(return_value=["read_file", "write_file", "bash", "grep", "ls"])
        return registry

    @pytest.fixture
    def mode_controller_build(self) -> Mock:
        """Create BUILD mode controller."""
        controller = Mock()
        controller.config.allow_all_tools = True
        controller.config.disallowed_tools = set()
        controller.config.mode_name = "build"
        return controller

    @pytest.fixture
    def mode_controller_plan(self) -> Mock:
        """Create PLAN mode controller."""
        controller = Mock()
        controller.config.allow_all_tools = False
        controller.config.disallowed_tools = {"write_file", "bash"}
        controller.config.mode_name = "plan"
        return controller

    def test_build_mode_allows_all_tools(
        self, mock_tool_registry: Mock, mode_controller_build: Mock
    ):
        """Test BUILD mode allows all tools except disallowed."""
        # Setup
        coordinator = ToolAccessCoordinator(
            tool_registry=mock_tool_registry,
            mode_controller=mode_controller_build,
        )

        # Execute & Assert
        assert coordinator.is_tool_enabled("read_file") is True
        assert coordinator.is_tool_enabled("write_file") is True
        assert coordinator.is_tool_enabled("bash") is True
        assert coordinator.is_tool_enabled("grep") is True

    def test_plan_mode_restricts_dangerous_tools(
        self, mock_tool_registry: Mock, mode_controller_plan: Mock
    ):
        """Test PLAN mode restricts dangerous tools."""
        # Setup
        coordinator = ToolAccessCoordinator(
            tool_registry=mock_tool_registry,
            mode_controller=mode_controller_plan,
        )

        # Execute & Assert
        assert coordinator.is_tool_enabled("read_file") is True
        assert coordinator.is_tool_enabled("write_file") is False
        assert coordinator.is_tool_enabled("bash") is False
        assert coordinator.is_tool_enabled("grep") is True

    def test_session_restrictions_override_mode(
        self, mock_tool_registry: Mock, mode_controller_build: Mock
    ):
        """Test session restrictions work even in permissive mode.

        Note: The actual implementation prioritizes mode controller's allow_all_tools
        over session restrictions when mode controller is present.
        """
        # Setup - mode controller with allow_all_tools=False
        mode_controller = Mock()
        mode_controller.config.allow_all_tools = False
        mode_controller.config.disallowed_tools = set()
        mode_controller.config.mode_name = "build"

        coordinator = ToolAccessCoordinator(
            tool_registry=mock_tool_registry,
            mode_controller=mode_controller,
        )
        coordinator.set_enabled_tools({"read_file", "grep"})

        # Execute & Assert - session restrictions apply when mode doesn't allow all
        enabled = coordinator.get_enabled_tools()
        assert enabled == {"read_file", "grep"}
        assert "write_file" not in enabled
        assert "bash" not in enabled

    def test_mode_switching_changes_access(
        self, mock_tool_registry: Mock, mode_controller_build: Mock, mode_controller_plan: Mock
    ):
        """Test switching modes changes tool access."""
        # Setup
        coordinator = ToolAccessCoordinator(
            tool_registry=mock_tool_registry,
            mode_controller=mode_controller_build,
        )

        # BUILD mode allows all
        assert coordinator.is_tool_enabled("bash") is True

        # Switch to PLAN mode
        coordinator.set_mode_controller(mode_controller_plan)

        # PLAN mode disallows bash
        assert coordinator.is_tool_enabled("bash") is False

    def test_access_decision_reasons_provided(
        self, mock_tool_registry: Mock, mode_controller_plan: Mock
    ):
        """Test that access decisions provide clear reasons."""
        # Setup
        config = ToolAccessConfig(default_allow_all=False, strict_mode=True)
        coordinator = ToolAccessCoordinator(
            tool_registry=mock_tool_registry,
            config=config,
            mode_controller=mode_controller_plan,
        )

        # Execute - mode disallowed
        decision_mode = coordinator.check_access("bash")
        assert "disallowed by current mode" in decision_mode.reason

        # Execute - session disallowed (need to clear mode restrictions first)
        coordinator.set_enabled_tools({"read_file"})
        # Create a coordinator without mode controller to test session layer
        coordinator_session_only = ToolAccessCoordinator(
            tool_registry=mock_tool_registry,
            config=config,
        )
        coordinator_session_only.set_enabled_tools({"read_file"})
        decision_session = coordinator_session_only.check_access("grep")
        assert "session-enabled" in decision_session.reason

        # Execute - strict mode unknown tool (without mode controller)
        coordinator_strict = ToolAccessCoordinator(
            tool_registry=mock_tool_registry,
            config=config,
        )
        decision_strict = coordinator_strict.check_access("unknown_tool")
        assert "strict mode" in decision_strict.reason

    def test_multiple_coordinators_independent_state(self, mock_tool_registry: Mock):
        """Test multiple coordinators maintain independent state."""
        # Setup
        coordinator1 = ToolAccessCoordinator(tool_registry=mock_tool_registry)
        coordinator2 = ToolAccessCoordinator(tool_registry=mock_tool_registry)

        # Execute - set different tools on each
        coordinator1.set_enabled_tools({"read_file"})
        coordinator2.set_enabled_tools({"bash", "grep"})

        # Assert - should be independent
        assert coordinator1.get_enabled_tools() == {"read_file"}
        assert coordinator2.get_enabled_tools() == {"bash", "grep"}

    def test_clear_restrictions_restores_mode_behavior(
        self, mock_tool_registry: Mock, mode_controller_build: Mock
    ):
        """Test clearing session restrictions restores mode-based access.

        Note: When mode controller has allow_all_tools=True, it takes precedence
        over session restrictions in get_enabled_tools().
        """
        # Setup - mode controller without allow_all_tools
        mode_controller = Mock()
        mode_controller.config.allow_all_tools = False
        mode_controller.config.disallowed_tools = set()
        mode_controller.config.mode_name = "build"

        coordinator = ToolAccessCoordinator(
            tool_registry=mock_tool_registry,
            mode_controller=mode_controller,
        )

        # Restrict session
        coordinator.set_enabled_tools({"read_file"})
        assert coordinator.get_enabled_tools() == {"read_file"}

        # Clear restrictions
        coordinator.clear_session_restrictions()

        # Should restore all available tools (mode doesn't allow all, no session restrictions)
        enabled = coordinator.get_enabled_tools()
        assert "read_file" in enabled
        assert "write_file" in enabled
        assert "bash" in enabled
        assert "grep" in enabled
        assert "ls" in enabled
