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

"""Unit tests for ModeControllerProtocol and ModeControllerAdapter.

TDD-first tests for Phase 1.1: Integrate AgentModeController with protocol.
These tests verify:
1. Protocol compliance
2. Adapter delegation to underlying AgentModeController
3. DI registration and resolution
4. Backward compatibility with existing orchestrator.current_mode property
"""

import pytest
from typing import Any
from unittest.mock import Mock, MagicMock, patch

from victor.agent.mode_controller import (
    AgentMode,
    AgentModeController,
    OperationalModeConfig,
    MODE_CONFIGS,
)


class TestModeControllerProtocol:
    """Tests for ModeControllerProtocol interface compliance."""

    def test_protocol_is_runtime_checkable(self):
        """Protocol should be runtime checkable for isinstance checks."""
        from victor.protocols.mode_controller import ModeControllerProtocol

        # Should be able to check isinstance
        assert hasattr(ModeControllerProtocol, "__protocol_attrs__") or hasattr(
            ModeControllerProtocol, "_is_protocol"
        )

    def test_agent_mode_controller_implements_protocol(self):
        """AgentModeController should implement ModeControllerProtocol."""
        from victor.protocols.mode_controller import ModeControllerProtocol

        controller = AgentModeController()

        # Verify required attributes exist
        assert hasattr(controller, "current_mode")
        assert hasattr(controller, "switch_mode")
        assert hasattr(controller, "is_tool_allowed")
        assert hasattr(controller, "get_exploration_multiplier")

    def test_protocol_methods_match_implementation(self):
        """Protocol methods should match AgentModeController implementation."""
        from victor.protocols.mode_controller import ModeControllerProtocol

        controller = AgentModeController(initial_mode=AgentMode.BUILD)

        # Test current_mode property
        assert controller.current_mode == AgentMode.BUILD

        # Test switch_mode
        result = controller.switch_mode(AgentMode.PLAN)
        assert result is True
        assert controller.current_mode == AgentMode.PLAN

        # Test is_tool_allowed
        assert controller.is_tool_allowed("read_file") is True
        assert controller.is_tool_allowed("bash") is False  # Disallowed in PLAN

        # Test get_exploration_multiplier
        multiplier = controller.get_exploration_multiplier()
        assert isinstance(multiplier, float)
        assert multiplier > 0


class TestModeControllerAdapter:
    """Tests for ModeControllerAdapter DI integration."""

    def test_adapter_wraps_controller(self):
        """Adapter should wrap and delegate to AgentModeController."""
        from victor.agent.coordinators.mode_adapter import ModeControllerAdapter

        controller = AgentModeController(initial_mode=AgentMode.EXPLORE)
        adapter = ModeControllerAdapter(controller)

        assert adapter.current_mode == AgentMode.EXPLORE

    def test_adapter_delegates_switch_mode(self):
        """Adapter should delegate switch_mode to underlying controller."""
        from victor.agent.coordinators.mode_adapter import ModeControllerAdapter

        controller = AgentModeController(initial_mode=AgentMode.BUILD)
        adapter = ModeControllerAdapter(controller)

        result = adapter.switch_mode(AgentMode.PLAN)

        assert result is True
        assert adapter.current_mode == AgentMode.PLAN
        assert controller.current_mode == AgentMode.PLAN

    def test_adapter_delegates_is_tool_allowed(self):
        """Adapter should delegate is_tool_allowed to controller."""
        from victor.agent.coordinators.mode_adapter import ModeControllerAdapter

        controller = AgentModeController(initial_mode=AgentMode.BUILD)
        adapter = ModeControllerAdapter(controller)

        # BUILD mode allows all tools
        assert adapter.is_tool_allowed("bash") is True
        assert adapter.is_tool_allowed("edit_files") is True

        # Switch to EXPLORE mode
        adapter.switch_mode(AgentMode.EXPLORE)

        # EXPLORE mode restricts certain tools
        assert adapter.is_tool_allowed("read_file") is True
        assert adapter.is_tool_allowed("bash") is False

    def test_adapter_delegates_get_exploration_multiplier(self):
        """Adapter should delegate get_exploration_multiplier to controller."""
        from victor.agent.coordinators.mode_adapter import ModeControllerAdapter

        controller = AgentModeController(initial_mode=AgentMode.BUILD)
        adapter = ModeControllerAdapter(controller)

        # BUILD mode has 5.0x multiplier
        assert adapter.get_exploration_multiplier() == 5.0

        # PLAN mode has 10.0x multiplier
        adapter.switch_mode(AgentMode.PLAN)
        assert adapter.get_exploration_multiplier() == 10.0

        # EXPLORE mode has 20.0x multiplier
        adapter.switch_mode(AgentMode.EXPLORE)
        assert adapter.get_exploration_multiplier() == 20.0

    def test_adapter_provides_config_access(self):
        """Adapter should provide access to mode configuration."""
        from victor.agent.coordinators.mode_adapter import ModeControllerAdapter

        controller = AgentModeController(initial_mode=AgentMode.BUILD)
        adapter = ModeControllerAdapter(controller)

        config = adapter.config

        assert isinstance(config, OperationalModeConfig)
        assert config.name == "Build"
        assert config.allow_all_tools is True

    def test_adapter_provides_system_prompt_addition(self):
        """Adapter should provide system prompt addition for current mode."""
        from victor.agent.coordinators.mode_adapter import ModeControllerAdapter

        controller = AgentModeController(initial_mode=AgentMode.BUILD)
        adapter = ModeControllerAdapter(controller)

        prompt = adapter.get_system_prompt_addition()

        assert isinstance(prompt, str)
        assert "BUILD mode" in prompt

    def test_adapter_provides_tool_priority(self):
        """Adapter should provide tool priority adjustment."""
        from victor.agent.coordinators.mode_adapter import ModeControllerAdapter

        controller = AgentModeController(initial_mode=AgentMode.BUILD)
        adapter = ModeControllerAdapter(controller)

        # BUILD mode boosts edit tools
        assert adapter.get_tool_priority("edit") == 1.5
        assert adapter.get_tool_priority("edit_files") == 1.5
        assert adapter.get_tool_priority("unknown_tool") == 1.0

    def test_adapter_callback_registration(self):
        """Adapter should support mode change callbacks."""
        from victor.agent.coordinators.mode_adapter import ModeControllerAdapter

        controller = AgentModeController(initial_mode=AgentMode.BUILD)
        adapter = ModeControllerAdapter(controller)

        callback_calls = []

        def on_mode_change(old_mode: AgentMode, new_mode: AgentMode):
            callback_calls.append((old_mode, new_mode))

        adapter.register_callback(on_mode_change)
        adapter.switch_mode(AgentMode.PLAN)

        assert len(callback_calls) == 1
        assert callback_calls[0] == (AgentMode.BUILD, AgentMode.PLAN)


class TestModeControllerDIRegistration:
    """Tests for DI container registration."""

    def test_protocol_registered_in_container(self):
        """ModeControllerProtocol should be registered in DI container."""
        from victor.protocols.mode_controller import ModeControllerProtocol
        from victor.core.container import ServiceContainer
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.config.settings import Settings

        container = ServiceContainer()
        settings = Settings()
        configure_orchestrator_services(container, settings)

        assert container.is_registered(ModeControllerProtocol)

    def test_resolve_mode_controller_from_container(self):
        """Should be able to resolve ModeControllerProtocol from container."""
        from victor.protocols.mode_controller import ModeControllerProtocol
        from victor.core.container import ServiceContainer
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.config.settings import Settings

        container = ServiceContainer()
        settings = Settings()
        configure_orchestrator_services(container, settings)

        controller = container.get(ModeControllerProtocol)

        assert controller is not None
        assert hasattr(controller, "current_mode")
        assert hasattr(controller, "switch_mode")

    def test_mode_controller_singleton_scope(self):
        """Mode controller should be registered as singleton."""
        from victor.protocols.mode_controller import ModeControllerProtocol
        from victor.core.container import ServiceContainer
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.config.settings import Settings

        container = ServiceContainer()
        settings = Settings()
        configure_orchestrator_services(container, settings)

        controller1 = container.get(ModeControllerProtocol)
        controller2 = container.get(ModeControllerProtocol)

        assert controller1 is controller2


class TestModeControllerIntegration:
    """Integration tests for mode controller with orchestrator."""

    def test_adapter_status_reporting(self):
        """Adapter should provide status information."""
        from victor.agent.coordinators.mode_adapter import ModeControllerAdapter

        controller = AgentModeController(initial_mode=AgentMode.PLAN)
        adapter = ModeControllerAdapter(controller)

        status = adapter.get_status()

        assert status["mode"] == "plan"
        assert status["name"] == "Plan"
        assert "description" in status
        assert "write_confirmation_required" in status

    def test_adapter_mode_list(self):
        """Adapter should provide list of available modes."""
        from victor.agent.coordinators.mode_adapter import ModeControllerAdapter

        controller = AgentModeController(initial_mode=AgentMode.BUILD)
        adapter = ModeControllerAdapter(controller)

        modes = adapter.get_mode_list()

        assert len(modes) == 3
        mode_names = [m["mode"] for m in modes]
        assert "build" in mode_names
        assert "plan" in mode_names
        assert "explore" in mode_names

        # Current mode should be marked
        current = next(m for m in modes if m["current"])
        assert current["mode"] == "build"

    def test_adapter_sandbox_settings(self):
        """Adapter should provide sandbox settings for restricted modes."""
        from victor.agent.coordinators.mode_adapter import ModeControllerAdapter

        controller = AgentModeController(initial_mode=AgentMode.PLAN)
        adapter = ModeControllerAdapter(controller)

        assert adapter.sandbox_dir == ".victor/sandbox"
        assert adapter.allow_sandbox_edits is True

        # BUILD mode has no sandbox
        adapter.switch_mode(AgentMode.BUILD)
        assert adapter.sandbox_dir is None

    def test_adapter_write_confirmation_required(self):
        """Adapter should indicate when write confirmation is required."""
        from victor.agent.coordinators.mode_adapter import ModeControllerAdapter

        controller = AgentModeController(initial_mode=AgentMode.BUILD)
        adapter = ModeControllerAdapter(controller)

        # BUILD mode doesn't require confirmation
        assert adapter.require_write_confirmation is False

        # PLAN mode requires confirmation
        adapter.switch_mode(AgentMode.PLAN)
        assert adapter.require_write_confirmation is True

    def test_adapter_max_files_per_operation(self):
        """Adapter should provide max files limit for operations."""
        from victor.agent.coordinators.mode_adapter import ModeControllerAdapter

        controller = AgentModeController(initial_mode=AgentMode.BUILD)
        adapter = ModeControllerAdapter(controller)

        # BUILD mode has no limit (0 = unlimited)
        assert adapter.max_files_per_operation == 0

        # PLAN mode has limited edits
        adapter.switch_mode(AgentMode.PLAN)
        assert adapter.max_files_per_operation == 5


class TestModeControllerBackwardCompatibility:
    """Tests ensuring backward compatibility."""

    def test_get_mode_controller_returns_adapter_compatible(self):
        """get_mode_controller() should return protocol-compatible instance."""
        from victor.agent.mode_controller import get_mode_controller, reset_mode_controller
        from victor.protocols.mode_controller import ModeControllerProtocol

        reset_mode_controller()
        controller = get_mode_controller()

        # Should implement protocol interface
        assert hasattr(controller, "current_mode")
        assert hasattr(controller, "switch_mode")
        assert hasattr(controller, "is_tool_allowed")
        assert hasattr(controller, "get_exploration_multiplier")

    def test_mode_switch_same_mode_returns_true(self):
        """Switching to current mode should return True (no-op)."""
        from victor.agent.coordinators.mode_adapter import ModeControllerAdapter

        controller = AgentModeController(initial_mode=AgentMode.BUILD)
        adapter = ModeControllerAdapter(controller)

        result = adapter.switch_mode(AgentMode.BUILD)

        assert result is True
        assert adapter.current_mode == AgentMode.BUILD

    def test_previous_mode_navigation(self):
        """Adapter should support navigating to previous mode."""
        from victor.agent.coordinators.mode_adapter import ModeControllerAdapter

        controller = AgentModeController(initial_mode=AgentMode.BUILD)
        adapter = ModeControllerAdapter(controller)

        adapter.switch_mode(AgentMode.PLAN)
        adapter.switch_mode(AgentMode.EXPLORE)

        prev = adapter.previous_mode()

        assert prev == AgentMode.PLAN
        assert adapter.current_mode == AgentMode.PLAN


class TestGetExplorationMultiplier:
    """Tests for get_exploration_multiplier method."""

    def test_build_mode_multiplier(self):
        """BUILD mode should have 5.0x exploration multiplier."""
        controller = AgentModeController(initial_mode=AgentMode.BUILD)

        # Access through config
        multiplier = controller.config.exploration_multiplier
        assert multiplier == 5.0

    def test_plan_mode_multiplier(self):
        """PLAN mode should have 10.0x exploration multiplier."""
        controller = AgentModeController(initial_mode=AgentMode.PLAN)

        multiplier = controller.config.exploration_multiplier
        assert multiplier == 10.0

    def test_explore_mode_multiplier(self):
        """EXPLORE mode should have 20.0x exploration multiplier."""
        controller = AgentModeController(initial_mode=AgentMode.EXPLORE)

        multiplier = controller.config.exploration_multiplier
        assert multiplier == 20.0
