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

"""Tests for the agent modes system."""

import pytest

from victor.agent.mode_controller import (
    AgentMode,
    AgentModeController,
    MODE_CONFIGS,
    get_mode_controller,
    reset_mode_controller,
)
from victor.core.container import reset_container


@pytest.fixture
def manager():
    """Create a fresh mode manager for each test."""
    reset_mode_controller()
    return AgentModeController()


class TestAgentMode:
    """Tests for AgentMode enum."""

    def test_mode_values(self):
        """Test that all modes have correct values."""
        assert AgentMode.BUILD.value == "build"
        assert AgentMode.PLAN.value == "plan"
        assert AgentMode.EXPLORE.value == "explore"

    def test_mode_from_string(self):
        """Test creating modes from strings."""
        assert AgentMode("build") == AgentMode.BUILD
        assert AgentMode("plan") == AgentMode.PLAN
        assert AgentMode("explore") == AgentMode.EXPLORE

    def test_invalid_mode_raises(self):
        """Test that invalid mode name raises ValueError."""
        with pytest.raises(ValueError):
            AgentMode("invalid")


class TestModeConfig:
    """Tests for ModeConfig dataclass."""

    def test_build_mode_config(self):
        """Test BUILD mode configuration."""
        config = MODE_CONFIGS[AgentMode.BUILD]

        assert config.name == "Build"
        assert config.allow_all_tools is True
        assert config.require_write_confirmation is False
        assert config.verbose_planning is False

    def test_plan_mode_config(self):
        """Test PLAN mode configuration."""
        config = MODE_CONFIGS[AgentMode.PLAN]

        assert config.name == "Plan"
        assert config.allow_all_tools is False
        assert config.require_write_confirmation is True
        assert config.verbose_planning is True
        # Should disallow file modifications
        assert "write_file" in config.disallowed_tools
        assert "edit_files" in config.disallowed_tools
        # Should allow read operations
        assert "read_file" in config.allowed_tools
        assert "code_search" in config.allowed_tools

    def test_explore_mode_config(self):
        """Test EXPLORE mode configuration."""
        config = MODE_CONFIGS[AgentMode.EXPLORE]

        assert config.name == "Explore"
        assert config.allow_all_tools is False
        assert config.require_write_confirmation is True
        # Should disallow modifications
        assert "write_file" in config.disallowed_tools
        assert "bash" in config.disallowed_tools
        # Should allow read operations
        assert "read_file" in config.allowed_tools


class TestAgentModeController:
    """Tests for AgentModeController class."""

    def test_initial_mode(self, manager):
        """Test that default mode is BUILD."""
        assert manager.current_mode == AgentMode.BUILD

    def test_custom_initial_mode(self):
        """Test creating manager with different initial mode."""
        reset_mode_controller()
        manager = AgentModeController(initial_mode=AgentMode.PLAN)
        assert manager.current_mode == AgentMode.PLAN

    def test_switch_mode(self, manager):
        """Test switching modes."""
        assert manager.current_mode == AgentMode.BUILD

        result = manager.switch_mode(AgentMode.PLAN)

        assert result is True
        assert manager.current_mode == AgentMode.PLAN

    def test_switch_to_same_mode(self, manager):
        """Test switching to the current mode."""
        assert manager.current_mode == AgentMode.BUILD

        result = manager.switch_mode(AgentMode.BUILD)

        assert result is True  # Should succeed but no change
        assert manager.current_mode == AgentMode.BUILD

    def test_mode_history(self, manager):
        """Test that mode history is tracked."""
        manager.switch_mode(AgentMode.PLAN)
        manager.switch_mode(AgentMode.EXPLORE)

        assert len(manager._mode_history) == 3
        assert manager._mode_history[0] == AgentMode.BUILD
        assert manager._mode_history[1] == AgentMode.PLAN
        assert manager._mode_history[2] == AgentMode.EXPLORE

    def test_previous_mode(self, manager):
        """Test returning to previous mode."""
        manager.switch_mode(AgentMode.PLAN)
        manager.switch_mode(AgentMode.EXPLORE)

        prev = manager.previous_mode()

        assert prev == AgentMode.PLAN
        assert manager.current_mode == AgentMode.PLAN

    def test_previous_mode_no_history(self, manager):
        """Test previous_mode when at initial mode."""
        prev = manager.previous_mode()
        assert prev is None

    def test_get_config(self, manager):
        """Test getting current mode config."""
        config = manager.config

        assert config.name == "Build"
        assert config.allow_all_tools is True


class TestToolAllowance:
    """Tests for tool allowance in different modes."""

    def test_build_mode_allows_all_tools(self, manager):
        """Test that BUILD mode allows all tools."""
        assert manager.is_tool_allowed("write_file")
        assert manager.is_tool_allowed("edit_files")
        assert manager.is_tool_allowed("bash")
        assert manager.is_tool_allowed("read_file")

    def test_plan_mode_restricts_tools(self, manager):
        """Test that PLAN mode restricts modification tools."""
        manager.switch_mode(AgentMode.PLAN)

        # Should allow read tools
        assert manager.is_tool_allowed("read_file")
        assert manager.is_tool_allowed("code_search")
        assert manager.is_tool_allowed("git_status")

        # Should disallow write tools
        assert not manager.is_tool_allowed("write_file")
        assert not manager.is_tool_allowed("edit_files")
        assert not manager.is_tool_allowed("bash")
        assert not manager.is_tool_allowed("git_commit")

    def test_explore_mode_restricts_tools(self, manager):
        """Test that EXPLORE mode restricts modification tools."""
        manager.switch_mode(AgentMode.EXPLORE)

        # Should allow read tools
        assert manager.is_tool_allowed("read_file")
        assert manager.is_tool_allowed("list_directory")
        assert manager.is_tool_allowed("web_search")

        # Should disallow write tools
        assert not manager.is_tool_allowed("write_file")
        assert not manager.is_tool_allowed("bash")


class TestToolPriority:
    """Tests for tool priority adjustments."""

    def test_build_mode_priorities(self, manager):
        """Test tool priorities in BUILD mode."""
        # edit_files should have higher priority
        assert manager.get_tool_priority("edit_files") > 1.0
        # Unknown tool should have default priority
        assert manager.get_tool_priority("unknown_tool") == 1.0

    def test_plan_mode_priorities(self, manager):
        """Test tool priorities in PLAN mode."""
        manager.switch_mode(AgentMode.PLAN)

        # code_search should have higher priority
        assert manager.get_tool_priority("code_search") > 1.0
        assert manager.get_tool_priority("plan_files") > 1.0


class TestModeCallbacks:
    """Tests for mode change callbacks."""

    def test_register_callback(self, manager):
        """Test registering a callback."""
        called = []

        def callback(old_mode, new_mode):
            called.append((old_mode, new_mode))

        manager.register_callback(callback)
        manager.switch_mode(AgentMode.PLAN)

        assert len(called) == 1
        assert called[0] == (AgentMode.BUILD, AgentMode.PLAN)

    def test_multiple_callbacks(self, manager):
        """Test multiple callbacks are called."""
        results1 = []
        results2 = []

        manager.register_callback(lambda o, n: results1.append(n))
        manager.register_callback(lambda o, n: results2.append(n))

        manager.switch_mode(AgentMode.EXPLORE)

        assert len(results1) == 1
        assert len(results2) == 1
        assert results1[0] == AgentMode.EXPLORE


class TestModeStatus:
    """Tests for mode status methods."""

    def test_get_status(self, manager):
        """Test getting mode status."""
        status = manager.get_status()

        assert status["mode"] == "build"
        assert status["name"] == "Build"
        assert "description" in status
        assert status["write_confirmation_required"] is False

    def test_get_mode_list(self, manager):
        """Test getting list of modes."""
        modes = manager.get_mode_list()

        assert len(modes) == 3  # BUILD, PLAN, EXPLORE

        mode_names = [m["mode"] for m in modes]
        assert "build" in mode_names
        assert "plan" in mode_names
        assert "explore" in mode_names

        # Check current mode is marked
        build_mode = next(m for m in modes if m["mode"] == "build")
        assert build_mode["current"] is True

        plan_mode = next(m for m in modes if m["mode"] == "plan")
        assert plan_mode["current"] is False


class TestGlobalModeController:
    """Tests for global mode controller functions."""

    def test_get_mode_manager_singleton(self):
        """Test that get_mode_controller returns singleton."""
        reset_mode_controller()
        manager1 = get_mode_controller()
        manager2 = get_mode_controller()

        assert manager1 is manager2

    def test_reset_mode_manager(self):
        """Test resetting the mode controller."""
        reset_container()
        reset_mode_controller()
        manager1 = get_mode_controller()
        reset_container()
        reset_mode_controller()
        manager2 = get_mode_controller()

        assert manager1 is not manager2


class TestSystemPromptAddition:
    """Tests for system prompt additions."""

    def test_build_mode_prompt(self, manager):
        """Test BUILD mode system prompt addition."""
        prompt = manager.get_system_prompt_addition()

        assert "BUILD mode" in prompt
        assert "implementation" in prompt.lower()

    def test_plan_mode_prompt(self, manager):
        """Test PLAN mode system prompt addition."""
        manager.switch_mode(AgentMode.PLAN)
        prompt = manager.get_system_prompt_addition()

        assert "PLAN mode" in prompt
        assert "DO NOT modify" in prompt
        assert "/mode build" in prompt

    def test_explore_mode_prompt(self, manager):
        """Test EXPLORE mode system prompt addition."""
        manager.switch_mode(AgentMode.EXPLORE)
        prompt = manager.get_system_prompt_addition()

        assert "EXPLORE mode" in prompt
        assert "DO NOT modify" in prompt
