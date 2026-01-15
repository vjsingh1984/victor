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

"""Tests for ModeCoordinator implementation.

Tests the ModeCoordinator which consolidates scattered mode logic
from the orchestrator into a single, focused coordinator.
"""

import pytest

from victor.agent.mode_controller import AgentMode, AgentModeController


class TestModeCoordinator:
    """Tests for ModeCoordinator implementation."""

    @pytest.mark.asyncio
    async def test_get_current_mode(self, coordinator):
        """Test getting current mode."""
        mode = coordinator.get_current_mode()
        assert mode == AgentMode.BUILD

    @pytest.mark.asyncio
    async def test_get_mode_config(self, coordinator):
        """Test getting current mode configuration."""
        config = coordinator.get_mode_config()
        assert config is not None
        assert config.name == "Build"
        assert config.allow_all_tools is True

    @pytest.mark.asyncio
    async def test_is_tool_allowed_build_mode(self, coordinator):
        """Test tool access check in BUILD mode."""
        # BUILD mode allows all tools (except disallowed)
        assert coordinator.is_tool_allowed("read_file") is True
        assert coordinator.is_tool_allowed("edit_files") is True
        assert coordinator.is_tool_allowed("bash") is True

    @pytest.mark.asyncio
    async def test_is_tool_allowed_plan_mode(self, coordinator_plan_mode):
        """Test tool access check in PLAN mode."""
        # PLAN mode allows read-only tools
        assert coordinator_plan_mode.is_tool_allowed("read_file") is True
        assert coordinator_plan_mode.is_tool_allowed("code_search") is True

        # But disallows bash in PLAN mode
        assert coordinator_plan_mode.is_tool_allowed("bash") is False

    @pytest.mark.asyncio
    async def test_is_tool_allowed_explore_mode(self, coordinator_explore_mode):
        """Test tool access check in EXPLORE mode."""
        # EXPLORE mode allows read-only tools
        assert coordinator_explore_mode.is_tool_allowed("read_file") is True
        assert coordinator_explore_mode.is_tool_allowed("code_search") is True

        # But disallows editing
        assert coordinator_explore_mode.is_tool_allowed("edit_files") is False
        assert coordinator_explore_mode.is_tool_allowed("bash") is False

    @pytest.mark.asyncio
    async def test_get_tool_priority(self, coordinator):
        """Test getting tool priority adjustment."""
        # edit_files should have boosted priority in BUILD mode
        priority = coordinator.get_tool_priority("edit_files")
        assert priority > 1.0

        # read_file should have slightly lower priority in BUILD mode
        priority = coordinator.get_tool_priority("read_file")
        assert priority < 1.0

    @pytest.mark.asyncio
    async def test_get_tool_priority_default(self, coordinator):
        """Test default tool priority for unconfigured tools."""
        # Unconfigured tools should have neutral priority
        priority = coordinator.get_tool_priority("some_random_tool")
        assert priority == 1.0

    @pytest.mark.asyncio
    async def test_get_exploration_multiplier(self, coordinator):
        """Test getting exploration multiplier."""
        multiplier = coordinator.get_exploration_multiplier()
        assert multiplier > 0
        # BUILD mode has 5x exploration
        assert multiplier == 5.0

    @pytest.mark.asyncio
    async def test_get_exploration_multiplier_plan_mode(self, coordinator_plan_mode):
        """Test exploration multiplier in PLAN mode."""
        multiplier = coordinator_plan_mode.get_exploration_multiplier()
        # PLAN mode has 10x exploration
        assert multiplier == 10.0

    @pytest.mark.asyncio
    async def test_switch_mode(self, coordinator):
        """Test switching modes."""
        # Start in BUILD mode
        assert coordinator.get_current_mode() == AgentMode.BUILD

        # Switch to PLAN mode
        result = coordinator.switch_mode(AgentMode.PLAN)
        assert result is True
        assert coordinator.get_current_mode() == AgentMode.PLAN

        # Switch to EXPLORE mode
        result = coordinator.switch_mode(AgentMode.EXPLORE)
        assert result is True
        assert coordinator.get_current_mode() == AgentMode.EXPLORE

    @pytest.mark.asyncio
    async def test_switch_to_same_mode(self, coordinator):
        """Test switching to the same mode is idempotent."""
        result = coordinator.switch_mode(AgentMode.BUILD)
        assert result is True
        assert coordinator.get_current_mode() == AgentMode.BUILD

    @pytest.mark.asyncio
    async def test_get_mode_status(self, coordinator):
        """Test getting comprehensive mode status."""
        status = coordinator.get_mode_status()
        assert "mode" in status
        assert "name" in status
        assert "description" in status
        assert status["mode"] == "build"
        assert status["name"] == "Build"

    @pytest.mark.asyncio
    async def test_resolve_shell_variant_build_mode(self, coordinator):
        """Test shell variant resolution in BUILD mode."""
        # BUILD mode should allow full shell
        variant = coordinator.resolve_shell_variant("bash")
        assert variant == "shell"

    @pytest.mark.asyncio
    async def test_resolve_shell_variant_plan_mode(self, coordinator_plan_mode):
        """Test shell variant resolution in PLAN mode."""
        # PLAN mode should resolve to readonly shell
        # (if shell is disallowed but shell_readonly is allowed)
        variant = coordinator_plan_mode.resolve_shell_variant("bash")
        # Should return "shell" if no tool registry available
        # or resolve based on mode config
        assert variant in ["shell", "shell_readonly", "bash"]

    @pytest.mark.asyncio
    async def test_get_system_prompt_addition(self, coordinator):
        """Test getting system prompt addition for current mode."""
        addition = coordinator.get_system_prompt_addition()
        assert isinstance(addition, str)
        assert len(addition) > 0
        # Should contain BUILD mode specific instructions
        assert "BUILD mode" in addition

    @pytest.mark.asyncio
    async def test_get_available_modes(self, coordinator):
        """Test getting list of available modes."""
        modes = coordinator.get_available_modes()
        assert len(modes) == 3
        mode_names = [m["mode"] for m in modes]
        assert "build" in mode_names
        assert "plan" in mode_names
        assert "explore" in mode_names


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mode_controller():
    """Provide a mode controller for testing."""
    return AgentModeController(initial_mode=AgentMode.BUILD)


@pytest.fixture
def mode_controller_plan():
    """Provide a mode controller in PLAN mode for testing."""
    return AgentModeController(initial_mode=AgentMode.PLAN)


@pytest.fixture
def mode_controller_explore():
    """Provide a mode controller in EXPLORE mode for testing."""
    return AgentModeController(initial_mode=AgentMode.EXPLORE)


@pytest.fixture
def coordinator(mode_controller):
    """Provide a ModeCoordinator instance for testing."""
    from victor.agent.coordinators.mode_coordinator import ModeCoordinator

    return ModeCoordinator(mode_controller=mode_controller)


@pytest.fixture
def coordinator_plan_mode(mode_controller_plan):
    """Provide a ModeCoordinator in PLAN mode for testing."""
    from victor.agent.coordinators.mode_coordinator import ModeCoordinator

    return ModeCoordinator(mode_controller=mode_controller_plan)


@pytest.fixture
def coordinator_explore_mode(mode_controller_explore):
    """Provide a ModeCoordinator in EXPLORE mode for testing."""
    from victor.agent.coordinators.mode_coordinator import ModeCoordinator

    return ModeCoordinator(mode_controller=mode_controller_explore)
