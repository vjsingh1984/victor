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

"""Property-based tests for AgentModeController.

Phase 3.3: Property-based tests for mode controller logic.

Uses Hypothesis to test invariants across many iterations:
1. Mode switch consistency
2. Tool filtering accuracy
3. Exploration multiplier bounds
4. Mode state persistence
"""

import pytest
from hypothesis import given, strategies as st, settings, Phase

from victor.agent.mode_controller import AgentModeController, AgentMode


# Strategies
mode_strategy = st.sampled_from(list(AgentMode))
tool_name_strategy = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz_"),
    min_size=1,
    max_size=20,
)


@pytest.fixture
def mode_controller():
    """Create a fresh mode controller for each test."""
    return AgentModeController()


class TestModeSwitchProperties:
    """Property-based tests for mode switching invariants."""

    @given(mode=mode_strategy)
    @settings(max_examples=50, phases=[Phase.generate])
    def test_switch_mode_updates_current_mode(self, mode: AgentMode):
        """Switching mode should update current_mode property."""
        controller = AgentModeController()

        result = controller.switch_mode(mode)

        # Mode switch should succeed
        assert result is True
        assert controller.current_mode.name == mode.name

    @given(modes=st.lists(mode_strategy, min_size=1, max_size=10))
    @settings(max_examples=30, phases=[Phase.generate])
    def test_sequential_mode_switches(self, modes: list[AgentMode]):
        """Sequential mode switches should result in last mode being current."""
        controller = AgentModeController()

        for mode in modes:
            controller.switch_mode(mode)

        # Final mode should be the last in the sequence
        assert controller.current_mode.name == modes[-1].name

    @given(mode=mode_strategy)
    @settings(max_examples=50, phases=[Phase.generate])
    def test_same_mode_switch_is_idempotent(self, mode: AgentMode):
        """Switching to the same mode twice should be idempotent."""
        controller = AgentModeController()

        controller.switch_mode(mode)
        mode_after_first = controller.current_mode

        controller.switch_mode(mode)
        mode_after_second = controller.current_mode

        assert mode_after_first.name == mode_after_second.name


class TestToolAllowedProperties:
    """Property-based tests for tool filtering."""

    @given(mode=mode_strategy, tool_name=tool_name_strategy)
    @settings(max_examples=100, phases=[Phase.generate])
    def test_tool_allowed_returns_boolean(self, mode: AgentMode, tool_name: str):
        """is_tool_allowed should always return a boolean."""
        controller = AgentModeController()
        controller.switch_mode(mode)

        result = controller.is_tool_allowed(tool_name)

        assert isinstance(result, bool)

    @given(tool_names=st.lists(tool_name_strategy, min_size=1, max_size=10, unique=True))
    @settings(max_examples=30, phases=[Phase.generate])
    def test_tool_allowed_consistency_across_queries(self, tool_names: list[str]):
        """Same tool should have same allowed status across repeated queries."""
        controller = AgentModeController()
        controller.switch_mode(AgentMode.BUILD)

        results = {}
        for tool_name in tool_names:
            results[tool_name] = controller.is_tool_allowed(tool_name)

        # Query again and verify consistency
        for tool_name in tool_names:
            assert controller.is_tool_allowed(tool_name) == results[tool_name]


class TestExplorationMultiplierProperties:
    """Property-based tests for exploration multiplier."""

    @given(mode=mode_strategy)
    @settings(max_examples=50, phases=[Phase.generate])
    def test_exploration_multiplier_is_positive(self, mode: AgentMode):
        """Exploration multiplier should always be positive."""
        controller = AgentModeController()
        controller.switch_mode(mode)

        multiplier = controller.get_exploration_multiplier()

        assert multiplier > 0

    @given(mode=mode_strategy)
    @settings(max_examples=50, phases=[Phase.generate])
    def test_exploration_multiplier_within_bounds(self, mode: AgentMode):
        """Exploration multiplier should be within reasonable bounds."""
        controller = AgentModeController()
        controller.switch_mode(mode)

        multiplier = controller.get_exploration_multiplier()

        # Reasonable bounds: 1.0 (min) to 25.0 (EXPLORE mode uses 20.0)
        assert 1.0 <= multiplier <= 25.0


class TestToolPriorityProperties:
    """Property-based tests for tool priority calculation."""

    @given(mode=mode_strategy, tool_name=tool_name_strategy)
    @settings(max_examples=100, phases=[Phase.generate])
    def test_tool_priority_is_non_negative(self, mode: AgentMode, tool_name: str):
        """Tool priority should always be non-negative."""
        controller = AgentModeController()
        controller.switch_mode(mode)

        priority = controller.get_tool_priority(tool_name)

        assert priority >= 0


class TestModeInvariantProperties:
    """Property-based tests for mode invariants."""

    @given(mode=mode_strategy)
    @settings(max_examples=50, phases=[Phase.generate])
    def test_mode_has_valid_name(self, mode: AgentMode):
        """Mode should have a valid string name."""
        controller = AgentModeController()
        controller.switch_mode(mode)

        assert isinstance(controller.current_mode.name, str)
        assert len(controller.current_mode.name) > 0

    @given(mode=mode_strategy)
    @settings(max_examples=50, phases=[Phase.generate])
    def test_system_prompt_addition_is_string(self, mode: AgentMode):
        """System prompt addition should always be a string."""
        controller = AgentModeController()
        controller.switch_mode(mode)

        addition = controller.get_system_prompt_addition()

        assert isinstance(addition, str)


class TestBuildModeBehavior:
    """Property-based tests specific to BUILD mode defaults."""

    @given(tool_name=tool_name_strategy)
    @settings(max_examples=50, phases=[Phase.generate])
    def test_build_mode_allows_most_tools(self, tool_name: str):
        """BUILD mode should allow most tools by default."""
        controller = AgentModeController()
        controller.switch_mode(AgentMode.BUILD)

        # In BUILD mode with no blacklist, tools should be allowed
        # unless they're in a specific deny list
        result = controller.is_tool_allowed(tool_name)
        # Just verify it returns a boolean (specific policy tested elsewhere)
        assert isinstance(result, bool)

    def test_build_mode_has_expected_multiplier(self):
        """BUILD mode should have exploration_multiplier = 5.0 (read before write)."""
        controller = AgentModeController()
        controller.switch_mode(AgentMode.BUILD)

        multiplier = controller.get_exploration_multiplier()

        # BUILD mode uses 5.0 for reading before writing
        assert multiplier == 5.0


class TestExploreModeProperties:
    """Property-based tests specific to EXPLORE mode."""

    def test_explore_mode_has_higher_multiplier(self):
        """EXPLORE mode should have higher exploration_multiplier than BUILD."""
        controller = AgentModeController()

        controller.switch_mode(AgentMode.BUILD)
        build_multiplier = controller.get_exploration_multiplier()

        controller.switch_mode(AgentMode.EXPLORE)
        explore_multiplier = controller.get_exploration_multiplier()

        assert explore_multiplier >= build_multiplier
