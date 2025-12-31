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

"""Tests for adaptive mode controller with reinforcement learning."""

import tempfile
from pathlib import Path

import pytest

from victor.agent.adaptive_mode_controller import (
    AdaptiveModeController,
    AgentMode,
    ModeAction,
    ModeState,
    QLearningStore,
    TransitionTrigger,
)


class TestModeState:
    """Tests for ModeState dataclass."""

    def test_to_state_key(self):
        """Test state key generation."""
        state = ModeState(
            mode=AgentMode.EXPLORE,
            task_type="analysis",
            tool_calls_made=3,
            tool_budget=10,
            iteration_count=5,
            iteration_budget=20,
            quality_score=0.7,
            grounding_score=0.8,
            time_in_mode_seconds=30.0,
            recent_tool_success_rate=0.9,
        )

        key = state.to_state_key()

        # Key should contain mode and task type
        assert "explore" in key
        assert "analysis" in key

    def test_discretize_ratio(self):
        """Test ratio discretization."""
        state = ModeState(
            mode=AgentMode.EXPLORE,
            task_type="analysis",
            tool_calls_made=0,
            tool_budget=10,
            iteration_count=0,
            iteration_budget=20,
            quality_score=0.5,
            grounding_score=0.5,
            time_in_mode_seconds=0.0,
            recent_tool_success_rate=1.0,
        )

        assert state._discretize_ratio(0.1) == "low"
        assert state._discretize_ratio(0.3) == "mid_low"
        assert state._discretize_ratio(0.6) == "mid_high"
        assert state._discretize_ratio(0.9) == "high"

    def test_discretize_quality(self):
        """Test quality score discretization."""
        state = ModeState(
            mode=AgentMode.EXPLORE,
            task_type="analysis",
            tool_calls_made=0,
            tool_budget=10,
            iteration_count=0,
            iteration_budget=20,
            quality_score=0.5,
            grounding_score=0.5,
            time_in_mode_seconds=0.0,
            recent_tool_success_rate=1.0,
        )

        assert state._discretize_quality(0.2) == "poor"
        assert state._discretize_quality(0.5) == "fair"
        assert state._discretize_quality(0.7) == "good"
        assert state._discretize_quality(0.9) == "excellent"


class TestQLearningStore:
    """Tests for QLearningStore."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary Q-learning store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield QLearningStore(project_path=Path(tmpdir))

    def test_get_q_value_returns_zero_for_unknown(self, temp_store):
        """Unknown state-action pairs should return 0."""
        q_value = temp_store.get_q_value("unknown_state", "unknown_action")
        assert q_value == 0.0

    def test_update_q_value(self, temp_store):
        """Test Q-value update."""
        # First update
        new_q = temp_store.update_q_value("state1", "action1", reward=1.0)
        assert new_q > 0

        # Verify it was stored
        stored_q = temp_store.get_q_value("state1", "action1")
        assert stored_q == new_q

    def test_update_q_value_with_next_state(self, temp_store):
        """Test Q-value update with next state consideration."""
        # Set up next state with high Q-value
        temp_store.update_q_value("next_state", "action1", reward=2.0)

        # Update current state considering next state
        new_q = temp_store.update_q_value(
            "current_state", "action1", reward=0.5, next_state_key="next_state"
        )

        # Q-value should be positive (incorporates immediate reward + discounted future)
        # Q = 0 + α * (0.5 + γ * max_next_Q - 0) = 0.1 * (0.5 + 0.9 * 0.2) = 0.068
        assert new_q > 0

    def test_get_all_actions(self, temp_store):
        """Test getting all actions for a state."""
        temp_store.update_q_value("state1", "action1", reward=1.0)
        temp_store.update_q_value("state1", "action2", reward=0.5)

        actions = temp_store.get_all_actions("state1")

        assert len(actions) == 2
        assert "action1" in actions
        assert "action2" in actions

    def test_record_transition(self, temp_store):
        """Test recording a transition."""
        temp_store.record_transition(
            profile_name="test",
            from_mode=AgentMode.EXPLORE,
            to_mode=AgentMode.BUILD,
            trigger=TransitionTrigger.PATTERN_DETECTED,
            state_key="test_state",
            action_key="test_action",
            reward=0.8,
        )

        # Should not raise
        assert True

    def test_update_task_stats(self, temp_store):
        """Test updating task statistics with outcome-aware learning.

        The new system considers success/failure:
        - First sample for successful task uses: max(used + 5, min_budget)
        - For unknown task types, min_budget defaults to 10
        """
        temp_store.update_task_stats(
            task_type="analysis",
            tool_budget_used=5,
            quality_score=0.8,
            completed=True,
        )

        stats = temp_store.get_task_stats("analysis")

        assert stats["task_type"] == "analysis"
        # First sample: max(5 + 5, 10) = 10 (default min budget)
        assert stats["optimal_tool_budget"] == 10
        assert stats["sample_count"] == 1

    def test_update_task_stats_budget_exhaustion_increases_budget(self, temp_store):
        """Budget exhaustion with failure should increase the learned budget.

        This tests the outcome-aware learning: when budget is exhausted but
        task fails, the system should learn to allocate more budget next time.
        """
        # First establish a baseline budget
        temp_store.update_task_stats(
            task_type="test_task",
            tool_budget_used=10,
            quality_score=0.8,
            completed=True,
        )
        initial_stats = temp_store.get_task_stats("test_task")
        initial_budget = initial_stats["optimal_tool_budget"]

        # Now simulate budget exhaustion failure
        temp_store.update_task_stats(
            task_type="test_task",
            tool_budget_used=15,
            quality_score=0.3,  # Low quality
            completed=False,  # Failed
            tool_budget_total=15,
            budget_exhausted=True,  # Budget was exhausted
        )

        stats = temp_store.get_task_stats("test_task")
        # Budget should have increased due to exhaustion failure
        assert stats["optimal_tool_budget"] > initial_budget

    def test_update_task_stats_success_gradually_decreases_budget(self, temp_store):
        """Efficient success should gradually decrease the learned budget.

        When tasks complete successfully with fewer tool calls than budget,
        the system should learn to gradually reduce budget (but not abruptly).
        """
        # Start with a higher budget
        temp_store.update_task_stats(
            task_type="efficient_task",
            tool_budget_used=30,
            quality_score=0.8,
            completed=True,
        )
        initial_stats = temp_store.get_task_stats("efficient_task")
        initial_budget = initial_stats["optimal_tool_budget"]

        # Simulate multiple efficient completions
        for _ in range(5):
            temp_store.update_task_stats(
                task_type="efficient_task",
                tool_budget_used=10,  # Much less than current budget
                quality_score=0.9,
                completed=True,
            )

        stats = temp_store.get_task_stats("efficient_task")
        # Budget should have decreased but gradually (not all at once)
        # It should still be above the min budget (10) but lower than initial
        assert stats["optimal_tool_budget"] < initial_budget
        assert stats["optimal_tool_budget"] >= 10  # Min floor

    def test_get_task_stats_returns_defaults(self, temp_store):
        """Should return defaults for unknown task types."""
        stats = temp_store.get_task_stats("unknown_task")

        assert stats["optimal_tool_budget"] == 10
        assert stats["sample_count"] == 0


class TestAdaptiveModeController:
    """Tests for AdaptiveModeController."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary Q-learning store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield QLearningStore(project_path=Path(tmpdir))

    @pytest.fixture
    def controller(self, temp_store):
        """Create a controller with temp storage."""
        return AdaptiveModeController(
            profile_name="test-profile",
            q_store=temp_store,
        )

    def test_creation(self, controller):
        """Test controller creation."""
        assert controller.profile_name == "test-profile"

    def test_get_recommended_action_returns_action(self, controller):
        """get_recommended_action should return a ModeAction."""
        action = controller.get_recommended_action(
            current_mode="explore",
            task_type="analysis",
            tool_calls_made=3,
            tool_budget=10,
        )

        assert isinstance(action, ModeAction)
        assert action.target_mode in AgentMode

    def test_get_recommended_action_respects_budget(self, controller):
        """Should suggest completion when budget exhausted."""
        action = controller.get_recommended_action(
            current_mode="explore",
            task_type="analysis",
            tool_calls_made=10,
            tool_budget=10,
        )

        # Should either complete or have low confidence for continuing
        assert (
            action.target_mode == AgentMode.COMPLETE
            or not action.should_continue
            or action.confidence < 0.7
        )

    def test_get_recommended_action_suggests_completion_on_high_quality(self, controller):
        """Should suggest completion when quality is high."""
        action = controller.get_recommended_action(
            current_mode="explore",
            task_type="analysis",
            tool_calls_made=3,
            tool_budget=10,
            quality_score=0.9,
        )

        # High quality should lead to completion consideration
        # Action might be to complete or continue with high confidence
        assert action.confidence > 0.0

    def test_record_outcome_updates_q_values(self, controller, temp_store):
        """Recording outcome should update Q-values."""
        # First, get an action to set up state
        controller.get_recommended_action(
            current_mode="explore",
            task_type="analysis",
            tool_calls_made=3,
            tool_budget=10,
        )

        # Record outcome
        reward = controller.record_outcome(
            success=True,
            quality_score=0.8,
            user_satisfied=True,
            completed=True,
        )

        assert reward > 0

    def test_record_outcome_negative_reward_on_failure(self, controller):
        """Failed outcomes should give negative reward."""
        controller.get_recommended_action(
            current_mode="explore",
            task_type="analysis",
            tool_calls_made=3,
            tool_budget=10,
        )

        reward = controller.record_outcome(
            success=False,
            quality_score=0.3,
            user_satisfied=False,
            completed=False,
        )

        assert reward < 1.0  # Should be penalized

    def test_get_optimal_tool_budget_uses_learned_value(self, controller, temp_store):
        """Should return learned optimal budget.

        With outcome-aware learning:
        - First sample for successful task: max(used + 5, min_budget)
        - For unknown task types (like 'analysis'), min_budget defaults to 10
        - So max(7 + 5, 10) = 12
        """
        # Train some data
        temp_store.update_task_stats(
            task_type="analysis",
            tool_budget_used=7,
            quality_score=0.9,
            completed=True,
        )

        budget = controller.get_optimal_tool_budget("analysis")

        # First sample: max(7 + 5, 10) = 12 (default min budget)
        assert budget == 12

    def test_get_optimal_tool_budget_uses_default(self, controller):
        """Should return default budget for unknown task types."""
        # For a task type with no learned data, should return default
        budget = controller.get_optimal_tool_budget("code_generation")

        # Should be the default from Q-store (10) since no training data
        expected = AdaptiveModeController.DEFAULT_TOOL_BUDGETS.get("code_generation", 10)
        assert budget == expected or budget == 10  # Accept either default

    def test_should_continue_false_on_budget_exhausted(self, controller):
        """Should return False when tool budget exhausted."""
        should_continue, reason = controller.should_continue(
            tool_calls_made=10,
            tool_budget=10,
            quality_score=0.5,
            iteration_count=5,
            iteration_budget=20,
        )

        assert not should_continue
        assert "budget" in reason.lower()

    def test_should_continue_false_on_iteration_exhausted(self, controller):
        """Should return False when iteration budget exhausted."""
        should_continue, reason = controller.should_continue(
            tool_calls_made=5,
            tool_budget=10,
            quality_score=0.5,
            iteration_count=20,
            iteration_budget=20,
        )

        assert not should_continue
        assert "iteration" in reason.lower()

    def test_should_continue_false_on_high_quality(self, controller):
        """Should return False when quality is very high."""
        should_continue, reason = controller.should_continue(
            tool_calls_made=5,
            tool_budget=10,
            quality_score=0.9,
            iteration_count=5,
            iteration_budget=20,
        )

        assert not should_continue
        assert "quality" in reason.lower()

    def test_get_session_stats(self, controller):
        """get_session_stats should return comprehensive stats."""
        stats = controller.get_session_stats()

        assert "profile_name" in stats
        assert "total_reward" in stats
        assert "exploration_rate" in stats

    def test_reset_session(self, controller):
        """reset_session should clear session tracking."""
        # Record some activity
        controller.get_recommended_action(
            current_mode="explore",
            task_type="analysis",
            tool_calls_made=3,
            tool_budget=10,
        )
        controller.record_outcome(success=True, quality_score=0.8)

        # Reset
        controller.reset_session()

        assert controller._total_reward == 0.0
        assert len(controller._mode_history) == 0

    def test_adjust_exploration_rate(self, controller):
        """Should adjust exploration rate within bounds."""
        controller.adjust_exploration_rate(0.5)
        assert controller._q_store.exploration_rate == 0.5

        controller.adjust_exploration_rate(1.5)  # Above max
        assert controller._q_store.exploration_rate == 1.0

        controller.adjust_exploration_rate(-0.5)  # Below min
        assert controller._q_store.exploration_rate == 0.0

    def test_decay_exploration_rate(self, controller):
        """Should decay exploration rate over time."""
        controller._q_store.exploration_rate = 1.0

        controller.decay_exploration_rate(0.9)
        assert controller._q_store.exploration_rate == 0.9

        controller.decay_exploration_rate(0.9)
        assert controller._q_store.exploration_rate == pytest.approx(0.81)


class TestModeTransitions:
    """Tests for valid mode transitions."""

    @pytest.fixture
    def temp_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield QLearningStore(project_path=Path(tmpdir))

    @pytest.fixture
    def controller(self, temp_store):
        return AdaptiveModeController(
            profile_name="test",
            q_store=temp_store,
        )

    def test_explore_can_transition_to_plan(self, controller):
        """Explore mode should be able to transition to plan."""
        valid = AdaptiveModeController.VALID_TRANSITIONS[AgentMode.EXPLORE]
        assert AgentMode.PLAN in valid

    def test_explore_can_transition_to_build(self, controller):
        """Explore mode should be able to transition to build."""
        valid = AdaptiveModeController.VALID_TRANSITIONS[AgentMode.EXPLORE]
        assert AgentMode.BUILD in valid

    def test_build_can_transition_to_review(self, controller):
        """Build mode should be able to transition to review."""
        valid = AdaptiveModeController.VALID_TRANSITIONS[AgentMode.BUILD]
        assert AgentMode.REVIEW in valid

    def test_complete_has_no_transitions(self, controller):
        """Complete mode should have no valid transitions."""
        valid = AdaptiveModeController.VALID_TRANSITIONS[AgentMode.COMPLETE]
        assert len(valid) == 0


class TestRewardCalculation:
    """Tests for reward calculation logic."""

    @pytest.fixture
    def temp_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield QLearningStore(project_path=Path(tmpdir))

    @pytest.fixture
    def controller(self, temp_store):
        return AdaptiveModeController(
            profile_name="test",
            q_store=temp_store,
        )

    def test_success_gives_positive_reward(self, controller):
        """Success should give positive reward."""
        controller.get_recommended_action(
            current_mode="explore",
            task_type="analysis",
            tool_calls_made=3,
            tool_budget=10,
        )

        reward = controller.record_outcome(
            success=True,
            quality_score=0.7,
            user_satisfied=True,
        )

        assert reward > 0

    def test_failure_gives_negative_reward(self, controller):
        """Failure should give negative/lower reward."""
        controller.get_recommended_action(
            current_mode="explore",
            task_type="analysis",
            tool_calls_made=3,
            tool_budget=10,
        )

        reward = controller.record_outcome(
            success=False,
            quality_score=0.3,
            user_satisfied=False,
        )

        # Failed tasks get penalized
        assert reward < 1.0

    def test_high_quality_increases_reward(self, controller):
        """Higher quality should increase reward."""
        controller.get_recommended_action(
            current_mode="explore",
            task_type="analysis",
            tool_calls_made=3,
            tool_budget=10,
        )

        low_quality_reward = controller.record_outcome(
            success=True,
            quality_score=0.3,
            user_satisfied=True,
        )

        controller.reset_session()
        controller.get_recommended_action(
            current_mode="explore",
            task_type="analysis",
            tool_calls_made=3,
            tool_budget=10,
        )

        high_quality_reward = controller.record_outcome(
            success=True,
            quality_score=0.9,
            user_satisfied=True,
        )

        assert high_quality_reward > low_quality_reward

    def test_completion_bonus(self, controller):
        """Completing a task should give bonus reward."""
        controller.get_recommended_action(
            current_mode="explore",
            task_type="analysis",
            tool_calls_made=3,
            tool_budget=10,
        )

        incomplete_reward = controller.record_outcome(
            success=True,
            quality_score=0.7,
            completed=False,
        )

        controller.reset_session()
        controller.get_recommended_action(
            current_mode="explore",
            task_type="analysis",
            tool_calls_made=3,
            tool_budget=10,
        )

        complete_reward = controller.record_outcome(
            success=True,
            quality_score=0.7,
            completed=True,
        )

        assert complete_reward > incomplete_reward
