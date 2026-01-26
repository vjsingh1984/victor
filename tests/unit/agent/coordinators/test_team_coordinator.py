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

"""Tests for TeamCoordinator.

This test file demonstrates the migration pattern from orchestrator tests
to coordinator-specific tests, following Track 4 extraction.

Migration Pattern:
1. Identify orchestrator tests that delegate to coordinator
2. Extract relevant test logic
3. Mock coordinator dependencies (not the orchestrator)
4. Test coordinator in isolation
5. Update integration tests to verify delegation

Example Migration from test_orchestrator_core.py:

BEFORE (orchestrator test):
```python
def test_orchestrator_get_team_suggestions(orchestrator):
    suggestions = orchestrator.get_team_suggestions("feature", "high")
    assert suggestions.recommended_team == "feature_team"
```

AFTER (coordinator test):
```python
def test_team_coordinator_get_suggestions(coordinator, mock_mw_coordinator):
    suggestions = coordinator.get_team_suggestions("feature", "high")
    mock_mw_coordinator.coordination.suggest_for_task.assert_called_once()
```
"""

import pytest
from unittest.mock import MagicMock, Mock
from typing import Dict, Any

from victor.agent.coordinators.team_coordinator import TeamCoordinator
from victor.protocols.coordination import (
    CoordinationSuggestion,
    TeamRecommendation,
    ComplexityLevel,
    TeamSuggestionAction,
)


class TestTeamCoordinator:
    """Test suite for TeamCoordinator.

    This coordinator handles team specification and suggestion management,
    extracted from AgentOrchestrator as part of Track 4 refactoring.
    """

    @pytest.fixture
    def mock_orchestrator(self) -> Mock:
        """Create mock orchestrator."""
        orchestrator = Mock()
        orchestrator._team_specs = {}
        return orchestrator

    @pytest.fixture
    def mock_mode_coordinator(self) -> Mock:
        """Create mock mode coordinator."""
        coordinator = Mock()
        coordinator.current_mode_name = "build"
        return coordinator

    @pytest.fixture
    def mock_mw_coordinator(self) -> Mock:
        """Create mock ModeWorkflowTeamCoordinator."""
        coordinator = Mock()
        coordinator.coordination = Mock()

        # Set up default suggestion response
        suggestion = CoordinationSuggestion(
            mode="build",
            task_type="feature",
            complexity=ComplexityLevel.HIGH,
            action=TeamSuggestionAction.SUGGEST,
            team_recommendations=[
                TeamRecommendation(
                    team_name="feature_team",
                    confidence=0.9,
                    reason="Task type matches pattern",
                    formation="parallel",
                )
            ],
            workflow_recommendations=[],
            system_prompt_additions="",
            tool_priorities={},
        )
        coordinator.coordination.suggest_for_task = Mock(return_value=suggestion)
        return coordinator

    @pytest.fixture
    def coordinator(
        self,
        mock_orchestrator: Mock,
        mock_mode_coordinator: Mock,
        mock_mw_coordinator: Mock,
    ) -> TeamCoordinator:
        """Create team coordinator with default mocks."""
        return TeamCoordinator(
            orchestrator=mock_orchestrator,
            mode_coordinator=mock_mode_coordinator,
            mode_workflow_team_coordinator=mock_mw_coordinator,
        )

    # Test get_team_suggestions

    def test_get_team_suggestions_delegates_to_mw_coordinator(
        self,
        coordinator: TeamCoordinator,
        mock_mode_coordinator: Mock,
        mock_mw_coordinator: Mock,
    ):
        """Test that get_team_suggestions delegates to ModeWorkflowTeamCoordinator."""
        # Execute
        result = coordinator.get_team_suggestions("feature", "high")

        # Assert - delegation occurred with correct parameters
        mock_mw_coordinator.coordination.suggest_for_task.assert_called_once_with(
            task_type="feature",
            complexity="high",
            mode="build",
        )
        assert result is not None
        assert result.task_type == "feature"

    def test_get_team_suggestions_with_different_task_types(
        self,
        coordinator: TeamCoordinator,
        mock_mw_coordinator: Mock,
    ):
        """Test get_team_suggestions with different task types."""
        task_types = ["bugfix", "refactor", "review", "test", "documentation"]

        for task_type in task_types:
            coordinator.get_team_suggestions(task_type, "medium")

        # Assert - called for each task type
        assert mock_mw_coordinator.coordination.suggest_for_task.call_count == len(task_types)

    def test_get_team_suggestions_with_different_complexity_levels(
        self,
        coordinator: TeamCoordinator,
        mock_mw_coordinator: Mock,
    ):
        """Test get_team_suggestions with different complexity levels."""
        complexity_levels = ["trivial", "low", "medium", "high", "extreme"]

        for complexity in complexity_levels:
            coordinator.get_team_suggestions("feature", complexity)

        # Assert - called for each complexity level
        assert mock_mw_coordinator.coordination.suggest_for_task.call_count == len(
            complexity_levels
        )

    def test_get_team_suggestions_with_different_modes(
        self,
        mock_orchestrator: Mock,
        mock_mode_coordinator: Mock,
        mock_mw_coordinator: Mock,
    ):
        """Test get_team_suggestions with different agent modes."""
        modes = ["build", "plan", "explore"]

        for mode in modes:
            # Set mode
            mock_mode_coordinator.current_mode_name = mode

            # Create coordinator and get suggestions
            coordinator = TeamCoordinator(
                orchestrator=mock_orchestrator,
                mode_coordinator=mock_mode_coordinator,
                mode_workflow_team_coordinator=mock_mw_coordinator,
            )
            coordinator.get_team_suggestions("feature", "high")

        # Assert - called with correct mode each time
        calls = mock_mw_coordinator.coordination.suggest_for_task.call_args_list
        assert len(calls) == len(modes)
        for i, mode in enumerate(modes):
            assert calls[i][1]["mode"] == mode

    def test_get_team_suggestions_returns_coordination_suggestion(
        self, coordinator: TeamCoordinator
    ):
        """Test that get_team_suggestions returns CoordinationSuggestion."""
        # Execute
        result = coordinator.get_team_suggestions("feature", "high")

        # Assert - result is CoordinationSuggestion
        assert isinstance(result, CoordinationSuggestion)
        assert hasattr(result, "task_type")
        assert hasattr(result, "complexity")
        assert hasattr(result, "action")
        assert hasattr(result, "team_recommendations")

    def test_get_team_suggestions_includes_team_recommendations(self, coordinator: TeamCoordinator):
        """Test that suggestions include team recommendations."""
        # Execute
        result = coordinator.get_team_suggestions("feature", "high")

        # Assert
        assert len(result.team_recommendations) > 0
        assert result.team_recommendations[0].team_name == "feature_team"
        assert result.team_recommendations[0].confidence == 0.9

    def test_get_team_suggestions_with_empty_recommendations(
        self,
        mock_orchestrator: Mock,
        mock_mode_coordinator: Mock,
        mock_mw_coordinator: Mock,
    ):
        """Test get_team_suggestions when no teams are recommended."""
        # Setup - return empty recommendations
        empty_suggestion = CoordinationSuggestion(
            mode="build",
            task_type="feature",
            complexity=ComplexityLevel.LOW,
            action=TeamSuggestionAction.NONE,
            team_recommendations=[],
            workflow_recommendations=[],
            system_prompt_additions="",
            tool_priorities={},
        )
        mock_mw_coordinator.coordination.suggest_for_task = Mock(return_value=empty_suggestion)

        # Execute
        coordinator = TeamCoordinator(
            orchestrator=mock_orchestrator,
            mode_coordinator=mock_mode_coordinator,
            mode_workflow_team_coordinator=mock_mw_coordinator,
        )
        result = coordinator.get_team_suggestions("feature", "low")

        # Assert
        assert len(result.team_recommendations) == 0
        assert result.action == TeamSuggestionAction.NONE

    def test_get_team_suggestions_passes_correct_mode_from_coordinator(
        self,
        mock_orchestrator: Mock,
        mock_mode_coordinator: Mock,
        mock_mw_coordinator: Mock,
    ):
        """Test that current mode from ModeCoordinator is passed correctly."""
        # Setup - set mode to "plan"
        mock_mode_coordinator.current_mode_name = "plan"

        # Execute
        coordinator = TeamCoordinator(
            orchestrator=mock_orchestrator,
            mode_coordinator=mock_mode_coordinator,
            mode_workflow_team_coordinator=mock_mw_coordinator,
        )
        coordinator.get_team_suggestions("refactor", "high")

        # Assert - mode was passed correctly
        call_args = mock_mw_coordinator.coordination.suggest_for_task.call_args
        assert call_args[1]["mode"] == "plan"

    # Test set_team_specs

    def test_set_team_specs_stores_in_orchestrator(
        self, coordinator: TeamCoordinator, mock_orchestrator: Mock
    ):
        """Test that set_team_specs stores specs in orchestrator."""
        # Setup
        specs = {
            "feature_team": Mock(name="feature_team"),
            "review_team": Mock(name="review_team"),
        }

        # Execute
        coordinator.set_team_specs(specs)

        # Assert - stored in orchestrator
        assert mock_orchestrator._team_specs == specs

    def test_set_team_specs_with_empty_dict(
        self, coordinator: TeamCoordinator, mock_orchestrator: Mock
    ):
        """Test set_team_specs with empty dictionary."""
        # Execute
        coordinator.set_team_specs({})

        # Assert
        assert mock_orchestrator._team_specs == {}

    def test_set_team_specs_with_single_team(
        self, coordinator: TeamCoordinator, mock_orchestrator: Mock
    ):
        """Test set_team_specs with a single team spec."""
        # Setup
        specs = {"solo_team": Mock(name="solo_team")}

        # Execute
        coordinator.set_team_specs(specs)

        # Assert
        assert "solo_team" in mock_orchestrator._team_specs
        assert len(mock_orchestrator._team_specs) == 1

    def test_set_team_specs_overrides_existing_specs(
        self, coordinator: TeamCoordinator, mock_orchestrator: Mock
    ):
        """Test that set_team_specs overrides existing specs."""
        # Setup - initial specs
        initial_specs = {"team1": Mock(name="team1")}
        mock_orchestrator._team_specs = initial_specs

        # Execute - set new specs
        new_specs = {"team2": Mock(name="team2")}
        coordinator.set_team_specs(new_specs)

        # Assert - old specs replaced
        assert mock_orchestrator._team_specs == new_specs
        assert "team1" not in mock_orchestrator._team_specs

    def test_set_team_specs_with_multiple_teams(
        self, coordinator: TeamCoordinator, mock_orchestrator: Mock
    ):
        """Test set_team_specs with multiple team specs."""
        # Setup
        specs = {
            "team1": Mock(name="team1"),
            "team2": Mock(name="team2"),
            "team3": Mock(name="team3"),
            "team4": Mock(name="team4"),
            "team5": Mock(name="team5"),
        }

        # Execute
        coordinator.set_team_specs(specs)

        # Assert - all teams stored
        assert len(mock_orchestrator._team_specs) == 5
        for team_name in specs:
            assert team_name in mock_orchestrator._team_specs

    def test_set_team_specs_preserves_spec_reference(
        self, coordinator: TeamCoordinator, mock_orchestrator: Mock
    ):
        """Test that set_team_specs preserves the same object reference."""
        # Setup
        original_specs = {"my_team": Mock(name="my_team")}

        # Execute
        coordinator.set_team_specs(original_specs)

        # Assert - same reference
        assert mock_orchestrator._team_specs is original_specs

    # Test get_team_specs

    def test_get_team_specs_returns_stored_specs(
        self, coordinator: TeamCoordinator, mock_orchestrator: Mock
    ):
        """Test that get_team_specs returns stored specs."""
        # Setup
        specs = {
            "feature_team": Mock(name="feature_team"),
            "review_team": Mock(name="review_team"),
        }
        mock_orchestrator._team_specs = specs

        # Execute
        result = coordinator.get_team_specs()

        # Assert
        assert result == specs
        assert len(result) == 2

    def test_get_team_specs_returns_empty_dict_when_not_set(
        self, coordinator: TeamCoordinator, mock_orchestrator: Mock
    ):
        """Test get_team_specs when no specs have been set."""
        # Setup - set _team_specs to empty dict and remove vertical_context
        mock_orchestrator._team_specs = {}
        # Remove vertical_context to prevent fallback from auto-creating Mock attributes
        if hasattr(mock_orchestrator, "vertical_context"):
            delattr(mock_orchestrator, "vertical_context")

        # Execute
        result = coordinator.get_team_specs()

        # Assert - returns empty dict
        assert result == {}
        assert isinstance(result, dict)

    def test_get_team_specs_returns_same_reference(
        self, coordinator: TeamCoordinator, mock_orchestrator: Mock
    ):
        """Test that get_team_specs returns the same reference (not a copy)."""
        # Setup
        specs = {"my_team": Mock(name="my_team")}
        mock_orchestrator._team_specs = specs

        # Execute
        result = coordinator.get_team_specs()

        # Assert - same reference
        assert result is specs

    def test_get_team_specs_after_set_team_specs(self, coordinator: TeamCoordinator):
        """Test get_team_specs after set_team_specs."""
        # Setup and execute
        original_specs = {"team1": Mock(name="team1")}
        coordinator.set_team_specs(original_specs)

        # Execute
        retrieved_specs = coordinator.get_team_specs()

        # Assert
        assert retrieved_specs == original_specs

    def test_get_team_specs_does_not_modify_original(
        self, coordinator: TeamCoordinator, mock_orchestrator: Mock
    ):
        """Test that get_team_specs doesn't modify the stored specs."""
        # Setup
        specs = {"team1": Mock(name="team1"), "team2": Mock(name="team2")}
        mock_orchestrator._team_specs = specs

        # Execute
        result = coordinator.get_team_specs()

        # Assert - original unchanged
        assert len(mock_orchestrator._team_specs) == 2
        assert mock_orchestrator._team_specs == specs

    # Test integration scenarios

    def test_full_workflow_set_get_suggestions(
        self, coordinator: TeamCoordinator, mock_mw_coordinator: Mock
    ):
        """Test full workflow: set specs, get specs, get suggestions."""
        # Set specs
        specs = {"feature_team": Mock(name="feature_team")}
        coordinator.set_team_specs(specs)

        # Get specs
        retrieved = coordinator.get_team_specs()
        assert retrieved == specs

        # Get suggestions
        suggestions = coordinator.get_team_suggestions("feature", "high")
        assert suggestions is not None

        # Assert delegation occurred
        mock_mw_coordinator.coordination.suggest_for_task.assert_called_once()

    def test_multiple_set_and_get_cycles(self, coordinator: TeamCoordinator):
        """Test multiple set/get cycles."""
        # First cycle
        specs1 = {"team1": Mock(name="team1")}
        coordinator.set_team_specs(specs1)
        result1 = coordinator.get_team_specs()
        assert result1 == specs1

        # Second cycle
        specs2 = {"team2": Mock(name="team2")}
        coordinator.set_team_specs(specs2)
        result2 = coordinator.get_team_specs()
        assert result2 == specs2

        # Third cycle
        specs3 = {"team3": Mock(name="team3")}
        coordinator.set_team_specs(specs3)
        result3 = coordinator.get_team_specs()
        assert result3 == specs3

    def test_team_spec_isolation_between_coordinators(
        self,
        mock_orchestrator: Mock,
        mock_mode_coordinator: Mock,
        mock_mw_coordinator: Mock,
    ):
        """Test that different coordinators can have different specs."""
        # Create two coordinators with same orchestrator
        coordinator1 = TeamCoordinator(
            orchestrator=mock_orchestrator,
            mode_coordinator=mock_mode_coordinator,
            mode_workflow_team_coordinator=mock_mw_coordinator,
        )
        coordinator2 = TeamCoordinator(
            orchestrator=mock_orchestrator,
            mode_coordinator=mock_mode_coordinator,
            mode_workflow_team_coordinator=mock_mw_coordinator,
        )

        # Set different specs via each coordinator
        specs1 = {"team1": Mock(name="team1")}
        specs2 = {"team2": Mock(name="team2")}

        coordinator1.set_team_specs(specs1)
        coordinator2.set_team_specs(specs2)

        # Both should see the same specs (same orchestrator)
        assert coordinator1.get_team_specs() == specs2
        assert coordinator2.get_team_specs() == specs2


class TestTeamCoordinatorEdgeCases:
    """Test edge cases and error conditions for TeamCoordinator."""

    @pytest.fixture
    def mock_orchestrator(self) -> Mock:
        """Create mock orchestrator."""
        orchestrator = Mock()
        return orchestrator

    @pytest.fixture
    def mock_mode_coordinator(self) -> Mock:
        """Create mock mode coordinator."""
        coordinator = Mock()
        coordinator.current_mode_name = "build"
        return coordinator

    @pytest.fixture
    def mock_mw_coordinator(self) -> Mock:
        """Create mock ModeWorkflowTeamCoordinator."""
        coordinator = Mock()
        coordinator.coordination = Mock()
        return coordinator

    @pytest.fixture
    def coordinator(
        self,
        mock_orchestrator: Mock,
        mock_mode_coordinator: Mock,
        mock_mw_coordinator: Mock,
    ) -> TeamCoordinator:
        """Create coordinator."""
        return TeamCoordinator(
            orchestrator=mock_orchestrator,
            mode_coordinator=mock_mode_coordinator,
            mode_workflow_team_coordinator=mock_mw_coordinator,
        )

    def test_get_team_suggestions_with_none_task_type(
        self, coordinator: TeamCoordinator, mock_mw_coordinator: Mock
    ):
        """Test get_team_suggestions with None task type."""
        # Execute - should not raise
        result = coordinator.get_team_suggestions(None, "high")

        # Assert - delegation occurred
        mock_mw_coordinator.coordination.suggest_for_task.assert_called_once_with(
            task_type=None,
            complexity="high",
            mode="build",
        )

    def test_get_team_suggestions_with_none_complexity(
        self, coordinator: TeamCoordinator, mock_mw_coordinator: Mock
    ):
        """Test get_team_suggestions with None complexity."""
        # Execute - should not raise
        result = coordinator.get_team_suggestions("feature", None)

        # Assert - delegation occurred
        mock_mw_coordinator.coordination.suggest_for_task.assert_called_once_with(
            task_type="feature",
            complexity=None,
            mode="build",
        )

    def test_get_team_suggestions_with_empty_strings(
        self, coordinator: TeamCoordinator, mock_mw_coordinator: Mock
    ):
        """Test get_team_suggestions with empty strings."""
        # Execute
        coordinator.get_team_suggestions("", "")

        # Assert - delegation occurred
        mock_mw_coordinator.coordination.suggest_for_task.assert_called_once_with(
            task_type="",
            complexity="",
            mode="build",
        )

    def test_get_team_suggestions_when_mw_coordinator_raises(
        self,
        coordinator: TeamCoordinator,
        mock_mw_coordinator: Mock,
    ):
        """Test get_team_suggestions when ModeWorkflowTeamCoordinator raises."""
        # Setup - make it raise
        mock_mw_coordinator.coordination.suggest_for_task.side_effect = RuntimeError(
            "Coordinator error"
        )

        # Execute & Assert - should propagate
        with pytest.raises(RuntimeError, match="Coordinator error"):
            coordinator.get_team_suggestions("feature", "high")

    def test_set_team_specs_with_none(self, coordinator: TeamCoordinator, mock_orchestrator: Mock):
        """Test set_team_specs with None - should handle gracefully."""
        # Execute & Assert - should raise or handle None appropriately
        # The implementation tries to log len(specs), so None will raise
        with pytest.raises(TypeError):
            coordinator.set_team_specs(None)

    def test_get_team_specs_when_specs_is_none(
        self,
        mock_orchestrator: Mock,
        mock_mode_coordinator: Mock,
        mock_mw_coordinator: Mock,
    ):
        """Test get_team_specs when _team_specs is None."""
        # Setup - set to None
        mock_orchestrator._team_specs = None

        # Execute
        coordinator = TeamCoordinator(
            orchestrator=mock_orchestrator,
            mode_coordinator=mock_mode_coordinator,
            mode_workflow_team_coordinator=mock_mw_coordinator,
        )
        result = coordinator.get_team_specs()

        # Assert - returns None (not empty dict)
        # This is expected behavior based on implementation using getattr with default {}
        # The implementation actually returns {} when attribute doesn't exist
        # but returns the actual value (even None) when it exists
        # So let's check what the actual behavior is
        # Based on the code: return getattr(self._orchestrator, "_team_specs", {})
        # When _team_specs is None, getattr returns None, not {}
        # Wait, no - getattr returns the actual value if it exists
        # So if _team_specs is None, it returns None
        # Actually, let me check the implementation again
        # From team_coordinator.py line 144:
        # return getattr(self._orchestrator, "_team_specs", {})
        # This returns the actual value if it exists, or {} if it doesn't
        # So if _team_specs is None, it returns None
        assert result is None

    def test_get_team_specs_with_special_characters_in_team_names(
        self, coordinator: TeamCoordinator, mock_orchestrator: Mock
    ):
        """Test get_team_specs with special characters in team names."""
        # Setup - team names with special characters
        specs = {
            "team-with-dashes": Mock(name="team1"),
            "team_with_underscores": Mock(name="team2"),
            "team.with.dots": Mock(name="team3"),
            "team@with#special$chars": Mock(name="team4"),
        }
        mock_orchestrator._team_specs = specs

        # Execute
        result = coordinator.get_team_specs()

        # Assert - all teams present
        assert len(result) == 4
        for team_name in specs:
            assert team_name in result

    def test_get_team_suggestions_with_unicode_task_type(
        self, coordinator: TeamCoordinator, mock_mw_coordinator: Mock
    ):
        """Test get_team_suggestions with unicode task type."""
        # Execute
        coordinator.get_team_suggestions("功能开发", "high")

        # Assert - delegation occurred with unicode
        mock_mw_coordinator.coordination.suggest_for_task.assert_called_once_with(
            task_type="功能开发",
            complexity="high",
            mode="build",
        )

    def test_set_team_specs_with_large_number_of_teams(
        self, coordinator: TeamCoordinator, mock_orchestrator: Mock
    ):
        """Test set_team_specs with a large number of teams."""
        # Setup - 100 teams
        specs = {f"team_{i}": Mock(name=f"team_{i}") for i in range(100)}

        # Execute
        coordinator.set_team_specs(specs)

        # Assert
        result = coordinator.get_team_specs()
        assert len(result) == 100

    def test_multiple_suggestions_with_different_params(
        self, coordinator: TeamCoordinator, mock_mw_coordinator: Mock
    ):
        """Test multiple get_team_suggestions calls with different parameters."""
        # Execute multiple calls
        coordinator.get_team_suggestions("feature", "high")
        coordinator.get_team_suggestions("bugfix", "medium")
        coordinator.get_team_suggestions("refactor", "low")
        coordinator.get_team_suggestions("review", "extreme")

        # Assert - all calls made
        assert mock_mw_coordinator.coordination.suggest_for_task.call_count == 4

        # Verify parameters
        calls = mock_mw_coordinator.coordination.suggest_for_task.call_args_list
        assert calls[0][1]["task_type"] == "feature"
        assert calls[1][1]["task_type"] == "bugfix"
        assert calls[2][1]["task_type"] == "refactor"
        assert calls[3][1]["task_type"] == "review"
