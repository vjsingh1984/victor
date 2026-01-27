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

"""Unit tests for TeamCoordinator.

Tests the team specification and suggestion coordinator extracted from
the monolithic orchestrator as part of Track 4 Phase 1 refactoring.
"""

import pytest
from unittest.mock import Mock, MagicMock

from victor.agent.coordinators.team_coordinator import TeamCoordinator


class TestTeamCoordinator:
    """Test suite for TeamCoordinator."""

    def test_initialization(self):
        """Test coordinator initialization with all dependencies."""
        # Arrange
        orchestrator = Mock()
        mode_coordinator = Mock()
        mode_workflow_team_coordinator = Mock()

        # Act
        coordinator = TeamCoordinator(
            orchestrator=orchestrator,
            mode_coordinator=mode_coordinator,
            mode_workflow_team_coordinator=mode_workflow_team_coordinator,
        )

        # Assert
        assert coordinator._orchestrator == orchestrator
        assert coordinator._mode_coordinator == mode_coordinator
        assert coordinator._mode_workflow_team_coordinator == mode_workflow_team_coordinator

    def test_get_team_suggestions_delegates_to_coordination(self):
        """Test that get_team_suggestions delegates to ModeWorkflowTeamCoordinator."""
        # Arrange
        orchestrator = Mock()
        mode_coordinator = Mock()
        mode_coordinator.current_mode_name = "build"

        mode_workflow_team_coordinator = Mock()
        suggestion = Mock()
        suggestion.recommended_team = "review_team"
        mode_workflow_team_coordinator.coordination.suggest_for_task.return_value = suggestion

        coordinator = TeamCoordinator(
            orchestrator=orchestrator,
            mode_coordinator=mode_coordinator,
            mode_workflow_team_coordinator=mode_workflow_team_coordinator,
        )

        # Act
        result = coordinator.get_team_suggestions("feature", "high")

        # Assert
        assert result == suggestion
        mode_workflow_team_coordinator.coordination.suggest_for_task.assert_called_once_with(
            task_type="feature",
            complexity="high",
            mode="build",
        )

    def test_get_team_suggestions_with_different_modes(self):
        """Test get_team_suggestions with different agent modes."""
        # Arrange
        orchestrator = Mock()
        mode_coordinator = Mock()
        mode_workflow_team_coordinator = Mock()

        coordinator = TeamCoordinator(
            orchestrator=orchestrator,
            mode_coordinator=mode_coordinator,
            mode_workflow_team_coordinator=mode_workflow_team_coordinator,
        )

        # Test with different modes
        for mode in ["build", "plan", "explore"]:
            mode_coordinator.current_mode_name = mode
            suggestion = Mock()
            mode_workflow_team_coordinator.coordination.suggest_for_task.return_value = suggestion

            # Act
            result = coordinator.get_team_suggestions("bugfix", "medium")

            # Assert
            assert result == suggestion
            mode_workflow_team_coordinator.coordination.suggest_for_task.assert_called_with(
                task_type="bugfix",
                complexity="medium",
                mode=mode,
            )

    def test_get_team_suggestions_with_different_task_types(self):
        """Test get_team_suggestions with different task types."""
        # Arrange
        orchestrator = Mock()
        mode_coordinator = Mock()
        mode_coordinator.current_mode_name = "build"
        mode_workflow_team_coordinator = Mock()

        coordinator = TeamCoordinator(
            orchestrator=orchestrator,
            mode_coordinator=mode_coordinator,
            mode_workflow_team_coordinator=mode_workflow_team_coordinator,
        )

        # Test different task types
        task_types = ["feature", "bugfix", "refactor", "documentation", "testing"]
        for task_type in task_types:
            suggestion = Mock()
            mode_workflow_team_coordinator.coordination.suggest_for_task.return_value = suggestion

            # Act
            result = coordinator.get_team_suggestions(task_type, "low")

            # Assert
            assert result == suggestion
            mode_workflow_team_coordinator.coordination.suggest_for_task.assert_called_with(
                task_type=task_type,
                complexity="low",
                mode="build",
            )

    def test_get_team_suggestions_with_different_complexities(self):
        """Test get_team_suggestions with different complexity levels."""
        # Arrange
        orchestrator = Mock()
        mode_coordinator = Mock()
        mode_coordinator.current_mode_name = "plan"
        mode_workflow_team_coordinator = Mock()

        coordinator = TeamCoordinator(
            orchestrator=orchestrator,
            mode_coordinator=mode_coordinator,
            mode_workflow_team_coordinator=mode_workflow_team_coordinator,
        )

        # Test different complexity levels
        complexities = ["low", "medium", "high", "extreme"]
        for complexity in complexities:
            suggestion = Mock()
            mode_workflow_team_coordinator.coordination.suggest_for_task.return_value = suggestion

            # Act
            result = coordinator.get_team_suggestions("refactor", complexity)

            # Assert
            assert result == suggestion
            mode_workflow_team_coordinator.coordination.suggest_for_task.assert_called_with(
                task_type="refactor",
                complexity=complexity,
                mode="plan",
            )

    def test_set_team_specs_stores_in_orchestrator(self):
        """Test that set_team_specs stores specs in orchestrator."""
        # Arrange
        orchestrator = Mock()
        mode_coordinator = Mock()
        mode_workflow_team_coordinator = Mock()

        coordinator = TeamCoordinator(
            orchestrator=orchestrator,
            mode_coordinator=mode_coordinator,
            mode_workflow_team_coordinator=mode_workflow_team_coordinator,
        )

        specs = {
            "review_team": Mock(name="ReviewTeam"),
            "refactor_team": Mock(name="RefactorTeam"),
        }

        # Act
        coordinator.set_team_specs(specs)

        # Assert
        assert orchestrator._team_specs == specs

    def test_set_team_specs_overwrites_existing_specs(self):
        """Test that set_team_specs overwrites existing specs."""
        # Arrange
        orchestrator = Mock()
        orchestrator._team_specs = {"old_team": Mock()}
        mode_coordinator = Mock()
        mode_workflow_team_coordinator = Mock()

        coordinator = TeamCoordinator(
            orchestrator=orchestrator,
            mode_coordinator=mode_coordinator,
            mode_workflow_team_coordinator=mode_workflow_team_coordinator,
        )

        new_specs = {"new_team": Mock()}

        # Act
        coordinator.set_team_specs(new_specs)

        # Assert
        assert orchestrator._team_specs == new_specs

    def test_set_team_specs_with_empty_dict(self):
        """Test setting empty team specs."""
        # Arrange
        orchestrator = Mock()
        mode_coordinator = Mock()
        mode_workflow_team_coordinator = Mock()

        coordinator = TeamCoordinator(
            orchestrator=orchestrator,
            mode_coordinator=mode_coordinator,
            mode_workflow_team_coordinator=mode_workflow_team_coordinator,
        )

        # Act
        coordinator.set_team_specs({})

        # Assert
        assert orchestrator._team_specs == {}

    def test_get_team_specs_returns_specs_from_orchestrator(self):
        """Test that get_team_specs retrieves specs from orchestrator."""
        # Arrange
        orchestrator = Mock()
        orchestrator._team_specs = {"team1": Mock(), "team2": Mock()}
        mode_coordinator = Mock()
        mode_workflow_team_coordinator = Mock()

        coordinator = TeamCoordinator(
            orchestrator=orchestrator,
            mode_coordinator=mode_coordinator,
            mode_workflow_team_coordinator=mode_workflow_team_coordinator,
        )

        # Act
        specs = coordinator.get_team_specs()

        # Assert
        assert specs == orchestrator._team_specs
        assert len(specs) == 2

    def test_get_team_specs_returns_empty_dict_when_not_set(self):
        """Test that get_team_specs returns empty dict when specs not set."""
        # Arrange
        orchestrator = Mock()
        # Configure mock to return None/empty values for fallback checks
        orchestrator.vertical_context = None
        orchestrator.configure_mock(**{"_team_specs": {}})
        mode_coordinator = Mock()
        mode_workflow_team_coordinator = Mock()

        coordinator = TeamCoordinator(
            orchestrator=orchestrator,
            mode_coordinator=mode_coordinator,
            mode_workflow_team_coordinator=mode_workflow_team_coordinator,
        )

        # Act
        specs = coordinator.get_team_specs()

        # Assert
        assert specs == {}

    def test_set_and_get_team_specs_roundtrip(self):
        """Test setting and getting team specs maintains data."""
        # Arrange
        orchestrator = Mock()
        mode_coordinator = Mock()
        mode_workflow_team_coordinator = Mock()

        coordinator = TeamCoordinator(
            orchestrator=orchestrator,
            mode_coordinator=mode_coordinator,
            mode_workflow_team_coordinator=mode_workflow_team_coordinator,
        )

        original_specs = {
            "team_a": Mock(name="TeamA"),
            "team_b": Mock(name="TeamB"),
            "team_c": Mock(name="TeamC"),
        }

        # Act
        coordinator.set_team_specs(original_specs)
        retrieved_specs = coordinator.get_team_specs()

        # Assert
        assert retrieved_specs == original_specs
        assert len(retrieved_specs) == 3

    def test_multiple_set_team_specs_calls(self):
        """Test multiple calls to set_team_specs."""
        # Arrange
        orchestrator = Mock()
        mode_coordinator = Mock()
        mode_workflow_team_coordinator = Mock()

        coordinator = TeamCoordinator(
            orchestrator=orchestrator,
            mode_coordinator=mode_coordinator,
            mode_workflow_team_coordinator=mode_workflow_team_coordinator,
        )

        # Act - Set multiple times
        team1 = Mock(name="team1")
        team2 = Mock(name="team2")
        team3 = Mock(name="team3")
        coordinator.set_team_specs({"team1": team1})
        coordinator.set_team_specs({"team2": team2})
        coordinator.set_team_specs({"team3": team3})

        # Assert - Should have the last value
        assert orchestrator._team_specs == {"team3": team3}

    def test_get_team_suggestions_complex_scenario(self):
        """Test complex scenario with multiple calls."""
        # Arrange
        orchestrator = Mock()
        mode_coordinator = Mock()
        mode_workflow_team_coordinator = Mock()

        coordinator = TeamCoordinator(
            orchestrator=orchestrator,
            mode_coordinator=mode_coordinator,
            mode_workflow_team_coordinator=mode_workflow_team_coordinator,
        )

        # Set some team specs
        specs = {
            "review_team": Mock(name="ReviewTeam"),
        }
        coordinator.set_team_specs(specs)

        # Get suggestions for different scenarios
        mode_coordinator.current_mode_name = "plan"
        suggestion1 = Mock()
        mode_workflow_team_coordinator.coordination.suggest_for_task.return_value = suggestion1

        # Act
        result1 = coordinator.get_team_suggestions("feature", "high")

        # Assert
        assert result1 == suggestion1
        mode_workflow_team_coordinator.coordination.suggest_for_task.assert_called_once_with(
            task_type="feature",
            complexity="high",
            mode="plan",
        )
