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

"""Tests for StateCoordinator.

Tests the state coordination functionality including:
- Stage tracking and transitions
- Message history management
- Transition history
"""

import pytest
from unittest.mock import MagicMock, PropertyMock

from victor.agent.conversation_state import ConversationStage
from victor.agent.state_coordinator import (
    StateCoordinator,
    StateCoordinatorConfig,
    StageTransition,
    create_state_coordinator,
)


class TestStateCoordinatorConfig:
    """Tests for StateCoordinatorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StateCoordinatorConfig()

        assert config.enable_auto_transitions is True
        assert config.enable_history_tracking is True
        assert config.max_history_length == 100
        assert config.emit_events is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = StateCoordinatorConfig(
            enable_auto_transitions=False,
            enable_history_tracking=False,
            max_history_length=50,
            emit_events=False,
        )

        assert config.enable_auto_transitions is False
        assert config.enable_history_tracking is False
        assert config.max_history_length == 50
        assert config.emit_events is False


class TestStageTransition:
    """Tests for StageTransition dataclass."""

    def test_basic_transition(self):
        """Test basic transition record."""
        transition = StageTransition(
            from_stage=ConversationStage.INITIAL,
            to_stage=ConversationStage.PLANNING,
            reason="User requested planning",
        )

        assert transition.from_stage == ConversationStage.INITIAL
        assert transition.to_stage == ConversationStage.PLANNING
        assert transition.reason == "User requested planning"
        assert transition.tool_name is None
        assert transition.confidence == 1.0

    def test_transition_with_tool(self):
        """Test transition record with tool name."""
        transition = StageTransition(
            from_stage=ConversationStage.READING,
            to_stage=ConversationStage.EXECUTION,
            reason="Edit tool triggered execution",
            tool_name="edit",
            confidence=0.9,
        )

        assert transition.tool_name == "edit"
        assert transition.confidence == 0.9


class TestStateCoordinator:
    """Tests for StateCoordinator."""

    @pytest.fixture
    def mock_controller(self):
        """Create mock conversation controller."""
        controller = MagicMock()
        controller.messages = []
        controller.message_count = 0
        type(controller).stage = PropertyMock(return_value=ConversationStage.INITIAL)
        return controller

    @pytest.fixture
    def mock_state_machine(self):
        """Create mock state machine."""
        machine = MagicMock()
        machine.get_stage.return_value = ConversationStage.INITIAL
        return machine

    @pytest.fixture
    def coordinator(self, mock_controller, mock_state_machine):
        """Create coordinator with mocks."""
        return StateCoordinator(
            conversation_controller=mock_controller,
            state_machine=mock_state_machine,
        )

    def test_init_default_config(self, mock_controller):
        """Test initialization with default config."""
        coordinator = StateCoordinator(conversation_controller=mock_controller)

        assert coordinator._config.enable_auto_transitions is True
        assert coordinator._config.emit_events is True

    def test_get_current_stage(self, coordinator, mock_state_machine):
        """Test getting current stage."""
        mock_state_machine.get_stage.return_value = ConversationStage.READING

        assert coordinator.get_current_stage() == ConversationStage.READING

    def test_stage_property(self, coordinator, mock_state_machine):
        """Test stage property."""
        mock_state_machine.get_stage.return_value = ConversationStage.ANALYSIS

        assert coordinator.stage == ConversationStage.ANALYSIS

    def test_transition_to_success(self, coordinator, mock_state_machine):
        """Test successful stage transition."""
        # Start from INITIAL stage
        mock_state_machine.get_stage.return_value = ConversationStage.INITIAL

        result = coordinator.transition_to(
            ConversationStage.PLANNING,
            reason="Planning phase",
        )

        assert result is True
        # The implementation uses _transition_to (internal method), not set_stage
        mock_state_machine._transition_to.assert_called_once_with(
            ConversationStage.PLANNING, confidence=0.8
        )

    def test_transition_to_same_stage(self, coordinator, mock_state_machine):
        """Test transition to same stage (no-op)."""
        mock_state_machine.get_stage.return_value = ConversationStage.INITIAL

        result = coordinator.transition_to(ConversationStage.INITIAL)

        assert result is True
        mock_state_machine.set_stage.assert_not_called()

    def test_transition_records_history(self, coordinator, mock_state_machine):
        """Test that transitions are recorded in history."""
        mock_state_machine.get_stage.return_value = ConversationStage.INITIAL

        coordinator.transition_to(ConversationStage.PLANNING, reason="Test")

        history = coordinator.get_transition_history()
        assert len(history) == 1
        assert history[0].from_stage == ConversationStage.INITIAL
        assert history[0].to_stage == ConversationStage.PLANNING

    def test_get_message_history(self, coordinator, mock_controller):
        """Test getting message history."""
        mock_messages = [MagicMock(), MagicMock()]
        mock_controller.messages = mock_messages

        result = coordinator.get_message_history()

        assert result == mock_messages

    def test_get_recent_messages(self, coordinator, mock_controller):
        """Test getting recent messages."""
        mock_messages = [
            MagicMock(role="system"),
            MagicMock(role="user"),
            MagicMock(role="assistant"),
            MagicMock(role="user"),
        ]
        mock_controller.messages = mock_messages

        # Without system messages
        result = coordinator.get_recent_messages(limit=2, include_system=False)
        assert len(result) == 2

    def test_is_in_exploration_phase(self, coordinator, mock_state_machine):
        """Test exploration phase detection."""
        mock_state_machine.get_stage.return_value = ConversationStage.READING

        assert coordinator.is_in_exploration_phase() is True

        mock_state_machine.get_stage.return_value = ConversationStage.EXECUTION
        assert coordinator.is_in_exploration_phase() is False

    def test_is_in_execution_phase(self, coordinator, mock_state_machine):
        """Test execution phase detection."""
        mock_state_machine.get_stage.return_value = ConversationStage.EXECUTION

        assert coordinator.is_in_execution_phase() is True

        mock_state_machine.get_stage.return_value = ConversationStage.READING
        assert coordinator.is_in_execution_phase() is False

    def test_is_in_completion_phase(self, coordinator, mock_state_machine):
        """Test completion phase detection."""
        mock_state_machine.get_stage.return_value = ConversationStage.COMPLETION

        assert coordinator.is_in_completion_phase() is True

        mock_state_machine.get_stage.return_value = ConversationStage.VERIFICATION
        assert coordinator.is_in_completion_phase() is True

        mock_state_machine.get_stage.return_value = ConversationStage.EXECUTION
        assert coordinator.is_in_completion_phase() is False

    def test_get_stage_sequence(self, coordinator, mock_state_machine):
        """Test getting stage sequence."""
        mock_state_machine.get_stage.return_value = ConversationStage.INITIAL
        coordinator.transition_to(ConversationStage.PLANNING)

        mock_state_machine.get_stage.return_value = ConversationStage.PLANNING
        coordinator.transition_to(ConversationStage.READING)

        sequence = coordinator.get_stage_sequence()
        assert len(sequence) == 3
        assert sequence[0] == ConversationStage.INITIAL
        assert sequence[1] == ConversationStage.PLANNING
        assert sequence[2] == ConversationStage.READING

    def test_get_state_snapshot(self, coordinator, mock_state_machine, mock_controller):
        """Test state snapshot creation."""
        mock_state_machine.get_stage.return_value = ConversationStage.READING
        mock_controller.message_count = 5

        snapshot = coordinator.get_state_snapshot()

        assert snapshot["stage"] == "reading"
        assert snapshot["message_count"] == 5
        assert "transition_count" in snapshot
        assert "last_transitions" in snapshot

    def test_clear_history(self, coordinator, mock_state_machine):
        """Test clearing transition history."""
        mock_state_machine.get_stage.return_value = ConversationStage.INITIAL
        coordinator.transition_to(ConversationStage.PLANNING)
        coordinator.transition_to(ConversationStage.READING)

        assert len(coordinator.get_transition_history()) == 2

        coordinator.clear_history()
        assert len(coordinator.get_transition_history()) == 0

    def test_on_stage_change_callback(self, mock_controller, mock_state_machine):
        """Test stage change callback."""
        callback_calls = []

        def on_change(old, new):
            callback_calls.append((old, new))

        coordinator = StateCoordinator(
            conversation_controller=mock_controller,
            state_machine=mock_state_machine,
            on_stage_change=on_change,
        )

        mock_state_machine.get_stage.return_value = ConversationStage.INITIAL
        coordinator.transition_to(ConversationStage.PLANNING)

        assert len(callback_calls) == 1
        assert callback_calls[0] == (ConversationStage.INITIAL, ConversationStage.PLANNING)


class TestCreateStateCoordinator:
    """Tests for create_state_coordinator factory function."""

    def test_create_basic(self):
        """Test basic factory creation."""
        mock_controller = MagicMock()
        type(mock_controller).stage = PropertyMock(return_value=ConversationStage.INITIAL)

        coordinator = create_state_coordinator(conversation_controller=mock_controller)

        assert isinstance(coordinator, StateCoordinator)

    def test_create_with_config(self):
        """Test factory creation with config."""
        mock_controller = MagicMock()
        type(mock_controller).stage = PropertyMock(return_value=ConversationStage.INITIAL)
        config = StateCoordinatorConfig(enable_auto_transitions=False)

        coordinator = create_state_coordinator(
            conversation_controller=mock_controller,
            config=config,
        )

        assert coordinator._config.enable_auto_transitions is False
