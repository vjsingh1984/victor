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

"""Tests for StateCoordinator (unified state management).

Tests the state coordination functionality including:
- Unified state access across scopes
- State change notifications (Observer pattern)
- Session state delegation
- Checkpoint state serialization
"""

import pytest
from unittest.mock import MagicMock, PropertyMock
from dataclasses import dataclass

from victor.agent.coordinators.state_coordinator import (
    StateCoordinator,
    StateScope,
    StateChange,
    StateObserver,
    create_state_coordinator,
)


@dataclass
class MockExecutionState:
    """Mock execution state."""
    executed_tools: list = None
    observed_files: set = None
    tool_calls_used: int = 0

    def __post_init__(self):
        if self.executed_tools is None:
            self.executed_tools = []
        if self.observed_files is None:
            self.observed_files = set()


class MockSessionStateManager:
    """Mock session state manager."""

    def __init__(self):
        self.execution_state = MockExecutionState()
        self._tool_budget = 100

    def get_checkpoint_state(self):
        """Get checkpoint state."""
        return {
            "executed_tools": self.execution_state.executed_tools,
            "observed_files": list(self.execution_state.observed_files),
            "tool_calls_used": self.execution_state.tool_calls_used,
            "tool_budget": self._tool_budget,
        }

    def apply_checkpoint_state(self, state):
        """Apply checkpoint state."""
        self.execution_state.executed_tools = state.get("executed_tools", [])
        self.execution_state.observed_files = set(state.get("observed_files", []))
        self.execution_state.tool_calls_used = state.get("tool_calls_used", 0)
        self._tool_budget = state.get("tool_budget", 100)

    @property
    def tool_calls_used(self):
        return self.execution_state.tool_calls_used

    @property
    def observed_files(self):
        return self.execution_state.observed_files

    @property
    def executed_tools(self):
        return self.execution_state.executed_tools

    @property
    def tool_budget(self):
        return self._tool_budget

    @tool_budget.setter
    def tool_budget(self, value):
        self._tool_budget = value

    def is_budget_exhausted(self):
        return self.execution_state.tool_calls_used >= self._tool_budget

    def get_remaining_budget(self):
        return self._tool_budget - self.execution_state.tool_calls_used

    def record_tool_call(self, tool_name, args):
        self.execution_state.executed_tools.append(tool_name)
        self.execution_state.tool_calls_used += 1

    def increment_tool_calls(self, count=1):
        self.execution_state.tool_calls_used += count
        return self.execution_state.tool_calls_used

    def record_file_read(self, filepath):
        self.execution_state.observed_files.add(filepath)

    def get_session_summary(self):
        return {
            "tool_calls_used": self.execution_state.tool_calls_used,
            "budget_remaining": self.get_remaining_budget(),
        }


class MockConversationStage:
    """Mock conversation stage."""
    INITIAL = None
    EXECUTION = None

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        if isinstance(other, MockConversationStage):
            return self.name == other.name
        return False

    def __repr__(self):
        return f"MockConversationStage({self.name})"


class MockConversationStateMachine:
    """Mock conversation state machine."""

    def __init__(self):
        self._stage = MockConversationStage("INITIAL")
        self._recorded_tools = []

    def get_stage(self):
        # Return a mock stage object with .name attribute
        if isinstance(self._stage, str):
            # For backward compatibility with tests that set _stage to string
            return MockConversationStage(self._stage)
        return self._stage

    def to_dict(self):
        return {"stage": self._stage.name if hasattr(self._stage, 'name') else self._stage}

    def _transition_to(self, stage, confidence=1.0):
        # Handle both string and enum-like objects
        if isinstance(stage, str):
            self._stage = stage
        elif hasattr(stage, 'name'):
            self._stage = stage.name
        elif hasattr(stage, 'value'):
            self._stage = stage.value
        else:
            self._stage = str(stage)

    def get_stage_tools(self):
        """Return tools relevant to current stage."""
        return {"read_file", "write_file"}  # Mock set of tools

    def record_tool_execution(self, tool_name, args):
        """Record tool execution in conversation state."""
        self._recorded_tools.append(tool_name)

    def get_state_summary(self):
        """Get state summary."""
        return {
            "stage": self._stage.name if hasattr(self._stage, 'name') else self._stage,
            "recorded_tools": self._recorded_tools,
        }


class TestStateScope:
    """Tests for StateScope enum."""

    def test_scope_values(self):
        """Test scope enum values."""
        assert StateScope.SESSION.value == "session"
        assert StateScope.CONVERSATION.value == "conversation"
        assert StateScope.CHECKPOINT.value == "checkpoint"
        assert StateScope.ALL.value == "all"


class TestStateChange:
    """Tests for StateChange dataclass."""

    def test_state_change_creation(self):
        """Test creating a state change event."""
        change = StateChange(
            scope=StateScope.SESSION,
            old_state={"stage": "initial"},
            new_state={"stage": "execution"},
        )

        assert change.scope == StateScope.SESSION
        assert change.old_state == {"stage": "initial"}
        assert change.new_state == {"stage": "execution"}

    def test_state_change_with_changes(self):
        """Test state change with specific changes."""
        change = StateChange(
            scope=StateScope.SESSION,
            old_state={"stage": "initial"},
            new_state={"stage": "execution"},
            changes={"stage": ("initial", "execution")},
        )

        assert change.changes == {"stage": ("initial", "execution")}

    def test_to_dict(self):
        """Test converting state change to dict."""
        change = StateChange(
            scope=StateScope.CONVERSATION,
            old_state={"stage": "initial"},
            new_state={"stage": "execution"},
        )

        result = change.to_dict()

        assert result["scope"] == "conversation"
        assert result["old_state"] == {"stage": "initial"}
        assert result["new_state"] == {"stage": "execution"}
        assert "timestamp" in result


class TestStateCoordinator:
    """Tests for StateCoordinator."""

    @pytest.fixture
    def mock_session_state(self):
        """Create mock session state manager."""
        return MockSessionStateManager()

    @pytest.fixture
    def mock_conversation_state(self):
        """Create mock conversation state machine."""
        return MockConversationStateMachine()

    @pytest.fixture
    def coordinator(self, mock_session_state, mock_conversation_state):
        """Create coordinator with mocks."""
        return StateCoordinator(
            session_state_manager=mock_session_state,
            conversation_state_machine=mock_conversation_state,
        )

    def test_init(self, mock_session_state, mock_conversation_state):
        """Test initialization."""
        coordinator = StateCoordinator(
            session_state_manager=mock_session_state,
            conversation_state_machine=mock_conversation_state,
            enable_history=True,
            max_history_size=50,
        )

        assert coordinator._session_state == mock_session_state
        assert coordinator._conversation_state == mock_conversation_state
        assert coordinator._enable_history is True
        assert coordinator._max_history_size == 50
        assert len(coordinator._observers) == 0

    def test_init_without_conversation_state(self, mock_session_state):
        """Test initialization without conversation state."""
        coordinator = StateCoordinator(
            session_state_manager=mock_session_state,
        )

        assert coordinator._conversation_state is None

    def test_get_state_all_scopes(self, coordinator):
        """Test getting state from all scopes."""
        state = coordinator.get_state(scope=StateScope.ALL)

        assert "session" in state
        assert "conversation" in state
        assert "checkpoint" in state
        assert "_metadata" in state

    def test_get_state_session_only(self, coordinator):
        """Test getting only session state."""
        state = coordinator.get_state(scope=StateScope.SESSION)

        assert "session" in state
        assert "conversation" not in state
        assert "checkpoint" not in state

    def test_get_state_without_metadata(self, coordinator):
        """Test getting state without metadata."""
        state = coordinator.get_state(include_metadata=False)

        assert "_metadata" not in state

    def test_set_state(self, coordinator, mock_session_state):
        """Test setting state."""
        new_state = {
            "session": {
                "executed_tools": ["test_tool"],
                "tool_calls_used": 5,
            }
        }

        coordinator.set_state(new_state, scope=StateScope.SESSION)

        # Verify state was applied
        assert "test_tool" in mock_session_state.executed_tools
        assert mock_session_state.tool_calls_used == 5

    def test_transition_to(self, coordinator, mock_conversation_state):
        """Test stage transition."""
        # Create mock stage using lowercase (matching ConversationStage enum values)
        execution_stage = MockConversationStage("execution")

        result = coordinator.transition_to(execution_stage)

        assert result is True
        # The _stage should now be a string (extracted from the mock stage object)
        assert mock_conversation_state._stage == "execution"

    def test_transition_to_without_conversation_state(
        self, mock_session_state
    ):
        """Test transition without conversation state."""
        coordinator = StateCoordinator(
            session_state_manager=mock_session_state,
        )

        execution_stage = MockConversationStage("execution")
        result = coordinator.transition_to(execution_stage)

        assert result is False

    def test_transition_to_string(self, coordinator, mock_conversation_state):
        """Test transition using string stage name."""
        # Use lowercase to match ConversationStage enum values
        result = coordinator.transition_to("EXECUTION")

        assert result is True
        # The _stage should be the lowercase string "execution" (enum value)
        assert mock_conversation_state._stage == "execution"

    def test_get_stage(self, coordinator, mock_conversation_state):
        """Test getting current stage."""
        # Set the stage as a string (as it would be stored internally)
        mock_conversation_state._stage = "READING"

        stage = coordinator.get_stage()

        # get_stage() returns the .name attribute from the mock stage object
        assert stage == "READING"

    def test_get_stage_without_conversation_state(self, mock_session_state):
        """Test getting stage without conversation state."""
        coordinator = StateCoordinator(
            session_state_manager=mock_session_state,
        )

        stage = coordinator.get_stage()

        assert stage is None

    def test_get_stage_tools(self, coordinator):
        """Test getting stage tools."""
        # This depends on the actual implementation
        tools = coordinator.get_stage_tools()

        # Just verify it returns a set
        assert isinstance(tools, set)

    def test_session_state_delegation_properties(self, coordinator):
        """Test session state property delegation."""
        # Set some values
        coordinator._session_state.execution_state.tool_calls_used = 10
        coordinator._session_state.execution_state.observed_files = {"/test/file"}

        assert coordinator.tool_calls_used == 10
        assert "/test/file" in coordinator.observed_files

    def test_tool_budget_property(self, coordinator):
        """Test tool budget property."""
        assert coordinator.tool_budget == 100

    def test_tool_budget_setter(self, coordinator):
        """Test setting tool budget."""
        notifications = []

        def observer(change):
            notifications.append(change)

        coordinator.subscribe(observer)

        coordinator.tool_budget = 200

        assert coordinator.tool_budget == 200
        assert len(notifications) == 1

    def test_is_budget_exhausted(self, coordinator):
        """Test budget exhaustion check."""
        coordinator._session_state.execution_state.tool_calls_used = 0
        assert coordinator.is_budget_exhausted() is False

        coordinator._session_state.execution_state.tool_calls_used = 100
        assert coordinator.is_budget_exhausted() is True

    def test_get_remaining_budget(self, coordinator):
        """Test getting remaining budget."""
        assert coordinator.get_remaining_budget() == 100

        coordinator._session_state.execution_state.tool_calls_used = 30
        assert coordinator.get_remaining_budget() == 70

    def test_record_tool_call(self, coordinator):
        """Test recording tool call."""
        coordinator.record_tool_call("read_file", {"path": "/test"})

        assert "read_file" in coordinator.executed_tools

    def test_increment_tool_calls(self, coordinator):
        """Test incrementing tool calls."""
        notifications = []

        def observer(change):
            notifications.append(change)

        coordinator.subscribe(observer)

        old_count = coordinator.tool_calls_used
        new_count = coordinator.increment_tool_calls(5)

        assert new_count == old_count + 5
        assert len(notifications) == 1

    def test_record_file_read(self, coordinator):
        """Test recording file read."""
        coordinator.record_file_read("/test/file.py")

        assert "/test/file.py" in coordinator.observed_files

    def test_subscribe(self, coordinator):
        """Test subscribing to state changes."""
        notifications = []

        def observer(change):
            notifications.append(change)

        unsubscribe = coordinator.subscribe(observer)

        # Trigger a state change
        coordinator.increment_tool_calls(1)

        assert len(notifications) == 1

        # Unsubscribe
        unsubscribe()
        coordinator.increment_tool_calls(1)

        # Should still be 1 (no new notification)
        assert len(notifications) == 1

    def test_unsubscribe(self, coordinator):
        """Test unsubscribing observer."""
        notifications = []

        def observer(change):
            notifications.append(change)

        coordinator.subscribe(observer)
        coordinator.unsubscribe(observer)

        coordinator.increment_tool_calls(1)

        assert len(notifications) == 0

    def test_unsubscribe_all(self, coordinator):
        """Test unsubscribing all observers."""
        observer1 = MagicMock()
        observer2 = MagicMock()

        coordinator.subscribe(observer1)
        coordinator.subscribe(observer2)

        assert len(coordinator._observers) == 2

        coordinator.unsubscribe_all()

        assert len(coordinator._observers) == 0

    def test_on_state_change_decorator(self, coordinator):
        """Test the on_state_change decorator."""
        notifications = []

        @coordinator.on_state_change
        def observer(change):
            notifications.append(change)

        coordinator.increment_tool_calls(1)

        assert len(notifications) == 1

    def test_get_state_history(self, coordinator):
        """Test getting state change history."""
        coordinator.increment_tool_calls(1)
        coordinator.increment_tool_calls(1)

        history = coordinator.get_state_history()

        assert len(history) == 2

    def test_get_state_history_with_limit(self, coordinator):
        """Test getting limited history."""
        for _ in range(10):
            coordinator.increment_tool_calls(1)

        history = coordinator.get_state_history(limit=5)

        assert len(history) == 5

    def test_get_state_changes_count(self, coordinator):
        """Test getting state changes count."""
        assert coordinator.get_state_changes_count() == 0

        coordinator.increment_tool_calls(1)

        assert coordinator.get_state_changes_count() == 1

    def test_clear_state_history(self, coordinator):
        """Test clearing state history."""
        coordinator.increment_tool_calls(1)
        coordinator.increment_tool_calls(1)

        assert coordinator.get_state_changes_count() == 2

        coordinator.clear_state_history()

        assert coordinator.get_state_changes_count() == 0

    def test_get_state_summary(self, coordinator):
        """Test getting state summary."""
        summary = coordinator.get_state_summary()

        assert "session" in summary
        assert "state_changes_count" in summary

    def test_repr(self, coordinator):
        """Test string representation."""
        coordinator._session_state.execution_state.tool_calls_used = 5
        # Set the stage as a string (as it would be stored internally)
        coordinator._conversation_state._stage = "execution"

        result = repr(coordinator)

        assert "StateCoordinator" in result
        assert "execution" in result


class TestCreateStateCoordinator:
    """Tests for create_state_coordinator factory function."""

    def test_create_basic(self):
        """Test basic factory creation."""
        mock_session = MagicMock()

        coordinator = create_state_coordinator(
            session_state_manager=mock_session,
        )

        assert isinstance(coordinator, StateCoordinator)

    def test_create_with_all_params(self):
        """Test factory with all parameters."""
        mock_session = MagicMock()
        mock_conv = MagicMock()

        coordinator = create_state_coordinator(
            session_state_manager=mock_session,
            conversation_state_machine=mock_conv,
            enable_history=False,
            max_history_size=50,
        )

        assert coordinator._enable_history is False
        assert coordinator._max_history_size == 50
