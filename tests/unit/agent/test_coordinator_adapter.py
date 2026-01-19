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

"""Unit tests for CoordinatorAdapter.

Tests adapter for coordinator integration (checkpointing, RL rewards).
"""

import pytest
from unittest.mock import AsyncMock, Mock

from victor.agent.adapters.coordinator_adapter import CoordinatorAdapter


# ============================================================================
# Test Fixtures
# ============================================================================


class MockStreamingSession:
    """Mock StreamingSession for testing."""

    def __init__(self, success=True, duration=1.5):
        self.success = success
        self.duration = duration
        self.tool_calls_used = 5
        self.latency = 0.5
        self.throughput = 10.0


class MockConversation:
    """Mock Conversation for testing."""

    def __init__(self):
        self.messages = [
            Mock(role="user", content="Hello"),
            Mock(role="assistant", content="Hi there!"),
        ]


class MockConversationController:
    """Mock ConversationController for testing."""

    def __init__(self):
        self._modified_files = {"file1.py", "file2.py"}
        self.conversation = MockConversation()


# ============================================================================
# CoordinatorAdapter Tests
# ============================================================================


class TestCoordinatorAdapter:
    """Test suite for CoordinatorAdapter."""

    def test_initialization_with_no_components(self):
        """Test adapter initialization with no components."""
        adapter = CoordinatorAdapter()

        assert adapter.state_coordinator is None
        assert adapter.evaluation_coordinator is None
        assert adapter.conversation_controller is None
        assert adapter.is_enabled is False

    def test_initialization_with_components(self):
        """Test adapter initialization with components."""
        mock_state_coordinator = Mock()
        mock_eval_coordinator = Mock()
        mock_conversation_controller = Mock()

        adapter = CoordinatorAdapter(
            state_coordinator=mock_state_coordinator,
            evaluation_coordinator=mock_eval_coordinator,
            conversation_controller=mock_conversation_controller,
        )

        assert adapter.state_coordinator is mock_state_coordinator
        assert adapter.evaluation_coordinator is mock_eval_coordinator
        assert adapter.conversation_controller is mock_conversation_controller
        assert adapter.is_enabled is True

    def test_send_rl_reward_signal_with_coordinator(self):
        """Test sending RL reward signal with coordinator available."""
        mock_eval_coordinator = Mock()
        mock_eval_coordinator.send_rl_reward_signal = Mock()

        adapter = CoordinatorAdapter(
            evaluation_coordinator=mock_eval_coordinator
        )

        session = MockStreamingSession(success=True)
        adapter.send_rl_reward_signal(session)

        mock_eval_coordinator.send_rl_reward_signal.assert_called_once_with(session)

    def test_send_rl_reward_signal_without_coordinator(self):
        """Test sending RL reward signal without coordinator."""
        adapter = CoordinatorAdapter()

        session = MockStreamingSession()
        # Should not raise exception
        adapter.send_rl_reward_signal(session)

    def test_send_rl_reward_signal_with_exception(self):
        """Test sending RL reward signal when coordinator raises exception."""
        mock_eval_coordinator = Mock()
        mock_eval_coordinator.send_rl_reward_signal = Mock(
            side_effect=Exception("Failed to send reward")
        )

        adapter = CoordinatorAdapter(
            evaluation_coordinator=mock_eval_coordinator
        )

        session = MockStreamingSession()
        # Should not raise exception
        adapter.send_rl_reward_signal(session)

    def test_get_checkpoint_state_with_coordinator(self):
        """Test getting checkpoint state with coordinator available."""
        mock_state_coordinator = Mock()
        mock_state_coordinator.get_state = Mock(
            return_value={
                "checkpoint": {
                    "stage": "ANALYZING",
                    "tool_history": ["read", "write"],
                    "observed_files": ["file1.py"],
                    "tool_calls_used": 5,
                }
            }
        )

        mock_conversation_controller = MockConversationController()

        adapter = CoordinatorAdapter(
            state_coordinator=mock_state_coordinator,
            conversation_controller=mock_conversation_controller,
        )

        state = adapter.get_checkpoint_state()

        assert state["stage"] == "ANALYZING"
        assert state["tool_history"] == ["read", "write"]
        assert state["observed_files"] == ["file1.py"]
        assert state["tool_calls_used"] == 5
        assert set(state["modified_files"]) == {"file1.py", "file2.py"}
        assert state["message_count"] == 2
        mock_state_coordinator.get_state.assert_called_once()

    def test_get_checkpoint_state_without_coordinator(self):
        """Test getting checkpoint state without coordinator."""
        adapter = CoordinatorAdapter()

        state = adapter.get_checkpoint_state()

        assert state == {}

    def test_get_checkpoint_state_with_exception(self):
        """Test getting checkpoint state when coordinator raises exception."""
        mock_state_coordinator = Mock()
        mock_state_coordinator.get_state = Mock(
            side_effect=Exception("Failed to get state")
        )

        adapter = CoordinatorAdapter(state_coordinator=mock_state_coordinator)

        state = adapter.get_checkpoint_state()

        assert state == {}

    def test_apply_checkpoint_state_with_coordinator(self):
        """Test applying checkpoint state with coordinator available."""
        mock_state_coordinator = Mock()
        mock_state_coordinator.set_state = Mock()

        adapter = CoordinatorAdapter(state_coordinator=mock_state_coordinator)

        checkpoint_state = {
            "stage": "EXECUTING",
            "tool_history": ["test", "debug"],
            "observed_files": ["test.py"],
            "tool_calls_used": 10,
        }

        adapter.apply_checkpoint_state(checkpoint_state)

        mock_state_coordinator.set_state.assert_called_once()
        call_args = mock_state_coordinator.set_state.call_args
        assert call_args[0][0]["checkpoint"]["stage"] == "EXECUTING"
        assert call_args[0][0]["checkpoint"]["tool_history"] == ["test", "debug"]

    def test_apply_checkpoint_state_without_coordinator(self):
        """Test applying checkpoint state without coordinator."""
        adapter = CoordinatorAdapter()

        # Should not raise exception
        adapter.apply_checkpoint_state({"stage": "ANALYZING"})

    def test_apply_checkpoint_state_with_empty_state(self):
        """Test applying empty checkpoint state."""
        mock_state_coordinator = Mock()
        mock_state_coordinator.set_state = Mock()

        adapter = CoordinatorAdapter(state_coordinator=mock_state_coordinator)

        adapter.apply_checkpoint_state({})

        # Should not call set_state with empty state
        mock_state_coordinator.set_state.assert_not_called()

    def test_apply_checkpoint_state_with_exception(self):
        """Test applying checkpoint state when coordinator raises exception."""
        mock_state_coordinator = Mock()
        mock_state_coordinator.set_state = Mock(
            side_effect=Exception("Failed to set state")
        )

        adapter = CoordinatorAdapter(state_coordinator=mock_state_coordinator)

        # Should not raise exception
        adapter.apply_checkpoint_state({"stage": "ANALYZING"})

    @pytest.mark.asyncio
    async def test_record_intelligent_outcome_with_coordinator(self):
        """Test recording intelligent outcome with coordinator available."""
        mock_eval_coordinator = Mock()
        mock_eval_coordinator.record_intelligent_outcome = AsyncMock()

        adapter = CoordinatorAdapter(
            evaluation_coordinator=mock_eval_coordinator
        )

        await adapter.record_intelligent_outcome(
            success=True,
            quality_score=0.9,
            user_satisfied=True,
            completed=True,
        )

        mock_eval_coordinator.record_intelligent_outcome.assert_called_once_with(
            success=True,
            quality_score=0.9,
            user_satisfied=True,
            completed=True,
        )

    @pytest.mark.asyncio
    async def test_record_intelligent_outcome_without_coordinator(self):
        """Test recording intelligent outcome without coordinator."""
        adapter = CoordinatorAdapter()

        # Should not raise exception
        await adapter.record_intelligent_outcome(
            success=True, quality_score=0.8, user_satisfied=False, completed=False
        )

    @pytest.mark.asyncio
    async def test_record_intelligent_outcome_with_exception(self):
        """Test recording intelligent outcome when coordinator raises exception."""
        mock_eval_coordinator = Mock()
        mock_eval_coordinator.record_intelligent_outcome = AsyncMock(
            side_effect=Exception("Recording failed")
        )

        adapter = CoordinatorAdapter(
            evaluation_coordinator=mock_eval_coordinator
        )

        # Should not raise exception
        await adapter.record_intelligent_outcome(
            success=False, quality_score=0.5, user_satisfied=True, completed=True
        )

    def test_full_checkpoint_flow(self):
        """Test complete checkpoint save and restore flow."""
        mock_state_coordinator = Mock()
        mock_state_coordinator.get_state = Mock(
            return_value={
                "checkpoint": {
                    "stage": "PLANNING",
                    "tool_history": ["read"],
                    "observed_files": [],
                    "tool_calls_used": 1,
                }
            }
        )
        mock_state_coordinator.set_state = Mock()

        mock_conversation_controller = MockConversationController()

        adapter = CoordinatorAdapter(
            state_coordinator=mock_state_coordinator,
            conversation_controller=mock_conversation_controller,
        )

        # Get checkpoint state
        state = adapter.get_checkpoint_state()
        assert state["stage"] == "PLANNING"
        assert state["tool_calls_used"] == 1
        assert len(state["modified_files"]) == 2

        # Apply checkpoint state
        new_state = {
            "stage": "EXECUTING",
            "tool_history": ["read", "write"],
            "observed_files": ["file1.py"],
            "tool_calls_used": 3,
        }
        adapter.apply_checkpoint_state(new_state)

        # Verify set_state was called
        mock_state_coordinator.set_state.assert_called_once()
        call_args = mock_state_coordinator.set_state.call_args
        assert call_args[0][0]["checkpoint"]["stage"] == "EXECUTING"

    def test_properties(self):
        """Test adapter properties."""
        mock_state_coordinator = Mock()
        mock_eval_coordinator = Mock()
        mock_conversation_controller = Mock()

        adapter = CoordinatorAdapter(
            state_coordinator=mock_state_coordinator,
            evaluation_coordinator=mock_eval_coordinator,
            conversation_controller=mock_conversation_controller,
        )

        assert adapter.state_coordinator is mock_state_coordinator
        assert adapter.evaluation_coordinator is mock_eval_coordinator
        assert adapter.conversation_controller is mock_conversation_controller
        assert adapter.is_enabled is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
