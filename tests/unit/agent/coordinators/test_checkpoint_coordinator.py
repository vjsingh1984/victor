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

"""Tests for CheckpointCoordinator.

This test file verifies the checkpoint coordinator functionality which handles
time-travel debugging operations including:

- Manual checkpoint saving and restoration
- Automatic checkpointing based on tool execution intervals
- State serialization and deserialization
- Session management

Migration Pattern from orchestrator tests:
1. Extract checkpoint-related test logic
2. Mock checkpoint manager dependencies
3. Test coordinator in isolation
4. Verify state serialization/deserialization
5. Test error handling for checkpoint failures
"""

import pytest
from unittest.mock import AsyncMock, Mock
from typing import Any

from victor.agent.coordinators.checkpoint_coordinator import CheckpointCoordinator


class TestCheckpointCoordinator:
    """Test suite for CheckpointCoordinator.

    This coordinator handles checkpoint operations for time-travel debugging,
    extracted from AgentOrchestrator as part of SOLID refactoring.
    """

    @pytest.fixture
    def mock_checkpoint_manager(self) -> Mock:
        """Create mock checkpoint manager."""
        manager = Mock()
        manager.save_checkpoint = AsyncMock()
        manager.restore_checkpoint = AsyncMock()
        manager.maybe_auto_checkpoint = AsyncMock()
        return manager

    @pytest.fixture
    def sample_state(self) -> dict[str, Any]:
        """Create sample conversation state for testing."""
        return {
            "stage": "EXECUTING",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "tool_history": ["read_file", "write_file"],
            "observed_files": ["file1.py", "file2.py"],
            "modified_files": ["file1.py"],
            "message_count": 2,
        }

    @pytest.fixture
    def get_state_fn(self, sample_state: dict[str, Any]) -> Mock:
        """Create mock get_state function."""
        return Mock(return_value=sample_state)

    @pytest.fixture
    def apply_state_fn(self) -> Mock:
        """Create mock apply_state function."""
        return Mock()

    @pytest.fixture
    def coordinator(
        self,
        mock_checkpoint_manager: Mock,
        get_state_fn: Mock,
        apply_state_fn: Mock,
    ) -> CheckpointCoordinator:
        """Create checkpoint coordinator with default configuration."""
        return CheckpointCoordinator(
            checkpoint_manager=mock_checkpoint_manager,
            session_id="test_session_123",
            get_state_fn=get_state_fn,
            apply_state_fn=apply_state_fn,
        )

    @pytest.fixture
    def coordinator_no_manager(
        self,
        get_state_fn: Mock,
        apply_state_fn: Mock,
    ) -> CheckpointCoordinator:
        """Create checkpoint coordinator without manager (disabled)."""
        return CheckpointCoordinator(
            checkpoint_manager=None,
            session_id="test_session_123",
            get_state_fn=get_state_fn,
            apply_state_fn=apply_state_fn,
        )

    # Test properties

    def test_checkpoint_manager_property(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test that checkpoint_manager property returns the manager."""
        # Execute
        manager = coordinator.checkpoint_manager

        # Assert
        assert manager == mock_checkpoint_manager

    def test_checkpoint_manager_property_when_disabled(
        self, coordinator_no_manager: CheckpointCoordinator
    ):
        """Test that checkpoint_manager returns None when disabled."""
        # Execute
        manager = coordinator_no_manager.checkpoint_manager

        # Assert
        assert manager is None

    def test_is_enabled_when_manager_exists(self, coordinator: CheckpointCoordinator):
        """Test that is_enabled returns True when manager is set."""
        # Execute
        is_enabled = coordinator.is_enabled

        # Assert
        assert is_enabled is True

    def test_is_enabled_when_manager_is_none(self, coordinator_no_manager: CheckpointCoordinator):
        """Test that is_enabled returns False when manager is None."""
        # Execute
        is_enabled = coordinator_no_manager.is_enabled

        # Assert
        assert is_enabled is False

    # Test save_checkpoint

    @pytest.mark.asyncio
    async def test_save_checkpoint_with_description(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test saving checkpoint with description."""
        # Setup
        mock_checkpoint_manager.save_checkpoint.return_value = "cp_abc123"

        # Execute
        checkpoint_id = await coordinator.save_checkpoint(description="Before refactoring")

        # Assert
        assert checkpoint_id == "cp_abc123"
        mock_checkpoint_manager.save_checkpoint.assert_called_once_with(
            session_id="test_session_123",
            state=coordinator._get_state_fn(),
            description="Before refactoring",
            tags=[],
        )

    @pytest.mark.asyncio
    async def test_save_checkpoint_with_tags(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test saving checkpoint with tags."""
        # Setup
        mock_checkpoint_manager.save_checkpoint.return_value = "cp_xyz789"

        # Execute
        checkpoint_id = await coordinator.save_checkpoint(
            description="Important state", tags=["important", "before-change"]
        )

        # Assert
        assert checkpoint_id == "cp_xyz789"
        mock_checkpoint_manager.save_checkpoint.assert_called_once()
        call_args = mock_checkpoint_manager.save_checkpoint.call_args
        assert call_args.kwargs["tags"] == ["important", "before-change"]

    @pytest.mark.asyncio
    async def test_save_checkpoint_without_description(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test saving checkpoint without description."""
        # Setup
        mock_checkpoint_manager.save_checkpoint.return_value = "cp_def456"

        # Execute
        checkpoint_id = await coordinator.save_checkpoint()

        # Assert
        assert checkpoint_id == "cp_def456"
        mock_checkpoint_manager.save_checkpoint.assert_called_once()
        call_args = mock_checkpoint_manager.save_checkpoint.call_args
        assert call_args.kwargs["description"] is None

    @pytest.mark.asyncio
    async def test_save_checkpoint_when_disabled(
        self, coordinator_no_manager: CheckpointCoordinator
    ):
        """Test that save_checkpoint returns None when checkpointing is disabled."""
        # Execute
        checkpoint_id = await coordinator_no_manager.save_checkpoint(description="Should not save")

        # Assert
        assert checkpoint_id is None

    @pytest.mark.asyncio
    async def test_save_checkpoint_with_serialization_error(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test that save_checkpoint handles TypeError gracefully."""
        # Setup
        mock_checkpoint_manager.save_checkpoint.side_effect = TypeError("Cannot serialize")

        # Execute
        checkpoint_id = await coordinator.save_checkpoint(description="Bad state")

        # Assert
        assert checkpoint_id is None

    @pytest.mark.asyncio
    async def test_save_checkpoint_with_value_error(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test that save_checkpoint handles ValueError gracefully."""
        # Setup
        mock_checkpoint_manager.save_checkpoint.side_effect = ValueError("Invalid value")

        # Execute
        checkpoint_id = await coordinator.save_checkpoint(description="Invalid state")

        # Assert
        assert checkpoint_id is None

    @pytest.mark.asyncio
    async def test_save_checkpoint_uses_session_id(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test that save_checkpoint uses the configured session ID."""
        # Setup
        mock_checkpoint_manager.save_checkpoint.return_value = "cp_session123"

        # Execute
        checkpoint_id = await coordinator.save_checkpoint()

        # Assert
        assert checkpoint_id == "cp_session123"
        mock_checkpoint_manager.save_checkpoint.assert_called_once()
        call_args = mock_checkpoint_manager.save_checkpoint.call_args
        assert call_args.kwargs["session_id"] == "test_session_123"

    @pytest.mark.asyncio
    async def test_save_checkpoint_with_empty_tags(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test that save_checkpoint uses empty list when tags is None."""
        # Setup
        mock_checkpoint_manager.save_checkpoint.return_value = "cp_empty_tags"

        # Execute
        checkpoint_id = await coordinator.save_checkpoint(tags=None)

        # Assert
        assert checkpoint_id == "cp_empty_tags"
        mock_checkpoint_manager.save_checkpoint.assert_called_once()
        call_args = mock_checkpoint_manager.save_checkpoint.call_args
        assert call_args.kwargs["tags"] == []

    # Test restore_checkpoint

    @pytest.mark.asyncio
    async def test_restore_checkpoint_success(
        self,
        coordinator: CheckpointCoordinator,
        mock_checkpoint_manager: Mock,
        sample_state: dict[str, Any],
        apply_state_fn: Mock,
    ):
        """Test successful checkpoint restoration."""
        # Setup
        mock_checkpoint_manager.restore_checkpoint.return_value = sample_state

        # Execute
        success = await coordinator.restore_checkpoint("cp_abc123")

        # Assert
        assert success is True
        mock_checkpoint_manager.restore_checkpoint.assert_called_once_with("cp_abc123")
        apply_state_fn.assert_called_once_with(sample_state)

    @pytest.mark.asyncio
    async def test_restore_checkpoint_when_disabled(
        self, coordinator_no_manager: CheckpointCoordinator, apply_state_fn: Mock
    ):
        """Test that restore_checkpoint returns False when checkpointing is disabled."""
        # Execute
        success = await coordinator_no_manager.restore_checkpoint("cp_xyz789")

        # Assert
        assert success is False
        apply_state_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_restore_checkpoint_with_os_error(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test that restore_checkpoint handles OSError gracefully."""
        # Setup
        mock_checkpoint_manager.restore_checkpoint.side_effect = OSError("File not found")

        # Execute
        success = await coordinator.restore_checkpoint("cp_missing")

        # Assert
        assert success is False

    @pytest.mark.asyncio
    async def test_restore_checkpoint_with_io_error(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test that restore_checkpoint handles IOError gracefully."""
        # Setup
        mock_checkpoint_manager.restore_checkpoint.side_effect = IOError("Read error")

        # Execute
        success = await coordinator.restore_checkpoint("cp_io_error")

        # Assert
        assert success is False

    @pytest.mark.asyncio
    async def test_restore_checkpoint_with_key_error(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test that restore_checkpoint handles KeyError gracefully."""
        # Setup
        mock_checkpoint_manager.restore_checkpoint.side_effect = KeyError("Invalid key")

        # Execute
        success = await coordinator.restore_checkpoint("cp_invalid")

        # Assert
        assert success is False

    @pytest.mark.asyncio
    async def test_restore_checkpoint_with_value_error(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test that restore_checkpoint handles ValueError gracefully."""
        # Setup
        mock_checkpoint_manager.restore_checkpoint.side_effect = ValueError("Invalid data format")

        # Execute
        success = await coordinator.restore_checkpoint("cp_bad_format")

        # Assert
        assert success is False

    @pytest.mark.asyncio
    async def test_restore_checkpoint_applies_state_correctly(
        self,
        coordinator: CheckpointCoordinator,
        mock_checkpoint_manager: Mock,
        apply_state_fn: Mock,
    ):
        """Test that restore_checkpoint applies the restored state correctly."""
        # Setup
        restored_state = {
            "stage": "ANALYZING",
            "messages": [{"role": "user", "content": "Restored"}],
            "tool_history": ["read_file"],
        }
        mock_checkpoint_manager.restore_checkpoint.return_value = restored_state

        # Execute
        success = await coordinator.restore_checkpoint("cp_restore")

        # Assert
        assert success is True
        apply_state_fn.assert_called_once_with(restored_state)

    # Test maybe_auto_checkpoint

    @pytest.mark.asyncio
    async def test_maybe_auto_checkpoint_creates_checkpoint(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test that maybe_auto_checkpoint creates a checkpoint when conditions are met."""
        # Setup
        mock_checkpoint_manager.maybe_auto_checkpoint.return_value = "cp_auto_123"

        # Execute
        checkpoint_id = await coordinator.maybe_auto_checkpoint()

        # Assert
        assert checkpoint_id == "cp_auto_123"
        mock_checkpoint_manager.maybe_auto_checkpoint.assert_called_once_with(
            session_id="test_session_123",
            state=coordinator._get_state_fn(),
        )

    @pytest.mark.asyncio
    async def test_maybe_auto_checkpoint_returns_none_when_not_needed(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test that maybe_auto_checkpoint returns None when conditions aren't met."""
        # Setup
        mock_checkpoint_manager.maybe_auto_checkpoint.return_value = None

        # Execute
        checkpoint_id = await coordinator.maybe_auto_checkpoint()

        # Assert
        assert checkpoint_id is None

    @pytest.mark.asyncio
    async def test_maybe_auto_checkpoint_when_disabled(
        self, coordinator_no_manager: CheckpointCoordinator
    ):
        """Test that maybe_auto_checkpoint returns None when checkpointing is disabled."""
        # Execute
        checkpoint_id = await coordinator_no_manager.maybe_auto_checkpoint()

        # Assert
        assert checkpoint_id is None

    @pytest.mark.asyncio
    async def test_maybe_auto_checkpoint_with_serialization_error(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test that maybe_auto_checkpoint handles TypeError gracefully."""
        # Setup
        mock_checkpoint_manager.maybe_auto_checkpoint.side_effect = TypeError("Cannot serialize")

        # Execute
        checkpoint_id = await coordinator.maybe_auto_checkpoint()

        # Assert
        assert checkpoint_id is None

    @pytest.mark.asyncio
    async def test_maybe_auto_checkpoint_with_value_error(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test that maybe_auto_checkpoint handles ValueError gracefully."""
        # Setup
        mock_checkpoint_manager.maybe_auto_checkpoint.side_effect = ValueError("Invalid value")

        # Execute
        checkpoint_id = await coordinator.maybe_auto_checkpoint()

        # Assert
        assert checkpoint_id is None

    @pytest.mark.asyncio
    async def test_maybe_auto_checkpoint_uses_session_id(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test that maybe_auto_checkpoint uses the configured session ID."""
        # Setup
        mock_checkpoint_manager.maybe_auto_checkpoint.return_value = "cp_auto_session"

        # Execute
        checkpoint_id = await coordinator.maybe_auto_checkpoint()

        # Assert
        assert checkpoint_id == "cp_auto_session"
        mock_checkpoint_manager.maybe_auto_checkpoint.assert_called_once()
        call_args = mock_checkpoint_manager.maybe_auto_checkpoint.call_args
        assert call_args.kwargs["session_id"] == "test_session_123"

    # Test update_session_id

    def test_update_session_id(self, coordinator: CheckpointCoordinator):
        """Test updating the session ID."""
        # Execute
        coordinator.update_session_id("new_session_456")

        # Assert
        assert coordinator._session_id == "new_session_456"

    def test_update_session_id_to_none(self, coordinator: CheckpointCoordinator):
        """Test updating session ID to None."""
        # Execute
        coordinator.update_session_id(None)

        # Assert
        assert coordinator._session_id is None

    def test_update_session_id_affects_subsequent_saves(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test that updating session ID affects subsequent save operations."""
        # Setup
        mock_checkpoint_manager.save_checkpoint = AsyncMock(return_value="cp_test")

        # Update session ID
        coordinator.update_session_id("updated_session")

        # Execute
        import asyncio

        asyncio.run(coordinator.save_checkpoint(description="After update"))

        # Assert
        mock_checkpoint_manager.save_checkpoint.assert_called_once()
        call_args = mock_checkpoint_manager.save_checkpoint.call_args
        assert call_args.kwargs["session_id"] == "updated_session"

    # Test integration scenarios

    @pytest.mark.asyncio
    async def test_checkpoint_workflow_save_and_restore(
        self,
        coordinator: CheckpointCoordinator,
        mock_checkpoint_manager: Mock,
        sample_state: dict[str, Any],
        apply_state_fn: Mock,
    ):
        """Test complete workflow: save checkpoint, then restore it."""
        # Setup
        mock_checkpoint_manager.save_checkpoint.return_value = "cp_workflow_123"
        mock_checkpoint_manager.restore_checkpoint.return_value = sample_state

        # Save checkpoint
        checkpoint_id = await coordinator.save_checkpoint(description="Workflow test")
        assert checkpoint_id == "cp_workflow_123"

        # Restore checkpoint
        success = await coordinator.restore_checkpoint(checkpoint_id)
        assert success is True

        # Verify state was applied
        apply_state_fn.assert_called_once_with(sample_state)

    @pytest.mark.asyncio
    async def test_multiple_checkpoints_same_session(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test saving multiple checkpoints in the same session."""
        # Setup
        mock_checkpoint_manager.save_checkpoint.side_effect = [
            "cp_1",
            "cp_2",
            "cp_3",
        ]

        # Execute
        cp1 = await coordinator.save_checkpoint(description="First checkpoint")
        cp2 = await coordinator.save_checkpoint(description="Second checkpoint")
        cp3 = await coordinator.save_checkpoint(description="Third checkpoint")

        # Assert
        assert cp1 == "cp_1"
        assert cp2 == "cp_2"
        assert cp3 == "cp_3"
        assert mock_checkpoint_manager.save_checkpoint.call_count == 3

    @pytest.mark.asyncio
    async def test_auto_checkpoint_after_tool_execution(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test auto-checkpoint triggering after tool execution."""
        # Setup
        mock_checkpoint_manager.maybe_auto_checkpoint.return_value = "cp_auto_tool"

        # Execute
        checkpoint_id = await coordinator.maybe_auto_checkpoint()

        # Assert
        assert checkpoint_id == "cp_auto_tool"
        mock_checkpoint_manager.maybe_auto_checkpoint.assert_called_once()


class TestCheckpointCoordinatorEdgeCases:
    """Test edge cases and error conditions for CheckpointCoordinator."""

    @pytest.fixture
    def mock_checkpoint_manager(self) -> Mock:
        """Create mock checkpoint manager."""
        manager = Mock()
        manager.save_checkpoint = AsyncMock()
        manager.restore_checkpoint = AsyncMock()
        manager.maybe_auto_checkpoint = AsyncMock()
        return manager

    @pytest.fixture
    def sample_state(self) -> dict[str, Any]:
        """Create sample conversation state for testing."""
        return {
            "stage": "EXECUTING",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "tool_history": ["read_file", "write_file"],
            "observed_files": ["file1.py", "file2.py"],
            "modified_files": ["file1.py"],
            "message_count": 2,
        }

    @pytest.fixture
    def apply_state_fn(self) -> Mock:
        """Create mock apply_state function."""
        return Mock()

    @pytest.fixture
    def coordinator(
        self,
        mock_checkpoint_manager: Mock,
        apply_state_fn: Mock,
    ) -> CheckpointCoordinator:
        """Create coordinator."""
        return CheckpointCoordinator(
            checkpoint_manager=mock_checkpoint_manager,
            session_id="test_session",
            get_state_fn=lambda: {
                "stage": "EXECUTING",
                "messages": [],
                "tool_history": [],
            },
            apply_state_fn=apply_state_fn,
        )

    @pytest.fixture
    def get_state_fn_raises(self) -> Mock:
        """Create get_state function that raises an error."""
        return Mock(side_effect=RuntimeError("State extraction failed"))

    @pytest.fixture
    def apply_state_fn_raises(self) -> Mock:
        """Create apply_state function that raises an error."""
        return Mock(side_effect=RuntimeError("State application failed"))

    @pytest.fixture
    def coordinator_with_broken_get_state(
        self, mock_checkpoint_manager: Mock, get_state_fn_raises: Mock
    ) -> CheckpointCoordinator:
        """Create coordinator with broken get_state function."""
        return CheckpointCoordinator(
            checkpoint_manager=mock_checkpoint_manager,
            session_id="test_session",
            get_state_fn=get_state_fn_raises,
            apply_state_fn=Mock(),
        )

    @pytest.fixture
    def coordinator_with_broken_apply_state(
        self, mock_checkpoint_manager: Mock, apply_state_fn_raises: Mock
    ) -> CheckpointCoordinator:
        """Create coordinator with broken apply_state function."""
        return CheckpointCoordinator(
            checkpoint_manager=mock_checkpoint_manager,
            session_id="test_session",
            get_state_fn=lambda: {},
            apply_state_fn=apply_state_fn_raises,
        )

    @pytest.mark.asyncio
    async def test_save_checkpoint_propagates_get_state_error(
        self, coordinator_with_broken_get_state: CheckpointCoordinator
    ):
        """Test that save_checkpoint propagates errors from get_state_fn."""
        # Execute & Assert - should propagate the error
        with pytest.raises(RuntimeError, match="State extraction failed"):
            await coordinator_with_broken_get_state.save_checkpoint()

    @pytest.mark.asyncio
    async def test_restore_checkpoint_propagates_apply_state_error(
        self,
        coordinator_with_broken_apply_state: CheckpointCoordinator,
        mock_checkpoint_manager: Mock,
    ):
        """Test that restore_checkpoint propagates errors from apply_state_fn."""
        # Setup
        mock_checkpoint_manager.restore_checkpoint.return_value = {"state": "data"}

        # Execute & Assert - should propagate the error
        with pytest.raises(RuntimeError, match="State application failed"):
            await coordinator_with_broken_apply_state.restore_checkpoint("cp_test")

    @pytest.mark.asyncio
    async def test_save_checkpoint_with_complex_state(self, mock_checkpoint_manager: Mock):
        """Test saving checkpoint with complex nested state."""
        # Setup
        complex_state = {
            "stage": "EXECUTING",
            "messages": [
                {
                    "role": "user",
                    "content": "Complex",
                    "metadata": {"timestamp": 123456, "nested": {"key": "value"}},
                }
            ],
            "tool_history": ["tool1", "tool2", "tool3"],
            "observed_files": ["file1.py", "file2.py", "file3.py"],
            "modified_files": ["file1.py"],
            "nested_data": {"level1": {"level2": {"level3": "deep value"}}},
            "list_data": [1, 2, 3, 4, 5],
            "message_count": 42,
        }
        get_state_fn = Mock(return_value=complex_state)
        mock_checkpoint_manager.save_checkpoint.return_value = "cp_complex"

        coordinator = CheckpointCoordinator(
            checkpoint_manager=mock_checkpoint_manager,
            session_id="complex_session",
            get_state_fn=get_state_fn,
            apply_state_fn=Mock(),
        )

        # Execute
        checkpoint_id = await coordinator.save_checkpoint(description="Complex state")

        # Assert
        assert checkpoint_id == "cp_complex"
        mock_checkpoint_manager.save_checkpoint.assert_called_once()
        call_args = mock_checkpoint_manager.save_checkpoint.call_args
        assert call_args.kwargs["state"] == complex_state

    @pytest.mark.asyncio
    async def test_save_checkpoint_with_special_characters_in_description(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test saving checkpoint with special characters in description."""
        # Setup
        mock_checkpoint_manager.save_checkpoint.return_value = "cp_special"
        description = "Checkpoint with emoji 🎉 and unicode 你好 and newlines\n"

        # Execute
        checkpoint_id = await coordinator.save_checkpoint(description=description)

        # Assert
        assert checkpoint_id == "cp_special"
        mock_checkpoint_manager.save_checkpoint.assert_called_once()
        call_args = mock_checkpoint_manager.save_checkpoint.call_args
        assert call_args.kwargs["description"] == description

    @pytest.mark.asyncio
    async def test_restore_checkpoint_with_empty_checkpoint_id(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test restoring with empty checkpoint ID."""
        # Setup
        mock_checkpoint_manager.restore_checkpoint.side_effect = KeyError("Empty ID")

        # Execute
        success = await coordinator.restore_checkpoint("")

        # Assert
        assert success is False

    @pytest.mark.asyncio
    async def test_restore_checkpoint_with_special_characters_in_id(
        self,
        coordinator: CheckpointCoordinator,
        mock_checkpoint_manager: Mock,
        sample_state: dict[str, Any],
    ):
        """Test restoring with special characters in checkpoint ID."""
        # Setup
        special_id = "cp_test-123_abc.456"
        mock_checkpoint_manager.restore_checkpoint.return_value = sample_state

        # Execute
        success = await coordinator.restore_checkpoint(special_id)

        # Assert
        assert success is True
        mock_checkpoint_manager.restore_checkpoint.assert_called_once_with(special_id)

    @pytest.mark.asyncio
    async def test_rapid_auto_checkpoint_calls(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test rapid consecutive auto_checkpoint calls."""
        # Setup
        mock_checkpoint_manager.maybe_auto_checkpoint.return_value = None

        # Execute - multiple rapid calls
        for _ in range(10):
            await coordinator.maybe_auto_checkpoint()

        # Assert
        assert mock_checkpoint_manager.maybe_auto_checkpoint.call_count == 10

    @pytest.mark.asyncio
    async def test_session_update_between_operations(
        self, coordinator: CheckpointCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test updating session ID between checkpoint operations."""
        # Setup
        mock_checkpoint_manager.save_checkpoint.side_effect = ["cp_old", "cp_new"]

        # Save with old session
        cp1 = await coordinator.save_checkpoint(description="Old session")
        assert cp1 == "cp_old"

        # Update session
        coordinator.update_session_id("new_session")

        # Save with new session
        cp2 = await coordinator.save_checkpoint(description="New session")
        assert cp2 == "cp_new"

        # Verify both calls used different session IDs
        assert mock_checkpoint_manager.save_checkpoint.call_count == 2
        first_call = mock_checkpoint_manager.save_checkpoint.call_args_list[0]
        second_call = mock_checkpoint_manager.save_checkpoint.call_args_list[1]
        assert first_call.kwargs["session_id"] == "test_session"
        assert second_call.kwargs["session_id"] == "new_session"

    @pytest.mark.asyncio
    async def test_multiple_restores_in_sequence(
        self,
        coordinator: CheckpointCoordinator,
        mock_checkpoint_manager: Mock,
        apply_state_fn: Mock,
    ):
        """Test multiple consecutive restore operations."""
        # Setup
        states = [
            {"stage": "INITIAL", "messages": []},
            {"stage": "EXECUTING", "messages": [{"role": "user"}]},
            {"stage": "COMPLETE", "messages": [{"role": "user"}, {"role": "assistant"}]},
        ]
        mock_checkpoint_manager.restore_checkpoint.side_effect = states

        # Execute
        success1 = await coordinator.restore_checkpoint("cp_1")
        success2 = await coordinator.restore_checkpoint("cp_2")
        success3 = await coordinator.restore_checkpoint("cp_3")

        # Assert
        assert success1 is True
        assert success2 is True
        assert success3 is True
        assert apply_state_fn.call_count == 3

        # Verify states were applied in order
        assert apply_state_fn.call_args_list[0][0][0] == states[0]
        assert apply_state_fn.call_args_list[1][0][0] == states[1]
        assert apply_state_fn.call_args_list[2][0][0] == states[2]

    @pytest.mark.asyncio
    async def test_save_and_restore_cycle(
        self,
        coordinator: CheckpointCoordinator,
        mock_checkpoint_manager: Mock,
        sample_state: dict[str, Any],
        apply_state_fn: Mock,
    ):
        """Test a complete save-restore-save cycle."""
        # Setup
        mock_checkpoint_manager.save_checkpoint.side_effect = ["cp_1", "cp_2"]
        mock_checkpoint_manager.restore_checkpoint.return_value = sample_state

        # Save initial checkpoint
        cp1 = await coordinator.save_checkpoint(description="Initial")
        assert cp1 == "cp_1"

        # Restore from checkpoint
        success = await coordinator.restore_checkpoint(cp1)
        assert success is True

        # Save new checkpoint after restoration
        cp2 = await coordinator.save_checkpoint(description="After restore")
        assert cp2 == "cp_2"

        # Verify operations
        assert mock_checkpoint_manager.save_checkpoint.call_count == 2
        assert mock_checkpoint_manager.restore_checkpoint.call_count == 1
        assert apply_state_fn.call_count == 1
