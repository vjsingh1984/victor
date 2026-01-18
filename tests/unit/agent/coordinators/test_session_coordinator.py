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

"""Tests for SessionCoordinator.

This test file provides comprehensive coverage for SessionCoordinator, which manages
session lifecycle and state coordination as part of Track 4 refactoring.

Migration Pattern:
1. SessionCoordinator extracted from AgentOrchestrator
2. Tests focus on session lifecycle management
3. Mock dependencies (SessionStateManager, LifecycleManager, etc.)
4. Test coordinator in isolation
5. Verify state tracking and lifecycle operations
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Any, Dict, List

from victor.agent.coordinators.session_coordinator import (
    SessionCoordinator,
    SessionInfo,
    SessionCostSummary,
    create_session_coordinator,
)


class TestSessionInfo:
    """Test SessionInfo dataclass."""

    def test_default_initialization(self):
        """Test SessionInfo with default values."""
        info = SessionInfo(session_id="test-session")

        assert info.session_id == "test-session"
        assert info.message_count == 0
        assert info.tool_calls_used == 0
        assert info.is_active is True
        assert isinstance(info.created_at, float)
        assert isinstance(info.last_activity, float)

    def test_custom_initialization(self):
        """Test SessionInfo with custom values."""
        info = SessionInfo(
            session_id="custom-session",
            message_count=10,
            tool_calls_used=5,
            is_active=False,
        )

        assert info.session_id == "custom-session"
        assert info.message_count == 10
        assert info.tool_calls_used == 5
        assert info.is_active is False

    def test_to_dict(self):
        """Test SessionInfo serialization."""
        info = SessionInfo(
            session_id="test-session",
            created_at=1234567890.0,
            last_activity=1234567895.0,
            message_count=15,
            tool_calls_used=7,
            is_active=True,
        )

        data = info.to_dict()

        assert data["session_id"] == "test-session"
        assert data["created_at"] == 1234567890.0
        assert data["last_activity"] == 1234567895.0
        assert data["message_count"] == 15
        assert data["tool_calls_used"] == 7
        assert data["is_active"] is True

    def test_from_dict(self):
        """Test SessionInfo deserialization."""
        data = {
            "session_id": "loaded-session",
            "created_at": 1234567890.0,
            "last_activity": 1234567900.0,
            "message_count": 20,
            "tool_calls_used": 10,
            "is_active": False,
        }

        info = SessionInfo.from_dict(data)

        assert info.session_id == "loaded-session"
        assert info.created_at == 1234567890.0
        assert info.last_activity == 1234567900.0
        assert info.message_count == 20
        assert info.tool_calls_used == 10
        assert info.is_active is False

    def test_from_dict_with_defaults(self):
        """Test SessionInfo deserialization with missing fields."""
        data = {"session_id": "minimal-session"}

        info = SessionInfo.from_dict(data)

        assert info.session_id == "minimal-session"
        assert info.message_count == 0
        assert info.tool_calls_used == 0
        assert info.is_active is True
        assert isinstance(info.created_at, float)


class TestSessionCostSummary:
    """Test SessionCostSummary dataclass."""

    def test_default_initialization(self):
        """Test SessionCostSummary with default values."""
        summary = SessionCostSummary()

        assert summary.total_cost == 0.0
        assert summary.input_cost == 0.0
        assert summary.output_cost == 0.0
        assert summary.total_tokens == 0
        assert summary.input_tokens == 0
        assert summary.output_tokens == 0

    def test_custom_initialization(self):
        """Test SessionCostSummary with custom values."""
        summary = SessionCostSummary(
            total_cost=0.1234,
            input_cost=0.05,
            output_cost=0.0734,
            total_tokens=1000,
            input_tokens=600,
            output_tokens=400,
        )

        assert summary.total_cost == 0.1234
        assert summary.input_cost == 0.05
        assert summary.output_cost == 0.0734
        assert summary.total_tokens == 1000
        assert summary.input_tokens == 600
        assert summary.output_tokens == 400

    def test_to_dict(self):
        """Test SessionCostSummary serialization."""
        summary = SessionCostSummary(
            total_cost=0.0256,
            input_cost=0.01,
            output_cost=0.0156,
            total_tokens=500,
            input_tokens=300,
            output_tokens=200,
        )

        data = summary.to_dict()

        assert data["total_cost"] == 0.0256
        assert data["input_cost"] == 0.01
        assert data["output_cost"] == 0.0156
        assert data["total_tokens"] == 500
        assert data["input_tokens"] == 300
        assert data["output_tokens"] == 200


class TestSessionCoordinator:
    """Test suite for SessionCoordinator.

    Tests session lifecycle management, state tracking, and cleanup operations.
    """

    @pytest.fixture
    def mock_session_state(self) -> Mock:
        """Create mock session state manager."""
        state = Mock()
        state.tool_budget = 100
        state.tool_calls_used = 0
        state.observed_files = set()
        state.executed_tools = []
        state.execution_state = Mock()
        state.execution_state.tool_calls_used = 0
        state.execution_state.observed_files = set()
        state.execution_state.executed_tools = []
        state.execution_state.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        state._tool_budget = 100

        # Configure methods
        state.reset = Mock()
        state.get_remaining_budget = Mock(return_value=100)
        state.is_budget_exhausted = Mock(return_value=False)
        state.get_token_usage = Mock(
            return_value={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        )
        state.get_session_summary = Mock(return_value={"summary": "test"})
        state.update_token_usage = Mock()
        state.reset_token_usage = Mock()
        return state

    @pytest.fixture
    def mock_lifecycle_manager(self) -> Mock:
        """Create mock lifecycle manager."""
        manager = Mock()
        manager.reset_conversation = Mock()
        manager.recover_session = Mock(return_value=True)
        return manager

    @pytest.fixture
    def mock_memory_manager(self) -> Mock:
        """Create mock memory manager."""
        memory = Mock()
        memory.create_session = Mock(return_value="mem-session-123")
        memory.end_session = Mock()
        memory.list_sessions = Mock(return_value=[])
        memory.get_session_stats = Mock(return_value={})
        memory.get_context_messages = Mock(return_value=[])
        memory._project_path = "/test/project"
        return memory

    @pytest.fixture
    def mock_checkpoint_manager(self) -> Mock:
        """Create mock checkpoint manager."""
        manager = AsyncMock()
        manager.save_checkpoint = AsyncMock(return_value="checkpoint-123")
        manager.restore_checkpoint = AsyncMock(return_value={"state": "restored"})
        manager.maybe_auto_checkpoint = AsyncMock(return_value=None)
        return manager

    @pytest.fixture
    def mock_cost_tracker(self) -> Mock:
        """Create mock cost tracker."""
        tracker = Mock()
        tracker.get_summary = Mock(return_value={"total_cost": 0.01})
        tracker.format_inline_cost = Mock(return_value="$0.01")
        return tracker

    @pytest.fixture
    def coordinator(
        self, mock_session_state: Mock, mock_lifecycle_manager: Mock
    ) -> SessionCoordinator:
        """Create session coordinator with default mocks."""
        return SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
            memory_manager=None,
            checkpoint_manager=None,
            cost_tracker=None,
        )

    @pytest.fixture
    def coordinator_full(
        self,
        mock_session_state: Mock,
        mock_lifecycle_manager: Mock,
        mock_memory_manager: Mock,
        mock_checkpoint_manager: Mock,
        mock_cost_tracker: Mock,
    ) -> SessionCoordinator:
        """Create coordinator with all dependencies."""
        return SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
            memory_manager=mock_memory_manager,
            checkpoint_manager=mock_checkpoint_manager,
            cost_tracker=mock_cost_tracker,
        )

    # ========================================================================
    # Initialization Tests
    # ========================================================================

    def test_initialization_with_required_dependencies(self, mock_session_state: Mock):
        """Test coordinator initialization with required dependencies."""
        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=None,
            memory_manager=None,
        )

        assert coordinator._session_state == mock_session_state
        assert coordinator._lifecycle_manager is None
        assert coordinator._memory_manager is None
        assert coordinator._checkpoint_manager is None
        assert coordinator._cost_tracker is None
        assert coordinator._current_session is None
        assert coordinator._memory_session_id is None

    def test_initialization_with_all_dependencies(
        self,
        mock_session_state: Mock,
        mock_lifecycle_manager: Mock,
        mock_memory_manager: Mock,
        mock_checkpoint_manager: Mock,
        mock_cost_tracker: Mock,
    ):
        """Test coordinator initialization with all dependencies."""
        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
            memory_manager=mock_memory_manager,
            checkpoint_manager=mock_checkpoint_manager,
            cost_tracker=mock_cost_tracker,
        )

        assert coordinator._session_state == mock_session_state
        assert coordinator._lifecycle_manager == mock_lifecycle_manager
        assert coordinator._memory_manager == mock_memory_manager
        assert coordinator._checkpoint_manager == mock_checkpoint_manager
        assert coordinator._cost_tracker == mock_cost_tracker

    # ========================================================================
    # Session Lifecycle Tests
    # ========================================================================

    def test_create_session_generates_session_id(self, coordinator: SessionCoordinator):
        """Test create_session generates a session ID when not provided."""
        session_id = coordinator.create_session()

        assert session_id is not None
        assert session_id.startswith("session-")
        assert len(session_id) > len("session-")
        assert coordinator._current_session is not None
        assert coordinator._current_session.session_id == session_id
        assert coordinator._current_session.is_active is True

    def test_create_session_with_custom_session_id(self, coordinator: SessionCoordinator):
        """Test create_session with custom session ID."""
        custom_id = "my-custom-session"
        session_id = coordinator.create_session(session_id=custom_id)

        assert session_id == custom_id
        assert coordinator._current_session.session_id == custom_id

    def test_create_session_resets_session_state(
        self, coordinator: SessionCoordinator, mock_session_state: Mock
    ):
        """Test create_session resets session state."""
        coordinator.create_session()

        mock_session_state.reset.assert_called_once_with()

    def test_create_session_with_memory_manager(
        self, coordinator_full: SessionCoordinator, mock_memory_manager: Mock
    ):
        """Test create_session initializes memory session."""
        session_id = coordinator_full.create_session()

        assert coordinator_full._memory_session_id == "mem-session-123"
        mock_memory_manager.create_session.assert_called_once()

    def test_create_session_memory_manager_failure(
        self,
        mock_session_state: Mock,
        mock_lifecycle_manager: Mock,
        mock_memory_manager: Mock,
    ):
        """Test create_session handles memory manager failure gracefully."""
        mock_memory_manager.create_session.side_effect = Exception("Memory error")

        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
            memory_manager=mock_memory_manager,
        )
        session_id = coordinator.create_session()

        # Session should still be created
        assert session_id is not None
        assert coordinator._memory_session_id is None

    def test_end_session_marks_inactive(self, coordinator: SessionCoordinator):
        """Test end_session marks current session as inactive."""
        coordinator.create_session()
        assert coordinator.is_active is True

        coordinator.end_session()

        assert coordinator.is_active is False

    def test_end_session_with_memory_manager(
        self, coordinator_full: SessionCoordinator, mock_memory_manager: Mock
    ):
        """Test end_session ends memory session when active."""
        coordinator_full.create_session()
        coordinator_full.end_session()

        mock_memory_manager.end_session.assert_called_once_with("mem-session-123")

    def test_end_session_without_current_session(self, coordinator: SessionCoordinator):
        """Test end_session when no current session exists."""
        # Should not raise
        coordinator.end_session()
        assert coordinator.is_active is False

    def test_end_session_memory_manager_failure(
        self,
        mock_session_state: Mock,
        mock_lifecycle_manager: Mock,
        mock_memory_manager: Mock,
    ):
        """Test end_session handles memory manager failure gracefully."""
        mock_memory_manager.end_session.side_effect = Exception("End failed")

        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
            memory_manager=mock_memory_manager,
        )
        coordinator.create_session()
        coordinator.end_session()

        # Session should still be marked inactive
        assert coordinator.is_active is False

    def test_reset_session(
        self,
        coordinator: SessionCoordinator,
        mock_session_state: Mock,
        mock_lifecycle_manager: Mock,
    ):
        """Test reset_session resets state and conversation."""
        coordinator.create_session()
        mock_session_state.reset.reset_mock()  # Reset mock after create_session
        mock_lifecycle_manager.reset_conversation.reset_mock()

        coordinator.reset_session()

        mock_session_state.reset.assert_called_once_with(preserve_token_usage=False)
        mock_lifecycle_manager.reset_conversation.assert_called_once()

    def test_reset_session_preserve_token_usage(
        self,
        coordinator: SessionCoordinator,
        mock_session_state: Mock,
        mock_lifecycle_manager: Mock,
    ):
        """Test reset_session with preserve_token_usage=True."""
        coordinator.create_session()
        mock_session_state.reset.reset_mock()  # Reset mock after create_session
        mock_lifecycle_manager.reset_conversation.reset_mock()

        coordinator.reset_session(preserve_token_usage=True)

        mock_session_state.reset.assert_called_once_with(preserve_token_usage=True)

    def test_reset_session_without_lifecycle_manager(self, mock_session_state: Mock):
        """Test reset_session when lifecycle manager is None."""
        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=None,
        )
        coordinator.create_session()
        mock_session_state.reset.reset_mock()  # Reset mock after create_session

        coordinator.reset_session()

        mock_session_state.reset.assert_called_once_with(preserve_token_usage=False)

    def test_reset_session_updates_activity_timestamp(self, coordinator: SessionCoordinator):
        """Test reset_session updates last_activity timestamp."""
        coordinator.create_session()
        original_activity = coordinator._current_session.last_activity

        import time

        time.sleep(0.01)  # Small delay to ensure timestamp difference

        coordinator.reset_session()

        assert coordinator._current_session.last_activity > original_activity

    def test_recover_session_success(
        self, coordinator_full: SessionCoordinator, mock_lifecycle_manager: Mock
    ):
        """Test successful session recovery."""
        success = coordinator_full.recover_session("recovery-session-123")

        assert success is True
        assert coordinator_full._memory_session_id == "recovery-session-123"
        assert coordinator_full._current_session is not None
        assert coordinator_full._current_session.session_id == "recovery-session-123"
        mock_lifecycle_manager.recover_session.assert_called_once()

    def test_recover_session_without_memory_manager(self, coordinator: SessionCoordinator):
        """Test recover_session when memory manager is not available."""
        success = coordinator.recover_session("some-session")

        assert success is False

    def test_recover_session_failure(
        self,
        mock_session_state: Mock,
        mock_lifecycle_manager: Mock,
        mock_memory_manager: Mock,
    ):
        """Test recover_session when recovery fails."""
        mock_lifecycle_manager.recover_session.return_value = False

        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
            memory_manager=mock_memory_manager,
        )
        success = coordinator.recover_session("bad-session")

        assert success is False
        assert coordinator._current_session is None

    # ========================================================================
    # Session State Access Tests
    # ========================================================================

    def test_session_id_property(self, coordinator: SessionCoordinator):
        """Test session_id property."""
        assert coordinator.session_id is None

        coordinator.create_session()
        assert coordinator.session_id is not None
        assert coordinator.session_id.startswith("session-")

    def test_memory_session_id_property(self, coordinator_full: SessionCoordinator):
        """Test memory_session_id property."""
        assert coordinator_full.memory_session_id is None

        coordinator_full.create_session()
        assert coordinator_full.memory_session_id == "mem-session-123"

    def test_session_state_property(self, coordinator: SessionCoordinator):
        """Test session_state property returns state manager."""
        assert coordinator.session_state == coordinator._session_state

    def test_is_active_property(self, coordinator: SessionCoordinator):
        """Test is_active property."""
        assert coordinator.is_active is False

        coordinator.create_session()
        assert coordinator.is_active is True

        coordinator.end_session()
        assert coordinator.is_active is False

    def test_tool_calls_used_property(
        self, coordinator: SessionCoordinator, mock_session_state: Mock
    ):
        """Test tool_calls_used property."""
        assert coordinator.tool_calls_used == 0

        mock_session_state.tool_calls_used = 15
        assert coordinator.tool_calls_used == 15

    def test_remaining_budget_property(
        self, coordinator: SessionCoordinator, mock_session_state: Mock
    ):
        """Test remaining_budget property."""
        mock_session_state.get_remaining_budget.return_value = 75
        assert coordinator.remaining_budget == 75

        mock_session_state.get_remaining_budget.assert_called_once()

    def test_is_budget_exhausted_property(
        self, coordinator: SessionCoordinator, mock_session_state: Mock
    ):
        """Test is_budget_exhausted property."""
        mock_session_state.is_budget_exhausted.return_value = False
        assert coordinator.is_budget_exhausted is False

        mock_session_state.is_budget_exhausted.return_value = True
        assert coordinator.is_budget_exhausted is True

    # ========================================================================
    # Session Statistics Tests
    # ========================================================================

    def test_get_session_info(self, coordinator: SessionCoordinator):
        """Test get_session_info returns current session."""
        assert coordinator.get_session_info() is None

        coordinator.create_session()
        info = coordinator.get_session_info()

        assert info is not None
        assert info.session_id == coordinator.session_id
        assert info.is_active is True

    def test_get_session_stats(self, coordinator: SessionCoordinator, mock_session_state: Mock):
        """Test get_session_stats returns comprehensive statistics."""
        coordinator.create_session()
        stats = coordinator.get_session_stats()

        assert stats["enabled"] is False
        assert stats["session_id"] is not None
        assert stats["memory_session_id"] is None
        assert stats["is_active"] is True
        assert stats["tool_calls_used"] == 0
        assert stats["tool_budget"] == 100
        assert stats["budget_remaining"] == 100
        assert stats["budget_exhausted"] is False
        assert "token_usage" in stats

    def test_get_session_stats_with_memory_manager(
        self, coordinator_full: SessionCoordinator, mock_memory_manager: Mock
    ):
        """Test get_session_stats includes memory manager stats."""
        mock_memory_manager.get_session_stats.return_value = {
            "messages": 10,
            "files_read": 5,
        }

        coordinator_full.create_session()
        stats = coordinator_full.get_session_stats()

        assert stats["enabled"] is True
        assert stats["messages"] == 10
        assert stats["files_read"] == 5

    def test_get_session_stats_memory_manager_failure(
        self,
        mock_session_state: Mock,
        mock_lifecycle_manager: Mock,
        mock_memory_manager: Mock,
    ):
        """Test get_session_stats handles memory manager failure."""
        mock_memory_manager.get_session_stats.side_effect = Exception("Stats error")
        mock_memory_manager.create_session = Mock(return_value="mem-123")

        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
            memory_manager=mock_memory_manager,
        )
        coordinator.create_session()
        stats = coordinator.get_session_stats()

        # Should return base stats without memory stats
        assert "enabled" in stats
        assert "session_id" in stats

    def test_get_session_summary(self, coordinator: SessionCoordinator, mock_session_state: Mock):
        """Test get_session_summary includes session IDs."""
        coordinator.create_session()
        summary = coordinator.get_session_summary()

        mock_session_state.get_session_summary.assert_called_once()
        assert summary["session_id"] == coordinator.session_id
        assert summary["memory_session_id"] is None

    def test_get_session_cost_summary(self, coordinator: SessionCoordinator):
        """Test get_session_cost_summary without cost tracker."""
        summary = coordinator.get_session_cost_summary()

        assert summary == {}

    def test_get_session_cost_summary_with_tracker(
        self, coordinator_full: SessionCoordinator, mock_cost_tracker: Mock
    ):
        """Test get_session_cost_summary with cost tracker."""
        summary = coordinator_full.get_session_cost_summary()

        mock_cost_tracker.get_summary.assert_called_once()
        assert summary == {"total_cost": 0.01}

    def test_get_session_cost_formatted(self, coordinator: SessionCoordinator):
        """Test get_session_cost_formatted without cost tracker."""
        cost_str = coordinator.get_session_cost_formatted()

        assert cost_str == "cost n/a"

    def test_get_session_cost_formatted_with_tracker(
        self, coordinator_full: SessionCoordinator, mock_cost_tracker: Mock
    ):
        """Test get_session_cost_formatted with cost tracker."""
        cost_str = coordinator_full.get_session_cost_formatted()

        mock_cost_tracker.format_inline_cost.assert_called_once()
        assert cost_str == "$0.01"

    # ========================================================================
    # Checkpoint/Recovery Tests
    # ========================================================================

    @pytest.mark.asyncio
    async def test_save_checkpoint_success(
        self, coordinator_full: SessionCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test successful checkpoint save."""
        coordinator_full.create_session()
        checkpoint_id = await coordinator_full.save_checkpoint(
            description="Test checkpoint", tags=["test"]
        )

        assert checkpoint_id == "checkpoint-123"
        mock_checkpoint_manager.save_checkpoint.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_checkpoint_without_manager(self, coordinator: SessionCoordinator):
        """Test save_checkpoint when checkpoint manager is None."""
        checkpoint_id = await coordinator.save_checkpoint()

        assert checkpoint_id is None

    @pytest.mark.asyncio
    async def test_save_checkpoint_io_error(
        self,
        mock_session_state: Mock,
        mock_lifecycle_manager: Mock,
        mock_checkpoint_manager: Mock,
    ):
        """Test save_checkpoint handles I/O errors."""
        mock_checkpoint_manager.save_checkpoint.side_effect = OSError("Disk full")

        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
            checkpoint_manager=mock_checkpoint_manager,
        )
        coordinator.create_session()
        checkpoint_id = await coordinator.save_checkpoint()

        assert checkpoint_id is None

    @pytest.mark.asyncio
    async def test_save_checkpoint_serialization_error(
        self,
        mock_session_state: Mock,
        mock_lifecycle_manager: Mock,
        mock_checkpoint_manager: Mock,
    ):
        """Test save_checkpoint handles serialization errors."""
        mock_checkpoint_manager.save_checkpoint.side_effect = ValueError("Cannot serialize")

        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
            checkpoint_manager=mock_checkpoint_manager,
        )
        coordinator.create_session()
        checkpoint_id = await coordinator.save_checkpoint()

        assert checkpoint_id is None

    @pytest.mark.asyncio
    async def test_restore_checkpoint_success(
        self,
        coordinator_full: SessionCoordinator,
        mock_checkpoint_manager: Mock,
        mock_session_state: Mock,
    ):
        """Test successful checkpoint restore."""
        coordinator_full.create_session()
        success = await coordinator_full.restore_checkpoint("checkpoint-123")

        assert success is True
        mock_checkpoint_manager.restore_checkpoint.assert_called_once_with("checkpoint-123")

    @pytest.mark.asyncio
    async def test_restore_checkpoint_without_manager(self, coordinator: SessionCoordinator):
        """Test restore_checkpoint when checkpoint manager is None."""
        success = await coordinator.restore_checkpoint("checkpoint-123")

        assert success is False

    @pytest.mark.asyncio
    async def test_restore_checkpoint_io_error(
        self,
        mock_session_state: Mock,
        mock_lifecycle_manager: Mock,
        mock_checkpoint_manager: Mock,
    ):
        """Test restore_checkpoint handles I/O errors."""
        mock_checkpoint_manager.restore_checkpoint.side_effect = IOError("File not found")

        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
            checkpoint_manager=mock_checkpoint_manager,
        )
        success = await coordinator.restore_checkpoint("bad-checkpoint")

        assert success is False

    @pytest.mark.asyncio
    async def test_restore_checkpoint_invalid_data(
        self,
        mock_session_state: Mock,
        mock_lifecycle_manager: Mock,
        mock_checkpoint_manager: Mock,
    ):
        """Test restore_checkpoint handles invalid data."""
        mock_checkpoint_manager.restore_checkpoint.side_effect = KeyError("Missing key")

        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
            checkpoint_manager=mock_checkpoint_manager,
        )
        success = await coordinator.restore_checkpoint("corrupt-checkpoint")

        assert success is False

    @pytest.mark.asyncio
    async def test_maybe_auto_checkpoint(
        self, coordinator_full: SessionCoordinator, mock_checkpoint_manager: Mock
    ):
        """Test maybe_auto_checkpoint delegates to checkpoint manager."""
        coordinator_full.create_session()
        checkpoint_id = await coordinator_full.maybe_auto_checkpoint()

        mock_checkpoint_manager.maybe_auto_checkpoint.assert_called_once()
        # Return value depends on checkpoint manager's threshold logic

    @pytest.mark.asyncio
    async def test_maybe_auto_checkpoint_without_manager(self, coordinator: SessionCoordinator):
        """Test maybe_auto_checkpoint when checkpoint manager is None."""
        checkpoint_id = await coordinator.maybe_auto_checkpoint()

        assert checkpoint_id is None

    @pytest.mark.asyncio
    async def test_maybe_auto_checkpoint_error_handling(
        self,
        mock_session_state: Mock,
        mock_lifecycle_manager: Mock,
        mock_checkpoint_manager: Mock,
    ):
        """Test maybe_auto_checkpoint handles errors gracefully."""
        mock_checkpoint_manager.maybe_auto_checkpoint.side_effect = OSError("Error")

        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
            checkpoint_manager=mock_checkpoint_manager,
        )
        coordinator.create_session()
        checkpoint_id = await coordinator.maybe_auto_checkpoint()

        assert checkpoint_id is None

    def test_get_checkpoint_state(self, coordinator: SessionCoordinator):
        """Test _get_checkpoint_state builds correct state dictionary."""
        coordinator.create_session()
        state = coordinator._get_checkpoint_state()

        assert "session_id" in state
        assert "tool_calls_used" in state
        assert "tool_budget" in state
        assert "token_usage" in state
        assert "observed_files" in state
        assert "executed_tools" in state

    def test_apply_checkpoint_state(
        self, coordinator: SessionCoordinator, mock_session_state: Mock
    ):
        """Test _apply_checkpoint_state restores state correctly."""
        coordinator.create_session()

        checkpoint_state = {
            "tool_calls_used": 10,
            "tool_budget": 150,
            "token_usage": {
                "prompt_tokens": 500,
                "completion_tokens": 250,
                "total_tokens": 750,
            },
            "observed_files": ["file1.py", "file2.py"],
            "executed_tools": ["read", "edit"],
        }

        coordinator._apply_checkpoint_state(checkpoint_state)

        assert mock_session_state.execution_state.tool_calls_used == 10
        assert mock_session_state._tool_budget == 150
        assert mock_session_state.execution_state.observed_files == {
            "file1.py",
            "file2.py",
        }
        assert mock_session_state.execution_state.executed_tools == ["read", "edit"]

    # ========================================================================
    # Recent Sessions Tests
    # ========================================================================

    def test_get_recent_sessions_success(
        self, coordinator_full: SessionCoordinator, mock_memory_manager: Mock
    ):
        """Test get_recent_sessions returns session list."""
        # Create a simple object that has the right attributes
        from types import SimpleNamespace

        mock_session = SimpleNamespace(
            session_id="session-1",
            created_at=None,
            last_activity=None,
            project_path="/test",
            provider="anthropic",
            model="claude-3",
        )
        # Override the fixture's list_sessions to return our test data
        mock_memory_manager.list_sessions = Mock(return_value=[mock_session])

        sessions = coordinator_full.get_recent_sessions(limit=10)

        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "session-1"
        assert sessions[0]["created_at"] is None
        assert sessions[0]["last_activity"] is None
        mock_memory_manager.list_sessions.assert_called_once_with(limit=10)

    def test_get_recent_sessions_without_memory_manager(self, coordinator: SessionCoordinator):
        """Test get_recent_sessions when memory manager is None."""
        sessions = coordinator.get_recent_sessions()

        assert sessions == []

    def test_get_recent_sessions_with_limit(
        self, coordinator_full: SessionCoordinator, mock_memory_manager: Mock
    ):
        """Test get_recent_sessions respects limit parameter."""
        coordinator_full.get_recent_sessions(limit=5)

        mock_memory_manager.list_sessions.assert_called_once_with(limit=5)

    def test_get_recent_sessions_handles_message_count(
        self,
        mock_session_state: Mock,
        mock_lifecycle_manager: Mock,
        mock_memory_manager: Mock,
    ):
        """Test get_recent_sessions includes message count when available."""
        mock_session = Mock()
        mock_session.session_id = "session-1"
        mock_session.created_at = None
        mock_session.last_activity = None
        mock_session.project_path = "/test"
        mock_session.provider = "anthropic"
        mock_session.model = "claude-3"
        mock_session.messages = ["msg1", "msg2", "msg3"]
        mock_memory_manager.list_sessions.return_value = [mock_session]

        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
            memory_manager=mock_memory_manager,
        )
        sessions = coordinator.get_recent_sessions()

        assert sessions[0]["message_count"] == 3

    def test_get_recent_sessions_error_handling(
        self,
        mock_session_state: Mock,
        mock_lifecycle_manager: Mock,
        mock_memory_manager: Mock,
    ):
        """Test get_recent_sessions handles errors gracefully."""
        mock_memory_manager.list_sessions.side_effect = Exception("List error")

        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
            memory_manager=mock_memory_manager,
        )
        sessions = coordinator.get_recent_sessions()

        assert sessions == []

    # ========================================================================
    # Token Usage Tests
    # ========================================================================

    def test_get_token_usage(self, coordinator: SessionCoordinator, mock_session_state: Mock):
        """Test get_token_usage delegates to session state."""
        usage = coordinator.get_token_usage()

        mock_session_state.get_token_usage.assert_called_once()
        assert usage == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

    def test_update_token_usage(self, coordinator: SessionCoordinator, mock_session_state: Mock):
        """Test update_token_usage delegates to session state."""
        coordinator.update_token_usage(
            prompt_tokens=200,
            completion_tokens=100,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=5,
        )

        mock_session_state.update_token_usage.assert_called_once_with(
            prompt_tokens=200,
            completion_tokens=100,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=5,
        )

    def test_update_token_usage_with_defaults(
        self, coordinator: SessionCoordinator, mock_session_state: Mock
    ):
        """Test update_token_usage with default values."""
        coordinator.update_token_usage()

        mock_session_state.update_token_usage.assert_called_once_with(
            prompt_tokens=0,
            completion_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )

    def test_reset_token_usage(self, coordinator: SessionCoordinator, mock_session_state: Mock):
        """Test reset_token_usage delegates to session state."""
        coordinator.reset_token_usage()

        mock_session_state.reset_token_usage.assert_called_once()

    # ========================================================================
    # Memory Context Tests
    # ========================================================================

    def test_get_memory_context_with_memory_manager(
        self, coordinator_full: SessionCoordinator, mock_memory_manager: Mock
    ):
        """Test get_memory_context with memory manager available."""
        coordinator_full.create_session()
        mock_memory_manager.get_context_messages.return_value = [
            {"role": "user", "content": "Hello"}
        ]

        context = coordinator_full.get_memory_context(max_tokens=1000)

        mock_memory_manager.get_context_messages.assert_called_once_with(
            session_id="mem-session-123", max_tokens=1000
        )
        assert context == [{"role": "user", "content": "Hello"}]

    def test_get_memory_context_without_memory_manager(self, coordinator: SessionCoordinator):
        """Test get_memory_context without memory manager."""
        fallback_messages = [{"role": "system", "content": "Fallback"}]

        context = coordinator.get_memory_context(messages=fallback_messages)

        assert context == fallback_messages

    def test_get_memory_context_with_fallback(
        self,
        mock_session_state: Mock,
        mock_lifecycle_manager: Mock,
        mock_memory_manager: Mock,
    ):
        """Test get_memory_context falls back to provided messages."""
        mock_memory_manager.get_context_messages.side_effect = Exception("Error")

        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
            memory_manager=mock_memory_manager,
        )
        coordinator.create_session()

        fallback_messages = [{"role": "user", "content": "Fallback"}]
        context = coordinator.get_memory_context(messages=fallback_messages)

        assert context == fallback_messages

    def test_get_memory_context_converts_message_objects(
        self,
        mock_session_state: Mock,
        mock_lifecycle_manager: Mock,
        mock_memory_manager: Mock,
    ):
        """Test get_memory_context converts Message objects to dict."""
        mock_memory_manager.get_context_messages.side_effect = Exception("Error")

        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
            memory_manager=mock_memory_manager,
        )

        # Create mock Message object with model_dump method
        mock_msg = Mock()
        mock_msg.model_dump.return_value = {"role": "user", "content": "Converted"}

        context = coordinator.get_memory_context(messages=[mock_msg])

        assert context == [{"role": "user", "content": "Converted"}]
        mock_msg.model_dump.assert_called_once()

    def test_get_memory_context_without_session(self, coordinator_full: SessionCoordinator):
        """Test get_memory_context when memory session not created."""
        # Don't create a session
        fallback_messages = [{"role": "user", "content": "No session"}]

        context = coordinator_full.get_memory_context(messages=fallback_messages)

        assert context == fallback_messages

    def test_get_memory_context_empty_fallback(self, coordinator: SessionCoordinator):
        """Test get_memory_context returns empty list when no fallback."""
        context = coordinator.get_memory_context(messages=None)

        assert context == []

    # ========================================================================
    # String Representation Tests
    # ========================================================================

    def test_repr(self, coordinator: SessionCoordinator):
        """Test __repr__ provides useful debugging information."""
        coordinator.create_session()

        repr_str = repr(coordinator)

        assert "SessionCoordinator" in repr_str
        assert coordinator.session_id in repr_str
        assert "active=True" in repr_str
        assert "tool_calls=" in repr_str

    def test_repr_without_session(self, coordinator: SessionCoordinator):
        """Test __repr__ when no session is active."""
        repr_str = repr(coordinator)

        assert "SessionCoordinator" in repr_str
        assert "active=False" in repr_str


class TestCreateSessionCoordinatorFactory:
    """Test factory function for creating SessionCoordinator."""

    def test_factory_with_required_dependencies(self):
        """Test factory with required dependencies only."""
        mock_state = Mock()
        coordinator = create_session_coordinator(session_state_manager=mock_state)

        assert isinstance(coordinator, SessionCoordinator)
        assert coordinator._session_state == mock_state
        assert coordinator._lifecycle_manager is None
        assert coordinator._memory_manager is None

    def test_factory_with_all_dependencies(self):
        """Test factory with all dependencies provided."""
        mock_state = Mock()
        mock_lifecycle = Mock()
        mock_memory = Mock()
        mock_checkpoint = Mock()
        mock_cost = Mock()

        coordinator = create_session_coordinator(
            session_state_manager=mock_state,
            lifecycle_manager=mock_lifecycle,
            memory_manager=mock_memory,
            checkpoint_manager=mock_checkpoint,
            cost_tracker=mock_cost,
        )

        assert isinstance(coordinator, SessionCoordinator)
        assert coordinator._session_state == mock_state
        assert coordinator._lifecycle_manager == mock_lifecycle
        assert coordinator._memory_manager == mock_memory
        assert coordinator._checkpoint_manager == mock_checkpoint
        assert coordinator._cost_tracker == mock_cost


class TestSessionCoordinatorIntegration:
    """Integration tests for SessionCoordinator lifecycle flows."""

    @pytest.fixture
    def mock_session_state(self) -> Mock:
        """Create mock session state manager."""
        state = Mock()
        state.tool_budget = 100
        state.tool_calls_used = 0
        state.observed_files = set()
        state.executed_tools = []
        state.execution_state = Mock()
        state.execution_state.tool_calls_used = 0
        state.execution_state.observed_files = set()
        state.execution_state.executed_tools = []
        state.execution_state.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        state._tool_budget = 100
        state.reset = Mock()
        state.get_remaining_budget = Mock(return_value=100)
        state.is_budget_exhausted = Mock(return_value=False)
        state.get_token_usage = Mock(
            return_value={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        )
        state.get_session_summary = Mock(return_value={"summary": "test"})
        state.update_token_usage = Mock()
        state.reset_token_usage = Mock()
        return state

    @pytest.fixture
    def mock_lifecycle_manager(self) -> Mock:
        """Create mock lifecycle manager."""
        manager = Mock()
        manager.reset_conversation = Mock()
        manager.recover_session = Mock(return_value=True)
        return manager

    def test_complete_session_lifecycle(
        self, mock_session_state: Mock, mock_lifecycle_manager: Mock
    ):
        """Test complete session lifecycle: create, use, end."""
        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
        )

        # Create session
        session_id = coordinator.create_session()
        assert coordinator.is_active is True

        # Use session
        coordinator.update_token_usage(prompt_tokens=100, completion_tokens=50)
        assert coordinator.get_token_usage()["total_tokens"] == 150

        # End session
        coordinator.end_session()
        assert coordinator.is_active is False
        assert coordinator.session_id == session_id  # ID preserved

    def test_session_reset_flow(self, mock_session_state: Mock, mock_lifecycle_manager: Mock):
        """Test session reset maintains session but clears state."""
        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
        )

        # Create and use session
        session_id = coordinator.create_session()
        coordinator.update_token_usage(prompt_tokens=100)

        # Reset with preservation
        coordinator.reset_session(preserve_token_usage=True)

        # Session still active
        assert coordinator.is_active is True
        assert coordinator.session_id == session_id

        # But state was reset
        mock_session_state.reset.assert_called_with(preserve_token_usage=True)
        mock_lifecycle_manager.reset_conversation.assert_called_once()

    def test_multiple_sessions_sequential(
        self, mock_session_state: Mock, mock_lifecycle_manager: Mock
    ):
        """Test creating multiple sessions sequentially."""
        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
        )

        # First session
        session1 = coordinator.create_session()
        assert coordinator.session_id == session1

        # End first session
        coordinator.end_session()
        assert coordinator.is_active is False

        # Second session
        session2 = coordinator.create_session()
        assert coordinator.session_id == session2
        assert coordinator.is_active is True
        assert session2 != session1

    @pytest.mark.asyncio
    async def test_checkpoint_restore_flow(
        self, mock_session_state: Mock, mock_lifecycle_manager: Mock
    ):
        """Test complete checkpoint and restore flow."""
        mock_checkpoint = AsyncMock()
        mock_checkpoint.save_checkpoint = AsyncMock(return_value="ckpt-123")
        mock_checkpoint.restore_checkpoint = AsyncMock(
            return_value={
                "tool_calls_used": 5,
                "tool_budget": 100,
                "token_usage": {"prompt_tokens": 200, "completion_tokens": 100},
                "observed_files": ["file1.py"],
                "executed_tools": ["read", "edit"],
            }
        )

        coordinator = SessionCoordinator(
            session_state_manager=mock_session_state,
            lifecycle_manager=mock_lifecycle_manager,
            checkpoint_manager=mock_checkpoint,
        )

        # Create session and save checkpoint
        coordinator.create_session()
        checkpoint_id = await coordinator.save_checkpoint(description="Test")
        assert checkpoint_id == "ckpt-123"

        # Restore from checkpoint
        success = await coordinator.restore_checkpoint(checkpoint_id)
        assert success is True

        # Verify state was restored
        assert mock_session_state.execution_state.tool_calls_used == 5
        assert mock_session_state.execution_state.observed_files == {"file1.py"}
