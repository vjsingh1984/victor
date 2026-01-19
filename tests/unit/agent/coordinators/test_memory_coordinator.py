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

"""Unit tests for MemoryCoordinator.

Tests memory management, context retrieval, session statistics, and recovery.
"""

from unittest.mock import MagicMock
from typing import Any, Dict

import pytest

from victor.agent.coordinators.memory_coordinator import (
    MemoryCoordinator,
    MemoryStats,
    SessionInfo,
    create_memory_coordinator,
)


@pytest.fixture
def mock_memory_manager():
    """Create a mock memory manager."""
    manager = MagicMock()
    return manager


@pytest.fixture
def mock_conversation_store():
    """Create a mock conversation store."""
    store = MagicMock()
    store.messages = []
    return store


class TestMemoryCoordinator:
    """Test suite for MemoryCoordinator."""

    def test_initialization(self):
        """Test coordinator initialization."""
        coordinator = MemoryCoordinator()

        assert coordinator._memory_manager is None
        assert coordinator._session_id is None
        assert coordinator._conversation_store is None

    def test_initialization_with_components(self, mock_memory_manager, mock_conversation_store):
        """Test coordinator initialization with components."""
        coordinator = MemoryCoordinator(
            memory_manager=mock_memory_manager,
            session_id="session-123",
            conversation_store=mock_conversation_store,
        )

        assert coordinator._memory_manager == mock_memory_manager
        assert coordinator._session_id == "session-123"
        assert coordinator._conversation_store == mock_conversation_store

    def test_get_memory_context_with_manager(self, mock_memory_manager):
        """Test get_memory_context with memory manager available."""
        # Setup
        expected_context = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        mock_memory_manager.get_context.return_value = expected_context

        coordinator = MemoryCoordinator(
            memory_manager=mock_memory_manager,
            session_id="session-123",
        )

        # Execute
        context = coordinator.get_memory_context(max_tokens=4000)

        # Verify
        assert context == expected_context
        mock_memory_manager.get_context.assert_called_once_with(max_tokens=4000)

    def test_get_memory_context_without_manager(self):
        """Test get_memory_context falls back without memory manager."""
        # Setup
        mock_store = MagicMock()
        mock_msg = MagicMock()
        mock_msg.model_dump.return_value = {"role": "user", "content": "Test"}
        mock_store.messages = [mock_msg]

        coordinator = MemoryCoordinator(
            conversation_store=mock_store,
        )

        # Execute
        context = coordinator.get_memory_context()

        # Verify
        assert len(context) == 1
        assert context[0]["role"] == "user"

    def test_get_memory_context_with_exception(self, mock_memory_manager):
        """Test get_memory_context handles exceptions gracefully."""
        # Setup
        mock_memory_manager.get_context.side_effect = Exception("Memory error")
        mock_store = MagicMock()
        mock_store.messages = []

        coordinator = MemoryCoordinator(
            memory_manager=mock_memory_manager,
            session_id="session-123",
            conversation_store=mock_store,
        )

        # Execute
        context = coordinator.get_memory_context()

        # Verify - should fall back to in-memory
        assert context == []

    def test_get_session_stats_with_manager(self, mock_memory_manager):
        """Test get_session_stats with memory manager available."""
        # Setup
        mock_stats = {
            "message_count": 10,
            "total_tokens": 5000,
            "max_tokens": 8000,
            "available_tokens": 3000,
        }
        mock_memory_manager.get_session_stats.return_value = mock_stats

        coordinator = MemoryCoordinator(
            memory_manager=mock_memory_manager,
            session_id="session-123",
        )

        # Execute
        stats = coordinator.get_session_stats(message_count=5)

        # Verify
        assert stats.enabled is True
        assert stats.session_id == "session-123"
        assert stats.message_count == 10
        assert stats.total_tokens == 5000
        assert stats.max_tokens == 8000
        assert stats.available_tokens == 3000
        assert stats.found is True

    def test_get_session_stats_without_manager(self):
        """Test get_session_stats without memory manager."""
        coordinator = MemoryCoordinator()

        # Execute
        stats = coordinator.get_session_stats(message_count=5)

        # Verify
        assert stats.enabled is False
        assert stats.session_id is None
        assert stats.message_count == 5

    def test_get_session_stats_session_not_found(self, mock_memory_manager):
        """Test get_session_stats when session not found."""
        # Setup - empty stats
        mock_memory_manager.get_session_stats.return_value = {}

        coordinator = MemoryCoordinator(
            memory_manager=mock_memory_manager,
            session_id="session-123",
        )

        # Execute
        stats = coordinator.get_session_stats(message_count=5)

        # Verify
        assert stats.enabled is True
        assert stats.found is False
        assert stats.error == "Session not found"

    def test_get_session_stats_with_exception(self, mock_memory_manager):
        """Test get_session_stats handles exceptions."""
        # Setup
        mock_memory_manager.get_session_stats.side_effect = Exception("Stats error")

        coordinator = MemoryCoordinator(
            memory_manager=mock_memory_manager,
            session_id="session-123",
        )

        # Execute
        stats = coordinator.get_session_stats(message_count=5)

        # Verify
        assert stats.enabled is True
        assert stats.error == "Stats error"

    def test_get_recent_sessions_with_manager(self, mock_memory_manager):
        """Test get_recent_sessions with memory manager available."""
        # Setup
        expected_sessions = [
            {"session_id": "session-1", "created_at": "2025-01-01"},
            {"session_id": "session-2", "created_at": "2025-01-02"},
        ]
        mock_memory_manager.get_recent_sessions.return_value = expected_sessions

        coordinator = MemoryCoordinator(
            memory_manager=mock_memory_manager,
        )

        # Execute
        sessions = coordinator.get_recent_sessions(limit=10)

        # Verify
        assert sessions == expected_sessions
        mock_memory_manager.get_recent_sessions.assert_called_once_with(limit=10)

    def test_get_recent_sessions_without_manager(self):
        """Test get_recent_sessions without memory manager."""
        coordinator = MemoryCoordinator()

        # Execute
        sessions = coordinator.get_recent_sessions(limit=10)

        # Verify
        assert sessions == []

    def test_get_recent_sessions_with_exception(self, mock_memory_manager):
        """Test get_recent_sessions handles exceptions."""
        # Setup
        mock_memory_manager.get_recent_sessions.side_effect = Exception("Sessions error")

        coordinator = MemoryCoordinator(
            memory_manager=mock_memory_manager,
        )

        # Execute
        sessions = coordinator.get_recent_sessions(limit=10)

        # Verify
        assert sessions == []

    def test_recover_session_success(self, mock_memory_manager):
        """Test successful session recovery."""
        # Setup
        mock_memory_manager.recover_session.return_value = True

        coordinator = MemoryCoordinator(
            memory_manager=mock_memory_manager,
        )

        # Execute
        success = coordinator.recover_session("session-456")

        # Verify
        assert success is True
        assert coordinator._session_id == "session-456"
        mock_memory_manager.recover_session.assert_called_once_with("session-456")

    def test_recover_session_failure(self, mock_memory_manager):
        """Test failed session recovery."""
        # Setup
        mock_memory_manager.recover_session.return_value = False

        coordinator = MemoryCoordinator(
            memory_manager=mock_memory_manager,
            session_id="session-123",
        )

        # Execute
        success = coordinator.recover_session("session-456")

        # Verify
        assert success is False
        # Session ID should not change on failure
        assert coordinator._session_id == "session-123"

    def test_recover_session_without_manager(self):
        """Test recover_session without memory manager."""
        coordinator = MemoryCoordinator()

        # Execute
        success = coordinator.recover_session("session-456")

        # Verify
        assert success is False

    def test_recover_session_with_exception(self, mock_memory_manager):
        """Test recover_session handles exceptions."""
        # Setup
        mock_memory_manager.recover_session.side_effect = Exception("Recovery error")

        coordinator = MemoryCoordinator(
            memory_manager=mock_memory_manager,
        )

        # Execute
        success = coordinator.recover_session("session-456")

        # Verify
        assert success is False

    def test_set_session_id(self):
        """Test setting session ID."""
        coordinator = MemoryCoordinator()

        assert coordinator.get_session_id() is None

        coordinator.set_session_id("session-789")

        assert coordinator.get_session_id() == "session-789"

    def test_clear_session_id(self):
        """Test clearing session ID."""
        coordinator = MemoryCoordinator(session_id="session-123")

        assert coordinator.get_session_id() == "session-123"

        coordinator.set_session_id(None)

        assert coordinator.get_session_id() is None

    def test_is_enabled_true(self, mock_memory_manager):
        """Test is_enabled returns True when manager and session exist."""
        coordinator = MemoryCoordinator(
            memory_manager=mock_memory_manager,
            session_id="session-123",
        )

        assert coordinator.is_enabled() is True

    def test_is_enabled_no_manager(self):
        """Test is_enabled returns False when no manager."""
        coordinator = MemoryCoordinator(
            session_id="session-123",
        )

        assert coordinator.is_enabled() is False

    def test_is_enabled_no_session(self, mock_memory_manager):
        """Test is_enabled returns False when no session."""
        coordinator = MemoryCoordinator(
            memory_manager=mock_memory_manager,
        )

        assert coordinator.is_enabled() is False

    def test_get_in_memory_messages_success(self, mock_conversation_store):
        """Test _get_in_memory_messages with valid store."""
        # Setup
        mock_msg = MagicMock()
        mock_msg.model_dump.return_value = {"role": "user", "content": "Test"}
        mock_conversation_store.messages = [mock_msg]

        coordinator = MemoryCoordinator(
            conversation_store=mock_conversation_store,
        )

        # Execute
        messages = coordinator._get_in_memory_messages()

        # Verify
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_get_in_memory_messages_no_store(self):
        """Test _get_in_memory_messages without store."""
        coordinator = MemoryCoordinator()

        # Execute
        messages = coordinator._get_in_memory_messages()

        # Verify
        assert messages == []

    def test_get_in_memory_messages_with_exception(self, mock_conversation_store):
        """Test _get_in_memory_messages handles exceptions."""
        # Setup
        mock_conversation_store.messages = MagicMock(side_effect=Exception("Messages error"))

        coordinator = MemoryCoordinator(
            conversation_store=mock_conversation_store,
        )

        # Execute
        messages = coordinator._get_in_memory_messages()

        # Verify
        assert messages == []


class TestCreateMemoryCoordinator:
    """Test suite for factory function."""

    def test_factory_with_defaults(self):
        """Test factory creates coordinator with defaults."""
        coordinator = create_memory_coordinator()

        assert coordinator._memory_manager is None
        assert coordinator._session_id is None
        assert coordinator._conversation_store is None

    def test_factory_with_all_components(self, mock_memory_manager, mock_conversation_store):
        """Test factory with all components."""
        coordinator = create_memory_coordinator(
            memory_manager=mock_memory_manager,
            session_id="session-123",
            conversation_store=mock_conversation_store,
        )

        assert coordinator._memory_manager == mock_memory_manager
        assert coordinator._session_id == "session-123"
        assert coordinator._conversation_store == mock_conversation_store
