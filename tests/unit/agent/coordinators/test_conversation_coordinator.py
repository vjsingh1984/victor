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

"""Tests for ConversationCoordinator.

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
def test_orchestrator_add_message(orchestrator):
    orchestrator.add_message("user", "Hello")
    assert len(orchestrator.messages) == 1
```

AFTER (coordinator test):
```python
def test_conversation_coordinator_add_message(coordinator, mock_conversation):
    coordinator.add_message("user", "Hello")
    mock_conversation.add_message.assert_called_once_with("user", "Hello")
```
"""

import pytest
from unittest.mock import MagicMock, Mock, patch
from typing import List, Any

from victor.agent.coordinators.conversation_coordinator import ConversationCoordinator


class TestConversationCoordinator:
    """Test suite for ConversationCoordinator.

    This coordinator handles conversation message management, extracted
    from AgentOrchestrator as part of Track 4 refactoring.
    """

    @pytest.fixture
    def mock_conversation(self) -> Mock:
        """Create mock conversation object."""
        conversation = Mock()
        conversation.messages = []
        conversation.add_message = Mock()
        return conversation

    @pytest.fixture
    def mock_lifecycle_manager(self) -> Mock:
        """Create mock lifecycle manager."""
        manager = Mock()
        manager.reset_conversation = Mock()
        return manager

    @pytest.fixture
    def mock_memory_manager(self) -> Mock:
        """Create mock memory manager wrapper."""
        memory = Mock()
        memory.is_enabled = True
        memory.add_message = Mock()
        return memory

    @pytest.fixture
    def mock_usage_logger(self) -> Mock:
        """Create mock usage logger."""
        logger = Mock()
        logger.log_event = Mock()
        return logger

    @pytest.fixture
    def coordinator(
        self,
        mock_conversation: Mock,
        mock_lifecycle_manager: Mock,
    ) -> ConversationCoordinator:
        """Create conversation coordinator with default mocks."""
        return ConversationCoordinator(
            conversation=mock_conversation,
            lifecycle_manager=mock_lifecycle_manager,
            memory_manager_wrapper=None,
            usage_logger=None,
        )

    @pytest.fixture
    def coordinator_with_memory(
        self,
        mock_conversation: Mock,
        mock_lifecycle_manager: Mock,
        mock_memory_manager: Mock,
    ) -> ConversationCoordinator:
        """Create coordinator with memory manager."""
        return ConversationCoordinator(
            conversation=mock_conversation,
            lifecycle_manager=mock_lifecycle_manager,
            memory_manager_wrapper=mock_memory_manager,
            usage_logger=None,
        )

    @pytest.fixture
    def coordinator_with_logging(
        self,
        mock_conversation: Mock,
        mock_lifecycle_manager: Mock,
        mock_usage_logger: Mock,
    ) -> ConversationCoordinator:
        """Create coordinator with usage logging."""
        return ConversationCoordinator(
            conversation=mock_conversation,
            lifecycle_manager=mock_lifecycle_manager,
            memory_manager_wrapper=None,
            usage_logger=mock_usage_logger,
        )

    # Test messages property

    def test_messages_property_returns_conversation_messages(
        self, coordinator: ConversationCoordinator, mock_conversation: Mock
    ):
        """Test that messages property delegates to conversation.messages."""
        # Setup
        mock_conversation.messages = [
            Mock(role="user", content="Hello"),
            Mock(role="assistant", content="Hi there"),
        ]

        # Execute
        messages = coordinator.messages

        # Assert
        assert messages == mock_conversation.messages
        assert len(messages) == 2

    def test_messages_property_returns_empty_list_when_no_messages(
        self, coordinator: ConversationCoordinator, mock_conversation: Mock
    ):
        """Test that messages property returns empty list when conversation is empty."""
        # Setup
        mock_conversation.messages = []

        # Execute
        messages = coordinator.messages

        # Assert
        assert messages == []

    # Test add_message

    def test_add_message_delegates_to_conversation(
        self, coordinator: ConversationCoordinator, mock_conversation: Mock
    ):
        """Test that add_message delegates to conversation.add_message."""
        # Execute
        coordinator.add_message("user", "Hello world")

        # Assert
        mock_conversation.add_message.assert_called_once_with("user", "Hello world")

    def test_add_message_with_user_role(
        self, coordinator: ConversationCoordinator, mock_conversation: Mock
    ):
        """Test adding user message."""
        # Execute
        coordinator.add_message("user", "My question")

        # Assert
        mock_conversation.add_message.assert_called_once_with("user", "My question")

    def test_add_message_with_assistant_role(
        self, coordinator: ConversationCoordinator, mock_conversation: Mock
    ):
        """Test adding assistant message."""
        # Execute
        coordinator.add_message("assistant", "My response")

        # Assert
        mock_conversation.add_message.assert_called_once_with(
            "assistant", "My response"
        )

    def test_add_message_with_system_role(
        self, coordinator: ConversationCoordinator, mock_conversation: Mock
    ):
        """Test adding system message."""
        # Execute
        coordinator.add_message("system", "System instruction")

        # Assert
        mock_conversation.add_message.assert_called_once_with(
            "system", "System instruction"
        )

    def test_add_message_with_empty_content(
        self, coordinator: ConversationCoordinator, mock_conversation: Mock
    ):
        """Test adding message with empty content."""
        # Execute
        coordinator.add_message("user", "")

        # Assert
        mock_conversation.add_message.assert_called_once_with("user", "")

    def test_add_message_with_long_content(
        self, coordinator: ConversationCoordinator, mock_conversation: Mock
    ):
        """Test adding message with long content."""
        # Setup
        long_content = "A" * 10000

        # Execute
        coordinator.add_message("user", long_content)

        # Assert
        mock_conversation.add_message.assert_called_once_with("user", long_content)

    def test_add_message_with_special_characters(
        self, coordinator: ConversationCoordinator, mock_conversation: Mock
    ):
        """Test adding message with special characters."""
        # Setup
        special_content = "Hello 👋\nNew line\tTab\0Null"

        # Execute
        coordinator.add_message("user", special_content)

        # Assert
        mock_conversation.add_message.assert_called_once_with("user", special_content)

    # Test add_message with memory manager

    def test_add_message_persists_to_memory_manager_when_enabled(
        self,
        coordinator_with_memory: ConversationCoordinator,
        mock_conversation: Mock,
        mock_memory_manager: Mock,
    ):
        """Test that add_message persists to memory manager when enabled."""
        # Execute
        coordinator_with_memory.add_message("user", "Save this")

        # Assert - both conversation and memory manager should be called
        mock_conversation.add_message.assert_called_once_with("user", "Save this")
        mock_memory_manager.add_message.assert_called_once_with("user", "Save this")

    def test_add_message_does_not_persist_to_memory_manager_when_disabled(
        self,
        coordinator: ConversationCoordinator,
        mock_conversation: Mock,
    ):
        """Test that add_message doesn't persist when memory manager is None."""
        # Execute
        coordinator.add_message("user", "Don't save this")

        # Assert - only conversation should be called
        mock_conversation.add_message.assert_called_once_with("user", "Don't save this")

    def test_add_message_with_memory_manager_disabled(
        self,
        mock_conversation: Mock,
        mock_lifecycle_manager: Mock,
        mock_memory_manager: Mock,
    ):
        """Test add_message when memory manager exists but is disabled."""
        # Setup
        mock_memory_manager.is_enabled = False
        coordinator = ConversationCoordinator(
            conversation=mock_conversation,
            lifecycle_manager=mock_lifecycle_manager,
            memory_manager_wrapper=mock_memory_manager,
            usage_logger=None,
        )

        # Execute
        coordinator.add_message("user", "Test")

        # Assert - memory manager should NOT be called
        mock_conversation.add_message.assert_called_once()
        mock_memory_manager.add_message.assert_not_called()

    # Test add_message with usage logging

    def test_add_message_logs_user_prompt_to_usage_analytics(
        self,
        coordinator_with_logging: ConversationCoordinator,
        mock_conversation: Mock,
        mock_usage_logger: Mock,
    ):
        """Test that user messages are logged to usage analytics."""
        # Execute
        coordinator_with_logging.add_message("user", "My question")

        # Assert
        mock_usage_logger.log_event.assert_called_once_with(
            "user_prompt", {"content": "My question"}
        )

    def test_add_message_logs_assistant_response_to_usage_analytics(
        self,
        coordinator_with_logging: ConversationCoordinator,
        mock_conversation: Mock,
        mock_usage_logger: Mock,
    ):
        """Test that assistant messages are logged to usage analytics."""
        # Execute
        coordinator_with_logging.add_message(
            "assistant", "Here is your answer"
        )

        # Assert
        mock_usage_logger.log_event.assert_called_once_with(
            "assistant_response", {"content": "Here is your answer"}
        )

    def test_add_message_does_not_log_system_messages(
        self,
        coordinator_with_logging: ConversationCoordinator,
        mock_conversation: Mock,
        mock_usage_logger: Mock,
    ):
        """Test that system messages are not logged to usage analytics."""
        # Execute
        coordinator_with_logging.add_message("system", "System instruction")

        # Assert - usage logger should not be called for system messages
        mock_usage_logger.log_event.assert_not_called()

    def test_add_message_without_usage_logger(
        self, coordinator: ConversationCoordinator, mock_conversation: Mock
    ):
        """Test add_message when usage logger is None."""
        # Execute - should not raise
        coordinator.add_message("user", "Test without logging")

        # Assert
        mock_conversation.add_message.assert_called_once()

    # Test add_message with both memory and logging

    def test_add_message_with_memory_and_logging(
        self,
        mock_conversation: Mock,
        mock_lifecycle_manager: Mock,
        mock_memory_manager: Mock,
        mock_usage_logger: Mock,
    ):
        """Test add_message with both memory manager and usage logging enabled."""
        # Setup
        coordinator = ConversationCoordinator(
            conversation=mock_conversation,
            lifecycle_manager=mock_lifecycle_manager,
            memory_manager_wrapper=mock_memory_manager,
            usage_logger=mock_usage_logger,
        )

        # Execute
        coordinator.add_message("user", "Comprehensive test")

        # Assert - all three should be called
        mock_conversation.add_message.assert_called_once_with("user", "Comprehensive test")
        mock_memory_manager.add_message.assert_called_once_with("user", "Comprehensive test")
        mock_usage_logger.log_event.assert_called_once_with(
            "user_prompt", {"content": "Comprehensive test"}
        )

    # Test reset_conversation

    def test_reset_conversation_delegates_to_lifecycle_manager(
        self, coordinator: ConversationCoordinator, mock_lifecycle_manager: Mock
    ):
        """Test that reset_conversation delegates to lifecycle manager."""
        # Execute
        coordinator.reset_conversation()

        # Assert
        mock_lifecycle_manager.reset_conversation.assert_called_once()

    def test_reset_conversation_clears_all_state(
        self, coordinator: ConversationCoordinator, mock_lifecycle_manager: Mock
    ):
        """Test that reset_conversation clears all session state."""
        # Execute
        coordinator.reset_conversation()

        # Assert - lifecycle manager should handle all reset logic
        mock_lifecycle_manager.reset_conversation.assert_called_once()

    def test_reset_conversation_multiple_calls(
        self, coordinator: ConversationCoordinator, mock_lifecycle_manager: Mock
    ):
        """Test that reset_conversation can be called multiple times."""
        # Execute
        coordinator.reset_conversation()
        coordinator.reset_conversation()
        coordinator.reset_conversation()

        # Assert
        assert mock_lifecycle_manager.reset_conversation.call_count == 3

    # Test integration scenarios

    def test_conversation_flow_add_and_reset(
        self,
        coordinator: ConversationCoordinator,
        mock_conversation: Mock,
        mock_lifecycle_manager: Mock,
    ):
        """Test a complete conversation flow: add messages, then reset."""
        # Add messages
        coordinator.add_message("user", "Question 1")
        coordinator.add_message("assistant", "Answer 1")
        coordinator.add_message("user", "Question 2")

        # Assert messages were added
        assert mock_conversation.add_message.call_count == 3

        # Reset
        coordinator.reset_conversation()

        # Assert reset was called
        mock_lifecycle_manager.reset_conversation.assert_called_once()

    def test_multiple_messages_same_role(
        self, coordinator: ConversationCoordinator, mock_conversation: Mock
    ):
        """Test adding multiple messages with the same role."""
        # Execute
        coordinator.add_message("user", "First message")
        coordinator.add_message("user", "Second message")
        coordinator.add_message("user", "Third message")

        # Assert
        assert mock_conversation.add_message.call_count == 3
        calls = mock_conversation.add_message.call_args_list
        assert calls[0][0] == ("user", "First message")
        assert calls[1][0] == ("user", "Second message")
        assert calls[2][0] == ("user", "Third message")

    def test_message_order_preserved(
        self, coordinator: ConversationCoordinator, mock_conversation: Mock
    ):
        """Test that message order is preserved when adding multiple messages."""
        # Execute
        messages = [
            ("user", "First"),
            ("assistant", "Response 1"),
            ("user", "Second"),
            ("assistant", "Response 2"),
            ("user", "Third"),
        ]

        for role, content in messages:
            coordinator.add_message(role, content)

        # Assert - order should match
        calls = mock_conversation.add_message.call_args_list
        for i, (role, content) in enumerate(messages):
            assert calls[i][0] == (role, content)


class TestConversationCoordinatorEdgeCases:
    """Test edge cases and error conditions for ConversationCoordinator."""

    @pytest.fixture
    def mock_conversation(self) -> Mock:
        """Create mock conversation."""
        conversation = Mock()
        conversation.messages = []
        conversation.add_message = Mock()
        return conversation

    @pytest.fixture
    def mock_lifecycle_manager(self) -> Mock:
        """Create mock lifecycle manager."""
        manager = Mock()
        manager.reset_conversation = Mock()
        return manager

    @pytest.fixture
    def coordinator(
        self, mock_conversation: Mock, mock_lifecycle_manager: Mock
    ) -> ConversationCoordinator:
        """Create coordinator."""
        return ConversationCoordinator(
            conversation=mock_conversation,
            lifecycle_manager=mock_lifecycle_manager,
        )

    def test_add_message_with_none_conversation_raises_error(self):
        """Test that creating coordinator with None conversation raises error."""
        with pytest.raises(AttributeError):
            # This should fail when trying to access None attributes
            coordinator = ConversationCoordinator(
                conversation=None,
                lifecycle_manager=Mock(),
            )
            coordinator.add_message("user", "Test")

    def test_add_message_with_unicode_content(
        self, coordinator: ConversationCoordinator, mock_conversation: Mock
    ):
        """Test adding message with unicode characters."""
        # Setup
        unicode_content = "Hello 世界 🌍 مرحبا"

        # Execute
        coordinator.add_message("user", unicode_content)

        # Assert
        mock_conversation.add_message.assert_called_once_with("user", unicode_content)

    def test_add_message_with_newlines_and_tabs(
        self, coordinator: ConversationCoordinator, mock_conversation: Mock
    ):
        """Test adding message with various whitespace characters."""
        # Setup
        content = "Line 1\nLine 2\tTabbed\r\nWindows line"

        # Execute
        coordinator.add_message("user", content)

        # Assert
        mock_conversation.add_message.assert_called_once_with("user", content)

    def test_messages_property_with_conversation_that_raises(
        self, mock_conversation: Mock, mock_lifecycle_manager: Mock
    ):
        """Test messages property when conversation raises an exception."""
        # Setup - use PropertyMock to make the property raise
        from unittest.mock import PropertyMock

        type(mock_conversation).messages = PropertyMock(
            side_effect=RuntimeError("Database error")
        )
        coordinator = ConversationCoordinator(
            conversation=mock_conversation,
            lifecycle_manager=mock_lifecycle_manager,
        )

        # Execute & Assert - should propagate the error
        with pytest.raises(RuntimeError, match="Database error"):
            _ = coordinator.messages

    def test_add_message_when_conversation_raises(
        self, coordinator: ConversationCoordinator, mock_conversation: Mock
    ):
        """Test add_message when conversation.add_message raises an exception."""
        # Setup
        mock_conversation.add_message.side_effect = ValueError("Invalid message")

        # Execute & Assert - should propagate the error
        with pytest.raises(ValueError, match="Invalid message"):
            coordinator.add_message("user", "Bad message")

    def test_reset_conversation_when_lifecycle_manager_raises(
        self, coordinator: ConversationCoordinator, mock_lifecycle_manager: Mock
    ):
        """Test reset_conversation when lifecycle manager raises an exception."""
        # Setup
        mock_lifecycle_manager.reset_conversation.side_effect = RuntimeError(
            "Reset failed"
        )

        # Execute & Assert - should propagate the error
        with pytest.raises(RuntimeError, match="Reset failed"):
            coordinator.reset_conversation()
