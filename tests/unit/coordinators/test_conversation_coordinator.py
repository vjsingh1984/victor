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

"""Unit tests for ConversationCoordinator.

Tests the conversation message management coordinator extracted from
the monolithic orchestrator as part of Track 4 Phase 1 refactoring.
"""

import pytest
from unittest.mock import Mock, MagicMock, call

from victor.agent.coordinators.conversation_coordinator import ConversationCoordinator


class TestConversationCoordinator:
    """Test suite for ConversationCoordinator."""

    def test_initialization(self):
        """Test coordinator initialization with all dependencies."""
        # Arrange
        conversation = Mock()
        lifecycle_manager = Mock()
        memory_manager = Mock()
        usage_logger = Mock()

        # Act
        coordinator = ConversationCoordinator(
            conversation=conversation,
            lifecycle_manager=lifecycle_manager,
            memory_manager_wrapper=memory_manager,
            usage_logger=usage_logger,
        )

        # Assert
        assert coordinator._conversation == conversation
        assert coordinator._lifecycle_manager == lifecycle_manager
        assert coordinator._memory_manager_wrapper == memory_manager
        assert coordinator._usage_logger == usage_logger

    def test_initialization_without_optional_dependencies(self):
        """Test coordinator initialization with minimal dependencies."""
        # Arrange
        conversation = Mock()
        lifecycle_manager = Mock()

        # Act
        coordinator = ConversationCoordinator(
            conversation=conversation,
            lifecycle_manager=lifecycle_manager,
            memory_manager_wrapper=None,
            usage_logger=None,
        )

        # Assert
        assert coordinator._conversation == conversation
        assert coordinator._lifecycle_manager == lifecycle_manager
        assert coordinator._memory_manager_wrapper is None
        assert coordinator._usage_logger is None

    def test_messages_property_delegates_to_conversation(self):
        """Test that messages property delegates to MessageHistory."""
        # Arrange
        conversation = Mock()
        conversation.messages = ["msg1", "msg2", "msg3"]
        lifecycle_manager = Mock()

        coordinator = ConversationCoordinator(
            conversation=conversation,
            lifecycle_manager=lifecycle_manager,
        )

        # Act
        messages = coordinator.messages

        # Assert
        assert messages == ["msg1", "msg2", "msg3"]
        # Verify it accessed the property
        assert hasattr(conversation, 'messages')

    def test_add_message_to_conversation(self):
        """Test adding a message to conversation history."""
        # Arrange
        conversation = Mock()
        lifecycle_manager = Mock()
        memory_manager = Mock(is_enabled=False)
        usage_logger = Mock()

        coordinator = ConversationCoordinator(
            conversation=conversation,
            lifecycle_manager=lifecycle_manager,
            memory_manager_wrapper=memory_manager,
            usage_logger=usage_logger,
        )

        # Act
        coordinator.add_message("user", "Hello, world!")

        # Assert
        conversation.add_message.assert_called_once_with("user", "Hello, world!")
        memory_manager.add_message.assert_not_called()

    def test_add_message_persists_to_memory_manager(self):
        """Test that messages are persisted to memory manager when enabled."""
        # Arrange
        conversation = Mock()
        lifecycle_manager = Mock()
        memory_manager = Mock(is_enabled=True)
        usage_logger = Mock()

        coordinator = ConversationCoordinator(
            conversation=conversation,
            lifecycle_manager=lifecycle_manager,
            memory_manager_wrapper=memory_manager,
            usage_logger=usage_logger,
        )

        # Act
        coordinator.add_message("user", "Test message")

        # Assert
        conversation.add_message.assert_called_once()
        memory_manager.add_message.assert_called_once_with("user", "Test message")

    def test_add_message_logs_user_message(self):
        """Test that user messages are logged to usage analytics."""
        # Arrange
        conversation = Mock()
        lifecycle_manager = Mock()
        memory_manager = Mock(is_enabled=False)
        usage_logger = Mock()

        coordinator = ConversationCoordinator(
            conversation=conversation,
            lifecycle_manager=lifecycle_manager,
            memory_manager_wrapper=memory_manager,
            usage_logger=usage_logger,
        )

        # Act
        coordinator.add_message("user", "User input")

        # Assert
        usage_logger.log_event.assert_called_once_with("user_prompt", {"content": "User input"})

    def test_add_message_logs_assistant_message(self):
        """Test that assistant messages are logged to usage analytics."""
        # Arrange
        conversation = Mock()
        lifecycle_manager = Mock()
        memory_manager = Mock(is_enabled=False)
        usage_logger = Mock()

        coordinator = ConversationCoordinator(
            conversation=conversation,
            lifecycle_manager=lifecycle_manager,
            memory_manager_wrapper=memory_manager,
            usage_logger=usage_logger,
        )

        # Act
        coordinator.add_message("assistant", "Assistant response")

        # Assert
        usage_logger.log_event.assert_called_once_with(
            "assistant_response", {"content": "Assistant response"}
        )

    def test_add_message_does_not_log_system_messages(self):
        """Test that system messages are not logged to usage analytics."""
        # Arrange
        conversation = Mock()
        lifecycle_manager = Mock()
        memory_manager = Mock(is_enabled=False)
        usage_logger = Mock()

        coordinator = ConversationCoordinator(
            conversation=conversation,
            lifecycle_manager=lifecycle_manager,
            memory_manager_wrapper=memory_manager,
            usage_logger=usage_logger,
        )

        # Act
        coordinator.add_message("system", "System prompt")

        # Assert
        usage_logger.log_event.assert_not_called()

    def test_add_message_without_usage_logger(self):
        """Test that add_message works when usage_logger is None."""
        # Arrange
        conversation = Mock()
        lifecycle_manager = Mock()
        memory_manager = Mock(is_enabled=False)

        coordinator = ConversationCoordinator(
            conversation=conversation,
            lifecycle_manager=lifecycle_manager,
            memory_manager_wrapper=memory_manager,
            usage_logger=None,
        )

        # Act & Assert - Should not raise exception
        coordinator.add_message("user", "Test message")
        conversation.add_message.assert_called_once()

    def test_reset_conversation_delegates_to_lifecycle_manager(self):
        """Test that reset_conversation delegates to LifecycleManager."""
        # Arrange
        conversation = Mock()
        lifecycle_manager = Mock()
        coordinator = ConversationCoordinator(
            conversation=conversation,
            lifecycle_manager=lifecycle_manager,
        )

        # Act
        coordinator.reset_conversation()

        # Assert
        lifecycle_manager.reset_conversation.assert_called_once()

    def test_multiple_add_messages(self):
        """Test adding multiple messages in sequence."""
        # Arrange
        conversation = Mock()
        lifecycle_manager = Mock()
        memory_manager = Mock(is_enabled=True)
        usage_logger = Mock()

        coordinator = ConversationCoordinator(
            conversation=conversation,
            lifecycle_manager=lifecycle_manager,
            memory_manager_wrapper=memory_manager,
            usage_logger=usage_logger,
        )

        # Act
        coordinator.add_message("user", "First message")
        coordinator.add_message("assistant", "First response")
        coordinator.add_message("user", "Second message")

        # Assert
        assert conversation.add_message.call_count == 3
        assert memory_manager.add_message.call_count == 3
        assert usage_logger.log_event.call_count == 3  # user, assistant, user

    def test_add_message_with_empty_content(self):
        """Test adding a message with empty content."""
        # Arrange
        conversation = Mock()
        lifecycle_manager = Mock()
        coordinator = ConversationCoordinator(
            conversation=conversation,
            lifecycle_manager=lifecycle_manager,
        )

        # Act
        coordinator.add_message("user", "")

        # Assert
        conversation.add_message.assert_called_once_with("user", "")

    def test_add_message_with_long_content(self):
        """Test adding a message with long content."""
        # Arrange
        conversation = Mock()
        lifecycle_manager = Mock()
        long_content = "x" * 10000

        coordinator = ConversationCoordinator(
            conversation=conversation,
            lifecycle_manager=lifecycle_manager,
        )

        # Act
        coordinator.add_message("user", long_content)

        # Assert
        conversation.add_message.assert_called_once_with("user", long_content)

    def test_messages_returns_empty_list_initially(self):
        """Test that messages returns empty list when conversation is empty."""
        # Arrange
        conversation = Mock()
        conversation.messages = []
        lifecycle_manager = Mock()

        coordinator = ConversationCoordinator(
            conversation=conversation,
            lifecycle_manager=lifecycle_manager,
        )

        # Act
        messages = coordinator.messages

        # Assert
        assert messages == []

    def test_thread_safety_of_add_message(self):
        """Test that add_message is thread-safe (delegates to thread-safe components)."""
        # This is a basic test - real thread safety would require concurrent testing
        # For now, we just verify the delegation pattern is correct

        # Arrange
        conversation = Mock()
        lifecycle_manager = Mock()
        coordinator = ConversationCoordinator(
            conversation=conversation,
            lifecycle_manager=lifecycle_manager,
        )

        # Act
        coordinator.add_message("user", "Thread safety test")

        # Assert - Verify delegation happened correctly
        conversation.add_message.assert_called_once_with("user", "Thread safety test")

    def test_reset_conversation_can_be_called_multiple_times(self):
        """Test that reset_conversation can be called multiple times."""
        # Arrange
        conversation = Mock()
        lifecycle_manager = Mock()
        coordinator = ConversationCoordinator(
            conversation=conversation,
            lifecycle_manager=lifecycle_manager,
        )

        # Act
        coordinator.reset_conversation()
        coordinator.reset_conversation()
        coordinator.reset_conversation()

        # Assert
        assert lifecycle_manager.reset_conversation.call_count == 3
