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

"""Unit tests for ConversationManager."""


from victor.agent.conversation import ConversationManager


class TestConversationManager:
    """Tests for ConversationManager class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        manager = ConversationManager()
        assert manager.system_prompt == ""
        assert manager.message_count() == 0
        assert manager.messages == []

    def test_init_with_system_prompt(self) -> None:
        """Test initialization with system prompt."""
        manager = ConversationManager(system_prompt="You are a helpful assistant.")
        assert manager.system_prompt == "You are a helpful assistant."

    def test_add_message(self) -> None:
        """Test adding messages."""
        manager = ConversationManager()
        msg = manager.add_message("user", "Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert manager.message_count() == 1

    def test_add_user_message(self) -> None:
        """Test adding user messages."""
        manager = ConversationManager()
        msg = manager.add_user_message("Hello, how are you?")
        assert msg.role == "user"
        assert msg.content == "Hello, how are you?"

    def test_add_assistant_message(self) -> None:
        """Test adding assistant messages."""
        manager = ConversationManager()
        msg = manager.add_assistant_message("I'm doing well, thanks!")
        assert msg.role == "assistant"
        assert msg.content == "I'm doing well, thanks!"

    def test_add_assistant_message_with_tool_calls(self) -> None:
        """Test adding assistant message with tool calls."""
        manager = ConversationManager()
        tool_calls = [{"id": "1", "name": "read_file", "arguments": {"path": "/tmp/test"}}]
        msg = manager.add_assistant_message("Let me check that file.", tool_calls=tool_calls)
        assert msg.role == "assistant"
        assert msg.tool_calls == tool_calls

    def test_add_tool_result(self) -> None:
        """Test adding tool result messages."""
        manager = ConversationManager()
        msg = manager.add_tool_result("tool_1", "File contents here", tool_name="read_file")
        assert msg.role == "tool"
        assert msg.content == "File contents here"
        assert msg.tool_call_id == "tool_1"
        assert msg.name == "read_file"

    def test_ensure_system_prompt(self) -> None:
        """Test system prompt is added at beginning."""
        manager = ConversationManager(system_prompt="System instructions")
        manager.add_user_message("Hello")
        manager.ensure_system_prompt()

        messages = manager.messages
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[0].content == "System instructions"
        assert messages[1].role == "user"

    def test_ensure_system_prompt_idempotent(self) -> None:
        """Test system prompt is only added once."""
        manager = ConversationManager(system_prompt="System instructions")
        manager.ensure_system_prompt()
        manager.ensure_system_prompt()  # Second call should be no-op
        assert manager.message_count() == 1

    def test_get_messages_for_provider(self) -> None:
        """Test getting messages formatted for provider."""
        manager = ConversationManager(system_prompt="Be helpful")
        manager.add_user_message("Hi")
        manager.add_assistant_message("Hello!")

        messages = manager.get_messages_for_provider()
        assert len(messages) == 3
        assert messages[0].role == "system"

    def test_clear(self) -> None:
        """Test clearing conversation."""
        manager = ConversationManager(system_prompt="System")
        manager.ensure_system_prompt()
        manager.add_user_message("Test")
        assert manager.message_count() == 2

        manager.clear()
        assert manager.message_count() == 0
        # System prompt should be re-addable
        manager.ensure_system_prompt()
        assert manager.message_count() == 1

    def test_trim_history(self) -> None:
        """Test history trimming respects max limit."""
        manager = ConversationManager(max_history_messages=5)
        for i in range(10):
            manager.add_user_message(f"Message {i}")

        assert manager.message_count() == 5
        # Should keep most recent messages
        messages = manager.messages
        assert messages[-1].content == "Message 9"

    def test_trim_history_preserves_system_prompt(self) -> None:
        """Test that system prompt is preserved when trimming."""
        manager = ConversationManager(system_prompt="System", max_history_messages=5)
        manager.ensure_system_prompt()
        for i in range(10):
            manager.add_user_message(f"Message {i}")

        messages = manager.messages
        assert messages[0].role == "system"
        assert messages[0].content == "System"

    def test_get_last_user_message(self) -> None:
        """Test getting last user message."""
        manager = ConversationManager()
        manager.add_user_message("First")
        manager.add_assistant_message("Response")
        manager.add_user_message("Second")

        assert manager.get_last_user_message() == "Second"

    def test_get_last_user_message_empty(self) -> None:
        """Test getting last user message when none exists."""
        manager = ConversationManager()
        assert manager.get_last_user_message() is None

    def test_get_last_assistant_message(self) -> None:
        """Test getting last assistant message."""
        manager = ConversationManager()
        manager.add_user_message("Question")
        manager.add_assistant_message("First answer")
        manager.add_assistant_message("Second answer")

        assert manager.get_last_assistant_message() == "Second answer"

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization and deserialization."""
        manager = ConversationManager(system_prompt="Test system", max_history_messages=50)
        manager.ensure_system_prompt()
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi there!")

        # Serialize
        data = manager.to_dict()
        assert data["system_prompt"] == "Test system"
        assert data["max_history"] == 50
        assert len(data["messages"]) == 3

        # Deserialize
        restored = ConversationManager.from_dict(data)
        assert restored.system_prompt == "Test system"
        assert restored.message_count() == 3
        assert restored.get_last_assistant_message() == "Hi there!"

    def test_system_prompt_setter(self) -> None:
        """Test setting system prompt."""
        manager = ConversationManager()
        manager.system_prompt = "New prompt"
        assert manager.system_prompt == "New prompt"

    def test_messages_returns_copy(self) -> None:
        """Test that messages property returns a copy."""
        manager = ConversationManager()
        manager.add_user_message("Test")
        messages1 = manager.messages
        messages2 = manager.messages
        assert messages1 is not messages2
        assert messages1 == messages2
