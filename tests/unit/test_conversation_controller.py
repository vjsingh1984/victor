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

"""Tests for ConversationController."""

import pytest
from unittest.mock import MagicMock

from victor.agent.conversation_controller import (
    ConversationController,
    ConversationConfig,
    ContextMetrics,
)
from victor.agent.conversation_state import ConversationStage
from victor.providers.base import Message


class TestContextMetrics:
    """Tests for ContextMetrics dataclass."""

    def test_utilization_zero_max(self):
        """Test utilization with zero max."""
        metrics = ContextMetrics(
            char_count=100,
            estimated_tokens=25,
            message_count=2,
            max_context_chars=0,
        )
        assert metrics.utilization == 0.0

    def test_utilization_normal(self):
        """Test normal utilization calculation."""
        metrics = ContextMetrics(
            char_count=50000,
            estimated_tokens=12500,
            message_count=10,
            max_context_chars=100000,
        )
        assert metrics.utilization == 0.5

    def test_utilization_capped_at_one(self):
        """Test utilization is capped at 1.0."""
        metrics = ContextMetrics(
            char_count=150000,
            estimated_tokens=37500,
            message_count=20,
            max_context_chars=100000,
        )
        assert metrics.utilization == 1.0


class TestConversationConfig:
    """Tests for ConversationConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ConversationConfig()
        assert config.max_context_chars == 200000
        assert config.chars_per_token_estimate == 4
        assert config.enable_stage_tracking is True
        assert config.enable_context_monitoring is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ConversationConfig(
            max_context_chars=100000,
            chars_per_token_estimate=3,
        )
        assert config.max_context_chars == 100000
        assert config.chars_per_token_estimate == 3


class TestConversationController:
    """Tests for ConversationController class."""

    def test_init_empty(self):
        """Test initialization with no messages."""
        controller = ConversationController()
        assert len(controller.messages) == 0
        assert controller.message_count == 0

    def test_add_user_message(self):
        """Test adding a user message."""
        controller = ConversationController()
        controller.set_system_prompt("You are helpful.")

        msg = controller.add_user_message("Hello!")

        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert controller.message_count == 2  # system + user

    def test_add_assistant_message(self):
        """Test adding an assistant message."""
        controller = ConversationController()

        msg = controller.add_assistant_message("Hi there!")

        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_add_assistant_with_tool_calls(self):
        """Test adding assistant message with tool calls."""
        controller = ConversationController()
        tool_calls = [{"name": "read_file", "arguments": {"path": "test.py"}}]

        msg = controller.add_assistant_message("Let me read that.", tool_calls=tool_calls)

        assert msg.tool_calls == tool_calls

    def test_add_tool_result(self):
        """Test adding a tool result."""
        controller = ConversationController()

        msg = controller.add_tool_result(
            tool_call_id="123",
            tool_name="read_file",
            result="file contents here",
        )

        assert msg.role == "tool"
        assert msg.content == "file contents here"
        assert msg.name == "read_file"
        assert msg.tool_call_id == "123"

    def test_system_prompt_added_once(self):
        """Test that system prompt is only added once."""
        controller = ConversationController()
        controller.set_system_prompt("You are helpful.")

        controller.add_user_message("Hello")
        controller.add_user_message("World")

        # Should only have 1 system message
        system_messages = [m for m in controller.messages if m.role == "system"]
        assert len(system_messages) == 1

    def test_get_context_metrics(self):
        """Test getting context metrics."""
        controller = ConversationController()
        controller.add_user_message("Hello, this is a test message.")
        controller.add_assistant_message("This is a response.")

        metrics = controller.get_context_metrics()

        assert metrics.char_count > 0
        assert metrics.estimated_tokens > 0
        assert metrics.message_count == 2

    def test_context_overflow_detection(self):
        """Test context overflow detection."""
        config = ConversationConfig(max_context_chars=100)
        controller = ConversationController(config=config)

        # Add a message that exceeds limit
        controller.add_user_message("x" * 150)

        assert controller.check_context_overflow() is True

    def test_reset(self):
        """Test resetting conversation."""
        controller = ConversationController()
        controller.set_system_prompt("System prompt")
        controller.add_user_message("Hello")
        controller.add_assistant_message("Hi")

        controller.reset()

        assert controller.message_count == 0
        assert controller._system_added is False

    def test_compact_history(self):
        """Test compacting history."""
        controller = ConversationController()
        controller.set_system_prompt("System")

        # Add many messages
        for i in range(20):
            controller.add_user_message(f"Message {i}")

        # Compact to keep 5 recent
        removed = controller.compact_history(keep_recent=5)

        assert removed > 0
        # System message + 5 recent
        assert controller.message_count == 6

    def test_get_last_user_message(self):
        """Test getting last user message."""
        controller = ConversationController()
        controller.add_user_message("First")
        controller.add_assistant_message("Response")
        controller.add_user_message("Second")

        assert controller.get_last_user_message() == "Second"

    def test_get_last_assistant_message(self):
        """Test getting last assistant message."""
        controller = ConversationController()
        controller.add_user_message("Question")
        controller.add_assistant_message("Answer 1")
        controller.add_assistant_message("Answer 2")

        assert controller.get_last_assistant_message() == "Answer 2"

    def test_on_context_overflow_callback(self):
        """Test context overflow callback."""
        config = ConversationConfig(max_context_chars=100)
        controller = ConversationController(config=config)

        callback_called = [False]
        def on_overflow(metrics):
            callback_called[0] = True

        controller.on_context_overflow(on_overflow)
        controller.add_user_message("x" * 150)

        assert callback_called[0] is True

    def test_to_dict(self):
        """Test exporting to dictionary."""
        controller = ConversationController()
        controller.add_user_message("Hello")
        controller.add_assistant_message("Hi")

        data = controller.to_dict()

        assert "messages" in data
        assert "stage" in data
        assert "metrics" in data
        assert len(data["messages"]) == 2

    def test_from_messages(self):
        """Test creating from existing messages."""
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ]

        controller = ConversationController.from_messages(messages)

        assert controller.message_count == 3
        assert controller.system_prompt == "You are helpful."
        assert controller._system_added is True

    def test_add_message_backward_compat(self):
        """Test add_message for backward compatibility."""
        controller = ConversationController()

        controller.add_message("user", "Hello")
        controller.add_message("assistant", "Hi")

        assert controller.message_count == 2
        assert controller.messages[0].role == "user"
        assert controller.messages[1].role == "assistant"

    def test_stage_tracking(self):
        """Test that conversation stage is tracked."""
        controller = ConversationController()

        # Initially should be in initial stage
        assert controller.stage == ConversationStage.INITIAL

    def test_get_stage_recommended_tools(self):
        """Test getting stage-recommended tools."""
        controller = ConversationController()

        tools = controller.get_stage_recommended_tools()

        assert isinstance(tools, set)
