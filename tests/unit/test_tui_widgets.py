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

"""Unit tests for Victor TUI widgets."""

import pytest
from unittest.mock import MagicMock

# Check if textual is available for testing
pytest.importorskip("textual")

from textual.app import App, ComposeResult

from victor.ui.tui.widgets import (
    MessageContainer,
    UserMessage,
    AssistantMessage,
    ChatInput,
    StatusBar,
    ToolIndicator,
)


class TestUserMessage:
    """Tests for UserMessage widget."""

    def test_user_message_stores_content(self):
        """Test that UserMessage stores content correctly."""
        msg = UserMessage("Hello, world!")
        assert msg._content == "Hello, world!"

    def test_user_message_with_special_chars(self):
        """Test UserMessage handles special characters."""
        content = "Code: `print('hello')` and **bold**"
        msg = UserMessage(content)
        assert msg._content == content


class TestAssistantMessage:
    """Tests for AssistantMessage widget."""

    def test_assistant_message_default_empty(self):
        """Test AssistantMessage initializes with empty content."""
        msg = AssistantMessage()
        assert msg._content == ""
        assert msg.content == ""

    def test_assistant_message_with_content(self):
        """Test AssistantMessage with initial content."""
        msg = AssistantMessage("Initial content")
        assert msg._content == "Initial content"

    def test_set_content(self):
        """Test set_content method."""
        msg = AssistantMessage()
        msg._markdown = MagicMock()  # Mock the markdown widget
        msg.set_content("New content")
        assert msg._content == "New content"
        msg._markdown.update.assert_called_once_with("New content")

    def test_append_content(self):
        """Test append_content method for streaming."""
        msg = AssistantMessage("Start")
        msg._markdown = MagicMock()
        msg.append_content(" more")
        assert msg._content == "Start more"
        msg._markdown.update.assert_called_once_with("Start more")


class TestToolIndicator:
    """Tests for ToolIndicator widget."""

    def test_tool_indicator_initial_state(self):
        """Test ToolIndicator initializes correctly."""
        indicator = ToolIndicator("code_search")
        assert indicator.tool_name == "code_search"
        assert indicator.status == "running"
        assert indicator.elapsed == 0.0

    def test_set_success(self):
        """Test set_success method."""
        indicator = ToolIndicator("test_tool")

        indicator.set_success(elapsed=1.5, preview="Result preview")

        assert indicator.status == "success"
        assert indicator.elapsed == 1.5

    def test_set_error(self):
        """Test set_error method."""
        indicator = ToolIndicator("test_tool")

        indicator.set_error(elapsed=2.0, error_msg="Something failed")

        assert indicator.status == "error"
        assert indicator.elapsed == 2.0


class TestStatusBar:
    """Tests for StatusBar widget."""

    def test_status_bar_initial_state(self):
        """Test StatusBar initializes correctly."""
        bar = StatusBar(
            provider="ollama",
            model="qwen2.5-coder:7b",
            tool_budget=15,
        )
        # StatusBar uses reactive properties (no underscore prefix)
        assert bar.provider == "ollama"
        assert bar.model == "qwen2.5-coder:7b"
        assert bar.tool_budget == 15

    def test_update_metrics(self):
        """Test update_metrics method."""
        bar = StatusBar(provider="test", model="test", tool_budget=10)
        bar.update_metrics(tokens=100, tool_calls=5, status="Processing")

        assert bar.tokens == 100
        assert bar.tool_calls == 5
        assert bar.status == "Processing"

    def test_set_thinking(self):
        """Test set_thinking method."""
        bar = StatusBar(provider="test", model="test", tool_budget=10)
        bar.set_thinking()
        assert bar.status == "Thinking..."

    def test_set_streaming(self):
        """Test set_streaming method."""
        bar = StatusBar(provider="test", model="test", tool_budget=10)
        bar.set_streaming()
        assert bar.status == "Streaming..."

    def test_set_tool_running(self):
        """Test set_tool_running method."""
        bar = StatusBar(provider="test", model="test", tool_budget=10)
        bar.set_tool_running("code_search")
        assert bar.status == "Running code_search..."

    def test_set_ready(self):
        """Test set_ready method."""
        bar = StatusBar(provider="test", model="test", tool_budget=10)
        bar.set_ready()
        assert bar.status == "Ready"


class TestMessageContainer:
    """Tests for MessageContainer widget."""

    def test_message_container_init(self):
        """Test MessageContainer initializes correctly."""
        container = MessageContainer()
        assert container._last_scroll_time == 0
        assert container._scroll_throttle == 0.1


class TestChatInput:
    """Tests for ChatInput widget."""

    def test_chat_input_placeholder(self):
        """Test ChatInput with custom placeholder."""
        chat_input = ChatInput(placeholder="Custom placeholder")
        assert chat_input._placeholder == "Custom placeholder"

    def test_chat_input_default_placeholder(self):
        """Test ChatInput has default placeholder."""
        chat_input = ChatInput()
        assert chat_input._placeholder == "Type your message..."

    def test_chat_input_has_submitted_message_class(self):
        """Test ChatInput has Submitted message class."""
        from victor.ui.tui.widgets.chat_input import ChatInput

        submitted = ChatInput.Submitted("test message")
        assert submitted.value == "test message"


# Integration tests using Textual's test runner
class MinimalTUIApp(App):
    """Minimal app for testing TUI widgets."""

    def compose(self) -> ComposeResult:
        yield MessageContainer()
        yield ChatInput()
        yield StatusBar(
            provider="test",
            model="test-model",
            tool_budget=10,
        )


class TestTUIIntegration:
    """Integration tests for TUI components."""

    @pytest.mark.asyncio
    async def test_app_mounts_widgets(self):
        """Test that the app mounts all widgets correctly."""
        app = MinimalTUIApp()
        async with app.run_test() as _:  # noqa: F841
            # Check that widgets are mounted (query by type)
            messages = app.query_one(MessageContainer)
            assert messages is not None

            chat_input = app.query_one(ChatInput)
            assert chat_input is not None

            status_bar = app.query_one(StatusBar)
            assert status_bar is not None

    @pytest.mark.asyncio
    async def test_message_container_add_messages(self):
        """Test adding messages to the container."""
        app = MinimalTUIApp()
        async with app.run_test() as _:  # noqa: F841
            messages = app.query_one(MessageContainer)

            # Add user message
            user_msg = messages.add_user_message("Hello!")
            assert user_msg is not None
            assert user_msg._content == "Hello!"

            # Add assistant message
            assistant_msg = messages.add_assistant_message("Hi there!")
            assert assistant_msg is not None
            assert assistant_msg._content == "Hi there!"

    @pytest.mark.asyncio
    async def test_clear_messages(self):
        """Test clearing messages from container."""
        app = MinimalTUIApp()
        async with app.run_test() as pilot:
            messages = app.query_one(MessageContainer)

            # Add some messages
            messages.add_user_message("Test 1")
            messages.add_assistant_message("Test 2")

            # Wait for any pending updates
            await pilot.pause()

            # Verify messages were added
            assert len(messages.children) == 2

            # Clear
            messages.clear_messages()

            # Wait for DOM update
            await pilot.pause()

            # Should have no children
            assert len(messages.children) == 0


class TestConfirmationModal:
    """Tests for ConfirmationModal widget."""

    def test_confirmation_modal_imports(self):
        """Test that ConfirmationModal can be imported."""
        from victor.ui.tui.widgets import ConfirmationModal
        from victor.agent.safety import ConfirmationRequest, RiskLevel

        request = ConfirmationRequest(
            tool_name="execute_bash",
            risk_level=RiskLevel.HIGH,
            description="Delete files",
            details=["rm -rf /tmp/test"],
            arguments={"command": "rm -rf /tmp/test"},
        )

        modal = ConfirmationModal(request)
        assert modal.request == request
        assert modal.request.risk_level == RiskLevel.HIGH


class TestTerminalCompatibility:
    """Tests for terminal compatibility detection."""

    def test_check_tui_compatibility_function_exists(self):
        """Test that the compatibility check function exists."""
        from victor.ui.cli import _check_tui_compatibility

        result, reason = _check_tui_compatibility()
        # Result should be a boolean
        assert isinstance(result, bool)
        # Reason should be a string
        assert isinstance(reason, str)
