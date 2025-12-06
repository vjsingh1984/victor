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

"""Message container widgets for Victor TUI."""

from typing import Any

from textual.app import ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Static, Markdown


class UserMessage(Static):
    """A user message bubble."""

    DEFAULT_CSS = """
    UserMessage {
        background: $primary-darken-2;
        color: $text;
        margin: 1 2;
        padding: 1 2;
        border: round $success;
    }

    UserMessage .label {
        color: $success;
        text-style: bold;
    }
    """

    def __init__(self, content: str) -> None:
        """Initialize user message.

        Args:
            content: The message content.
        """
        super().__init__()
        self._content = content

    def compose(self) -> ComposeResult:
        """Compose the user message widget."""
        yield Static("[bold green]You[/]", classes="label")
        yield Static(self._content)


class AssistantMessage(Static):
    """An assistant message bubble with streaming support."""

    DEFAULT_CSS = """
    AssistantMessage {
        background: $surface;
        color: $text;
        margin: 1 2;
        padding: 1 2;
        border: round $primary;
    }

    AssistantMessage .label {
        color: $primary;
        text-style: bold;
    }

    AssistantMessage .content {
        margin-top: 1;
    }
    """

    def __init__(self, content: str = "") -> None:
        """Initialize assistant message.

        Args:
            content: Initial message content (can be empty for streaming).
        """
        super().__init__()
        self._content = content
        self._markdown: Markdown | None = None

    def compose(self) -> ComposeResult:
        """Compose the assistant message widget."""
        yield Static("[bold blue]Assistant[/]", classes="label")
        self._markdown = Markdown(self._content, classes="content")
        yield self._markdown

    def append_content(self, text: str) -> None:
        """Append content to the message (for streaming).

        Args:
            text: Text to append to the message.
        """
        self._content += text
        if self._markdown:
            self._markdown.update(self._content)

    def set_content(self, text: str) -> None:
        """Set the full message content.

        Args:
            text: The complete message content.
        """
        self._content = text
        if self._markdown:
            self._markdown.update(self._content)

    @property  # type: ignore[misc]
    def content(self) -> str:
        """Get the current message content."""
        return self._content


class MessageContainer(ScrollableContainer):
    """Scrollable container for chat messages."""

    DEFAULT_CSS = """
    MessageContainer {
        height: 1fr;
        scrollbar-gutter: stable;
        background: $background;
        padding: 1 0;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the message container.

        Args:
            **kwargs: Additional arguments passed to parent ScrollableContainer.
        """
        super().__init__(**kwargs)
        self._last_scroll_time: float = 0
        self._scroll_throttle: float = 0.1  # Throttle scrolls to every 100ms

    def add_user_message(self, content: str) -> UserMessage:
        """Add a user message to the container.

        Args:
            content: The message content.

        Returns:
            The created UserMessage widget.
        """
        message = UserMessage(content)
        self.mount(message)
        self._scroll_to_end()
        return message

    def add_assistant_message(self, content: str = "") -> AssistantMessage:
        """Add an assistant message to the container.

        Args:
            content: Initial content (can be empty for streaming).

        Returns:
            The created AssistantMessage widget for streaming updates.
        """
        message = AssistantMessage(content)
        self.mount(message)
        self._scroll_to_end()
        return message

    def _scroll_to_end(self, force: bool = False) -> None:
        """Scroll to end with throttling.

        Args:
            force: Force scroll even if throttled.
        """
        import time

        current_time = time.time()
        if force or (current_time - self._last_scroll_time) >= self._scroll_throttle:
            self.scroll_end(animate=False)
            self._last_scroll_time = current_time

    def scroll_to_end_throttled(self) -> None:
        """Scroll to end (throttled for streaming updates)."""
        self._scroll_to_end(force=False)

    def scroll_to_end_now(self) -> None:
        """Scroll to end immediately (bypasses throttle)."""
        self._scroll_to_end(force=True)

    def clear_messages(self) -> None:
        """Clear all messages from the container."""
        self.remove_children()
