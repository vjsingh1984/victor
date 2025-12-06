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

"""Chat input widget for Victor TUI."""

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Input, Static


class ChatInput(Static):
    """Chat input widget with prompt label and text input."""

    DEFAULT_CSS = """
    ChatInput {
        dock: bottom;
        height: auto;
        min-height: 3;
        padding: 0 1;
        background: $surface;
        border-top: solid $primary;
    }

    ChatInput Horizontal {
        height: auto;
        min-height: 3;
    }

    ChatInput .prompt-label {
        width: 6;
        padding: 1 0 1 0;
        color: $success;
        text-style: bold;
    }

    ChatInput Input {
        width: 1fr;
        border: none;
        background: $surface;
    }

    ChatInput Input:focus {
        border: none;
    }
    """

    class Submitted(Message):
        """Event fired when user submits input."""

        def __init__(self, value: str) -> None:
            """Initialize submitted event.

            Args:
                value: The submitted text.
            """
            super().__init__()
            self.value = value

    def __init__(self, placeholder: str = "Type your message...", **kwargs: Any) -> None:
        """Initialize chat input.

        Args:
            placeholder: Placeholder text for the input.
            **kwargs: Additional arguments passed to parent Static widget.
        """
        super().__init__(**kwargs)
        self._placeholder = placeholder
        self._input: Input | None = None

    def compose(self) -> ComposeResult:
        """Compose the chat input widget."""
        with Horizontal():
            yield Static("You > ", classes="prompt-label")
            self._input = Input(placeholder=self._placeholder, id="chat-input")
            yield self._input

    def on_mount(self) -> None:
        """Focus the input when mounted."""
        if self._input:
            self._input.focus()

    @on(Input.Submitted)
    def handle_submit(self, event: Input.Submitted) -> None:
        """Handle input submission.

        Args:
            event: The input submitted event.
        """
        value = event.value.strip()
        if value:
            self.post_message(self.Submitted(value))
            if self._input:
                self._input.clear()

    def focus_input(self) -> None:
        """Focus the input field."""
        if self._input:
            self._input.focus()

    def set_disabled(self, disabled: bool) -> None:
        """Enable or disable the input.

        Args:
            disabled: Whether to disable the input.
        """
        if self._input:
            self._input.disabled = disabled
