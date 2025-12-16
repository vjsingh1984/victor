"""Custom widgets for Victor TUI.

Provides message display, input handling, and status widgets
for the modern chat interface.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from rich.markdown import Markdown
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, Label, Static, RichLog

if TYPE_CHECKING:
    pass


def _get_input_history_from_db(limit: int = 100) -> List[str]:
    """Load user message history from conversation database.

    Returns recent unique user messages from the project's conversation.db.
    Filters out tool outputs and system-like messages.
    """
    try:
        from victor.config.settings import get_project_paths

        db_path = get_project_paths().conversation_db
        if not db_path.exists():
            return []

        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                """
                SELECT DISTINCT content
                FROM messages
                WHERE role = 'user'
                  AND content IS NOT NULL
                  AND content != ''
                  AND LENGTH(content) < 2000  -- Skip very long messages
                  AND content NOT LIKE '<TOOL_OUTPUT%'  -- Filter tool outputs
                  AND content NOT LIKE '<%'  -- Filter XML-like tags
                  AND content NOT LIKE '{%'  -- Filter JSON blobs
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            # Reverse to get chronological order (oldest first)
            messages = [row[0] for row in cursor.fetchall()]
            return list(reversed(messages))
    except Exception:
        # Silently fail if DB not available
        return []


class StatusBar(Static):
    """Top status bar showing provider, model, and session info.

    Displays connection status, current provider/model, and
    helpful keyboard shortcuts.
    """

    DEFAULT_CSS = """
    StatusBar {
        dock: top;
        height: 3;
        background: #1e1e2e;
        color: #cdd6f4;
        padding: 0 2;
    }

    StatusBar .status-content {
        width: 100%;
        height: 100%;
        align: center middle;
    }

    StatusBar .provider-info {
        color: #a6adc8;
    }

    StatusBar .model-info {
        color: #bac2de;
    }

    StatusBar .shortcuts {
        color: #6c7086;
        text-align: right;
    }
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-3-5-sonnet",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.provider = provider
        self.model = model

    def compose(self) -> ComposeResult:
        with Horizontal(classes="status-content"):
            yield Label(
                Text.assemble(
                    ("Victor ", "bold #89b4fa"),
                    ("| ", "#6c7086"),
                    (f"{self.provider}", "#a6adc8"),
                    (" / ", "#6c7086"),
                    (f"{self.model}", "#cdd6f4"),
                ),
                classes="provider-info",
            )
            yield Label(
                Text.assemble(
                    ("Ctrl+C", "bold #cdd6f4"),
                    (" exit  ", "#6c7086"),
                    ("Enter", "bold #cdd6f4"),
                    (" send", "#6c7086"),
                ),
                classes="shortcuts",
            )

    def update_info(self, provider: str, model: str) -> None:
        """Update provider and model display."""
        self.provider = provider
        self.model = model
        self.refresh()


class MessageWidget(Static):
    """Widget for displaying a single chat message.

    Supports both user and assistant messages with different
    styling and markdown rendering for assistant responses.
    """

    DEFAULT_CSS = """
    MessageWidget {
        width: 100%;
        padding: 1 2;
        margin: 1 0;
    }

    MessageWidget.user {
        background: #313244;
        border: round #45475a;
    }

    MessageWidget.assistant {
        background: #1e1e2e;
        border: round #313244;
    }

    MessageWidget.system {
        background: #313244;
        border: round #585b70;
        text-style: italic;
    }

    MessageWidget.error {
        background: #302030;
        border: round #f38ba8;
    }

    MessageWidget .message-header {
        height: 1;
        margin-bottom: 1;
        color: #6c7086;
    }

    MessageWidget .message-content {
        width: 100%;
    }
    """

    def __init__(
        self,
        content: str,
        role: str = "assistant",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.content = content
        self.role = role
        self.add_class(role)

    def compose(self) -> ComposeResult:
        role_label = {
            "user": "You",
            "assistant": "Victor",
            "system": "System",
            "error": "Error",
        }.get(self.role, self.role.title())

        role_color = {
            "user": "#a6e3a1",  # Soft green
            "assistant": "#89b4fa",  # Soft blue
            "system": "#a6adc8",  # Neutral gray
            "error": "#f38ba8",  # Soft red
        }.get(self.role, "#cdd6f4")

        with Vertical():
            yield Label(
                Text(f"{role_label}", style=f"bold {role_color}"),
                classes="message-header",
            )
            if self.role == "assistant":
                yield Static(Markdown(self.content), classes="message-content")
            else:
                yield Static(self.content, classes="message-content")

    def update_content(self, content: str) -> None:
        """Update message content (for streaming)."""
        self.content = content
        content_widget = self.query_one(".message-content", Static)
        if self.role == "assistant":
            content_widget.update(Markdown(content))
        else:
            content_widget.update(content)


class ConversationLog(RichLog):
    """Scrollable log for displaying conversation history.

    Provides auto-scrolling, message formatting, and support
    for streaming responses.
    """

    DEFAULT_CSS = """
    ConversationLog {
        height: 1fr;
        border: round #313244;
        background: #11111b;
        padding: 1 2;
        scrollbar-gutter: stable;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            highlight=True,
            markup=True,
            wrap=True,
            auto_scroll=True,
            **kwargs,
        )
        self._streaming_message: str | None = None

    def add_user_message(self, content: str) -> None:
        """Add a user message to the log."""
        self.write(Text())  # Spacing
        self.write(Text("You", style="bold #a6e3a1"))  # Soft green
        self.write(Text(content))
        self.write(Text())  # Spacing

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the log."""
        self.write(Text())  # Spacing
        self.write(Text("Victor", style="bold #89b4fa"))  # Soft blue
        self.write(Markdown(content))
        self.write(Text())  # Spacing

    def add_system_message(self, content: str) -> None:
        """Add a system/status message to the log."""
        self.write(Text(f"[{content}]", style="dim italic #a6adc8"))  # Neutral gray

    def add_error_message(self, content: str) -> None:
        """Add an error message to the log."""
        self.write(Text(f"Error: {content}", style="bold #f38ba8"))  # Soft red

    def start_streaming(self) -> None:
        """Start a streaming response."""
        self.write(Text())  # Spacing
        self.write(Text("Victor", style="bold #89b4fa"))  # Soft blue
        self._streaming_message = ""

    def update_streaming(self, content: str) -> None:
        """Update the streaming response content."""
        self._streaming_message = content

    def finish_streaming(self) -> None:
        """Finish the streaming response."""
        if self._streaming_message:
            self.write(Markdown(self._streaming_message))
            self.write(Text())  # Spacing
        self._streaming_message = None


class InputWidget(Static):
    """Input area at the bottom of the screen.

    Uses standard Input widget for reliable slash command support.
    Up/Down arrows navigate input history loaded from conversation DB.
    """

    DEFAULT_CSS = """
    InputWidget {
        dock: bottom;
        height: 3;
        padding: 0 1;
        background: $primary-darken-3;
        border-top: solid $primary-darken-1;
    }

    InputWidget .input-row {
        width: 100%;
        height: 1;
    }

    InputWidget .prompt-indicator {
        width: 2;
        color: $success;
    }

    InputWidget Input {
        width: 1fr;
        border: none;
        background: transparent;
        padding: 0;
    }

    InputWidget Input:focus {
        border: none;
    }

    InputWidget .input-hint {
        width: 100%;
        height: 1;
        color: $text-muted;
        text-align: right;
    }
    """

    # Class-level history shared across instances (persists across sessions)
    _history: list[str] = []
    _max_history: int = 100
    _history_loaded: bool = False  # Track if DB history has been loaded

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._input: Input | None = None
        self._history_index: int = -1  # -1 means not browsing history
        self._draft: str = ""  # Save current draft when browsing history

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(classes="input-row"):
                yield Label("> ", classes="prompt-indicator")
                yield Input(
                    placeholder="Type message or /help...",
                    id="message-input",
                )
            yield Label(
                "Enter send · ↑↓ history · /exit quit",
                classes="input-hint",
            )

    def on_mount(self) -> None:
        """Focus the input on mount and load history from DB."""
        self._input = self.query_one("#message-input", Input)
        self._input.focus()

        # Load persistent history from conversation database (once per session)
        if not InputWidget._history_loaded:
            InputWidget._history_loaded = True
            db_history = _get_input_history_from_db(limit=InputWidget._max_history)
            if db_history:
                # Merge DB history with any in-session history
                # DB history goes first (older), session history last (newer)
                seen = set(InputWidget._history)
                for msg in db_history:
                    if msg not in seen:
                        InputWidget._history.insert(0, msg)
                        seen.add(msg)
                # Trim to max size
                if len(InputWidget._history) > InputWidget._max_history:
                    InputWidget._history = InputWidget._history[-InputWidget._max_history :]

    def on_key(self, event) -> None:
        """Handle up/down arrow keys for history navigation."""
        if event.key == "up":
            self._history_prev()
            event.prevent_default()
            event.stop()
        elif event.key == "down":
            self._history_next()
            event.prevent_default()
            event.stop()

    def _history_prev(self) -> None:
        """Navigate to previous history entry (Up arrow)."""
        if not InputWidget._history:
            return

        if self._history_index == -1:
            # Starting to browse history, save current draft
            self._draft = self._input.value if self._input else ""
            self._history_index = len(InputWidget._history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        else:
            return  # Already at oldest entry

        if self._input:
            self._input.value = InputWidget._history[self._history_index]
            self._input.cursor_position = len(self._input.value)

    def _history_next(self) -> None:
        """Navigate to next history entry (Down arrow)."""
        if self._history_index == -1:
            return  # Not browsing history

        if self._history_index < len(InputWidget._history) - 1:
            self._history_index += 1
            if self._input:
                self._input.value = InputWidget._history[self._history_index]
                self._input.cursor_position = len(self._input.value)
        else:
            # Return to draft
            self._history_index = -1
            if self._input:
                self._input.value = self._draft
                self._input.cursor_position = len(self._input.value)

    @property
    def value(self) -> str:
        """Get current input value."""
        if self._input:
            return self._input.value
        return ""

    def clear(self) -> None:
        """Clear the input field."""
        if self._input:
            self._input.value = ""
        self._history_index = -1
        self._draft = ""

    def focus_input(self) -> None:
        """Focus the input field."""
        if self._input:
            self._input.focus()

    def add_to_history(self, text: str) -> None:
        """Add message to history (called externally after successful submit)."""
        text = text.strip()
        if not text:
            return
        # Don't add duplicates of the last entry
        if InputWidget._history and InputWidget._history[-1] == text:
            return
        InputWidget._history.append(text)
        # Trim history to max size
        if len(InputWidget._history) > InputWidget._max_history:
            InputWidget._history = InputWidget._history[-InputWidget._max_history :]


class ToolCallWidget(Static):
    """Widget for displaying tool call status.

    Shows tool name, arguments, and execution status
    during agent tool invocations.
    """

    DEFAULT_CSS = """
    ToolCallWidget {
        width: 100%;
        padding: 0 2;
        margin: 0 0 1 0;
        background: $surface-darken-2;
        border: round $secondary-darken-1;
    }

    ToolCallWidget.pending {
        border: round $warning;
    }

    ToolCallWidget.success {
        border: round $success;
    }

    ToolCallWidget.error {
        border: round $error;
    }

    ToolCallWidget .tool-header {
        height: 1;
        color: $text-muted;
    }

    ToolCallWidget .tool-status {
        color: $warning;
    }

    ToolCallWidget.success .tool-status {
        color: $success;
    }

    ToolCallWidget.error .tool-status {
        color: $error;
    }
    """

    def __init__(
        self,
        tool_name: str,
        arguments: dict | None = None,
        status: str = "pending",
        elapsed: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.arguments = arguments or {}
        self.status = status
        self.elapsed = elapsed
        self.add_class(status)

    def compose(self) -> ComposeResult:
        status_icon = {
            "pending": "...",
            "success": "✓",
            "error": "✗",
        }.get(self.status, "?")

        elapsed_str = f" ({self.elapsed:.1f}s)" if self.elapsed else ""

        args_preview = ""
        if self.arguments:
            first_key = next(iter(self.arguments.keys()), None)
            if first_key:
                first_val = str(self.arguments[first_key])[:30]
                if len(str(self.arguments[first_key])) > 30:
                    first_val += "..."
                args_preview = f"({first_key}={first_val})"

        yield Label(
            Text.assemble(
                (status_icon, "bold"),
                " ",
                (self.tool_name, "cyan"),
                (args_preview, "dim"),
                (elapsed_str, "dim"),
            ),
            classes="tool-header",
        )

    def update_status(self, status: str, elapsed: float | None = None) -> None:
        """Update tool call status."""
        self.remove_class(self.status)
        self.status = status
        self.elapsed = elapsed
        self.add_class(status)
        self.refresh()


class ThinkingWidget(Static):
    """Widget for displaying thinking/reasoning content.

    Shows the model's extended thinking process when
    thinking mode is enabled.
    """

    DEFAULT_CSS = """
    ThinkingWidget {
        width: 100%;
        padding: 1 2;
        margin: 0 0 1 0;
        background: $primary-darken-3;
        border: round $primary-darken-1;
        color: $text-muted;
    }

    ThinkingWidget .thinking-header {
        height: 1;
        color: $primary;
        margin-bottom: 1;
    }

    ThinkingWidget .thinking-content {
        width: 100%;
        text-style: italic;
    }
    """

    def __init__(self, content: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.content = content

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(
                Text("Thinking...", style="bold magenta"),
                classes="thinking-header",
            )
            yield Static(self.content, classes="thinking-content")

    def update_content(self, content: str) -> None:
        """Update thinking content."""
        self.content = content
        try:
            content_widget = self.query_one(".thinking-content", Static)
            content_widget.update(content)
        except Exception:
            pass
