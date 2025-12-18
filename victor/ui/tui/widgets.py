"""Custom widgets for Victor TUI.

Provides message display, input handling, and status widgets
for the modern chat interface.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, List, Optional

from rich.markdown import Markdown
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Label, Static, RichLog, TextArea
from textual.message import Message

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
                  AND LENGTH(content) < 4000  # Skip very long messages
                  AND content NOT LIKE '<TOOL_OUTPUT%'  # Filter tool outputs
                  AND content NOT LIKE '<%'  # Filter XML-like tags
                  AND content NOT LIKE '{%'  # Filter JSON blobs
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

    DEFAULT_CSS = ""

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
                    ("Victor ", "bold #7cb7ff"),
                    "| ",
                    (f"{self.provider}", ""),
                    (" / ", ""),
                    (f"{self.model}", "bold"),
                ),
                classes="provider-info",
            )
            yield Label(
                Text.assemble(
                    ("Ctrl+C", "bold"),
                    " exit  ",
                    ("Ctrl+Enter", "bold"),
                    " send",
                ),
                classes="shortcuts",
            )

    def update_info(self, provider: str, model: str) -> None:
        """Update provider and model display."""
        self.provider = provider
        self.model = model
        provider_label = self.query_one(".provider-info")
        provider_label.update(Text.assemble(
            ("Victor ", "bold #7cb7ff"),
            "| ",
            (f"{self.provider}", ""),
            (" / ", ""),
            (f"{self.model}", "bold"),
        ))
        self.refresh()


class MessageWidget(Static):
    """Widget for displaying a single chat message.

    Supports both user and assistant messages with different
    styling and markdown rendering for assistant responses.
    """

    DEFAULT_CSS = ""

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

        header = Label(role_label, classes="message-header")
        header.add_class(self.role)

        with Vertical():
            yield header
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

    DEFAULT_CSS = ""

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
        self.write(Text())
        self.write(Text("You", style="bold green"))
        self.write(Text(content))
        self.write(Text())

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the log."""
        self.write(Text())
        self.write(Text("Victor", style="bold blue"))
        self.write(Markdown(content))
        self.write(Text())

    def add_system_message(self, content: str) -> None:
        """Add a system/status message to the log."""
        self.write(Text(f"[{content}]", style="dim italic"))

    def add_error_message(self, content: str) -> None:
        """Add an error message to the log."""
        self.write(Text(f"Error: {content}", style="bold red"))

    def start_streaming(self) -> None:
        """Start a streaming response."""
        self.write(Text())
        self.write(Text("Victor", style="bold blue"))
        self._streaming_message = ""

    def update_streaming(self, content: str) -> None:
        """Update the streaming response content."""
        self._streaming_message = content

    def finish_streaming(self) -> None:
        """Finish the streaming response."""
        if self._streaming_message:
            self.write(Markdown(self._streaming_message))
            self.write(Text())
        self._streaming_message = None


class InputWidget(Static):
    """Input area at the bottom of the screen.

    Uses a multi-line TextArea for better input handling.
    `Ctrl+Enter` sends the message.
    Up/Down arrows navigate input history loaded from conversation DB.
    """

    DEFAULT_CSS = ""

    class Submitted(Message):
        """Custom message to bubble up when input is submitted."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    BINDINGS = [
        ("ctrl+enter", "submit", "Send Message"),
    ]

    # Class-level history shared across instances (persists across sessions)
    _history: list[str] = []
    _max_history: int = 100
    _history_loaded: bool = False  # Track if DB history has been loaded

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._input: TextArea | None = None
        self._history_index: int = -1  # -1 means not browsing history
        self._draft: str = ""  # Save current draft when browsing history

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(classes="input-row"):
                yield Label("> ", classes="prompt-indicator")
                yield TextArea(
                    placeholder="Type message or /help...",
                    id="message-input",
                )
            yield Label(
                "Ctrl+Enter send · ↑↓ history · /exit quit",
                classes="input-hint",
            )

    def on_mount(self) -> None:
        """Focus the input on mount and load history from DB."""
        self._input = self.query_one("#message-input", TextArea)
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
        if not self._input:
            return
        # Only trigger history if cursor is at the top of the text area
        is_at_top = self._input.cursor_location[0] == 0

        if event.key == "up" and is_at_top:
            self._history_prev()
            event.prevent_default()
            event.stop()
        elif event.key == "down":
            # Allow "down" to navigate history only if already browsing
            if self._history_index != -1:
                is_at_bottom = (
                    self._input.cursor_location[0] == len(self._input.document.lines) - 1
                )
                if is_at_bottom:
                    self._history_next()
                    event.prevent_default()
                    event.stop()

    def action_submit(self) -> None:
        """Handle submission via Ctrl+Enter."""
        if self._input and self._input.text:
            self.post_message(self.Submitted(self._input.text))

    def _history_prev(self) -> None:
        """Navigate to previous history entry (Up arrow)."""
        if not InputWidget._history:
            return

        if self._history_index == -1:
            # Starting to browse history, save current draft
            self._draft = self._input.text if self._input else ""
            self._history_index = len(InputWidget._history) - 1
        elif self._history_index > 0:
            self._history_index -= 1
        else:
            return  # Already at oldest entry

        if self._input:
            self._input.load_text(InputWidget._history[self._history_index])
            self._input.cursor_location = (999, 999)  # Move to end

    def _history_next(self) -> None:
        """Navigate to next history entry (Down arrow)."""
        if self._history_index == -1:
            return  # Not browsing history

        if self._history_index < len(InputWidget._history) - 1:
            self._history_index += 1
            if self._input:
                self._input.load_text(InputWidget._history[self._history_index])
                self._input.cursor_location = (999, 999)  # Move to end
        else:
            # Return to draft
            self._history_index = -1
            if self._input:
                self._input.load_text(self._draft)
                self._input.cursor_location = (999, 999)  # Move to end

    @property
    def value(self) -> str:
        """Get current input value."""
        if self._input:
            return self._input.text
        return ""

    def clear(self) -> None:
        """Clear the input field."""
        if self._input:
            self._input.load_text("")
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

    DEFAULT_CSS = ""

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

    DEFAULT_CSS = ""

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
