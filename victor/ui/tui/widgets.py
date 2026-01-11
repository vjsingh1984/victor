"""Custom widgets for Victor TUI.

Provides message display, input handling, and status widgets
for the modern chat interface.

Enhanced widgets include:
- EnhancedConversationLog: Better streaming support with ScrollableContainer
- CodeBlock: Syntax-highlighted code with copy functionality
- ToolProgressPanel: Real-time tool execution visualization
- ThinkingSidebar: Collapsible panel for model reasoning
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
import time
from typing import TYPE_CHECKING, Callable, List, Optional

from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Label, Static, RichLog, TextArea, Button, ProgressBar
from textual.message import Message
from textual.reactive import reactive
from textual.binding import Binding

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
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (limit,),
            )
            # Already in chronological order (oldest first) from ORDER BY timestamp ASC
            messages = [row[0] for row in cursor.fetchall()]
            return messages
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
        self.status = "Idle"

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
            status_label = Label(self.status, classes="status-indicator")
            status_label.add_class("idle")
            yield status_label
            yield Label(
                Text.assemble(
                    ("Ctrl+C", "bold"),
                    " exit  ",
                    ("Enter", "bold"),
                    " send",
                ),
                classes="shortcuts",
            )

    def update_info(self, provider: str, model: str) -> None:
        """Update provider and model display."""
        self.provider = provider
        self.model = model
        provider_label = self.query_one(".provider-info")
        provider_label.update(
            Text.assemble(
                ("Victor ", "bold #7cb7ff"),
                "| ",
                (f"{self.provider}", ""),
                (" / ", ""),
                (f"{self.model}", "bold"),
            )
        )
        self.refresh()

    def update_status(self, status: str, state: str = "idle") -> None:
        """Update status text and styling."""
        self.status = status
        label = self.query_one(".status-indicator", Label)
        label.update(self.status)
        label.remove_class("idle", "busy", "streaming")
        label.add_class(state)


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


class SubmitTextArea(TextArea):
    """Custom TextArea that submits on Enter and allows Shift+Enter for newlines.

    This subclass overrides the default Enter behavior to emit a Submit message
    instead of inserting a newline. Shift+Enter still inserts newlines.
    """

    class Submit(Message):
        """Message emitted when Enter is pressed."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def _on_key(self, event) -> None:
        """Handle key events before TextArea processes them."""
        # Check for plain Enter (not shift+enter)
        # In Textual, shift+enter appears as "shift+enter" in event.key
        if event.key == "enter":
            # Plain Enter: submit instead of newline
            if self.text.strip():
                self.post_message(self.Submit(self.text))
            event.prevent_default()
            event.stop()
            return
        # Shift+Enter and all other keys handled normally by parent
        super()._on_key(event)


class InputWidget(Static):
    """Input area at the bottom of the screen.

    Uses a multi-line TextArea for better input handling.
    `Enter` sends the message, `Shift+Enter` adds a newline.
    Up/Down arrows navigate input history loaded from conversation DB.
    """

    DEFAULT_CSS = ""

    class Submitted(Message):
        """Custom message to bubble up when input is submitted."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    BINDINGS = [
        ("ctrl+enter", "submit", "Send Message"),  # Alternative binding
    ]

    # Class-level history shared across instances (persists across sessions)
    _history: list[str] = []
    _max_history: int = 100
    _history_loaded: bool = False  # Track if DB history has been loaded

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._input: SubmitTextArea | None = None
        self._history_index: int = -1  # -1 means not browsing history
        self._draft: str = ""  # Save current draft when browsing history
        self._history_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(classes="input-row"):
                yield Label("> ", classes="prompt-indicator")
                yield SubmitTextArea(
                    placeholder="Type message or /help...",
                    id="message-input",
                )
            yield Label(
                "Enter send · Shift+Enter newline · ↑↓ history · /help",
                classes="input-hint",
            )

    def on_submit_text_area_submit(self, event: SubmitTextArea.Submit) -> None:
        """Handle submit from the custom TextArea."""
        if event.value.strip():
            self.post_message(self.Submitted(event.value))

    def on_mount(self) -> None:
        """Focus the input on mount and load history from DB."""
        self._input = self.query_one("#message-input", SubmitTextArea)
        self._input.focus()

        # Load persistent history from conversation database (once per session)
        if not InputWidget._history_loaded:
            InputWidget._history_loaded = True
            self._history_task = asyncio.create_task(self._load_history_async())

    async def _load_history_async(self) -> None:
        """Load input history without blocking the UI."""
        try:
            db_history = await asyncio.to_thread(
                _get_input_history_from_db,
                limit=InputWidget._max_history,
            )
        except Exception:
            return

        if not db_history:
            return

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
        """Handle key events for history navigation.

        - Up/Down: Navigate history (when at top/bottom of input)
        Note: Enter/Shift+Enter handled by SubmitTextArea
        """
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
                is_at_bottom = self._input.cursor_location[0] == len(self._input.document.lines) - 1
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


# =============================================================================
# Enhanced TUI Widgets (Phase 1 TUI Enhancement)
# =============================================================================


class CodeBlock(Static):
    """Syntax-highlighted code block with copy functionality.

    Features:
    - Automatic language detection from code fences
    - Syntax highlighting via Rich Syntax
    - Copy-to-clipboard button
    - Line numbers option
    """

    DEFAULT_CSS = """
    CodeBlock {
        background: $surface;
        border: solid $primary-darken-2;
        padding: 0 1;
        margin: 1 0;
    }

    CodeBlock .code-header {
        color: $text-muted;
        text-style: italic;
    }

    CodeBlock .copy-button {
        dock: right;
        width: auto;
        min-width: 6;
        height: 1;
        margin: 0 0 0 1;
    }
    """

    def __init__(
        self,
        code: str,
        language: str = "python",
        show_line_numbers: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.code = code
        self.language = language
        self.show_line_numbers = show_line_numbers

    def compose(self) -> ComposeResult:
        with Horizontal(classes="code-header-row"):
            yield Label(f"[{self.language}]", classes="code-header")
            yield Button("Copy", classes="copy-button", variant="default")
        yield Static(
            Syntax(
                self.code,
                self.language,
                theme="monokai",
                line_numbers=self.show_line_numbers,
                word_wrap=True,
            ),
            classes="code-content",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle copy button press."""
        try:
            import pyperclip

            pyperclip.copy(self.code)
            # Update button text briefly
            event.button.label = "Copied!"
            self.set_timer(1.5, lambda: setattr(event.button, "label", "Copy"))
        except ImportError:
            # pyperclip not available
            event.button.label = "N/A"


class StreamingMessageBlock(Static):
    """Message block that supports live streaming updates.

    Unlike MessageWidget which requires full content, this widget
    can receive content chunks and update in real-time with proper
    markdown rendering.
    """

    DEFAULT_CSS = """
    StreamingMessageBlock {
        padding: 1;
        margin: 0 0 1 0;
    }

    StreamingMessageBlock.user {
        background: $surface;
    }

    StreamingMessageBlock.assistant {
        background: $surface-darken-1;
    }

    StreamingMessageBlock .message-header {
        text-style: bold;
        margin-bottom: 1;
    }

    StreamingMessageBlock .message-header.user {
        color: $success;
    }

    StreamingMessageBlock .message-header.assistant {
        color: $primary;
    }

    StreamingMessageBlock .streaming-indicator {
        color: $warning;
        text-style: italic;
    }
    """

    content = reactive("")
    is_streaming = reactive(False)

    def __init__(
        self,
        role: str = "assistant",
        initial_content: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.role = role
        self.content = initial_content
        self.add_class(role)
        self._code_blocks: List[tuple[str, str]] = []  # (code, language)

    def compose(self) -> ComposeResult:
        role_label = {
            "user": "You",
            "assistant": "Victor",
            "system": "System",
            "error": "Error",
        }.get(self.role, self.role.title())

        with Vertical():
            header = Label(role_label, classes="message-header")
            header.add_class(self.role)
            yield header
            yield Static(id="message-body", classes="message-content")
            yield Label("▌", classes="streaming-indicator", id="cursor")

    def on_mount(self) -> None:
        """Hide cursor initially."""
        cursor = self.query_one("#cursor")
        cursor.display = self.is_streaming

    def watch_content(self, content: str) -> None:
        """React to content changes."""
        self._update_display()

    def watch_is_streaming(self, streaming: bool) -> None:
        """Show/hide streaming cursor."""
        try:
            cursor = self.query_one("#cursor")
            cursor.display = streaming
        except Exception:
            pass

    def _update_display(self) -> None:
        """Update the message body with current content."""
        try:
            body = self.query_one("#message-body", Static)
            if self.role == "assistant":
                # Parse and render markdown, extracting code blocks
                body.update(Markdown(self.content))
            else:
                body.update(self.content)
        except Exception:
            pass

    def append_chunk(self, chunk: str) -> None:
        """Append a streaming chunk to the content."""
        self.content += chunk

    def finish_streaming(self) -> None:
        """Mark streaming as complete."""
        self.is_streaming = False


class ToolProgressPanel(Static):
    """Panel showing real-time tool execution progress.

    Features:
    - Progress bar for long-running tools
    - Expandable output preview
    - Success/failure status with elapsed time
    - Cancel button for interruptible tools
    """

    DEFAULT_CSS = """
    ToolProgressPanel {
        background: $surface;
        border: solid $primary-darken-2;
        padding: 1;
        margin: 1 0;
        height: auto;
        max-height: 12;
    }

    ToolProgressPanel .tool-header {
        text-style: bold;
    }

    ToolProgressPanel .tool-name {
        color: $secondary;
    }

    ToolProgressPanel .tool-status {
        margin-left: 1;
    }

    ToolProgressPanel .tool-status.pending {
        color: $warning;
    }

    ToolProgressPanel .tool-status.running {
        color: $primary;
    }

    ToolProgressPanel .tool-status.success {
        color: $success;
    }

    ToolProgressPanel .tool-status.error {
        color: $error;
    }

    ToolProgressPanel .progress-container {
        margin: 1 0;
    }

    ToolProgressPanel .output-preview {
        color: $text-muted;
        max-height: 4;
        overflow: hidden;
    }

    ToolProgressPanel .elapsed-time {
        color: $text-muted;
        text-style: italic;
    }
    """

    status = reactive("pending")
    progress = reactive(0.0)
    elapsed = reactive(0.0)

    def __init__(
        self,
        tool_name: str,
        arguments: Optional[dict] = None,
        on_cancel: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.arguments = arguments or {}
        self.on_cancel = on_cancel
        self._start_time = time.time()
        self._output_preview = ""

    def compose(self) -> ComposeResult:
        args_preview = self._format_args_preview()

        with Vertical():
            with Horizontal(classes="tool-header-row"):
                yield Label(
                    Text.assemble(
                        ("⚙ ", ""),
                        (self.tool_name, "bold cyan"),
                        (f" {args_preview}" if args_preview else "", "dim"),
                    ),
                    classes="tool-header",
                )
                yield Label("⏳", classes="tool-status pending", id="status-icon")
                yield Label("", classes="elapsed-time", id="elapsed")

            yield ProgressBar(total=100, show_eta=False, id="progress-bar")

            yield Static("", classes="output-preview", id="output")

            if self.on_cancel:
                yield Button("Cancel", variant="error", id="cancel-btn")

    def _format_args_preview(self) -> str:
        """Format arguments as a preview string."""
        if not self.arguments:
            return ""
        first_key = next(iter(self.arguments.keys()), None)
        if first_key:
            first_val = str(self.arguments[first_key])[:40]
            if len(str(self.arguments[first_key])) > 40:
                first_val += "..."
            return f"({first_key}={first_val})"
        return ""

    def watch_status(self, status: str) -> None:
        """Update status icon when status changes."""
        try:
            icon = self.query_one("#status-icon")
            icons = {
                "pending": ("⏳", "pending"),
                "running": ("▶", "running"),
                "success": ("✓", "success"),
                "error": ("✗", "error"),
            }
            symbol, css_class = icons.get(status, ("?", "pending"))
            icon.update(symbol)
            icon.remove_class("pending", "running", "success", "error")
            icon.add_class(css_class)
        except Exception:
            pass

    def watch_elapsed(self, elapsed: float) -> None:
        """Update elapsed time display."""
        try:
            elapsed_label = self.query_one("#elapsed")
            elapsed_label.update(f"{elapsed:.1f}s")
        except Exception:
            pass

    def watch_progress(self, progress: float) -> None:
        """Update progress bar."""
        try:
            bar = self.query_one("#progress-bar", ProgressBar)
            bar.update(progress=progress)
        except Exception:
            pass

    def update_output_preview(self, text: str) -> None:
        """Update the output preview text."""
        self._output_preview = text[:200]  # Truncate
        try:
            output = self.query_one("#output")
            output.update(self._output_preview)
        except Exception:
            pass

    def set_running(self) -> None:
        """Mark tool as running."""
        self.status = "running"
        self._start_time = time.time()

    def set_complete(self, success: bool, output: str = "") -> None:
        """Mark tool as complete."""
        self.status = "success" if success else "error"
        self.elapsed = time.time() - self._start_time
        self.progress = 100.0
        if output:
            self.update_output_preview(output)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle cancel button press."""
        if event.button.id == "cancel-btn" and self.on_cancel:
            self.on_cancel()
            self.status = "error"


class EnhancedConversationLog(VerticalScroll):
    """Enhanced conversation log with streaming support.

    Replaces RichLog-based ConversationLog with a ScrollableContainer
    that supports:
    - Live streaming updates
    - Message widgets with proper styling
    - Code blocks with syntax highlighting
    - Auto-scroll to bottom
    """

    DEFAULT_CSS = """
    EnhancedConversationLog {
        background: $background;
        padding: 1;
        scrollbar-gutter: stable;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._streaming_message: Optional[StreamingMessageBlock] = None
        self._message_count = 0
        self._auto_scroll = True

    @property
    def auto_scroll_enabled(self) -> bool:
        return self._auto_scroll

    def add_user_message(self, content: str) -> None:
        """Add a user message to the log."""
        msg = StreamingMessageBlock(
            role="user",
            initial_content=content,
            id=f"msg-{self._message_count}",
        )
        self._message_count += 1
        self.mount(msg)
        self._maybe_scroll_end()

    def add_assistant_message(self, content: str) -> None:
        """Add a complete assistant message to the log."""
        msg = StreamingMessageBlock(
            role="assistant",
            initial_content=content,
            id=f"msg-{self._message_count}",
        )
        self._message_count += 1
        self.mount(msg)
        self._maybe_scroll_end()

    def add_system_message(self, content: str) -> None:
        """Add a system/status message to the log."""
        msg = Static(
            Text(f"[{content}]", style="dim italic"),
            id=f"msg-{self._message_count}",
        )
        self._message_count += 1
        self.mount(msg)
        self._maybe_scroll_end()

    def add_error_message(self, content: str) -> None:
        """Add an error message to the log."""
        msg = Static(
            Text(f"Error: {content}", style="bold red"),
            id=f"msg-{self._message_count}",
        )
        self._message_count += 1
        self.mount(msg)
        self._maybe_scroll_end()

    def start_streaming(self) -> StreamingMessageBlock:
        """Start a streaming response and return the message block."""
        self._streaming_message = StreamingMessageBlock(
            role="assistant",
            initial_content="",
            id=f"msg-{self._message_count}",
        )
        self._streaming_message.is_streaming = True
        self._message_count += 1
        self.mount(self._streaming_message)
        self._maybe_scroll_end()
        return self._streaming_message

    def update_streaming(self, content: str) -> None:
        """Update the current streaming message."""
        if self._streaming_message:
            self._streaming_message.content = content
            self._maybe_scroll_end()

    def append_streaming_chunk(self, chunk: str) -> None:
        """Append a chunk to the current streaming message."""
        if self._streaming_message:
            self._streaming_message.append_chunk(chunk)
            self._maybe_scroll_end()

    def finish_streaming(self) -> None:
        """Finish the current streaming response."""
        if self._streaming_message:
            self._streaming_message.finish_streaming()
            self._streaming_message = None

    def add_tool_progress(
        self,
        tool_name: str,
        arguments: Optional[dict] = None,
        on_cancel: Optional[Callable] = None,
    ) -> ToolProgressPanel:
        """Add a tool progress panel and return it for updates."""
        panel = ToolProgressPanel(
            tool_name=tool_name,
            arguments=arguments,
            on_cancel=on_cancel,
            id=f"tool-{self._message_count}",
        )
        self._message_count += 1
        self.mount(panel)
        self._maybe_scroll_end()
        return panel

    def add_code_block(
        self,
        code: str,
        language: str = "python",
    ) -> None:
        """Add a standalone code block."""
        block = CodeBlock(
            code=code,
            language=language,
            id=f"code-{self._message_count}",
        )
        self._message_count += 1
        self.mount(block)
        self._maybe_scroll_end()

    def clear(self) -> None:
        """Clear all messages from the log."""
        for child in list(self.children):
            child.remove()
        self._message_count = 0
        self._streaming_message = None
        self._auto_scroll = True

    def on_scroll(self, _event) -> None:
        """Update auto-scroll when the user scrolls."""
        self.update_auto_scroll_state()

    def scroll_to_bottom(self, animate: bool = False) -> None:
        """Scroll to bottom and re-enable auto-scroll."""
        self._auto_scroll = True
        self.scroll_end(animate=animate)

    def disable_auto_scroll(self) -> None:
        """Disable auto-scroll until user returns to bottom."""
        self._auto_scroll = False

    def update_auto_scroll_state(self) -> None:
        """Update auto-scroll based on scroll position."""
        self._auto_scroll = self._is_at_bottom()

    def _maybe_scroll_end(self) -> None:
        if self._auto_scroll:
            self.scroll_end(animate=False)

    def _is_at_bottom(self) -> bool:
        try:
            return self.scroll_y >= self.max_scroll_y - 1
        except Exception:
            return True
