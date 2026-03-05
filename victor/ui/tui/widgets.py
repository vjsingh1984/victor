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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text
from victor.ui.rendering.markdown import render_markdown_with_hooks
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual import events
from textual.widgets import Label, Static, RichLog, TextArea, Button, ProgressBar
from textual.message import Message
from textual.messages import UpdateScroll
from textual.reactive import reactive
from textual.binding import Binding
from textual.widget import Widget

if TYPE_CHECKING:
    pass


@dataclass
class ToolPreviewData:
    """Structured preview data for tool outputs."""

    tool_name: str
    preview_type: str
    path: Optional[str] = None
    snippet: Optional[str] = None
    content: Optional[str] = None
    diff: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolPreviewBlock(Static):
    """Collapsible tool preview block for transcript display."""

    DEFAULT_CSS = """
    ToolPreviewBlock {
        border: round $border-muted;
        background: $panel;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    ToolPreviewBlock.collapsed #preview-body {
        display: none;
    }

    ToolPreviewBlock .preview-header {
        text-style: bold;
        color: $primary;
    }

    ToolPreviewBlock .preview-meta {
        color: $text-muted;
    }
    """

    BINDINGS = [("enter", "toggle", "Toggle preview")]

    def __init__(self, preview: ToolPreviewData, **kwargs) -> None:
        super().__init__(**kwargs)
        self.preview = preview
        self._collapsed = True
        self._body_widget: Optional[Static] = None

    def compose(self) -> ComposeResult:
        header_text = f"{self.preview.tool_name} · {self.preview.preview_type.title()}"
        if self.preview.path:
            header_text += f" · {self.preview.path}"
        yield Label(header_text, classes="preview-header")
        if self.preview.metadata:
            meta_parts = [
                f"{key}: {value}"
                for key, value in self.preview.metadata.items()
                if value is not None
            ]
            if meta_parts:
                yield Label(" · ".join(meta_parts), classes="preview-meta")
        yield Label("Press Enter or click to expand/collapse", classes="preview-meta")
        yield Static("", id="preview-body")

    def on_mount(self) -> None:
        self._body_widget = self.query_one("#preview-body", Static)
        self._render_body()

    def _render_body(self) -> None:
        if not self._body_widget:
            return
        if self._collapsed:
            text = self.preview.snippet or "(preview hidden)"
        else:
            if self.preview.preview_type == "diff" and self.preview.diff:
                self._body_widget.update(
                    Syntax(self.preview.diff, "diff", theme="monokai", word_wrap=True)
                )
                return
            text = self.preview.content or self.preview.snippet or "(no content)"
        self._body_widget.update(text)

    def action_toggle(self) -> None:
        self._collapsed = not self._collapsed
        if self._collapsed:
            self.add_class("collapsed")
        else:
            self.remove_class("collapsed")
        self._render_body()

    def on_click(self) -> None:
        self.action_toggle()


class ConversationTurn(Vertical):
    """Container representing a user/assistant exchange."""

    DEFAULT_CSS = """
    ConversationTurn {
        padding: 0 0 1 0;
        margin: 0;
    }

    ConversationTurn .turn-user {
        color: $primary;
        margin-bottom: 0;
    }

    ConversationTurn .turn-assistant {
        margin-top: 0;
    }

    ConversationTurn .turn-tools {
        margin-top: 0;
    }
    """

    def __init__(self, turn_id: str):
        super().__init__(id=turn_id, classes="conversation-turn")
        self._assistant_block: Optional[StreamingMessageBlock] = None
        self._tool_container: Optional[Vertical] = None
        self._pending_user_content: Optional[str] = None
        self._has_pending_user_content = False
        self._pending_tool_previews: list[ToolPreviewData] = []
        self._assistant_started = False

    def compose(self) -> ComposeResult:
        yield Label("", classes="turn-user", id="turn-user")
        yield Container(id="assistant-area")
        yield Vertical(id="turn-tools", classes="turn-tools")

    def on_mount(self) -> None:
        """Apply deferred content once child widgets exist."""
        if self._has_pending_user_content:
            self.set_user_message(self._pending_user_content)
        if self._assistant_block is not None and not self._assistant_block.is_mounted:
            self.ensure_assistant_block()
        if self._pending_tool_previews:
            pending = list(self._pending_tool_previews)
            self._pending_tool_previews.clear()
            for preview in pending:
                self.add_tool_preview(preview)

    def _apply_user_message_to_label(self, label: Label, content: Optional[str]) -> None:
        if content:
            label.update(
                Text.assemble(
                    ("You: ", "bold #64b5f6"),
                    (content, ""),
                )
            )
            label.display = True
        else:
            label.display = False

    def set_user_message(self, content: Optional[str]) -> None:
        self._pending_user_content = content
        self._has_pending_user_content = True
        try:
            label = self.query_one("#turn-user", Label)
        except Exception:
            # Turn may be updated before compose/mount completes.
            return
        self._apply_user_message_to_label(label, content)

    def ensure_assistant_block(self) -> StreamingMessageBlock:
        if self._assistant_block is None:
            self._assistant_block = StreamingMessageBlock(role="assistant", initial_content="")
            self._assistant_block.add_class("turn-assistant")
        if self._assistant_block.is_mounted:
            return self._assistant_block
        # In pre-mount/testing paths we may not yet be attached to the DOM.
        if not self.is_attached:
            return self._assistant_block
        try:
            container = self.query_one("#assistant-area", Container)
            container.mount(self._assistant_block)
        except Exception:
            try:
                self.mount(self._assistant_block)
            except Exception:
                pass
        return self._assistant_block

    def start_assistant_stream(self) -> StreamingMessageBlock:
        block = self.ensure_assistant_block()
        self._assistant_started = True
        block.is_streaming = True
        block._update_display()
        return block

    def set_assistant_message(self, content: str) -> None:
        block = self.ensure_assistant_block()
        self._assistant_started = True
        block.is_streaming = False
        block.content = content
        block._update_display()

    def has_assistant_output(self) -> bool:
        return self._assistant_started

    def add_tool_preview(self, preview: ToolPreviewData) -> None:
        if self._tool_container is None:
            try:
                self._tool_container = self.query_one("#turn-tools", Vertical)
            except Exception:
                if not self.is_attached:
                    self._pending_tool_previews.append(preview)
                    return
                self._tool_container = Vertical(classes="turn-tools")
                try:
                    self.mount(self._tool_container)
                except Exception:
                    self._pending_tool_previews.append(preview)
                    return
        block = ToolPreviewBlock(preview=preview)
        if not self._tool_container.is_attached:
            self._pending_tool_previews.append(preview)
            return
        self._tool_container.mount(block)


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
    helpful keyboard shortcuts that update based on context.
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
        self._state = "idle"  # idle, streaming, error
        self._follow_paused = False
        self._unread_count = 0

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
            follow_label = Label("Following", classes="follow-indicator")
            follow_label.add_class("following")
            yield follow_label
            yield Label("", classes="unread-indicator", id="unread-indicator")
            yield Label(
                Text.assemble(
                    ("Ctrl+C", "bold"),
                    " exit  ",
                    ("Enter", "bold"),
                    " send",
                ),
                classes="shortcuts",
                id="shortcut-hints",
            )

    def _update_shortcuts_idle(self) -> None:
        """Update shortcuts for idle state."""
        try:
            hints = self.query_one("#shortcut-hints", Label)
            parts: list[object] = [
                ("Ctrl+L", "bold"),
                " clear  ",
                ("Ctrl+S", "bold"),
                " save  ",
                ("Ctrl+F", "bold"),
                f" {self._follow_action_text()}  ",
            ]
            if self._follow_paused:
                parts.extend(
                    [
                        ("Ctrl+End", "bold"),
                        " latest  ",
                    ]
                )
            parts.extend([("Enter", "bold"), " send"])
            hints.update(Text.assemble(*parts))
        except Exception:
            pass

    def _update_shortcuts_streaming(self) -> None:
        """Update shortcuts for streaming state."""
        try:
            hints = self.query_one("#shortcut-hints", Label)
            parts: list[object] = [
                ("Ctrl+X", "bold"),
                " cancel  ",
                ("Ctrl+F", "bold"),
                f" {self._follow_action_text()}  ",
            ]
            if self._follow_paused:
                parts.extend(
                    [
                        ("Ctrl+End", "bold"),
                        " latest  ",
                    ]
                )
            parts.extend([("Ctrl+D", "bold"), " toggle details"])
            hints.update(Text.assemble(*parts))
        except Exception:
            pass

    def _update_shortcuts_error(self) -> None:
        """Update shortcuts for error state."""
        try:
            hints = self.query_one("#shortcut-hints", Label)
            parts: list[object] = [
                ("Ctrl+L", "bold"),
                " clear  ",
                ("Ctrl+F", "bold"),
                f" {self._follow_action_text()}  ",
            ]
            if self._follow_paused:
                parts.extend(
                    [
                        ("Ctrl+End", "bold"),
                        " latest  ",
                    ]
                )
            parts.extend([("Ctrl+C", "bold"), " exit"])
            hints.update(Text.assemble(*parts))
        except Exception:
            pass

    def _follow_action_text(self) -> str:
        """Return the current follow shortcut action text."""
        return "resume follow" if self._follow_paused else "pause follow"

    def update_shortcuts(self, state: str) -> None:
        """Update shortcut hints based on application state.

        Args:
            state: One of 'idle', 'streaming', 'error'
        """
        self._state = state
        if state == "streaming":
            self._update_shortcuts_streaming()
        elif state == "error":
            self._update_shortcuts_error()
        else:  # idle
            self._update_shortcuts_idle()

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
        """Update status text, styling, and dynamic shortcut hints.

        Args:
            status: Status text to display (e.g., "Idle", "Streaming", "Error")
            state: State for styling and shortcuts ('idle', 'streaming', 'error', 'busy')
        """
        self.status = status
        label = self.query_one(".status-indicator", Label)
        label.update(self.status)
        label.remove_class("idle", "busy", "streaming")
        label.add_class(state)
        # Update shortcut hints based on state
        self.update_shortcuts(state)

    def update_follow(self, paused: bool) -> None:
        """Update follow-mode indicator."""
        if paused == self._follow_paused:
            return
        self._follow_paused = paused
        label = self.query_one(".follow-indicator", Label)
        label.update("Paused" if paused else "Following")
        label.remove_class("following", "paused")
        label.add_class("paused" if paused else "following")
        self.update_shortcuts(self._state)

    def update_unread(self, unread_count: int) -> None:
        """Update unread counter badge."""
        if unread_count == self._unread_count:
            return
        self._unread_count = unread_count
        label = self.query_one("#unread-indicator", Label)
        if unread_count > 0:
            label.update(f"{unread_count} new")
            label.add_class("visible")
        else:
            label.update("")
            label.remove_class("visible")


class SubmitTextArea(TextArea):
    """Custom TextArea that submits on Enter and allows Shift+Enter for newlines.

    This provides a Claude Code-like multi-line input experience:
    - Enter: Submit message (if not at end of line with text)
    - Shift+Enter: Always insert newline
    - Ctrl+Enter: Always submit
    - Proper multi-line editing with word wrap
    """

    class Submit(Message):
        """Message emitted when Enter is pressed."""

        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    async def _on_key(self, event: events.Key) -> None:
        """Handle key events before TextArea processes them."""
        # Ctrl+Enter: Always submit
        if event.key == "ctrl+enter":
            if self.text.strip():
                self.post_message(self.Submit(self.text))
            event.prevent_default()
            event.stop()
            return

        # Check for plain Enter (not shift+enter)
        if event.key == "enter":
            # Check cursor position - if in middle of line, insert newline
            # If at end of line, submit
            cursor_location = self.cursor_location
            text = self.text

            # cursor_location is a tuple (line, column)
            line_idx, col_idx = cursor_location

            # Get the current line text using the document
            lines = text.splitlines()
            current_line = lines[line_idx] if line_idx < len(lines) else ""
            is_at_end_of_line = col_idx >= len(current_line)

            # If at end of line and there's text, submit
            # Otherwise insert newline
            if is_at_end_of_line and text.strip():
                self.post_message(self.Submit(self.text))
                event.prevent_default()
                event.stop()
                return

        # Shift+Enter and all other keys handled normally by parent
        await super()._on_key(event)


class InputWidget(Static):
    """Input area at the bottom of the screen.

    Uses a multi-line TextArea for better input handling.
    `Enter` sends the message, `Shift+Enter` adds a newline.
    Up/Down arrows navigate input history loaded from conversation DB.
    """

    DEFAULT_CSS = ""
    _IDLE_HINT = "Enter send | Shift+Enter newline | ↑/↓ history | /help commands"
    _BUSY_HINT = "Sending... | Ctrl+X cancel current response"

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
        self._prompt_label: Label | None = None
        self._hint_label: Label | None = None
        self._history_index: int = -1  # -1 means not browsing history
        self._draft: str = ""  # Save current draft when browsing history
        self._history_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(classes="input-row"):
                yield Label("❯", classes="prompt-indicator", id="prompt-indicator")
                yield SubmitTextArea(
                    placeholder="Enter your message here...",
                    id="message-input",
                )
            yield Label(
                self._IDLE_HINT,
                classes="input-hint",
                id="input-hint",
            )

    def on_submit_text_area_submit(self, event: SubmitTextArea.Submit) -> None:
        """Handle submit from the custom TextArea."""
        if event.value.strip():
            self.post_message(self.Submitted(event.value))

    def on_mount(self) -> None:
        """Focus the input on mount and defer history loading."""
        self._input = self.query_one("#message-input", SubmitTextArea)
        self._prompt_label = self.query_one("#prompt-indicator", Label)
        self._hint_label = self.query_one("#input-hint", Label)
        self._input.focus()
        self.set_busy(False)

        # Defer history loading until after TUI has rendered (non-blocking)
        # Load persistent history from conversation database (once per session)
        if not InputWidget._history_loaded:
            InputWidget._history_loaded = True
            # Use a short timer to defer loading, allowing TUI to render first
            # Widget has access to app via self.app when mounted
            if hasattr(self, "app") and self.app is not None:
                self.app.set_timer(0.1, self._start_history_load)
            else:
                # Fallback: load immediately if no app context
                self._history_task = asyncio.create_task(self._load_history_async())

    def _start_history_load(self) -> None:
        """Start async history loading."""
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

    def set_busy(self, busy: bool) -> None:
        """Set input busy/idle visuals while processing a prompt."""
        if self._input:
            self._input.disabled = busy
        if self._prompt_label:
            self._prompt_label.update("⋯" if busy else "❯")
            self._prompt_label.remove_class("busy")
            if busy:
                self._prompt_label.add_class("busy")
        if self._hint_label:
            self._hint_label.update(self._BUSY_HINT if busy else self._IDLE_HINT)

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
        background: transparent;
        border: none;
        padding: 0;
        margin: 0;
    }

    StreamingMessageBlock.user {
        background: transparent;
    }

    StreamingMessageBlock.assistant {
        background: transparent;
    }

    StreamingMessageBlock .message-header {
        text-style: bold;
        margin-bottom: 0;
    }

    StreamingMessageBlock .message-header.user {
        color: $success;
    }

    StreamingMessageBlock .message-header.assistant {
        color: $primary;
    }

    StreamingMessageBlock .message-content {
        width: 100%;
        height: auto;
        max-height: 10;
        overflow-x: hidden;
        overflow-y: auto;
    }

    StreamingMessageBlock .streaming-indicator {
        color: $text-muted;
        text-style: italic;
    }
    """

    content = reactive("")
    is_streaming = reactive(False)
    _STREAM_RENDER_INTERVAL_SECONDS = 1 / 30

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
        self._last_stream_render_ts = 0.0
        self._pending_stream_buffer = ""
        self._body_widget: Static | None = None
        self._cursor_widget: Label | None = None

    def compose(self) -> ComposeResult:
        role_label = {
            "user": "You",
            "assistant": "Victor",
            "system": "System",
            "error": "Error",
        }.get(self.role, self.role.title())

        with Vertical():
            if self.role != "user":
                with Horizontal(classes="header-row"):
                    header = Label(role_label, classes="message-header")
                    header.add_class(self.role)
                    yield header
                    if self.role == "assistant":
                        yield Button("Copy", id="copy-btn", variant="default")
            yield Static(id="message-body", classes="message-content")
            if self.role == "assistant":
                yield Label("▌", classes="streaming-indicator", id="cursor")

    def on_mount(self) -> None:
        """Hide cursor initially."""
        self._body_widget = self.query_one("#message-body", Static)
        if self.role == "assistant":
            try:
                self._cursor_widget = self.query_one("#cursor", Label)
                self._cursor_widget.display = self.is_streaming
            except Exception:
                self._cursor_widget = None
        else:
            self._cursor_widget = None

    def watch_content(self, content: str) -> None:
        """React to content changes."""
        self._update_display()

    def watch_is_streaming(self, streaming: bool) -> None:
        """Show/hide streaming cursor."""
        if self.role != "assistant":
            return
        try:
            if self._cursor_widget is None:
                self._cursor_widget = self.query_one("#cursor", Label)
            self._cursor_widget.display = streaming
        except Exception:
            pass
        if streaming:
            self._last_stream_render_ts = 0.0
        self._update_display()

    def _update_display(self) -> None:
        """Update the message body with current content."""
        try:
            if self._body_widget is None:
                self._body_widget = self.query_one("#message-body", Static)
            body = self._body_widget
            if self.role == "assistant":
                # Render plain text while streaming for responsiveness,
                # then re-render with markdown hooks when complete.
                if self.is_streaming:
                    body.update(self.content)
                else:
                    body.update(render_markdown_with_hooks(self.content))
            elif self.role == "user":
                text = Text.assemble(
                    ("You: ", "bold " + "#64b5f6"),
                    (self.content, ""),
                )
                body.update(text)
            else:
                body.update(self.content)
        except Exception:
            pass

    def append_chunk(self, chunk: str) -> None:
        """Append a streaming chunk to the content."""
        if not chunk:
            return
        self._pending_stream_buffer += chunk
        if not self.is_streaming:
            self._flush_stream_buffer()
            return

        now = time.monotonic()
        if (
            not self._last_stream_render_ts
            or now - self._last_stream_render_ts >= self._STREAM_RENDER_INTERVAL_SECONDS
        ):
            self._flush_stream_buffer(now)

    def finish_streaming(self) -> None:
        """Mark streaming as complete."""
        self._flush_stream_buffer()
        self.is_streaming = False
        self._update_display()

    def _flush_stream_buffer(self, timestamp: float | None = None) -> None:
        if not self._pending_stream_buffer:
            return
        self.content += self._pending_stream_buffer
        self._pending_stream_buffer = ""
        self._last_stream_render_ts = timestamp or time.monotonic()


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

    Provides a Claude Code-like message display:
    - Messages flow naturally one after another
    - Auto-scrolls to show new content
    - Smooth scrolling through history
    - Proper message separation and styling
    """

    DEFAULT_CSS = """
    EnhancedConversationLog {
        background: $background;
        padding: 0;
        scrollbar-gutter: stable;
    }

    EnhancedConversationLog > Static {
        margin: 0;
    }

    EnhancedConversationLog > StreamingMessageBlock {
        margin: 0;
    }

    EnhancedConversationLog .unread-separator {
        margin: 0 0 1 0;
        color: $warning;
        text-style: bold;
    }
    """

    _FOLLOW_SCROLL_INTERVAL_SECONDS = 1 / 30
    _PROGRAMMATIC_SCROLL_GUARD_SECONDS = 0.08
    _RESIZE_SCROLL_GUARD_SECONDS = 0.2

    def __init__(self, show_unread_separator: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self._streaming_message: Optional[StreamingMessageBlock] = None
        self._message_count = 0
        self._auto_scroll = True
        self._user_scrolled = False
        self._sticky_follow_paused = False
        self._last_follow_scroll_ts = 0.0
        self._ignore_scroll_update_until = 0.0
        self._ignore_resize_scroll_update_until = 0.0
        self._unread_count = 0
        self._show_unread_separator = show_unread_separator
        self._unread_separator: Optional[Static] = None
        self._unread_boundary_id: Optional[str] = None
        self._turn_count = 0
        self._current_turn: Optional[ConversationTurn] = None

    @property
    def auto_scroll_enabled(self) -> bool:
        return self._auto_scroll

    @property
    def unread_count(self) -> int:
        return self._unread_count

    @property
    def follow_paused(self) -> bool:
        return self._sticky_follow_paused

    @property
    def unread_separator_enabled(self) -> bool:
        return self._show_unread_separator

    def set_unread_separator_enabled(self, enabled: bool) -> None:
        """Enable or disable unread separator marker."""
        self._show_unread_separator = enabled
        if not enabled:
            self._remove_unread_separator()
            return
        if not self._auto_scroll and self._unread_count > 0:
            self._ensure_unread_separator()

    def jump_to_unread_separator(self) -> bool:
        """Scroll the transcript to unread boundary (marker or first unread message)."""
        target: Optional[Widget] = self._unread_separator or self._get_unread_boundary_target()
        if target is None:
            return False
        try:
            self.scroll_to_widget(
                target,
                animate=False,
                top=True,
                immediate=True,
                force=True,
            )
            return True
        except Exception:
            return False

    def set_follow_paused(self, paused: bool, *, jump_to_bottom: bool = False) -> None:
        """Set sticky follow mode.

        When paused, auto-follow stays disabled until explicitly resumed.
        """
        self._sticky_follow_paused = paused
        if paused:
            self._auto_scroll = False
            return
        self._auto_scroll = True
        self._last_follow_scroll_ts = 0.0
        self._unread_count = 0
        self._unread_boundary_id = None
        self._remove_unread_separator()
        if jump_to_bottom:
            self._scroll_end_with_guard(animate=False)

    def add_user_message(self, content: str) -> None:
        """Add a user message to the log."""
        self._mark_unread_boundary()
        self._begin_new_turn(content)
        if self._sticky_follow_paused:
            self._auto_scroll = False
            return
        # Always scroll to show user message
        self._auto_scroll = True
        self._unread_count = 0
        self._unread_boundary_id = None
        self._remove_unread_separator()
        self._scroll_end_with_guard(animate=False)

    def add_assistant_message(self, content: str) -> None:
        """Add a complete assistant message to the log."""
        self._mark_unread_boundary()
        turn = self._current_turn
        if turn is None or turn.has_assistant_output():
            turn = self._begin_new_turn(None)
        turn.set_assistant_message(content)
        # Auto-scroll to show assistant response
        self._maybe_scroll_end(mark_unread=True, force=True)

    def add_system_message(self, content: str) -> None:
        """Add a system/status message to the log."""
        self._mark_unread_boundary()
        msg = Static(
            Text(f"[{content}]", style="dim italic"),
            id=f"msg-{self._message_count}",
        )
        self._message_count += 1
        self.mount(msg)
        # Follow transcript if enabled; otherwise count as unread activity.
        self._maybe_scroll_end(mark_unread=True)

    def add_error_message(self, content: str) -> None:
        """Add an error message to the log."""
        self._mark_unread_boundary()
        msg = Static(
            Text(f"Error: {content}", style="bold red"),
            id=f"msg-{self._message_count}",
        )
        self._message_count += 1
        self.mount(msg)
        self._maybe_scroll_end(mark_unread=True, force=True)

    def add_tool_preview(self, preview: ToolPreviewData) -> None:
        """Add a tool preview block to the log."""
        target = self._current_turn
        if target:
            target.add_tool_preview(preview)
            self._maybe_scroll_end(mark_unread=True, force=True)
            return
        self._mark_unread_boundary()
        block = ToolPreviewBlock(preview=preview)
        self._message_count += 1
        self.mount(block)
        self._maybe_scroll_end(mark_unread=True, force=True)

    def _begin_new_turn(self, user_content: Optional[str]) -> ConversationTurn:
        turn = ConversationTurn(turn_id=f"turn-{self._turn_count}")
        self._turn_count += 1
        self._message_count += 1
        self.mount(turn)
        turn.set_user_message(user_content)
        self._current_turn = turn
        return turn

    def start_streaming(self) -> StreamingMessageBlock:
        """Start a streaming response and return the message block."""
        self._mark_unread_boundary()
        turn = self._current_turn
        if turn is None or turn.has_assistant_output():
            turn = self._begin_new_turn(None)
        self._streaming_message = turn.start_assistant_stream()
        self._maybe_scroll_end(mark_unread=True, force=True)
        return self._streaming_message

    def update_streaming(self, content: str) -> None:
        """Update the current streaming message.

        Shows new content as it arrives, like Claude Code.
        Auto-scrolls only when user is already at bottom.
        """
        if self._streaming_message:
            self._streaming_message.content = content
            self._maybe_scroll_end()

    def append_streaming_chunk(self, chunk: str) -> None:
        """Append a chunk to the current streaming message.

        Shows new content as it arrives, like Claude Code.
        """
        if self._streaming_message:
            self._streaming_message.append_chunk(chunk)
            self._maybe_scroll_end()

    def finish_streaming(self) -> None:
        """Finish the current streaming response."""
        if self._streaming_message:
            self._streaming_message.finish_streaming()
            self._streaming_message = None
        self._maybe_scroll_end(force=True)

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
        self._maybe_scroll_end(force=True)
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
        self._maybe_scroll_end(force=True)

    def add_history_message(self, role: str, content: str) -> None:
        """Replay an existing message without triggering follow/unread side effects."""
        widget: Widget | None = None
        if role == "user":
            widget = StreamingMessageBlock(
                role="user",
                initial_content=content,
                id=f"msg-{self._message_count}",
            )
        elif role == "assistant":
            widget = StreamingMessageBlock(
                role="assistant",
                initial_content=content,
                id=f"msg-{self._message_count}",
            )
        elif role == "system":
            widget = Static(
                Text(f"[{content}]", style="dim italic"),
                id=f"msg-{self._message_count}",
            )
        elif role == "error":
            widget = Static(
                Text(f"Error: {content}", style="bold red"),
                id=f"msg-{self._message_count}",
            )
        if widget is None:
            return
        self._message_count += 1
        self.mount(widget)

    def clear(self) -> None:
        """Clear all messages from the log."""
        for child in list(self.children):
            child.remove()
        self._message_count = 0
        self._streaming_message = None
        self._auto_scroll = not self._sticky_follow_paused
        self._last_follow_scroll_ts = 0.0
        self._ignore_scroll_update_until = 0.0
        self._ignore_resize_scroll_update_until = 0.0
        self._unread_count = 0
        self._unread_separator = None
        self._unread_boundary_id = None

    def on_update_scroll(self, _event: UpdateScroll) -> None:
        """Update auto-scroll whenever this scroll view moves."""
        now = time.monotonic()
        if now < self._ignore_resize_scroll_update_until:
            return
        if now < self._ignore_scroll_update_until and self._auto_scroll:
            # Ignore transient programmatic follow-scroll events while still at bottom.
            # If user has already scrolled away from bottom, process immediately.
            if self._is_at_bottom():
                return
        self.update_auto_scroll_state()

    def on_resize(self, _event: events.Resize) -> None:
        """Preserve paused/off-bottom state across terminal resize events."""
        if self._auto_scroll:
            return
        guard_until = time.monotonic() + self._RESIZE_SCROLL_GUARD_SECONDS
        if guard_until > self._ignore_resize_scroll_update_until:
            self._ignore_resize_scroll_update_until = guard_until

    def scroll_to_bottom(self, animate: bool = False) -> None:
        """Scroll to bottom and re-enable auto-scroll."""
        self._auto_scroll = not self._sticky_follow_paused
        self._last_follow_scroll_ts = 0.0
        self._ignore_scroll_update_until = 0.0
        self._ignore_resize_scroll_update_until = 0.0
        self._unread_count = 0
        self._unread_boundary_id = None
        self._remove_unread_separator()
        self._scroll_end_with_guard(animate=animate)

    def disable_auto_scroll(self) -> None:
        """Disable auto-scroll until user returns to bottom."""
        self._auto_scroll = False

    def update_auto_scroll_state(self) -> None:
        """Update auto-scroll based on scroll position."""
        if self._sticky_follow_paused:
            if self._is_at_bottom():
                self._unread_count = 0
                self._unread_boundary_id = None
                self._remove_unread_separator()
            self._auto_scroll = False
            return
        self._auto_scroll = self._is_at_bottom()
        if self._auto_scroll:
            self._unread_count = 0
            self._unread_boundary_id = None
            self._remove_unread_separator()

    def _maybe_scroll_end(self, mark_unread: bool = False, force: bool = False) -> None:
        if self._auto_scroll:
            if not force:
                now = time.monotonic()
                if (
                    self._last_follow_scroll_ts
                    and now - self._last_follow_scroll_ts < self._FOLLOW_SCROLL_INTERVAL_SECONDS
                ):
                    return
                self._last_follow_scroll_ts = now
                self._scroll_end_with_guard(animate=False, now=now)
            else:
                now = time.monotonic()
                self._last_follow_scroll_ts = now
                self._scroll_end_with_guard(animate=False, now=now)
            return
        if mark_unread:
            self._unread_count += 1
            self._refresh_unread_separator_label()

    def _scroll_end_with_guard(self, animate: bool = False, now: float | None = None) -> None:
        timestamp = now if now is not None else time.monotonic()
        guard_until = timestamp + self._PROGRAMMATIC_SCROLL_GUARD_SECONDS
        if guard_until > self._ignore_scroll_update_until:
            self._ignore_scroll_update_until = guard_until
        self.scroll_end(animate=animate)

    def _mark_unread_boundary(self) -> None:
        """Insert unread separator before first unread message."""
        if self._auto_scroll or self._unread_count > 0:
            return
        self._unread_boundary_id = f"msg-{self._message_count}"
        if self._show_unread_separator:
            self._ensure_unread_separator()

    def _ensure_unread_separator(self) -> None:
        if self._unread_separator is not None:
            self._refresh_unread_separator_label()
            return
        separator = Static(
            self._build_unread_separator_text(),
            classes="unread-separator",
        )
        self._unread_separator = separator
        target = self._get_unread_boundary_target()
        try:
            if target is not None:
                self.mount(separator, before=target)
            else:
                self.mount(separator)
        except Exception:
            self.mount(separator)

    def _build_unread_separator_text(self) -> Text:
        if self._unread_count <= 0:
            label = "── New messages ──"
        elif self._unread_count == 1:
            label = "── 1 new message ──"
        else:
            label = f"── {self._unread_count} new messages ──"
        return Text(label, style="bold yellow")

    def _refresh_unread_separator_label(self) -> None:
        if self._unread_separator is None:
            return
        try:
            self._unread_separator.update(self._build_unread_separator_text())
        except Exception:
            pass

    def _get_unread_boundary_target(self) -> Optional[Widget]:
        if not self._unread_boundary_id:
            return None
        try:
            return self.query_one(f"#{self._unread_boundary_id}")
        except Exception:
            return None

    def _remove_unread_separator(self) -> None:
        if self._unread_separator is None:
            return
        try:
            self._unread_separator.remove()
        except Exception:
            pass
        self._unread_separator = None

    def _is_at_bottom(self) -> bool:
        try:
            return self.scroll_y >= self.max_scroll_y - 1
        except Exception:
            return True
