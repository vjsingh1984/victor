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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from victor.ui.common.constants import CURSOR_COL_INDEX, CURSOR_ROW_INDEX

from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.timer import Timer
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
            messages = [row[CURSOR_ROW_INDEX] for row in cursor.fetchall()]
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
                id="shortcut-hints",
            )

    def _update_shortcuts_idle(self) -> None:
        """Update shortcuts for idle state."""
        try:
            hints = self.query_one("#shortcut-hints", Label)
            hints.update(
                Text.assemble(
                    ("Ctrl+L", "bold"),
                    " clear  ",
                    ("Ctrl+S", "bold"),
                    " save  ",
                    ("Enter", "bold"),
                    " send",
                )
            )
        except Exception:
            pass

    def _update_shortcuts_streaming(self) -> None:
        """Update shortcuts for streaming state."""
        try:
            hints = self.query_one("#shortcut-hints", Label)
            hints.update(
                Text.assemble(
                    ("Ctrl+X", "bold"),
                    " cancel  ",
                    ("Ctrl+D", "bold"),
                    " toggle details",
                )
            )
        except Exception:
            pass

    def _update_shortcuts_error(self) -> None:
        """Update shortcuts for error state."""
        try:
            hints = self.query_one("#shortcut-hints", Label)
            hints.update(
                Text.assemble(
                    ("Ctrl+L", "bold"),
                    " clear  ",
                    ("Ctrl+C", "bold"),
                    " exit",
                )
            )
        except Exception:
            pass

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
        provider_label.update(Text.assemble(
            ("Victor ", "bold #7cb7ff"),
            "| ",
            (f"{self.provider}", ""),
            (" / ", ""),
            (f"{self.model}", "bold"),
        ))
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

    async def _on_key(self, event) -> None:
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
                yield Label("❯", classes="prompt-indicator", id="prompt")
                yield SubmitTextArea(
                    placeholder="Enter your message here...",
                    id="message-input",
                )
            yield Label(
                "Enter: send  Shift+Enter: newline  ↑/↓: history  Ctrl+C: exit  /help: commands",
                classes="input-hint",
            )

    def on_submit_text_area_submit(self, event: SubmitTextArea.Submit) -> None:
        """Handle submit from the custom TextArea."""
        if event.value.strip():
            self.post_message(self.Submitted(event.value))

    def on_mount(self) -> None:
        """Focus the input on mount and defer history loading."""
        self._input = self.query_one("#message-input", SubmitTextArea)
        self._input.focus()

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

    def _update_prompt_hint(self) -> None:
        """Update prompt based on cursor position.

        Shows what Enter key will do:
        - ⏎ (Enter key symbol): Submit message
        - ↵ (Newline symbol): Insert newline
        """
        if not self._input:
            return

        try:
            cursor_location = self._input.cursor_location
            text = self._input.text

            # cursor_location is a tuple (line, column)
            line_idx, col_idx = cursor_location

            # Get the current line text
            lines = text.splitlines()
            current_line = lines[line_idx] if line_idx < len(lines) else ""
            is_at_end = col_idx >= len(current_line)

            prompt = self.query_one("#prompt", Label)
            if is_at_end and text.strip():
                # Enter will submit
                prompt.update("⏎")
                prompt.add_class("will-submit")
            else:
                # Enter will insert newline
                prompt.update("↵")
                prompt.remove_class("will-submit")
        except Exception:
            # Silently fail if query fails
            pass

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

        # Update prompt hint on any key event
        self._update_prompt_hint()

        # Only trigger history if cursor is at the top of the text area
        is_at_top = self._input.cursor_location[CURSOR_ROW_INDEX] == 0

        if event.key == "up" and is_at_top:
            self._history_prev()
            event.prevent_default()
            event.stop()
        elif event.key == "down":
            # Allow "down" to navigate history only if already browsing
            if self._history_index != -1:
                is_at_bottom = (
                    self._input.cursor_location[CURSOR_ROW_INDEX]
                    == len(self._input.document.lines) - 1
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

    Auto-collapses to compact form 2 seconds after completion.
    Click to toggle expand/collapse.
    Expandable error details for failed tool calls.
    """

    DEFAULT_CSS = """
    ToolCallWidget .error-details {
        margin-top: 1;
        padding: 1;
        background: $error-bg;
        border: round $error;
        max-height: 20;
        overflow-y: auto;
        text-style: italic;
        color: $error;
    }

    ToolCallWidget .error-details.hidden {
        display: none;
    }

    ToolCallWidget .error-summary {
        color: $error;
        text-style: italic;
    }

    ToolCallWidget .expand-btn {
        margin-top: 1;
        min-width: 18;
    }
    """

    def __init__(
        self,
        tool_name: str,
        arguments: dict | None = None,
        status: str = "pending",
        elapsed: float | None = None,
        error_message: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.arguments = arguments or {}
        self.status = status
        self.elapsed = elapsed
        self.error_message = error_message
        self.add_class(status)

        # Collapse state management
        self._is_collapsed = False
        self._is_complete = False
        self._collapse_timer: Optional[Timer] = None

        # Error details state
        self._error_summary = ""
        self._show_expand_button = False

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

        with Vertical():
            yield Label(
                Text.assemble(
                    (status_icon, "bold"),
                    " ",
                    (self.tool_name, "cyan"),
                    (args_preview, "dim"),
                    (elapsed_str, "dim"),
                    (" ▼", "dim") if not self._is_collapsed else (" ▶", "dim"),
                ),
                classes="tool-header",
                id="tool-header",
            )
            yield Label(
                Text.assemble(
                    ("Click to ", "dim"),
                    ("collapse" if not self._is_collapsed else "expand", "dim italic"),
                ),
                classes="tool-hint",
                id="tool-hint",
            )

            # Add error section if error exists
            if self.status == "error" and self.error_message:
                # Calculate error summary (first line, truncated to 60 chars)
                first_line = self.error_message.split("\n")[0][:60]
                if len(self.error_message.split("\n")[0]) > 60:
                    first_line += "..."
                self._error_summary = first_line
                self._show_expand_button = (
                    "\n" in self.error_message or len(self.error_message) > 60
                )

                yield Label(f"Error: {self._error_summary}", classes="error-summary")
                if self._show_expand_button:
                    yield Button("[+] Show Details", id="expand-error", classes="expand-btn")
                    # Truncate error to 20 lines max
                    error_lines = self.error_message.split("\n")[:20]
                    truncated_error = "\n".join(error_lines)
                    if len(self.error_message.split("\n")) > 20:
                        truncated_error += "\n... (truncated)"
                    yield Vertical(
                        Label(truncated_error, id="error-text"),
                        id="error-details",
                        classes="error-details hidden",
                    )

    def update_status(
        self, status: str, elapsed: float | None = None, error_message: str | None = None
    ) -> None:
        """Update tool call status and auto-collapse if complete.

        Args:
            status: New status ("pending", "success", "error")
            elapsed: Optional elapsed time in seconds
            error_message: Optional error message for failed calls
        """
        self.remove_class(self.status)
        self.status = status
        self.elapsed = elapsed
        self.error_message = error_message
        self.add_class(status)

        # Mark as complete and schedule auto-collapse
        self._is_complete = status in ("success", "error")
        if self._is_complete:
            # Auto-collapse after 2 seconds if complete
            if self._collapse_timer:
                self._collapse_timer.stop()
            self._collapse_timer = self.set_timer(2.0, self._collapse)

        self.refresh()

    def _collapse(self) -> None:
        """Collapse to compact form."""
        if not self._is_collapsed:
            self._is_collapsed = True
            self.add_class("collapsed")
            self._update_header()
            self.refresh()

    def _expand(self) -> None:
        """Expand to full form."""
        if self._is_collapsed:
            self._is_collapsed = False
            self.remove_class("collapsed")
            self._update_header()
            self.refresh()

    def _update_header(self) -> None:
        """Update header to reflect collapse state."""
        try:
            header = self.query_one("#tool-header", Label)
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

            header.update(
                Text.assemble(
                    (status_icon, "bold"),
                    " ",
                    (self.tool_name, "cyan"),
                    (args_preview, "dim"),
                    (elapsed_str, "dim"),
                    (" ▼", "dim") if not self._is_collapsed else (" ▶", "dim"),
                )
            )
        except Exception:
            pass

    def on_click(self) -> None:
        """Toggle collapse on click."""
        if self._is_collapsed:
            self._expand()
        else:
            self._collapse()

        # Stop collapse timer when manually toggled
        if self._collapse_timer:
            self._collapse_timer.stop()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses for expand/collapse error details."""
        if event.button.id == "expand-error":
            try:
                error_details = self.query_one("#error-details", Vertical)
                expand_btn = self.query_one("#expand-error", Button)

                if error_details.has_class("hidden"):
                    # Expand error details
                    error_details.remove_class("hidden")
                    expand_btn.label = "[-] Hide Details"
                else:
                    # Collapse error details
                    error_details.add_class("hidden")
                    expand_btn.label = "[+] Show Details"
            except Exception:
                pass  # Silently fail if widgets not found


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


class IncrementalMarkdownRenderer:
    """Render large markdown documents incrementally.

    Splits markdown into chunks and renders progressively to avoid UI freeze
    on large documents (500+ lines). This prevents the TUI from becoming
    unresponsive when rendering large code blocks or documentation.

    Features:
    - Chunked rendering at safe boundaries
    - Progressive display
    - Configurable chunk size
    - Maintains markdown structure
    """

    # Render chunks of 50 lines at a time
    CHUNK_SIZE = 50

    def __init__(self, markdown_content: str, chunk_size: int = CHUNK_SIZE) -> None:
        """Initialize incremental renderer.

        Args:
            markdown_content: Full markdown content to render
            chunk_size: Lines per chunk (default: 50)
        """
        self.content = markdown_content
        self.chunk_size = chunk_size
        self.chunks = self._split_into_chunks()
        self.current_chunk = 0
        self._total_chunks = len(self.chunks)

    def _split_into_chunks(self) -> List[str]:
        """Split markdown into renderable chunks at safe boundaries.

        Tries to split at blank lines or code block boundaries to avoid
        breaking markdown structures in the middle.
        """
        lines = self.content.splitlines()
        chunks = []
        current_chunk = []
        current_size = 0

        in_code_block = False
        code_block_indent = 0

        for line in lines:
            # Track code blocks
            if line.strip().startswith("```"):
                in_code_block = not in_code_block

            # Add line to current chunk
            current_chunk.append(line)
            current_size += 1

            # Check if we should split
            if current_size >= self.chunk_size:
                # Prefer splitting at blank lines or code block boundaries
                if line.strip() == "" or not in_code_block:
                    # Safe to split here
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                elif current_size >= self.chunk_size * 2:
                    # Force split if we're 2x over limit (even in code block)
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

        # Add remaining content
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def render_next_chunk(self, widget: Static) -> bool:
        """Render next chunk to the widget.

        Args:
            widget: Static widget to update with rendered markdown

        Returns:
            True if more chunks remain, False if rendering complete
        """
        if self.current_chunk >= self._total_chunks:
            return False

        chunk = self.chunks[self.current_chunk]

        # For incremental rendering, we append to existing content
        # This is a simplified approach - in production you'd use a buffer
        current_content = widget.text if hasattr(widget, "text") else ""

        # Combine previous content with new chunk
        combined_content = current_content + ("\n\n" if current_content else "") + chunk

        # Render markdown
        try:
            widget.update(Markdown(combined_content))
        except Exception:
            # Fallback to plain text if markdown parsing fails
            widget.update(combined_content)

        self.current_chunk += 1
        return self.current_chunk < self._total_chunks

    def get_progress(self) -> tuple[int, int]:
        """Get rendering progress.

        Returns:
            Tuple of (current_chunk, total_chunks)
        """
        return (self.current_chunk, self._total_chunks)

    def reset(self) -> None:
        """Reset rendering to beginning."""
        self.current_chunk = 0


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
            import pyperclip  # type: ignore[import-untyped]

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

    Performance optimizations:
    - Debounced markdown rendering (10fps = 100ms throttle)
    - Cached markdown parsing to avoid re-parsing unchanged content
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
        # Performance optimization: Initialize debounce attributes BEFORE setting content
        # This prevents AttributeError when watch_content is called during __init__
        self._render_timer = None
        self._cached_markdown = None
        self._cached_content = None
        self._render_debounce_ms = 100  # 10fps max render rate
        self._code_blocks: List[tuple[str, str]] = []  # (code, language)

        super().__init__(**kwargs)
        self.role = role
        self.content = initial_content
        self.add_class(role)

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
        """React to content changes with debounced rendering.

        Debounces markdown re-renders to 10fps (100ms) to avoid excessive
        CPU usage during rapid streaming. This reduces markdown parsing from
        every chunk to at most 10 times per second.
        """
        # Skip debouncing during initialization (no app context yet)
        # Check if widget is mounted by testing if we can access app
        from textual._context import NoActiveAppError

        try:
            # Try to access app - will raise NoActiveAppError if not mounted
            _ = self.app
        except NoActiveAppError:
            # No app context yet (widget not mounted), render immediately
            self._do_render()
            return
        except Exception:
            # Other errors during app access, render immediately
            self._do_render()
            return

        # We have an app context, check if timer is initialized
        if not hasattr(self, "_render_timer"):
            self._do_render()
            return

        # Cancel any pending render
        if self._render_timer:
            self._render_timer.stop()

        # Schedule debounced render
        self._render_timer: Timer | None = self.set_timer(self._render_debounce_ms / 1000.0, self._do_render)

    def watch_is_streaming(self, streaming: bool) -> None:
        """Show/hide streaming cursor."""
        try:
            cursor = self.query_one("#cursor")
            cursor.display = streaming
        except Exception:
            pass

    def _do_render(self) -> None:
        """Actually update the display with markdown caching and incremental rendering.

        Only re-parses markdown if content has changed since last render.
        Uses incremental rendering for large documents (>100 lines) to avoid UI freeze.
        This provides significant CPU savings during streaming.
        """
        try:
            body = self.query_one("#message-body", Static)
            if self.role == "assistant":
                # Check if content is large enough for incremental rendering
                content_lines = len(self.content.splitlines())

                if content_lines > 100 and not self.is_streaming:
                    # Use incremental rendering for large completed messages
                    if (
                        not hasattr(self, "_incremental_renderer")
                        or self._incremental_renderer is None
                    ):
                        self._incremental_renderer = IncrementalMarkdownRenderer(self.content)
                        self._incremental_renderer.render_next_chunk(body)
                        # Schedule remaining chunks
                        self._schedule_next_incremental_chunk(body)
                    return

                # Cache parsed markdown to avoid re-parsing unchanged content
                if self._cached_content != self.content:
                    self._cached_markdown = Markdown(self.content)
                    self._cached_content = self.content
                    # Clear incremental renderer when content changes
                    if hasattr(self, "_incremental_renderer"):
                        self._incremental_renderer = None
                body.update(self._cached_markdown)
            else:
                body.update(self.content)
        except Exception:
            pass

    def _schedule_next_incremental_chunk(self, body: Static) -> None:
        """Schedule next incremental chunk rendering.

        Uses set_timer to avoid blocking the UI thread during rendering.
        """
        if not hasattr(self, "_incremental_renderer") or self._incremental_renderer is None:
            return

        has_more = self._incremental_renderer.render_next_chunk(body)
        if has_more:
            # Schedule next chunk with minimal delay to keep UI responsive
            self.set_timer(0.001, lambda: self._schedule_next_incremental_chunk(body))

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
        arguments: Optional[Dict[str, Any]] = None,
        on_cancel: Optional[Callable[..., Any]] = None,
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


class VirtualScrollContainer(VerticalScroll):
    """Virtual scrolling container for large conversation logs.

    Only renders messages visible in viewport + buffer zone.
    Efficiently handles 1000+ message conversations by keeping only
    a subset of messages in the DOM at any time.

    Features:
    - Windowed rendering with buffer zone
    - Automatic viewport detection
    - Smooth scrolling with minimal DOM manipulation
    - Memory efficient for large sessions
    - Backward compatible: behaves like EnhancedConversationLog for small sessions
    """

    DEFAULT_CSS = """
    VirtualScrollContainer {
        background: $background;
        padding: 1;
        scrollbar-gutter: stable;
    }

    VirtualScrollContainer > Static {
        margin: 1 0;
    }

    VirtualScrollContainer > StreamingMessageBlock {
        margin: 1 0;
    }
    """

    # Configuration
    BUFFER_SIZE = 10  # Render 10 messages above/below viewport
    VIRTUAL_SCROLL_THRESHOLD = 100  # Enable virtual scrolling after 100 messages

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Store all message data (lightweight dicts)
        self._all_messages: List[Dict[str, Any]] = []
        # Track which messages are currently in DOM
        self._visible_start = 0
        self._visible_end = 0
        # Streaming message always stays in DOM
        self._streaming_message: Optional[StreamingMessageBlock] = None
        self._message_count = 0
        self._auto_scroll = True
        # Virtual scrolling state
        self._use_virtual_scrolling = False
        self._viewport_height = 0
        self._avg_message_height = 10  # Estimate, will be refined

        # Performance optimization: Throttled scrolling
        self._last_scroll_time: float = 0
        self._scroll_throttle_ms = 50  # Scroll max 20fps
        self._update_pending = False

    @property
    def auto_scroll_enabled(self) -> bool:
        return self._auto_scroll

    def _check_enable_virtual_scrolling(self) -> None:
        """Enable virtual scrolling if message count exceeds threshold."""
        should_enable = len(self._all_messages) >= self.VIRTUAL_SCROLL_THRESHOLD
        if should_enable and not self._use_virtual_scrolling:
            self._use_virtual_scrolling = True
            # Initial render when enabling virtual scrolling
            self._update_visible_range()
        elif not should_enable and self._use_virtual_scrolling:
            self._use_virtual_scrolling = False

    def _add_message_to_store(self, role: str, content: str, msg_type: str = "message") -> None:
        """Add message data to virtual store (lightweight operation)."""
        self._all_messages.append(
            {
                "role": role,
                "content": content,
                "type": msg_type,
                "id": self._message_count,
            }
        )
        self._message_count += 1
        self._check_enable_virtual_scrolling()

    def _create_message_widget(self, msg_data: Dict[str, Any]):
        """Create a widget from message data."""
        role = msg_data["role"]
        content = msg_data["content"]
        msg_id = msg_data["id"]

        if role == "user":
            return StreamingMessageBlock(
                role="user",
                initial_content=content,
                id=f"msg-{msg_id}",
            )
        elif role == "assistant":
            return StreamingMessageBlock(
                role="assistant",
                initial_content=content,
                id=f"msg-{msg_id}",
            )
        elif role == "system":
            return Static(
                Text(f"[{content}]", style="dim italic"),
                id=f"msg-{msg_id}",
            )
        elif role == "error":
            return Static(
                Text(f"Error: {content}", style="bold red"),
                id=f"msg-{msg_id}",
            )
        else:
            return Static(content, id=f"msg-{msg_id}")

    def _update_visible_range(self) -> None:
        """Update which messages should be visible based on scroll position."""
        if not self._use_virtual_scrolling:
            return

        # Prevent update cycles
        if self._update_pending:
            return
        self._update_pending = True

        try:
            # Calculate viewport
            scroll_y = self.scroll_y
            viewport_height = self.window_height or 50

            # Estimate which messages are visible
            # Start with a simple estimate based on count
            total_messages = len(self._all_messages)
            if total_messages == 0:
                return

            # Estimate visible range based on scroll position
            # This is a simplified approach - in production you'd track actual heights
            scroll_ratio = scroll_y / max(self.max_scroll_y, 1)
            estimated_visible_idx = int(scroll_ratio * total_messages)

            # Calculate range with buffer
            new_start = max(0, estimated_visible_idx - self.BUFFER_SIZE)
            new_end = min(total_messages, estimated_visible_idx + self.BUFFER_SIZE + 1)

            # Only update if range changed significantly
            if abs(new_start - self._visible_start) > 5 or abs(new_end - self._visible_end) > 5:
                self._visible_start = new_start
                self._visible_end = new_end
                self._render_visible_messages()

        finally:
            self._update_pending = False

    def _render_visible_messages(self) -> None:
        """Render only the visible range of messages."""
        if not self._use_virtual_scrolling:
            return

        # Remove all current children except streaming message
        for child in list(self.children):
            if child is not self._streaming_message:
                child.remove()

        # Render visible range
        for i in range(self._visible_start, self._visible_end):
            if i < len(self._all_messages):
                msg_data = self._all_messages[i]
                widget = self._create_message_widget(msg_data)
                self.mount(widget)

        # Re-add streaming message if active
        if self._streaming_message and self._streaming_message not in self.children:
            self.mount(self._streaming_message)

    def add_user_message(self, content: str) -> None:
        """Add a user message to the log."""
        self._add_message_to_store("user", content)

        if self._use_virtual_scrolling:
            # Update visible range to show new message
            if self._auto_scroll:
                self._visible_end = len(self._all_messages)
                self._visible_start = max(0, self._visible_end - 2 * self.BUFFER_SIZE)
                self._render_visible_messages()
                self.scroll_end(animate=False)
        else:
            # Normal rendering for small sessions
            msg = StreamingMessageBlock(
                role="user",
                initial_content=content,
                id=f"msg-{self._message_count - 1}",
            )
            self.mount(msg)
            self.scroll_end(animate=False)

    def add_assistant_message(self, content: str) -> None:
        """Add a complete assistant message to the log."""
        self._add_message_to_store("assistant", content)

        if self._use_virtual_scrolling:
            if self._auto_scroll:
                self._visible_end = len(self._all_messages)
                self._visible_start = max(0, self._visible_end - 2 * self.BUFFER_SIZE)
                self._render_visible_messages()
                self.scroll_end(animate=False)
        else:
            msg = StreamingMessageBlock(
                role="assistant",
                initial_content=content,
                id=f"msg-{self._message_count - 1}",
            )
            self.mount(msg)
            if self._auto_scroll:
                self.scroll_end(animate=False)

    def add_system_message(self, content: str) -> None:
        """Add a system/status message to the log."""
        self._add_message_to_store("system", content)

        if not self._use_virtual_scrolling:
            msg = Static(
                Text(f"[{content}]", style="dim italic"),
                id=f"msg-{self._message_count - 1}",
            )
            self.mount(msg)

    def add_error_message(self, content: str) -> None:
        """Add an error message to the log."""
        self._add_message_to_store("error", content)

        if self._use_virtual_scrolling:
            if self._auto_scroll:
                self._visible_end = len(self._all_messages)
                self._visible_start = max(0, self._visible_end - 2 * self.BUFFER_SIZE)
                self._render_visible_messages()
                self.scroll_end(animate=False)
        else:
            msg = Static(
                Text(f"Error: {content}", style="bold red"),
                id=f"msg-{self._message_count - 1}",
            )
            self.mount(msg)
            self.scroll_end(animate=False)

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
        """Update the current streaming message.

        Shows new content as it arrives, like Claude Code.
        Always scrolls during streaming to show progress (throttled to 20fps).
        """
        if self._streaming_message:
            self._streaming_message.content = content
            # Throttle scrolling to reduce DOM thrashing during fast streaming
            self._throttled_scroll_end()

    def append_streaming_chunk(self, chunk: str) -> None:
        """Append a chunk to the current streaming message.

        Shows new content as it arrives, like Claude Code.
        Scrolling is throttled to 20fps to reduce DOM manipulation overhead.
        """
        if self._streaming_message:
            self._streaming_message.append_chunk(chunk)
            # Throttle scrolling to reduce DOM thrashing during fast streaming
            self._throttled_scroll_end()

    def _throttled_scroll_end(self) -> None:
        """Scroll to end with throttling (max 20fps).

        Reduces scroll calls from ~100 per second to ~20 per second during
        fast streaming, significantly reducing DOM manipulation overhead.
        """
        now = time.time() * 1000  # Convert to milliseconds
        if now - self._last_scroll_time > self._scroll_throttle_ms:
            self.scroll_end(animate=False)
            self._last_scroll_time = now

    def finish_streaming(self) -> None:
        """Finish the current streaming response."""
        if self._streaming_message:
            # Save streaming message to store
            self._add_message_to_store("assistant", self._streaming_message.content)
            self._streaming_message.finish_streaming()
            self._streaming_message = None
            # Re-render with virtual scrolling if enabled
            if self._use_virtual_scrolling:
                self._render_visible_messages()
        # Final scroll to ensure full response is visible
        self.scroll_end(animate=False)

    def add_tool_progress(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        on_cancel: Optional[Callable[..., Any]] = None,
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
        self._all_messages.clear()
        for child in list(self.children):
            child.remove()
        self._message_count = 0
        self._streaming_message = None
        self._auto_scroll = True
        self._use_virtual_scrolling = False
        self._visible_start = 0
        self._visible_end = 0

    def on_scroll(self, _event) -> None:
        """Update auto-scroll when the user scrolls."""
        self.update_auto_scroll_state()
        # Update visible range for virtual scrolling
        if self._use_virtual_scrolling:
            self._update_visible_range()

    def scroll_to_bottom(self, animate: bool = False) -> None:
        """Scroll to bottom and re-enable auto-scroll."""
        self._auto_scroll = True
        self.scroll_end(animate=animate)
        # Update visible range to show latest messages
        if self._use_virtual_scrolling:
            self._visible_end = len(self._all_messages)
            self._visible_start = max(0, self._visible_end - 2 * self.BUFFER_SIZE)
            self._render_visible_messages()

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


# Backward compatibility alias
EnhancedConversationLog = VirtualScrollContainer
