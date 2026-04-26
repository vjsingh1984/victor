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
import logging
import re
import time
from typing import TYPE_CHECKING, Callable, List, Optional

logger = logging.getLogger(__name__)

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

from victor.ui.history_utils import load_input_history_from_db


def _get_input_history_from_db(limit: int = 100) -> List[str]:
    """Load user message history from conversation database.

    Returns recent unique user messages from the project's consolidated database.
    Uses project.db (consolidated database) which includes conversation history.
    Filters out tool outputs and system-like messages.
    """
    try:
        from victor.config.settings import get_project_paths

        db_path = get_project_paths().project_db
        return load_input_history_from_db(db_path, limit=limit)
    except Exception as e:
        # DB not available - this is expected in some configurations
        logger.debug(f"Failed to load recent messages from database: {e}")
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
        self._follow_paused: bool = False
        self._last_unread_count: int = 0

    def compose(self) -> ComposeResult:
        """Compose the status bar widget.

        Returns:
            ComposeResult: The child widgets for the status bar.

        Layout:
            - Provider info (e.g., "Victor | anthropic / claude-3-5-sonnet-20241022")
            - Status indicator (idle, streaming, error)
            - Follow indicator (shows when auto-following is active)
            - Unread indicator (shows unread message count)
            - Keyboard shortcuts (Ctrl+C exit, Enter send)
        """
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
            follow_label = Label("Following", classes="follow-indicator", id="follow-indicator")
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
        """Update shortcuts for idle state.

        Shows follow/pause action and jump hint when follow is paused.
        """
        try:
            hints = self.query_one("#shortcut-hints", Label)
            if self._follow_paused:
                hints.update(
                    Text.assemble(
                        ("Ctrl+F", "bold"),
                        " resume follow  ",
                        ("Ctrl+End", "bold"),
                        " latest  ",
                        ("Enter", "bold"),
                        " send",
                    )
                )
            else:
                hints.update(
                    Text.assemble(
                        ("Ctrl+F", "bold"),
                        " pause follow  ",
                        ("Ctrl+S", "bold"),
                        " save  ",
                        ("Enter", "bold"),
                        " send",
                    )
                )
        except Exception as e:
            logger.debug(f"Failed to update idle shortcuts: {e}")

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
        except Exception as e:
            logger.debug(f"Failed to update streaming shortcuts: {e}")

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
        except Exception as e:
            logger.debug(f"Failed to update error shortcuts: {e}")

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
        """Update the follow/paused indicator.

        No-op when the state has not changed.
        """
        if paused == self._follow_paused:
            return
        self._follow_paused = paused
        try:
            label = self.query_one("#follow-indicator", Label)
            label.update("Paused" if paused else "Following")
            label.remove_class("following", "paused")
            label.add_class("paused" if paused else "following")
        except Exception as e:
            logger.debug(f"Failed to update follow indicator: {e}")
        self.update_shortcuts(self._state)

    def update_unread(self, count: int) -> None:
        """Update the unread badge.

        No-op when the count has not changed.
        """
        if count == self._last_unread_count:
            return
        self._last_unread_count = count
        try:
            label = self.query_one("#unread-indicator", Label)
            if count > 0:
                label.update(f"{count} new")
                label.add_class("visible")
            else:
                label.update("")
                label.remove_class("visible")
        except Exception as e:
            logger.debug(f"Failed to update unread indicator: {e}")


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

    _APP_SHORTCUT_ACTIONS: dict[str, str] = {
        "ctrl+f": "action_toggle_follow_mode",
        "ctrl+n": "action_jump_unread",
        "ctrl+u": "action_toggle_unread_marker",
        "ctrl+end": "action_scroll_bottom",
        "ctrl+x": "action_cancel_stream",
    }

    def _dispatch_app_shortcut(self, event) -> bool:
        """Forward selected app-level shortcuts while editing the prompt."""
        action_name = self._APP_SHORTCUT_ACTIONS.get(event.key)
        if not action_name:
            return False
        app = self.app
        if app is None:
            return False
        action = getattr(app, action_name, None)
        if not callable(action):
            return False
        result = action()
        if asyncio.iscoroutine(result):
            asyncio.create_task(result)
        event.prevent_default()
        event.stop()
        return True

    async def _on_key(self, event) -> None:
        """Handle key events before TextArea processes them."""
        if self._dispatch_app_shortcut(event):
            return

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

    _BUSY_HINT = "Working…"
    _IDLE_HINT = "Enter: send  Shift+Enter: newline  ↑/↓: history  Ctrl+C: exit  /help: commands"

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
        """Compose the input widget.

        Creates a text input area with a prompt indicator and hint label.
        Supports Up/Down arrow navigation through conversation history.

        Returns:
            ComposeResult: The child widgets for the input area.

        Layout:
            - Prompt indicator (❯): Visual prompt marker
            - SubmitTextArea: Multi-line text input with submit on Ctrl+Enter/Ctrl+J
            - Hint label: Shows keyboard shortcuts and status messages
        """
        with Vertical():
            with Horizontal(classes="input-row"):
                yield Label("❯", classes="prompt-indicator")
                yield SubmitTextArea(
                    placeholder="Enter your message here...",
                    id="message-input",
                )
            yield Label(
                self._IDLE_HINT,
                classes="input-hint",
            )

    def on_submit_text_area_submit(self, event: SubmitTextArea.Submit) -> None:
        """Handle submit from the custom TextArea."""
        if event.value.strip():
            self.post_message(self.Submitted(event.value))

    def on_mount(self) -> None:
        """Focus the input on mount and defer history loading."""
        self._input = self.query_one("#message-input", SubmitTextArea)
        self._prompt_label = self.query_one(".prompt-indicator", Label)
        self._hint_label = self.query_one(".input-hint", Label)
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

    async def _load_history_async(self) -> None:
        """Load input history without blocking the UI."""
        try:
            db_history = await asyncio.to_thread(
                _get_input_history_from_db,
                limit=InputWidget._max_history,
            )
        except Exception as e:
            logger.debug(f"Failed to load input history from database: {e}")
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

    def set_value(self, value: str) -> None:
        """Set the input field content programmatically."""
        if self._input:
            self._input.load_text(value)
            self._input.cursor_location = (999, 999)
        self._history_index = -1
        self._draft = ""

    def focus_input(self) -> None:
        """Focus the input field."""
        if self._input:
            self._input.focus()

    def set_busy(self, busy: bool) -> None:
        """Set the busy state of the input widget."""
        if busy:
            self.add_class("busy")
            if self._input:
                self._input.disabled = True
            if self._prompt_label:
                self._prompt_label.update("⋯")
                self._prompt_label.add_class("busy")
            if self._hint_label:
                self._hint_label.update(self._BUSY_HINT)
        else:
            self.remove_class("busy")
            if self._input:
                self._input.disabled = False
            if self._prompt_label:
                self._prompt_label.update("❯")
                self._prompt_label.remove_class("busy")
            if self._hint_label:
                self._hint_label.update(self._IDLE_HINT)

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
    ToolCallWidget .tool-follow-ups {
        margin-top: 1;
        height: auto;
    }

    ToolCallWidget .tool-output-preview {
        margin-top: 1;
        color: $text-muted;
        max-height: 4;
        overflow: hidden;
    }

    ToolCallWidget .follow-up-button {
        width: auto;
        min-width: 18;
        margin-right: 1;
    }
    """

    class FollowUpSelected(Message):
        """Message emitted when a follow-up suggestion is selected."""

        def __init__(self, command: str) -> None:
            super().__init__()
            self.command = command

    def __init__(
        self,
        tool_name: str,
        arguments: dict | None = None,
        status: str = "pending",
        elapsed: float | None = None,
        follow_up_suggestions: list[dict] | None = None,
        output_preview: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tool_name = tool_name
        self.arguments = arguments or {}
        self.status = status
        self.elapsed = elapsed
        self.follow_up_suggestions = self._normalize_follow_up_suggestions(follow_up_suggestions)
        self._output_preview = self._normalize_output_preview(output_preview)
        self.add_class(status)

    @staticmethod
    def _normalize_follow_up_suggestions(suggestions: list[dict] | None) -> list[dict]:
        """Normalize tool follow-up suggestions for display."""
        if not suggestions:
            return []
        normalized = []
        for suggestion in suggestions:
            if not isinstance(suggestion, dict):
                continue
            command = suggestion.get("command")
            if not isinstance(command, str) or not command.strip():
                continue
            normalized.append(suggestion)
        return normalized

    @staticmethod
    def _follow_up_label(suggestion: dict) -> str:
        """Build a compact button label for a follow-up suggestion."""
        description = suggestion.get("description")
        if isinstance(description, str) and description.strip():
            label = description.strip()
        else:
            label = str(suggestion.get("command", "")).strip()
        if len(label) > 32:
            return label[:29] + "..."
        return label

    @staticmethod
    def _normalize_output_preview(output_preview: str | None) -> str:
        """Build a compact multi-line preview for tool output."""
        if not isinstance(output_preview, str) or not output_preview.strip():
            return ""
        lines = output_preview.splitlines() or [output_preview]
        preview_lines = lines[:3]
        normalized = "\n".join(line[:120] for line in preview_lines)
        if len(lines) > 3 or len(output_preview) > len(normalized):
            normalized += "\n..."
        return normalized

    def compose(self) -> ComposeResult:
        """Compose the tool call widget.

        Displays:
            - Status icon (pending: ..., success: ✓, error: ✗)
            - Tool name
            - Arguments preview (first argument, truncated if too long)
            - Elapsed time
            - Follow-up suggestions (if available)

        Returns:
            ComposeResult: The child widgets for the tool call display.
        """
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
            id="tool-header-label",
        )
        yield Static(self._output_preview, classes="tool-output-preview", id="tool-output-preview")
        if self.follow_up_suggestions:
            with Horizontal(classes="tool-follow-ups"):
                for idx, suggestion in enumerate(self.follow_up_suggestions[:2]):
                    yield Button(
                        self._follow_up_label(suggestion),
                        id=f"follow-up-{idx}",
                        classes="follow-up-button",
                        variant="primary" if idx == 0 else "default",
                    )

    def update_status(
        self,
        status: str,
        elapsed: float | None = None,
        follow_up_suggestions: list[dict] | None = None,
        output_preview: str | None = None,
    ) -> None:
        """Update tool call status."""
        self.remove_class(self.status)
        self.status = status
        self.elapsed = elapsed
        follow_ups_changed = False
        if follow_up_suggestions is not None:
            normalized_follow_ups = self._normalize_follow_up_suggestions(follow_up_suggestions)
            follow_ups_changed = normalized_follow_ups != self.follow_up_suggestions
            self.follow_up_suggestions = normalized_follow_ups
        preview_changed = False
        if output_preview is not None:
            normalized_preview = self._normalize_output_preview(output_preview)
            preview_changed = normalized_preview != self._output_preview
            self._output_preview = normalized_preview
        self.add_class(status)
        # Targeted header update instead of expensive refresh(recompose=True)
        try:
            header = self.query_one("#tool-header-label", Label)
            status_icon = {
                "pending": "...",
                "success": "✓",
                "error": "✗",
            }.get(status, "?")
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
                )
            )
        except Exception as e:
            logger.debug(f"Failed to update tool header label: {e}")
        if preview_changed:
            try:
                preview = self.query_one("#tool-output-preview", Static)
                preview.update(self._output_preview)
            except Exception as e:
                logger.debug(f"Failed to update tool output preview: {e}")
        if follow_ups_changed:
            try:
                self.refresh(recompose=True)
            except Exception as e:
                logger.debug(f"Failed to recompose tool follow-ups: {e}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle follow-up suggestion buttons."""
        button_id = event.button.id or ""
        if not button_id.startswith("follow-up-"):
            return
        try:
            index = int(button_id.rsplit("-", 1)[-1])
        except ValueError:
            return
        if index < 0 or index >= len(self.follow_up_suggestions):
            return
        command = self.follow_up_suggestions[index].get("command")
        if isinstance(command, str) and command.strip():
            self.post_message(self.FollowUpSelected(command.strip()))
            event.stop()


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
        """Compose the thinking widget.

        Displays the model's extended thinking/reasoning process in a styled panel.

        Returns:
            ComposeResult: The child widgets for the thinking display.

        Layout:
            - Header label: "Thinking..." in bold magenta
            - Content Static: Shows the thinking/reasoning text content
        """
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
        except Exception as e:
            logger.debug(f"Failed to update thinking content: {e}")


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
        """Compose the code block widget.

        Displays syntax-highlighted code with a copy button and language label.

        Returns:
            ComposeResult: The child widgets for the code block display.

        Layout:
            - Header row: Language label (e.g., [python]) and Copy button
            - Static Syntax widget: Syntax-highlighted code with Monokai theme
        """
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

    # Throttle interval for streaming display updates (seconds).
    _STREAM_THROTTLE = 0.05

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
        self._chunk_buffer: str = ""
        self._last_update_time: float = 0.0

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
        except Exception as e:
            logger.debug(f"Failed to update streaming cursor: {e}")

    def _update_display(self) -> None:
        """Update the message body with current content.

        During streaming, renders as plain text for performance.
        After streaming finishes, renders as Markdown.
        """
        try:
            body = self.query_one("#message-body", Static)
            if self.is_streaming:
                # Fast plain-text update while streaming
                body.update(self.content)
            elif self.role == "assistant":
                # Final render as Markdown
                body.update(Markdown(self.content))
            else:
                body.update(self.content)
        except Exception as e:
            logger.debug(f"Failed to update message display: {e}")

    def append_chunk(self, chunk: str) -> None:
        """Append a streaming chunk to the content.

        Chunks are buffered and flushed at a throttled rate to avoid
        excessive redraws during fast token delivery.
        """
        now = time.monotonic()
        self._chunk_buffer += chunk
        if now - self._last_update_time >= self._STREAM_THROTTLE:
            self.content += self._chunk_buffer
            self._chunk_buffer = ""
            self._last_update_time = now

    def finish_streaming(self) -> None:
        """Mark streaming as complete and flush any buffered content."""
        # Flush remaining buffer
        if self._chunk_buffer:
            self.content += self._chunk_buffer
            self._chunk_buffer = ""
        self.is_streaming = False
        # Force final Markdown render
        self._update_display()


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
        """Compose the tool progress panel.

        Displays real-time tool execution progress with status, progress bar,
        and optional cancel button.

        Returns:
            ComposeResult: The child widgets for the progress panel.

        Layout:
            - Header row: Tool name with icon, status icon (⏳/✓/✗), elapsed time
            - ProgressBar: Visual progress indicator (0-100%)
            - Output preview: Shows recent tool output (max 4 lines)
            - Cancel button: Only shown if on_cancel callback provided
        """
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
        except Exception as e:
            logger.debug(f"Failed to update tool status icon: {e}")

    def watch_elapsed(self, elapsed: float) -> None:
        """Update elapsed time display."""
        try:
            elapsed_label = self.query_one("#elapsed")
            elapsed_label.update(f"{elapsed:.1f}s")
        except Exception as e:
            logger.debug(f"Failed to update elapsed time: {e}")

    def watch_progress(self, progress: float) -> None:
        """Update progress bar."""
        try:
            bar = self.query_one("#progress-bar", ProgressBar)
            bar.update(progress=progress)
        except Exception as e:
            logger.debug(f"Failed to update progress bar: {e}")

    def update_output_preview(self, text: str) -> None:
        """Update the output preview text."""
        self._output_preview = text[:200]  # Truncate
        try:
            output = self.query_one("#output")
            output.update(self._output_preview)
        except Exception as e:
            logger.debug(f"Failed to update output preview: {e}")

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
    - Auto-scrolls to show new content (respects user scroll-away)
    - Smooth scrolling through history
    - Proper message separation and styling
    - Unread message tracking with optional separator widget
    - Sticky follow-pause mode (Ctrl+F)
    - Programmatic scroll guard to avoid false scroll-away detection
    """

    DEFAULT_CSS = """
    EnhancedConversationLog {
        background: $background;
        padding: 1;
        scrollbar-gutter: stable;
    }

    EnhancedConversationLog > Static {
        margin: 1 0;
    }

    EnhancedConversationLog > StreamingMessageBlock {
        margin: 1 0;
    }

    EnhancedConversationLog > .unread-separator {
        color: $warning;
        text-style: bold;
        margin: 0;
        padding: 0;
    }
    """

    # Duration (seconds) of the guard window after a programmatic scroll_end.
    _PROGRAMMATIC_GUARD_SECONDS = 0.15
    # Duration (seconds) of the guard window after a resize event.
    _RESIZE_GUARD_SECONDS = 0.3
    # Minimum interval between scroll_end calls during streaming (200ms).
    _SCROLL_THROTTLE_SECONDS = 0.2
    # Max widgets before trimming old messages (virtual scrolling).
    MAX_VISIBLE_MESSAGES = 100

    def __init__(self, show_unread_separator: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._streaming_message: Optional[StreamingMessageBlock] = None
        self._message_count = 0
        self._auto_scroll = True
        self._user_scrolled = False
        # Follow-pause (sticky)
        self._follow_paused = False
        # Unread tracking
        self._unread_count = 0
        self._unread_separator: Optional[Static] = None
        self._unread_boundary_id: Optional[str] = None
        self._show_unread_separator = show_unread_separator
        # Programmatic scroll guard timestamps
        self._ignore_scroll_update_until: float = 0.0
        self._ignore_resize_scroll_update_until: float = 0.0
        # Streaming scroll throttle
        self._last_scroll_end_time: float = 0.0

    # -- public properties ---------------------------------------------------

    @property
    def auto_scroll_enabled(self) -> bool:
        return self._auto_scroll

    @property
    def follow_paused(self) -> bool:
        return self._follow_paused

    @property
    def unread_count(self) -> int:
        return self._unread_count

    @property
    def unread_separator_enabled(self) -> bool:
        return self._show_unread_separator

    # -- follow pause --------------------------------------------------------

    def set_follow_paused(self, paused: bool, jump_to_bottom: bool = False) -> None:
        """Enable or disable sticky follow-pause mode.

        When paused, auto-scroll is disabled and user messages do not
        re-enable it.  ``jump_to_bottom=True`` scrolls and clears unread.
        """
        self._follow_paused = paused
        if paused:
            self._auto_scroll = False
        elif jump_to_bottom:
            self.scroll_to_bottom()

    # -- message methods -----------------------------------------------------

    def add_user_message(self, content: str) -> None:
        """Add a user message to the log."""
        msg = StreamingMessageBlock(
            role="user",
            initial_content=content,
            id=f"msg-{self._message_count}",
        )
        self._message_count += 1
        self.mount(msg)
        self._trim_old_messages()
        # User messages always re-enable follow (unless sticky-paused)
        if not self._follow_paused:
            self._auto_scroll = True
            self._maybe_scroll_end(force=True)

    def add_assistant_message(self, content: str) -> None:
        """Add a complete assistant message to the log."""
        msg_id = f"msg-{self._message_count}"
        msg = StreamingMessageBlock(
            role="assistant",
            initial_content=content,
            id=msg_id,
        )
        self._message_count += 1
        if not self._auto_scroll:
            self._increment_unread(msg_id)
        self.mount(msg)
        self._trim_old_messages()
        self._maybe_scroll_end()

    def add_system_message(self, content: str) -> None:
        """Add a system/status message to the log."""
        msg_id = f"msg-{self._message_count}"
        msg = Static(
            Text(f"[{content}]", style="dim italic"),
            id=msg_id,
        )
        self._message_count += 1
        if not self._auto_scroll:
            self._increment_unread(msg_id)
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
        # Always show errors
        self._maybe_scroll_end(force=True)

    def add_history_message(self, role: str, content: str) -> None:
        """Add a replayed history message without scroll or unread side effects."""
        if role == "user":
            msg = StreamingMessageBlock(
                role="user",
                initial_content=content,
                id=f"msg-{self._message_count}",
            )
        elif role == "assistant":
            msg = StreamingMessageBlock(
                role="assistant",
                initial_content=content,
                id=f"msg-{self._message_count}",
            )
        elif role == "system":
            msg = Static(
                Text(f"[{content}]", style="dim italic"),
                id=f"msg-{self._message_count}",
            )
        elif role == "error":
            msg = Static(
                Text(f"Error: {content}", style="bold red"),
                id=f"msg-{self._message_count}",
            )
        else:
            # Unknown role — skip silently.
            return
        self._message_count += 1
        self.mount(msg)

    def add_history_code_block(
        self,
        code: str,
        language: str = "python",
    ) -> None:
        """Add a replayed code block without scroll or unread side effects."""
        block = CodeBlock(
            code=code,
            language=language,
            id=f"code-{self._message_count}",
        )
        self._message_count += 1
        self.mount(block)

    # -- streaming -----------------------------------------------------------

    def start_streaming(self) -> StreamingMessageBlock:
        """Start a streaming response and return the message block."""
        self._streaming_message = StreamingMessageBlock(
            role="assistant",
            initial_content="",
            id=f"msg-{self._message_count}",
        )
        self._streaming_message.is_streaming = True
        msg_id = f"msg-{self._message_count}"
        self._message_count += 1
        if not self._auto_scroll:
            self._increment_unread(msg_id)
        self.mount(self._streaming_message)
        self._maybe_scroll_end(force=True)
        return self._streaming_message

    def update_streaming(self, content: str) -> None:
        """Update the current streaming message.

        Respects auto-scroll: only scrolls if following.
        """
        if self._streaming_message:
            self._streaming_message.content = content
            self._maybe_scroll_end()

    def append_streaming_chunk(self, chunk: str) -> None:
        """Append a chunk to the current streaming message.

        Respects auto-scroll: only scrolls if following.
        """
        if self._streaming_message:
            self._streaming_message.append_chunk(chunk)
            self._maybe_scroll_end()

    def finish_streaming(self) -> None:
        """Finish the current streaming response."""
        if self._streaming_message:
            self._streaming_message.finish_streaming()
            self._streaming_message = None
        self._maybe_scroll_end(bypass_throttle=True)

    # -- tool / code helpers -------------------------------------------------

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

    # -- clear / scroll ------------------------------------------------------

    def clear(self) -> None:
        """Clear all messages from the log."""
        for child in list(self.children):
            child.remove()
        self._message_count = 0
        self._streaming_message = None
        self._auto_scroll = True
        self._unread_count = 0
        self._unread_separator = None
        self._unread_boundary_id = None

    def scroll_to_bottom(self, animate: bool = False) -> None:
        """Scroll to bottom, re-enable auto-scroll, and clear unread state."""
        self._auto_scroll = True
        self._clear_unread()
        self.scroll_end(animate=animate)

    def disable_auto_scroll(self) -> None:
        """Disable auto-scroll until user returns to bottom."""
        self._auto_scroll = False

    # -- auto-scroll state ---------------------------------------------------

    def update_auto_scroll_state(self) -> None:
        """Update auto-scroll based on scroll position.

        Respects sticky follow-pause: if paused, auto-scroll stays off
        but unread count is cleared when at bottom.
        """
        at_bottom = self._is_at_bottom()
        if self._follow_paused:
            # Sticky pause keeps auto-scroll off, but clears unread at bottom
            self._auto_scroll = False
            if at_bottom:
                self._unread_count = 0
        else:
            self._auto_scroll = at_bottom

    def on_update_scroll(self, event) -> None:
        """Handle Textual UpdateScroll messages with guard windows."""
        now = time.monotonic()

        # Resize guard — always skip when inside window (auto-scroll is off)
        if now < self._ignore_resize_scroll_update_until:
            return

        # Programmatic guard — skip unless user actively scrolled away
        if now < self._ignore_scroll_update_until:
            if self._auto_scroll and self._is_at_bottom():
                return
            # User scrolled away during guard — fall through to update

        self.update_auto_scroll_state()

    def on_resize(self, event) -> None:
        """Set a short guard after resize to suppress transient scroll events."""
        if not self._auto_scroll:
            self._ignore_resize_scroll_update_until = time.monotonic() + self._RESIZE_GUARD_SECONDS
        else:
            self._ignore_resize_scroll_update_until = 0.0

    # -- unread tracking -----------------------------------------------------

    def _increment_unread(self, msg_id: str) -> None:
        """Increment unread count and optionally insert/update separator."""
        self._unread_count += 1
        if self._unread_boundary_id is None:
            self._unread_boundary_id = msg_id
        if self._show_unread_separator:
            self._upsert_unread_separator()

    def _upsert_unread_separator(self) -> None:
        """Create or update the unread separator label."""
        label = self._unread_label_text()
        if self._unread_separator is None:
            self._unread_separator = Static(label, classes="unread-separator")
            if self._unread_boundary_id:
                try:
                    target = self.query_one(f"#{self._unread_boundary_id}")
                    self.mount(self._unread_separator, before=target)
                except Exception:
                    self.mount(self._unread_separator)
            else:
                self.mount(self._unread_separator)
        else:
            self._unread_separator._Static__content = label  # type: ignore[attr-defined]

    def _unread_label_text(self) -> str:
        n = self._unread_count
        suffix = "s" if n != 1 else ""
        return f"--- {n} new message{suffix} ---"

    def _clear_unread(self) -> None:
        """Reset unread state and remove separator widget."""
        self._unread_count = 0
        if self._unread_separator is not None:
            try:
                self._unread_separator.remove()
            except Exception:
                pass
            self._unread_separator = None
        self._unread_boundary_id = None

    def set_unread_separator_enabled(self, enabled: bool) -> None:
        """Toggle unread separator marker on or off at runtime."""
        self._show_unread_separator = enabled
        if enabled and self._unread_count > 0 and self._unread_separator is None:
            # Insert marker at the saved boundary
            self._upsert_unread_separator()
        elif not enabled and self._unread_separator is not None:
            try:
                self._unread_separator.remove()
            except Exception:
                pass
            self._unread_separator = None

    def jump_to_unread_separator(self) -> bool:
        """Scroll to the unread separator widget.

        Returns True if a target was found and scrolled to.
        """
        if self._unread_separator is not None:
            self.scroll_to_widget(self._unread_separator)
            return True
        if self._unread_boundary_id is not None:
            try:
                target = self.query_one(f"#{self._unread_boundary_id}")
                self.scroll_to_widget(target)
                return True
            except Exception:
                pass
        return False

    # -- internal helpers ----------------------------------------------------

    def _maybe_scroll_end(self, force: bool = False, bypass_throttle: bool = False) -> None:
        """Conditionally scroll to end.

        ``force=True``: scroll regardless of _auto_scroll (anchors new messages).
        ``bypass_throttle=True``: skip the rate-limit but still respect _auto_scroll.
        Both ``force`` and ``bypass_throttle`` are combined with OR for throttle bypass.
        Sets a short guard window to suppress transient UpdateScroll events.
        """
        if not (force or self._auto_scroll):
            return
        now = time.monotonic()
        skip_throttle = force or bypass_throttle
        if not skip_throttle and (now - self._last_scroll_end_time) < self._SCROLL_THROTTLE_SECONDS:
            return
        self._last_scroll_end_time = now
        self.scroll_end(animate=False)
        self._ignore_scroll_update_until = now + self._PROGRAMMATIC_GUARD_SECONDS

    def _trim_old_messages(self) -> None:
        """Remove oldest message widgets when the DOM grows too large.

        Keeps at most MAX_VISIBLE_MESSAGES children mounted to prevent
        unbounded widget tree growth across long conversations.
        """
        children = [c for c in self.children if not isinstance(c, type(self._unread_separator))]
        excess = len(children) - self.MAX_VISIBLE_MESSAGES
        if excess > 0:
            for child in children[:excess]:
                try:
                    child.remove()
                except Exception:
                    pass

    def _is_at_bottom(self) -> bool:
        try:
            return self.scroll_y >= self.max_scroll_y - 1
        except Exception as e:
            logger.debug(f"Failed to determine scroll position, assuming at bottom: {e}")
            return True
