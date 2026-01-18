"""Victor TUI Application.

Main Textual application providing a modern chat interface
with input at the bottom and conversation history in the middle.
"""

from __future__ import annotations

import asyncio
import io
from typing import TYPE_CHECKING, Any, Callable, Optional

from rich.console import Console
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Input, Label

from victor.ui.common.constants import SESSION_TYPE_INDEX, SESSION_UUID_INDEX
from victor.ui.tui.session import Message
from victor.ui.tui.theme import THEME_CSS
from victor.ui.tui.widgets import (
    VirtualScrollContainer,
    InputWidget,
    StatusBar,
    ThinkingWidget,
    ToolCallWidget,
)
from victor.config.tui_themes import THEMES, get_theme
from victor.config.keybindings import KeybindingConfig, DEFAULT_KEYBINDINGS

if TYPE_CHECKING:
    from victor.config.settings import Settings
    from victor.protocols import UIAgentProtocol


class TUIConsoleAdapter:
    """Adapter that captures Rich console output for TUI display.

    SlashCommandHandler uses rich.console.Console for output.
    This adapter captures that output and redirects it to the TUI's
    conversation log as system messages.
    """

    def __init__(
        self,
        conversation_log: EnhancedConversationLog,
        on_line: Optional[Callable[[str], None]] = None,
    ):
        self._log = conversation_log
        self._buffer = io.StringIO()
        self._console = Console(file=self._buffer, force_terminal=False, width=120)
        self._on_line = on_line

    def print(self, *args, **kwargs) -> None:
        """Capture print output and display in TUI."""
        # Reset buffer
        self._buffer.seek(0)
        self._buffer.truncate(0)

        # Render to buffer
        self._console.print(*args, **kwargs)

        # Get rendered text (strip ANSI since TUI handles styling)
        output = self._buffer.getvalue().strip()
        if output:
            # Split by lines and add each as system message
            for line in output.split("\n"):
                line = line.strip()
                if line:
                    if self._on_line:
                        self._on_line(line)
                    self._log.add_system_message(line)

    def __getattr__(self, name):
        """Delegate other console methods to the internal console."""
        return getattr(self._console, name)


class SessionRestoreProgress(ModalScreen[None]):
    """Progress modal for session restore.

    Shows a progress bar and message count during session restoration.
    """

    BINDINGS = [("escape", "app.pop_screen", "Close")]

    def __init__(self, total: int, session_name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.total = total
        self.current = 0
        self.session_name = session_name

    def compose(self) -> ComposeResult:
        with Vertical(id="restore-progress-container"):
            yield Label(
                f"Restoring: {self.session_name}", classes="progress-title", id="progress-title"
            )
            yield ProgressBar(
                total=100, show_eta=False, classes="session-progress", id="progress-bar"
            )
            yield Label("0/0 messages (0%)", id="progress-label")

    def on_mount(self) -> None:
        """Initialize progress bar."""
        self.update(0)

    def update(self, current: int) -> None:
        """Update progress bar.

        Args:
            current: Current message count
        """
        self.current = current
        try:
            progress_bar = self.query_one("#progress-bar", ProgressBar)
            percentage = (current / self.total) * 100 if self.total > 0 else 0
            progress_bar.update(progress=percentage)

            label = self.query_one("#progress-label", Label)
            label.update(f"{current}/{self.total} messages ({percentage:.0f}%)")
        except Exception:
            pass


class SessionPicker(ModalScreen[Optional[str]]):
    """Modal screen to select a saved TUI session."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("enter", "select", "Select", show=False),
        Binding("ctrl+f", "focus_filter", "Filter", show=False),
        Binding("ctrl+l", "clear_filter", "Clear Filter", show=False),
        Binding("ctrl+u", "sort_recent", "Sort Recent", show=False),
        Binding("ctrl+m", "sort_messages", "Sort Messages", show=False),
        Binding("ctrl+n", "sort_name", "Sort Name", show=False),
    ]

    def __init__(
        self,
        sessions: list[dict[str, Any]],
        title: str = "Resume Session",
        help_text: str = (
            "Enter: resume  Esc: cancel  Ctrl+F: filter  Ctrl+L: clear  "
            "Ctrl+U: recent  Ctrl+M: messages  Ctrl+N: name"
        ),
    ) -> None:
        super().__init__()
        self._sessions = sessions
        self._session_ids: list[str] = []
        self._title = title
        self._help_text = help_text

    def compose(self) -> ComposeResult:
        with Container(id="session-picker"):
            yield Label(self._title, id="session-title")
            yield Input(placeholder="Filter sessions...", id="session-filter")
            yield DataTable(id="session-table", zebra_stripes=True)
            yield Label(self._help_text, id="session-help")

    def on_mount(self) -> None:
        table = self.query_one("#session-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Source", "ID", "Name", "Provider", "Model", "Updated", "Messages")
        self._populate_table(self._sessions)
        if self._session_ids:
            table.focus()
            table.move_cursor(row=0, column=0)

    def action_focus_filter(self) -> None:
        self.query_one("#session-filter", Input).focus()

    def action_clear_filter(self) -> None:
        filter_input = self.query_one("#session-filter", Input)
        filter_input.value = ""
        self._apply_filter("")

    def action_sort_recent(self) -> None:
        self._sort_sessions(lambda item: item.get("updated_at", ""), reverse=True)

    def action_sort_messages(self) -> None:
        self._sort_sessions(lambda item: item.get("message_count", 0), reverse=True)

    def action_sort_name(self) -> None:
        self._sort_sessions(lambda item: (item.get("name") or "").lower(), reverse=False)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_select(self) -> None:
        if not self._session_ids:
            self.dismiss(None)
            return
        table = self.query_one("#session-table", DataTable)
        row = table.cursor_row or 0
        if row < 0 or row >= len(self._session_ids):
            self.dismiss(None)
            return
        self.dismiss(self._session_ids[row])

    def on_data_table_row_selected(self, _event) -> None:
        self.action_select()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "session-filter":
            return
        self._apply_filter(event.value)

    def _format_time(self, value: str) -> str:
        try:
            from datetime import datetime

            return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return value[:16] if value else ""

    def _populate_table(self, sessions: list[dict[str, Any]]) -> None:
        table = self.query_one("#session-table", DataTable)
        table.clear()
        self._session_ids.clear()
        for session in sessions:
            session_key = session.get("key", session["id"])
            self._session_ids.append(session_key)
            table.add_row(
                session.get("source", ""),
                session["id"][:8],
                session["name"],
                session.get("provider", ""),
                session["model"],
                self._format_time(session.get("updated_at", "")),
                str(session.get("message_count", 0)),
            )
        if self._session_ids:
            table.move_cursor(row=0, column=0)

    def _apply_filter(self, value: str) -> None:
        query = value.strip().lower()
        if not query:
            self._populate_table(self._sessions)
            return
        filtered = []
        for session in self._sessions:
            haystack = " ".join(
                str(session.get(key, "")) for key in ("id", "name", "provider", "model", "source")
            ).lower()
            if query in haystack:
                filtered.append(session)
        self._populate_table(filtered)

    def _sort_sessions(self, key, reverse: bool) -> None:
        self._sessions.sort(key=key, reverse=reverse)
        filter_value = self.query_one("#session-filter", Input).value
        self._apply_filter(filter_value)


class VictorTUI(App):
    """Modern TUI for Victor AI assistant.

    Features:
    - Input box at the bottom (like Claude Code, Gemini CLI)
    - Scrollable conversation history in the middle
    - Status bar at the top showing provider/model
    - Beautiful spacing and aesthetics

    Layout:
    ┌─────────────────────────────────────┐
    │ Victor | anthropic / claude-3-5-sonnet │  <- StatusBar
    ├─────────────────────────────────────┤
    │                                     │
    │  You                                │  <- ConversationLog
    │  Hello, can you help me?            │      (scrollable)
    │                                     │
    │  Victor                             │
    │  Of course! I'd be happy to help... │
    │                                     │
    ├─────────────────────────────────────┤
    │ > Type your message...              │  <- InputWidget
    │   Enter send | Shift+Enter newline  │
    └─────────────────────────────────────┘
    """

    CSS = (
        THEME_CSS
        + """
    Screen {
        background: $background;
        color: $text;
        layout: vertical;
    }

    #main-container {
        width: 100%;
        height: 100%;
        padding: 0 1 0 1;
        layout: vertical;
        min-height: 0;
    }

    #conversation-area {
        width: 100%;
        height: 1fr;
        min-height: 0;
        padding: 0 0 0 0;
        layout: vertical;
    }

    VirtualScrollContainer {
        height: 1fr;
        min-height: 0;
        background: $background;
        border: none;
        padding: 0;
        margin: 0;
        scrollbar-gutter: stable;
    }

    #jump-to-bottom {
        dock: bottom;
        display: none;
        height: 1;
        width: auto;
        padding: 0 1;
        margin: 0 0 1 0;
        background: $panel;
        border: round $border-muted;
        color: $text-muted;
    }

    #jump-to-bottom.visible {
        display: block;
    }

    /* Status bar - compact */
    StatusBar {
        dock: top;
        height: 1;
        background: $panel;
        color: $text;
        padding: 0 1;
        border-bottom: solid $border-muted;
        margin: 0;
    }

    StatusBar .status-content {
        width: 100%;
        height: 100%;
        align: center middle;
    }

    StatusBar .provider-info {
        color: $text-muted;
        text-style: bold;
        width: 1fr;
    }

    StatusBar .status-indicator {
        color: $text-muted;
        text-style: bold;
        text-align: center;
        width: auto;
    }

    StatusBar .status-indicator.idle {
        color: $text-muted;
    }

    StatusBar .status-indicator.busy {
        color: $primary;
    }

    StatusBar .status-indicator.streaming {
        color: $warning;
    }

    StatusBar .provider-info .victor-name {
        color: $primary;
    }

    StatusBar .shortcuts {
        color: $text-muted;
        text-align: right;
        width: 1fr;
    }

    /* Messages - more compact */
    MessageWidget {
        width: 100%;
        padding: 0 1;
        margin: 0 0 1 0;
        background: transparent;
        border: none;
    }

    MessageWidget.user {
        background: $panel-alt;
        border: round $border-muted;
        padding: 0 1;
    }

    MessageWidget.assistant {
        background: transparent;
        border: none;
    }

    MessageWidget.system {
        background: transparent;
        border: none;
        text-style: italic;
    }

    MessageWidget.error {
        background: $error-bg;
        border: round $error;
    }

    MessageWidget.error .message-header {
        color: $error;
        text-style: bold;
    }

    MessageWidget.error .message-content {
        color: $error;
    }

    MessageWidget .message-header {
        height: 1;
        margin-bottom: 0;
        text-style: bold;
    }

    MessageWidget .message-header.user { color: $success; }
    MessageWidget .message-header.assistant { color: $primary; }
    MessageWidget .message-header.system { color: $text-muted; }
    MessageWidget .message-header.error { color: $error; }

    MessageWidget .message-content {
        width: 100%;
        color: $text;
        max-height: 20;
        overflow-y: auto;
    }

    /* Input - compact at bottom */
    InputWidget {
        dock: bottom;
        height: auto;
        max-height: 10;
        padding: 1;
        background: $panel;
        border-top: solid $border-strong;
        margin: 0;
    }

    InputWidget .input-row {
        width: 100%;
        height: auto;
        padding: 0;
        margin: 0 0 1 0;
    }

    InputWidget .prompt-indicator {
        width: 2;
        height: 3;
        color: $primary;
        text-style: bold;
    }

    InputWidget .prompt-indicator.will-submit {
        color: $success;
        text-style: bold;
    }

    InputWidget SubmitTextArea,
    InputWidget TextArea {
        width: 1fr;
        height: auto;
        min-height: 3;
        max-height: 6;
        border: thick $primary;
        background: $background;
        color: $text;
        padding: 1;
        scrollbar-gutter: stable;
        text-style: none;
    }

    InputWidget SubmitTextArea:focus,
    InputWidget TextArea:focus {
        border: thick $primary;
        background: $background;
        text-style: bold;
    }

    InputWidget .input-hint {
        width: 100%;
        height: 1;
        color: $text-muted;
        text-align: right;
        padding: 0;
        margin: 0;
        text-style: italic;
    }

    /* Tool calls - compact */
    ToolCallWidget {
        width: 100%;
        padding: 0 1;
        margin: 0 0 1 0;
        background: $panel;
        border: round $border-muted;
        height: auto;
    }

    ToolCallWidget.pending { border: round $warning; }
    ToolCallWidget.success { border: round $success; }
    ToolCallWidget.error { border: round $error; }

    ToolCallWidget .tool-header { height: 1; color: $text-muted; }
    ToolCallWidget .tool-status { color: $warning; }
    ToolCallWidget.success .tool-status { color: $success; }
    ToolCallWidget.error .tool-status { color: $error; }

    ToolCallWidget .tool-hint {
        height: 1;
        color: $text-muted;
        text-style: italic;
        margin-top: 0;
    }

    /* Collapsed state */
    ToolCallWidget.collapsed {
        height: 2;
        max-height: 2;
        overflow: hidden;
    }

    ToolCallWidget.collapsed .tool-hint {
        display: none;
    }

    /* Thinking - compact */
    ThinkingWidget {
        width: 100%;
        padding: 0 1;
        margin: 0 0 1 0;
        background: $panel;
        border: round $border-muted;
        color: $text-muted;
    }

    ThinkingWidget .thinking-header {
        height: 1;
        color: $primary;
        margin-bottom: 0;
    }

    ThinkingWidget .thinking-content {
        width: 100%;
        max-height: 10;
        overflow-y: auto;
        text-style: italic;
    }

    /* Containers */
    #tool-calls-container,
    #thinking-container {
        width: 100%;
        padding: 0 1;
        margin: 0 0 1 0;
        display: none;
    }

    #tool-calls-container.visible,
    #thinking-container.visible {
        display: block;
    }

    #session-picker {
        width: 80%;
        max-width: 100;
        height: 70%;
        padding: 1 2;
        background: $panel;
        border: round $border-strong;
    }

    #session-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #session-filter {
        width: 100%;
        margin-bottom: 1;
    }

    #session-help {
        color: $text-muted;
        margin-top: 1;
    }

    /* Session restore progress modal */
    #restore-progress-container {
        width: 60%;
        max-width: 80;
        height: auto;
        padding: 2 3;
        background: $panel;
        border: round $border-strong;
        align: center_middle;
    }

    .progress-title {
        text-style: bold;
        text-align: center;
        margin-bottom: 2;
        color: $primary;
    }

    .session-progress {
        width: 100%;
        margin: 1 0;
    }

    #progress-label {
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }

    Footer {
        background: $panel;
        color: $text-muted;
        border-top: solid $border-muted;
    }
    """
    )

    BINDINGS = [
        Binding("ctrl+c", "quit", "Exit", show=True),
        Binding("ctrl+l", "clear", "Clear", show=True),
        Binding("escape", "focus_input", "Focus Input", show=False),
        # Phase 1.6 Enhanced keyboard shortcuts
        Binding("ctrl+t", "toggle_thinking", "Toggle Thinking", show=True),
        Binding("ctrl+y", "toggle_tools", "Toggle Tools", show=True),
        Binding("ctrl+d", "toggle_details", "Toggle Details", show=True),
        Binding("ctrl+x", "cancel_stream", "Cancel", show=True),
        Binding("ctrl+g", "resume_any_session", "Resume Any Session", show=True),
        Binding("ctrl+p", "resume_project_session", "Resume Project Session", show=True),
        Binding("ctrl+r", "resume_session", "Resume Session", show=True),
        Binding("ctrl+s", "save_session", "Save Session", show=True),
        Binding("ctrl+e", "export_session", "Export Session", show=True),
        Binding("ctrl+slash", "show_help", "Help", show=True),
        Binding("ctrl+up", "scroll_up", "Scroll Up", show=False),
        Binding("ctrl+down", "scroll_down", "Scroll Down", show=False),
        Binding("ctrl+home", "scroll_top", "Scroll Top", show=False),
        Binding("ctrl+end", "scroll_bottom", "Scroll Bottom", show=False),
        # Phase 3: Theme switching
        Binding("ctrl+right", "next_theme", "Next Theme", show=True),
        Binding("ctrl+left", "prev_theme", "Previous Theme", show=False),
    ]

    def __init__(
        self,
        agent: Optional["UIAgentProtocol"] = None,
        provider: str = "anthropic",
        model: str = "claude-3-5-sonnet",
        stream: bool = True,
        on_message: Optional[Callable[[str], Any]] = None,
        settings: Optional["Settings"] = None,
        theme: str = "default",
        keybindings: Optional[KeybindingConfig] = None,
        **kwargs,
    ) -> None:
        """Initialize Victor TUI.

        Args:
            agent: Optional UIAgentProtocol instance (orchestrator or compatible)
            provider: Provider name for display
            model: Model name for display
            stream: Whether to stream responses
            on_message: Callback when user sends a message
            settings: Optional Settings instance for slash commands
            theme: Theme name (default, dark, light, high_contrast, dracula, nord)
            keybindings: Optional keybinding configuration
        """
        super().__init__(**kwargs)
        self.agent = agent
        self.provider = provider
        self.model = model
        self.stream = stream
        self.on_message = on_message
        self.settings = settings
        self._conversation_log: VirtualScrollContainer | None = None
        self._input_widget: InputWidget | None = None
        self._status_bar: StatusBar | None = None
        self._jump_button: Button | None = None
        self._is_processing = False
        self._current_tool_widget: ToolCallWidget | None = None
        self._tool_widgets: list[ToolCallWidget] = []
        self._tool_widget_limit = 5
        self._thinking_widget: ThinkingWidget | None = None
        self._slash_handler = None  # Initialized in on_mount when conversation_log is ready
        self._console_adapter = None
        self._session_messages: list[Message] = []
        self._cancellation_timer = None
        self._poll_timer = None

        # Theme support
        self._current_theme = theme
        self._apply_theme(theme)

        # Keybinding support
        self._keybindings = keybindings or KeybindingConfig(
            bindings=DEFAULT_KEYBINDINGS.copy(), preset_name="default"
        )

    def compose(self) -> ComposeResult:
        """Compose the TUI layout."""
        yield StatusBar(provider=self.provider, model=self.model)
        with Container(id="main-container"):
            with Vertical(id="conversation-area"):
                yield VirtualScrollContainer(id="conversation-log")
                yield Button("Jump to bottom", id="jump-to-bottom", variant="default")
            with Container(id="thinking-container"):
                yield ThinkingWidget(id="thinking-widget")
            with Container(id="tool-calls-container"):
                pass  # Tool calls added dynamically
            yield InputWidget(id="input-widget")
        yield Footer()

    def on_mount(self) -> None:
        """Handle mount event."""
        self._conversation_log = self.query_one("#conversation-log", VirtualScrollContainer)
        self._input_widget = self.query_one("#input-widget", InputWidget)
        self._thinking_widget = self.query_one("#thinking-widget", ThinkingWidget)
        self._status_bar = self.query_one(StatusBar)
        self._jump_button = self.query_one("#jump-to-bottom", Button)

        # Initialize slash command handler with TUI console adapter
        if self.settings:
            from victor.ui.slash_commands import SlashCommandHandler

            self._console_adapter = TUIConsoleAdapter(
                self._conversation_log,
                on_line=lambda line: self._record_message("system", line),
            )
            self._slash_handler = SlashCommandHandler(
                console=self._console_adapter,
                settings=self.settings,
                agent=self.agent,
            )
            # Override /exit handler to use TUI exit via public API
            self._slash_handler.set_exit_callback(self.exit)

        # Show welcome message
        self._add_system_message(f"Connected to {self.provider}/{self.model}")
        self._add_system_message("Type /help for commands, Ctrl+C to exit")
        self._set_status("Idle", "idle")
        self._update_jump_to_bottom()

        # Focus input
        self._input_widget.focus_input()

    async def on_input_widget_submitted(self, event: InputWidget.Submitted) -> None:
        """Handle input submission from the custom InputWidget."""
        message = event.value.strip()
        if not message:
            return

        # Add to history before clearing
        self._input_widget.add_to_history(message)

        # Clear input
        self._input_widget.clear()

        # Handle commands
        if message.startswith("/"):
            await self._handle_command(message)
            return

        # Add user message to log
        self._add_user_message(message)

        # Process message directly (not using @work decorator to avoid thread issues)
        await self._process_message_async(message)

    async def _handle_command(self, command: str) -> None:
        """Handle slash commands.

        Delegates to SlashCommandHandler for all commands, which provides
        41+ commands dynamically registered at startup.
        """
        cmd = command.lower().strip()

        # Handle TUI-specific exit (must bypass SlashCommandHandler's sys.exit)
        if cmd in ("/exit", "/quit", "/bye"):
            self.exit()
            return

        # Handle /clear in TUI to also clear the visual log
        if cmd in ("/clear", "/reset"):
            self._conversation_log.clear()
            if self.agent:
                self.agent.reset_conversation()
            self._conversation_log.add_system_message("Conversation cleared")
            self._input_widget.focus_input()
            return

        # Delegate to SlashCommandHandler if available
        if self._slash_handler:
            try:
                await self._slash_handler.execute(command)
            except Exception as e:
                self._add_error_message(f"Command error: {e}")
        else:
            # Fallback for when settings not provided
            if cmd == "/help":
                self._add_system_message("Available commands:")
                self._add_system_message("  /clear - Clear conversation")
                self._add_system_message("  /exit  - Exit application")
                self._add_system_message("  /help  - Show this help")
            else:
                self._add_system_message(f"Unknown command: {command}")

        self._input_widget.focus_input()

    async def _process_message_async(self, message: str) -> None:
        """Process user message and get response."""
        if self._is_processing:
            return

        self._is_processing = True
        self._set_status("Working", "busy")

        try:
            if self.agent:
                await self._process_with_agent(message)
            elif self.on_message:
                # Callback mode for external handling
                result = self.on_message(message)
                if asyncio.iscoroutine(result):
                    result = await result
                if result:
                    self._add_assistant_message(str(result))
            else:
                # Demo mode
                self._add_assistant_message(
                    f"You said: *{message}*\n\n" "(No agent configured - running in demo mode)",
                )
        except Exception as e:
            self._add_error_message(str(e))
        finally:
            self._is_processing = False
            self._set_status("Idle", "idle")
            if self._input_widget:
                # Call directly without _call_ui since we're in the UI thread
                try:
                    self._input_widget.focus_input()
                except Exception:
                    pass

    async def _process_with_agent(self, message: str) -> None:
        """Process message with the agent."""
        if not self.agent:
            return

        if self.stream and self.agent.provider.supports_streaming():
            await self._stream_response(message)
        else:
            response = await self.agent.chat(message)
            try:
                self._add_assistant_message(response.content)
            except Exception:
                pass

    async def _stream_response(self, message: str) -> None:
        """Stream response from agent."""
        content_buffer = ""

        # Start streaming display
        await self._start_streaming_ui()
        self._set_status("Streaming", "streaming")

        try:
            async for chunk in self.agent.stream_chat(message):
                # Handle different chunk types
                if hasattr(chunk, "type"):
                    chunk_type = chunk.type

                    if chunk_type == "content":
                        content_buffer += chunk.content or ""
                        if self._conversation_log:
                            try:
                                self._conversation_log.update_streaming(content_buffer)
                            except Exception:
                                pass
                            # Update jump-to-bottom button visibility during streaming
                            try:
                                self._update_jump_to_bottom()
                            except Exception:
                                pass

                    elif chunk_type == "thinking_start":
                        try:
                            self._show_thinking()
                        except Exception:
                            pass

                    elif chunk_type == "thinking":
                        try:
                            self._update_thinking(chunk.content or "")
                        except Exception:
                            pass

                    elif chunk_type == "thinking_end":
                        try:
                            self._hide_thinking()
                        except Exception:
                            pass

                    elif chunk_type == "tool_start":
                        try:
                            self._show_tool_call(
                                chunk.tool_name or "unknown",
                                chunk.arguments or {},
                            )
                        except Exception:
                            pass

                    elif chunk_type == "tool_end":
                        try:
                            self._finish_tool_call(
                                success=chunk.success if hasattr(chunk, "success") else True,
                                elapsed=chunk.elapsed if hasattr(chunk, "elapsed") else None,
                                error_message=chunk.error if hasattr(chunk, "error") else None,
                            )
                        except Exception:
                            pass

                elif hasattr(chunk, "content") and chunk.content:
                    # Simple content chunk
                    content_buffer += chunk.content
                    if self._conversation_log:
                        try:
                            self._conversation_log.update_streaming(content_buffer)
                        except Exception:
                            pass
                        # Update jump-to-bottom button visibility during streaming
                        try:
                            self._update_jump_to_bottom()
                        except Exception:
                            pass

        finally:
            # Finish streaming
            if self._conversation_log:
                try:
                    self._conversation_log.finish_streaming()
                except Exception:
                    pass
            try:
                self._hide_thinking()
            except Exception:
                pass
            self._set_status("Idle", "idle")
            if content_buffer.strip():
                self._record_message("assistant", content_buffer)

    def _show_thinking(self) -> None:
        """Show thinking panel."""
        container = self.query_one("#thinking-container")
        container.add_class("visible")
        self._thinking_widget.update_content("")

    def _update_thinking(self, content: str) -> None:
        """Update thinking content."""
        self._thinking_widget.update_content(content)

    def _hide_thinking(self) -> None:
        """Hide thinking panel."""
        container = self.query_one("#thinking-container")
        container.remove_class("visible")

    def _show_tool_call(self, tool_name: str, arguments: dict) -> None:
        """Show tool call widget."""
        container = self.query_one("#tool-calls-container")
        container.add_class("visible")

        # Create tool widget
        self._current_tool_widget = ToolCallWidget(
            tool_name=tool_name,
            arguments=arguments,
            status="pending",
        )
        container.mount(self._current_tool_widget)
        self._tool_widgets.append(self._current_tool_widget)
        self._prune_tool_widgets()

    def _finish_tool_call(
        self,
        success: bool = True,
        elapsed: float | None = None,
        error_message: str | None = None,
    ) -> None:
        """Finish current tool call.

        Args:
            success: Whether the tool call succeeded
            elapsed: Optional elapsed time in seconds
            error_message: Optional error message for failed calls
        """
        if self._current_tool_widget:
            status = "success" if success else "error"
            finished_widget = self._current_tool_widget
            finished_widget.update_status(status, elapsed, error_message)
            self._current_tool_widget = None
            self._schedule_tool_widget_cleanup(finished_widget)
            self._prune_tool_widgets()

    def _schedule_tool_widget_cleanup(self, widget: ToolCallWidget) -> None:
        def _remove() -> None:
            self._remove_tool_widget(widget)

        self.set_timer(6.0, _remove)

    def _remove_tool_widget(self, widget: ToolCallWidget) -> None:
        if widget in self._tool_widgets:
            self._tool_widgets.remove(widget)
        try:
            widget.remove()
        except Exception:
            pass
        if not self._tool_widgets:
            container = self.query_one("#tool-calls-container")
            container.remove_class("visible")

    def _prune_tool_widgets(self) -> None:
        """Remove oldest tool widgets if limit exceeded.

        Also hides the container if no widgets remain.
        """
        while len(self._tool_widgets) > self._tool_widget_limit:
            widget = self._tool_widgets.pop(0)
            try:
                widget.remove()
            except Exception:
                pass
        # Hide container if no widgets remain (defensive check)
        if not self._tool_widgets:
            try:
                container = self.query_one("#tool-calls-container")
                container.remove_class("visible")
            except Exception:
                pass

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_clear(self) -> None:
        """Clear conversation."""
        self._conversation_log.clear()
        self._session_messages.clear()
        if self.agent:
            self.agent.reset_conversation()
        self._add_system_message("Conversation cleared")

    def action_focus_input(self) -> None:
        """Focus the input widget."""
        self._input_widget.focus_input()

    def action_toggle_thinking(self) -> None:
        """Toggle the thinking panel visibility."""
        container = self.query_one("#thinking-container")
        if "visible" in container.classes:
            container.remove_class("visible")
            self._add_system_message("Thinking panel hidden")
        else:
            container.add_class("visible")
            self._add_system_message("Thinking panel shown")

    def action_toggle_tools(self) -> None:
        """Toggle the tool calls panel visibility."""
        container = self.query_one("#tool-calls-container")
        if "visible" in container.classes:
            container.remove_class("visible")
            self._add_system_message("Tools panel hidden")
        else:
            container.add_class("visible")
            self._add_system_message("Tools panel shown")

    def action_toggle_details(self) -> None:
        """Toggle both thinking and tools panel visibility (details mode)."""
        thinking_container = self.query_one("#thinking-container")
        tools_container = self.query_one("#tool-calls-container")
        thinking_visible = "visible" in thinking_container.classes
        tools_visible = "visible" in tools_container.classes

        # If either is visible, hide both. If neither is visible, show both.
        if thinking_visible or tools_visible:
            if thinking_visible:
                thinking_container.remove_class("visible")
            if tools_visible:
                tools_container.remove_class("visible")
            self._add_system_message("Details panels hidden")
        else:
            thinking_container.add_class("visible")
            tools_container.add_class("visible")
            self._add_system_message("Details panels shown")

    def action_cancel_stream(self) -> None:
        """Request cancellation of the current stream if active with enhanced feedback."""
        if not self.agent:
            self._add_system_message("No active agent to cancel")
            return

        if not hasattr(self.agent, "is_streaming") or not self.agent.is_streaming():
            self._add_system_message("No active stream to cancel")
            return

        if hasattr(self.agent, "request_cancellation"):
            self.agent.request_cancellation()
            self._add_system_message("Cancellation requested...")
            self._set_status("Canceling...", "busy")

            # Use timer for polling (100ms intervals, up to 5 seconds)
            self._poll_timer = self.set_timer(
                0.1, self._poll_cancellation_sync, repeat=50  # Poll for 5 seconds (50 * 0.1)
            )

            # Set timeout for cancellation
            self._cancellation_timer = self.set_timer(5.0, self._on_cancellation_timeout)
        else:
            self._add_system_message("Streaming cancellation not supported")

    def _poll_cancellation_sync(self) -> None:
        """Check if cancellation completed (synchronous version for Textual timer)."""
        if not self.agent or not self.agent.is_streaming():
            # Cancellation succeeded
            self._set_status("Canceled", "idle")
            self._add_system_message("✓ Stream canceled successfully")

            # Stop timers
            if hasattr(self, "_poll_timer") and self._poll_timer:
                self._poll_timer.stop()
            if hasattr(self, "_cancellation_timer") and self._cancellation_timer:
                self._cancellation_timer.stop()

    def _on_cancellation_timeout(self) -> None:
        """Handle cancellation timeout."""
        if self.agent and self.agent.is_streaming():
            self._set_status("Cancellation timed out", "error")
            self._add_system_message("⚠ Cancellation timed out - stream may still be active")

        # Stop poll timer
        if hasattr(self, "_poll_timer") and self._poll_timer:
            self._poll_timer.stop()

    def action_resume_session(self) -> None:
        """Show session picker and load a saved session."""
        from victor.ui.tui.session import SessionManager

        manager = SessionManager()
        sessions = manager.list_sessions(limit=20)
        if not sessions:
            self._add_system_message("No saved sessions found")
            return
        self.push_screen(SessionPicker(sessions), self._handle_session_choice)

    def action_resume_project_session(self) -> None:
        """Resume a project session from the project SQLite database."""
        from victor.agent.sqlite_session_persistence import get_sqlite_session_persistence

        persistence = get_sqlite_session_persistence()
        sessions = persistence.list_sessions(limit=20)
        if not sessions:
            self._add_system_message("No project sessions found")
            return
        mapped = [
            {
                "id": session["session_id"],
                "name": session.get("title") or session["session_id"][:8],
                "source": "Project",
                "key": f"project:{session['session_id']}",
                "provider": session.get("provider", ""),
                "model": session.get("model", ""),
                "updated_at": session.get("updated_at", ""),
                "message_count": session.get("message_count", 0),
            }
            for session in sessions
        ]
        self.push_screen(
            SessionPicker(mapped, title="Resume Project Session"),
            self._handle_project_session_choice,
        )

    def action_resume_any_session(self) -> None:
        """Resume a session from either TUI or project history."""
        from victor.agent.sqlite_session_persistence import get_sqlite_session_persistence
        from victor.ui.tui.session import SessionManager

        manager = SessionManager()
        tui_sessions = manager.list_sessions(limit=20)

        persistence = get_sqlite_session_persistence()
        project_sessions = persistence.list_sessions(limit=20)

        combined = []
        for session in tui_sessions:
            combined.append(
                {
                    "id": session["id"],
                    "name": session["name"],
                    "source": "TUI",
                    "key": f"tui:{session['id']}",
                    "provider": session.get("provider", ""),
                    "model": session.get("model", ""),
                    "updated_at": session.get("updated_at", ""),
                    "message_count": session.get("message_count", 0),
                }
            )
        for session in project_sessions:
            combined.append(
                {
                    "id": session["session_id"],
                    "name": session.get("title") or session["session_id"][:8],
                    "source": "Project",
                    "key": f"project:{session['session_id']}",
                    "provider": session.get("provider", ""),
                    "model": session.get("model", ""),
                    "updated_at": session.get("updated_at", ""),
                    "message_count": session.get("message_count", 0),
                }
            )

        if not combined:
            self._add_system_message("No sessions found")
            return

        combined.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        self.push_screen(
            SessionPicker(combined, title="Resume Session (All Sources)"),
            self._handle_any_session_choice,
        )

    def _handle_session_choice(self, session_id: Optional[str]) -> None:
        if not session_id:
            self._add_system_message("Session restore cancelled")
            return
        if session_id.startswith("tui:"):
            self._load_session(session_id.split(":", 1)[SESSION_UUID_INDEX])
        else:
            self._load_session(session_id)

    def _handle_project_session_choice(self, session_id: Optional[str]) -> None:
        if not session_id:
            self._add_system_message("Project session restore cancelled")
            return
        if session_id.startswith("project:"):
            self._load_project_session(session_id.split(":", 1)[SESSION_UUID_INDEX])
        else:
            self._load_project_session(session_id)

    def _handle_any_session_choice(self, session_key: Optional[str]) -> None:
        if not session_key:
            self._add_system_message("Session restore cancelled")
            return
        if session_key.startswith("tui:"):
            self._load_session(session_key.split(":", 1)[SESSION_UUID_INDEX])
        elif session_key.startswith("project:"):
            self._load_project_session(session_key.split(":", 1)[SESSION_UUID_INDEX])
        else:
            self._load_session(session_key)

    def _load_session(self, session_id: str) -> None:
        """Load a TUI session with progress indication."""
        import asyncio
        from victor.ui.tui.session import SessionManager

        manager = SessionManager()
        session = manager.load(session_id)
        if not session:
            self._add_error_message(f"Session not found: {session_id}")
            return

        message_count = len(session.messages)
        session_name = session.name or session.id[:8]

        # Show progress modal for sessions > 20 messages
        if message_count > 20:
            progress = SessionRestoreProgress(message_count, session_name)
            self.push_screen(progress)

            try:
                if self._conversation_log:
                    self._conversation_log.clear()

                self._session_messages = list(session.messages)
                for i, msg in enumerate(session.messages):
                    self._render_message(msg.role, msg.content)
                    progress.update(i + 1)
                    # Yield to UI to update progress bar
                    asyncio.sleep(0)

                self._restore_agent_conversation(session.messages)
            finally:
                self.pop_screen()
        else:
            # Fast path for small sessions
            if message_count > 50:
                self._add_system_message(f"Loading {message_count} messages...")

            if self._conversation_log:
                self._conversation_log.clear()

            self._session_messages = list(session.messages)
            for i, msg in enumerate(session.messages):
                self._render_message(msg.role, msg.content)
                # Show progress for large sessions (old behavior)
                if message_count > 50 and (i + 1) % 25 == 0:
                    self._add_system_message(f"Loading... {i + 1}/{message_count}")

            self._restore_agent_conversation(session.messages)

        self._add_system_message(f"Session loaded: {session_name} ({message_count} messages)")

    def _load_project_session(self, session_id: str) -> None:
        """Load a project session with progress indication."""
        import asyncio
        from victor.agent.conversation_state import ConversationStateMachine
        from victor.agent.message_history import MessageHistory
        from victor.agent.sqlite_session_persistence import get_sqlite_session_persistence

        persistence = get_sqlite_session_persistence()
        session = persistence.load_session(session_id)
        if not session:
            self._add_error_message(f"Project session not found: {session_id}")
            return

        conversation = session.get("conversation", {})
        history = MessageHistory.from_dict(conversation) if conversation else MessageHistory()
        messages = history.messages

        message_count = len(messages)
        metadata = session.get("metadata", {})
        session_name = metadata.get("title") or session_id[:8]

        # Show progress modal for sessions > 20 messages
        if message_count > 20:
            progress = SessionRestoreProgress(message_count, f"Project: {session_name}")
            self.push_screen(progress)

            try:
                if self._conversation_log:
                    self._conversation_log.clear()

                self._session_messages = []
                for i, msg in enumerate(messages):
                    role = msg.role
                    content = msg.content
                    if role == "tool":
                        role = "system"
                        if msg.name:
                            content = f"Tool result ({msg.name}): {content}"
                        else:
                            content = f"Tool result: {content}"
                    if not content and getattr(msg, "tool_calls", None):
                        content = "Tool calls requested."
                    if not content:
                        continue
                    self._render_message(role, content)
                    self._session_messages.append(Message(role=role, content=content, metadata={}))
                    progress.update(i + 1)
                    # Yield to UI to update progress bar
                    asyncio.sleep(0)
            finally:
                self.pop_screen()
        else:
            # Fast path for small sessions
            if message_count > 50:
                self._add_system_message(
                    f"Loading {message_count} messages from project session..."
                )

            if self._conversation_log:
                self._conversation_log.clear()

            self._session_messages = []
            for i, msg in enumerate(messages):
                role = msg.role
                content = msg.content
                if role == "tool":
                    role = "system"
                    if msg.name:
                        content = f"Tool result ({msg.name}): {content}"
                    else:
                        content = f"Tool result: {content}"
                if not content and getattr(msg, "tool_calls", None):
                    content = "Tool calls requested."
                if not content:
                    continue
                self._render_message(role, content)
                self._session_messages.append(Message(role=role, content=content, metadata={}))
                # Show progress for large sessions (old behavior)
                if message_count > 50 and (i + 1) % 25 == 0:
                    self._add_system_message(f"Loading... {i + 1}/{message_count}")

        if self.agent:
            self.agent.conversation = history
            self.agent.active_session_id = session_id
            conversation_state = session.get("conversation_state")
            if conversation_state:
                try:
                    self.agent.conversation_state = ConversationStateMachine.from_dict(
                        conversation_state
                    )
                except Exception as exc:
                    self._add_error_message(f"Failed to restore conversation state: {exc}")

        self._add_system_message(
            f"Project session loaded: {session_name} ({message_count} messages)"
        )

    def _render_message(self, role: str, content: str) -> None:
        if not self._conversation_log:
            return
        if role == "user":
            self._conversation_log.add_user_message(content)
        elif role == "assistant":
            self._conversation_log.add_assistant_message(content)
        elif role == "system":
            self._conversation_log.add_system_message(content)
        elif role == "error":
            self._conversation_log.add_error_message(content)

    def _restore_agent_conversation(self, messages: list[Message]) -> None:
        if not self.agent or not hasattr(self.agent, "conversation"):
            return
        try:
            from victor.agent.message_history import MessageHistory

            system_prompt = ""
            try:
                system_prompt = self.agent.conversation.system_prompt
            except Exception:
                system_prompt = ""

            history = MessageHistory(system_prompt=system_prompt)
            for msg in messages:
                role = msg.role
                if role == "error":
                    role = "system"
                if role not in ("system", "user", "assistant", "tool"):
                    continue
                history.add_message(role, msg.content)
            self.agent.conversation = history
        except Exception as exc:
            self._add_error_message(f"Failed to restore agent context: {exc}")

    def action_save_session(self) -> None:
        """Save the current session."""
        try:
            from victor.ui.tui.session import SessionManager

            manager = SessionManager()
            # Create a session from current conversation
            session = manager.create_session(
                name=f"Session {self.provider}/{self.model}",
                provider=self.provider,
                model=self.model,
            )
            session.messages = list(self._session_messages)
            manager.save(session)
            self._add_system_message(f"Session saved: {session.id[:8]} (use Ctrl+R to resume)")
        except Exception as e:
            self._add_error_message(f"Failed to save session: {e}")

    def action_export_session(self) -> None:
        """Export the current session to a markdown file."""
        import tempfile
        from pathlib import Path

        try:
            from victor.ui.tui.session import Session, SessionManager

            # Create a temporary session for export
            session = Session(
                name=f"Session {self.provider}/{self.model}",
                provider=self.provider,
                model=self.model,
            )
            session.messages = list(self._session_messages)

            # Export to temp file first, then show user the path
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False, prefix="victor_session_"
            ) as f:
                f.write(session.to_markdown())
                temp_path = Path(f.name)

            self._add_system_message(f"Session exported to: {temp_path}")
            self._add_system_message(f"Message count: {len(self._session_messages)}")
        except Exception as e:
            self._add_error_message(f"Failed to export session: {e}")

    def action_show_help(self) -> None:
        """Show help overlay with keyboard shortcuts."""
        help_text = """
Keyboard Shortcuts:
  Enter        Send message
  Shift+Enter  Add newline
  Ctrl+C       Exit
  Ctrl+L       Clear conversation
  Ctrl+T       Toggle thinking panel
  Ctrl+Y       Toggle tools panel
  Ctrl+D       Toggle all details (thinking + tools)
  Ctrl+X       Cancel streaming
  Ctrl+G       Resume any session
  Ctrl+P       Resume project session
  Ctrl+R       Resume TUI session
  Ctrl+S       Save session
  Ctrl+E       Export session to markdown
  Ctrl+/       Show this help
  Ctrl+→/←     Next/Previous theme
  Ctrl+↑/↓     Scroll conversation
  Ctrl+Home/End Jump to top/bottom
  ↑/↓          Navigate input history
  Escape       Focus input

Slash Commands:
  /help        Show all commands
  /clear       Clear conversation
  /model       Switch model
  /provider    Switch provider
  /exit        Exit TUI

Available Themes:
  default      Default dark theme
  dark         GitHub dark theme
  light        Light theme
  high_contrast High contrast theme
  dracula      Dracula theme
  nord         Nord theme
"""
        self._add_system_message(help_text.strip())

    def action_scroll_up(self) -> None:
        """Scroll conversation up."""
        if not self._conversation_log:
            return
        self._conversation_log.disable_auto_scroll()
        self._conversation_log.scroll_up(animate=False)
        self._conversation_log.update_auto_scroll_state()
        self._update_jump_to_bottom()

    def action_scroll_down(self) -> None:
        """Scroll conversation down."""
        if not self._conversation_log:
            return
        self._conversation_log.scroll_down(animate=False)
        self._conversation_log.update_auto_scroll_state()
        self._update_jump_to_bottom()

    def action_scroll_top(self) -> None:
        """Scroll to top of conversation."""
        if not self._conversation_log:
            return
        self._conversation_log.disable_auto_scroll()
        self._conversation_log.scroll_home(animate=False)
        self._conversation_log.update_auto_scroll_state()
        self._update_jump_to_bottom()

    def action_scroll_bottom(self) -> None:
        """Scroll to bottom of conversation."""
        if not self._conversation_log:
            return
        self._conversation_log.scroll_to_bottom(animate=False)
        self._update_jump_to_bottom()

    def on_scroll(self, event) -> None:
        if event.sender is self._conversation_log:
            self._update_jump_to_bottom()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "jump-to-bottom":
            return
        if self._conversation_log:
            self._conversation_log.scroll_to_bottom(animate=False)
        self._update_jump_to_bottom()

    def add_message(self, content: str, role: str = "assistant") -> None:
        """Add a message to the conversation log.

        This method can be called from outside the TUI to add messages.

        Args:
            content: Message content
            role: Message role (user, assistant, system, error)
        """
        if role == "user":
            self._add_user_message(content)
        elif role == "assistant":
            self._add_assistant_message(content)
        elif role == "system":
            self._add_system_message(content)
        elif role == "error":
            self._add_error_message(content)

    def _call_ui(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        try:
            self.call_from_thread(func, *args, **kwargs)
        except RuntimeError:
            func(*args, **kwargs)

    def _set_status(self, status: str, state: str = "idle") -> None:
        if not self._status_bar:
            return
        try:
            self._status_bar.update_status(status, state)
        except Exception:
            pass

    def _update_jump_to_bottom(self) -> None:
        if not self._jump_button or not self._conversation_log:
            return
        if self._conversation_log.auto_scroll_enabled:
            self._jump_button.remove_class("visible")
        else:
            self._jump_button.add_class("visible")

    async def _start_streaming_ui(self) -> None:
        if not self._conversation_log:
            return
        started = asyncio.Event()

        def _start() -> None:
            self._conversation_log.start_streaming()
            started.set()

        self._call_ui(_start)
        await started.wait()

    def _record_message(self, role: str, content: str, **metadata: Any) -> None:
        if not content:
            return
        self._session_messages.append(Message(role=role, content=content, metadata=metadata))

    def _add_user_message(self, content: str) -> None:
        if self._conversation_log:
            self._conversation_log.add_user_message(content)
        self._record_message("user", content)

    def _add_assistant_message(self, content: str) -> None:
        if self._conversation_log:
            self._conversation_log.add_assistant_message(content)
        self._record_message("assistant", content)

    def _add_system_message(self, content: str) -> None:
        if self._conversation_log:
            self._conversation_log.add_system_message(content)
        self._record_message("system", content)

    def _add_error_message(self, content: str) -> None:
        if self._conversation_log:
            self._conversation_log.add_error_message(content)
        self._record_message("error", content)

    # =========================================================================
    # THEME MANAGEMENT
    # =========================================================================

    def _apply_theme(self, theme_name: str) -> None:
        """Apply a theme to the TUI.

        Args:
            theme_name: Name of the theme to apply
        """
        try:
            theme = get_theme(theme_name)
            # Update CSS variables
            self.stylesheet._update_colors(theme.to_dict())
            self._current_theme = theme_name
        except Exception as e:
            # Silently fail if theme not found
            pass

    def action_next_theme(self) -> None:
        """Cycle to the next available theme."""
        theme_names = list(THEMES.keys())
        if not theme_names:
            return

        try:
            current_idx = theme_names.index(self._current_theme)
        except ValueError:
            current_idx = 0

        next_idx = (current_idx + 1) % len(theme_names)
        next_theme = theme_names[next_idx]
        self._apply_theme(next_theme)
        self._add_system_message(f"Theme changed to: {THEMES[next_theme].display_name}")

    def action_prev_theme(self) -> None:
        """Cycle to the previous available theme."""
        theme_names = list(THEMES.keys())
        if not theme_names:
            return

        try:
            current_idx = theme_names.index(self._current_theme)
        except ValueError:
            current_idx = 0

        prev_idx = (current_idx - 1) % len(theme_names)
        prev_theme = theme_names[prev_idx]
        self._apply_theme(prev_theme)
        self._add_system_message(f"Theme changed to: {THEMES[prev_theme].display_name}")

    def set_theme(self, theme_name: str) -> None:
        """Set a specific theme by name.

        Args:
            theme_name: Name of the theme to apply
        """
        self._apply_theme(theme_name)
        try:
            theme = get_theme(theme_name)
            self._add_system_message(f"Theme changed to: {theme.display_name}")
        except Exception:
            self._add_system_message(f"Failed to load theme: {theme_name}")

    def get_theme(self) -> str:
        """Get the current theme name.

        Returns:
            Current theme name
        """
        return self._current_theme

    # =========================================================================
    # KEYBINDING MANAGEMENT
    # =========================================================================

    def get_keybinding(self, action: str) -> Optional[str]:
        """Get the keybinding for an action.

        Args:
            action: Action name (e.g., "quit", "clear", "toggle_thinking")

        Returns:
            Key combination string or None if not found
        """
        return self._keybindings.get_binding(action)

    def set_keybinding(self, action: str, keys: str) -> None:
        """Set a custom keybinding for an action.

        Args:
            action: Action name
            keys: Key combination (e.g., "ctrl+x")
        """
        self._keybindings.set_binding(action, keys)
        self._add_system_message(f"Keybinding changed: {action} → {keys}")

    def reset_keybindings(self) -> None:
        """Reset keybindings to default preset."""
        self._keybindings = KeybindingConfig(
            bindings=DEFAULT_KEYBINDINGS.copy(), preset_name="default"
        )
        self._add_system_message("Keybindings reset to default")

    def get_keybindings(self) -> Dict[str, str]:
        """Get all keybindings.

        Returns:
            Dictionary of action → keybinding
        """
        return self._keybindings.to_dict()


async def run_tui(
    agent: Optional["UIAgentProtocol"] = None,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet",
    stream: bool = True,
    settings: Optional["Settings"] = None,
    theme: str = "default",
    keybindings: Optional[KeybindingConfig] = None,
) -> None:
    """Run the Victor TUI.

    Args:
        agent: Optional UIAgentProtocol instance (orchestrator or compatible)
        provider: Provider name for display
        model: Model name for display
        stream: Whether to stream responses
        settings: Optional Settings instance for slash commands
        theme: Theme name (default, dark, light, high_contrast, dracula, nord)
        keybindings: Optional keybinding configuration
    """
    import os

    # Set environment variable to disable event backend dispatcher
    # The dispatcher conflicts with Textual's event loop
    os.environ["VICTOR_TUI_MODE"] = "1"

    app = VictorTUI(
        agent=agent,
        provider=provider,
        model=model,
        stream=stream,
        settings=settings,
        theme=theme,
        keybindings=keybindings,
    )
    try:
        await app.run_async()
    finally:
        # Clean up environment variable
        os.environ.pop("VICTOR_TUI_MODE", None)
