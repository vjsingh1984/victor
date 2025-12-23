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
from textual.widgets import Footer
from victor.ui.tui.theme import THEME_CSS
from victor.ui.tui.widgets import (
    ConversationLog,
    EnhancedConversationLog,
    InputWidget,
    StatusBar,
    ThinkingWidget,
    ToolCallWidget,
    CodeBlock,
    StreamingMessageBlock,
    ToolProgressPanel,
)

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings


class TUIConsoleAdapter:
    """Adapter that captures Rich console output for TUI display.

    SlashCommandHandler uses rich.console.Console for output.
    This adapter captures that output and redirects it to the TUI's
    conversation log as system messages.
    """

    def __init__(self, conversation_log: ConversationLog):
        self._log = conversation_log
        self._buffer = io.StringIO()
        self._console = Console(file=self._buffer, force_terminal=False, width=120)

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
                    self._log.add_system_message(line)

    def __getattr__(self, name):
        """Delegate other console methods to the internal console."""
        return getattr(self._console, name)


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
    │  You                                │
    │  Hello, can you help me?            │
    │                                     │
    │  Victor                             │
    │  Of course! I'd be happy to help... │
    │                                     │  <- ConversationLog
    │                                     │
    │                                     │
    ├─────────────────────────────────────┤
    │ > Type your message...              │  <- InputWidget
    │   Enter to send | Ctrl+C to exit    │
    └─────────────────────────────────────┘
    """

    CSS = THEME_CSS + """
    Screen {
        background: $background;
        color: $text;
    }

    #main-container {
        width: 100%;
        height: 100%;
        padding: 0 2 1 2;
        layout: vertical;
        min-height: 0;
    }

    #conversation-area {
        width: 100%;
        height: 1fr;
        min-height: 0;
        padding: 1 0 1 0;
        layout: vertical;
    }

    ConversationLog {
        height: 1fr;
        min-height: 0;
        background: $panel;
        border: round $border-strong;
        padding: 1 2;
        margin: 0 0 1 0;
        scrollbar-gutter: stable;
    }

    /* Status bar */
    StatusBar {
        dock: top;
        height: 2;
        background: $panel;
        color: $text;
        padding: 0 1;
        border-bottom: solid $border-strong;
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
    }

    StatusBar .provider-info .victor-name {
        color: $primary;
    }

    StatusBar .shortcuts {
        color: $text-muted;
        text-align: right;
    }

    /* Messages */
    MessageWidget {
        width: 100%;
        padding: 1 2;
        margin: 1 0;
        background: $panel;
        border: round $border-muted;
    }

    MessageWidget.user {
        background: $panel-alt;
        border: round $border-strong;
    }

    MessageWidget.assistant {
        background: $panel;
        border: round $border-muted;
    }

    MessageWidget.system {
        background: $panel;
        border: round $border-muted;
        text-style: italic;
    }

    MessageWidget.error {
        background: $panel;
        border: round $error;
    }

    MessageWidget .message-header {
        height: 1;
        margin-bottom: 1;
        text-style: bold;
    }

    MessageWidget .message-header.user { color: $success; }
    MessageWidget .message-header.assistant { color: $primary; }
    MessageWidget .message-header.system { color: $text-muted; }
    MessageWidget .message-header.error { color: $error; }

    MessageWidget .message-content {
        width: 100%;
        color: $text;
    }

    /* Input */
    InputWidget {
        dock: bottom;
        height: auto;
        max-height: 50%;
        padding: 0 2 1 2;
        background: $panel;
        border-top: solid $border-muted;
        margin: 0;
    }

    InputWidget .input-row {
        width: 100%;
        height: auto;
        padding-top: 1;
    }

    InputWidget .prompt-indicator {
        width: 2;
        height: 1;
        color: $primary;
    }

    InputWidget TextArea {
        width: 1fr;
        height: auto;
        min-height: 3;
        max-height: 16;
        border: round $border-strong;
        background: $surface;
        color: $text;
        padding: 1 1;
        scrollbar-gutter: stable;
    }

    InputWidget TextArea:focus {
        border: round $border-strong;
        background: $background;
    }

    InputWidget .input-hint {
        width: 100%;
        height: 1;
        color: $text-muted;
        text-align: right;
    }

    /* Tool calls */
    ToolCallWidget {
        width: 100%;
        padding: 1 2;
        margin: 0 0 1 0;
        background: $panel;
        border: round $border-strong;
    }

    ToolCallWidget.pending { border: round $warning; }
    ToolCallWidget.success { border: round $success; }
    ToolCallWidget.error { border: round $error; }

    ToolCallWidget .tool-header { height: 1; color: $text-muted; }
    ToolCallWidget .tool-status { color: $warning; }
    ToolCallWidget.success .tool-status { color: $success; }
    ToolCallWidget.error .tool-status { color: $error; }

    /* Thinking */
    ThinkingWidget {
        width: 100%;
        padding: 1 2;
        margin: 0 0 1 0;
        background: $panel;
        border: round $border-strong;
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

    /* Containers */
    #tool-calls-container,
    #thinking-container {
        width: 100%;
        padding: 0 2;
        margin: 0 0 1 0;
        display: none;
    }

    #tool-calls-container.visible,
    #thinking-container.visible {
        display: block;
    }

    Footer {
        background: $panel;
        color: $text-muted;
        border-top: solid $border-muted;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Exit", show=True),
        Binding("ctrl+l", "clear", "Clear", show=True),
        Binding("escape", "focus_input", "Focus Input", show=False),
    ]

    def __init__(
        self,
        agent: Optional["AgentOrchestrator"] = None,
        provider: str = "anthropic",
        model: str = "claude-3-5-sonnet",
        stream: bool = True,
        on_message: Optional[Callable[[str], Any]] = None,
        settings: Optional["Settings"] = None,
        **kwargs,
    ) -> None:
        """Initialize Victor TUI.

        Args:
            agent: Optional AgentOrchestrator instance
            provider: Provider name for display
            model: Model name for display
            stream: Whether to stream responses
            on_message: Callback when user sends a message
            settings: Optional Settings instance for slash commands
        """
        super().__init__(**kwargs)
        self.agent = agent
        self.provider = provider
        self.model = model
        self.stream = stream
        self.on_message = on_message
        self.settings = settings
        self._conversation_log: ConversationLog | None = None
        self._input_widget: InputWidget | None = None
        self._is_processing = False
        self._current_tool_widget: ToolCallWidget | None = None
        self._thinking_widget: ThinkingWidget | None = None
        self._slash_handler = None  # Initialized in on_mount when conversation_log is ready
        self._console_adapter = None

    def compose(self) -> ComposeResult:
        """Compose the TUI layout."""
        yield StatusBar(provider=self.provider, model=self.model)
        with Container(id="main-container"):
            with Vertical(id="conversation-area"):
                yield ConversationLog(id="conversation-log")
            with Container(id="thinking-container"):
                yield ThinkingWidget(id="thinking-widget")
            with Container(id="tool-calls-container"):
                pass  # Tool calls added dynamically
            yield InputWidget(id="input-widget")
        yield Footer()

    def on_mount(self) -> None:
        """Handle mount event."""
        self._conversation_log = self.query_one("#conversation-log", ConversationLog)
        self._input_widget = self.query_one("#input-widget", InputWidget)
        self._thinking_widget = self.query_one("#thinking-widget", ThinkingWidget)

        # Initialize slash command handler with TUI console adapter
        if self.settings:
            from victor.ui.slash_commands import SlashCommandHandler

            self._console_adapter = TUIConsoleAdapter(self._conversation_log)
            self._slash_handler = SlashCommandHandler(
                console=self._console_adapter,
                settings=self.settings,
                agent=self.agent,
            )
            # Override /exit handler to use TUI exit
            self._slash_handler._tui_exit_callback = self.exit

        # Show welcome message
        self._conversation_log.add_system_message(f"Connected to {self.provider}/{self.model}")
        self._conversation_log.add_system_message("Type /help for commands, Ctrl+C to exit")

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
        self._conversation_log.add_user_message(message)

        # Process message (note: @work decorator makes this return a Worker, not a coroutine)
        self._process_message(message)

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
                self._conversation_log.add_error_message(f"Command error: {e}")
        else:
            # Fallback for when settings not provided
            if cmd == "/help":
                self._conversation_log.add_system_message("Available commands:")
                self._conversation_log.add_system_message("  /clear - Clear conversation")
                self._conversation_log.add_system_message("  /exit  - Exit application")
                self._conversation_log.add_system_message("  /help  - Show this help")
            else:
                self._conversation_log.add_system_message(f"Unknown command: {command}")

        self._input_widget.focus_input()

    @work(exclusive=True)
    async def _process_message(self, message: str) -> None:
        """Process user message and get response."""
        if self._is_processing:
            return

        self._is_processing = True

        try:
            if self.agent:
                await self._process_with_agent(message)
            elif self.on_message:
                # Callback mode for external handling
                result = self.on_message(message)
                if asyncio.iscoroutine(result):
                    result = await result
                if result:
                    self._conversation_log.add_assistant_message(str(result))
            else:
                # Demo mode
                self._conversation_log.add_assistant_message(
                    f"You said: *{message}*\n\n" "(No agent configured - running in demo mode)"
                )
        except Exception as e:
            self._conversation_log.add_error_message(str(e))
        finally:
            self._is_processing = False
            self._input_widget.focus_input()

    async def _process_with_agent(self, message: str) -> None:
        """Process message with the agent."""
        if not self.agent:
            return

        if self.stream and self.agent.provider.supports_streaming():
            await self._stream_response(message)
        else:
            response = await self.agent.chat(message)
            self._conversation_log.add_assistant_message(response.content)

    async def _stream_response(self, message: str) -> None:
        """Stream response from agent."""
        content_buffer = ""

        # Start streaming display
        self._conversation_log.start_streaming()

        try:
            async for chunk in self.agent.stream_chat(message):
                # Handle different chunk types
                if hasattr(chunk, "type"):
                    chunk_type = chunk.type

                    if chunk_type == "content":
                        content_buffer += chunk.content or ""
                        self._conversation_log.update_streaming(content_buffer)

                    elif chunk_type == "thinking_start":
                        self._show_thinking()

                    elif chunk_type == "thinking":
                        self._update_thinking(chunk.content or "")

                    elif chunk_type == "thinking_end":
                        self._hide_thinking()

                    elif chunk_type == "tool_start":
                        self._show_tool_call(
                            chunk.tool_name or "unknown",
                            chunk.arguments or {},
                        )

                    elif chunk_type == "tool_end":
                        self._finish_tool_call(
                            success=chunk.success if hasattr(chunk, "success") else True,
                            elapsed=chunk.elapsed if hasattr(chunk, "elapsed") else None,
                        )

                elif hasattr(chunk, "content") and chunk.content:
                    # Simple content chunk
                    content_buffer += chunk.content
                    self._conversation_log.update_streaming(content_buffer)

        finally:
            # Finish streaming
            self._conversation_log.finish_streaming()
            self._hide_thinking()

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

    def _finish_tool_call(
        self,
        success: bool = True,
        elapsed: float | None = None,
    ) -> None:
        """Finish current tool call."""
        if self._current_tool_widget:
            status = "success" if success else "error"
            self._current_tool_widget.update_status(status, elapsed)
            self._current_tool_widget = None

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_clear(self) -> None:
        """Clear conversation."""
        self._conversation_log.clear()
        if self.agent:
            self.agent.reset_conversation()
        self._conversation_log.add_system_message("Conversation cleared")

    def action_focus_input(self) -> None:
        """Focus the input widget."""
        self._input_widget.focus_input()

    def add_message(self, content: str, role: str = "assistant") -> None:
        """Add a message to the conversation log.

        This method can be called from outside the TUI to add messages.

        Args:
            content: Message content
            role: Message role (user, assistant, system, error)
        """
        if role == "user":
            self._conversation_log.add_user_message(content)
        elif role == "assistant":
            self._conversation_log.add_assistant_message(content)
        elif role == "system":
            self._conversation_log.add_system_message(content)
        elif role == "error":
            self._conversation_log.add_error_message(content)


async def run_tui(
    agent: Optional["AgentOrchestrator"] = None,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet",
    stream: bool = True,
    settings: Optional["Settings"] = None,
) -> None:
    """Run the Victor TUI.

    Args:
        agent: Optional AgentOrchestrator instance
        provider: Provider name for display
        model: Model name for display
        stream: Whether to stream responses
        settings: Optional Settings instance for slash commands
    """
    app = VictorTUI(
        agent=agent,
        provider=provider,
        model=model,
        stream=stream,
        settings=settings,
    )
    await app.run_async()
