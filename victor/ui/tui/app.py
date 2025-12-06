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

"""Victor TUI Application - Main Textual App."""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Static

from victor.ui.tui.widgets import (
    MessageContainer,
    ChatInput,
    StatusBar,
    ToolIndicator,
    AssistantMessage,
    ConfirmationModal,
)
from victor.agent.safety import ConfirmationRequest, set_confirmation_callback

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.config.settings import Settings
    from victor.ui.commands import SlashCommandHandler

logger = logging.getLogger(__name__)


VICTOR_ASCII_LOGO = """[bold blue]╔═══ VICTOR ═══╗[/] [dim]AI Coding Assistant[/]"""


class WelcomePanel(Static):
    """Welcome panel shown at startup - compact version."""

    DEFAULT_CSS = """
    WelcomePanel {
        margin: 0 1;
        padding: 0 1;
        border: none;
        background: $surface;
        width: 100%;
    }
    """

    def __init__(self, provider: str, model: str, context_loaded: bool = False) -> None:
        """Initialize welcome panel.

        Args:
            provider: LLM provider name.
            model: Model name.
            context_loaded: Whether project context is loaded.
        """
        # Build content string
        context_icon = "[green]✓[/]" if context_loaded else "[dim]○[/]"
        content = (
            f"{VICTOR_ASCII_LOGO}  "
            f"[cyan]{provider}[/]:[cyan]{model}[/]  "
            f"ctx:{context_icon}  "
            f"[dim]/help | Ctrl+D to exit[/]"
        )
        # Static widgets should set content directly, not use compose()
        super().__init__(content)


class VictorApp(App):
    """Victor TUI Application."""

    CSS_PATH = Path(__file__).parent / "styles" / "victor.tcss"

    TITLE = "Victor"
    SUB_TITLE = "Enterprise-Ready AI Coding Assistant"

    BINDINGS = [
        Binding("ctrl+c", "cancel", "Cancel", show=True),
        Binding("ctrl+d", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear", "Clear", show=True),
        Binding("ctrl+t", "toggle_theme", "Theme", show=True),
        Binding("escape", "focus_input", "Focus Input", show=False),
    ]

    def __init__(
        self,
        agent: AgentOrchestrator,
        settings: Settings,
        cmd_handler: SlashCommandHandler,
        provider: str = "ollama",
        model: str = "unknown",
    ) -> None:
        """Initialize Victor TUI.

        Args:
            agent: The AgentOrchestrator instance.
            settings: Application settings.
            cmd_handler: Slash command handler.
            provider: LLM provider name.
            model: Model name.
        """
        super().__init__()
        self.agent = agent
        self.settings = settings
        self.cmd_handler = cmd_handler
        self._provider = provider
        self._model = model
        self._streaming = False
        self._current_assistant_msg: AssistantMessage | None = None

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()
        yield MessageContainer(id="messages")
        yield ChatInput(id="chat-input")
        yield StatusBar(
            provider=self._provider,
            model=self._model,
            tool_budget=getattr(self.settings, "tool_call_budget", 15),
            id="status-bar",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Handle app mount - show welcome message and set up safety callback."""
        messages = self.query_one("#messages", MessageContainer)
        context_loaded = bool(
            hasattr(self.agent, "project_context")
            and self.agent.project_context
            and self.agent.project_context.content
        )
        welcome = WelcomePanel(
            provider=self._provider,
            model=self._model,
            context_loaded=context_loaded,
        )
        messages.mount(welcome)

        # Set up safety confirmation callback for TUI
        set_confirmation_callback(self._handle_confirmation)

    async def _handle_confirmation(self, request: ConfirmationRequest) -> bool:
        """Handle confirmation request by showing modal dialog.

        Args:
            request: The confirmation request with risk details.

        Returns:
            True if user confirmed, False if cancelled.
        """
        modal = ConfirmationModal(request)
        result = await self.push_screen_wait(modal)
        return result if result is not None else False

    @on(ChatInput.Submitted)
    async def handle_chat_submit(self, event: ChatInput.Submitted) -> None:
        """Handle chat input submission.

        Args:
            event: The chat submission event.
        """
        user_input = event.value.strip()
        if not user_input:
            return

        # Handle exit commands
        if user_input.lower() in ("exit", "quit"):
            self.exit()
            return

        # Handle slash commands
        if self.cmd_handler.is_command(user_input):
            await self._handle_slash_command(user_input)
            return

        # Handle legacy clear command
        if user_input.lower() == "clear":
            self.action_clear()
            return

        # Regular chat message - show user message IMMEDIATELY for responsiveness
        messages = self.query_one("#messages", MessageContainer)
        messages.add_user_message(user_input)
        messages.scroll_to_end_now()
        self.refresh(layout=True)  # Force immediate UI update

        # Then start background streaming (worker handles the rest)
        self._stream_response(user_input)

    async def _handle_slash_command(self, command: str) -> None:
        """Handle a slash command.

        Args:
            command: The slash command to execute.
        """
        messages = self.query_one("#messages", MessageContainer)

        # Add user message showing the command
        messages.add_user_message(command)

        # Capture Rich console output to display in TUI
        string_io = io.StringIO()
        capture_console = Console(file=string_io, force_terminal=True, width=100)

        # Temporarily swap the command handler's console
        original_console = self.cmd_handler.console
        self.cmd_handler.console = capture_console

        try:
            await self.cmd_handler.execute(command)

            # Get captured output
            output = string_io.getvalue()

            # Create assistant message with the output
            assistant_msg = messages.add_assistant_message()
            if output.strip():
                # Convert Rich markup to Markdown-compatible format
                # Remove ANSI codes for cleaner display
                clean_output = self._clean_rich_output(output)
                assistant_msg.set_content(clean_output)
            else:
                assistant_msg.set_content(f"[dim]Command '{command}' executed.[/]")

        except Exception as e:
            assistant_msg = messages.add_assistant_message()
            assistant_msg.set_content(f"[red]Error executing command: {e}[/]")
        finally:
            # Restore original console
            self.cmd_handler.console = original_console
            string_io.close()

        messages.scroll_to_end_now()

    def _clean_rich_output(self, output: str) -> str:
        """Clean Rich console output for display in Markdown widget.

        Args:
            output: Raw Rich console output with ANSI codes.

        Returns:
            Cleaned output suitable for Markdown display.
        """
        import re

        # Remove ANSI escape codes
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        cleaned = ansi_escape.sub("", output)

        # Convert Rich box characters to ASCII for better TUI compatibility
        box_replacements = {
            "─": "-",
            "│": "|",
            "┌": "+",
            "┐": "+",
            "└": "+",
            "┘": "+",
            "├": "+",
            "┤": "+",
            "┬": "+",
            "┴": "+",
            "┼": "+",
            "═": "=",
            "║": "|",
            "╔": "+",
            "╗": "+",
            "╚": "+",
            "╝": "+",
        }
        for rich_char, ascii_char in box_replacements.items():
            cleaned = cleaned.replace(rich_char, ascii_char)

        return cleaned.strip()

    @work(exclusive=True, thread=False)
    async def _stream_response(self, user_message: str) -> None:
        """Stream a response from the agent.

        Args:
            user_message: The user's message.
        """
        messages = self.query_one("#messages", MessageContainer)
        status_bar = self.query_one("#status-bar", StatusBar)
        chat_input = self.query_one("#chat-input", ChatInput)

        # Disable input while streaming
        chat_input.set_disabled(True)
        self._streaming = True

        # User message already added in handle_chat_submit for immediate feedback
        # Add assistant message placeholder
        self._current_assistant_msg = messages.add_assistant_message()
        status_bar.set_thinking()

        content_buffer = ""
        start_time = time.time()
        first_content_time: float | None = None
        tool_indicators: dict[str, ToolIndicator] = {}
        tool_start_times: dict[str, float] = {}
        chunk_count = 0

        try:
            from victor.agent.response_sanitizer import sanitize_response

            async for chunk in self.agent.stream_chat(user_message):
                # Handle status metadata (tool status, thinking indicator)
                if chunk.metadata and "status" in chunk.metadata:
                    status_bar.update_metrics(status=chunk.metadata["status"])
                    continue

                # Handle file preview (consistent with CLI)
                if chunk.metadata and "file_preview" in chunk.metadata:
                    path = chunk.metadata.get("path", "")
                    preview = chunk.metadata["file_preview"]
                    # Add file preview as a separate message
                    preview_msg = messages.add_assistant_message()
                    preview_msg.set_content(f"```\n{path}:\n{preview}\n```")
                    messages.scroll_to_end_now()
                    continue

                # Handle edit preview (consistent with CLI)
                if chunk.metadata and "edit_preview" in chunk.metadata:
                    path = chunk.metadata.get("path", "")
                    preview = chunk.metadata["edit_preview"]
                    # Format diff with markdown
                    diff_lines = []
                    for line in preview.split("\n"):
                        if line.startswith("-"):
                            diff_lines.append(f"[red]{line}[/]")
                        elif line.startswith("+"):
                            diff_lines.append(f"[green]{line}[/]")
                        else:
                            diff_lines.append(f"[dim]{line}[/]")
                    preview_msg = messages.add_assistant_message()
                    preview_msg.set_content(f"[bold]{path}:[/]\n" + "\n".join(diff_lines))
                    messages.scroll_to_end_now()
                    continue

                # Handle content - DO NOT sanitize per-chunk (consistent with CLI)
                # Sanitize only once at the end to preserve whitespace
                if chunk.content:
                    # Track time to first content
                    if first_content_time is None:
                        first_content_time = time.time()
                        status_bar.set_streaming()

                    # Accumulate raw content (like CLI does)
                    content_buffer += chunk.content
                    self._current_assistant_msg.set_content(content_buffer)

                    # Throttled scroll (every 10 chunks or so)
                    chunk_count += 1
                    if chunk_count % 10 == 0:
                        messages.scroll_to_end_throttled()

                # Handle tool calls
                if chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        tool_name = tc.get("name", "tool")
                        tool_id = tc.get("id", tool_name)

                        if tool_id not in tool_indicators:
                            # Track start time for this tool
                            tool_start_times[tool_id] = time.time()

                            # Create indicator
                            indicator = ToolIndicator(tool_name)
                            messages.mount(indicator)
                            tool_indicators[tool_id] = indicator
                            status_bar.set_tool_running(tool_name)
                            messages.scroll_to_end_now()

            # Sanitize the full response once streaming is complete (consistent with CLI)
            # This removes thinking tokens and artifacts without losing spaces
            content_buffer = sanitize_response(content_buffer)
            self._current_assistant_msg.set_content(content_buffer)

            # Mark all tool indicators as complete with individual elapsed times
            current_time = time.time()
            for tool_id, indicator in tool_indicators.items():
                tool_elapsed = current_time - tool_start_times.get(tool_id, start_time)
                indicator.set_success(elapsed=tool_elapsed)

            # Update status bar
            token_estimate = len(content_buffer) // 4  # Rough estimate
            status_bar.update_metrics(
                tokens=token_estimate,
                tool_calls=len(tool_indicators),
            )
            status_bar.set_ready()

            # Final scroll to ensure everything is visible
            messages.scroll_to_end_now()

        except Exception as e:
            logger.exception("Error during streaming")
            if self._current_assistant_msg:
                self._current_assistant_msg.set_content(f"{content_buffer}\n\n[red]Error: {e}[/]")
            status_bar.status = f"Error: {e}"

        finally:
            self._streaming = False
            chat_input.set_disabled(False)
            chat_input.focus_input()
            self._current_assistant_msg = None

    def action_cancel(self) -> None:
        """Handle cancel action (Ctrl+C)."""
        if self._streaming:
            # Request cancellation from the orchestrator
            self.agent.request_cancellation()
            self.notify("Cancelling...", severity="warning")
        else:
            chat_input = self.query_one("#chat-input", ChatInput)
            chat_input.focus_input()

    def action_clear(self) -> None:
        """Clear conversation history."""
        messages = self.query_one("#messages", MessageContainer)
        messages.clear_messages()
        self.agent.reset_conversation()

        # Re-add welcome panel
        context_loaded = bool(
            hasattr(self.agent, "project_context")
            and self.agent.project_context
            and self.agent.project_context.content
        )
        welcome = WelcomePanel(
            provider=self._provider,
            model=self._model,
            context_loaded=context_loaded,
        )
        messages.mount(welcome)

        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.update_metrics(tokens=0, tool_calls=0, status="Ready")

        self.notify("Conversation cleared", severity="information")

    def action_focus_input(self) -> None:
        """Focus the input field."""
        chat_input = self.query_one("#chat-input", ChatInput)
        chat_input.focus_input()

    def action_quit(self) -> None:  # type: ignore[override]
        """Quit the application."""
        self.exit()

    def action_toggle_theme(self) -> None:
        """Toggle between dark and light theme."""
        self.dark = not self.dark  # type: ignore[has-type]
        theme_name = "dark" if self.dark else "light"
        self.notify(f"Switched to {theme_name} theme", severity="information")
