"""LiveDisplayRenderer - Rich Live display-based stream renderer.

This renderer uses Rich's Live display for real-time markdown rendering,
suitable for interactive CLI usage.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from victor.ui.rendering.utils import (
    format_tool_args,
    render_edit_preview,
    render_file_preview,
    render_thinking_indicator,
    render_thinking_text,
)


class LiveDisplayRenderer:
    """Renderer using Rich Live display for interactive mode.

    This renderer manages a Live display that updates in real-time
    with markdown rendering.

    Uses shared utility functions for common rendering operations
    (file preview, edit preview, thinking display, arg formatting).

    Attributes:
        console: Rich Console for output
    """

    def __init__(self, console: Console):
        """Initialize LiveDisplayRenderer.

        Args:
            console: Rich Console for output
        """
        self.console = console
        self._live: Live | None = None
        self._content_buffer = ""
        self._is_paused = False  # Track pause state to avoid redundant operations
        self._thinking_buffer = ""  # Accumulate thinking text for clean display
        self._pending_tool: dict | None = None  # Track tool waiting for result
        self._last_thinking_rendered = ""  # Track last rendered thinking to avoid dupes
        self._thinking_indicator_shown = False  # Track if we've shown the indicator

    def start(self) -> None:
        """Start the Live display."""
        self._live = Live(Markdown(""), console=self.console, refresh_per_second=10)
        self._live.start()
        self._is_paused = False

    def pause(self) -> None:
        """Pause the Live display to show status output."""
        if self._live and not self._is_paused:
            self._live.stop()
            self._is_paused = True

    def resume(self) -> None:
        """Resume the Live display with current content."""
        if self._is_paused:
            self._live = Live(
                Markdown(self._content_buffer),
                console=self.console,
                refresh_per_second=10,
            )
            self._live.start()
            self._is_paused = False

    def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        """Handle tool execution start - store for later consolidation.

        Args:
            name: Tool name
            arguments: Tool arguments
        """
        # Store pending tool info - will print consolidated output on result
        self._pending_tool = {"name": name, "arguments": arguments}

    def on_tool_result(
        self,
        name: str,
        success: bool,
        elapsed: float,
        arguments: dict[str, Any],
        error: str | None = None,
    ) -> None:
        """Handle tool execution result - print consolidated single line.

        Args:
            name: Tool name
            success: Whether tool succeeded
            elapsed: Execution time in seconds
            arguments: Tool arguments
            error: Error message if failed
        """
        self.pause()
        args_display = format_tool_args(arguments)
        args_str = f"({args_display})" if args_display else ""
        icon = "✓" if success else "✗"
        color = "green" if success else "red"
        # Single consolidated line: icon + name + args + time
        self.console.print(f"[{color}]{icon}[/] {name}{args_str} [dim]({elapsed:.1f}s)[/]")
        self._pending_tool = None
        self.resume()

    def on_status(self, message: str) -> None:
        """Handle status message.

        Args:
            message: Status message to display
        """
        self.pause()
        self.console.print(f"[dim]{message}[/]")
        self.resume()

    def on_file_preview(self, path: str, content: str) -> None:
        """Handle file content preview.

        Args:
            path: File path
            content: File content to preview
        """
        self.pause()
        render_file_preview(self.console, path, content)
        self.resume()

    def on_edit_preview(self, path: str, diff: str) -> None:
        """Handle edit diff preview.

        Args:
            path: File path being edited
            diff: Diff content to display
        """
        self.pause()
        render_edit_preview(self.console, path, diff)
        self.resume()

    def on_content(self, text: str) -> None:
        """Handle content chunk - update Live display.

        Args:
            text: Content text to append
        """
        self._content_buffer += text
        if self._live:
            self._live.update(Markdown(self._content_buffer))

    def on_thinking_content(self, text: str) -> None:
        """Accumulate thinking content - will render on thinking_end.

        Args:
            text: Thinking text to accumulate
        """
        # Just accumulate - don't print each chunk to avoid duplication
        self._thinking_buffer += text

    def on_thinking_start(self) -> None:
        """Show thinking indicator and pause Live display."""
        self.pause()
        self._thinking_buffer = ""  # Reset thinking buffer
        # Only show indicator once per thinking session
        if not self._thinking_indicator_shown:
            render_thinking_indicator(self.console)
            self._thinking_indicator_shown = True

    def on_thinking_end(self) -> None:
        """End thinking - render accumulated content and resume Live display."""
        # Render only the NEW content (not already rendered) to avoid duplication
        if self._thinking_buffer:
            # Only render the portion we haven't already shown
            if self._thinking_buffer.startswith(self._last_thinking_rendered):
                new_content = self._thinking_buffer[len(self._last_thinking_rendered) :]
            else:
                new_content = self._thinking_buffer

            if new_content.strip():
                render_thinking_text(self.console, new_content)
                self.console.print()  # Newline after thinking content

            self._last_thinking_rendered = self._thinking_buffer
            self._thinking_buffer = ""
        self.resume()

    def finalize(self) -> str:
        """Finalize response and return accumulated content.

        Returns:
            Accumulated response content
        """
        # Render any remaining thinking content
        if self._thinking_buffer:
            render_thinking_text(self.console, self._thinking_buffer)
            self._thinking_buffer = ""
            self.console.print()
        self.cleanup()
        return self._content_buffer

    def cleanup(self) -> None:
        """Clean up the Live display."""
        if self._live:
            self._live.stop()
            self._live = None
        self._is_paused = False
        self._thinking_buffer = ""
        self._last_thinking_rendered = ""
        self._pending_tool = None
        self._thinking_indicator_shown = False
