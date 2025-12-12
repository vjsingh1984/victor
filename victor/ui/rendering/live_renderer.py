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

    def start(self) -> None:
        """Start the Live display."""
        self._live = Live(Markdown(""), console=self.console, refresh_per_second=10)
        self._live.start()

    def pause(self) -> None:
        """Pause the Live display to show status output."""
        if self._live:
            self._live.stop()

    def resume(self) -> None:
        """Resume the Live display with current content."""
        self._live = Live(
            Markdown(self._content_buffer),
            console=self.console,
            refresh_per_second=10,
        )
        self._live.start()

    def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        """Handle tool execution start.

        Args:
            name: Tool name
            arguments: Tool arguments
        """
        self.pause()
        args_display = format_tool_args(arguments)
        args_str = f"({args_display})" if args_display else ""
        self.console.print(f"[dim]🔧 {name}{args_str}...[/]")
        self.resume()

    def on_tool_result(
        self,
        name: str,
        success: bool,
        elapsed: float,
        arguments: dict[str, Any],
        error: str | None = None,
    ) -> None:
        """Handle tool execution result.

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
        self.console.print(f"[{color}]{icon}[/] {name}{args_str} [dim]({elapsed:.1f}s)[/]")
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
        """Render thinking content as dimmed/italic text.

        Args:
            text: Thinking text to display
        """
        self.pause()
        render_thinking_text(self.console, text)

    def on_thinking_start(self) -> None:
        """Show thinking indicator and pause Live display."""
        self.pause()
        render_thinking_indicator(self.console)

    def on_thinking_end(self) -> None:
        """End thinking and resume Live display."""
        self.console.print()  # Newline after thinking content
        self.resume()

    def finalize(self) -> str:
        """Finalize response and return accumulated content.

        Returns:
            Accumulated response content
        """
        self.cleanup()
        return self._content_buffer

    def cleanup(self) -> None:
        """Clean up the Live display."""
        if self._live:
            self._live.stop()
            self._live = None
