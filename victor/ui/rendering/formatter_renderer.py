"""FormatterRenderer - OutputFormatter-based stream renderer.

This renderer delegates to OutputFormatter for oneshot mode rendering,
suitable for non-interactive CLI usage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console

from victor.ui.rendering.utils import (
    render_edit_preview,
    render_file_preview,
    render_thinking_indicator,
    render_thinking_text,
)

if TYPE_CHECKING:
    from victor.ui.output_formatter import OutputFormatter


class FormatterRenderer:
    """Renderer using OutputFormatter for oneshot mode.

    This renderer delegates to OutputFormatter methods which handle
    the visual presentation with proper formatting.

    Uses shared utility functions for common rendering operations
    (file preview, edit preview, thinking display).

    Attributes:
        formatter: OutputFormatter instance for formatting output
        console: Rich Console for direct output
    """

    def __init__(self, formatter: OutputFormatter, console: Console):
        """Initialize FormatterRenderer.

        Args:
            formatter: OutputFormatter to delegate to
            console: Rich Console for direct output
        """
        self.formatter = formatter
        self.console = console
        self._content_buffer = ""

    def start(self) -> None:
        """Start streaming mode in the formatter."""
        self.formatter.start_streaming()

    def pause(self) -> None:
        """End streaming to allow status output."""
        self.formatter.end_streaming()

    def resume(self) -> None:
        """Resume streaming mode in the formatter."""
        self.formatter.start_streaming()

    def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        """Handle tool execution start.

        Args:
            name: Tool name
            arguments: Tool arguments
        """
        self.pause()
        self.formatter.tool_start(name, arguments)
        self.formatter.status(f"🔧 Running {name}...")
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
        self.formatter.tool_result(
            tool_name=name,
            success=success,
            error=error,
        )
        self.resume()

    def on_status(self, message: str) -> None:
        """Handle status message.

        Args:
            message: Status message to display
        """
        self.pause()
        self.formatter.status(message)
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
        """Handle content chunk.

        Args:
            text: Content text to append
        """
        self._content_buffer += text
        self.formatter.stream_chunk(text)

    def on_thinking_content(self, text: str) -> None:
        """Render thinking content as dimmed/italic text.

        Args:
            text: Thinking text to display
        """
        # Don't add to content buffer (thinking is ephemeral)
        render_thinking_text(self.console, text)

    def on_thinking_start(self) -> None:
        """Show thinking indicator."""
        self.pause()
        render_thinking_indicator(self.console)

    def on_thinking_end(self) -> None:
        """Show end of thinking."""
        self.console.print()  # Newline after thinking
        self.resume()

    def finalize(self) -> str:
        """Finalize response and return accumulated content.

        Returns:
            Accumulated response content
        """
        self.formatter.response(content=self._content_buffer)
        return self._content_buffer

    def cleanup(self) -> None:
        """Clean up resources (no-op for FormatterRenderer)."""
        pass
