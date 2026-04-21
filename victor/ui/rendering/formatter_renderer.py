"""FormatterRenderer - OutputFormatter-based stream renderer.

This renderer delegates to OutputFormatter for oneshot mode rendering,
suitable for non-interactive CLI usage.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from rich.console import Console

logger = logging.getLogger(__name__)

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
        self._pause_count = 0  # Depth counter for nested pause/resume
        self._last_tool_result: dict | None = None

    def start(self) -> None:
        """Start streaming mode in the formatter."""
        self.formatter.start_streaming()

    def pause(self) -> None:
        """End streaming to allow status output.

        Supports nested calls via depth counting — the underlying formatter
        is only paused on the first call, preventing redundant end/start cycles.
        """
        self._pause_count += 1
        if self._pause_count == 1:
            self.formatter.end_streaming(finalize=False)
            logger.debug("FormatterRenderer: paused (depth=1)")

    def resume(self) -> None:
        """Resume streaming mode in the formatter.

        Only resumes the underlying formatter when the depth counter reaches zero,
        allowing nested pause/resume callers to unwind correctly.
        """
        if self._pause_count <= 0:
            logger.warning("FormatterRenderer: resume() called with no matching pause — ignoring")
            return
        self._pause_count -= 1
        if self._pause_count == 0:
            self.formatter.start_streaming(preserve_buffer=True)
            logger.debug("FormatterRenderer: resumed (depth=0)")

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
        follow_up_suggestions: list[dict[str, Any]] | None = None,
        result: Any = None,  # Tool output for preview
    ) -> None:
        """Handle tool execution result.

        Args:
            name: Tool name
            success: Whether tool succeeded
            elapsed: Execution time in seconds
            arguments: Tool arguments
            error: Error message if failed
            follow_up_suggestions: Optional follow-up suggestions
            result: Tool output (for preview display)
        """
        tool_output = str(result) if result is not None else None
        self._last_tool_result = {
            "name": name,
            "success": success,
            "result": tool_output or "",
            "arguments": arguments,
            "elapsed": elapsed,
        }
        self.pause()
        self.formatter.tool_result(
            tool_name=name,
            success=success,
            error=error,
            follow_up_suggestions=follow_up_suggestions,
            original_result=tool_output,
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

    def expand_last_output(self) -> None:
        """Expand the last tool output to show full content."""
        if not self._last_tool_result:
            self.console.print("[dim]No tool output to expand[/]")
            return

        data = self._last_tool_result
        if not data["success"] or not data["result"]:
            return

        from rich.panel import Panel
        from rich.syntax import Syntax

        content = data["result"]
        tool_name = data["name"]

        if len(content) > 10000:
            self.console.print(
                f"[dim yellow]⚠ Output is {len(content)} chars, showing first 10000[/]"
            )
            content = content[:10000]

        try:
            ext = tool_name.split("_")[-1] if "_" in tool_name else "txt"
            syntax = Syntax(content, ext, theme="monokai", line_numbers=True, word_wrap=True)
            self.console.print(
                Panel(syntax, title=f"[bold]{tool_name}[/] - Full Output", border_style="blue")
            )
        except Exception:
            self.console.print(
                Panel(content, title=f"[bold]{tool_name}[/] - Full Output", border_style="blue")
            )

    def cleanup(self) -> None:
        """Clean up resources."""
        self._last_tool_result = None
