"""LiveDisplayRenderer - Rich Live display-based stream renderer.

This renderer uses Rich's Live display for real-time markdown rendering,
suitable for interactive CLI usage.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from victor.ui.rendering.metrics import StreamingMetrics

logger = logging.getLogger(__name__)

from victor.ui.rendering.utils import (
    expand_tool_output,
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
        self._content_buffer = ""  # Single source of truth for all content
        self._is_paused = False
        self._pause_count = 0  # Depth counter for nested pause/resume
        self._pause_start_ms: float | None = None
        self._metrics = StreamingMetrics()
        # REMOVED: self._thinking_buffer = ""  # No longer needed - caused duplication
        self._pending_tool: dict | None = None
        self._last_tool_result: dict | None = None
        # REMOVED: self._last_thinking_rendered = ""  # No longer needed
        self._thinking_indicator_shown = False
        self._in_thinking_mode = False
        self._content_shown_before_pause = ""

    def start(self) -> None:
        """Start the Live display."""
        self._live = Live(Markdown(""), console=self.console, refresh_per_second=10)
        self._live.start()
        self._is_paused = False
        self._pause_count = 0

    def get_metrics(self) -> StreamingMetrics:
        """Return accumulated streaming metrics for this session."""
        return self._metrics

    def pause(self) -> None:
        """Pause the Live display to show status output.

        Supports nested calls via depth counting — the Live display is only
        stopped on the first call, preventing unnecessary stop/restart cycles.
        """
        self._pause_count += 1
        if self._pause_count == 1 and self._live and not self._is_paused:
            self._pause_start_ms = time.monotonic() * 1000
            self._live.stop()
            self._is_paused = True
            self._content_shown_before_pause = self._content_buffer
            logger.debug("LiveDisplayRenderer: paused (depth=1)")

    def resume(self) -> None:
        """Resume the Live display with only NEW content since the outermost pause.

        Only restarts the Live display when the depth counter reaches zero,
        preventing duplication from nested pause/resume callers (e.g., thinking
        blocks that contain tool calls).
        """
        if self._pause_count <= 0:
            logger.warning("LiveDisplayRenderer: resume() called with no matching pause — ignoring")
            return
        self._pause_count -= 1
        if self._pause_count == 0 and self._is_paused:
            if self._pause_start_ms is not None:
                duration_ms = time.monotonic() * 1000 - self._pause_start_ms
                self._metrics.record_pause(duration_ms)
                self._pause_start_ms = None
            self._metrics.record_resume()
            new_content = self._content_buffer[len(self._content_shown_before_pause) :]
            self._live = Live(
                Markdown(new_content),
                console=self.console,
                refresh_per_second=10,
            )
            self._live.start()
            self._is_paused = False
            logger.debug("LiveDisplayRenderer: resumed (depth=0)")

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
        follow_up_suggestions: list[dict[str, Any]] | None = None,
        # New parameters for preview
        original_result: str | None = None,
        preview_lines: int = 3,
        was_pruned: bool = False,
        result: Any = None,  # Alias for original_result (for compatibility)
    ) -> None:
        """Handle tool execution result - print consolidated single line with preview.

        Args:
            name: Tool name
            success: Whether tool succeeded
            elapsed: Execution time in seconds
            arguments: Tool arguments
            error: Error message if failed
            original_result: Unmodified tool output (before pruning)
            preview_lines: Number of lines to show in preview
            was_pruned: Whether output was pruned before sending to LLM
            result: Tool output (alias for original_result)
        """
        from victor.config.tool_settings import get_tool_settings

        tool_settings = get_tool_settings()
        show_preview = tool_settings.tool_output_preview_enabled

        # Use result parameter if original_result not provided
        tool_output = original_result or (str(result) if result is not None else None)

        self.pause()

        # Print status line
        args_display = format_tool_args(arguments)
        args_str = f"({args_display})" if args_display else ""
        icon = "✓" if success else "✗"
        color = "green" if success else "red"
        # Single consolidated line: icon + name + args + time
        self.console.print(f"[{color}]{icon}[/] {name}{args_str} [dim]({elapsed:.1f}s)[/]")

        # Show preview if enabled
        if show_preview and success and tool_output:
            preview_text = self._generate_preview(tool_output, preview_lines)
            if preview_text:
                self.console.print(f"[dim]↳ {preview_text}[/]")

                # Show expand hint if output is longer than preview
                num_lines = len(tool_output.split("\n"))
                if num_lines > preview_lines:
                    hotkey = tool_settings.tool_output_expand_hotkey
                    self.console.print(f"[dim italic]Press {hotkey} to see all {num_lines} lines[/]")

        # Show pruning transparency
        if was_pruned and tool_settings.tool_output_show_transparency:
            self.console.print("[dim yellow]⚠ Output was pruned before sending to LLM[/]")

        # Store result for potential expansion
        self._last_tool_result = {
            "name": name,
            "success": success,
            "result": tool_output or "",
            "arguments": arguments,
            "elapsed": elapsed,
        }

        # Show follow-up suggestions
        if success and follow_up_suggestions:
            for suggestion in follow_up_suggestions[:2]:
                if not isinstance(suggestion, dict):
                    continue
                command = suggestion.get("command")
                if not isinstance(command, str) or not command.strip():
                    continue
                self.console.print(f"[dim]  next: {command}[/]")

        self._pending_tool = None
        self._metrics.record_tool_result()
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
        """Handle content chunk - route to appropriate handler based on state.

        Args:
            text: Content text to append
        """
        # CRITICAL: Always preserve content for final display
        self._content_buffer += text

        # State-based routing (State Pattern)
        if self._in_thinking_mode:
            self._handle_thinking_content(text)
        else:
            self._handle_normal_content(text)

    def _handle_thinking_content(self, text: str) -> None:
        """Handle content during thinking mode - print immediately.

        Key design decisions:
        - Print directly to console (NOT Live display which is paused)
        - Don't buffer to _thinking_buffer (eliminates duplication)
        - Don't call _live.update() (wasted cycles, display is paused)

        Args:
            text: Content text to display
        """
        from rich.text import Text

        # Print thinking content immediately with distinct styling
        thinking_text = Text(text, style="dim italic")
        self.console.print(thinking_text)

    def _handle_normal_content(self, text: str) -> None:
        """Handle content during normal mode - stream to Live display.

        Key design decisions:
        - Update Live display with full buffer (Rich handles delta internally)
        - Track metrics for performance monitoring
        - No delta calculation needed (Rich is efficient)

        Args:
            text: Content text to append
        """
        t0 = time.monotonic() * 1000
        if self._live:
            # Rich Live display is smart - it only re-renders changed portions
            self._live.update(Markdown(self._content_buffer))
        duration_ms = time.monotonic() * 1000 - t0
        self._metrics.record_content_chunk(duration_ms)
        if duration_ms > 100:
            logger.debug("LiveDisplayRenderer: slow render %.1fms", duration_ms)

    def on_thinking_content(self, text: str) -> None:
        """Display thinking content immediately during streaming.

        Design principle:
        - Thinking content from API (DeepSeek) is different from inline markers
        - API reasoning: print immediately, don't buffer
        - Inline markers: handled by StreamingContentFilter
        - Single responsibility: just display, don't accumulate

        Args:
            text: Thinking text to display
        """
        if not text or not text.strip():
            return

        # Pause live display to show thinking content
        self.pause()

        # Render thinking content immediately (no buffering)
        from rich.text import Text
        thinking_text = Text(text, style="dim italic")
        self.console.print(thinking_text)

        # Resume live display
        self.resume()

        # REMOVED: self._thinking_buffer += text  # Eliminates duplication
        # REMOVED: self._last_thinking_rendered = text  # No longer needed

    def on_thinking_start(self) -> None:
        """Show thinking indicator and pause Live display."""
        self.pause()
        # REMOVED: self._thinking_buffer = ""  # No longer needed
        # Only show indicator once per response (reset in cleanup)
        if not self._thinking_indicator_shown:
            render_thinking_indicator(self.console)
            self._thinking_indicator_shown = True
        # Mark that we're in thinking mode - content will be separate
        self._in_thinking_mode = True

    def on_thinking_end(self) -> None:
        """Exit thinking state and resume Live display.

        Simplified approach:
        - Just exit thinking mode and resume
        - No manual rendering (resume() handles delta correctly)
        - No buffer duplication (single source of truth)
        """
        # Exit thinking mode first
        self._in_thinking_mode = False

        # Resume Live display (shows all content buffered during thinking)
        self.resume()

        # Note: _thinking_buffer is now unused - removed in __init__

    def finalize(self) -> str:
        """Finalize response and return accumulated content.

        Returns:
            Accumulated response content
        """
        from rich.markdown import Markdown
        import time

        # REMOVED: Flush thinking buffer (no longer exists - caused duplication)

        # FAIL-SAFE: Ensure content buffer is displayed to user
        # This catches any edge cases where content wasn't shown during streaming
        if self._content_buffer and not self._live:
            # Live display was never started, render content directly
            self.console.print(Markdown(self._content_buffer))
        elif self._live and self._content_buffer:
            # Live display exists - ensure final update
            final_content = self._content_buffer[len(self._content_shown_before_pause) :]
            if final_content.strip():
                self._live.update(Markdown(final_content))
                # Small delay to ensure user sees final content
                time.sleep(0.1)

        # Log for debugging
        logger.debug(
            f"LiveDisplayRenderer.finalize(): "
            f"content_buffer_len={len(self._content_buffer)}, "
            f"in_thinking_mode={self._in_thinking_mode}"
        )

        self.cleanup()
        return self._content_buffer

    def _generate_preview(self, text: str, num_lines: int = 3) -> str:
        """Generate a preview of the first few lines of output.

        Args:
            text: Full text to generate preview from
            num_lines: Number of lines to include in preview

        Returns:
            Preview text with ellipsis if truncated
        """
        if not text:
            return ""
        lines = text.split("\n")
        preview = "\n".join(lines[:num_lines])
        if len(lines) > num_lines:
            preview += "..."
        # Truncate long lines
        max_line_length = 120
        preview = "\n".join(
            line[:max_line_length] + "..." if len(line) > max_line_length else line
            for line in preview.split("\n")
        )
        return preview

    def expand_last_output(self) -> None:
        """Expand the last tool output to show full content."""
        if not self._last_tool_result:
            self.console.print("[dim]No tool output to expand[/]")
            return

        data = self._last_tool_result
        if not data["success"] or not data["result"]:
            return

        expand_tool_output(
            self.console,
            data["name"],
            data["result"],
            pause_fn=self.pause,
            resume_fn=lambda: self.resume(),
        )

    def cleanup(self) -> None:
        """Clean up the Live display."""
        if self._live:
            self._live.stop()
            self._live = None
        self._is_paused = False
        self._pause_count = 0
        # REMOVED: self._thinking_buffer = ""  # No longer needed
        # REMOVED: self._last_thinking_rendered = ""  # No longer needed
        self._pending_tool = None
        self._thinking_indicator_shown = False
        self._in_thinking_mode = False
        self._content_shown_before_pause = ""
        self._last_tool_result = None
