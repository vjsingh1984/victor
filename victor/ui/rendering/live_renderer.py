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
from rich.markup import escape as _markup_escape

from victor.ui.rendering.metrics import StreamingMetrics

logger = logging.getLogger(__name__)

from victor.ui.rendering.utils import (
    expand_tool_output,
    format_duration,
    format_tool_args,
    format_tool_display_name,
    render_content_badge,
    render_edit_preview,
    render_file_preview,
    render_status_message,
    render_thinking_indicator,
    render_thinking_text,
    render_tool_preview,
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

    # Class-level constants for buffer management (Fix 3 & 5)
    _MAX_CONTENT_BUFFER_SIZE = 50_000  # 50K chars — prevents unbounded memory growth
    _THINKING_BUFFER_LIMIT = 10_000  # Discard thinking content beyond this

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
        self._pending_tool: dict | None = None
        self._last_tool_result: dict | None = None
        self._thinking_indicator_shown = False
        self._in_thinking_mode = False
        self._content_shown_before_pause = ""
        self._tool_section_shown = False  # Track if tool section separator shown
        self._current_tool_start_time: float | None = None  # Track tool execution start time
        self._current_tool_category: str | None = None  # Track current tool category for grouping

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
        # Record start time for progress tracking
        self._current_tool_start_time = time.monotonic()

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

        # Show tool section separator before first tool call
        if not self._tool_section_shown:
            self._print_section_separator("Tool Execution")
            self._tool_section_shown = True

        # Check if tool took a long time and show progress (if applicable)
        if self._current_tool_start_time:
            tool_elapsed = time.monotonic() - self._current_tool_start_time
            if tool_elapsed > 3.0:
                self._update_tool_progress(name, tool_elapsed)

        tool_settings = get_tool_settings()
        show_preview = tool_settings.tool_output_preview_enabled

        # Check for tool category changes and show group headers (if enabled)
        if tool_settings.enable_tool_grouping:
            tool_category = self._categorize_tool(name)
            if tool_category != self._current_tool_category:
                # Add spacing before new group (but not before first tool)
                if self._current_tool_category is not None:
                    self.console.print("")  # Blank line between groups
                # Show group header
                self.console.print(f"[dim bold]▸ {tool_category}[/]")
                self._current_tool_category = tool_category

        preview_output = str(result) if result is not None else original_result
        full_output = original_result or (str(result) if result is not None else None)

        self.pause()

        args_display = format_tool_args(arguments)
        icon = "✓" if success else "✗"
        color = "green" if success else "red"
        status_line = f"[{color}]{icon}[/] [bold]{format_tool_display_name(name)}[/]"
        if args_display:
            status_line += f" [dim]{args_display}[/]"
        status_line += f" [dim]• {format_duration(elapsed)}[/]"
        if error:
            status_line += f" [red]{error[:80]}[/]"
        self.console.print(status_line)

        # Show preview if enabled
        if show_preview and success and preview_output:
            # Calculate adaptive preview lines if enabled
            if tool_settings.tool_output_preview_adaptive:
                adaptive_lines = self._calculate_adaptive_preview_lines(
                    preview_output, error, preview_lines, tool_settings
                )
            else:
                adaptive_lines = preview_lines

            from victor.ui.rendering.tool_preview import renderer as _tool_preview_renderer

            preview = _tool_preview_renderer.render(
                name, arguments, preview_output, max_lines=adaptive_lines
            )
            if preview.header or preview.lines:
                if preview.header:
                    self.console.print(f"[dim]│ {_markup_escape(preview.header)}[/]")
                if preview.lines:
                    # Check if preview contains Rich markup (e.g., formatted diffs)
                    if preview.contains_rich_markup:
                        # Don't escape markup - the lines already contain Rich formatting
                        preview_text = "\n".join(preview.lines)
                    else:
                        # Escape markup to prevent rendering issues
                        preview_text = "\n".join(
                            _markup_escape(line_text) for line_text in preview.lines
                        )

                    render_tool_preview(
                        self.console,
                        preview_text,
                        total_lines=preview.total_line_count,
                        preview_lines=adaptive_lines,
                        hotkey=tool_settings.tool_output_expand_hotkey,
                        contains_rich_markup=preview.contains_rich_markup,
                    )

        # Show pruning transparency
        if was_pruned and tool_settings.tool_output_show_transparency:
            self.console.print("[dim yellow]! Preview truncated (full output sent to model)[/]")

        # Store result for potential expansion
        self._last_tool_result = {
            "name": name,
            "success": success,
            "result": full_output or "",
            "arguments": arguments,
            "elapsed": elapsed,
        }

        # Show follow-up suggestions
        if follow_up_suggestions:
            for suggestion in follow_up_suggestions[:2]:
                if not isinstance(suggestion, dict):
                    continue
                command = suggestion.get("command")
                if not isinstance(command, str) or not command.strip():
                    continue
                self.console.print(f"[dim]  next: {command}[/]")

        self._pending_tool = None
        self._current_tool_start_time = None  # Reset tool start time
        self._metrics.record_tool_result()
        self.resume()

    def on_status(self, message: str) -> None:
        """Handle status message.

        Args:
            message: Status message to display
        """
        self.pause()
        render_status_message(self.console, message)
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

        All content goes into _content_buffer regardless of mode (single source
        of truth).  During thinking mode the text is also printed directly to
        console and _content_shown_before_pause is advanced so that resume()
        does not re-display thinking content in the Live panel.

        Args:
            text: Content text to append
        """
        # Always buffer — cap to prevent unbounded memory growth
        if len(self._content_buffer) + len(text) > self._MAX_CONTENT_BUFFER_SIZE:
            excess = len(self._content_buffer) + len(text) - self._MAX_CONTENT_BUFFER_SIZE
            self._content_buffer = self._content_buffer[excess:]
        self._content_buffer += text

        if self._in_thinking_mode:
            # Advance the "shown" pointer so resume() starts from here, not
            # from before thinking started — prevents thinking text appearing
            # a second time in the Live panel when normal content resumes.
            self._content_shown_before_pause = self._content_buffer
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
        render_thinking_text(self.console, text)

    def _handle_normal_content(self, text: str) -> None:
        """Handle content during normal mode - stream to Live display.

        Key design decisions:
        - Show only the post-pause slice of the buffer. Pre-pause content
          was already committed to the terminal by the previous Live session
          stop; re-rendering it makes old iterations' text reappear after
          each tool execution.
        - Track metrics for performance monitoring

        Args:
            text: Content text to append
        """
        t0 = time.monotonic() * 1000
        if self._live:
            visible = self._content_buffer[len(self._content_shown_before_pause) :]
            self._live.update(Markdown(visible))
        duration_ms = time.monotonic() * 1000 - t0
        self._metrics.record_content_chunk(duration_ms)
        if duration_ms > 100:
            logger.debug("LiveDisplayRenderer: slow render %.1fms", duration_ms)

    def on_thinking_content(self, text: str) -> None:
        """Display thinking content immediately during streaming.

        Design principle:
        - Thinking content from API (DeepSeek, Z.AI) is different from inline markers
        - API reasoning: print immediately, don't buffer
        - Inline markers: handled by StreamingContentFilter
        - Single responsibility: just display, don't accumulate

        Note: Delta normalization is handled in stream_response() handler,
        so this method receives only the new portion to display.

        Args:
            text: Thinking text to display (already normalized/delta-extracted)
        """
        if not text or not text.strip():
            return

        # Pause live display to show thinking content
        self.pause()
        render_thinking_text(self.console, text)
        # Resume live display
        self.resume()

    def on_thinking_start(self) -> None:
        """Show thinking indicator and pause Live display."""
        self.pause()
        # Only show indicator once per response (reset in cleanup)
        if not self._thinking_indicator_shown:
            # Add section separator for visual hierarchy
            self._print_section_separator("Thinking")
            # Show content type badge
            render_content_badge(self.console, "thinking")
            render_thinking_indicator(self.console)
            self._thinking_indicator_shown = True
        # Mark that we're in thinking mode - content will be separate
        self._in_thinking_mode = True

    def on_thinking_end(self) -> None:
        """Exit thinking state and resume Live display."""
        self._in_thinking_mode = False
        self.resume()

    def _print_section_separator(self, title: str = "") -> None:
        """Print a subtle section separator for visual hierarchy.

        Args:
            title: Optional title to display in the separator
        """
        if title:
            # Styled separator with title
            self.console.print(f"[dim]{'─' * 20} {title} {'─' * 20}[/]")
        else:
            # Simple separator line
            self.console.print("[dim]" + "─" * 60 + "[/]")

    def _update_tool_progress(self, tool_name: str, elapsed: float) -> None:
        """Show progress indicator for long-running tools.

        Args:
            tool_name: Name of the tool being executed
            elapsed: Time elapsed since tool started (in seconds)
        """
        if elapsed > 3.0 and int(elapsed) % 2 == 0:  # Every 2 seconds after 3s
            dots = "." * (int(elapsed) % 3 + 1)
            display_name = format_tool_display_name(tool_name)
            self.console.print(f"[dim]  {display_name} still running{dots} ({elapsed:.1f}s)[/]")
            self.resume()  # Resume after printing progress update

    def had_tool_calls(self) -> bool:
        """Return True if at least one tool call was processed this turn."""
        return self._tool_section_shown

    def finalize(self) -> str:
        """Finalize response and return accumulated content.

        Returns:
            Accumulated response content
        """
        from rich.markdown import Markdown
        import time

        # REMOVED: Flush thinking buffer (no longer exists - caused duplication)

        # FAIL-SAFE: Ensure content buffer is displayed to user
        # Only print the portion not already shown directly (e.g., via thinking-mode print)
        unshown = self._content_buffer[len(self._content_shown_before_pause) :]
        if unshown and not self._live:
            self.console.print(Markdown(unshown))
        elif self._live and self._content_buffer:
            # Live display exists - ensure final update
            final_content = self._content_buffer[len(self._content_shown_before_pause) :]
            if final_content.strip():
                self._live.update(Markdown(final_content))
                # Small delay to ensure user sees final content
                time.sleep(0.1)

        # Add section separator before final response if we have content
        if self._content_buffer.strip() and (
            self._thinking_indicator_shown or self._tool_section_shown
        ):
            self._print_section_separator("Response")
            # Show content type badge
            render_content_badge(self.console, "response")

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

    def _calculate_adaptive_preview_lines(
        self,
        tool_output: str,
        error: str | None,
        default_lines: int,
        tool_settings: Any,
    ) -> int:
        """Calculate adaptive preview lines based on content characteristics.

        Args:
            tool_output: The tool output text
            error: Error message if any
            default_lines: Default preview lines from caller
            tool_settings: ToolSettings instance with adaptive config

        Returns:
            Number of lines to show in preview
        """
        # If there's an error, show all output (no preview limit)
        if error:
            return len(tool_output.split("\n"))

        total_lines = len(tool_output.split("\n"))
        min_lines = tool_settings.tool_output_preview_lines_min
        max_lines = tool_settings.tool_output_preview_lines_max

        # Small outputs: show everything
        if total_lines <= 5:
            return total_lines

        # Medium outputs (5-50 lines): show moderate preview
        if 5 < total_lines <= 50:
            # Use 3-5 lines for medium outputs, bounded by min/max
            adaptive = min(5, max_lines)
            return max(min_lines, adaptive)

        # Large outputs (>50 lines): show minimal preview
        # Use 1-2 lines for large outputs, bounded by min/max
        adaptive = min(2, max_lines)
        return max(min_lines, adaptive)

    def _categorize_tool(self, tool_name: str) -> str:
        """Categorize a tool name into a logical group.

        Args:
            tool_name: Name of the tool to categorize

        Returns:
            Category name for the tool
        """
        # Define tool categories based on name patterns
        categories = {
            "File System": ["read", "write", "ls", "grep", "file_info"],
            "Search": ["code_search", "semantic_code_search", "search"],
            "Git": ["git_status", "git_diff", "git_log", "git_blame"],
            "Analysis": ["overview", "analyze", "inspect"],
            "Build": ["build", "compile", "test"],
            "Execution": ["bash", "shell", "run"],
            "Web": ["web_search", "fetch", "http"],
            "Database": ["db_query", "db_execute", "sql"],
        }

        # Find matching category
        tool_lower = tool_name.lower()
        for category, patterns in categories.items():
            if any(pattern in tool_lower for pattern in patterns):
                return category

        # Default: use "Other" category
        return "Other"

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
        self._pending_tool = None
        self._thinking_indicator_shown = False
        self._in_thinking_mode = False
        self._content_shown_before_pause = ""
        self._last_tool_result = None
        self._tool_section_shown = False  # Reset tool section flag
        self._current_tool_category = None  # Reset tool category tracker
