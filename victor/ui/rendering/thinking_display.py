"""Enhanced thinking display manager for interactive streaming.

Handles thinking/reasoning content with visual hierarchy, progressive
disclosure, and structured thinking blocks. Extracted from LiveDisplayRenderer.

Features:
- **Visual hierarchy**: Section headers, thinking badges, dimmed content
- **Progressive disclosure**: Expandable thinking blocks for long reasoning
- **State machine**: Clean start/content/end transitions
- **Performance**: Buffered rendering for fast thinking streams
"""

from __future__ import annotations

import logging
import time

from rich.console import Group
from rich.markup import escape as _markup_escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from victor.ui.rendering.live_manager import LiveManager
from victor.ui.rendering.utils import (
    render_content_badge,
    render_thinking_indicator,
    render_thinking_text,
)

logger = logging.getLogger(__name__)


class ThinkingDisplayManager:
    """Manages thinking/reasoning content display with visual hierarchy.

    Responsibilities:
    - Thinking state machine (start/content/end)
    - Content routing during thinking vs normal mode
    - Thinking indicator display with duration tracking
    - Section separators for thinking blocks
    - Progressive disclosure for long reasoning chains

    All display output goes through the LiveManager's pause/resume cycle.
    """

    _THINKING_BUFFER_LIMIT = 10_000
    _LONG_THINKING_THRESHOLD_CHARS = 500  # Show collapsed if longer

    def __init__(self, live_manager: LiveManager):
        """Initialize ThinkingDisplayManager.

        Args:
            live_manager: The LiveManager for pause/resume/lifecycle
        """
        self._live_manager = live_manager
        self._thinking_indicator_shown = False
        self._in_thinking_mode = False
        self._thinking_start_time: float | None = None
        self._thinking_char_count = 0
        self._thinking_lines: list[str] = []

    @property
    def in_thinking_mode(self) -> bool:
        """Whether we are currently in thinking mode."""
        return self._in_thinking_mode

    @property
    def thinking_duration(self) -> float:
        """Duration of current thinking session in seconds."""
        if self._thinking_start_time is None:
            return 0.0
        return time.monotonic() - self._thinking_start_time

    @property
    def thinking_char_count(self) -> int:
        """Total characters of thinking content received."""
        return self._thinking_char_count

    def on_thinking_start(self) -> None:
        """Show thinking indicator and pause Live display.

        Displays a structured thinking header with icon and starts
        tracking duration for the thinking session.
        """
        self._live_manager.pause()

        if not self._thinking_indicator_shown:
            self._live_manager.print_section_separator("Thinking")
            render_content_badge(self._live_manager.console, "thinking")
            render_thinking_indicator(self._live_manager.console)
            self._thinking_indicator_shown = True
            self._thinking_start_time = time.monotonic()

        self._in_thinking_mode = True

    def on_thinking_content(self, text: str) -> None:
        """Display thinking content with progressive rendering.

        For short thinking content, renders inline. For longer chains,
        uses a buffered approach to avoid overwhelming the display.

        Args:
            text: Thinking text to display (already normalized/delta-extracted)
        """
        if not text or not text.strip():
            return

        self._thinking_char_count += len(text)
        self._thinking_lines.append(text)

        self._live_manager.pause()

        # For short thinking, render inline immediately
        if self._thinking_char_count < self._LONG_THINKING_THRESHOLD_CHARS:
            render_thinking_text(self._live_manager.console, text)
        else:
            # For long thinking, show periodic summaries
            elapsed = self.thinking_duration
            if len(self._thinking_lines) % 5 == 0:
                self._live_manager.console.print(
                    f"  [dim]... continuing reasoning "
                    f"({self._thinking_char_count} chars, "
                    f"{elapsed:.1f}s)[/]"
                )
            else:
                # Still render last chunk but dimmed
                render_thinking_text(self._live_manager.console, text)

        self._live_manager.resume()

    def on_thinking_end(self) -> None:
        """Exit thinking state and resume Live display.

        Shows a summary of the thinking session if it was long,
        then transitions back to normal content display.
        """
        self._in_thinking_mode = False

        # Show thinking summary for long sessions
        if self._thinking_char_count > self._LONG_THINKING_THRESHOLD_CHARS:
            elapsed = self.thinking_duration
            self._live_manager.pause()
            self._live_manager.console.print(
                f"  [dim]✓ Reasoning complete "
                f"({self._thinking_char_count} chars, "
                f"{elapsed:.1f}s)[/]"
            )
            self._live_manager.resume()

        self._live_manager.resume()

    def handle_content(self, text: str) -> None:
        """Route content based on thinking state.

        During thinking mode, content prints directly to console and the
        'shown' pointer is advanced to avoid re-display on resume.
        During normal mode, content goes to the Live display buffer.

        Args:
            text: Content text to display
        """
        self._live_manager.append_content(text)

        if self._in_thinking_mode:
            self._live_manager.pause()
            render_thinking_text(self._live_manager.console, text)
            self._live_manager.advance_shown_content()
            self._live_manager.resume()
        else:
            t0 = time.monotonic() * 1000
            self._live_manager.update_live()
            duration_ms = time.monotonic() * 1000 - t0
            self._live_manager.get_metrics().record_content_chunk(duration_ms)
            if duration_ms > 100:
                logger.debug("LiveDisplayRenderer: slow render %.1fms", duration_ms)

    def reset(self) -> None:
        """Reset thinking state (for cleanup between turns)."""
        self._thinking_indicator_shown = False
        self._in_thinking_mode = False
        self._thinking_start_time = None
        self._thinking_char_count = 0
        self._thinking_lines.clear()
