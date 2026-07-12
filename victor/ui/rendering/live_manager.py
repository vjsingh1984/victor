"""Live display lifecycle manager for interactive streaming.

Extracted from LiveDisplayRenderer to separate Live lifecycle concerns
(content buffer, pause/resume, incremental rendering) from tool display
and thinking display logic.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.rule import Rule

from victor.ui.rendering.markdown import find_safe_split, render_markdown_with_hooks
from victor.ui.rendering.metrics import StreamingMetrics

logger = logging.getLogger(__name__)


class LiveManager:
    """Manages the Rich Live display lifecycle and content buffer.

    Responsibilities:
    - Start/stop/pause/resume the Rich Live display
    - Maintain the content buffer (single source of truth)
    - Incremental streaming render (HEAD/TAIL split)
    - Track pause depth for nested pause/resume callers

    Attributes:
        console: Rich Console for output
        content_buffer: Single source of truth for all content
    """

    # Cap to prevent unbounded memory growth
    MAX_CONTENT_BUFFER_SIZE = 50_000

    def __init__(self, console: Console):
        """Initialize LiveManager.

        Args:
            console: Rich Console for output
        """
        self.console = console
        self._live: Live | None = None
        self._content_buffer = ""
        self._is_paused = False
        self._pause_count = 0
        self._pause_start_ms: float | None = None
        self._metrics = StreamingMetrics()
        self._content_shown_before_pause = ""

        # Incremental streaming render: cached HEAD (complete markdown blocks) so
        # only the active TAIL (in-progress last block) is re-rendered each tick.
        self._rendered_head: Any = None
        self._rendered_head_source: str = ""

    # ── Public API ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the Live display."""
        self._live = Live(
            render_markdown_with_hooks(""),
            console=self.console,
            refresh_per_second=10,
        )
        self._live.start()
        self._is_paused = False
        self._pause_count = 0
        self._invalidate_head_cache()

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
            logger.debug("LiveManager: paused (depth=1)")

    def resume(self) -> None:
        """Resume the Live display with only NEW content since the outermost pause.

        Only restarts the Live display when the depth counter reaches zero,
        preventing duplication from nested pause/resume callers (e.g., thinking
        blocks that contain tool calls).
        """
        if self._pause_count <= 0:
            logger.debug("LiveManager: resume() called with no matching pause — ignoring")
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
            self._invalidate_head_cache()
            logger.debug("LiveManager: resumed (depth=0)")

    def stop(self) -> None:
        """Stop the Live display and reset state."""
        if self._live:
            self._live.stop()
            self._live = None
        self._is_paused = False
        self._pause_count = 0
        self._content_shown_before_pause = ""

    def get_metrics(self) -> StreamingMetrics:
        """Return accumulated streaming metrics for this session."""
        return self._metrics

    # ── Content Buffer ──────────────────────────────────────────────────

    def append_content(self, text: str) -> None:
        """Append text to the content buffer with size cap.

        Args:
            text: Content text to append
        """
        if len(self._content_buffer) + len(text) > self.MAX_CONTENT_BUFFER_SIZE:
            excess = len(self._content_buffer) + len(text) - self.MAX_CONTENT_BUFFER_SIZE
            self._content_buffer = self._content_buffer[excess:]
        self._content_buffer += text

    @property
    def content_buffer(self) -> str:
        """The full accumulated content buffer."""
        return self._content_buffer

    @property
    def visible_content(self) -> str:
        """Post-pause slice of the content buffer (what the live panel shows)."""
        return self._content_buffer[len(self._content_shown_before_pause) :]

    def advance_shown_content(self) -> None:
        """Advance the 'shown' pointer so resume() starts from here."""
        self._content_shown_before_pause = self._content_buffer

    @property
    def is_paused(self) -> bool:
        """Whether the Live display is currently paused."""
        return self._is_paused

    @property
    def live(self) -> Live | None:
        """The underlying Rich Live instance."""
        return self._live

    # ── Incremental Rendering ───────────────────────────────────────────

    def update_live(self) -> None:
        """Update the Live display with current content.

        Uses incremental (HEAD/TAIL) rendering when enabled:
        - Caches the stable HEAD (complete markdown blocks)
        - Re-renders only the active TAIL (in-progress last block)
        - Falls back to full re-render when disabled or buffer is empty
        """
        if not self._live:
            return

        visible = self.visible_content
        if not self._incremental_render_enabled() or not visible:
            self._live.update(render_markdown_with_hooks(visible))
            return

        split = find_safe_split(visible)
        head_source = visible[:split]
        if head_source != self._rendered_head_source:
            self._rendered_head = render_markdown_with_hooks(head_source) if head_source else None
            self._rendered_head_source = head_source

        tail_render = render_markdown_with_hooks(visible[split:])
        if self._rendered_head is not None:
            self._live.update(Group(self._rendered_head, tail_render))
        else:
            self._live.update(tail_render)

    def update_live_with_renderable(self, renderable: Any) -> None:
        """Update the Live display with a specific renderable.

        Args:
            renderable: Any Rich renderable to display
        """
        if self._live:
            self._live.update(renderable)

    @staticmethod
    def _incremental_render_enabled() -> bool:
        """Incremental (head/tail) streaming render — default on, env-gated."""
        raw = os.getenv("VICTOR_INCREMENTAL_RENDER", "1").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def _invalidate_head_cache(self) -> None:
        """Drop the cached HEAD render (new slice / fallback / reset)."""
        self._rendered_head = None
        self._rendered_head_source = ""

    # ── Section Separator ───────────────────────────────────────────────

    def print_section_separator(self, title: str = "") -> None:
        """Print a subtle section separator for visual hierarchy.

        Args:
            title: Optional title to display in the separator
        """
        if title:
            self.console.print(Rule(f"[dim]{title}[/]", style="dim"))
        else:
            self.console.print(Rule(style="dim"))
