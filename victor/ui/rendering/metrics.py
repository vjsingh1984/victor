"""Streaming performance metrics.

Collected by renderers during a streaming session and accessible via
renderer.get_metrics() after the session ends.
"""

from __future__ import annotations

from dataclasses import dataclass

SLOW_RENDER_THRESHOLD_MS: float = 100.0


@dataclass
class StreamingMetrics:
    """Performance counters for a single streaming session.

    All timing values are in milliseconds.
    """

    pause_count: int = 0
    resume_count: int = 0
    content_chunks: int = 0
    tool_results: int = 0
    total_pause_ms: float = 0.0
    total_content_ms: float = 0.0
    slow_renders: int = 0  # content renders that exceeded SLOW_RENDER_THRESHOLD_MS

    def record_pause(self, duration_ms: float) -> None:
        self.pause_count += 1
        self.total_pause_ms += duration_ms

    def record_resume(self) -> None:
        self.resume_count += 1

    def record_content_chunk(self, duration_ms: float) -> None:
        self.content_chunks += 1
        self.total_content_ms += duration_ms
        if duration_ms > SLOW_RENDER_THRESHOLD_MS:
            self.slow_renders += 1

    def record_tool_result(self) -> None:
        self.tool_results += 1

    @property
    def avg_content_ms(self) -> float:
        if self.content_chunks == 0:
            return 0.0
        return self.total_content_ms / self.content_chunks
