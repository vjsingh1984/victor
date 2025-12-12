"""StreamRenderer protocol definition.

This module defines the protocol (interface) that all stream renderers
must implement for consistent streaming response handling.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StreamRenderer(Protocol):
    """Protocol defining the interface for stream renderers.

    Implementations handle the visual presentation of streaming responses,
    allowing different rendering strategies for different CLI modes.

    Methods:
        start: Initialize the streaming display
        pause: Temporarily pause to show status/tool output
        resume: Resume streaming after pause
        on_tool_start: Handle tool execution start event
        on_tool_result: Handle tool execution result event
        on_status: Handle status message
        on_file_preview: Handle file content preview
        on_edit_preview: Handle edit diff preview
        on_content: Handle content chunk
        on_thinking_content: Handle thinking content
        on_thinking_start: Handle transition into thinking state
        on_thinking_end: Handle transition out of thinking state
        finalize: Finalize and return accumulated content
        cleanup: Clean up resources
    """

    def start(self) -> None:
        """Start the streaming display."""
        ...

    def pause(self) -> None:
        """Pause streaming to show status/tool output."""
        ...

    def resume(self) -> None:
        """Resume streaming after pause."""
        ...

    def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        """Handle tool execution start event."""
        ...

    def on_tool_result(
        self,
        name: str,
        success: bool,
        elapsed: float,
        arguments: dict[str, Any],
        error: str | None = None,
    ) -> None:
        """Handle tool execution result event."""
        ...

    def on_status(self, message: str) -> None:
        """Handle status message (thinking indicator, etc.)."""
        ...

    def on_file_preview(self, path: str, content: str) -> None:
        """Handle file content preview."""
        ...

    def on_edit_preview(self, path: str, diff: str) -> None:
        """Handle edit diff preview."""
        ...

    def on_content(self, text: str) -> None:
        """Handle content chunk."""
        ...

    def on_thinking_content(self, text: str) -> None:
        """Handle thinking content (rendered dimmed/italic)."""
        ...

    def on_thinking_start(self) -> None:
        """Handle transition into thinking state."""
        ...

    def on_thinking_end(self) -> None:
        """Handle transition out of thinking state."""
        ...

    def finalize(self) -> str:
        """Finalize the response and return accumulated content."""
        ...

    def cleanup(self) -> None:
        """Clean up resources."""
        ...
