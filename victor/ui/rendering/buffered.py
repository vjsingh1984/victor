"""Buffered renderer for non-streaming mode.

Collects streaming events and renders them at completion.
Used for --no-stream mode to capture tool calls, reasoning,
and final content that would otherwise be swallowed.
"""

from __future__ import annotations

from typing import Any


class BufferedRenderer:
    """Collects streaming events and renders them at completion.

    Implements the StreamRenderer protocol but buffers all output
    until flush() or finalize() is called.
    """

    def __init__(self, show_reasoning: bool = False, plain: bool = False):
        self._show_reasoning = show_reasoning
        self._plain = plain
        self._tool_calls: list[dict[str, Any]] = []
        self._reasoning_chunks: list[str] = []
        self._content_chunks: list[str] = []
        self._statuses: list[str] = []

    def start(self) -> None:
        """Start the buffered display."""
        pass

    def pause(self) -> None:
        """Pause (no-op for buffered)."""
        pass

    def resume(self) -> None:
        """Resume (no-op for buffered)."""
        pass

    def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        """Record tool execution start."""
        self._tool_calls.append({"name": name, "arguments": arguments, "result": None})

    def on_tool_result(
        self,
        name: str,
        success: bool,
        elapsed: float,
        arguments: dict[str, Any],
        error: str | None = None,
        follow_up_suggestions: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record tool execution result."""
        # Update last matching tool call with result
        for tc in reversed(self._tool_calls):
            if tc["name"] == name and tc["result"] is None:
                tc["result"] = {
                    "success": success,
                    "elapsed": elapsed,
                    "error": error,
                }
                break

    def on_status(self, message: str) -> None:
        """Record status message."""
        self._statuses.append(message)

    def on_file_preview(self, path: str, content: str) -> None:
        """Record file preview (included in content)."""
        pass

    def on_edit_preview(self, path: str, diff: str) -> None:
        """Record edit preview (included in content)."""
        pass

    def on_content(self, text: str) -> None:
        """Buffer content chunk."""
        self._content_chunks.append(text)

    def on_thinking_content(self, text: str) -> None:
        """Buffer thinking content if show_reasoning is enabled."""
        if self._show_reasoning:
            self._reasoning_chunks.append(text)

    def on_thinking_start(self) -> None:
        """Handle transition into thinking state."""
        pass

    def on_thinking_end(self) -> None:
        """Handle transition out of thinking state."""
        pass

    def finalize(self) -> str:
        """Return accumulated content."""
        return "".join(self._content_chunks)

    def cleanup(self) -> None:
        """Clean up resources."""
        pass

    def flush(self, console: Any) -> None:
        """Print collected output to console.

        Args:
            console: Rich Console instance for output
        """
        from rich.markdown import Markdown

        # Print tool calls summary
        if self._tool_calls:
            for tc in self._tool_calls:
                result = tc.get("result", {})
                if result:
                    status = "[green]ok[/]" if result.get("success") else "[red]fail[/]"
                    elapsed = result.get("elapsed", 0)
                    error_msg = f" - {result['error']}" if result.get("error") else ""
                    console.print(
                        f"  [dim]Tool:[/] {tc['name']} {status} "
                        f"[dim]({elapsed:.1f}s){error_msg}[/]"
                    )
                else:
                    console.print(f"  [dim]Tool:[/] {tc['name']}")

        # Print reasoning if --show-reasoning
        if self._reasoning_chunks:
            reasoning_text = "".join(self._reasoning_chunks)
            console.print(f"\n[dim italic]{reasoning_text}[/]\n")

        # Print final content
        content = "".join(self._content_chunks)
        if content.strip():
            if self._plain:
                console.print(content)
            else:
                console.print(Markdown(content))
