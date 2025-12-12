"""Stream rendering abstraction for CLI output.

This module implements the Strategy pattern to unify streaming response handling
across different CLI modes (oneshot vs interactive) while allowing pluggable
rendering strategies.

Design Pattern: Strategy + Protocol
- StreamRenderer protocol defines the interface
- FormatterRenderer uses OutputFormatter (for oneshot mode)
- LiveDisplayRenderer uses Rich Live display (for interactive mode)
- stream_response() is the unified handler that works with any renderer

Benefits:
- Single streaming loop eliminates code duplication
- Easy to add new renderers (JSON, plain text, TUI, etc.)
- Consistent behavior across all modes
- Testable - can mock the renderer for unit tests

Thinking Content Handling (dual-mode):
- **API-based reasoning**: DeepSeek API sends reasoning via metadata field
  (`chunk.metadata["reasoning_content"]`), rendered as dim/italic text
- **Inline markers**: Qwen3/Ollama local models use inline markers
  (`<think>...</think>`, `<|begin_of_thinking|>`), processed by
  StreamingContentFilter from response_sanitizer
- Automatic state transitions when switching between reasoning and normal output
- `suppress_thinking` option to completely hide thinking content
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from victor.agent.response_sanitizer import StreamingContentFilter

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.ui.output_formatter import OutputFormatter


@runtime_checkable
class StreamRenderer(Protocol):
    """Protocol defining the interface for stream renderers.

    Implementations handle the visual presentation of streaming responses,
    allowing different rendering strategies for different CLI modes.
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


class FormatterRenderer:
    """Renderer using OutputFormatter for oneshot mode.

    This renderer delegates to OutputFormatter methods which handle
    the visual presentation with proper formatting.
    """

    def __init__(self, formatter: OutputFormatter, console: Console):
        self.formatter = formatter
        self.console = console
        self._content_buffer = ""

    def start(self) -> None:
        self.formatter.start_streaming()

    def pause(self) -> None:
        self.formatter.end_streaming()

    def resume(self) -> None:
        self.formatter.start_streaming()

    def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
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
        self.pause()
        self.formatter.tool_result(
            tool_name=name,
            success=success,
            error=error,
        )
        self.resume()

    def on_status(self, message: str) -> None:
        self.pause()
        self.formatter.status(message)
        self.resume()

    def on_file_preview(self, path: str, content: str) -> None:
        self.pause()
        ext = path.split(".")[-1] if "." in path else "txt"
        syntax = Syntax(content, ext, theme="monokai", line_numbers=False)
        self.console.print(Panel(syntax, title=f"[dim]{path}[/]", border_style="dim"))
        self.resume()

    def on_edit_preview(self, path: str, diff: str) -> None:
        self.pause()
        self.console.print(f"[dim]{path}:[/]")
        for line in diff.split("\n"):
            if line.startswith("-"):
                self.console.print(f"[red]{line}[/]")
            elif line.startswith("+"):
                self.console.print(f"[green]{line}[/]")
            else:
                self.console.print(f"[dim]{line}[/]")
        self.resume()

    def on_content(self, text: str) -> None:
        self._content_buffer += text
        self.formatter.stream_chunk(text)

    def on_thinking_content(self, text: str) -> None:
        """Render thinking content as dimmed/italic text."""
        # Don't add to content buffer (thinking is ephemeral)
        # Use dim italic styling for thinking text
        styled = Text(text, style="dim italic")
        self.console.print(styled, end="")

    def on_thinking_start(self) -> None:
        """Show thinking indicator."""
        self.pause()
        self.console.print("[dim italic]💭 Thinking...[/]")

    def on_thinking_end(self) -> None:
        """Show end of thinking."""
        self.console.print()  # Newline after thinking
        self.resume()

    def finalize(self) -> str:
        self.formatter.response(content=self._content_buffer)
        return self._content_buffer

    def cleanup(self) -> None:
        # FormatterRenderer doesn't need explicit cleanup
        pass


class LiveDisplayRenderer:
    """Renderer using Rich Live display for interactive mode.

    This renderer manages a Live display that updates in real-time
    with markdown rendering.
    """

    def __init__(self, console: Console):
        self.console = console
        self._live: Live | None = None
        self._content_buffer = ""

    def start(self) -> None:
        self._live = Live(Markdown(""), console=self.console, refresh_per_second=10)
        self._live.start()

    def pause(self) -> None:
        if self._live:
            self._live.stop()

    def resume(self) -> None:
        self._live = Live(
            Markdown(self._content_buffer),
            console=self.console,
            refresh_per_second=10,
        )
        self._live.start()

    def _format_args(self, arguments: dict[str, Any]) -> str:
        """Format arguments for compact display."""
        if not arguments:
            return ""
        args_str = ", ".join(f"{k}={repr(v)[:30]}" for k, v in arguments.items())
        return f"({args_str})"

    def on_tool_start(self, name: str, arguments: dict[str, Any]) -> None:
        self.pause()
        args_display = self._format_args(arguments)
        self.console.print(f"[dim]🔧 {name}{args_display}...[/]")
        self.resume()

    def on_tool_result(
        self,
        name: str,
        success: bool,
        elapsed: float,
        arguments: dict[str, Any],
        error: str | None = None,
    ) -> None:
        self.pause()
        args_display = self._format_args(arguments)
        icon = "✓" if success else "✗"
        color = "green" if success else "red"
        self.console.print(f"[{color}]{icon}[/] {name}{args_display} [dim]({elapsed:.1f}s)[/]")
        self.resume()

    def on_status(self, message: str) -> None:
        self.pause()
        self.console.print(f"[dim]{message}[/]")
        self.resume()

    def on_file_preview(self, path: str, content: str) -> None:
        self.pause()
        ext = path.split(".")[-1] if "." in path else "txt"
        syntax = Syntax(content, ext, theme="monokai", line_numbers=False)
        self.console.print(Panel(syntax, title=f"[dim]{path}[/]", border_style="dim"))
        self.resume()

    def on_edit_preview(self, path: str, diff: str) -> None:
        self.pause()
        self.console.print(f"[dim]{path}:[/]")
        for line in diff.split("\n"):
            if line.startswith("-"):
                self.console.print(f"[red]{line}[/]")
            elif line.startswith("+"):
                self.console.print(f"[green]{line}[/]")
            else:
                self.console.print(f"[dim]{line}[/]")
        self.resume()

    def on_content(self, text: str) -> None:
        self._content_buffer += text
        if self._live:
            self._live.update(Markdown(self._content_buffer))

    def on_thinking_content(self, text: str) -> None:
        """Render thinking content as dimmed/italic text in Live display."""
        self.pause()
        styled = Text(text, style="dim italic")
        self.console.print(styled, end="")

    def on_thinking_start(self) -> None:
        """Show thinking indicator and pause Live display."""
        self.pause()
        self.console.print("[dim italic]💭 Thinking...[/]")

    def on_thinking_end(self) -> None:
        """End thinking and resume Live display."""
        self.console.print()  # Newline after thinking content
        self.resume()

    def finalize(self) -> str:
        self.cleanup()
        return self._content_buffer

    def cleanup(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None


async def stream_response(
    agent: AgentOrchestrator,
    message: str,
    renderer: StreamRenderer,
    suppress_thinking: bool = False,
) -> str:
    """Unified streaming response handler.

    This is the single source of truth for streaming response handling.
    It processes chunks from the agent and delegates rendering to the
    provided renderer implementation.

    Thinking Content Handling (dual-mode):
    - **API-based reasoning**: DeepSeek API sends reasoning content via
      `chunk.metadata["reasoning_content"]`. This is handled directly and
      rendered through `renderer.on_thinking_content()`.
    - **Inline markers**: Models like Qwen3 or local Ollama models may use
      inline markers (`<think>...</think>`, `<|begin_of_thinking|>`).
      These are processed by `StreamingContentFilter` from response_sanitizer.

    State management automatically handles transitions between thinking
    and normal content, including when switching from API-based reasoning
    to regular content output.

    Args:
        agent: The agent orchestrator to stream from
        message: The user message to send
        renderer: The renderer to use for output
        suppress_thinking: If True, completely hide thinking content

    Returns:
        The accumulated response content
    """
    renderer.start()
    stream_gen = agent.stream_chat(message)

    # Initialize content filter for thinking markers
    content_filter = StreamingContentFilter(suppress_thinking=suppress_thinking)
    was_thinking = False

    try:
        async for chunk in stream_gen:
            # Handle structured tool events
            if chunk.metadata and "tool_start" in chunk.metadata:
                tool_data = chunk.metadata["tool_start"]
                renderer.on_tool_start(
                    name=tool_data["name"],
                    arguments=tool_data.get("arguments", {}),
                )
            elif chunk.metadata and "tool_result" in chunk.metadata:
                tool_data = chunk.metadata["tool_result"]
                renderer.on_tool_result(
                    name=tool_data["name"],
                    success=tool_data.get("success", True),
                    elapsed=tool_data.get("elapsed", 0),
                    arguments=tool_data.get("arguments", {}),
                    error=tool_data.get("error"),
                )
            # Handle status messages (thinking indicator, etc.)
            elif chunk.metadata and "status" in chunk.metadata:
                renderer.on_status(chunk.metadata["status"])
            # Handle file preview
            elif chunk.metadata and "file_preview" in chunk.metadata:
                renderer.on_file_preview(
                    path=chunk.metadata.get("path", ""),
                    content=chunk.metadata["file_preview"],
                )
            # Handle edit preview
            elif chunk.metadata and "edit_preview" in chunk.metadata:
                renderer.on_edit_preview(
                    path=chunk.metadata.get("path", ""),
                    diff=chunk.metadata["edit_preview"],
                )
            # Handle reasoning_content from DeepSeek API (separate from inline markers)
            # DeepSeek sends reasoning via metadata, not inline <think> markers
            elif chunk.metadata and "reasoning_content" in chunk.metadata:
                reasoning = chunk.metadata["reasoning_content"]
                if reasoning and not suppress_thinking:
                    # Start thinking state if not already active
                    if not was_thinking:
                        renderer.on_thinking_start()
                        was_thinking = True
                    renderer.on_thinking_content(reasoning)
            # Handle content - filter through StreamingContentFilter
            elif chunk.content:
                # End API-based thinking state when regular content arrives
                # This handles the transition from DeepSeek reasoning to regular output
                if was_thinking and not content_filter.is_thinking:
                    renderer.on_thinking_end()
                    was_thinking = False
                result = content_filter.process_chunk(chunk.content)

                # Handle state transitions
                if result.entering_thinking and not was_thinking:
                    renderer.on_thinking_start()
                    was_thinking = True

                # Render content based on thinking state
                if result.content:
                    if result.is_thinking:
                        renderer.on_thinking_content(result.content)
                    else:
                        renderer.on_content(result.content)

                if result.exiting_thinking and was_thinking:
                    renderer.on_thinking_end()
                    was_thinking = False

                # Check if we should abort due to excessive thinking
                if content_filter.should_abort():
                    renderer.on_status(f"⚠️ {content_filter.abort_reason}")
                    break

        # Flush any remaining buffered content
        flush_result = content_filter.flush()
        if flush_result.content:
            if flush_result.is_thinking:
                renderer.on_thinking_content(flush_result.content)
            else:
                renderer.on_content(flush_result.content)

        # End thinking state if still active
        if was_thinking:
            renderer.on_thinking_end()

        return renderer.finalize()

    finally:
        renderer.cleanup()
        # Graceful cleanup of async generator to prevent RuntimeError on abort
        if hasattr(stream_gen, "aclose"):
            try:
                await stream_gen.aclose()
            except RuntimeError:
                # Generator already closed or running - ignore
                pass
