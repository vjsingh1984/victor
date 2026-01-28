"""Stream response handler - unified streaming response processing.

This module provides the stream_response() function which is the single
source of truth for streaming response handling across all CLI modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator

from victor.agent.response_sanitizer import StreamingContentFilter
from victor.ui.rendering.protocol import StreamRenderer

if TYPE_CHECKING:
    from victor.protocols import UIAgentProtocol


async def stream_response(
    agent: "UIAgentProtocol",
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
    stream_gen: AsyncIterator[Any] = agent.stream_chat(message)

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
