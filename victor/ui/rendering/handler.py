"""Stream response handler - unified streaming response processing.

This module provides the stream_response() function which is the single
source of truth for streaming response handling across all CLI modes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from victor.agent.response_sanitizer import create_streaming_filter
from victor.ui.rendering.protocol import StreamRenderer
from victor.ui.rendering.utils import StreamDeltaNormalizer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator


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
    content_filter = create_streaming_filter(suppress_thinking=suppress_thinking)
    content_normalizer = StreamDeltaNormalizer()
    reasoning_normalizer = StreamDeltaNormalizer()
    was_thinking = False

    # ENHANCEMENT: Log chunk boundaries for debugging streaming issues
    chunk_count = 0

    def process_content(raw_content: str) -> None:
        """Normalize and render regular or inline-thinking content."""
        nonlocal was_thinking

        normalized_content = content_normalizer.consume(raw_content)
        if not normalized_content:
            return

        # End API-based thinking state when regular content arrives.
        if was_thinking and not content_filter.is_thinking:
            logger.debug("← Exiting thinking mode (API reasoning ended)")
            renderer.on_thinking_end()
            reasoning_normalizer.reset()
            was_thinking = False

        result = content_filter.process_chunk(normalized_content)

        # Log inline thinking state transitions for debugging.
        if result.entering_thinking and not was_thinking:
            logger.debug("→ Entering thinking mode (inline markers)")
            renderer.on_thinking_start()
            was_thinking = True

        if result.content:
            if result.is_thinking:
                thinking_delta = reasoning_normalizer.consume(result.content)
                if thinking_delta:
                    renderer.on_thinking_content(thinking_delta)
            else:
                renderer.on_content(result.content)

        if result.exiting_thinking and was_thinking:
            logger.debug("← Exiting thinking mode (inline markers ended)")
            renderer.on_thinking_end()
            reasoning_normalizer.reset()
            was_thinking = False

    try:
        async for chunk in stream_gen:
            chunk_count += 1

            # Log every 100th chunk to diagnose duplication/buffering issues
            if chunk_count % 100 == 0:
                content_len = len(chunk.content) if chunk.content else 0
                metadata_keys = list(chunk.metadata.keys()) if chunk.metadata else []
                logger.debug(
                    f"Stream chunk #{chunk_count}: "
                    f"content_len={content_len}, "
                    f"metadata_keys={metadata_keys}, "
                    f"is_thinking={content_filter.is_thinking}"
                )

            # Handle structured tool events
            if chunk.metadata and "tool_start" in chunk.metadata:
                tool_data = chunk.metadata["tool_start"]
                renderer.on_tool_start(
                    name=tool_data["name"],
                    arguments=tool_data.get("arguments", {}),
                )
            elif chunk.metadata and "tool_result" in chunk.metadata:
                tool_data = chunk.metadata["tool_result"]
                tool_result_kwargs = {
                    "name": tool_data["name"],
                    "success": tool_data.get("success", True),
                    "elapsed": tool_data.get("elapsed", 0),
                    "arguments": tool_data.get("arguments", {}),
                    "error": tool_data.get("error"),
                    "follow_up_suggestions": tool_data.get("follow_up_suggestions"),
                    "result": tool_data.get("result"),
                }
                if tool_data.get("was_pruned"):
                    tool_result_kwargs["was_pruned"] = True
                renderer.on_tool_result(
                    **tool_result_kwargs,
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
                    # Filter out continuation markers from provider responses
                    # (e.g., "💭 Thinking...", "🤔 Thinking...") to avoid redundancy with our badge
                    reasoning = reasoning.lstrip()
                    for marker in ["💭 Thinking...", "💭", "🤔 Thinking...", "🤔", "Thinking..."]:
                        if reasoning.startswith(marker):
                            reasoning = reasoning[len(marker):].lstrip()
                            break

                    # Log state transition for debugging
                    if not was_thinking:
                        logger.debug("→ Entering thinking mode (API reasoning)")
                        renderer.on_thinking_start()
                        reasoning_normalizer.reset()
                        was_thinking = True
                    reasoning_delta = reasoning_normalizer.consume(reasoning)
                    if reasoning_delta:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "reasoning_normalizer: input_len=%d, delta_len=%d, "
                                "input_preview=%r..., delta_preview=%r...",
                                len(reasoning),
                                len(reasoning_delta),
                                reasoning[:60],
                                reasoning_delta[:60],
                            )
                        renderer.on_thinking_content(reasoning_delta)
                    elif logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "reasoning_normalizer: filtered duplicate, input_len=%d, "
                            "input_preview=%r...",
                            len(reasoning),
                            reasoning[:60],
                        )
                if chunk.content:
                    process_content(chunk.content)
            # Handle content - filter through StreamingContentFilter
            elif chunk.content:
                process_content(chunk.content)

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

        # CRITICAL FIX: Ensure thinking state is properly closed
        if was_thinking:
            renderer.on_thinking_end()
            reasoning_normalizer.reset()
            was_thinking = False

        # CRITICAL FIX: Add comprehensive logging before finalize
        logger.debug(
            "stream_response completion - content_buffer_len=%d, "
            "in_thinking_mode=%s, live_active=%s",
            len(getattr(renderer, "_content_buffer", "")),
            getattr(renderer, "_in_thinking_mode", False),
            getattr(renderer, "_live", None) is not None,
        )

        # Log first 100 chars of content buffer for verification
        if logger.isEnabledFor(logging.DEBUG):
            content_preview = getattr(renderer, "_content_buffer", "")[:100]
            logger.debug("Content buffer preview: %r...", content_preview)

        # FAIL-SAFE: Verify content was displayed
        final_content = renderer.finalize()
        if not final_content:
            logger.warning("stream_response returned empty content - this may indicate a bug")

        return final_content

    finally:
        renderer.cleanup()
        # Graceful cleanup of async generator to prevent RuntimeError on abort
        if hasattr(stream_gen, "aclose"):
            try:
                await stream_gen.aclose()
            except RuntimeError:
                # Generator already closed or running - ignore
                pass
