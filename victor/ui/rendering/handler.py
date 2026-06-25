"""Stream response handler - unified streaming response processing.

This module provides the stream_response() function which is the single
source of truth for streaming response handling across all CLI modes.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, TYPE_CHECKING

from victor.agent.response_sanitizer import create_streaming_filter
from victor.framework.events import EventType
from victor.ui.rendering.protocol import StreamRenderer
from victor.ui.rendering.utils import (
    extract_tool_call_arguments,
    extract_tool_result_payload,
    StreamDeltaNormalizer,
    is_thinking_status_message,
    normalize_reasoning_content,
)

logger = logging.getLogger(__name__)

# ✅ PROPER: No TYPE_CHECKING imports needed (using Any for agent parameter)


def _resolve_stream_factory(agent: Any) -> Any:
    """Resolve the streaming entrypoint while preserving legacy compatibility."""

    agent_dict = getattr(agent, "__dict__", {})

    if "stream" in agent_dict or hasattr(type(agent), "stream"):
        return agent.stream
    if "stream_chat" in agent_dict or hasattr(type(agent), "stream_chat"):
        return agent.stream_chat

    stream_factory = getattr(agent, "stream", None)
    if callable(stream_factory):
        return stream_factory

    legacy_stream_factory = getattr(agent, "stream_chat", None)
    if callable(legacy_stream_factory):
        return legacy_stream_factory

    raise AttributeError("Agent does not expose stream() or stream_chat()")


async def stream_response(
    agent: Any,  # ✅ PROPER: Use Any instead of AgentOrchestrator (duck typing)
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
      `event.metadata["reasoning_content"]`. This is handled directly and
      rendered through `renderer.on_thinking_content()`.
    - **Inline markers**: Models like Qwen3 or local Ollama models may use
      inline markers (`<think>...</think>`, `<|begin_of_thinking|>`).
      These are processed by `StreamingContentFilter` from response_sanitizer.

    State management automatically handles transitions between thinking
    and normal content, including when switching from API-based reasoning
    to regular content output.

    Args:
        agent: The agent instance to stream from (Agent facade with stream method)
        message: The user message to send
        renderer: The renderer to use for output
        suppress_thinking: If True, completely hide thinking content

    Returns:
        The accumulated response content
    """
    renderer.start()

    # Register a UI-ephemeral progress sink so long-running tools can stream
    # partial output into the live display while they run. Best-effort and
    # gated by settings; cleared in the finally below so it never leaks across
    # turns. Progress never enters the conversation or reaches the model.
    _progress_sink_registered = False
    on_progress = getattr(renderer, "on_tool_progress", None)
    if callable(on_progress):
        try:
            from victor.config.tool_settings import get_tool_settings
            from victor.framework.tool_progress import set_progress_sink

            if get_tool_settings().tool_progress_streaming_enabled:
                set_progress_sink(on_progress)
                _progress_sink_registered = True
        except Exception:  # pragma: no cover - defensive wiring
            logger.debug("Failed to register tool progress sink", exc_info=True)

    stream_gen = _resolve_stream_factory(agent)(message)

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

    error_surfaced = False
    try:
        async for event in stream_gen:
            chunk_count += 1
            event_type = getattr(event, "type", None)
            event_tool_name = getattr(event, "tool_name", None)
            event_metadata = getattr(event, "metadata", None)
            event_content = getattr(event, "content", "")

            # Log every 100th event to diagnose duplication/buffering issues
            if chunk_count % 100 == 0:
                content_len = len(event_content) if event_content else 0
                logger.debug(
                    f"Stream event #{chunk_count}: "
                    f"type={event_type}, "
                    f"content_len={content_len}, "
                    f"is_thinking={content_filter.is_thinking}"
                )

            # Handle structured tool events
            if event_metadata and "tool_start" in event_metadata:
                tool_start = event_metadata["tool_start"] or {}
                renderer.on_tool_start(
                    name=str(tool_start.get("name", event_tool_name or "unknown")),
                    arguments=tool_start.get("arguments", {}),
                )
            elif event_metadata and "tool_result" in event_metadata:
                result_data = event_metadata["tool_result"] or {}
                tool_result_kwargs = {
                    "name": str(result_data.get("name", event_tool_name or "unknown")),
                    "success": result_data.get("success", True),
                    "elapsed": result_data.get("elapsed", 0),
                    "arguments": result_data.get("arguments", {}),
                    "error": result_data.get("error"),
                    "follow_up_suggestions": result_data.get("follow_up_suggestions"),
                    "result": result_data.get("result"),
                }
                if "original_result" in result_data:
                    tool_result_kwargs["original_result"] = result_data.get("original_result")
                if result_data.get("was_pruned"):
                    tool_result_kwargs["was_pruned"] = True
                renderer.on_tool_result(**tool_result_kwargs)
            elif event_type == EventType.TOOL_CALL:
                renderer.on_tool_start(
                    name=event_tool_name or "unknown",
                    arguments=extract_tool_call_arguments(event),
                )
            elif event_type == EventType.TOOL_RESULT:
                result_data = extract_tool_result_payload(event)
                tool_result_kwargs = {
                    "name": str(result_data.get("name", event_tool_name or "unknown")),
                    "success": result_data.get("success", True),
                    "elapsed": result_data.get("elapsed", 0),
                    "arguments": result_data.get("arguments", {}),
                    "error": result_data.get("error"),
                    "follow_up_suggestions": result_data.get("follow_up_suggestions"),
                    "result": result_data.get("result"),
                }
                if "original_result" in result_data:
                    tool_result_kwargs["original_result"] = result_data.get("original_result")
                if result_data.get("was_pruned"):
                    tool_result_kwargs["was_pruned"] = True
                renderer.on_tool_result(
                    **tool_result_kwargs,
                )
            # Surface terminal stream errors (e.g. provider model-not-found) to
            # the user instead of dropping them — otherwise the stream simply
            # ends empty and we misreport it as a possible bug below.
            elif event_type == EventType.ERROR:
                error_text = (
                    getattr(event, "error", None)
                    or event_content
                    or "The provider returned an error."
                )
                renderer.on_status(f"❌ {error_text}")
                error_surfaced = True
                break
            # Handle streamed tool progress (live terminal block). UI-ephemeral:
            # never added to conversation or sent to the model.
            elif (
                event_metadata and "tool_progress" in event_metadata
            ) or event_type == EventType.TOOL_PROGRESS:
                progress_data = (event_metadata or {}).get("tool_progress") or {}
                on_progress = getattr(renderer, "on_tool_progress", None)
                if callable(on_progress):
                    on_progress(
                        name=str(progress_data.get("name", event_tool_name or "unknown")),
                        stdout=progress_data.get("stdout", event_content or ""),
                        stderr=progress_data.get("stderr", ""),
                        progress=progress_data.get("progress", 0.0),
                        is_final=bool(progress_data.get("is_final", False)),
                    )
            # Handle status messages (thinking indicator, etc.)
            elif event_metadata and "status" in event_metadata:
                status_message = str(event_metadata["status"])
                if is_thinking_status_message(status_message):
                    if not suppress_thinking and not was_thinking:
                        logger.debug("→ Entering thinking mode (status event)")
                        renderer.on_thinking_start()
                        reasoning_normalizer.reset()
                        was_thinking = True
                else:
                    renderer.on_status(status_message)
            # Handle file preview
            elif event_metadata and "file_preview" in event_metadata:
                renderer.on_file_preview(
                    path=event_metadata.get("path", ""),
                    content=event_metadata["file_preview"],
                )
            # Handle edit preview
            elif event_metadata and "edit_preview" in event_metadata:
                renderer.on_edit_preview(
                    path=event_metadata.get("path", ""),
                    diff=event_metadata["edit_preview"],
                )
            # Handle reasoning_content from DeepSeek API (separate from inline markers)
            # DeepSeek sends reasoning via metadata, not inline <think> markers
            elif event_metadata and "reasoning_content" in event_metadata:
                reasoning = event_metadata["reasoning_content"]
                if reasoning and not suppress_thinking:
                    # Filter out provider-added continuation markers to avoid
                    # duplicating our own thinking indicator in the transcript.
                    reasoning = normalize_reasoning_content(reasoning)

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
                if event_content:
                    process_content(event_content)
            # Handle content - filter through StreamingContentFilter
            elif event_content:
                process_content(event_content)

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
        # Empty content is valid in some flows (e.g. empty stream or metadata-only events),
        # so return the renderer output directly and only log for visibility. A surfaced
        # error already explains the empty result, so don't misreport it as a bug.
        if not final_content and not renderer.had_tool_calls() and not error_surfaced:
            logger.warning("stream_response returned empty content - this may indicate a bug")

        return final_content

    finally:
        if _progress_sink_registered:
            try:
                from victor.framework.tool_progress import clear_progress_sink

                clear_progress_sink()
            except Exception:  # pragma: no cover - defensive
                pass
        renderer.cleanup()
        # Graceful cleanup of async generator to prevent RuntimeError on abort
        aclose = getattr(stream_gen, "aclose", None)
        if callable(aclose):
            try:
                maybe_awaitable = aclose()
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
            except RuntimeError:
                # Generator already closed or running - ignore
                pass
