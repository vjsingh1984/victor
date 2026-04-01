"""Tests for stream failure content preservation.

Phase 3: When a provider stream fails mid-response, partial content
accumulated so far is preserved in conversation history with a
'[Stream interrupted]' suffix.
"""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.coordinators.streaming_chat_coordinator import (
    StreamingChatCoordinator,
)
from victor.providers.base import StreamChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(content: str = "") -> StreamChunk:
    return StreamChunk(content=content)


async def _async_chunks(
    chunks: list[StreamChunk],
    *,
    fail_after: int | None = None,
) -> AsyncIterator[StreamChunk]:
    """Yield *chunks*, optionally raising after *fail_after* items."""
    for idx, chunk in enumerate(chunks):
        if fail_after is not None and idx >= fail_after:
            raise ConnectionError("provider connection lost")
        yield chunk


def _build_coordinator() -> tuple[StreamingChatCoordinator, MagicMock]:
    """Return a coordinator wired to lightweight mocks."""
    chat_ctx = MagicMock()
    chat_ctx.conversation.ensure_system_prompt = MagicMock()
    chat_ctx.messages = []
    chat_ctx.add_message = MagicMock()

    tool_ctx = MagicMock()
    tool_ctx.tool_calls_used = 0
    tool_ctx.tool_budget = 10

    provider = MagicMock()
    provider.supports_tools = MagicMock(return_value=False)

    provider_ctx = MagicMock()
    provider_ctx.provider = provider
    provider_ctx._check_cancellation = MagicMock(return_value=False)
    provider_ctx.model = "test-model"
    provider_ctx.temperature = 0.7
    provider_ctx.max_tokens = 1024
    provider_ctx.thinking = None

    coord = StreamingChatCoordinator(
        chat_context=chat_ctx,
        tool_context=tool_ctx,
        provider_context=provider_ctx,
    )
    return coord, chat_ctx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_partial_content_saved_on_stream_failure() -> None:
    """Provider yields 3 chunks then raises -> partial content saved."""
    coord, chat_ctx = _build_coordinator()

    chunks = [_make_chunk("Hello"), _make_chunk(" cruel"), _make_chunk(" world")]
    provider = coord._provider_context.provider
    provider.stream = MagicMock(
        return_value=_async_chunks(chunks, fail_after=3),
    )

    # We need a 4th chunk to trigger the error (fail_after=3 means the
    # error fires when *attempting* to yield index 3).  Add a dummy so
    # the generator has a 4th iteration.
    all_chunks = chunks + [_make_chunk("!")]
    provider.stream = MagicMock(
        return_value=_async_chunks(all_chunks, fail_after=3),
    )

    collected: list[StreamChunk] = []
    with pytest.raises(ConnectionError, match="provider connection lost"):
        async for c in coord.stream_chat("hi"):
            collected.append(c)

    # Three chunks should have been yielded before the failure
    assert len(collected) == 3

    # Partial content must be saved with the interrupted suffix
    chat_ctx.add_message.assert_any_call(
        "assistant", "Hello cruel world\n\n[Stream interrupted]"
    )


async def test_normal_completion_no_suffix() -> None:
    """Provider completes normally -> no '[Stream interrupted]' suffix."""
    coord, chat_ctx = _build_coordinator()

    chunks = [_make_chunk("Good"), _make_chunk(" morning")]
    provider = coord._provider_context.provider
    provider.stream = MagicMock(return_value=_async_chunks(chunks))

    collected: list[StreamChunk] = []
    async for c in coord.stream_chat("hi"):
        collected.append(c)

    assert len(collected) == 2

    # The assistant message must be the full content without suffix
    chat_ctx.add_message.assert_any_call("assistant", "Good morning")

    # Ensure no interrupted suffix was ever stored
    for call in chat_ctx.add_message.call_args_list:
        args = call[0]
        if len(args) >= 2:
            assert "[Stream interrupted]" not in str(args[1])


async def test_failure_on_first_chunk_no_partial_saved() -> None:
    """Provider fails on the very first chunk -> no partial message added."""
    coord, chat_ctx = _build_coordinator()

    chunks = [_make_chunk("data")]
    provider = coord._provider_context.provider
    provider.stream = MagicMock(
        return_value=_async_chunks(chunks, fail_after=0),
    )

    collected: list[StreamChunk] = []
    with pytest.raises(ConnectionError, match="provider connection lost"):
        async for c in coord.stream_chat("hi"):
            collected.append(c)

    # Nothing was yielded
    assert len(collected) == 0

    # Only the "user" message should have been added (from stream_chat),
    # no assistant message at all.
    for call in chat_ctx.add_message.call_args_list:
        args = call[0]
        if len(args) >= 1:
            assert args[0] != "assistant"
