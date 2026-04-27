"""Tests for deprecated streaming shim fail-fast behavior."""

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

    with pytest.warns(
        DeprecationWarning,
        match="StreamingChatCoordinator without a bound ChatService is deprecated",
    ):
        coord = StreamingChatCoordinator(
            chat_context=chat_ctx,
            tool_context=tool_ctx,
            provider_context=provider_ctx,
        )
    return coord, chat_ctx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_streaming_shim_requires_bound_chat_service_on_failure_path() -> None:
    coord, chat_ctx = _build_coordinator()
    with pytest.raises(RuntimeError, match="no bound ChatService"):
        with pytest.warns(
            DeprecationWarning,
            match="StreamingChatCoordinator.stream_chat\\(\\) is deprecated compatibility surface",
        ):
            _ = [c async for c in coord.stream_chat("hi")]
    chat_ctx.add_message.assert_not_called()


async def test_streaming_shim_requires_bound_chat_service_on_success_path() -> None:
    coord, chat_ctx = _build_coordinator()
    with pytest.raises(RuntimeError, match="no bound ChatService"):
        with pytest.warns(
            DeprecationWarning,
            match="StreamingChatCoordinator.stream_chat\\(\\) is deprecated compatibility surface",
        ):
            _ = [c async for c in coord.stream_chat("hi")]
    chat_ctx.add_message.assert_not_called()


async def test_streaming_shim_public_entrypoint_no_longer_owns_provider_streaming() -> None:
    coord, chat_ctx = _build_coordinator()
    with pytest.raises(RuntimeError, match="no bound ChatService"):
        with pytest.warns(
            DeprecationWarning,
            match="StreamingChatCoordinator.stream_chat\\(\\) is deprecated compatibility surface",
        ):
            _ = [c async for c in coord.stream_chat("hi")]
    chat_ctx.add_message.assert_not_called()
