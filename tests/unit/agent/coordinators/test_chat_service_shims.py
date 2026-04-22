from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.coordinators.protocols import ExecutionMode
from victor.agent.coordinators.streaming_chat_coordinator import (
    StreamingChatCoordinator,
)
from victor.agent.coordinators.sync_chat_coordinator import SyncChatCoordinator
from victor.agent.coordinators.unified_chat_coordinator import UnifiedChatCoordinator
from victor.providers.base import CompletionResponse, StreamChunk


@pytest.mark.asyncio
async def test_sync_chat_coordinator_delegates_to_bound_chat_service():
    chat_service = AsyncMock()
    response = CompletionResponse(content="service", role="assistant")
    chat_service.chat.return_value = response

    coordinator = SyncChatCoordinator(
        chat_context=MagicMock(),
        tool_context=MagicMock(),
        provider_context=MagicMock(),
        turn_executor=MagicMock(),
        chat_service=chat_service,
    )

    result = await coordinator.chat("hello", use_planning=True)

    assert result is response
    chat_service.chat.assert_awaited_once_with("hello", use_planning=True)


@pytest.mark.asyncio
async def test_streaming_chat_coordinator_delegates_to_bound_chat_service():
    chunk = StreamChunk(content="partial", role="assistant")
    chat_service = MagicMock()

    async def _stream_chat(user_message: str):
        assert user_message == "hello"
        yield chunk

    chat_service.stream_chat = _stream_chat

    coordinator = StreamingChatCoordinator(
        chat_context=MagicMock(),
        tool_context=MagicMock(),
        provider_context=MagicMock(),
        chat_service=chat_service,
    )

    chunks = [item async for item in coordinator.stream_chat("hello")]

    assert chunks == [chunk]


@pytest.mark.asyncio
async def test_unified_chat_coordinator_delegates_to_bound_chat_service():
    sync = MagicMock()
    streaming = MagicMock()
    chat_service = AsyncMock()
    response = CompletionResponse(content="unified", role="assistant")
    chat_service.chat.return_value = response

    coordinator = UnifiedChatCoordinator(
        sync_coordinator=sync,
        streaming_coordinator=streaming,
        default_mode=ExecutionMode.SYNC,
        chat_service=chat_service,
    )

    result = await coordinator.chat("hello", mode=ExecutionMode.STREAMING, use_planning=False)

    assert result is response
    chat_service.chat.assert_awaited_once_with("hello", stream=True, use_planning=False)
