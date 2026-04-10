"""Tests for ChatServiceAdapter."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.adapters.chat_adapter import ChatServiceAdapter


@pytest.fixture
def mock_chat_coordinator():
    coordinator = MagicMock()
    coordinator.chat = AsyncMock(return_value=MagicMock(content="response"))
    coordinator.chat_with_planning = AsyncMock(return_value=MagicMock(content="planned"))
    coordinator.stream_chat = MagicMock()
    return coordinator


@pytest.fixture
def chat_adapter(mock_chat_coordinator):
    return ChatServiceAdapter(mock_chat_coordinator)


async def test_chat_delegates_to_coordinator(chat_adapter, mock_chat_coordinator):
    result = await chat_adapter.chat("hello")
    mock_chat_coordinator.chat.assert_awaited_once_with("hello")
    assert result.content == "response"


async def test_chat_with_planning_delegates(chat_adapter, mock_chat_coordinator):
    result = await chat_adapter.chat_with_planning("complex task", use_planning=True)
    mock_chat_coordinator.chat_with_planning.assert_awaited_once_with("complex task", True)
    assert result.content == "planned"


async def test_stream_chat_delegates(chat_adapter, mock_chat_coordinator):
    chunks = [MagicMock(content="chunk1"), MagicMock(content="chunk2")]

    async def mock_stream(msg):
        for chunk in chunks:
            yield chunk

    mock_chat_coordinator.stream_chat = mock_stream

    collected = []
    async for chunk in chat_adapter.stream_chat("hello"):
        collected.append(chunk)
    assert len(collected) == 2
    assert collected[0].content == "chunk1"


def test_is_healthy(chat_adapter):
    assert chat_adapter.is_healthy() is True


def test_is_healthy_with_none():
    adapter = ChatServiceAdapter(None)
    assert adapter.is_healthy() is False


def test_reset_conversation_delegates(chat_adapter, mock_chat_coordinator):
    mock_chat_coordinator.reset_conversation = MagicMock()
    chat_adapter.reset_conversation()
    mock_chat_coordinator.reset_conversation.assert_called_once()
