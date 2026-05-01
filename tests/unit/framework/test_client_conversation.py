"""Tests for VictorClient conversation management methods."""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from victor.framework.session_config import SessionConfig
from victor.framework.client import VictorClient


@pytest.mark.asyncio
async def test_victor_client_reset_conversation_delegates_to_chat_service() -> None:
    """Test reset_conversation delegates to ChatService."""
    config = SessionConfig()
    client = VictorClient(config, container=object())

    mock_chat_service = AsyncMock()
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=mock_chat_service))

    client._context = execution_context
    client._initialized = True

    await client.reset_conversation()

    mock_chat_service.reset_conversation.assert_called_once()


@pytest.mark.asyncio
async def test_victor_client_reset_conversation_raises_when_not_initialized() -> None:
    """Test reset_conversation raises RuntimeError when not initialized."""
    config = SessionConfig()
    client = VictorClient(config, container=object())

    with pytest.raises(RuntimeError, match="not initialized"):
        await client.reset_conversation()


@pytest.mark.asyncio
async def test_victor_client_get_messages_delegates_to_context_service() -> None:
    """Test get_messages delegates to ContextService."""
    config = SessionConfig()
    client = VictorClient(config, container=object())

    fake_messages = [MagicMock(role="user", content="hello")]
    mock_context_service = AsyncMock()
    mock_context_service.get_messages.return_value = fake_messages

    execution_context = SimpleNamespace(services=SimpleNamespace(context=mock_context_service))

    client._context = execution_context
    client._initialized = True

    messages = await client.get_messages(limit=10, role="user")

    mock_context_service.get_messages.assert_called_once_with(limit=10, role="user")
    assert messages == fake_messages


@pytest.mark.asyncio
async def test_victor_client_get_messages_raises_when_not_initialized() -> None:
    """Test get_messages raises RuntimeError when not initialized."""
    config = SessionConfig()
    client = VictorClient(config, container=object())

    with pytest.raises(RuntimeError, match="not initialized"):
        await client.get_messages()


@pytest.mark.asyncio
async def test_victor_client_get_messages_returns_empty_when_service_unavailable() -> None:
    """Test get_messages returns empty list when ContextService unavailable."""
    config = SessionConfig()
    client = VictorClient(config, container=object())

    execution_context = SimpleNamespace(services=SimpleNamespace(context=None))

    client._context = execution_context
    client._initialized = True

    messages = await client.get_messages()

    assert messages == []


@pytest.mark.asyncio
async def test_victor_client_reset_conversation_handles_missing_service() -> None:
    """Test reset_conversation handles missing ChatService gracefully."""
    config = SessionConfig()
    client = VictorClient(config, container=object())

    execution_context = SimpleNamespace(services=SimpleNamespace(chat=None))

    client._context = execution_context
    client._initialized = True

    # Should not raise, just log warning
    await client.reset_conversation()
