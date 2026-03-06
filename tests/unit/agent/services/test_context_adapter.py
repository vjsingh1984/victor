"""Tests for ContextServiceAdapter."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.adapters.context_adapter import ContextServiceAdapter


@pytest.fixture
def mock_conversation_controller():
    controller = MagicMock()
    controller.get_context_metrics.return_value = MagicMock(total_tokens=5000, message_count=10)
    controller.messages = [
        MagicMock(role="user", content="hello"),
        MagicMock(role="assistant", content="hi"),
    ]
    controller.add_message = MagicMock()
    return controller


@pytest.fixture
def mock_context_compactor():
    compactor = MagicMock()
    compactor.check_and_compact = AsyncMock(return_value=3)
    return compactor


@pytest.fixture
def context_adapter(mock_conversation_controller, mock_context_compactor):
    return ContextServiceAdapter(mock_conversation_controller, mock_context_compactor)


def test_get_context_metrics(context_adapter, mock_conversation_controller):
    metrics = context_adapter.get_context_metrics()
    mock_conversation_controller.get_context_metrics.assert_called_once()
    assert metrics.total_tokens == 5000


async def test_compact_context(context_adapter, mock_context_compactor):
    removed = await context_adapter.compact_context()
    mock_context_compactor.check_and_compact.assert_awaited_once()
    assert removed == 3


async def test_compact_context_no_compactor(mock_conversation_controller):
    adapter = ContextServiceAdapter(mock_conversation_controller, None)
    removed = await adapter.compact_context()
    assert removed == 0


def test_add_message(context_adapter, mock_conversation_controller):
    context_adapter.add_message("user", "test message")
    mock_conversation_controller.add_message.assert_called_once_with("user", "test message")


def test_get_messages(context_adapter):
    messages = context_adapter.get_messages()
    assert len(messages) == 2


def test_get_messages_with_limit(context_adapter):
    messages = context_adapter.get_messages(limit=1)
    assert len(messages) == 1


def test_get_messages_with_role_filter(context_adapter):
    messages = context_adapter.get_messages(role="user")
    assert len(messages) == 1
    assert messages[0].role == "user"


def test_is_healthy(context_adapter):
    assert context_adapter.is_healthy() is True


def test_is_healthy_with_none():
    adapter = ContextServiceAdapter(None)
    assert adapter.is_healthy() is False
