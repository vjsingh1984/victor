"""Tests for SessionServiceAdapter."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.adapters.session_adapter import SessionServiceAdapter


@pytest.fixture
def mock_session_coordinator():
    coordinator = MagicMock()
    coordinator.get_recent_sessions.return_value = [
        {"session_id": "abc", "created_at": "2024-01-01"}
    ]
    coordinator.recover_session.return_value = True
    coordinator.get_session_stats.return_value = {"messages": 10, "tool_calls": 5}
    coordinator.get_memory_context.return_value = [{"role": "user", "content": "hi"}]
    coordinator.save_checkpoint = AsyncMock(return_value="ckpt-123")
    coordinator.restore_checkpoint = AsyncMock(return_value=True)
    coordinator.maybe_auto_checkpoint = AsyncMock(return_value=None)
    return coordinator


@pytest.fixture
def session_adapter(mock_session_coordinator):
    return SessionServiceAdapter(mock_session_coordinator)


def test_get_recent_sessions(session_adapter, mock_session_coordinator):
    result = session_adapter.get_recent_sessions(5)
    mock_session_coordinator.get_recent_sessions.assert_called_once_with(5)
    assert len(result) == 1


def test_recover_session(session_adapter, mock_session_coordinator):
    assert session_adapter.recover_session("abc") is True
    mock_session_coordinator.recover_session.assert_called_once_with("abc")


def test_get_session_stats(session_adapter, mock_session_coordinator):
    stats = session_adapter.get_session_stats()
    assert stats["messages"] == 10


def test_get_memory_context(session_adapter, mock_session_coordinator):
    ctx = session_adapter.get_memory_context(max_tokens=100, messages=[])
    mock_session_coordinator.get_memory_context.assert_called_once_with(max_tokens=100, messages=[])
    assert len(ctx) == 1


async def test_save_checkpoint(session_adapter, mock_session_coordinator):
    result = await session_adapter.save_checkpoint("test", ["tag1"])
    mock_session_coordinator.save_checkpoint.assert_awaited_once_with("test", ["tag1"])
    assert result == "ckpt-123"


async def test_restore_checkpoint(session_adapter, mock_session_coordinator):
    result = await session_adapter.restore_checkpoint("ckpt-123")
    mock_session_coordinator.restore_checkpoint.assert_awaited_once_with("ckpt-123")
    assert result is True


async def test_maybe_auto_checkpoint(session_adapter, mock_session_coordinator):
    result = await session_adapter.maybe_auto_checkpoint()
    mock_session_coordinator.maybe_auto_checkpoint.assert_awaited_once()
    assert result is None


def test_is_healthy(session_adapter):
    assert session_adapter.is_healthy() is True


def test_is_healthy_with_none():
    adapter = SessionServiceAdapter(None)
    assert adapter.is_healthy() is False
