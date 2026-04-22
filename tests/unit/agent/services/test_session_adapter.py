"""Tests for SessionServiceAdapter."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.adapters.session_adapter import SessionServiceAdapter


@pytest.fixture
def mock_session_service():
    return SimpleNamespace(
        get_recent_sessions=MagicMock(
            return_value=[{"session_id": "abc", "created_at": "2024-01-01"}]
        ),
        recover_session=MagicMock(return_value=True),
        get_session_stats=MagicMock(return_value={"messages": 10, "tool_calls": 5}),
        get_memory_context=MagicMock(return_value=[{"role": "user", "content": "hi"}]),
        save_checkpoint=AsyncMock(return_value="ckpt-123"),
        restore_checkpoint=AsyncMock(return_value=True),
        maybe_auto_checkpoint=AsyncMock(return_value=None),
    )


@pytest.fixture
def mock_session_coordinator():
    return SimpleNamespace(
        get_recent_sessions=MagicMock(return_value=[{"session_id": "legacy"}]),
        recover_session=MagicMock(return_value=False),
        get_session_stats=MagicMock(return_value={"messages": 1}),
        get_memory_context=MagicMock(return_value=[]),
        save_checkpoint=AsyncMock(return_value="legacy-ckpt"),
        restore_checkpoint=AsyncMock(return_value=False),
        maybe_auto_checkpoint=AsyncMock(return_value="legacy-auto"),
    )


@pytest.fixture
def session_adapter(mock_session_service, mock_session_coordinator):
    return SessionServiceAdapter(
        mock_session_service,
        deprecated_session_coordinator=mock_session_coordinator,
    )


def test_get_recent_sessions_prefers_service(
    session_adapter,
    mock_session_service,
    mock_session_coordinator,
):
    result = session_adapter.get_recent_sessions(5)

    mock_session_service.get_recent_sessions.assert_called_once_with(5)
    mock_session_coordinator.get_recent_sessions.assert_not_called()
    assert len(result) == 1


def test_recover_session_prefers_service(
    session_adapter,
    mock_session_service,
    mock_session_coordinator,
):
    assert session_adapter.recover_session("abc") is True

    mock_session_service.recover_session.assert_called_once_with("abc")
    mock_session_coordinator.recover_session.assert_not_called()


def test_get_session_stats_prefers_service(
    session_adapter,
    mock_session_service,
    mock_session_coordinator,
):
    stats = session_adapter.get_session_stats()

    mock_session_service.get_session_stats.assert_called_once()
    mock_session_coordinator.get_session_stats.assert_not_called()
    assert stats["messages"] == 10


def test_get_memory_context_prefers_service(
    session_adapter,
    mock_session_service,
    mock_session_coordinator,
):
    ctx = session_adapter.get_memory_context(max_tokens=100, messages=[])

    mock_session_service.get_memory_context.assert_called_once_with(max_tokens=100, messages=[])
    mock_session_coordinator.get_memory_context.assert_not_called()
    assert len(ctx) == 1


@pytest.mark.asyncio
async def test_save_checkpoint_prefers_service(
    session_adapter,
    mock_session_service,
    mock_session_coordinator,
):
    result = await session_adapter.save_checkpoint("test", ["tag1"])

    mock_session_service.save_checkpoint.assert_awaited_once_with("test", ["tag1"])
    mock_session_coordinator.save_checkpoint.assert_not_awaited()
    assert result == "ckpt-123"


@pytest.mark.asyncio
async def test_restore_checkpoint_prefers_service(
    session_adapter,
    mock_session_service,
    mock_session_coordinator,
):
    result = await session_adapter.restore_checkpoint("ckpt-123")

    mock_session_service.restore_checkpoint.assert_awaited_once_with("ckpt-123")
    mock_session_coordinator.restore_checkpoint.assert_not_awaited()
    assert result is True


@pytest.mark.asyncio
async def test_maybe_auto_checkpoint_prefers_service(
    session_adapter,
    mock_session_service,
    mock_session_coordinator,
):
    result = await session_adapter.maybe_auto_checkpoint()

    mock_session_service.maybe_auto_checkpoint.assert_awaited_once()
    mock_session_coordinator.maybe_auto_checkpoint.assert_not_awaited()
    assert result is None


def test_is_healthy(session_adapter):
    assert session_adapter.is_healthy() is True


def test_is_healthy_with_none():
    adapter = SessionServiceAdapter(None)
    assert adapter.is_healthy() is False


def test_explicit_coordinator_fallback_warns(mock_session_coordinator):
    with pytest.warns(
        DeprecationWarning,
        match="coordinator fallback only",
    ):
        adapter = SessionServiceAdapter(
            None,
            deprecated_session_coordinator=mock_session_coordinator,
        )

    assert adapter.get_session_stats() == {"messages": 1}
    mock_session_coordinator.get_session_stats.assert_called_once()


def test_old_session_coordinator_kwarg_warns(mock_session_service, mock_session_coordinator):
    with pytest.warns(
        DeprecationWarning,
        match="SessionServiceAdapter\\(session_coordinator=...\\) is deprecated",
    ):
        adapter = SessionServiceAdapter(
            mock_session_service,
            session_coordinator=mock_session_coordinator,
        )

    assert adapter.get_session_stats() == {"messages": 10, "tool_calls": 5}


def test_old_and_new_session_coordinator_kwargs_conflict(
    mock_session_service,
    mock_session_coordinator,
):
    with pytest.raises(
        TypeError,
        match="Use only one of session_coordinator or deprecated_session_coordinator",
    ):
        SessionServiceAdapter(
            mock_session_service,
            session_coordinator=mock_session_coordinator,
            deprecated_session_coordinator=mock_session_coordinator,
        )


def test_legacy_positional_coordinator_is_still_supported(mock_session_coordinator):
    adapter = SessionServiceAdapter(mock_session_coordinator)

    assert adapter.get_session_stats() == {"messages": 1}
    mock_session_coordinator.get_session_stats.assert_called_once()
