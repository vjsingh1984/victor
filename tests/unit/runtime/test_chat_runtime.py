from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.runtime import resolve_chat_runtime, resolve_chat_service


class _RuntimeOwner(MagicMock):
    """MagicMock subclass for readable runtime-owner doubles."""


def _make_runtime_owner() -> _RuntimeOwner:
    owner = _RuntimeOwner()
    owner._container = None
    return owner


def test_resolve_chat_runtime_prefers_execution_context_chat_service() -> None:
    runtime_owner = _make_runtime_owner()
    chat_service = MagicMock()
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=chat_service))

    runtime = resolve_chat_runtime(runtime_owner, execution_context)

    assert runtime is chat_service


def test_resolve_chat_service_returns_execution_context_chat_service() -> None:
    runtime_owner = _make_runtime_owner()
    chat_service = MagicMock()
    execution_context = SimpleNamespace(services=SimpleNamespace(chat=chat_service))

    resolved = resolve_chat_service(runtime_owner, execution_context)

    assert resolved is chat_service


@pytest.mark.asyncio
async def test_resolve_chat_runtime_wraps_orchestrator_fallback_without_warning() -> None:
    runtime_owner = _make_runtime_owner()
    runtime_owner.chat = AsyncMock(return_value="fallback-result")
    runtime_owner._container = None

    async def _stream_chat(_message: str):
        yield SimpleNamespace(content="chunk")

    runtime_owner.stream_chat = _stream_chat

    runtime = resolve_chat_runtime(runtime_owner)

    assert runtime is not runtime_owner

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", DeprecationWarning)
        result = await runtime.chat("hello")
        chunks = [chunk async for chunk in runtime.stream_chat("hello")]

    assert result == "fallback-result"
    assert [chunk.content for chunk in chunks] == ["chunk"]
    assert len(recorded) == 0
