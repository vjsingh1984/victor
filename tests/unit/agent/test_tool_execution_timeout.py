"""Tests for per-tool execution timeout via asyncio.wait_for."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_tool(execute_coro, *, timeout_seconds=None, name="mock_tool"):
    """Create a mock tool with an async execute and optional timeout_seconds."""
    tool = MagicMock()
    tool.name = name
    tool.execute = AsyncMock(side_effect=execute_coro)
    if timeout_seconds is not None:
        tool.timeout_seconds = timeout_seconds
    else:
        # Ensure the attribute is absent so getattr falls back to default
        if hasattr(tool, "timeout_seconds"):
            del tool.timeout_seconds
    return tool


async def test_hanging_tool_raises_timeout_error():
    """A tool that sleeps forever should be cancelled by wait_for."""

    async def hang(**kwargs):
        await asyncio.sleep(999)

    tool = _make_tool(hang, timeout_seconds=0.1)

    per_attempt_timeout = getattr(tool, "timeout_seconds", 30.0)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            tool.execute(),
            timeout=per_attempt_timeout,
        )


async def test_fast_tool_succeeds():
    """A tool that completes quickly should return its result normally."""

    async def fast(**kwargs):
        await asyncio.sleep(0.01)
        return "done"

    tool = _make_tool(fast, timeout_seconds=30.0)

    per_attempt_timeout = getattr(tool, "timeout_seconds", 30.0)
    result = await asyncio.wait_for(
        tool.execute(),
        timeout=per_attempt_timeout,
    )
    assert result == "done"


async def test_default_timeout_when_attr_missing():
    """When a tool has no timeout_seconds attribute, default to 30.0."""

    tool = MagicMock(spec=[])
    tool.name = "no_timeout_tool"
    tool.execute = AsyncMock(return_value="ok")

    per_attempt_timeout = getattr(tool, "timeout_seconds", 30.0)
    assert per_attempt_timeout == 30.0

    result = await asyncio.wait_for(
        tool.execute(),
        timeout=per_attempt_timeout,
    )
    assert result == "ok"
