"""Focused tests for ToolRetryExecutor cache invalidation behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.services.tool_retry import ToolRetryExecutor


class _Config:
    retry_enabled = True
    max_retry_attempts = 1
    retry_base_delay = 0.0
    retry_max_delay = 0.0


def _make_executor(cache: MagicMock) -> ToolRetryExecutor:
    pipeline = MagicMock()
    pipeline._execute_single_tool = AsyncMock(
        return_value=SimpleNamespace(success=True, error=None)
    )
    cache.get.return_value = None
    return ToolRetryExecutor(config=_Config(), pipeline=pipeline, cache=cache)


@pytest.mark.asyncio
async def test_execute_tool_with_retry_invalidates_paths_for_canonical_write():
    cache = MagicMock()
    executor = _make_executor(cache)

    result, success, error = await executor.execute_tool_with_retry(
        "write",
        {"path": "/tmp/example.py"},
        {},
    )

    assert success is True
    assert error is None
    assert result is not None
    cache.invalidate_paths.assert_called_once_with(["/tmp/example.py"])
    cache.clear_namespaces.assert_not_called()


@pytest.mark.asyncio
async def test_execute_tool_with_retry_invalidates_paths_for_create_file_alias():
    cache = MagicMock()
    executor = _make_executor(cache)

    await executor.execute_tool_with_retry(
        "create_file",
        {"path": "/tmp/example.py"},
        {},
    )

    cache.invalidate_paths.assert_called_once_with(["/tmp/example.py"])


@pytest.mark.asyncio
async def test_execute_tool_with_retry_clears_canonical_namespaces_for_shell():
    cache = MagicMock()
    executor = _make_executor(cache)

    await executor.execute_tool_with_retry(
        "shell",
        {"cmd": "pytest"},
        {},
    )

    cache.clear_namespaces.assert_called_once_with(["read", "ls"])
