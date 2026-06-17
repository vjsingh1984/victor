"""Focused tests for workflow sandbox executor async boundaries."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.workflows.isolation import IsolationConfig, ResourceLimits
from victor.workflows.sandbox_executor import SandboxedExecutor


@pytest.mark.asyncio
async def test_process_sandbox_uses_asyncio_to_thread_for_communicate() -> None:
    executor = SandboxedExecutor(docker_available=False)
    isolation = IsolationConfig(
        sandbox_type="process",
        network_allowed=False,
        resource_limits=ResourceLimits(timeout_seconds=5.0),
    )

    process = MagicMock()
    process.returncode = 0
    process.communicate = MagicMock(return_value=(b"stdout", b""))

    sandbox = MagicMock()
    sandbox.start = AsyncMock(return_value=process)
    sandbox.terminate = AsyncMock()

    async def call_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    with (
        patch("victor.integrations.mcp.sandbox.SandboxedProcess", return_value=sandbox),
        patch(
            "victor.workflows.sandbox_executor.asyncio.to_thread",
            side_effect=call_to_thread,
        ) as mock_to_thread,
    ):
        result = await executor._execute_process(
            ["python", "-c", "print('ok')"],
            isolation,
            working_dir=None,
            env={"CUSTOM_ENV": "1"},
            input_data="payload",
        )

    assert result.success is True
    assert result.output == "stdout"
    assert result.error == ""
    sandbox.start.assert_awaited_once()
    sandbox.terminate.assert_awaited_once_with(process)
    mock_to_thread.assert_awaited_once()
    called = mock_to_thread.await_args
    assert called.args[0] is process.communicate
    assert called.args[1] == b"payload"
    sandbox.start.assert_awaited_once()
    assert sandbox.start.await_args.kwargs["env"]["CUSTOM_ENV"] == "1"
    assert sandbox.start.await_args.kwargs["env"]["VICTOR_NETWORK_DISABLED"] == "1"
