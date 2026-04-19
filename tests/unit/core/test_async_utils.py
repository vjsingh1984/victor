# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for async/sync boundary helpers."""

import asyncio

import pytest

from victor.core.async_utils import run_blocking, run_sync, run_sync_in_thread


async def _sample_value() -> str:
    return "ok"


def test_run_sync_executes_coroutine_from_sync_context():
    """run_sync should execute coroutines when no event loop is active."""
    assert run_sync(_sample_value()) == "ok"


@pytest.mark.asyncio
async def test_run_sync_raises_from_async_context():
    """run_sync must reject nested event-loop usage."""
    with pytest.raises(
        RuntimeError, match="Cannot use run_sync\\(\\) from within an async context"
    ):
        run_sync(_sample_value())


def test_run_sync_in_thread_executes_coroutine():
    """run_sync_in_thread should execute coroutines from sync contexts."""
    assert run_sync_in_thread(_sample_value()) == "ok"


def test_run_sync_in_thread_supports_timeouts():
    """run_sync_in_thread should fail fast when the worker times out."""

    async def _slow_value() -> str:
        await asyncio.sleep(0.1)
        return "slow"

    with pytest.raises(TimeoutError, match="Async operation timed out"):
        run_sync_in_thread(_slow_value(), timeout=0.01)


@pytest.mark.asyncio
async def test_run_blocking_offloads_sync_function():
    """run_blocking should run sync functions without blocking the event loop."""
    import time

    def blocking_work():
        time.sleep(0.01)
        return "done"

    result = await run_blocking(blocking_work)
    assert result == "done"


@pytest.mark.asyncio
async def test_run_blocking_passes_args_and_kwargs():
    """run_blocking should forward positional and keyword arguments."""

    def add(a, b, offset=0):
        return a + b + offset

    result = await run_blocking(add, 3, 4, offset=10)
    assert result == 17


@pytest.mark.asyncio
async def test_run_blocking_propagates_exceptions():
    """run_blocking should propagate exceptions from the sync function."""

    def fail():
        raise ValueError("sync error")

    with pytest.raises(ValueError, match="sync error"):
        await run_blocking(fail)
