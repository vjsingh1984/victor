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

"""Cross-task cleanup regression tests for the streaming consumer.

The provider's httpx SSE generator must be entered, iterated, and closed in a single
asyncio task, else httpcore/anyio raises "Attempted to exit cancel scope in a different
task" / "async generator ignored GeneratorExit". A producer task now owns the generator's
whole lifecycle via ``aclosing``. These tests assert the generator is closed exactly once,
in the same task it was iterated, on every exit path (normal end, tool-call break,
cancellation, stall timeout).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from victor.agent.services.chat_stream_helpers import ChatStreamHelperMixin
from victor.agent.streaming.context import StreamingChatContext
from victor.core.errors import ProviderTimeoutError
from victor.providers.base import StreamChunk


class _Helper(ChatStreamHelperMixin):
    def __init__(self, orchestrator):
        self._orchestrator = orchestrator


def _make_orch(stream_factory, *, heartbeat=0.05, stall=5.0, grace=100.0):
    return SimpleNamespace(
        get_assembled_messages=MagicMock(return_value=[]),
        provider=SimpleNamespace(stream=stream_factory),
        model="m",
        temperature=0.7,
        max_tokens=100,
        settings=SimpleNamespace(
            stream_provider_wait_heartbeat_seconds=heartbeat,
            stream_provider_stall_timeout_seconds=stall,
            stream_provider_loop_stall_grace_seconds=grace,
            stream_idle_timeout_seconds=300.0,
        ),
        _metrics_collector=SimpleNamespace(record_first_token=MagicMock()),
        sanitizer=SimpleNamespace(is_garbage_content=lambda _c: False, sanitize=lambda c: c),
    )


def _instrumented_stream(chunks, record, *, hang_forever=False):
    """Return an async-generator factory that records the task at iterate and at close."""

    async def _stream(**_kwargs):
        try:
            for chunk in chunks:
                record["iter_task"] = id(asyncio.current_task())
                yield chunk
            if hang_forever:
                record["iter_task"] = id(asyncio.current_task())
                await asyncio.sleep(3600)
        finally:
            record["close_task"] = id(asyncio.current_task())
            record["aclose_count"] = record.get("aclose_count", 0) + 1

    return _stream


def _ctx():
    return StreamingChatContext(user_message="x", total_iterations=1)


async def test_full_consume_closes_generator_once_in_task():
    record: dict = {}
    chunks = [StreamChunk(content="hello "), StreamChunk(content="world")]
    helper = _Helper(_make_orch(_instrumented_stream(chunks, record)))

    content, tool_calls, _tokens, _garbage = await helper._stream_provider_response_inner(
        {}, {}, _ctx()
    )

    assert content == "hello world"
    assert tool_calls is None
    assert record["aclose_count"] == 1  # closed exactly once
    assert record["close_task"] == record["iter_task"]  # enter and exit share one task


async def test_tool_call_break_closes_generator_once_in_task():
    record: dict = {}
    chunks = [
        StreamChunk(content="thinking"),
        StreamChunk(content="", tool_calls=[{"name": "shell", "arguments": {}}]),
        StreamChunk(content="unreached"),
    ]
    helper = _Helper(_make_orch(_instrumented_stream(chunks, record)))

    _content, tool_calls, _tokens, _garbage = await helper._stream_provider_response_inner(
        {}, {}, _ctx()
    )

    assert tool_calls == [{"name": "shell", "arguments": {}}]
    assert record["aclose_count"] == 1
    assert record["close_task"] == record["iter_task"]


async def test_cancel_mid_stream_closes_generator_in_task():
    record: dict = {}
    chunks = [StreamChunk(content="partial")]
    helper = _Helper(_make_orch(_instrumented_stream(chunks, record, hang_forever=True)))

    task = asyncio.create_task(helper._stream_provider_response_inner({}, {}, _ctx()))
    # Let it consume the first chunk and then block on the hanging producer.
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The generator was finalized deterministically in the producer task during cancellation,
    # not left to GC (which would close it off-task and raise GeneratorExit warnings).
    assert record["aclose_count"] == 1


async def test_stall_timeout_raises_and_closes_generator():
    record: dict = {}
    # A generator that never yields: the consumer's stall timeout must fire and the producer
    # must still be finalized in-task.
    helper = _Helper(
        _make_orch(
            _instrumented_stream([], record, hang_forever=True),
            heartbeat=0.05,
            stall=0.1,
            grace=100.0,
        )
    )

    with pytest.raises(ProviderTimeoutError):
        await helper._stream_provider_response_inner({}, {}, _ctx())

    assert record["aclose_count"] == 1
