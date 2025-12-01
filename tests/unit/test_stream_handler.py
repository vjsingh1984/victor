# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Tests for stream_handler module."""

import pytest
import asyncio
from typing import AsyncIterator
from victor.agent.stream_handler import (
    StreamHandler,
    StreamResult,
    StreamMetrics,
    StreamBuffer,
)
from victor.providers.base import StreamChunk


async def mock_stream(
    chunks: list[StreamChunk],
) -> AsyncIterator[StreamChunk]:
    """Create an async iterator from a list of chunks."""
    for chunk in chunks:
        yield chunk


class TestStreamHandler:
    """Tests for StreamHandler class."""

    def test_stream_handler_init(self):
        """Test StreamHandler initialization."""
        handler = StreamHandler()
        assert handler.on_content is None
        assert handler.on_tool_call is None
        assert handler.timeout == 300.0

    def test_stream_handler_with_callbacks(self):
        """Test StreamHandler with callbacks."""
        content_received = []

        def on_content(chunk):
            content_received.append(chunk)

        handler = StreamHandler(on_content=on_content)
        assert handler.on_content is not None

    def test_stream_handler_timeout(self):
        """Test StreamHandler timeout setting."""
        handler = StreamHandler(timeout=60.0)
        assert handler.timeout == 60.0


class TestStreamResult:
    """Tests for StreamResult dataclass."""

    def test_stream_result_default(self):
        """Test StreamResult with defaults."""
        result = StreamResult()
        assert result.content == ""
        assert result.tool_calls == []
        assert result.stop_reason is None
        assert result.error is None

    def test_stream_result_with_content(self):
        """Test StreamResult with content."""
        result = StreamResult(content="Hello, World!")
        assert result.content == "Hello, World!"

    def test_stream_result_with_tool_calls(self):
        """Test StreamResult with tool calls."""
        tool_calls = [{"name": "test_tool", "args": {}}]
        result = StreamResult(tool_calls=tool_calls)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "test_tool"


class TestStreamMetrics:
    """Tests for StreamMetrics dataclass."""

    def test_stream_metrics_default(self):
        """Test StreamMetrics with defaults."""
        metrics = StreamMetrics()
        assert metrics.start_time == 0.0
        assert metrics.first_token_time is None
        assert metrics.total_chunks == 0

    def test_stream_metrics_time_to_first_token(self):
        """Test time to first token calculation."""
        metrics = StreamMetrics(start_time=1.0, first_token_time=1.5)
        assert metrics.time_to_first_token == 0.5

    def test_stream_metrics_time_to_first_token_none(self):
        """Test time to first token when not set."""
        metrics = StreamMetrics(start_time=1.0)
        assert metrics.time_to_first_token is None

    def test_stream_metrics_total_duration(self):
        """Test total duration calculation."""
        metrics = StreamMetrics(start_time=1.0, end_time=5.0)
        assert metrics.total_duration == 4.0

    def test_stream_metrics_tokens_per_second(self):
        """Test tokens per second calculation."""
        metrics = StreamMetrics(start_time=1.0, end_time=2.0, total_content_length=400)
        # 400 chars / 4 = 100 tokens, 100 tokens / 1 sec = 100 tps
        assert metrics.tokens_per_second == 100.0

    def test_stream_metrics_tokens_per_second_zero_duration(self):
        """Test tokens per second with zero duration."""
        metrics = StreamMetrics(start_time=1.0, end_time=1.0)
        assert metrics.tokens_per_second == 0.0


class TestStreamHandlerProcessStream:
    """Tests for StreamHandler.process_stream method."""

    @pytest.mark.asyncio
    async def test_process_stream_content(self):
        """Test processing stream with content chunks."""
        handler = StreamHandler()

        chunks = [
            StreamChunk(content="Hello, "),
            StreamChunk(content="World!"),
            StreamChunk(is_final=True),
        ]

        result = await handler.process_stream(mock_stream(chunks))

        assert result.content == "Hello, World!"
        assert result.metrics.total_chunks == 3
        assert result.metrics.total_content_length == 13

    @pytest.mark.asyncio
    async def test_process_stream_with_callback(self):
        """Test process_stream with content callback."""
        received = []

        def on_content(chunk):
            received.append(chunk)

        handler = StreamHandler(on_content=on_content)

        chunks = [
            StreamChunk(content="Hello"),
            StreamChunk(content="World"),
            StreamChunk(is_final=True),
        ]

        await handler.process_stream(mock_stream(chunks))

        assert received == ["Hello", "World"]

    @pytest.mark.asyncio
    async def test_process_stream_with_tool_calls(self):
        """Test processing stream with tool calls."""
        tool_calls_received = []

        def on_tool_call(tc):
            tool_calls_received.append(tc)

        handler = StreamHandler(on_tool_call=on_tool_call)

        chunks = [
            StreamChunk(content="Using tool..."),
            StreamChunk(tool_calls=[{"name": "test_tool", "args": {"x": 1}}]),
            StreamChunk(is_final=True),
        ]

        result = await handler.process_stream(mock_stream(chunks))

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "test_tool"
        assert len(tool_calls_received) == 1

    @pytest.mark.asyncio
    async def test_process_stream_stop_reason(self):
        """Test process_stream captures stop reason."""
        handler = StreamHandler()

        chunks = [
            StreamChunk(content="Done"),
            StreamChunk(stop_reason="end_turn", is_final=True),
        ]

        result = await handler.process_stream(mock_stream(chunks))

        assert result.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_process_stream_on_complete_callback(self):
        """Test on_complete callback is called."""
        completed = []

        def on_complete(result):
            completed.append(result)

        handler = StreamHandler(on_complete=on_complete)

        chunks = [
            StreamChunk(content="Done", is_final=True),
        ]

        await handler.process_stream(mock_stream(chunks))

        assert len(completed) == 1
        assert completed[0].content == "Done"

    @pytest.mark.asyncio
    async def test_process_stream_callback_error_handled(self):
        """Test that callback errors are handled gracefully."""

        def bad_callback(chunk):
            raise ValueError("Callback error")

        handler = StreamHandler(on_content=bad_callback)

        chunks = [
            StreamChunk(content="Test"),
            StreamChunk(is_final=True),
        ]

        # Should not raise despite callback error
        result = await handler.process_stream(mock_stream(chunks))
        assert result.content == "Test"

    @pytest.mark.asyncio
    async def test_process_stream_metrics(self):
        """Test that metrics are collected during streaming."""
        handler = StreamHandler()

        chunks = [
            StreamChunk(content="Hello"),
            StreamChunk(content=" World"),
            StreamChunk(is_final=True),
        ]

        result = await handler.process_stream(mock_stream(chunks))

        assert result.metrics.total_chunks == 3
        assert result.metrics.total_content_length == 11
        assert result.metrics.first_token_time is not None
        assert result.metrics.end_time > result.metrics.start_time

    @pytest.mark.asyncio
    async def test_cancel_stream(self):
        """Test cancelling stream processing."""
        handler = StreamHandler()

        async def slow_stream():
            for i in range(10):
                if i == 2:
                    handler.cancel()
                yield StreamChunk(content=f"chunk{i}")
                await asyncio.sleep(0.01)

        result = await handler.process_stream(slow_stream())
        # Stream should be cancelled partway through
        assert result.metrics.total_chunks < 10

    def test_reset(self):
        """Test resetting handler state."""
        handler = StreamHandler()
        handler._cancelled = True
        handler._current_content = "test"
        handler._tool_calls = [{"test": "data"}]

        handler.reset()

        assert handler._cancelled is False
        assert handler._current_content == ""
        assert handler._tool_calls == []


class TestStreamBuffer:
    """Tests for StreamBuffer class."""

    def test_buffer_init(self):
        """Test StreamBuffer initialization."""
        buffer = StreamBuffer()
        assert buffer._buffers == {}

    def test_buffer_add_chunk_new(self):
        """Test adding chunk to new buffer."""
        buffer = StreamBuffer()

        chunk = {
            "function": {"name": "test_tool"},
            "index": 0,
        }

        result = buffer.add_chunk("call_1", chunk)
        # Should return None (not complete yet - no arguments)
        assert result is None
        assert "call_1" in buffer._buffers

    def test_buffer_add_chunk_complete(self):
        """Test adding chunk that completes the tool call."""
        buffer = StreamBuffer()

        # First chunk with name
        buffer.add_chunk("call_1", {"function": {"name": "test_tool"}})

        # Second chunk with complete arguments
        result = buffer.add_chunk("call_1", {"function": {"arguments": '{"x": 1}'}})

        # Should be complete (has name and args ending with })
        assert result is not None
        assert result["function"]["name"] == "test_tool"
        assert result["function"]["arguments"] == '{"x": 1}'

    def test_buffer_add_chunk_incremental_args(self):
        """Test adding arguments incrementally."""
        buffer = StreamBuffer()

        buffer.add_chunk("call_1", {"function": {"name": "test"}})
        buffer.add_chunk("call_1", {"function": {"arguments": '{"x":'}})
        result = buffer.add_chunk("call_1", {"function": {"arguments": " 1}"}})

        # Should be complete after full JSON
        assert result is not None
        assert result["function"]["arguments"] == '{"x": 1}'

    def test_buffer_clear(self):
        """Test clearing buffer."""
        buffer = StreamBuffer()
        buffer.add_chunk("call_1", {"function": {"name": "test"}})

        buffer.clear()

        assert buffer._buffers == {}

    def test_buffer_flush(self):
        """Test flushing incomplete buffers."""
        buffer = StreamBuffer()
        buffer.add_chunk("call_1", {"function": {"name": "test1"}})
        buffer.add_chunk("call_2", {"function": {"name": "test2"}})

        flushed = buffer.flush()

        assert len(flushed) == 2
        assert buffer._buffers == {}

    def test_buffer_flush_empty(self):
        """Test flushing empty buffer."""
        buffer = StreamBuffer()
        flushed = buffer.flush()
        assert flushed == []
