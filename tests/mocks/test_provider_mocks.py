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

"""Tests for provider mock infrastructure.

These tests validate that the mock providers work correctly and
can be used in other test files.
"""

import pytest

from tests.mocks.provider_mocks import (
    FailingProvider,
    LatencySimulationProvider,
    MockBaseProvider,
    ProviderTestHelpers,
    StreamingTestProvider,
    ToolCallMockProvider,
)
from victor.core.errors import ProviderRateLimitError, ProviderTimeoutError
from victor.providers.base import Message


class TestMockBaseProvider:
    """Tests for MockBaseProvider."""

    @pytest.mark.asyncio
    async def test_basic_chat(self):
        """Test basic chat functionality."""
        provider = MockBaseProvider(response_text="Hello, world!")
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Say hello"),
        ]

        response = await provider.chat(messages, model="test-model")

        assert response.content == "Hello, world!"
        assert response.role == "assistant"
        assert response.model == "test-model"
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_streaming(self):
        """Test streaming functionality."""
        provider = MockBaseProvider(response_text="Hello world", supports_streaming=True)
        messages = [Message(role="user", content="Test")]

        chunks = []
        async for chunk in provider.stream(messages, model="test"):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert chunks[-1].is_final
        full_content = "".join(c.content for c in chunks)
        assert "Hello world" in full_content

    @pytest.mark.asyncio
    async def test_call_tracking(self):
        """Test that provider tracks calls."""
        provider = MockBaseProvider()
        messages = [Message(role="user", content="Test")]

        await provider.chat(messages, model="test")
        await provider.chat(messages, model="test")

        assert provider.call_count == 2
        assert provider.last_request is not None
        assert provider.last_request["model"] == "test"

    @pytest.mark.asyncio
    async def test_reset(self):
        """Test reset functionality."""
        provider = MockBaseProvider()
        messages = [Message(role="user", content="Test")]

        await provider.chat(messages, model="test")
        provider.reset()

        assert provider.call_count == 0
        assert provider.last_request is None


class TestFailingProvider:
    """Tests for FailingProvider."""

    @pytest.mark.asyncio
    async def test_immediate_failure(self):
        """Test provider that fails immediately."""
        provider = FailingProvider(error_type="timeout")
        messages = [Message(role="user", content="Test")]

        with pytest.raises(ProviderTimeoutError):
            await provider.chat(messages, model="test")

    @pytest.mark.asyncio
    async def test_fail_after_successes(self):
        """Test provider that succeeds then fails."""
        provider = FailingProvider(error_type="rate_limit", fail_after=2)
        messages = [Message(role="user", content="Test")]

        # First two calls succeed
        response1 = await provider.chat(messages, model="test")
        assert "Success" in response1.content

        response2 = await provider.chat(messages, model="test")
        assert "Success" in response2.content

        # Third call fails
        with pytest.raises(ProviderRateLimitError):
            await provider.chat(messages, model="test")

        assert provider.call_count == 3

    @pytest.mark.asyncio
    async def test_different_error_types(self):
        """Test different error types."""
        error_types = ["timeout", "rate_limit", "auth", "connection", "generic"]

        for error_type in error_types:
            provider = FailingProvider(error_type=error_type)
            messages = [Message(role="user", content="Test")]

            with pytest.raises(Exception):  # Different errors raised
                await provider.chat(messages, model="test")


class TestStreamingTestProvider:
    """Tests for StreamingTestProvider."""

    @pytest.mark.asyncio
    async def test_controlled_chunks(self):
        """Test streaming with controlled chunks."""
        chunks_list = ["Hello", " ", "world", "!"]
        provider = StreamingTestProvider(chunks=chunks_list)
        messages = [Message(role="user", content="Test")]

        streamed = []
        async for chunk in provider.stream(messages, model="test"):
            streamed.append(chunk)

        assert len(streamed) == len(chunks_list) + 1  # +1 for final metadata
        assert streamed[0].content == "Hello"
        assert streamed[-1].is_final

    @pytest.mark.asyncio
    async def test_chunk_tracking(self):
        """Test that provider tracks streamed chunks."""
        provider = StreamingTestProvider(chunks=["chunk1", "chunk2"])
        messages = [Message(role="user", content="Test")]

        async for _ in provider.stream(messages, model="test"):
            pass

        streamed_chunks = provider.streamed_chunks
        assert len(streamed_chunks) == 2
        assert streamed_chunks[0].content == "chunk1"


class TestToolCallMockProvider:
    """Tests for ToolCallMockProvider."""

    @pytest.mark.asyncio
    async def test_tool_calls_in_response(self):
        """Test that tool calls are included in response."""
        tool_calls = ProviderTestHelpers.create_test_tool_call(
            name="search", arguments={"query": "test"}
        )

        provider = ToolCallMockProvider(response_text="Searching...", tool_calls=[tool_calls])
        messages = [Message(role="user", content="Search for test")]

        response = await provider.chat(messages, model="test")

        assert response.content == "Searching..."
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_call_sequence(self):
        """Test multi-turn call sequence."""
        sequence = [
            {"content": "First call", "tool_calls": []},
            {"content": "Second call", "tool_calls": []},
            {"content": "Third call", "tool_calls": []},
        ]

        provider = ToolCallMockProvider(call_sequence=sequence)
        messages = [Message(role="user", content="Test")]

        # First call uses sequence
        response1 = await provider.chat(messages, model="test")
        assert response1.content == "First call"

        response2 = await provider.chat(messages, model="test")
        assert response2.content == "Second call"

        response3 = await provider.chat(messages, model="test")
        assert response3.content == "Third call"

        # Fourth call uses default
        response4 = await provider.chat(messages, model="test")
        assert "Executing" in response4.content


class TestProviderTestHelpers:
    """Tests for ProviderTestHelpers utility functions."""

    def test_create_simple_mock(self):
        """Test creating simple mock provider."""
        provider = ProviderTestHelpers.create_simple_mock("Test response")
        assert isinstance(provider, MockBaseProvider)

    def test_create_failing_mock(self):
        """Test creating failing mock provider."""
        provider = ProviderTestHelpers.create_failing_mock("timeout", fail_after=3)
        assert isinstance(provider, FailingProvider)

    def test_create_test_messages(self):
        """Test creating test messages."""
        messages = ProviderTestHelpers.create_test_messages("Hello")
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[1].content == "Hello"

    def test_create_test_tool_call(self):
        """Test creating tool call dict."""
        tool_call = ProviderTestHelpers.create_test_tool_call(
            name="test_tool", arguments={"param": "value"}
        )

        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "test_tool"
        assert "param" in tool_call["function"]["arguments"]
        assert "id" in tool_call

    @pytest.mark.asyncio
    async def test_collect_stream_chunks(self):
        """Test collecting stream chunks."""
        provider = ProviderTestHelpers.create_streaming_mock(
            chunks=["a", "b", "c"], chunk_delay=0.01
        )
        messages = ProviderTestHelpers.create_test_messages()

        chunks = await ProviderTestHelpers.collect_stream_chunks(provider, messages, model="test")

        assert len(chunks) > 0
        assert isinstance(chunks[0].content, str)

    def test_assert_valid_response(self):
        """Test response validation helper."""
        from victor.providers.base import CompletionResponse

        response = CompletionResponse(
            content="Test",
            role="assistant",
            usage={"total_tokens": 10},
        )

        # Should not raise
        ProviderTestHelpers.assert_valid_response(response)

    def test_assert_valid_stream_chunks(self):
        """Test stream chunks validation helper."""
        from victor.providers.base import StreamChunk

        chunks = [
            StreamChunk(content="a"),
            StreamChunk(content="b", is_final=True),
        ]

        # Should not raise
        ProviderTestHelpers.assert_valid_stream_chunks(chunks)


class TestLatencySimulationProvider:
    """Tests for LatencySimulationProvider."""

    @pytest.mark.asyncio
    async def test_constant_latency(self):
        """Test provider with constant latency."""
        provider = LatencySimulationProvider(base_latency=0.1)
        messages = [Message(role="user", content="Test")]

        import time

        start = time.time()
        await provider.chat(messages, model="test")
        elapsed = time.time() - start

        # Should take approximately 0.1 seconds
        assert 0.08 < elapsed < 0.15

    @pytest.mark.asyncio
    async def test_latency_tracking(self):
        """Test that provider tracks latency history."""
        provider = LatencySimulationProvider(base_latency=0.05)
        messages = [Message(role="user", content="Test")]

        await provider.chat(messages, model="test")
        await provider.chat(messages, model="test")
        await provider.chat(messages, model="test")

        assert len(provider.latency_history) == 3
        assert provider.average_latency > 0
        assert provider.max_latency > 0

    @pytest.mark.asyncio
    async def test_increasing_latency_pattern(self):
        """Test increasing latency pattern."""
        provider = LatencySimulationProvider(base_latency=0.05, latency_pattern="increasing")
        messages = [Message(role="user", content="Test")]

        await provider.chat(messages, model="test")
        await provider.chat(messages, model="test")
        await provider.chat(messages, model="test")

        # Latency should increase with each call
        latencies = provider.latency_history
        assert latencies[0] < latencies[1] < latencies[2]

    @pytest.mark.asyncio
    async def test_random_latency_pattern(self):
        """Test random latency pattern with jitter."""
        provider = LatencySimulationProvider(
            base_latency=0.1, jitter=0.03, latency_pattern="random"
        )
        messages = [Message(role="user", content="Test")]

        for _ in range(5):
            await provider.chat(messages, model="test")

        latencies = provider.latency_history
        # All latencies should be within [0.07, 0.13]
        for latency in latencies:
            assert 0.06 < latency < 0.14

    @pytest.mark.asyncio
    async def test_timeout_simulation(self):
        """Test timeout simulation."""
        provider = LatencySimulationProvider(base_latency=0.2, timeout_after=0.15)
        messages = [Message(role="user", content="Test")]

        with pytest.raises(ProviderTimeoutError):
            await provider.chat(messages, model="test")

    @pytest.mark.asyncio
    async def test_streaming_with_latency(self):
        """Test streaming with latency simulation."""
        provider = LatencySimulationProvider(base_latency=0.1, response_text="Hello world")
        messages = [Message(role="user", content="Test")]

        import time

        start = time.time()
        chunks = []
        async for chunk in provider.stream(messages, model="test"):
            chunks.append(chunk)
        elapsed = time.time() - start

        # Should take approximately base_latency time
        assert len(chunks) > 0
        assert elapsed >= 0.08  # Account for timing variance

    @pytest.mark.asyncio
    async def test_latency_reset(self):
        """Test resetting latency history."""
        provider = LatencySimulationProvider(base_latency=0.05)
        messages = [Message(role="user", content="Test")]

        await provider.chat(messages, model="test")
        await provider.chat(messages, model="test")

        assert len(provider.latency_history) == 2

        provider.reset()

        assert len(provider.latency_history) == 0
        assert provider.average_latency == 0.0

    @pytest.mark.asyncio
    async def test_average_latency_calculation(self):
        """Test average latency calculation."""
        provider = LatencySimulationProvider(base_latency=0.1)
        messages = [Message(role="user", content="Test")]

        await provider.chat(messages, model="test")
        await provider.chat(messages, model="test")
        await provider.chat(messages, model="test")

        avg = provider.average_latency
        assert avg == pytest.approx(0.1)  # All calls have 0.1s latency

    @pytest.mark.asyncio
    async def test_degrading_latency_pattern(self):
        """Test degrading latency pattern."""
        provider = LatencySimulationProvider(base_latency=0.05, latency_pattern="degrading")
        messages = [Message(role="user", content="Test")]

        await provider.chat(messages, model="test")
        await provider.chat(messages, model="test")
        await provider.chat(messages, model="test")

        # Latency should increase (degrading performance)
        latencies = provider.latency_history
        assert latencies[2] > latencies[0]
