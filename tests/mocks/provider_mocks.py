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

"""Comprehensive mock provider infrastructure for testing.

This module provides production-quality mock provider classes that simulate
various provider behaviors for testing purposes. All mocks implement the
BaseProvider interface and can be used interchangeably with real providers
in unit and integration tests.

Usage:
    from tests.mocks.provider_mocks import MockBaseProvider, FailingProvider

    # Simple mock
    provider = MockBaseProvider(response_text="Hello, world!")
    response = await provider.chat(messages, model="test-model")

    # Failing provider
    failing = FailingProvider(error_type="rate_limit")
    with pytest.raises(ProviderRateLimitError):
        await failing.chat(messages, model="test-model")
"""

from asyncio import sleep
from typing import Any, AsyncIterator, Callable, Dict, List, Optional
from unittest.mock import MagicMock

from victor.core.errors import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderInvalidResponseError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    StreamChunk,
    ToolDefinition,
)


# =============================================================================
# MockBaseProvider - Configurable Base Mock
# =============================================================================


class MockBaseProvider(BaseProvider):
    """Mock provider with configurable behavior for testing.

    This provider simulates successful responses with configurable content,
    token usage, and metadata. It supports both regular and streaming modes,
    as well as tool calling.

    Args:
        response_text: Text to return in chat responses
        response_delay: Simulated network delay in seconds (default: 0)
        supports_streaming: Whether to support streaming (default: True)
        supports_tools: Whether to support tool calling (default: True)
        token_usage: Custom token usage dict (default: auto-calculated)
        tool_calls: Predefined tool calls to return (default: None)
        metadata: Additional metadata to include in responses (default: None)
        **kwargs: Additional arguments passed to BaseProvider

    Example:
        provider = MockBaseProvider(
            response_text="Test response",
            response_delay=0.1,
            supports_tools=True,
        )
        response = await provider.chat(messages, model="test")
        assert response.content == "Test response"
    """

    def __init__(
        self,
        response_text: str = "Mock response",
        response_delay: float = 0.0,
        supports_streaming: bool = True,
        supports_tools: bool = True,
        token_usage: Optional[Dict[str, int]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize mock provider with configurable behavior."""
        super().__init__(**kwargs)
        self._response_text = response_text
        self._response_delay = response_delay
        self._supports_streaming = supports_streaming
        self._supports_tools = supports_tools
        self._token_usage = token_usage
        self._tool_calls = tool_calls
        self._metadata = metadata or {}
        self._call_count = 0
        self._last_request: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        """Provider name."""
        return "mock_base"

    def supports_tools(self) -> bool:
        """Check if provider supports tool calling."""
        return self._supports_tools

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        return self._supports_streaming

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send a chat completion request.

        Args:
            messages: List of conversation messages
            model: Model identifier
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            tools: Available tools for the model to use
            **kwargs: Additional provider-specific parameters

        Returns:
            CompletionResponse with configured mock data
        """
        # Record request for testing
        self._call_count += 1
        self._last_request = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": tools,
            "kwargs": kwargs,
        }

        # Simulate network delay
        if self._response_delay > 0:
            await sleep(self._response_delay)

        # Calculate token usage if not provided
        if self._token_usage is None:
            prompt_tokens = sum(len(m.content) // 4 for m in messages)
            completion_tokens = len(self._response_text) // 4
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        else:
            usage = self._token_usage

        return CompletionResponse(
            content=self._response_text,
            role="assistant",
            tool_calls=self._tool_calls,
            stop_reason="stop",
            usage=usage,
            model=model,
            metadata=self._metadata if self._metadata else None,
        )

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion response.

        Yields the configured response text in chunks, simulating
        real streaming behavior.

        Args:
            messages: List of conversation messages
            model: Model identifier
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            tools: Available tools for the model to use
            **kwargs: Additional provider-specific parameters

        Yields:
            StreamChunk objects with incremental content
        """
        # Record request
        self._call_count += 1
        self._last_request = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": tools,
            "kwargs": kwargs,
        }

        # Split response into words for realistic streaming
        words = self._response_text.split()

        for i, word in enumerate(words):
            if self._response_delay > 0:
                await sleep(self._response_delay / len(words))

            is_final = i == len(words) - 1

            yield StreamChunk(
                content=word + " " if not is_final else word,
                is_final=is_final,
            )

        # Final chunk with metadata
        if self._token_usage:
            yield StreamChunk(
                is_final=True,
                usage=self._token_usage,
                model_name=model,
            )

    async def close(self) -> None:
        """Close any open connections or resources."""
        pass

    @property
    def call_count(self) -> int:
        """Get number of times chat/stream was called."""
        return self._call_count

    @property
    def last_request(self) -> Optional[Dict[str, Any]]:
        """Get the last request parameters."""
        return self._last_request

    def reset(self) -> None:
        """Reset call count and last request."""
        self._call_count = 0
        self._last_request = None


# =============================================================================
# FailingProvider - Simulate Various Failure Modes
# =============================================================================


class FailingProvider(BaseProvider):
    """Provider that simulates various failure scenarios.

    This provider is useful for testing error handling, retry logic,
    circuit breakers, and resilience patterns.

    Args:
        error_type: Type of error to simulate:
            - "timeout": ProviderTimeoutError
            - "rate_limit": ProviderRateLimitError
            - "auth": ProviderAuthError
            - "connection": ProviderConnectionError
            - "invalid_response": ProviderInvalidResponseError
            - "generic": ProviderError
        fail_after: Number of successful calls before failing (default: 0)
        error_message: Custom error message (default: auto-generated)
        **kwargs: Additional arguments passed to BaseProvider

    Example:
        # Fail immediately with rate limit error
        provider = FailingProvider(error_type="rate_limit")

        # Fail after 3 successful calls
        provider = FailingProvider(error_type="timeout", fail_after=3)

        # Test retry logic
        for i in range(5):
            try:
                await provider.chat(messages, model="test")
            except ProviderRateLimitError:
                print(f"Failed on attempt {i+1}")
    """

    def __init__(
        self,
        error_type: str = "generic",
        fail_after: int = 0,
        error_message: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize failing provider."""
        super().__init__(**kwargs)
        self._error_type = error_type
        self._fail_after = fail_after
        self._error_message = error_message
        self._call_count = 0

    @property
    def name(self) -> str:
        """Provider name."""
        return "failing_provider"

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send a chat completion request that may fail."""
        self._call_count += 1

        # Check if we should fail
        if self._call_count > self._fail_after:
            error = self._create_error()
            raise error

        # Return success response
        return CompletionResponse(
            content=f"Success call {self._call_count}",
            role="assistant",
            model=model,
        )

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion response that may fail."""
        self._call_count += 1

        if self._call_count > self._fail_after:
            error = self._create_error()
            raise error

        yield StreamChunk(content=f"Success stream {self._call_count}")

    async def close(self) -> None:
        """Close any open connections or resources."""
        pass

    def _create_error(self) -> ProviderError:
        """Create the appropriate error based on error_type."""
        message = self._error_message or f"Simulated {self._error_type} error"

        error_map = {
            "timeout": lambda: ProviderTimeoutError(
                message, timeout=self.timeout, provider=self.name
            ),
            "rate_limit": lambda: ProviderRateLimitError(
                message, retry_after=60, provider=self.name
            ),
            "auth": lambda: ProviderAuthError(message, provider=self.name),
            "connection": lambda: ProviderConnectionError(
                message, provider=self.name
            ),
            "invalid_response": lambda: ProviderInvalidResponseError(
                message, provider=self.name
            ),
            "generic": lambda: ProviderError(message, provider=self.name),
        }

        error_factory = error_map.get(self._error_type, error_map["generic"])
        return error_factory()

    @property
    def call_count(self) -> int:
        """Get number of calls made."""
        return self._call_count


# =============================================================================
# StreamingTestProvider - Specialized for Streaming Tests
# =============================================================================


class StreamingTestProvider(BaseProvider):
    """Provider optimized for testing streaming functionality.

    This provider offers fine-grained control over streaming behavior,
    including chunk size, delays, and intermediate tool calls.

    Args:
        chunks: List of content chunks to yield in order
        chunk_delay: Delay between chunks in seconds (default: 0)
        include_tool_calls: Whether to include tool calls in stream (default: False)
        **kwargs: Additional arguments passed to BaseProvider

    Example:
        provider = StreamingTestProvider(
            chunks=["Hello", " world", "!"],
            chunk_delay=0.05,
        )

        chunks = []
        async for chunk in provider.stream(messages, model="test"):
            chunks.append(chunk.content)

        assert chunks == ["Hello", " world", "!"]
    """

    def __init__(
        self,
        chunks: Optional[List[str]] = None,
        chunk_delay: float = 0.0,
        include_tool_calls: bool = False,
        **kwargs: Any,
    ):
        """Initialize streaming test provider."""
        super().__init__(**kwargs)
        self._chunks = chunks or ["chunk1", "chunk2", "chunk3"]
        self._chunk_delay = chunk_delay
        self._include_tool_calls = include_tool_calls
        self._streamed_chunks: List[StreamChunk] = []

    @property
    def name(self) -> str:
        """Provider name."""
        return "streaming_test"

    def supports_streaming(self) -> bool:
        """Provider supports streaming."""
        return True

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send a chat completion request (non-streaming)."""
        return CompletionResponse(
            content="".join(self._chunks),
            role="assistant",
            model=model,
        )

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion response with controlled chunks."""
        self._streamed_chunks.clear()

        for i, chunk_content in enumerate(self._chunks):
            if self._chunk_delay > 0:
                await sleep(self._chunk_delay)

            is_final = i == len(self._chunks) - 1

            chunk = StreamChunk(
                content=chunk_content,
                is_final=is_final,
            )

            self._streamed_chunks.append(chunk)
            yield chunk

        # Final metadata chunk
        yield StreamChunk(
            is_final=True,
            model_name=model,
        )

    async def close(self) -> None:
        """Close any open connections or resources."""
        pass

    @property
    def streamed_chunks(self) -> List[StreamChunk]:
        """Get list of chunks that were streamed."""
        return self._streamed_chunks.copy()


# =============================================================================
# ToolCallMockProvider - Predefined Tool Call Responses
# =============================================================================


class ToolCallMockProvider(BaseProvider):
    """Provider that returns predefined tool calls in responses.

    This is useful for testing tool calling logic, tool execution,
    and multi-turn conversations with tools.

    Args:
        response_text: Text content to include with tool calls
        tool_calls: List of tool call dicts to return
        call_sequence: Sequence of responses for multi-turn testing (default: None)
        **kwargs: Additional arguments passed to BaseProvider

    Example:
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "test"}'
                }
            }
        ]

        provider = ToolCallMockProvider(
            response_text="I'll search for that.",
            tool_calls=tool_calls,
        )

        response = await provider.chat(messages, model="test")
        assert response.tool_calls == tool_calls
    """

    def __init__(
        self,
        response_text: str = "Executing tools...",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        call_sequence: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        """Initialize tool call mock provider."""
        super().__init__(**kwargs)
        self._response_text = response_text
        self._default_tool_calls = tool_calls or []
        self._call_sequence = call_sequence or []
        self._sequence_index = 0
        self._call_history: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        """Provider name."""
        return "tool_call_mock"

    def supports_tools(self) -> bool:
        """Provider supports tool calling."""
        return True

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send a chat completion request with tool calls."""
        # Record call
        call_info = {
            "messages": messages,
            "model": model,
            "tools": tools,
        }
        self._call_history.append(call_info)

        # Use sequence if available, otherwise default
        if self._sequence_index < len(self._call_sequence):
            response_data = self._call_sequence[self._sequence_index]
            self._sequence_index += 1

            content = response_data.get("content", self._response_text)
            tool_calls = response_data.get("tool_calls", self._default_tool_calls)
        else:
            content = self._response_text
            tool_calls = self._default_tool_calls

        return CompletionResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls if tool_calls else None,
            model=model,
        )

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion response (not implemented for tool calls)."""
        raise NotImplementedError(
            "Streaming not implemented for ToolCallMockProvider"
        )

    async def close(self) -> None:
        """Close any open connections or resources."""
        pass

    @property
    def call_history(self) -> List[Dict[str, Any]]:
        """Get history of all calls made."""
        return self._call_history.copy()

    def reset_sequence(self) -> None:
        """Reset call sequence index to start."""
        self._sequence_index = 0


# =============================================================================
# ProviderTestHelpers - Utility Functions
# =============================================================================


class ProviderTestHelpers:
    """Helper functions for creating and configuring test providers."""

    @staticmethod
    def create_simple_mock(response_text: str = "Test response") -> MockBaseProvider:
        """Create a simple mock provider with default settings.

        Args:
            response_text: Text to return in responses

        Returns:
            Configured MockBaseProvider instance
        """
        return MockBaseProvider(response_text=response_text)

    @staticmethod
    def create_streaming_mock(
        chunks: Optional[List[str]] = None, chunk_delay: float = 0.0
    ) -> StreamingTestProvider:
        """Create a streaming mock provider.

        Args:
            chunks: List of chunks to stream
            chunk_delay: Delay between chunks

        Returns:
            Configured StreamingTestProvider instance
        """
        return StreamingTestProvider(chunks=chunks, chunk_delay=chunk_delay)

    @staticmethod
    def create_failing_mock(
        error_type: str = "generic", fail_after: int = 0
    ) -> FailingProvider:
        """Create a failing mock provider.

        Args:
            error_type: Type of error to simulate
            fail_after: Number of successful calls before failing

        Returns:
            Configured FailingProvider instance
        """
        return FailingProvider(error_type=error_type, fail_after=fail_after)

    @staticmethod
    def create_tool_call_mock(
        tool_calls: List[Dict[str, Any]], response_text: str = "Executing tools..."
    ) -> ToolCallMockProvider:
        """Create a tool call mock provider.

        Args:
            tool_calls: List of tool call dicts to return
            response_text: Text content to include

        Returns:
            Configured ToolCallMockProvider instance
        """
        return ToolCallMockProvider(
            tool_calls=tool_calls, response_text=response_text
        )

    @staticmethod
    def create_test_messages(content: str = "Test message") -> List[Message]:
        """Create a list of test messages.

        Args:
            content: Content for the user message

        Returns:
            List with system and user messages
        """
        return [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content=content),
        ]

    @staticmethod
    def create_test_tool_call(
        name: str, arguments: Dict[str, Any], call_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a properly formatted tool call dict.

        Args:
            name: Tool/function name
            arguments: Tool arguments as dict (will be converted to JSON)
            call_id: Optional call ID (auto-generated if not provided)

        Returns:
            Tool call dict in standard format
        """
        import json
        import uuid

        return {
            "id": call_id or f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments),
            },
        }

    @staticmethod
    async def collect_stream_chunks(
        provider: BaseProvider, messages: List[Message], model: str = "test-model"
    ) -> List[StreamChunk]:
        """Collect all chunks from a streaming response.

        Args:
            provider: Provider to stream from
            messages: Messages to send
            model: Model identifier

        Returns:
            List of all StreamChunk objects
        """
        chunks = []
        async for chunk in provider.stream(messages, model=model):
            chunks.append(chunk)
        return chunks

    @staticmethod
    def assert_valid_response(response: CompletionResponse) -> None:
        """Assert that a response is valid and well-formed.

        Args:
            response: Response to validate

        Raises:
            AssertionError: If response is invalid
        """
        assert isinstance(response.content, str)
        assert len(response.content) > 0
        assert response.role == "assistant"
        assert response.usage is not None
        assert "total_tokens" in response.usage

    @staticmethod
    def assert_valid_stream_chunks(chunks: List[StreamChunk]) -> None:
        """Assert that streamed chunks are valid.

        Args:
            chunks: List of chunks to validate

        Raises:
            AssertionError: If chunks are invalid
        """
        assert len(chunks) > 0

        # All chunks should be StreamChunk instances
        for chunk in chunks:
            assert isinstance(chunk, StreamChunk)

        # Last chunk should be marked as final
        assert chunks[-1].is_final


# =============================================================================
# LatencySimulationProvider - Simulate Network Latency and Jitter
# =============================================================================


class LatencySimulationProvider(BaseProvider):
    """Provider that simulates various network latency patterns.

    This provider is useful for testing timeout handling, retry logic,
    performance monitoring, and latency-sensitive operations.

    Args:
        base_latency: Base network delay in seconds (default: 0.1)
        jitter: Random jitter amount in seconds (default: 0.02)
        timeout_after: Latency threshold that triggers timeout (default: None)
        latency_pattern: Latency pattern to use:
            - "constant": Always use base_latency
            - "increasing": Latency increases with each call
            - "degrading": Latency increases over time within call
            - "random": Random latency between base_latency +/- jitter
        response_text: Text to return in responses (default: "Simulated response")
        **kwargs: Additional arguments passed to BaseProvider

    Example:
        # Constant latency of 200ms
        provider = LatencySimulationProvider(base_latency=0.2)

        # Latency that increases with each call (100ms, 150ms, 200ms, ...)
        provider = LatencySimulationProvider(
            base_latency=0.1,
            latency_pattern="increasing"
        )

        # Random latency between 80ms and 120ms
        provider = LatencySimulationProvider(
            base_latency=0.1,
            jitter=0.02,
            latency_pattern="random"
        )

        # Simulate timeout (>5 seconds)
        provider = LatencySimulationProvider(base_latency=6.0)
    """

    def __init__(
        self,
        base_latency: float = 0.1,
        jitter: float = 0.02,
        timeout_after: Optional[float] = None,
        latency_pattern: str = "constant",
        response_text: str = "Simulated response",
        **kwargs: Any,
    ):
        """Initialize latency simulation provider."""
        super().__init__(**kwargs)
        self._base_latency = base_latency
        self._jitter = jitter
        self._timeout_after = timeout_after
        self._latency_pattern = latency_pattern
        self._response_text = response_text
        self._call_count = 0
        self._latency_history: List[float] = []

    @property
    def name(self) -> str:
        """Provider name."""
        return "latency_simulation"

    def _calculate_latency(self) -> float:
        """Calculate latency for current call based on pattern."""
        import random

        self._call_count += 1

        if self._latency_pattern == "constant":
            latency = self._base_latency

        elif self._latency_pattern == "increasing":
            # Increase by 10% each call
            increase_factor = 1.0 + (0.1 * (self._call_count - 1))
            latency = self._base_latency * increase_factor

        elif self._latency_pattern == "degrading":
            # Start fast, get slower (simulate performance degradation)
            degradation_factor = 1.0 + (0.05 * (self._call_count - 1))
            latency = self._base_latency * degradation_factor

        elif self._latency_pattern == "random":
            # Random latency within [base_latency - jitter, base_latency + jitter]
            latency = self._base_latency + random.uniform(-self._jitter, self._jitter)
            latency = max(0, latency)  # Ensure non-negative

        else:
            # Default to constant
            latency = self._base_latency

        return latency

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Send a chat completion request with simulated latency."""
        latency = self._calculate_latency()
        self._latency_history.append(latency)

        # Check if this should timeout
        if self._timeout_after and latency > self._timeout_after:
            from victor.core.errors import ProviderTimeoutError

            raise ProviderTimeoutError(
                f"Simulated timeout after {latency:.2f}s",
                timeout=int(latency),
                provider=self.name,
            )

        # Simulate network delay
        await sleep(latency)

        return CompletionResponse(
            content=self._response_text,
            role="assistant",
            model=model,
        )

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion response with simulated latency."""
        words = self._response_text.split()
        latency_per_chunk = self._calculate_latency() / len(words)

        for i, word in enumerate(words):
            await sleep(latency_per_chunk)

            is_final = i == len(words) - 1
            yield StreamChunk(
                content=word + " " if not is_final else word,
                is_final=is_final,
            )

        # Final metadata chunk
        yield StreamChunk(is_final=True, model_name=model)

    async def close(self) -> None:
        """Close any open connections or resources."""
        pass

    @property
    def latency_history(self) -> List[float]:
        """Get history of latencies for all calls."""
        return self._latency_history.copy()

    @property
    def average_latency(self) -> float:
        """Get average latency across all calls."""
        if not self._latency_history:
            return 0.0
        return sum(self._latency_history) / len(self._latency_history)

    @property
    def max_latency(self) -> float:
        """Get maximum latency observed."""
        if not self._latency_history:
            return 0.0
        return max(self._latency_history)

    def reset(self) -> None:
        """Reset latency history and call count."""
        self._call_count = 0
        self._latency_history.clear()


# =============================================================================
# Convenience Exports
# =============================================================================


__all__ = [
    "MockBaseProvider",
    "FailingProvider",
    "StreamingTestProvider",
    "ToolCallMockProvider",
    "LatencySimulationProvider",
    "ProviderTestHelpers",
]
