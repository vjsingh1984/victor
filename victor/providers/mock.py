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

"""Mock provider for testing purposes.

This provider implements the BaseProvider interface for testing without
requiring actual API calls. It's designed to be used in unit tests and
integration tests where you need predictable, controllable responses.
"""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

from victor.providers.base import BaseProvider, CompletionResponse, Message, StreamChunk


class MockProvider(BaseProvider):
    """Mock LLM provider for testing.

    This provider provides configurable responses without making actual API calls.
    It's ideal for unit tests and integration tests where you need predictable
    behavior and want to avoid external dependencies.

    Features:
    - Configurable responses (single or sequence)
    - Simulated streaming support
    - Tool calling simulation
    - Error simulation
    - Latency simulation
    - Request/response tracking

    Example:
        ```python
        provider = MockProvider(
            model="test-model",
            responses=["Hello", "World"],
            simulate_latency=0.1
        )

        # First call returns "Hello", second returns "World"
        response1 = await provider.chat([Message(role="user", content="Hi")])
        response2 = await provider.chat([Message(role="user", content="Hi")])

        # Configure to raise an error
        provider.configure_error(ConnectionError("API unavailable"))
        response3 = await provider.chat(...)  # Raises ConnectionError
        ```

    Args:
        model: Model name to report
        responses: List of responses to cycle through (default: ["Mock response"])
        simulate_latency: Simulated API latency in seconds (default: 0)
        supports_tools: Whether to report tool support (default: True)
        supports_streaming: Whether to report streaming support (default: True)
        default_tool_calls: Default tool calls to include in responses
    """

    def __init__(
        self,
        model: str = "mock-model",
        responses: Optional[List[str]] = None,
        simulate_latency: float = 0.0,
        supports_tools: bool = True,
        supports_streaming: bool = True,
        default_tool_calls: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the mock provider.

        Args:
            model: Model identifier
            responses: List of responses to cycle through
            simulate_latency: Artificial delay in seconds
            supports_tools: Report tool calling capability
            supports_streaming: Report streaming capability
            default_tool_calls: Default tool calls for responses
            **kwargs: Additional arguments passed to BaseProvider
        """
        super().__init__(**kwargs)

        self._model = model
        self._responses = responses or ["Mock response"]
        self._response_index = 0
        self._simulate_latency = simulate_latency
        self._supports_tools = supports_tools
        self._supports_streaming = supports_streaming
        self._default_tool_calls = default_tool_calls

        # Request tracking
        self._call_count = 0
        self._request_history: List[Dict[str, Any]] = []

        # Error simulation
        self._error_to_raise: Optional[Exception] = None
        self._error_after_calls: int = -1  # -1 means never

    @property
    def name(self) -> str:
        """Provider name."""
        return "mock"

    @property
    def model(self) -> str:
        """Current model name."""
        return self._model

    def set_model(self, model: str) -> None:
        """Set the model name."""
        self._model = model

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Generate a mock chat completion.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature (ignored in mock)
            max_tokens: Maximum tokens (ignored in mock)
            tools: Available tools (ignored in mock)
            **kwargs: Additional arguments (ignored in mock)

        Returns:
            CompletionResponse with mock content

        Raises:
            Exception: If error simulation is configured
        """
        # Track request
        self._call_count += 1
        self._request_history.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "tools": tools,
                "call_number": self._call_count,
            }
        )

        # Check for error simulation
        self._check_for_error()

        # Simulate latency if configured
        if self._simulate_latency > 0:
            await asyncio.sleep(self._simulate_latency)

        # Get response from cycle
        content = self._responses[self._response_index % len(self._responses)]
        self._response_index += 1

        return CompletionResponse(
            content=content,
            role="assistant",
            model=self._model,
            stop_reason="stop",
            tool_calls=self._default_tool_calls,
            usage={"prompt_tokens": 10, "completion_tokens": len(content.split())},
            raw_response=None,
            metadata={"mock": True, "call_number": self._call_count},
        )

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream mock chat completion.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature (ignored)
            max_tokens: Maximum tokens (ignored)
            tools: Available tools (ignored)
            **kwargs: Additional arguments (ignored)

        Yields:
            StreamChunk objects with incremental content

        Raises:
            Exception: If error simulation is configured
        """
        # Track request
        self._call_count += 1
        self._check_for_error()

        # Get response
        content = self._responses[self._response_index % len(self._responses)]
        self._response_index += 1

        # Split content into words and stream them
        words = content.split()

        # Simulate latency before streaming
        if self._simulate_latency > 0:
            await asyncio.sleep(self._simulate_latency)

        for i, word in enumerate(words):
            is_final = (i == len(words) - 1)
            yield StreamChunk(
                content=word + " ",
                is_final=is_final,
                stop_reason="stop" if is_final else None,
                usage=None if not is_final else {"prompt_tokens": 10, "completion_tokens": len(words)},
            )

            # Small delay between chunks
            if self._simulate_latency > 0:
                await asyncio.sleep(self._simulate_latency / len(words))

    async def close(self) -> None:
        """Close any open connections or resources.

        This is a no-op for the mock provider, but required by BaseProvider.
        """
        pass

    def supports_tools(self) -> bool:
        """Check if provider supports tool calling."""
        return self._supports_tools

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        return self._supports_streaming

    # ============ Configuration Methods ============

    def set_responses(self, responses: List[str]) -> None:
        """Set the responses to cycle through.

        Args:
            responses: List of response strings
        """
        self._responses = responses
        self._response_index = 0

    def add_response(self, response: str) -> None:
        """Add a response to the cycle.

        Args:
            response: Response string to add
        """
        self._responses.append(response)

    def configure_error(
        self, error: Exception, after_calls: int = 0
    ) -> None:
        """Configure the provider to raise an error.

        Args:
            error: Exception to raise
            after_calls: Number of successful calls before error (0 = immediately)
        """
        self._error_to_raise = error
        self._error_after_calls = after_calls

    def clear_error(self) -> None:
        """Clear error configuration."""
        self._error_to_raise = None
        self._error_after_calls = -1

    def set_latency(self, seconds: float) -> None:
        """Set simulated latency.

        Args:
            seconds: Latency in seconds
        """
        self._simulate_latency = seconds

    # ============ Query Methods ============

    def get_call_count(self) -> int:
        """Get the number of times chat() was called.

        Returns:
            Number of calls
        """
        return self._call_count

    def get_request_history(self) -> List[Dict[str, Any]]:
        """Get the history of all requests.

        Returns:
            List of request dictionaries
        """
        return self._request_history.copy()

    def get_last_request(self) -> Optional[Dict[str, Any]]:
        """Get the most recent request.

        Returns:
            Last request dict or None if no calls
        """
        return self._request_history[-1] if self._request_history else None

    def reset(self) -> None:
        """Reset the provider state.

        Resets call count, response index, and request history.
        """
        self._call_count = 0
        self._response_index = 0
        self._request_history.clear()
        self.clear_error()

    # ============ Private Methods ============

    def _check_for_error(self) -> None:
        """Check if error should be raised.

        Raises:
            Exception: If error simulation is triggered
        """
        if self._error_to_raise is not None:
            if self._error_after_calls == 0:
                error = self._error_to_raise
                self.clear_error()  # Clear after raising
                raise error
            elif self._error_after_calls > 0:
                self._error_after_calls -= 1


class MockStreamingProvider(MockProvider):
    """Mock provider that always supports streaming.

    This is a convenience class for tests that specifically need
    a streaming provider.

    Example:
        provider = MockStreamingProvider(
            responses=["First chunk", "Second chunk"]
        )

        async for chunk in provider.stream(messages, model="test"):
            print(chunk.delta)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with streaming enabled by default."""
        super().__init__(supports_streaming=True, **kwargs)


class MockToolCallingProvider(MockProvider):
    """Mock provider that supports tool calling.

    This provider can simulate tool calling responses for testing
    tool execution logic.

    Example:
        provider = MockToolCallingProvider(
            tool_calls=[
                {
                    "id": "call_123",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "test.txt"}'
                    }
                }
            ]
        )

        response = await provider.chat(messages)
        assert response.tool_calls is not None
    """

    def __init__(
        self,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with tool calling support.

        Args:
            tool_calls: Default tool calls to include in responses
            **kwargs: Additional arguments passed to MockProvider
        """
        super().__init__(supports_tools=True, default_tool_calls=tool_calls, **kwargs)

    def set_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """Set the tool calls to include in responses.

        Args:
            tool_calls: List of tool call dictionaries
        """
        self._default_tool_calls = tool_calls


def create_mock_provider(
    model: str = "mock-model",
    response: str = "Mock response",
    **kwargs: Any,
) -> MockProvider:
    """Factory function to create a MockProvider.

    This is a convenience function for creating a mock provider
    with a single response.

    Args:
        model: Model name
        response: Default response
        **kwargs: Additional arguments for MockProvider

    Returns:
        Configured MockProvider instance

    Example:
        provider = create_mock_provider(
            model="test-model",
            response="Hello, world!",
            simulate_latency=0.1
        )
    """
    return MockProvider(model=model, responses=[response], **kwargs)
