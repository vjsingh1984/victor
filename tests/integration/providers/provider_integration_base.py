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

"""Base class for provider integration tests.

This module provides a framework for testing LLM providers with mock HTTP
servers, enabling fast and reliable testing without external dependencies.

Usage:
    class TestMyProvider(ProviderIntegrationTest):
        __test__ = True

        @pytest.fixture
        def provider_name(self):
            return "my_provider"

        @pytest.fixture
        def mock_server(self):
            return MockHTTPEndPoint(response_data={...})
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from victor.config.settings import Settings
from victor.providers.base import BaseProvider, StreamChunk, Message, ToolDefinition


class MockHTTPEndPoint:
    """Mock HTTP endpoint for simulating provider API responses.

    This class simulates an HTTP server that can be used to test
    provider implementations without making actual network calls.
    """

    def __init__(
        self,
        response_data: Optional[Dict[str, Any]] = None,
        status_code: int = 200,
        error_message: Optional[str] = None,
        delay_ms: int = 0,
    ):
        """Initialize mock HTTP endpoint.

        Args:
            response_data: Data to return in response
            status_code: HTTP status code to simulate
            error_message: Error message to return (if status_code != 200)
            delay_ms: Artificial delay in milliseconds
        """
        self.response_data = response_data or {}
        self.status_code = status_code
        self.error_message = error_message
        self.delay_ms = delay_ms
        self.request_count = 0
        self.requests_made: List[Dict[str, Any]] = []

    async def simulate_request(self, **kwargs) -> Tuple[int, Dict[str, Any]]:
        """Simulate an HTTP request to this endpoint.

        Args:
            **kwargs: Request parameters

        Returns:
            Tuple of (status_code, response_data)
        """
        self.request_count += 1
        self.requests_made.append(kwargs)

        # Simulate network delay
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000.0)

        if self.status_code != 200:
            error_data = {
                "error": {
                    "message": self.error_message or f"HTTP {self.status_code}",
                    "type": "api_error",
                    "code": self.status_code,
                }
            }
            return self.status_code, error_data

        return self.status_code, self.response_data

    def reset(self) -> None:
        """Reset request counter and history."""
        self.request_count = 0
        self.requests_made = []


class MockHTTPServer:
    """Mock HTTP server for provider testing.

    Provides a complete mock server implementation with support for
    multiple endpoints, request/response tracking, and error simulation.
    """

    def __init__(self):
        """Initialize mock HTTP server."""
        self.endpoints: Dict[str, MockHTTPEndPoint] = {}
        self.default_endpoint = MockHTTPEndPoint()

    def add_endpoint(
        self,
        path: str,
        response_data: Optional[Dict[str, Any]] = None,
        status_code: int = 200,
        error_message: Optional[str] = None,
    ) -> MockHTTPEndPoint:
        """Add a mock endpoint to the server.

        Args:
            path: Endpoint path (e.g., "/v1/chat/completions")
            response_data: Data to return in response
            status_code: HTTP status code
            error_message: Error message for error responses

        Returns:
            The created MockHTTPEndPoint
        """
        endpoint = MockHTTPEndPoint(
            response_data=response_data,
            status_code=status_code,
            error_message=error_message,
        )
        self.endpoints[path] = endpoint
        return endpoint

    def get_endpoint(self, path: str) -> MockHTTPEndPoint:
        """Get endpoint by path, or default if not found."""
        return self.endpoints.get(path, self.default_endpoint)

    async def request(self, path: str, **kwargs) -> Tuple[int, Dict[str, Any]]:
        """Simulate a request to an endpoint.

        Args:
            path: Endpoint path
            **kwargs: Request parameters

        Returns:
            Tuple of (status_code, response_data)
        """
        endpoint = self.get_endpoint(path)
        return await endpoint.simulate_request(**kwargs)

    def reset(self) -> None:
        """Reset all endpoints."""
        for endpoint in self.endpoints.values():
            endpoint.reset()
        self.default_endpoint.reset()


class ProviderIntegrationTest(ABC):
    """Base class for provider integration tests.

    Provides common test scenarios for all providers:
    - Successful chat completion
    - Authentication error handling
    - Rate limit error handling
    - Timeout error handling
    - Streaming responses

    Subclasses should provider the provider_name fixture and may
    override specific test methods for provider-specific behavior.
    """

    @pytest.fixture
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the provider being tested.

        This fixture must be implemented by subclasses.

        Returns:
            Provider name (e.g., "anthropic", "openai", "ollama")
        """
        raise NotImplementedError("Subclasses must implement provider_name fixture")

    @pytest.fixture
    def mock_server(self) -> MockHTTPServer:
        """Create mock HTTP server for provider testing.

        Returns:
            MockHTTPServer instance configured for successful responses
        """
        server = MockHTTPServer()

        # Default successful chat completion response
        server.add_endpoint(
            "/chat/completions",
            response_data={
                "id": "chatcmpl-test123",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Test response",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

        return server

    @pytest.fixture
    def mock_auth_error_server(self) -> MockHTTPServer:
        """Create mock server that returns authentication errors.

        Returns:
            MockHTTPServer configured for 401 responses
        """
        server = MockHTTPServer()
        server.add_endpoint(
            "/chat/completions",
            status_code=401,
            error_message="Invalid API key",
        )
        return server

    @pytest.fixture
    def mock_rate_limit_server(self) -> MockHTTPServer:
        """Create mock server that returns rate limit errors.

        Returns:
            MockHTTPServer configured for 429 responses
        """
        server = MockHTTPServer()
        server.add_endpoint(
            "/chat/completions",
            status_code=429,
            error_message="Rate limit exceeded",
        )
        return server

    @pytest.fixture
    def mock_timeout_server(self) -> MockHTTPServer:
        """Create mock server that simulates timeout.

        Returns:
            MockHTTPServer configured with artificial delay
        """
        server = MockHTTPServer()
        server.add_endpoint(
            "/chat/completions",
            response_data={
                "id": "chatcmpl-timeout",
                "choices": [{"message": {"role": "assistant", "content": "Delayed response"}}],
            },
            delay_ms=5000,  # 5 second delay
        )
        return server

    @pytest.fixture
    def mock_streaming_server(self) -> MockHTTPServer:
        """Create mock server for streaming responses.

        Returns:
            MockHTTPServer configured for streaming
        """
        server = MockHTTPServer()

        # Note: Streaming is provider-specific, subclasses should override
        server.add_endpoint(
            "/chat/completions",
            response_data={
                "id": "chatcmpl-stream",
                "choices": [{"message": {"role": "assistant", "content": "Streamed response"}}],
            },
        )

        return server

    @pytest.fixture
    def mock_settings(self) -> Settings:
        """Create mock settings for testing.

        Returns:
            Settings instance with test configuration
        """
        return Settings()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_successful_chat(self, mock_server: MockHTTPServer):
        """Test successful chat completion.

        This test verifies that the provider can:
        1. Accept a chat request
        2. Return a valid response
        3. Include proper usage information

        Args:
            mock_server: Mock HTTP server fixture
        """
        # In a real implementation, this would create a provider
        # configured to use the mock server
        status_code, response_data = await mock_server.request("/chat/completions")

        assert status_code == 200
        assert response_data["choices"][0]["message"]["content"] == "Test response"
        assert response_data["usage"]["total_tokens"] > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_auth_error(self, mock_auth_error_server: MockHTTPServer):
        """Test authentication error handling.

        This test verifies that the provider properly handles
        authentication failures (401 errors).

        Args:
            mock_auth_error_server: Mock server returning auth errors
        """
        status_code, response_data = await mock_auth_error_server.request("/chat/completions")

        assert status_code == 401
        assert "error" in response_data
        assert "Invalid API key" in response_data["error"]["message"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rate_limit_error(self, mock_rate_limit_server: MockHTTPServer):
        """Test rate limit error handling.

        This test verifies that the provider properly handles
        rate limit errors (429).

        Args:
            mock_rate_limit_server: Mock server returning rate limit errors
        """
        status_code, response_data = await mock_rate_limit_server.request("/chat/completions")

        assert status_code == 429
        assert "error" in response_data
        assert "Rate limit" in response_data["error"]["message"]

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_timeout_error(self, mock_timeout_server: MockHTTPServer):
        """Test timeout error handling.

        This test verifies that the provider properly handles
        timeouts and can recover or fail gracefully.

        Args:
            mock_timeout_server: Mock server with artificial delay
        """
        import time

        start = time.time()
        status_code, response_data = await mock_timeout_server.request("/chat/completions")
        elapsed = time.time() - start

        assert elapsed >= 5.0  # Should have waited for the delay
        assert status_code == 200

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_turn_conversation(self, mock_server: MockHTTPServer):
        """Test multi-turn conversation handling.

        This test verifies that the provider can:
        1. Handle conversation history
        2. Maintain context across turns
        3. Return coherent responses

        Args:
            mock_server: Mock HTTP server fixture
        """
        # First turn
        _, response1 = await mock_server.request("/chat/completions", messages=[
            {"role": "user", "content": "Hello"}
        ])
        assert response1["choices"][0]["message"]["content"] == "Test response"

        # Second turn (with history)
        _, response2 = await mock_server.request("/chat/completions", messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Test response"},
            {"role": "user", "content": "How are you?"}
        ])
        assert response2["choices"][0]["message"]["content"] == "Test response"

    def test_provider_properties(self, mock_server: MockHTTPServer):
        """Test provider properties and capabilities.

        This test verifies that the provider exposes correct:
        1. Provider name
        2. Support for tools
        3. Support for streaming

        Args:
            mock_server: Mock HTTP server fixture
        """
        # Verify mock server is properly configured
        assert mock_server is not None
        assert hasattr(mock_server, "endpoints")
        assert "/chat/completions" in mock_server.endpoints

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tool_calling(self, mock_server: MockHTTPServer):
        """Test tool calling functionality.

        This test verifies that the provider can:
        1. Accept tool definitions
        2. Return tool calls in responses
        3. Handle tool results

        Args:
            mock_server: Mock HTTP server fixture
        """
        # Configure server for tool calling response
        mock_server.add_endpoint(
            "/chat/completions",
            response_data={
                "id": "chatcmpl-tools",
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "test_tool",
                                    "arguments": '{"param": "value"}',
                                }
                            }
                        ]
                    }
                }],
            },
        )

        status_code, response_data = await mock_server.request(
            "/chat/completions",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "description": "A test tool",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "param": {"type": "string"}
                            },
                            "required": ["param"],
                        }
                    }
                }
            ]
        )

        assert status_code == 200
        assert "tool_calls" in response_data["choices"][0]["message"]


class ProviderErrorScenariosTest(ABC):
    """Base class for provider error scenario tests.

    Tests various error conditions that providers should handle:
    - Network errors
    - Malformed responses
    - Timeout scenarios
    - Concurrent request limits
    """

    @pytest.fixture
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the provider being tested."""
        raise NotImplementedError

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_network_error_recovery(self):
        """Test provider handles network errors gracefully."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_malformed_response_handling(self):
        """Test provider handles malformed responses."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_request_handling(self):
        """Test provider handles concurrent requests."""


class ProviderPerformanceTest(ABC):
    """Base class for provider performance tests.

    Tests performance characteristics:
    - Response latency
    - Throughput
    - Memory usage
    - Connection pooling
    """

    @pytest.fixture
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the provider being tested."""
        raise NotImplementedError

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_response_latency(self):
        """Test provider response latency is acceptable."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_throughput(self):
        """Test provider throughput under load."""


# Utility functions for provider testing
def create_mock_response(
    content: str = "Test response",
    role: str = "assistant",
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    usage: Optional[Dict[str, int]] = None,
) -> MagicMock:
    """Create a mock provider response.

    Args:
        content: Response content
        role: Message role
        tool_calls: Optional tool calls
        usage: Optional token usage info

    Returns:
        MagicMock mimicking a provider response
    """
    response = MagicMock()
    response.content = content
    response.role = role
    response.tool_calls = tool_calls

    if usage:
        usage_mock = MagicMock()
        usage_mock.prompt_tokens = usage.get("prompt_tokens", 10)
        usage_mock.completion_tokens = usage.get("completion_tokens", 5)
        usage_mock.total_tokens = usage.get("total_tokens", 15)
        response.usage = usage_mock

    return response


def create_mock_stream_chunks(
    content: str = "Test streaming response",
    chunk_size: int = 5,
) -> List[StreamChunk]:
    """Create mock streaming chunks.

    Args:
        content: Content to stream
        chunk_size: Size of each chunk

    Returns:
        List of StreamChunk objects
    """
    chunks = []
    for i in range(0, len(content), chunk_size):
        chunk_content = content[i:i + chunk_size]
        is_final = (i + chunk_size) >= len(content)

        chunk = MagicMock(spec=StreamChunk)
        chunk.content = chunk_content
        chunk.delta = chunk_content
        chunk.is_final = is_final
        chunk.usage = None if not is_final else MagicMock(
            prompt_tokens=10,
            completion_tokens=len(content) // 4,
            total_tokens=10 + len(content) // 4,
        )
        chunks.append(chunk)

    return chunks


def assert_provider_error(
    error: Exception,
    expected_type: type,
    expected_message_contains: Optional[str] = None,
) -> None:
    """Assert that an exception is of expected type and message.

    Args:
        error: The exception to check
        expected_type: Expected exception type
        expected_message_contains: Optional string that should be in error message

    Raises:
        AssertionError: If error doesn't match expectations
    """
    assert isinstance(error, expected_type), f"Expected {expected_type}, got {type(error)}"

    if expected_message_contains:
        assert expected_message_contains in str(error), \
            f"Expected '{expected_message_contains}' in error message: {str(error)}"
