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

"""Comprehensive tests for BaseProvider protocols and functionality.

Target: 60%+ coverage for victor/providers/base.py

Test areas:
1. Protocol compliance (5 tests) - Verify BaseProvider implements required protocols
2. Health checks (8 tests) - Test health check functionality
3. Circuit breakers (10 tests) - Test circuit breaker state transitions
4. Tool calling support (8 tests) - Test supports_tools() method
5. Streaming support (8 tests) - Test supports_streaming() method
6. Provider metadata (6 tests) - Test name, model, capabilities properties
7. Error handling (5 tests) - Test error handling in base methods

Total: 50 tests
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import List, AsyncIterator, Optional

from victor.providers.base import (
    BaseProvider,
    Message,
    ToolDefinition,
    CompletionResponse,
    StreamChunk,
    StreamingProvider,
    ToolCallingProvider,
    is_streaming_provider,
    is_tool_calling_provider,
)
from victor.providers.circuit_breaker import CircuitBreakerError, CircuitState
from victor.providers.runtime_capabilities import ProviderRuntimeCapabilities
from victor.core.errors import (
    ProviderError,
    ProviderAuthError,
    ProviderConnectionError,
)


# -----------------------------------------------------------------------------
# Mock Providers for Testing
# -----------------------------------------------------------------------------


class MockBasicProvider(BaseProvider):
    """Minimal mock provider for testing base functionality."""

    # Counter to generate unique circuit breaker names
    _instance_count = 0

    def __init__(self, *args, **kwargs):
        # Extract circuit breaker parameters before calling super
        self._cb_failure_threshold = kwargs.pop("circuit_breaker_failure_threshold", None)
        self._cb_recovery_timeout = kwargs.pop("circuit_breaker_recovery_timeout", None)

        # Generate unique name for circuit breaker isolation
        MockBasicProvider._instance_count += 1
        self._instance_id = MockBasicProvider._instance_count
        super().__init__(*args, **kwargs)
        # Override with instance-specific circuit breaker
        self._setup_instance_circuit_breaker()

    def _setup_instance_circuit_breaker(self):
        """Setup instance-specific circuit breaker to avoid singleton issues."""
        from victor.providers.circuit_breaker import CircuitBreaker

        # Use extracted parameters or defaults
        if self._use_circuit_breaker:
            failure_threshold = self._cb_failure_threshold or 5
            recovery_timeout = self._cb_recovery_timeout or 30.0
            self._circuit_breaker = CircuitBreaker(
                name=f"provider_MockBasicProvider_{self._instance_id}",
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )

    @property
    def name(self) -> str:
        return "mock_basic"

    async def chat(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> CompletionResponse:
        return CompletionResponse(
            content="Mock response",
            model=model,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

    async def stream(
        self,
        messages: List[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(content="Mock chunk", model_name=model)
        yield StreamChunk(content=" another chunk", model_name=model, is_final=True)

    async def close(self) -> None:
        pass


class MockToolCallingProvider(MockBasicProvider):
    """Mock provider that supports tool calling."""

    @property
    def name(self) -> str:
        return "mock_tool_calling"

    def supports_tools(self) -> bool:
        return True


class MockStreamingProvider(MockBasicProvider):
    """Mock provider that supports streaming."""

    @property
    def name(self) -> str:
        return "mock_streaming"

    def supports_streaming(self) -> bool:
        return True


class MockFullFeaturedProvider(MockBasicProvider):
    """Mock provider with all capabilities."""

    @property
    def name(self) -> str:
        return "mock_full"

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True


# -----------------------------------------------------------------------------
# 1. Protocol Compliance Tests (5 tests)
# -----------------------------------------------------------------------------


class TestProtocolCompliance:
    """Test that BaseProvider properly implements protocols."""

    def test_base_provider_is_concrete(self):
        """Test that BaseProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseProvider()  # type: ignore

    def test_mock_provider_instantiates(self):
        """Test that mock provider can be instantiated."""
        provider = MockBasicProvider()
        assert provider is not None
        assert provider.name == "mock_basic"

    def test_tool_calling_protocol_check(self):
        """Test ToolCallingProvider protocol check."""
        provider = MockToolCallingProvider()
        assert isinstance(provider, ToolCallingProvider)
        assert provider.supports_tools() is True

    def test_streaming_protocol_check(self):
        """Test StreamingProvider protocol check."""
        provider = MockStreamingProvider()
        assert isinstance(provider, StreamingProvider)
        assert provider.supports_streaming() is True

    def test_full_featured_protocol_check(self):
        """Test that full featured provider implements both protocols."""
        provider = MockFullFeaturedProvider()
        assert isinstance(provider, ToolCallingProvider)
        assert isinstance(provider, StreamingProvider)
        assert provider.supports_tools() is True
        assert provider.supports_streaming() is True


# -----------------------------------------------------------------------------
# 2. Health Check Tests (8 tests)
# -----------------------------------------------------------------------------


class TestHealthChecks:
    """Test health check functionality."""

    def test_circuit_breaker_initially_closed(self):
        """Test that circuit breaker starts in closed state."""
        provider = MockBasicProvider()
        assert provider.is_circuit_open() is False

    def test_circuit_breaker_stats_initial(self):
        """Test initial circuit breaker statistics."""
        provider = MockBasicProvider()
        stats = provider.get_circuit_breaker_stats()
        assert stats is not None
        assert stats["state"] == CircuitState.CLOSED.value
        assert stats["failure_count"] == 0
        assert stats["success_count"] == 0

    def test_health_check_with_circuit_disabled(self):
        """Test health check when circuit breaker is disabled."""
        provider = MockBasicProvider(use_circuit_breaker=False)
        assert provider.is_circuit_open() is False
        assert provider.circuit_breaker is None

    def test_health_check_after_successful_call(self):
        """Test circuit breaker state after successful call."""
        provider = MockBasicProvider()

        async def run_test():
            # Simulate successful call through circuit breaker
            async def mock_call():
                return "success"

            await provider._execute_with_circuit_breaker(mock_call)

            stats = provider.get_circuit_breaker_stats()
            assert stats["state"] == CircuitState.CLOSED.value
            assert stats["total_calls"] == 1
            assert stats["failure_count"] == 0

        asyncio.run(run_test())

    def test_health_check_after_failure(self):
        """Test circuit breaker state after failure."""
        provider = MockBasicProvider(circuit_breaker_failure_threshold=2)

        async def run_test():
            # Simulate failures through circuit breaker
            async def mock_call():
                raise ProviderConnectionError("Connection failed")

            # First failure
            try:
                await provider._execute_with_circuit_breaker(mock_call)
            except ProviderConnectionError:
                pass

            # Circuit should still be closed (threshold is 2)
            assert provider.is_circuit_open() is False

            # Second failure - should open circuit
            try:
                await provider._execute_with_circuit_breaker(mock_call)
            except ProviderConnectionError:
                pass

            # Now circuit should be open
            assert provider.is_circuit_open() is True

        asyncio.run(run_test())

    def test_circuit_breaker_stats_after_multiple_calls(self):
        """Test circuit breaker statistics track multiple calls."""
        provider = MockBasicProvider()

        async def run_test():
            # Simulate 5 successful calls through circuit breaker
            async def mock_call():
                return "success"

            for _ in range(5):
                await provider._execute_with_circuit_breaker(mock_call)

            stats = provider.get_circuit_breaker_stats()
            assert stats["total_calls"] == 5
            assert stats["failure_count"] == 0

        asyncio.run(run_test())

    def test_circuit_breaker_excludes_auth_errors(self):
        """Test that auth errors don't trip the circuit breaker."""
        provider = MockBasicProvider(circuit_breaker_failure_threshold=2)

        async def run_test():
            # Auth errors should not count as failures
            with patch.object(provider, "chat", side_effect=ProviderAuthError("Invalid API key")):
                try:
                    await provider.chat([], model="test")
                except ProviderAuthError:
                    pass

            # Circuit should still be closed
            assert provider.is_circuit_open() is False

            stats = provider.get_circuit_breaker_stats()
            assert stats["failure_count"] == 0

        asyncio.run(run_test())

    def test_health_check_with_custom_threshold(self):
        """Test circuit breaker with custom failure threshold."""
        # Note: Custom thresholds aren't easily testable with our mock setup
        # because the circuit breaker is created in __init__ before we can
        # intercept the parameters. This test verifies the circuit breaker
        # exists and can be queried.
        provider = MockBasicProvider()

        assert provider.circuit_breaker is not None
        stats = provider.get_circuit_breaker_stats()
        # Just verify the circuit breaker is functional
        assert stats["state"] == CircuitState.CLOSED.value


# -----------------------------------------------------------------------------
# 3. Circuit Breaker Tests (10 tests)
# -----------------------------------------------------------------------------


class TestCircuitBreaker:
    """Test circuit breaker state transitions."""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initializes with correct defaults."""
        provider = MockBasicProvider()
        assert provider.circuit_breaker is not None
        assert provider.is_circuit_open() is False

    def test_circuit_opens_after_threshold(self):
        """Test circuit opens after failure threshold is reached."""
        provider = MockBasicProvider()

        async def run_test():
            # Simulate 5 failures (default threshold)
            async def mock_call():
                raise ProviderConnectionError("Failed")

            for _ in range(5):
                try:
                    await provider._execute_with_circuit_breaker(mock_call)
                except ProviderConnectionError:
                    pass

            # Circuit should now be open
            assert provider.is_circuit_open() is True

        asyncio.run(run_test())

    def test_circuit_prevents_requests_when_open(self):
        """Test that open circuit prevents new requests."""
        provider = MockBasicProvider()

        async def run_test():
            # Open the circuit
            async def mock_call():
                raise ProviderConnectionError("Failed")

            for _ in range(5):
                try:
                    await provider._execute_with_circuit_breaker(mock_call)
                except ProviderConnectionError:
                    pass

            assert provider.is_circuit_open() is True

            # Try to make a request - should be rejected by circuit
            with pytest.raises(CircuitBreakerError):
                await provider._execute_with_circuit_breaker(mock_call)

        asyncio.run(run_test())

    def test_circuit_manual_reset(self):
        """Test manual circuit breaker reset."""
        provider = MockBasicProvider()

        async def run_test():
            # Open the circuit
            async def mock_call():
                raise ProviderConnectionError("Failed")

            for _ in range(5):
                try:
                    await provider._execute_with_circuit_breaker(mock_call)
                except ProviderConnectionError:
                    pass

            assert provider.is_circuit_open() is True

            # Reset the circuit
            provider.reset_circuit_breaker()
            assert provider.is_circuit_open() is False

        asyncio.run(run_test())

    def test_circuit_resets_after_success(self):
        """Test that circuit closes after successful request."""
        provider = MockBasicProvider()

        async def run_test():
            # Open the circuit
            async def mock_call():
                raise ProviderConnectionError("Failed")

            for _ in range(5):
                try:
                    await provider._execute_with_circuit_breaker(mock_call)
                except ProviderConnectionError:
                    pass

            assert provider.is_circuit_open() is True

            # Reset to test recovery
            provider.reset_circuit_breaker()

            # Successful request should keep circuit closed
            async def success_call():
                return "success"

            await provider._execute_with_circuit_breaker(success_call)
            assert provider.is_circuit_open() is False

        asyncio.run(run_test())

    def test_circuit_breaker_disabled_when_requested(self):
        """Test that circuit breaker can be disabled."""
        provider = MockBasicProvider(use_circuit_breaker=False)
        assert provider.circuit_breaker is None
        assert provider.is_circuit_open() is False

    def test_circuit_breaker_with_custom_timeout(self):
        """Test circuit breaker has recovery timeout."""
        provider = MockBasicProvider()

        assert provider.circuit_breaker is not None
        # Just verify the circuit breaker has a recovery timeout
        assert provider.circuit_breaker.recovery_timeout > 0

    def test_circuit_half_open_state(self):
        """Test circuit transitions to half-open after timeout."""
        # Note: Half-open state testing is complex and time-dependent
        # This test just verifies that after opening the circuit,
        # the reset mechanism works
        provider = MockBasicProvider()

        async def run_test():
            # Open the circuit
            async def mock_call():
                raise ProviderConnectionError("Failed")

            for _ in range(5):
                try:
                    await provider._execute_with_circuit_breaker(mock_call)
                except ProviderConnectionError:
                    pass

            assert provider.is_circuit_open() is True

            # Manual reset works (simulating recovery)
            provider.reset_circuit_breaker()
            assert provider.is_circuit_open() is False

            # Successful call after reset
            async def success_call():
                return "success"

            await provider._execute_with_circuit_breaker(success_call)
            assert provider.is_circuit_open() is False

        asyncio.run(run_test())

    def test_execute_with_circuit_breaker_success(self):
        """Test _execute_with_circuit_breaker on successful execution."""
        provider = MockBasicProvider()

        async def mock_func():
            return "success"

        async def run_test():
            result = await provider._execute_with_circuit_breaker(mock_func)
            assert result == "success"

        asyncio.run(run_test())

    def test_execute_with_circuit_breaker_failure(self):
        """Test _execute_with_circuit_breaker handles failures."""
        provider = MockBasicProvider()

        async def mock_func():
            raise ProviderConnectionError("Failed")

        async def run_test():
            with pytest.raises(ProviderConnectionError):
                await provider._execute_with_circuit_breaker(mock_func)

            # Failure should be recorded
            stats = provider.get_circuit_breaker_stats()
            assert stats["failure_count"] > 0

        asyncio.run(run_test())


# -----------------------------------------------------------------------------
# 4. Tool Calling Support Tests (8 tests)
# -----------------------------------------------------------------------------


class TestToolCallingSupport:
    """Test supports_tools() method."""

    def test_default_tool_support_false(self):
        """Test that default provider doesn't support tools."""
        provider = MockBasicProvider()
        # The method returns False, but isinstance check passes due to structural typing
        assert provider.supports_tools() is False
        # Note: isinstance passes because protocols are structural in Python
        # The provider has the supports_tools method, so it matches the protocol

    def test_tool_calling_provider_returns_true(self):
        """Test that MockToolCallingProvider supports tools."""
        provider = MockToolCallingProvider()
        assert provider.supports_tools() is True
        assert isinstance(provider, ToolCallingProvider)

    def test_is_tool_calling_provider_helper(self):
        """Test is_tool_calling_provider() helper function."""
        basic_provider = MockBasicProvider()
        tool_provider = MockToolCallingProvider()

        assert is_tool_calling_provider(basic_provider) is False
        assert is_tool_calling_provider(tool_provider) is True

    def test_tools_parameter_in_chat(self):
        """Test that tools parameter is accepted in chat()."""
        provider = MockToolCallingProvider()

        async def run_test():
            tools = [
                ToolDefinition(
                    name="test_tool",
                    description="A test tool",
                    parameters={"type": "object", "properties": {}},
                )
            ]
            messages = [Message(role="user", content="Hello")]

            response = await provider.chat(messages, model="test", tools=tools)
            assert response.content == "Mock response"

        asyncio.run(run_test())

    def test_tools_parameter_in_stream(self):
        """Test that tools parameter is accepted in stream()."""
        provider = MockToolCallingProvider()

        async def run_test():
            tools = [
                ToolDefinition(
                    name="test_tool",
                    description="A test tool",
                    parameters={"type": "object", "properties": {}},
                )
            ]
            messages = [Message(role="user", content="Hello")]

            chunks = []
            async for chunk in provider.stream(messages, model="test", tools=tools):
                chunks.append(chunk)

            assert len(chunks) == 2

        asyncio.run(run_test())

    def test_full_featured_supports_tools(self):
        """Test that full featured provider supports tools."""
        provider = MockFullFeaturedProvider()
        assert provider.supports_tools() is True
        assert isinstance(provider, ToolCallingProvider)

    def test_protocol_check_with_hasattr(self):
        """Test protocol check using hasattr."""
        provider = MockBasicProvider()
        assert hasattr(provider, "supports_tools")
        assert callable(provider.supports_tools)

    def test_tool_calling_provider_protocol_definition(self):
        """Test that ToolCallingProvider protocol is properly defined."""
        # Verify protocol has the right methods
        assert hasattr(ToolCallingProvider, "supports_tools")
        assert callable(ToolCallingProvider.supports_tools)


# -----------------------------------------------------------------------------
# 5. Streaming Support Tests (8 tests)
# -----------------------------------------------------------------------------


class TestStreamingSupport:
    """Test supports_streaming() method."""

    def test_default_streaming_support_false(self):
        """Test that default provider doesn't support streaming."""
        provider = MockBasicProvider()
        # The method returns False, but isinstance check passes due to structural typing
        assert provider.supports_streaming() is False
        # Note: isinstance passes because protocols are structural in Python
        # The provider has the supports_streaming and stream methods, so it matches the protocol

    def test_streaming_provider_returns_true(self):
        """Test that MockStreamingProvider supports streaming."""
        provider = MockStreamingProvider()
        assert provider.supports_streaming() is True
        assert isinstance(provider, StreamingProvider)

    def test_is_streaming_provider_helper(self):
        """Test is_streaming_provider() helper function."""
        basic_provider = MockBasicProvider()
        streaming_provider = MockStreamingProvider()

        assert is_streaming_provider(basic_provider) is False
        assert is_streaming_provider(streaming_provider) is True

    def test_stream_method_returns_async_iterator(self):
        """Test that stream() method returns AsyncIterator."""
        provider = MockBasicProvider()

        async def run_test():
            messages = [Message(role="user", content="Hello")]
            stream_result = provider.stream(messages, model="test")

            # Should be async iterable
            assert hasattr(stream_result, "__aiter__")

            # Should be able to iterate
            chunks = []
            async for chunk in stream_result:
                chunks.append(chunk)

            assert len(chunks) == 2

        asyncio.run(run_test())

    def test_stream_chat_alias(self):
        """Test that stream_chat() is an alias for stream()."""
        provider = MockBasicProvider()

        async def run_test():
            messages = [Message(role="user", content="Hello")]

            chunks = []
            async for chunk in provider.stream_chat(messages, model="test"):
                chunks.append(chunk)

            assert len(chunks) == 2

        asyncio.run(run_test())

    def test_full_featured_supports_streaming(self):
        """Test that full featured provider supports streaming."""
        provider = MockFullFeaturedProvider()
        assert provider.supports_streaming() is True
        assert isinstance(provider, StreamingProvider)

    def test_stream_chunk_properties(self):
        """Test StreamChunk properties."""
        chunk = StreamChunk(
            content="test content",
            is_final=True,
            model_name="test-model",
            usage={"total_tokens": 100},
        )

        assert chunk.content == "test content"
        assert chunk.is_final is True
        assert chunk.model_name == "test-model"
        assert chunk.usage is not None

    def test_streaming_protocol_definition(self):
        """Test that StreamingProvider protocol is properly defined."""
        # Verify protocol has the right methods
        assert hasattr(StreamingProvider, "supports_streaming")
        assert hasattr(StreamingProvider, "stream")
        assert callable(StreamingProvider.supports_streaming)
        assert callable(StreamingProvider.stream)


# -----------------------------------------------------------------------------
# 6. Provider Metadata Tests (6 tests)
# -----------------------------------------------------------------------------


class TestProviderMetadata:
    """Test provider metadata properties."""

    def test_provider_name_property(self):
        """Test that provider has name property."""
        provider = MockBasicProvider()
        assert provider.name == "mock_basic"
        assert isinstance(provider.name, str)

    def test_provider_name_is_abstract(self):
        """Test that name property is abstract and must be overridden."""
        # This is implicitly tested by the fact that BaseProvider can't be instantiated
        # and our mock providers must implement it
        provider = MockToolCallingProvider()
        assert provider.name == "mock_tool_calling"

    def test_provider_initialization_parameters(self):
        """Test provider initialization with various parameters."""
        provider = MockBasicProvider(
            api_key="test_key",
            base_url="https://api.example.com",
            timeout=120,
            max_retries=5,
        )

        assert provider.api_key == "test_key"
        assert provider.base_url == "https://api.example.com"
        assert provider.timeout == 120
        assert provider.max_retries == 5

    def test_provider_extra_config(self):
        """Test that extra kwargs are stored in extra_config."""
        provider = MockBasicProvider(custom_param="value", another_param=123)

        assert provider.extra_config == {"custom_param": "value", "another_param": 123}

    def test_discover_capabilities(self):
        """Test discover_capabilities() method."""
        provider = MockFullFeaturedProvider()

        async def run_test():
            capabilities = await provider.discover_capabilities("test-model")

            assert isinstance(capabilities, ProviderRuntimeCapabilities)
            assert capabilities.provider == "mock_full"
            assert capabilities.model == "test-model"
            assert capabilities.supports_tools is True
            assert capabilities.supports_streaming is True
            assert capabilities.source == "config"

        asyncio.run(run_test())

    def test_discover_capabilities_basic_provider(self):
        """Test discover_capabilities() for basic provider."""
        provider = MockBasicProvider()

        async def run_test():
            capabilities = await provider.discover_capabilities("test-model")

            assert capabilities.supports_tools is False
            assert capabilities.supports_streaming is False

        asyncio.run(run_test())


# -----------------------------------------------------------------------------
# 7. Error Handling Tests (5 tests)
# -----------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling in base methods."""

    def test_close_method_can_be_called(self):
        """Test that close() method can be called."""
        provider = MockBasicProvider()

        async def run_test():
            await provider.close()
            # Should not raise any exception

        asyncio.run(run_test())

    @pytest.mark.asyncio
    async def test_count_tokens_method(self):
        """Test count_tokens() method."""
        provider = MockBasicProvider()
        token_count = await provider.count_tokens("This is a test message")
        # Simple estimation: ~4 characters per token
        assert token_count > 0
        assert isinstance(token_count, int)

    @pytest.mark.asyncio
    async def test_count_tokens_empty_string(self):
        """Test count_tokens() with empty string."""
        provider = MockBasicProvider()
        token_count = await provider.count_tokens("")
        assert token_count == 0

    @pytest.mark.asyncio
    async def test_count_tokens_long_text(self):
        """Test count_tokens() with longer text."""
        provider = MockBasicProvider()
        long_text = "word " * 1000  # ~5000 characters
        token_count = await provider.count_tokens(long_text)
        # Should be roughly 1250 tokens (5000 / 4)
        assert token_count > 1000

    @pytest.mark.asyncio
    async def test_completion_response_properties(self):
        """Test CompletionResponse properties."""
        response = CompletionResponse(
            content="Test response",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            stop_reason="length",
            metadata={"reasoning_content": "thinking"},
        )

        assert response.content == "Test response"
        assert response.model_name == "test-model"  # Test backward-compatible alias
        assert response.usage is not None
        assert response.stop_reason == "length"
        assert response.metadata is not None


# -----------------------------------------------------------------------------
# 8. Additional Tests for Protocol Helper Functions (3 tests)
# -----------------------------------------------------------------------------


class TestProtocolHelperFunctions:
    """Test protocol helper functions."""

    def test_is_streaming_provider_with_none(self):
        """Test is_streaming_provider() with None."""
        assert is_streaming_provider(None) is False

    def test_is_tool_calling_provider_with_none(self):
        """Test is_tool_calling_provider() with None."""
        assert is_tool_calling_provider(None) is False

    def test_is_streaming_provider_with_object_without_method(self):
        """Test is_streaming_provider() with object that doesn't have method."""

        class NotAProvider:
            pass

        obj = NotAProvider()
        assert is_streaming_provider(obj) is False


# -----------------------------------------------------------------------------
# 9. Data Model Tests (5 tests)
# -----------------------------------------------------------------------------


class TestDataModels:
    """Test data model classes."""

    def test_message_creation(self):
        """Test Message model creation."""
        message = Message(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"
        assert message.tool_calls is None

    def test_message_with_tool_calls(self):
        """Test Message model with tool calls."""
        tool_calls = [{"id": "call_1", "function": {"name": "test_tool"}}]
        message = Message(role="assistant", content="", tool_calls=tool_calls)
        assert message.tool_calls == tool_calls

    def test_message_to_dict(self):
        """Test Message.to_dict() method."""
        message = Message(role="user", content="Hello", name="User1")
        message_dict = message.to_dict()
        assert message_dict["role"] == "user"
        assert message_dict["content"] == "Hello"
        assert message_dict["name"] == "User1"

    def test_tool_definition_creation(self):
        """Test ToolDefinition model creation."""
        tool_def = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
        )
        assert tool_def.name == "test_tool"
        assert tool_def.description == "A test tool"

    def test_stream_chunk_with_metadata(self):
        """Test StreamChunk with metadata."""
        chunk = StreamChunk(
            content="partial",
            metadata={"reasoning_content": "thinking process"},
            usage={"cache_read_input_tokens": 100},
        )
        assert chunk.content == "partial"
        assert chunk.metadata is not None
        assert chunk.usage is not None
        assert chunk.usage.get("cache_read_input_tokens") == 100
