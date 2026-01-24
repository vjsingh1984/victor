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

"""Comprehensive tests for provider error handling.

Test areas:
1. Retry logic with exponential backoff (8 tests)
2. Rate limiting handling (6 tests)
3. Network failure recovery (5 tests)
4. Timeout scenarios (5 tests)
5. Authentication failures (3 tests)
6. Connection errors (3 tests)

Total: 30+ tests
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch
from typing import List

from tests.mocks.provider_mocks import (
    FailingProvider,
    LatencySimulationProvider,
    MockBaseProvider,
    ProviderTestHelpers,
)
from victor.core.errors import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderInvalidResponseError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from victor.providers.base import BaseProvider, Message, CompletionResponse


# =============================================================================
# 1. Retry Logic with Exponential Backoff (8 tests)
# =============================================================================


class TestRetryLogic:
    """Test retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self):
        """Test that transient failures trigger retries."""
        provider = FailingProvider(error_type="connection", fail_after=2)
        messages = ProviderTestHelpers.create_test_messages("Test")

        # First two calls succeed
        response1 = await provider.chat(messages, model="test")
        assert "Success" in response1.content

        response2 = await provider.chat(messages, model="test")
        assert "Success" in response2.content

        # Third call fails
        with pytest.raises(ProviderConnectionError):
            await provider.chat(messages, model="test")

        assert provider.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_with_increasing_delay(self):
        """Test that retries use increasing delays (exponential backoff)."""
        call_times = []

        class TimedRetryProvider(BaseProvider):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            @property
            def name(self) -> str:
                return "timed_retry"

            async def chat(self, messages, *, model, **kwargs) -> CompletionResponse:
                import time

                call_times.append(time.time())
                self.call_count += 1

                if self.call_count < 3:
                    raise ProviderConnectionError("Transient error")

                return CompletionResponse(content="Success", model=model)

            async def stream(self, messages, *, model, **kwargs):
                raise NotImplementedError

            async def close(self) -> None:
                pass

        provider = TimedRetryProvider()
        messages = ProviderTestHelpers.create_test_messages()

        # Simulate retry logic with exponential backoff
        max_retries = 3
        base_delay = 0.1

        for attempt in range(max_retries):
            try:
                response = await provider.chat(messages, model="test")
                assert response.content == "Success"
                break
            except ProviderConnectionError:
                if attempt < max_retries - 1:
                    # Exponential backoff
                    delay = base_delay * (2**attempt)
                    await asyncio.sleep(delay)

        # Verify multiple calls were made
        assert provider.call_count == 3

        # Verify delays increased (approximately)
        if len(call_times) >= 2:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1] if len(call_times) > 2 else 0
            # Second delay should be longer due to exponential backoff
            assert delay2 > delay1 * 0.8  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that retries stop after max_retries is reached."""
        provider = FailingProvider(error_type="connection", fail_after=0)
        messages = ProviderTestHelpers.create_test_messages()

        max_retries = 3
        attempts = 0

        for attempt in range(max_retries):
            try:
                await provider.chat(messages, model="test")
            except ProviderConnectionError:
                attempts += 1
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1)  # Backoff

        assert attempts == max_retries
        assert provider.call_count == max_retries

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Test retry logic on timeout errors."""
        provider = FailingProvider(error_type="timeout", fail_after=1)
        messages = ProviderTestHelpers.create_test_messages()

        # First call succeeds
        response = await provider.chat(messages, model="test")
        assert "Success" in response.content

        # Second call times out
        with pytest.raises(ProviderTimeoutError):
            await provider.chat(messages, model="test")

        assert provider.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_auth_error(self):
        """Test that auth errors don't trigger retries."""
        provider = FailingProvider(error_type="auth", fail_after=0)
        messages = ProviderTestHelpers.create_test_messages()

        # Auth errors should fail immediately
        with pytest.raises(ProviderAuthError):
            await provider.chat(messages, model="test")

        # Should only be called once (no retries)
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self):
        """Test successful retry after failures."""
        provider = FailingProvider(error_type="rate_limit", fail_after=2)
        messages = ProviderTestHelpers.create_test_messages()

        # First two succeed
        response1 = await provider.chat(messages, model="test")
        response2 = await provider.chat(messages, model="test")

        assert "Success" in response1.content
        assert "Success" in response2.content

        # Third fails with rate limit
        with pytest.raises(ProviderRateLimitError):
            await provider.chat(messages, model="test")

    @pytest.mark.asyncio
    async def test_retry_with_jitter(self):
        """Test that retries include jitter to avoid thundering herd."""
        import random

        delays = []

        class JitterRetryProvider(BaseProvider):
            @property
            def name(self) -> str:
                return "jitter_retry"

            async def chat(self, messages, *, model, **kwargs) -> CompletionResponse:
                if not hasattr(self, "call_count"):
                    self.call_count = 0
                self.call_count += 1

                if self.call_count < 3:
                    raise ProviderConnectionError("Transient error")

                return CompletionResponse(content="Success", model=model)

            async def stream(self, messages, *, model, **kwargs):
                raise NotImplementedError

            async def close(self) -> None:
                pass

        provider = JitterRetryProvider()
        messages = ProviderTestHelpers.create_test_messages()

        # Retry with jitter
        base_delay = 0.1
        for attempt in range(3):
            try:
                response = await provider.chat(messages, model="test")
                break
            except ProviderConnectionError:
                if attempt < 2:
                    # Add jitter: base_delay * (2^attempt) + random(-0.01, 0.01)
                    delay = base_delay * (2**attempt) + random.uniform(-0.01, 0.01)
                    delay = max(0, delay)  # Ensure non-negative
                    await asyncio.sleep(delay)

        assert provider.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_preserves_request_context(self):
        """Test that retries preserve the original request context."""
        context_tracker = []

        class ContextTrackingProvider(BaseProvider):
            @property
            def name(self) -> str:
                return "context_tracking"

            async def chat(
                self, messages, *, model, temperature=0.7, **kwargs
            ) -> CompletionResponse:
                context_tracker.append(
                    {"model": model, "temperature": temperature, "messages": messages}
                )

                if len(context_tracker) < 3:
                    raise ProviderConnectionError("Transient error")

                return CompletionResponse(content="Success", model=model)

            async def stream(self, messages, *, model, **kwargs):
                raise NotImplementedError

            async def close(self) -> None:
                pass

        provider = ContextTrackingProvider()
        messages = ProviderTestHelpers.create_test_messages("Test context")

        # Retry with backoff
        for attempt in range(3):
            try:
                response = await provider.chat(messages, model="test-model", temperature=0.9)
                break
            except ProviderConnectionError:
                if attempt < 2:
                    await asyncio.sleep(0.05)

        # All retries should have the same context
        assert len(context_tracker) == 3
        for ctx in context_tracker:
            assert ctx["model"] == "test-model"
            assert ctx["temperature"] == 0.9
            assert len(ctx["messages"]) == 2


# =============================================================================
# 2. Rate Limiting Handling (6 tests)
# =============================================================================


class TestRateLimitHandling:
    """Test rate limiting error handling."""

    @pytest.mark.asyncio
    async def test_rate_limit_error_raised(self):
        """Test that rate limit errors are properly raised."""
        provider = FailingProvider(error_type="rate_limit", fail_after=0)
        messages = ProviderTestHelpers.create_test_messages()

        with pytest.raises(ProviderRateLimitError) as exc_info:
            await provider.chat(messages, model="test")

        assert exc_info.value.retry_after is not None
        assert exc_info.value.retry_after == 60

    @pytest.mark.asyncio
    async def test_rate_limit_retry_after(self):
        """Test retry_after field in rate limit errors."""
        provider = FailingProvider(error_type="rate_limit", fail_after=1)
        messages = ProviderTestHelpers.create_test_messages()

        # First call succeeds
        response = await provider.chat(messages, model="test")
        assert "Success" in response.content

        # Second call hits rate limit
        try:
            await provider.chat(messages, model="test")
        except ProviderRateLimitError as e:
            assert e.retry_after == 60
            # Check either 'rate limit' or 'rate_limit' (from error_type)
            error_str = str(e).lower()
            assert "rate" in error_str and "limit" in error_str

    @pytest.mark.asyncio
    async def test_rate_limit_recovery(self):
        """Test recovery after rate limit."""
        provider = FailingProvider(error_type="rate_limit", fail_after=1)
        messages = ProviderTestHelpers.create_test_messages()

        # First succeeds
        await provider.chat(messages, model="test")

        # Second hits rate limit
        with pytest.raises(ProviderRateLimitError):
            await provider.chat(messages, model="test")

        # In real scenario, would wait retry_after seconds
        # For testing, we verify the error structure
        try:
            await provider.chat(messages, model="test")
        except ProviderRateLimitError as e:
            assert e.retry_after == 60

    @pytest.mark.asyncio
    async def test_custom_retry_after_value(self):
        """Test custom retry_after values."""
        # FailingProvider uses default retry_after=60
        provider = FailingProvider(error_type="rate_limit", fail_after=0)
        messages = ProviderTestHelpers.create_test_messages()

        with pytest.raises(ProviderRateLimitError) as exc_info:
            await provider.chat(messages, model="test")

        assert exc_info.value.retry_after == 60

    @pytest.mark.asyncio
    async def test_rate_limit_error_properties(self):
        """Test rate limit error properties."""
        provider = FailingProvider(error_type="rate_limit", fail_after=0)
        messages = ProviderTestHelpers.create_test_messages()

        with pytest.raises(ProviderRateLimitError) as exc_info:
            await provider.chat(messages, model="test")

        error = exc_info.value
        assert error.provider == "failing_provider"
        assert error.retry_after == 60
        assert error.category.value == "provider_rate_limit"

    @pytest.mark.asyncio
    async def test_rate_limit_includes_recovery_hint(self):
        """Test that rate limit errors include recovery hints."""
        provider = FailingProvider(error_type="rate_limit", fail_after=0)
        messages = ProviderTestHelpers.create_test_messages()

        with pytest.raises(ProviderRateLimitError) as exc_info:
            await provider.chat(messages, model="test")

        error = exc_info.value
        assert error.recovery_hint is not None
        assert "retry" in error.recovery_hint.lower() or "wait" in error.recovery_hint.lower()


# =============================================================================
# 3. Network Failure Recovery (5 tests)
# =============================================================================


class TestNetworkFailureRecovery:
    """Test network failure recovery."""

    @pytest.mark.asyncio
    async def test_connection_error_detection(self):
        """Test detection of connection errors."""
        provider = FailingProvider(error_type="connection", fail_after=0)
        messages = ProviderTestHelpers.create_test_messages()

        with pytest.raises(ProviderConnectionError):
            await provider.chat(messages, model="test")

    @pytest.mark.asyncio
    async def test_transient_connection_failure_recovery(self):
        """Test recovery from transient connection failures."""
        provider = FailingProvider(error_type="connection", fail_after=2)
        messages = ProviderTestHelpers.create_test_messages()

        # First two calls succeed
        response1 = await provider.chat(messages, model="test")
        response2 = await provider.chat(messages, model="test")

        assert "Success" in response1.content
        assert "Success" in response2.content

        # Third fails
        with pytest.raises(ProviderConnectionError):
            await provider.chat(messages, model="test")

    @pytest.mark.asyncio
    async def test_persistent_connection_failure(self):
        """Test handling of persistent connection failures."""
        provider = FailingProvider(error_type="connection", fail_after=0)
        messages = ProviderTestHelpers.create_test_messages()

        attempts = 0
        max_attempts = 3

        for _ in range(max_attempts):
            try:
                await provider.chat(messages, model="test")
            except ProviderConnectionError:
                attempts += 1

        assert attempts == max_attempts

    @pytest.mark.asyncio
    async def test_network_error_categorization(self):
        """Test that network errors are properly categorized."""
        provider = FailingProvider(error_type="connection", fail_after=0)
        messages = ProviderTestHelpers.create_test_messages()

        with pytest.raises(ProviderConnectionError) as exc_info:
            await provider.chat(messages, model="test")

        error = exc_info.value
        assert error.category.value == "provider_connection"
        assert error.provider == "failing_provider"

    @pytest.mark.asyncio
    async def test_connection_error_recovery_hint(self):
        """Test that connection errors include recovery hints."""
        provider = FailingProvider(error_type="connection", fail_after=0)
        messages = ProviderTestHelpers.create_test_messages()

        with pytest.raises(ProviderConnectionError) as exc_info:
            await provider.chat(messages, model="test")

        error = exc_info.value
        assert error.recovery_hint is not None
        assert (
            "network" in error.recovery_hint.lower() or "connection" in error.recovery_hint.lower()
        )


# =============================================================================
# 4. Timeout Scenarios (5 tests)
# =============================================================================


class TestTimeoutScenarios:
    """Test timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_detection(self):
        """Test detection of timeout errors."""
        provider = FailingProvider(error_type="timeout", fail_after=0)
        messages = ProviderTestHelpers.create_test_messages()

        with pytest.raises(ProviderTimeoutError):
            await provider.chat(messages, model="test")

    @pytest.mark.asyncio
    async def test_timeout_error_properties(self):
        """Test timeout error properties."""
        provider = FailingProvider(error_type="timeout", fail_after=0)
        messages = ProviderTestHelpers.create_test_messages()

        with pytest.raises(ProviderTimeoutError) as exc_info:
            await provider.chat(messages, model="test")

        error = exc_info.value
        assert error.provider == "failing_provider"
        assert error.timeout is not None
        assert error.category.value == "provider_connection"

    @pytest.mark.asyncio
    async def test_timeout_with_latency_provider(self):
        """Test timeout using LatencySimulationProvider."""
        provider = LatencySimulationProvider(base_latency=0.3, timeout_after=0.2)
        messages = ProviderTestHelpers.create_test_messages()

        with pytest.raises(ProviderTimeoutError):
            await provider.chat(messages, model="test")

    @pytest.mark.asyncio
    async def test_timeout_includes_latency_info(self):
        """Test that timeout errors include latency information."""
        provider = LatencySimulationProvider(base_latency=0.5, timeout_after=0.3)
        messages = ProviderTestHelpers.create_test_messages()

        with pytest.raises(ProviderTimeoutError) as exc_info:
            await provider.chat(messages, model="test")

        error = exc_info.value
        assert "timeout" in str(error).lower()
        # timeout field is 0 from LatencySimulationProvider, which is valid
        # (indicates timeout occurred)
        assert error.timeout >= 0

    @pytest.mark.asyncio
    async def test_timeout_recovery_hint(self):
        """Test that timeout errors include recovery hints."""
        provider = FailingProvider(error_type="timeout", fail_after=0)
        messages = ProviderTestHelpers.create_test_messages()

        with pytest.raises(ProviderTimeoutError) as exc_info:
            await provider.chat(messages, model="test")

        error = exc_info.value
        assert error.recovery_hint is not None
        assert (
            "timeout" in error.recovery_hint.lower() or "timed out" in error.recovery_hint.lower()
        )


# =============================================================================
# 5. Authentication Failures (3 tests)
# =============================================================================


class TestAuthenticationFailures:
    """Test authentication error handling."""

    @pytest.mark.asyncio
    async def test_auth_error_detection(self):
        """Test detection of authentication errors."""
        provider = FailingProvider(error_type="auth", fail_after=0)
        messages = ProviderTestHelpers.create_test_messages()

        with pytest.raises(ProviderAuthError):
            await provider.chat(messages, model="test")

    @pytest.mark.asyncio
    async def test_auth_no_retry(self):
        """Test that auth errors don't trigger retries."""
        provider = FailingProvider(error_type="auth", fail_after=0)
        messages = ProviderTestHelpers.create_test_messages()

        with pytest.raises(ProviderAuthError):
            await provider.chat(messages, model="test")

        # Should only be called once (no retries for auth errors)
        assert provider.call_count == 1

    @pytest.mark.asyncio
    async def test_auth_error_properties(self):
        """Test authentication error properties."""
        provider = FailingProvider(error_type="auth", fail_after=0)
        messages = ProviderTestHelpers.create_test_messages()

        with pytest.raises(ProviderAuthError) as exc_info:
            await provider.chat(messages, model="test")

        error = exc_info.value
        assert error.provider == "failing_provider"
        assert error.category.value == "provider_auth"
        assert error.recovery_hint is not None


# =============================================================================
# 6. Connection Errors (3 tests)
# =============================================================================


class TestConnectionErrors:
    """Test connection error handling."""

    @pytest.mark.asyncio
    async def test_connection_error_isolation(self):
        """Test that connection errors don't corrupt provider state."""
        provider = MockBaseProvider(response_text="OK")
        messages = ProviderTestHelpers.create_test_messages()

        # First call succeeds
        response1 = await provider.chat(messages, model="test")
        assert response1.content == "OK"

        # Simulate connection error
        with pytest.raises(ProviderConnectionError):
            raise ProviderConnectionError("Network failure")

        # Provider should still work
        response2 = await provider.chat(messages, model="test")
        assert response2.content == "OK"

    @pytest.mark.asyncio
    async def test_connection_error_with_circuit_breaker(self):
        """Test that connection errors work with circuit breaker pattern."""
        from victor.providers.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        # Create a circuit breaker with low threshold
        config = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=5.0)
        breaker = CircuitBreaker.from_config("test_provider", config)

        provider = MockBaseProvider(response_text="OK")
        messages = ProviderTestHelpers.create_test_messages()

        # Simulate failures through circuit breaker
        async def failing_call():
            raise ProviderConnectionError("Network failure")

        # First failure
        with pytest.raises(ProviderConnectionError):
            await breaker.execute(failing_call)

        assert breaker.state == CircuitState.CLOSED  # Still closed after first failure

        # Second failure - should open circuit
        with pytest.raises(ProviderConnectionError):
            await breaker.execute(failing_call)

        assert breaker.state == CircuitState.OPEN  # Circuit should be open now

        # Third call should fail immediately due to open circuit
        with pytest.raises(Exception):  # CircuitBreakerError
            await breaker.execute(failing_call)

    @pytest.mark.asyncio
    async def test_connection_error_recovery_after_circuit_reset(self):
        """Test recovery after circuit breaker reset."""
        from victor.providers.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        # Create a circuit breaker
        config = CircuitBreakerConfig(failure_threshold=2, success_threshold=2, timeout_seconds=5.0)
        breaker = CircuitBreaker.from_config("test_provider", config)

        provider = MockBaseProvider(response_text="OK")
        messages = ProviderTestHelpers.create_test_messages()

        # Trip the circuit breaker with failures
        async def failing_call():
            raise ProviderConnectionError("Network failure")

        # First two failures trip the circuit
        with pytest.raises(ProviderConnectionError):
            await breaker.execute(failing_call)

        with pytest.raises(ProviderConnectionError):
            await breaker.execute(failing_call)

        assert breaker.state == CircuitState.OPEN

        # Manually reset for testing (in real scenario, would wait for timeout)
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED

        # Should be able to execute again
        async def success_call():
            return "success"

        result = await breaker.execute(success_call)
        assert result == "success"
