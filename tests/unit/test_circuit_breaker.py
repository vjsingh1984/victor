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

"""Unit tests for CircuitBreaker."""

import asyncio
import pytest

from victor.providers.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState,
)


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    @pytest.fixture
    def breaker(self) -> CircuitBreaker:
        """Create a fresh circuit breaker for each test."""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=0.1,  # Short timeout for testing
            half_open_max_calls=2,
            success_threshold=2,
            name="test_breaker",
        )

    def test_init_default_state(self, breaker: CircuitBreaker) -> None:
        """Test breaker starts in closed state."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed
        assert not breaker.is_open

    @pytest.mark.asyncio
    async def test_successful_call(self, breaker: CircuitBreaker) -> None:
        """Test successful call passes through."""

        async def success_func() -> str:
            return "success"

        result = await breaker.execute(success_func)
        assert result == "success"
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_opens_after_failures(self, breaker: CircuitBreaker) -> None:
        """Test circuit opens after reaching failure threshold."""

        async def failing_func() -> None:
            raise ValueError("Simulated failure")

        # Fail up to threshold
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.execute(failing_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open

    @pytest.mark.asyncio
    async def test_open_rejects_calls(self, breaker: CircuitBreaker) -> None:
        """Test open circuit rejects calls immediately."""

        async def failing_func() -> None:
            raise ValueError("Simulated failure")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.execute(failing_func)

        # Now calls should be rejected
        async def should_not_run() -> str:
            return "should not reach here"

        with pytest.raises(CircuitBreakerError) as exc_info:
            await breaker.execute(should_not_run)

        assert exc_info.value.state == CircuitState.OPEN
        assert exc_info.value.retry_after > 0

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self, breaker: CircuitBreaker) -> None:
        """Test circuit transitions to half-open after recovery timeout."""

        async def failing_func() -> None:
            raise ValueError("Simulated failure")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.execute(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Should now be half-open
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_closes_on_success(self, breaker: CircuitBreaker) -> None:
        """Test half-open circuit closes after successful calls."""

        async def failing_func() -> None:
            raise ValueError("Simulated failure")

        async def success_func() -> str:
            return "success"

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.execute(failing_func)

        # Wait for recovery timeout
        await asyncio.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # Successful calls should close it (need success_threshold successes)
        await breaker.execute(success_func)
        await breaker.execute(success_func)

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_opens_on_failure(self, breaker: CircuitBreaker) -> None:
        """Test half-open circuit reopens on failure."""

        async def failing_func() -> None:
            raise ValueError("Simulated failure")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await breaker.execute(failing_func)

        # Wait for recovery timeout
        await asyncio.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # Failure in half-open should reopen
        with pytest.raises(ValueError):
            await breaker.execute(failing_func)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_excluded_exceptions(self) -> None:
        """Test that excluded exceptions don't count as failures."""
        breaker = CircuitBreaker(
            failure_threshold=3,
            excluded_exceptions=(ValueError,),
            name="test_excluded",
        )

        async def raise_value_error() -> None:
            raise ValueError("This is excluded")

        # These should not open the circuit
        for _ in range(5):
            with pytest.raises(ValueError):
                await breaker.execute(raise_value_error)

        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_decorator_usage(self, breaker: CircuitBreaker) -> None:
        """Test circuit breaker as decorator."""
        call_count = 0

        @breaker
        async def decorated_func() -> str:
            nonlocal call_count
            call_count += 1
            return "decorated result"

        result = await decorated_func()
        assert result == "decorated result"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_context_manager_success(self, breaker: CircuitBreaker) -> None:
        """Test circuit breaker as async context manager."""
        async with breaker:
            pass  # Success
        assert breaker.is_closed

    @pytest.mark.asyncio
    async def test_context_manager_failure(self, breaker: CircuitBreaker) -> None:
        """Test context manager records failure on exception."""
        for _ in range(3):
            with pytest.raises(ValueError):
                async with breaker:
                    raise ValueError("Simulated failure")

        assert breaker.is_open

    def test_reset(self, breaker: CircuitBreaker) -> None:
        """Test manual reset returns to closed state."""
        # Force open state by modifying internal counter
        breaker._failure_count = breaker.failure_threshold
        breaker._record_failure()
        assert breaker.is_open

        breaker.reset()
        assert breaker.is_closed

    def test_get_stats(self, breaker: CircuitBreaker) -> None:
        """Test statistics collection."""
        stats = breaker.get_stats()
        assert stats["name"] == "test_breaker"
        assert stats["state"] == "closed"
        assert stats["total_calls"] == 0


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        CircuitBreakerRegistry._breakers.clear()

    def test_get_or_create(self) -> None:
        """Test get_or_create returns same instance."""
        breaker1 = CircuitBreakerRegistry.get_or_create("test", failure_threshold=5)
        breaker2 = CircuitBreakerRegistry.get_or_create("test", failure_threshold=10)

        assert breaker1 is breaker2
        assert breaker1.failure_threshold == 5  # First creation wins

    def test_get(self) -> None:
        """Test get returns existing breaker."""
        CircuitBreakerRegistry.get_or_create("existing")
        assert CircuitBreakerRegistry.get("existing") is not None
        assert CircuitBreakerRegistry.get("nonexistent") is None

    def test_reset_all(self) -> None:
        """Test reset_all resets all breakers."""
        breaker1 = CircuitBreakerRegistry.get_or_create("b1", failure_threshold=1)
        breaker2 = CircuitBreakerRegistry.get_or_create("b2", failure_threshold=1)

        # Open both
        breaker1._failure_count = 1
        breaker1._record_failure()
        breaker2._failure_count = 1
        breaker2._record_failure()

        CircuitBreakerRegistry.reset_all()

        assert breaker1.is_closed
        assert breaker2.is_closed

    def test_get_all_stats(self) -> None:
        """Test getting stats from all breakers."""
        CircuitBreakerRegistry.get_or_create("stats1")
        CircuitBreakerRegistry.get_or_create("stats2")

        stats = CircuitBreakerRegistry.get_all_stats()
        assert "stats1" in stats
        assert "stats2" in stats
