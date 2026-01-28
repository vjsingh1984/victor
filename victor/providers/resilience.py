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

"""
Error Recovery and Resilience System for LLM Providers.

This module provides robust error handling for production systems:
- Circuit breaker pattern for failure isolation
- Retry strategy with exponential backoff and jitter
- Fallback provider chain support
- Rate limit handling with Retry-After respect

Note: This module contains an embedded CircuitBreaker implementation optimized
for the ResilientProvider workflow. For standalone circuit breaker usage with
decorator/context manager support, see victor.providers.circuit_breaker.

Usage:
    from victor.providers.resilience import ResilientProvider, CircuitBreakerConfig

    # Wrap provider with resilience features
    resilient = ResilientProvider(
        provider=anthropic_provider,
        fallback_providers=[openai_provider],
    )

    # Use like normal provider
    response = await resilient.chat(messages, model=model)
"""

import asyncio
import logging
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Import canonical types from circuit_breaker.py to avoid duplication
from victor.providers.circuit_breaker import (
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreakerError as CanonicalCircuitBreakerError,
)


@dataclass
class CircuitBreakerState:
    """Runtime state of circuit breaker."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.now)
    half_open_calls: int = 0
    consecutive_successes: int = 0


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, circuit_name: str, retry_after: Optional[float] = None):
        self.circuit_name = circuit_name
        self.retry_after = retry_after
        message = f"Circuit breaker '{circuit_name}' is open"
        if retry_after:
            message += f". Retry after {retry_after:.1f}s"
        super().__init__(message)


class ProviderCircuitBreaker:
    """
    Circuit breaker optimized for provider resilience workflow.

    Renamed from CircuitBreaker to be semantically distinct:
    - CircuitBreaker (victor.providers.circuit_breaker): Standalone with decorator/context manager
    - MultiCircuitBreaker (victor.agent.resilience): Manages multiple named circuits
    - ObservableCircuitBreaker (victor.observability.resilience): Metrics/callback focused
    - ProviderCircuitBreaker (here): ResilientProvider workflow with execute(), is_available

    States:
    - CLOSED: Normal operation, tracking failures
    - OPEN: Too many failures, rejecting requests immediately
    - HALF_OPEN: Testing if service recovered

    Usage:
        cb = ProviderCircuitBreaker("anthropic")

        async def make_request():
            return await provider.chat(messages, model=model)

        try:
            result = await cb.execute(make_request)
        except CircuitOpenError:
            # Circuit is open, use fallback
            pass
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Name for logging and identification
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitBreakerState()
        self._lock = asyncio.Lock()

        logger.debug(
            f"CircuitBreaker '{name}' initialized. "
            f"Threshold: {self.config.failure_threshold}, "
            f"Timeout: {self.config.timeout_seconds}s"
        )

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state.state

    @property
    def failure_count(self) -> int:
        """Current failure count."""
        return self._state.failure_count

    @property
    def is_available(self) -> bool:
        """Check if circuit allows requests."""
        if self._state.state == CircuitState.CLOSED:
            return True

        if self._state.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self._state.last_failure_time:
                elapsed = (datetime.now() - self._state.last_failure_time).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    return True  # Will transition to half-open
            return False

        # Half-open: allow limited calls
        return self._state.half_open_calls < self.config.half_open_max_calls

    @property
    def time_until_retry(self) -> Optional[float]:
        """Seconds until circuit might allow requests."""
        if self._state.state != CircuitState.OPEN:
            return None

        if self._state.last_failure_time:
            elapsed = (datetime.now() - self._state.last_failure_time).total_seconds()
            remaining = self.config.timeout_seconds - elapsed
            return max(0, remaining)

        return None

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs,
    ) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            CircuitOpenError: If circuit is open
        """
        async with self._lock:
            if not self.is_available:
                raise CircuitOpenError(self.name, self.time_until_retry)

            # Transition to half-open if timeout passed
            if self._state.state == CircuitState.OPEN:
                self._transition_to(CircuitState.HALF_OPEN)

            if self._state.state == CircuitState.HALF_OPEN:
                self._state.half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure(e)
            raise

    async def _record_success(self):
        """Record successful call."""
        async with self._lock:
            self._state.consecutive_successes += 1

            if self._state.state == CircuitState.HALF_OPEN:
                self._state.success_count += 1
                if self._state.success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            else:
                # Reset failure count on success in closed state
                self._state.failure_count = max(0, self._state.failure_count - 1)

    async def _record_failure(self, error: Exception):
        """Record failed call."""
        async with self._lock:
            self._state.failure_count += 1
            self._state.last_failure_time = datetime.now()
            self._state.consecutive_successes = 0

            if self._state.state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to(CircuitState.OPEN)
            elif self._state.failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)

            logger.warning(
                f"CircuitBreaker '{self.name}' recorded failure: {error}. "
                f"State: {self._state.state.value}, "
                f"Failures: {self._state.failure_count}/{self.config.failure_threshold}"
            )

    def _transition_to(self, new_state: CircuitState):
        """Transition to new state."""
        old_state = self._state.state
        self._state.state = new_state
        self._state.last_state_change = datetime.now()

        if new_state == CircuitState.HALF_OPEN:
            self._state.half_open_calls = 0
            self._state.success_count = 0
        elif new_state == CircuitState.CLOSED:
            self._state.failure_count = 0
            self._state.success_count = 0

        logger.info(
            f"CircuitBreaker '{self.name}' transitioned: " f"{old_state.value} -> {new_state.value}"
        )

    def reset(self):
        """Reset circuit breaker to initial state."""
        self._state = CircuitBreakerState()
        logger.info(f"CircuitBreaker '{self.name}' reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.state.value,
            "failure_count": self._state.failure_count,
            "success_count": self._state.success_count,
            "consecutive_successes": self._state.consecutive_successes,
            "is_available": self.is_available,
            "time_until_retry": self.time_until_retry,
            "last_failure_time": (
                self._state.last_failure_time.isoformat() if self._state.last_failure_time else None
            ),
            "last_state_change": self._state.last_state_change.isoformat(),
        }


@dataclass
class ProviderRetryConfig:
    """Configuration for provider retry strategy.

    Renamed from RetryConfig to be semantically distinct:
    - ProviderRetryConfig (here): Provider-specific with retryable_patterns
    - AgentRetryConfig (victor.agent.resilience): Agent-specific with jitter flag
    - ObservabilityRetryConfig (victor.observability.resilience): With BackoffStrategy

    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay_seconds: Initial delay between retries
        max_delay_seconds: Maximum delay cap
        exponential_base: Base for exponential backoff
        jitter_factor: Random jitter factor (0.0 to 1.0)
        retryable_exceptions: Exception types that should be retried
        retryable_status_codes: HTTP status codes that should be retried
    """

    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter_factor: float = 0.1

    # Retryable error conditions
    retryable_exceptions: tuple = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    )

    retryable_status_codes: tuple = (429, 500, 502, 503, 504)

    # Retryable error message patterns
    retryable_patterns: tuple = (
        r"rate.?limit",
        r"overloaded",
        r"capacity",
        r"temporarily.?unavailable",
        r"server.?error",
        r"timeout",
    )


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, max_retries: int, last_error: Exception):
        self.max_retries = max_retries
        self.last_error = last_error
        super().__init__(f"Max retries ({max_retries}) exhausted. Last error: {last_error}")


class ProviderRetryStrategy:
    """
    Intelligent provider retry strategy with exponential backoff and jitter.

    Renamed from RetryStrategy to be semantically distinct:
    - ProviderRetryStrategy (here): Concrete provider retry with execute()
    - BaseRetryStrategy (victor.core.retry): Abstract base with should_retry(), get_delay()
    - BatchRetryStrategy (victor.workflows.batch_executor): Enum for batch retry modes

    Features:
    - Exponential backoff with configurable base
    - Random jitter to prevent thundering herd
    - Respects Retry-After headers from rate limit responses
    - Configurable retryable conditions

    Usage:
        retry = ProviderRetryStrategy()

        async def make_request():
            return await provider.chat(messages, model=model)

        result = await retry.execute(make_request)
    """

    def __init__(self, config: Optional[ProviderRetryConfig] = None):
        """Initialize retry strategy.

        Args:
            config: Retry configuration
        """
        self.config = config or ProviderRetryConfig()
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.config.retryable_patterns
        ]

        logger.debug(
            f"ProviderRetryStrategy initialized. "
            f"Max retries: {self.config.max_retries}, "
            f"Base delay: {self.config.base_delay_seconds}s"
        )

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs,
    ) -> T:
        """Execute function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            RetryExhaustedError: If all retries are exhausted
        """
        last_exception: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                # Check if retryable
                if not self._is_retryable(e):
                    logger.debug(f"Non-retryable error: {e}")
                    raise

                # Check if max retries exceeded
                if attempt >= self.config.max_retries:
                    logger.error(
                        f"Max retries ({self.config.max_retries}) exceeded. " f"Last error: {e}"
                    )
                    raise RetryExhaustedError(self.config.max_retries, e)

                # Calculate delay
                delay = self._calculate_delay(attempt, e)

                logger.warning(
                    f"Retry {attempt + 1}/{self.config.max_retries} "
                    f"after {delay:.2f}s. Error: {e}"
                )

                await asyncio.sleep(delay)

        # Should not reach here, but just in case
        if last_exception:
            raise RetryExhaustedError(self.config.max_retries, last_exception)
        raise RuntimeError("Unexpected state in retry loop")

    def _is_retryable(self, error: Exception) -> bool:
        """Check if error is retryable.

        Args:
            error: Exception to check

        Returns:
            True if error should be retried
        """
        # Check exception type
        if isinstance(error, self.config.retryable_exceptions):
            return True

        # Check error message patterns
        error_str = str(error).lower()
        for pattern in self._compiled_patterns:
            if pattern.search(error_str):
                return True

        # Check for status code in error
        for status_code in self.config.retryable_status_codes:
            if str(status_code) in error_str:
                return True

        return False

    def _calculate_delay(self, attempt: int, error: Exception) -> float:
        """Calculate delay before next retry.

        Args:
            attempt: Current attempt number (0-based)
            error: Exception that triggered retry

        Returns:
            Delay in seconds
        """
        # Check for Retry-After header in error
        retry_after = self._extract_retry_after(error)
        if retry_after is not None:
            return min(retry_after, self.config.max_delay_seconds)

        # Exponential backoff
        delay = self.config.base_delay_seconds * (self.config.exponential_base**attempt)

        # Add jitter
        jitter_range = delay * self.config.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)
        delay += jitter

        # Cap at max delay
        return min(max(0, delay), self.config.max_delay_seconds)

    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract Retry-After value from error if present.

        Args:
            error: Exception to check

        Returns:
            Retry-After value in seconds, or None
        """
        error_str = str(error)

        # Look for "retry after X seconds" pattern
        patterns = [
            r"retry[- ]?after[:\s]+(\d+(?:\.\d+)?)",
            r"wait[:\s]+(\d+(?:\.\d+)?)\s*s",
            r"(\d+(?:\.\d+)?)\s*seconds?\s*(?:before|until)",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue

        return None


class ProviderUnavailableError(Exception):
    """Raised when no providers are available."""

    def __init__(self, primary_error: Exception, fallback_errors: Optional[List[Exception]] = None):
        self.primary_error = primary_error
        self.fallback_errors = fallback_errors or []

        errors_summary = [f"Primary: {primary_error}"]
        for i, err in enumerate(self.fallback_errors):
            errors_summary.append(f"Fallback {i + 1}: {err}")

        super().__init__(f"All providers failed. {'; '.join(errors_summary)}")


class ResilientProvider:
    """
    Wrapper that adds resilience features to any provider.

    Features:
    - Circuit breaker per provider
    - Retry with exponential backoff
    - Fallback to alternative providers
    - Request timeout handling
    - Comprehensive error logging

    Usage:
        resilient = ResilientProvider(
            provider=anthropic_provider,
            fallback_providers=[openai_provider, ollama_provider],
        )

        response = await resilient.chat(messages, model=model)
    """

    def __init__(
        self,
        provider: Any,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[ProviderRetryConfig] = None,
        fallback_providers: Optional[List[Any]] = None,
        request_timeout: float = 120.0,
    ):
        """Initialize resilient provider.

        Args:
            provider: Primary provider instance
            circuit_config: Circuit breaker configuration
            retry_config: Retry strategy configuration
            fallback_providers: List of fallback providers
            request_timeout: Request timeout in seconds
        """
        self.provider = provider
        self.fallback_providers = fallback_providers or []
        self.request_timeout = request_timeout

        # Get provider name
        self._provider_name = getattr(provider, "name", "unknown")

        # Create circuit breaker
        self.circuit_breaker = ProviderCircuitBreaker(
            name=f"cb_{self._provider_name}",
            config=circuit_config,
        )

        # Create retry strategy
        self.retry_strategy = ProviderRetryStrategy(config=retry_config)

        # Create circuit breakers for fallbacks
        self._fallback_circuits: Dict[str, ProviderCircuitBreaker] = {}
        for fb in self.fallback_providers:
            fb_name = getattr(fb, "name", f"fallback_{len(self._fallback_circuits)}")
            self._fallback_circuits[fb_name] = ProviderCircuitBreaker(
                name=f"cb_{fb_name}",
                config=circuit_config,
            )

        # Statistics
        self._stats = {
            "total_requests": 0,
            "primary_successes": 0,
            "fallback_successes": 0,
            "total_failures": 0,
            "retry_attempts": 0,
        }

        logger.info(
            f"ResilientProvider initialized for '{self._provider_name}'. "
            f"Fallbacks: {len(self.fallback_providers)}, "
            f"Timeout: {request_timeout}s"
        )

    @property
    def name(self) -> str:
        """Provider name."""
        return self._provider_name

    def supports_tools(self) -> bool:
        """Check if provider supports tools."""
        return getattr(self.provider, "supports_tools", lambda: False)()

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        return getattr(self.provider, "supports_streaming", lambda: False)()

    async def chat(
        self,
        messages: List[Any],
        *,
        model: str,
        **kwargs,
    ) -> Any:
        """Execute chat with resilience features.

        Args:
            messages: List of messages
            model: Model identifier
            **kwargs: Additional arguments for provider

        Returns:
            Chat response

        Raises:
            ProviderUnavailableError: If all providers fail
        """
        self._stats["total_requests"] += 1

        async def _execute_primary():
            return await asyncio.wait_for(
                self.provider.chat(messages, model=model, **kwargs),
                timeout=self.request_timeout,
            )

        # Try primary provider with circuit breaker and retry
        try:
            result = await self.circuit_breaker.execute(
                self.retry_strategy.execute,
                _execute_primary,
            )
            self._stats["primary_successes"] += 1
            return result

        except CircuitOpenError as e:
            logger.warning(f"Primary provider circuit open: {e}")
            primary_error_var = e
        except Exception as e:
            logger.warning(f"Primary provider failed: {e}")
            primary_error: Exception = e

        # Try fallback providers
        fallback_errors: List[Exception] = []

        for fallback in self.fallback_providers:
            fb_name = getattr(fallback, "name", "unknown")
            fb_circuit = self._fallback_circuits.get(fb_name)

            try:
                logger.info(f"Trying fallback provider: {fb_name}")

                async def _execute_fallback(fb=fallback):
                    return await asyncio.wait_for(
                        fb.chat(messages, model=model, **kwargs),
                        timeout=self.request_timeout,
                    )

                if fb_circuit:
                    result = await fb_circuit.execute(_execute_fallback)
                else:
                    result = await _execute_fallback()

                self._stats["fallback_successes"] += 1
                logger.info(f"Fallback provider '{fb_name}' succeeded")
                return result

            except CircuitOpenError as e:
                logger.warning(f"Fallback '{fb_name}' circuit open: {e}")
                fallback_errors.append(e)
            except Exception as e:
                logger.warning(f"Fallback provider '{fb_name}' failed: {e}")
                fallback_errors.append(e)
                continue

        # All providers failed
        self._stats["total_failures"] += 1
        raise ProviderUnavailableError(primary_error or primary_error_var, fallback_errors)

    async def stream(
        self,
        messages: List[Any],
        *,
        model: str,
        **kwargs,
    ):
        """Stream chat with resilience features.

        Note: Streaming has limited retry capability. If the stream fails
        midway, it cannot be resumed.

        Args:
            messages: List of messages
            model: Model identifier
            **kwargs: Additional arguments for provider

        Yields:
            Stream chunks
        """
        self._stats["total_requests"] += 1

        # For streaming, we only retry the initial connection
        async def _start_stream():
            return self.provider.stream(messages, model=model, **kwargs)

        try:
            stream = await self.circuit_breaker.execute(
                self.retry_strategy.execute,
                _start_stream,
            )

            async for chunk in stream:
                yield chunk

            self._stats["primary_successes"] += 1

        except CircuitOpenError:
            # Try fallbacks for streaming
            for fallback in self.fallback_providers:
                fb_name = getattr(fallback, "name", "unknown")
                try:
                    logger.info(f"Trying fallback stream: {fb_name}")
                    stream = fallback.stream(messages, model=model, **kwargs)
                    async for chunk in stream:
                        yield chunk
                    self._stats["fallback_successes"] += 1
                    return
                except Exception as e:
                    logger.warning(f"Fallback stream '{fb_name}' failed: {e}")
                    continue

            self._stats["total_failures"] += 1
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get resilience statistics."""
        return {
            **self._stats,
            "primary_circuit": self.circuit_breaker.get_stats(),
            "fallback_circuits": {
                name: cb.get_stats() for name, cb in self._fallback_circuits.items()
            },
            "success_rate": (
                (self._stats["primary_successes"] + self._stats["fallback_successes"])
                / self._stats["total_requests"]
                if self._stats["total_requests"] > 0
                else 0
            ),
        }

    def reset_circuits(self):
        """Reset all circuit breakers."""
        self.circuit_breaker.reset()
        for cb in self._fallback_circuits.values():
            cb.reset()
        logger.info("All circuit breakers reset")
