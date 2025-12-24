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

"""Framework-layer facade for resilience patterns.

This module provides a unified access point to resilience components from
the framework layer. It re-exports existing implementations from core modules
without duplicating code, following the Facade Pattern.

Delegated modules:
- victor.providers.circuit_breaker: Standalone circuit breaker with decorator support
- victor.providers.resilience: ResilientProvider with circuit breaker + retry + fallback
- victor.core.retry: Unified retry strategies (exponential, linear, fixed)

Design Pattern: Facade
- Single import point for framework users
- No code duplication - pure re-exports
- Maintains backward compatibility with original modules
- Enables discovery of resilience capabilities through framework namespace

Example:
    from victor.framework import (
        # Circuit Breaker
        CircuitBreaker,
        CircuitBreakerRegistry,
        CircuitState,

        # Resilient Provider (complete solution)
        ResilientProvider,
        CircuitBreakerConfig,
        RetryConfig,

        # Retry Strategies
        ExponentialBackoffStrategy,
        LinearBackoffStrategy,
        with_retry,
    )

    # Use standalone circuit breaker
    @CircuitBreaker(failure_threshold=5)
    async def call_api():
        ...

    # Use retry decorator
    @with_retry(ExponentialBackoffStrategy(max_attempts=3))
    async def retriable_operation():
        ...

    # Use resilient provider (combines everything)
    provider = ResilientProvider(
        primary_provider=anthropic_provider,
        fallback_providers=[openai_provider],
        circuit_config=CircuitBreakerConfig(failure_threshold=5),
        retry_config=RetryConfig(max_retries=3),
    )
"""

from __future__ import annotations

# =============================================================================
# Circuit Breaker (Standalone)
# From: victor/providers/circuit_breaker.py
# =============================================================================
from victor.providers.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState,
)

# =============================================================================
# Resilient Provider (Circuit Breaker + Retry + Fallback)
# From: victor/providers/resilience.py
# =============================================================================
from victor.providers.resilience import (
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitOpenError,
    ProviderUnavailableError,
    ResilientProvider,
    RetryConfig,
    RetryExhaustedError,
    RetryStrategy as ResilientRetryStrategy,  # Renamed to avoid conflict with unified
)

# =============================================================================
# Unified Retry Strategies
# From: victor/core/retry.py
# =============================================================================
from victor.core.retry import (
    ExponentialBackoffStrategy,
    FixedDelayStrategy,
    LinearBackoffStrategy,
    NoRetryStrategy,
    RetryContext,
    RetryExecutor,
    RetryOutcome,
    RetryResult,
    RetryStrategy,
    connection_retry_strategy,
    provider_retry_strategy,
    tool_retry_strategy,
    with_retry,
    with_retry_sync,
)

__all__ = [
    # Circuit Breaker (Standalone)
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitBreakerRegistry",
    "CircuitState",
    # Resilient Provider
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "CircuitOpenError",
    "ProviderUnavailableError",
    "ResilientProvider",
    "RetryConfig",
    "RetryExhaustedError",
    "ResilientRetryStrategy",
    # Unified Retry Strategies
    "ExponentialBackoffStrategy",
    "FixedDelayStrategy",
    "LinearBackoffStrategy",
    "NoRetryStrategy",
    "RetryContext",
    "RetryExecutor",
    "RetryOutcome",
    "RetryResult",
    "RetryStrategy",
    "connection_retry_strategy",
    "provider_retry_strategy",
    "tool_retry_strategy",
    "with_retry",
    "with_retry_sync",
]
