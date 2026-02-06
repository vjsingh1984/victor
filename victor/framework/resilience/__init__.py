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

"""Framework resilience module with retry handlers and circuit breakers.

This module provides a unified access point to resilience components from
the framework layer. It re-exports existing implementations from core modules
and provides new workflow-specific retry handlers.

Delegated modules:
- victor.providers.circuit_breaker: Standalone circuit breaker with decorator support
- victor.providers.resilience: ResilientProvider with circuit breaker + retry + fallback
- victor.core.retry: Unified retry strategies (exponential, linear, fixed)
- victor.framework.resilience.retry: New workflow retry handlers

Quick Start:
    from victor.framework.resilience import (
        # Circuit Breaker
        CircuitBreaker,
        # Retry Strategies
        ExponentialBackoffStrategy,
        with_retry,
        # Workflow Handlers
        RetryHandler,
        retry_with_backoff,
    )

    # Use standalone circuit breaker
    @CircuitBreaker(failure_threshold=5)
    async def call_api():
        ...

    # Use retry decorator
    @with_retry(ExponentialBackoffStrategy(max_attempts=3))
    async def retriable_operation():
        ...

    # Use retry handler in code
    result = await retry_with_backoff(api_call, max_retries=5)

    # In YAML workflows:
    #   - id: fetch_with_retry
    #     type: compute
    #     handler: retry_with_backoff
    #     tools: [http_get]
"""

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
from victor.providers.resilience import (  # type: ignore[attr-defined]
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitOpenError,
    ProviderRetryConfig,
    ProviderRetryStrategy,
    ProviderUnavailableError,
    ResilientProvider,
    RetryExhaustedError,
)

# =============================================================================
# Unified Retry Strategies
# From: victor/core/retry.py
# =============================================================================
from victor.core.retry import (
    BaseRetryStrategy,
    ExponentialBackoffStrategy,
    FixedDelayStrategy,
    LinearBackoffStrategy,
    NoRetryStrategy,
    RetryContext,
    RetryExecutor,
    RetryOutcome,
    RetryResult,
    connection_retry_strategy,
    provider_retry_strategy,
    tool_retry_strategy,
    with_retry,
    with_retry_sync,
)

# =============================================================================
# Framework-Specific Retry Handlers
# From: victor/framework/resilience/retry.py
# =============================================================================
from victor.framework.resilience.retry import (
    DatabaseRetryHandler,
    FRAMEWORK_RETRY_HANDLERS,
    NetworkRetryHandler,
    RateLimitRetryHandler,
    register_framework_retry_handlers,
    retry_with_backoff,
    retry_with_backoff_sync,
    RetryConfig,
    RetryHandler,
    RetryHandlerConfig,
    with_exponential_backoff,
    with_exponential_backoff_sync,
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
    "ProviderRetryConfig",
    "RetryExhaustedError",
    "ProviderRetryStrategy",
    # Unified Retry Strategies
    "ExponentialBackoffStrategy",
    "FixedDelayStrategy",
    "LinearBackoffStrategy",
    "NoRetryStrategy",
    "RetryContext",
    "RetryExecutor",
    "RetryOutcome",
    "RetryResult",
    "BaseRetryStrategy",
    "connection_retry_strategy",
    "provider_retry_strategy",
    "tool_retry_strategy",
    "with_retry",
    "with_retry_sync",
    # Framework-Specific Retry Handlers
    "RetryConfig",
    "RetryHandlerConfig",
    "RetryHandler",
    "retry_with_backoff",
    "retry_with_backoff_sync",
    "with_exponential_backoff",
    "with_exponential_backoff_sync",
    # Specialized handlers
    "NetworkRetryHandler",
    "RateLimitRetryHandler",
    "DatabaseRetryHandler",
    "FRAMEWORK_RETRY_HANDLERS",
    "register_framework_retry_handlers",
]
