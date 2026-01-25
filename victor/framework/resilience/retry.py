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

"""Framework-level retry handler for workflows.

This module provides a retry handler that can be used in YAML workflows
and directly in code. It re-exports the core retry strategies from
victor.core.retry and provides workflow-specific handlers.

Design Pattern: Facade + Handler
- Re-exports BaseRetryStrategy implementations from core
- Provides RetryHandler for workflow compute nodes
- Enables YAML workflow integration via handler registration

Usage in YAML workflows:
    - id: fetch_with_retry
      type: compute
      handler: retry_with_backoff
      tools: [call_external_api]
      config:
        max_retries: 5
        base_delay: 2.0
        max_delay: 60.0

Usage in code:
    from victor.framework.resilience.retry import (
        retry_with_backoff,
        ExponentialBackoffStrategy,
        RetryHandler,
    )
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    Set,
    Type,
    TypeVar,
    cast,
)

# Re-export core retry types (canonical source)
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

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Framework-Specific Retry Handler for Workflows
# =============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry handler in workflows.

    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay cap in seconds
        exponential_base: Multiplier for exponential backoff
        jitter_factor: Random jitter factor (0.0 to 1.0)
        retryable_exceptions: Exception types that should be retried
        non_retryable_exceptions: Exception types to never retry
        retryable_patterns: Error message patterns to retry
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter_factor: float = 0.1

    retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
    )

    non_retryable_exceptions: tuple[type[BaseException], ...] = ()

    retryable_patterns: tuple[str, ...] = (
        r"rate.?limit",
        r"overloaded",
        r"capacity",
        r"temporarily.?unavailable",
        r"server.?error",
        r"timeout",
        r"connection",
        r"network",
    )

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "RetryConfig":
        """Create RetryConfig from dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            RetryConfig instance
        """
        return cls(
            max_retries=config.get("max_retries", 3),
            base_delay=config.get("base_delay", 1.0),
            max_delay=config.get("max_delay", 60.0),
            exponential_base=config.get("exponential_base", 2.0),
            jitter_factor=config.get("jitter_factor", 0.1),
        )


@dataclass
class RetryHandlerConfig:
    """Configuration for RetryHandler used in workflow compute nodes.

    Attributes:
        retry_config: Core retry configuration
        fail_fast: If True, stop workflow on final failure
        log_attempts: If True, log each retry attempt
        backoff_strategy: Strategy type to use ("exponential", "linear", "fixed")
    """

    retry_config: RetryConfig = field(default_factory=RetryConfig)
    fail_fast: bool = False
    log_attempts: bool = True
    backoff_strategy: str = "exponential"


class RetryHandler:
    """Workflow compute handler for retry with exponential backoff.

    This handler wraps tool execution with retry logic, using
    exponential backoff between attempts. It integrates with
    the workflow system as a compute handler.

    Example:
        from victor.workflows.executor import register_compute_handler
        from victor.framework.resilience.retry import RetryHandler

        register_compute_handler("retry_with_backoff", RetryHandler())

    In YAML:
        - id: fetch_api
          type: compute
          handler: retry_with_backoff
          tools: [call_external_api]
          config:
            max_retries: 5
            base_delay: 2.0
    """

    def __init__(
        self,
        config: Optional[RetryHandlerConfig] = None,
    ):
        """Initialize retry handler.

        Args:
            config: Handler configuration
        """
        self._config = config or RetryHandlerConfig()
        self._strategy_cache: dict[str, BaseRetryStrategy] = {}

    def _get_strategy(self, strategy_type: str) -> BaseRetryStrategy:
        """Get or create retry strategy.

        Args:
            strategy_type: Strategy type name

        Returns:
            BaseRetryStrategy instance
        """
        if strategy_type not in self._strategy_cache:
            retry_cfg = self._config.retry_config
            if strategy_type == "exponential":
                self._strategy_cache[strategy_type] = ExponentialBackoffStrategy(
                    max_attempts=retry_cfg.max_retries + 1,
                    base_delay=retry_cfg.base_delay,
                    max_delay=retry_cfg.max_delay,
                    multiplier=retry_cfg.exponential_base,
                    jitter=retry_cfg.jitter_factor,
                    non_retryable_exceptions=set(retry_cfg.non_retryable_exceptions) if retry_cfg.non_retryable_exceptions else None,
                )
            elif strategy_type == "linear":
                self._strategy_cache[strategy_type] = LinearBackoffStrategy(
                    max_attempts=retry_cfg.max_retries + 1,
                    base_delay=retry_cfg.base_delay,
                    increment=retry_cfg.base_delay,
                    max_delay=retry_cfg.max_delay,
                )
            elif strategy_type == "fixed":
                self._strategy_cache[strategy_type] = FixedDelayStrategy(
                    max_attempts=retry_cfg.max_retries + 1,
                    delay=retry_cfg.base_delay,
                )
            else:
                # Default to exponential
                return self._get_strategy("exponential")

        return self._strategy_cache[strategy_type]

    async def __call__(
        self,
        node: Any,  # ComputeNode from workflows.definition
        context: Any,  # WorkflowContext from workflows.executor
        tool_registry: Any,  # ToolRegistry from tools.registry
    ) -> Any:  # NodeResult from workflows.executor
        """Execute tools with retry logic.

        Args:
            node: Workflow compute node
            context: Workflow execution context
            tool_registry: Tool registry for execution

        Returns:
            NodeResult with execution outcome
        """
        from victor.workflows.executor import NodeResult, ExecutorNodeStatus

        start_time = time.time()
        outputs: dict[str, Any] = {}
        errors: list[str] = []
        tool_calls_used = 0

        # Get config from node if present
        node_config = self._extract_node_config(node)
        strategy = self._get_strategy(node_config.get("backoff_strategy", "exponential"))

        # Build params from input mapping
        params = self._build_params(node, context)

        for tool_name in node.tools:
            if not node.constraints.allows_tool(tool_name):
                continue

            last_error = None
            attempt = 0
            max_attempts = self._config.retry_config.max_retries + 1

            while attempt < max_attempts:
                try:
                    if self._config.log_attempts and attempt > 0:
                        logger.info(
                            f"Retry attempt {attempt}/{max_attempts - 1} " f"for tool '{tool_name}'"
                        )

                    result = await asyncio.wait_for(
                        tool_registry.execute(
                            tool_name,
                            _exec_ctx={
                                "workflow_context": context.data,
                                "constraints": node.constraints.to_dict(),
                                "retry_attempt": attempt,
                            },
                            **params,
                        ),
                        timeout=node.constraints.timeout,
                    )
                    tool_calls_used += 1

                    if result.success:
                        outputs[tool_name] = result.output
                        break
                    else:
                        last_error = result.error
                        if not self._is_retryable_error(result.error):
                            # Non-retryable error, stop immediately
                            attempt = max_attempts
                            break

                except asyncio.TimeoutError:
                    last_error = f"Timed out after {node.constraints.timeout}s"

                except Exception as e:
                    last_error = str(e)
                    if not self._is_retryable_exception(type(e)):
                        # Non-retryable exception, stop immediately
                        attempt = max_attempts
                        break

                # Retry logic
                attempt += 1
                if attempt < max_attempts:
                    delay = strategy.get_delay(
                        RetryContext(
                            attempt=attempt,
                            max_attempts=max_attempts,
                        )
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)

            if last_error and not outputs.get(tool_name):
                errors.append(f"{tool_name}: {last_error}")

        # Store outputs
        if node.output_key:
            context.set(node.output_key, outputs)

        # Determine status
        if errors and self._config.fail_fast:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                output=outputs,
                error="; ".join(errors),
                duration_seconds=time.time() - start_time,
                tool_calls_used=tool_calls_used,
            )

        return NodeResult(
            node_id=node.id,
            status=(
                ExecutorNodeStatus.COMPLETED if outputs or not errors else ExecutorNodeStatus.FAILED
            ),
            output=outputs,
            error="; ".join(errors) if errors else None,
            duration_seconds=time.time() - start_time,
            tool_calls_used=tool_calls_used,
        )

    def _extract_node_config(self, node: Any) -> dict[str, Any]:
        """Extract config from workflow node.

        Args:
            node: Compute node

        Returns:
            Configuration dictionary
        """
        config = {}
        if hasattr(node, "config") and isinstance(node.config, dict):
            config = node.config.copy()
        return config

    def _build_params(self, node: Any, context: Any) -> dict[str, Any]:
        """Build tool parameters from context.

        Args:
            node: Compute node
            context: Workflow context

        Returns:
            Parameter dictionary
        """
        params = {}
        for param_name, context_key in node.input_mapping.items():
            value = context.get(context_key)
            if value is not None:
                params[param_name] = value
            else:
                params[param_name] = context_key
        return params

    def _is_retryable_error(self, error: Optional[str]) -> bool:
        """Check if error message indicates retryable condition.

        Args:
            error: Error message

        Returns:
            True if error is retryable
        """
        if not error:
            return True

        import re

        error_lower = error.lower()
        for pattern in self._config.retry_config.retryable_patterns:
            if isinstance(pattern, str):
                # Handle regex patterns (strings starting with r" or containing .? etc)
                # For simple strings, do substring match
                # For regex patterns, use regex matching
                try:
                    # Remove r prefix if present and escape the pattern for literal matching
                    # or use as regex if it looks like a regex pattern
                    clean_pattern = pattern.lower()
                    if (
                        "?)" in clean_pattern
                        or "?." in clean_pattern
                        or clean_pattern.startswith("^")
                    ):
                        # Use regex matching
                        if re.search(clean_pattern, error_lower):
                            return True
                    else:
                        # Simple substring match (remove common regex markers)
                        simple_pattern = clean_pattern.replace('r"', "").replace(".?", "")
                        if simple_pattern in error_lower:
                            return True
                except (re.error, ValueError):
                    # Fallback to substring matching
                    if pattern.lower() in error_lower:
                        return True

        return False

    def _is_retryable_exception(self, exc_type: type) -> bool:
        """Check if exception type is retryable.

        Args:
            exc_type: Exception type

        Returns:
            True if exception should be retried
        """
        # Check non-retryable first
        if exc_type in self._config.retry_config.non_retryable_exceptions:
            return False

        # Check retryable
        if exc_type in self._config.retry_config.retryable_exceptions:
            return True

        # Default to retry for unknown exceptions
        return True


# =============================================================================
# Standalone retry_with_backoff handler function
# =============================================================================


async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: float = 0.1,
    retryable_exceptions: Optional[Set[Type[Exception]]] = None,
    **kwargs: Any,
) -> T:
    """Execute async function with exponential backoff retry.

    This is a standalone async function for use in code (not YAML workflows).

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Exponential multiplier
        jitter: Random jitter factor
        retryable_exceptions: Only retry these exception types
        **kwargs: Keyword arguments for func

    Returns:
        Result of func

    Raises:
        Exception: The last exception if all retries exhausted

    Example:
        result = await retry_with_backoff(
            api_client.call,
            "https://api.example.com/data",
            max_retries=5,
            base_delay=2.0,
        )
    """
    strategy = ExponentialBackoffStrategy(
        max_attempts=max_retries + 1,
        base_delay=base_delay,
        max_delay=max_delay,
        multiplier=exponential_base,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions,
    )

    executor = RetryExecutor(strategy)
    result = await executor.execute_async(func, *args, **kwargs)

    if result.success:
        return cast(T, result.result)
    elif result.exception:
        raise result.exception
    else:
        raise RuntimeError("Retry failed with unknown error")


def retry_with_backoff_sync(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: float = 0.1,
    retryable_exceptions: Optional[Set[Type[Exception]]] = None,
    **kwargs: Any,
) -> T:
    """Execute sync function with exponential backoff retry.

    This is a standalone sync function for use in code (not YAML workflows).

    Args:
        func: Sync function to execute
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Exponential multiplier
        jitter: Random jitter factor
        retryable_exceptions: Only retry these exception types
        **kwargs: Keyword arguments for func

    Returns:
        Result of func

    Raises:
        Exception: The last exception if all retries exhausted

    Example:
        result = retry_with_backoff_sync(
            file_operation,
            "/path/to/file",
            max_retries=3,
        )
    """
    strategy = ExponentialBackoffStrategy(
        max_attempts=max_retries + 1,
        base_delay=base_delay,
        max_delay=max_delay,
        multiplier=exponential_base,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions,
    )

    executor = RetryExecutor(strategy)
    result = executor.execute_sync(func, *args, **kwargs)

    if result.success:
        return cast(T, result.result)
    elif result.exception:
        raise result.exception
    else:
        raise RuntimeError("Retry failed with unknown error")


# =============================================================================
# Decorators for easy retry application
# =============================================================================


def with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: float = 0.1,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for async functions with exponential backoff retry.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Exponential multiplier
        jitter: Random jitter factor

    Usage:
        @with_exponential_backoff(max_retries=5)
        async def fetch_data(url: str) -> dict:
            return await http_client.get(url)
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry_with_backoff(
                func,
                *args,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter,
                **kwargs,
            )

        return wrapper

    return decorator


def with_exponential_backoff_sync(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: float = 0.1,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for sync functions with exponential backoff retry.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Exponential multiplier
        jitter: Random jitter factor

    Usage:
        @with_exponential_backoff_sync(max_retries=3)
        def read_file(path: str) -> str:
            return Path(path).read_text()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return retry_with_backoff_sync(
                func,
                *args,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter,
                **kwargs,
            )

        return wrapper

    return decorator


# =============================================================================
# Specialized retry handlers for common use cases
# =============================================================================


class NetworkRetryHandler(RetryHandler):
    """Retry handler optimized for network operations.

    Features:
    - More retries (default 5)
    - Longer delays for network latency
    - Higher jitter for distributed thundering herd prevention
    - Specific network exception handling

    Example:
        register_compute_handler("network_retry", NetworkRetryHandler())

    In YAML:
        - id: fetch_remote
          type: compute
          handler: network_retry
          tools: [http_get]
    """

    def __init__(self, config: Optional[RetryHandlerConfig] = None):
        if config is None:
            config = RetryHandlerConfig(
                retry_config=RetryConfig(
                    max_retries=5,
                    base_delay=2.0,
                    max_delay=120.0,
                    jitter_factor=0.25,
                ),
            )
        super().__init__(config)


class RateLimitRetryHandler(RetryHandler):
    """Retry handler optimized for rate-limited APIs.

    Features:
    - Longer base delay to respect rate limits
    - Detects rate limit error messages
    - Exponential backoff with high multiplier

    Example:
        register_compute_handler("rate_limit_retry", RateLimitRetryHandler())

    In YAML:
        - id: api_call
          type: compute
          handler: rate_limit_retry
          tools: [call_openai_api]
    """

    def __init__(self, config: Optional[RetryHandlerConfig] = None):
        if config is None:
            config = RetryHandlerConfig(
                retry_config=RetryConfig(
                    max_retries=4,
                    base_delay=5.0,
                    max_delay=300.0,
                    exponential_base=2.5,
                    jitter_factor=0.2,
                    retryable_patterns=(
                        r"rate.?limit",
                        r"429",
                        r"too.?many.?requests",
                        r"quota",
                    ),
                ),
            )
        super().__init__(config)


class DatabaseRetryHandler(RetryHandler):
    """Retry handler optimized for database operations.

    Features:
    - Moderate retries (transaction conflicts)
    - Shorter delays (quick conflict resolution)
    - Database-specific exception patterns

    Example:
        register_compute_handler("database_retry", DatabaseRetryHandler())

    In YAML:
        - id: query_db
          type: compute
          handler: database_retry
          tools: [execute_query]
    """

    def __init__(self, config: Optional[RetryHandlerConfig] = None):
        if config is None:
            config = RetryHandlerConfig(
                retry_config=RetryConfig(
                    max_retries=3,
                    base_delay=0.5,
                    max_delay=10.0,
                    exponential_base=2.0,
                    jitter_factor=0.15,
                    retryable_patterns=(
                        r"deadlock",
                        r"lock",
                        r"conflict",
                        r"constraint",
                    ),
                ),
            )
        super().__init__(config)


# =============================================================================
# Framework handler instances for registration
# =============================================================================

FRAMEWORK_RETRY_HANDLERS: dict[str, RetryHandler] = {
    "retry_with_backoff": RetryHandler(),
    "network_retry": NetworkRetryHandler(),
    "rate_limit_retry": RateLimitRetryHandler(),
    "database_retry": DatabaseRetryHandler(),
}


def register_framework_retry_handlers() -> None:
    """Register all framework retry handlers with workflow executor.

    Call this during application initialization to make retry handlers
    available in YAML workflows.

    Example:
        from victor.framework.resilience.retry import register_framework_retry_handlers
        register_framework_retry_handlers()
    """
    try:
        from victor.workflows.executor import register_compute_handler

        for name, handler in FRAMEWORK_RETRY_HANDLERS.items():
            register_compute_handler(name, handler)
            logger.debug(f"Registered retry handler: {name}")
    except ImportError:
        logger.warning("Workflow executor not available, skipping handler registration")


__all__ = [
    # Re-exported from core.retry
    "BaseRetryStrategy",
    "ExponentialBackoffStrategy",
    "FixedDelayStrategy",
    "LinearBackoffStrategy",
    "NoRetryStrategy",
    "RetryContext",
    "RetryExecutor",
    "RetryOutcome",
    "RetryResult",
    "connection_retry_strategy",
    "provider_retry_strategy",
    "tool_retry_strategy",
    "with_retry",
    "with_retry_sync",
    # Framework-specific
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
