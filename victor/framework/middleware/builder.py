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

"""Middleware Builder - Fluent API for middleware composition.

This module provides a builder pattern for constructing middleware chains,
eliminating code duplication across verticals.

Design Philosophy:
- Builder Pattern: Fluent API for middleware composition
- Presets: Pre-configured middleware sets for common use cases
- Extensible: Easy to add custom middleware
- Type-Safe: Protocol-based middleware interface

Usage:
    # Build a middleware chain
    chain = (
        MiddlewareBuilder()
        .with_logging()
        .with_git_safety(block_dangerous=False)
        .with_secret_masking()
        .with_metrics()
        .build()
    )

    # Use presets
    chain = MiddlewareBuilder.standard()  # Logging + GitSafety + SecretMasking
    chain = MiddlewareBuilder.safe()  # All safety middleware
    chain = MiddlewareBuilder.production()  # All middleware with metrics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from victor.framework.middleware import (
        GitSafetyMiddleware,
        LoggingMiddleware,
        MetricsMiddleware,
        SecretMaskingMiddleware,
        ValidationMiddleware,
        SafetyCheckMiddleware,
        OutputValidationMiddleware,
        CacheMiddleware,
        RateLimitMiddleware,
    )

logger = logging.getLogger(__name__)


# =============================================================================
# Middleware Protocol
# =============================================================================


class MiddlewareProtocol(Protocol):
    """Protocol for middleware components.

    All middleware implementations must implement this protocol.
    """

    async def process_before(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, Optional[str], Dict[str, Any]]:
        """Process before tool execution.

        Args:
            tool_name: Name of tool being executed
            arguments: Tool arguments
            context: Optional execution context

        Returns:
            Tuple of (should_proceed, error_message, modified_arguments)
        """
        ...

    async def process_after(
        self,
        tool_name: str,
        result: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, Optional[str], Any]:
        """Process after tool execution.

        Args:
            tool_name: Name of tool that was executed
            result: Tool execution result
            context: Optional execution context

        Returns:
            Tuple of (should_continue, error_message, modified_result)
        """
        ...

    @property
    def enabled(self) -> bool:
        """Check if middleware is enabled."""
        ...

    def enable(self) -> None:
        """Enable the middleware."""
        ...

    def disable(self) -> None:
        """Disable the middleware."""
        ...

    @property
    def name(self) -> str:
        """Get middleware name."""
        ...


# =============================================================================
# Middleware Chain
# =============================================================================


@dataclass
class MiddlewareChain:
    """A chain of middleware components.

    Executes middleware in order, with before/after hooks.

    Attributes:
        middleware: List of middleware in execution order
        stop_on_first_error: Whether to stop on first middleware error
    """

    middleware: List[MiddlewareProtocol] = field(default_factory=list)
    stop_on_first_error: bool = True

    async def process_before(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, Optional[str], Dict[str, Any]]:
        """Process all middleware before tool execution.

        Args:
            tool_name: Name of tool being executed
            arguments: Tool arguments
            context: Optional execution context

        Returns:
            Tuple of (should_proceed, error_message, modified_arguments)
        """
        current_args = arguments
        error_message = None

        for mw in self.middleware:
            if not mw.enabled:
                continue

            try:
                should_proceed, error, current_args = await mw.process_before(
                    tool_name, current_args, context
                )

                if not should_proceed:
                    if error:
                        error_message = error
                    if self.stop_on_first_error:
                        return False, error_message, current_args

            except Exception as e:
                logger.error(f"Middleware {mw.name} process_before failed: {e}")
                if self.stop_on_first_error:
                    return False, f"Middleware error: {e}", current_args

        return True, None, current_args

    async def process_after(
        self,
        tool_name: str,
        result: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, Optional[str], Any]:
        """Process all middleware after tool execution.

        Args:
            tool_name: Name of tool that was executed
            result: Tool execution result
            context: Optional execution context

        Returns:
            Tuple of (should_continue, error_message, modified_result)
        """
        current_result = result
        error_message = None

        for mw in reversed(self.middleware):  # Reverse order for after hooks
            if not mw.enabled:
                continue

            try:
                should_continue, error, current_result = await mw.process_after(
                    tool_name, current_result, context
                )

                if not should_continue:
                    if error:
                        error_message = error
                    if self.stop_on_first_error:
                        return False, error_message, current_result

            except Exception as e:
                logger.error(f"Middleware {mw.name} process_after failed: {e}")
                if self.stop_on_first_error:
                    return False, f"Middleware error: {e}", current_result

        return True, None, current_result

    def add(self, middleware: MiddlewareProtocol) -> "MiddlewareChain":
        """Add middleware to the chain.

        Args:
            middleware: Middleware to add

        Returns:
            Self for fluent chaining
        """
        self.middleware.append(middleware)
        return self

    def remove(self, name: str) -> bool:
        """Remove middleware by name.

        Args:
            name: Name of middleware to remove

        Returns:
            True if middleware was removed
        """
        for i, mw in enumerate(self.middleware):
            if mw.name == name:
                self.middleware.pop(i)
                return True
        return False

    def enable(self, name: str) -> bool:
        """Enable middleware by name.

        Args:
            name: Name of middleware to enable

        Returns:
            True if middleware was enabled
        """
        for mw in self.middleware:
            if mw.name == name:
                mw.enable()
                return True
        return False

    def disable(self, name: str) -> bool:
        """Disable middleware by name.

        Args:
            name: Name of middleware to disable

        Returns:
            True if middleware was disabled
        """
        for mw in self.middleware:
            if mw.name == name:
                mw.disable()
                return True
        return False

    def get_names(self) -> List[str]:
        """Get names of all middleware in chain.

        Returns:
            List of middleware names
        """
        return [mw.name for mw in self.middleware]


# =============================================================================
# Middleware Builder
# =============================================================================


class MiddlewareBuilder:
    """Builder for constructing middleware chains.

    Provides a fluent API for building middleware chains with presets.

    Example:
        chain = (
            MiddlewareBuilder()
            .with_logging()
            .with_git_safety(block_dangerous=False)
            .with_secret_masking()
            .build()
        )
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._middleware_factories: List[Callable[[], MiddlewareProtocol]] = []
        self._stop_on_first_error: bool = True

    # =====================================================================
    # Fluent API
    # =====================================================================

    def with_logging(self, level: int = logging.INFO) -> "MiddlewareBuilder":
        """Add logging middleware.

        Args:
            level: Logging level

        Returns:
            Self for fluent chaining
        """
        def factory() -> MiddlewareProtocol:
            from victor.framework.middleware import LoggingMiddleware

            return LoggingMiddleware(level=level)

        self._middleware_factories.append(factory)
        return self

    def with_git_safety(
        self,
        block_dangerous: bool = False,
        warn_on_risky: bool = True,
    ) -> "MiddlewareBuilder":
        """Add git safety middleware.

        Args:
            block_dangerous: Whether to block dangerous operations
            warn_on_risky: Whether to warn on risky operations

        Returns:
            Self for fluent chaining
        """
        def factory() -> MiddlewareProtocol:
            from victor.framework.middleware import GitSafetyMiddleware

            return GitSafetyMiddleware(
                block_dangerous=block_dangerous,
                warn_on_risky=warn_on_risky,
            )

        self._middleware_factories.append(factory)
        return self

    def with_secret_masking(self, mask_char: str = "***") -> "MiddlewareBuilder":
        """Add secret masking middleware.

        Args:
            mask_char: Character to use for masking

        Returns:
            Self for fluent chaining
        """
        def factory() -> MiddlewareProtocol:
            from victor.framework.middleware import SecretMaskingMiddleware

            return SecretMaskingMiddleware(mask_char=mask_char)

        self._middleware_factories.append(factory)
        return self

    def with_metrics(self, collect_tool_usage: bool = True) -> "MiddlewareBuilder":
        """Add metrics collection middleware.

        Args:
            collect_tool_usage: Whether to collect tool usage metrics

        Returns:
            Self for fluent chaining
        """
        def factory() -> MiddlewareProtocol:
            from victor.framework.middleware import MetricsMiddleware

            return MetricsMiddleware(collect_tool_usage=collect_tool_usage)

        self._middleware_factories.append(factory)
        return self

    def with_validation(
        self,
        validate_inputs: bool = True,
        validate_outputs: bool = True,
    ) -> "MiddlewareBuilder":
        """Add validation middleware.

        Args:
            validate_inputs: Whether to validate inputs
            validate_outputs: Whether to validate outputs

        Returns:
            Self for fluent chaining
        """
        def factory() -> MiddlewareProtocol:
            from victor.framework.middleware import ValidationMiddleware

            return ValidationMiddleware(
                validate_inputs=validate_inputs,
                validate_outputs=validate_outputs,
            )

        self._middleware_factories.append(factory)
        return self

    def with_cache(
        self,
        ttl: int = 300,
        max_size: int = 1000,
    ) -> "MiddlewareBuilder":
        """Add caching middleware.

        Args:
            ttl: Time to live for cache entries (seconds)
            max_size: Maximum cache size

        Returns:
            Self for fluent chaining
        """
        def factory() -> MiddlewareProtocol:
            from victor.framework.middleware import CacheMiddleware

            return CacheMiddleware(ttl=ttl, max_size=max_size)

        self._middleware_factories.append(factory)
        return self

    def with_rate_limit(
        self,
        max_calls: int = 100,
        window: int = 60,
    ) -> "MiddlewareBuilder":
        """Add rate limiting middleware.

        Args:
            max_calls: Maximum calls per window
            window: Time window in seconds

        Returns:
            Self for fluent chaining
        """
        def factory() -> MiddlewareProtocol:
            from victor.framework.middleware import RateLimitMiddleware

            return RateLimitMiddleware(max_calls=max_calls, window=window)

        self._middleware_factories.append(factory)
        return self

    def with_custom(
        self,
        middleware_factory: Callable[[], MiddlewareProtocol],
    ) -> "MiddlewareBuilder":
        """Add custom middleware.

        Args:
            middleware_factory: Factory function for middleware

        Returns:
            Self for fluent chaining
        """
        self._middleware_factories.append(middleware_factory)
        return self

    def stop_on_error(self, stop: bool = True) -> "MiddlewareBuilder":
        """Configure error handling behavior.

        Args:
            stop: Whether to stop on first error

        Returns:
            Self for fluent chaining
        """
        self._stop_on_first_error = stop
        return self

    # =====================================================================
    # Build Methods
    # =====================================================================

    def build(self) -> MiddlewareChain:
        """Build the middleware chain.

        Returns:
            MiddlewareChain with configured middleware
        """
        chain = MiddlewareChain(stop_on_first_error=self._stop_on_first_error)

        for factory in self._middleware_factories:
            try:
                middleware = factory()
                chain.add(middleware)
            except Exception as e:
                logger.warning(f"Failed to create middleware: {e}")

        return chain

    # =====================================================================
    # Presets
    # =====================================================================

    @classmethod
    def minimal(cls) -> "MiddlewareBuilder":
        """Minimal preset: Only essential middleware.

        Returns:
            Builder configured with minimal middleware
        """
        return cls().with_logging()

    @classmethod
    def standard(cls) -> "MiddlewareBuilder":
        """Standard preset: Common middleware for development.

        Returns:
            Builder configured with standard middleware
        """
        return (
            cls()
            .with_logging()
            .with_git_safety(block_dangerous=False)
            .with_secret_masking()
        )

    @classmethod
    def safe(cls) -> "MiddlewareBuilder":
        """Safe preset: All safety-related middleware.

        Returns:
            Builder configured with safety middleware
        """
        return (
            cls()
            .with_git_safety(block_dangerous=True, warn_on_risky=True)
            .with_secret_masking()
            .with_validation()
        )

    @classmethod
    def production(cls) -> "MiddlewareBuilder":
        """Production preset: All middleware with full monitoring.

        Returns:
            Builder configured with production middleware
        """
        return (
            cls()
            .with_logging(logging.WARNING)
            .with_git_safety(block_dangerous=False, warn_on_risky=True)
            .with_secret_masking()
            .with_metrics()
            .with_validation()
            .with_cache()
        )

    @classmethod
    def performance(cls) -> "MiddlewareBuilder":
        """Performance preset: Caching and rate limiting.

        Returns:
            Builder configured with performance middleware
        """
        return (
            cls()
            .with_cache(ttl=600, max_size=2000)
            .with_rate_limit(max_calls=200, window=60)
        )


__all__ = [
    "MiddlewareProtocol",
    "MiddlewareChain",
    "MiddlewareBuilder",
]
