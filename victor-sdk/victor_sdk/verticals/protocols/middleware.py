"""Middleware-related protocol definitions.

These protocols define how verticals provide middleware configurations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Callable, Any, Dict, List, Awaitable


@runtime_checkable
class MiddlewareProvider(Protocol):
    """Protocol for providing middleware.

    Middleware can intercept and modify execution flow at various points.
    """

    def get_pre_execution_middleware(
        self,
    ) -> List[Callable[[Dict[str, Any]], Awaitable[None]]]:
        """Return middleware to run before execution.

        Returns:
            List of async functions that take context
        """
        ...

    def get_post_execution_middleware(
        self,
    ) -> List[Callable[[Dict[str, Any]], Awaitable[None]]]:
        """Return middleware to run after execution.

        Returns:
            List of async functions that take context
        """
        ...

    def get_error_handling_middleware(
        self,
    ) -> List[Callable[[Exception, Dict[str, Any]], Awaitable[None]]]:
        """Return middleware for error handling.

        Returns:
            List of async functions that take exception and context
        """
        ...

    def get_middleware_order(self) -> List[str]:
        """Return ordered list of middleware to apply.

        Returns:
            List of middleware identifiers in execution order
        """
        ...
