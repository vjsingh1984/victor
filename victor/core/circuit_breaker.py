"""Shared circuit-breaker types.

Layer-specific circuit breakers live in providers, agent, and observability
modules, but they should use these shared state/config/error types so callers
can compare and handle circuit state consistently across layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration shared by circuit-breaker implementations."""

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 30.0
    half_open_max_calls: int = 3

    @property
    def recovery_timeout(self) -> float:
        """Alias for timeout_seconds for backward compatibility."""
        return self.timeout_seconds


class CircuitBreakerError(Exception):
    """Raised when a circuit rejects a call."""

    def __init__(
        self,
        message: str,
        state: CircuitState,
        retry_after: float,
        last_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.state = state
        self.retry_after = retry_after
        self.last_error = last_error


__all__ = [
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
]
