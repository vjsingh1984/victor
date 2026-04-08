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

"""Recovery service implementation.

Extracts error recovery from the AgentOrchestrator into
a focused, single-responsibility service following SOLID principles.

This service handles:
- Error classification and analysis
- Recovery action selection
- Automatic retry with exponential backoff
- Recovery metrics and tracking
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RecoveryContextImpl:
    """Implementation of recovery context."""

    def __init__(
        self,
        error: Exception,
        error_type: str,
        attempt_count: int,
        state: Dict[str, Any],
        metadata: Dict[str, Any],
    ):
        self.error = error
        self.error_type = error_type
        self.attempt_count = attempt_count
        self.state = state
        self.metadata = metadata


class RecoveryService:
    """Service for error recovery and resilience.

    Extracted from AgentOrchestrator to handle:
    - Error classification and analysis
    - Recovery action selection
    - Automatic retry with exponential backoff
    - Recovery metrics and tracking

    This service follows SOLID principles:
    - SRP: Only handles recovery operations
    - OCP: Extensible through strategy pattern
    - LSP: Implements RecoveryServiceProtocol
    - ISP: Focused interface
    - DIP: Depends on abstractions

    Example:
        service = RecoveryService()
        error_type = await service.classify_error(error)
        context = RecoveryContextImpl(error, error_type, 1, {}, {})
        success = await service.execute_recovery(context)
    """

    def __init__(
        self,
        max_retry_attempts: int = 3,
        base_retry_delay: float = 1.0,
        max_retry_delay: float = 60.0,
    ):
        """Initialize the recovery service.

        Args:
            max_retry_attempts: Maximum retry attempts
            base_retry_delay: Base delay for exponential backoff
            max_retry_delay: Maximum delay between retries
        """
        self._max_retry_attempts = max_retry_attempts
        self._base_retry_delay = base_retry_delay
        self._max_retry_delay = max_retry_delay
        self._metrics: Dict[str, Any] = {
            "total_attempts": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "by_error_type": {},
        }
        self._logger = logging.getLogger(f"{__name__}.{id(self)}")

    async def classify_error(self, error: Exception) -> str:
        """Classify an error for recovery strategy selection.

        Args:
            error: The error to classify

        Returns:
            Error type string
        """
        error_type = type(error).__name__

        # Map common errors to types
        error_mapping = {
            "TimeoutError": "timeout",
            "RateLimitError": "rate_limit",
            "AuthError": "auth",
            "ConnectionError": "connection",
        }

        return error_mapping.get(error_type, "unknown")

    async def select_recovery_action(
        self,
        context: RecoveryContextImpl,
    ) -> str:
        """Select appropriate recovery action for the context.

        Args:
            context: Recovery context with error and state info

        Returns:
            Recovery action name
        """
        error_type = context.error_type

        # Select action based on error type
        action_mapping = {
            "timeout": "retry",
            "rate_limit": "backoff",
            "auth": "fail",
            "connection": "retry",
            "unknown": "retry",
        }

        return action_mapping.get(error_type, "fail")

    async def execute_recovery(
        self,
        context: RecoveryContextImpl,
    ) -> bool:
        """Execute appropriate recovery action for the context.

        Args:
            context: Recovery context with error and state info

        Returns:
            True if recovery succeeded, False otherwise
        """
        self._metrics["total_attempts"] += 1

        action = await self.select_recovery_action(context)

        self._logger.info(
            f"Executing recovery action: {action} for error: {context.error_type}"
        )

        # Track by error type
        error_type = context.error_type
        if error_type not in self._metrics["by_error_type"]:
            self._metrics["by_error_type"][error_type] = {
                "attempts": 0,
                "successes": 0,
            }
        self._metrics["by_error_type"][error_type]["attempts"] += 1

        # Execute action
        success = False

        if action == "retry":
            success = await self._retry_action(context)
        elif action == "backoff":
            success = await self._backoff_action(context)
        elif action == "fail":
            success = False
        else:
            success = False

        # Track results
        if success:
            self._metrics["successful_recoveries"] += 1
            self._metrics["by_error_type"][error_type]["successes"] += 1
        else:
            self._metrics["failed_recoveries"] += 1

        return success

    def can_retry(
        self,
        error: Exception,
        attempt_count: int,
    ) -> bool:
        """Check if an operation can be retried.

        Args:
            error: The error that occurred
            attempt_count: Number of attempts made so far

        Returns:
            True if operation can be retried, False otherwise
        """
        # Don't retry auth errors
        if isinstance(error, (PermissionError, AuthError)):
            return False

        # Check attempt limit
        return attempt_count < self._max_retry_attempts

    def get_recovery_metrics(self) -> Dict[str, Any]:
        """Get recovery statistics and metrics.

        Returns:
            Dictionary with recovery metrics
        """
        total = self._metrics["total_attempts"]
        successful = self._metrics["successful_recoveries"]

        return {
            **self._metrics,
            "success_rate": successful / total if total > 0 else 0.0,
        }

    def reset_metrics(self) -> None:
        """Reset recovery metrics."""
        self._metrics = {
            "total_attempts": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "by_error_type": {},
        }

    def is_healthy(self) -> bool:
        """Check if the recovery service is healthy.

        Returns:
            True if the service is healthy
        """
        return self._max_retry_attempts > 0

    # ==========================================================================
    # Private Methods
    # ==========================================================================

    async def _retry_action(self, context: RecoveryContextImpl) -> bool:
        """Execute retry action.

        Args:
            context: Recovery context

        Returns:
            True if retry should be attempted
        """
        # Just indicate retry is possible
        # The actual retry is handled by the caller
        return context.attempt_count < self._max_retry_attempts

    async def _backoff_action(self, context: RecoveryContextImpl) -> bool:
        """Execute backoff action.

        Args:
            context: Recovery context

        Returns:
            True if backoff completed successfully
        """
        delay = self._calculate_retry_delay(context.attempt_count)
        self._logger.info(f"Backing off for {delay:.1f} seconds")
        await asyncio.sleep(delay)
        return True

    def _calculate_retry_delay(
        self,
        attempt: int,
    ) -> float:
        """Calculate retry delay with exponential backoff.

        Args:
            attempt: Current attempt number

        Returns:
            Delay in seconds
        """
        delay = self._base_retry_delay * (2**attempt)
        return min(delay, self._max_retry_delay)


class AuthError(Exception):
    """Authentication error."""

    pass
