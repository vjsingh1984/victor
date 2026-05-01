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

"""Tool tracker service implementation.

Handles budget tracking, usage metrics, and statistics for tool executions.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ToolTrackerServiceConfig:
    """Configuration for ToolTrackerService.

    Attributes:
        initial_budget: Initial tool budget limit
        budget_multiplier: Multiplier for budget adjustments
        enable_tracking: Enable usage tracking
    """

    def __init__(
        self,
        initial_budget: int = 100,
        budget_multiplier: float = 1.0,
        enable_tracking: bool = True,
    ):
        self.initial_budget = initial_budget
        self.budget_multiplier = budget_multiplier
        self.enable_tracking = enable_tracking


class ToolTrackerService:
    """Service for tracking tool budget and usage metrics.

    Responsible for:
    - Budget management and enforcement
    - Usage statistics collection
    - Error tracking
    - Performance metrics

    This service does NOT handle:
    - Tool selection (delegated to ToolSelectorService)
    - Tool execution (delegated to ToolExecutorService)
    - Execution planning (delegated to ToolPlannerService)
    - Result processing (delegated to ToolResultProcessor)

    Example:
        config = ToolTrackerServiceConfig(initial_budget=50)
        tracker = ToolTrackerService(config=config)

        # Check budget
        if not tracker.is_budget_exhausted():
            tracker.consume_budget(1)

        # Record execution
        tracker.record_execution("search", success=True, duration_ms=150)

        # Get stats
        stats = tracker.get_tool_usage_stats()
    """

    def __init__(self, config: ToolTrackerServiceConfig):
        """Initialize ToolTrackerService.

        Args:
            config: Service configuration
        """
        self.config = config

        # Budget management
        self._budget_limit = config.initial_budget
        self._budget_used = 0

        # Usage tracking
        self._tool_call_counts: Dict[str, int] = defaultdict(int)
        self._tool_error_counts: Dict[str, int] = defaultdict(int)
        self._tool_durations: Dict[str, list[float]] = defaultdict(list)

        # Health tracking
        self._healthy = True

    def is_budget_exhausted(self) -> bool:
        """Check if tool budget is exhausted.

        Returns:
            True if budget exhausted, False otherwise
        """
        return self._budget_used >= self._budget_limit

    def get_remaining_budget(self) -> int:
        """Get remaining tool budget.

        Returns:
            Remaining budget count
        """
        return max(0, self._budget_limit - self._budget_used)

    def consume_budget(self, amount: int = 1) -> None:
        """Consume from the tool budget.

        Args:
            amount: Amount to consume (default: 1)

        Raises:
            ValueError: If amount exceeds remaining budget
        """
        remaining = self.get_remaining_budget()
        if amount > remaining:
            raise ValueError(f"Cannot consume {amount} from budget (remaining: {remaining})")

        self._budget_used += amount
        logger.debug(f"Consumed {amount} from budget, {self.get_remaining_budget()} remaining")

    def reset_tool_budget(self) -> None:
        """Reset the tool budget to initial limit."""
        self._budget_used = 0
        logger.debug(f"Reset tool budget to {self._budget_limit}")

    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics.

        Returns:
            Dictionary with usage metrics
        """
        # Calculate average durations
        avg_durations = {}
        for tool_name, durations in self._tool_durations.items():
            if durations:
                avg_durations[tool_name] = sum(durations) / len(durations)

        return {
            "budget_limit": self._budget_limit,
            "budget_used": self._budget_used,
            "budget_remaining": self.get_remaining_budget(),
            "total_calls": sum(self._tool_call_counts.values()),
            "total_errors": sum(self._tool_error_counts.values()),
            "tool_call_counts": dict(self._tool_call_counts),
            "tool_error_counts": dict(self._tool_error_counts),
            "average_durations_ms": avg_durations,
            "success_rate": self._calculate_success_rate(),
        }

    def get_tool_call_count(self, tool_name: str) -> int:
        """Get call count for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Number of times tool was called
        """
        return self._tool_call_counts.get(tool_name, 0)

    def get_tool_error_count(self, tool_name: str) -> int:
        """Get error count for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Number of times tool errored
        """
        return self._tool_error_counts.get(tool_name, 0)

    def record_execution(self, tool_name: str, success: bool, duration_ms: float) -> None:
        """Record a tool execution for metrics.

        Args:
            tool_name: Name of the tool executed
            success: Whether execution succeeded
            duration_ms: Execution duration in milliseconds
        """
        if not self.config.enable_tracking:
            return

        # Record call count
        self._tool_call_counts[tool_name] += 1

        # Record error if failed
        if not success:
            self._tool_error_counts[tool_name] += 1

        # Record duration
        self._tool_durations[tool_name].append(duration_ms)

        logger.debug(
            f"Recorded execution: {tool_name} " f"(success={success}, duration={duration_ms}ms)"
        )

    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate.

        Returns:
            Success rate as percentage (0-100)
        """
        total_calls = sum(self._tool_call_counts.values())
        total_errors = sum(self._tool_error_counts.values())

        if total_calls == 0:
            return 100.0

        return ((total_calls - total_errors) / total_calls) * 100

    @property
    def budget_limit(self) -> int:
        """Get budget limit."""
        return self._budget_limit

    @budget_limit.setter
    def budget_limit(self, value: int) -> None:
        """Set budget limit."""
        if value < 0:
            raise ValueError("Budget limit cannot be negative")
        self._budget_limit = value

    @property
    def budget_used(self) -> int:
        """Get budget used."""
        return self._budget_used

    @property
    def execution_count(self) -> int:
        """Get total execution count."""
        return sum(self._tool_call_counts.values())

    def is_healthy(self) -> bool:
        """Check if service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        return self._healthy

    def reset_metrics(self) -> None:
        """Reset all usage metrics."""
        self._tool_call_counts.clear()
        self._tool_error_counts.clear()
        self._tool_durations.clear()
        logger.debug("Reset all usage metrics")
