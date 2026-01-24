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

"""Tool Budget Coordinator - Manages tool call budget enforcement.

This module extracts budget management logic from ToolCoordinator,
following SRP (Single Responsibility Principle).

Responsibilities:
- Budget tracking (used, remaining, total)
- Budget consumption with warnings
- Budget reset and multiplier application
- Budget exhaustion checking

Design Philosophy:
- Single Responsibility: Only handles budget tracking
- Callback-based: Notifies on budget warnings
- Stateless: Budget state is encapsulated
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.budget_manager import BudgetManager

from victor.agent.coordinators.base_config import BaseCoordinatorConfig

logger = logging.getLogger(__name__)


@dataclass
class ToolBudgetConfig(BaseCoordinatorConfig):
    """Configuration for ToolBudgetCoordinator.

    Inherits common configuration from BaseCoordinatorConfig:
        enabled: Whether the coordinator is enabled
        timeout: Default timeout in seconds for operations
        max_retries: Maximum number of retry attempts for failed operations
        retry_enabled: Whether retry logic is enabled
        log_level: Logging level for coordinator messages
        enable_metrics: Whether to collect metrics

    Attributes:
        default_budget: Default tool call budget
        budget_multiplier: Multiplier for budget based on complexity
        warning_threshold: Percentage (0-1) to trigger warnings (default: 0.2 = 20%)
    """

    default_budget: int = 25
    budget_multiplier: float = 1.0
    warning_threshold: float = 0.2


@dataclass
class BudgetStatus:
    """Status of the tool budget.

    Attributes:
        total: Total budget available
        used: Budget units consumed
        remaining: Budget units remaining
        is_exhausted: Whether budget is exhausted
        utilization: Fraction of budget used (0-1)
    """

    total: int
    used: int
    remaining: int
    is_exhausted: bool
    utilization: float


@dataclass
class BudgetStats:
    """Statistics for budget usage.

    Attributes:
        total_consumed: Total budget consumed over session
        total_reset_count: Number of times budget was reset
        multiplier_history: History of multiplier changes
        warning_count: Number of warnings issued
    """

    total_consumed: int = 0
    total_reset_count: int = 0
    multiplier_history: list[tuple[float, float]] = field(default_factory=list)
    warning_count: int = 0


class ToolBudgetCoordinator:
    """Coordinator for tool call budget management.

    Extracts budget management logic from ToolCoordinator following SRP.
    This coordinator is responsible only for tracking and enforcing
    tool call budgets.

    Example:
        coordinator = ToolBudgetCoordinator(
            config=ToolBudgetConfig(default_budget=30),
            on_warning=lambda r, t: logger.warning(f"Low budget: {r}/{t}")
        )

        # Consume budget
        coordinator.consume(5)
        if coordinator.is_exhausted():
            logger.error("Budget exhausted!")

        # Check status
        status = coordinator.get_status()
        print(f"{status.remaining}/{status.total} remaining")

        # Reset for new session
        coordinator.reset(new_budget=50)
    """

    def __init__(
        self,
        budget_manager: Optional["BudgetManager"] = None,
        config: Optional[ToolBudgetConfig] = None,
        on_warning: Optional[Callable[[int, int], None]] = None,  # remaining, total
    ) -> None:
        """Initialize the budget coordinator.

        Args:
            budget_manager: Optional BudgetManager for integration
            config: Budget configuration
            on_warning: Callback when budget is running low
        """
        self._budget_manager = budget_manager
        self._config = config or ToolBudgetConfig()
        self._on_warning = on_warning

        # Internal state (used only if no budget_manager provided)
        self._budget_used: int = 0
        self._total_budget: int = self._config.default_budget

        # Statistics
        self._stats = BudgetStats()

        logger.debug(
            f"ToolBudgetCoordinator initialized with budget={self._total_budget}, "
            f"warning_threshold={self._config.warning_threshold}"
        )

    # =====================================================================
    # Budget Queries
    # =====================================================================

    @property
    def budget(self) -> int:
        """Get the total tool budget."""
        if self._budget_manager:
            # TODO: Fix this to use get_status(BudgetType.TOOL_CALLS)
            max_calls: int = getattr(self._budget_manager, "get_max_tool_calls", lambda: self._total_budget)()
            return max_calls
        return self._total_budget

    @budget.setter
    def budget(self, value: int) -> None:
        """Set the total tool budget."""
        if self._budget_manager:
            # TODO: Fix this to use proper API
            if hasattr(self._budget_manager, "config"):
                config = self._budget_manager.config
                if hasattr(config, "base_tool_calls"):
                    config.base_tool_calls = value
        else:
            self._total_budget = max(0, value)

    @property
    def budget_used(self) -> int:
        """Get the number of budget units used."""
        if self._budget_manager:
            # TODO: Fix this to use get_status(BudgetType.TOOL_CALLS)
            used: int = getattr(self._budget_manager, "get_used_tool_calls", lambda: self._budget_used)()
            return used
        return self._budget_used

    @property
    def budget_multiplier(self) -> float:
        """Get the current budget multiplier."""
        return self._config.budget_multiplier

    def get_remaining_budget(self) -> int:
        """Get remaining tool call budget.

        Returns:
            Number of tool calls remaining
        """
        if self._budget_manager:
            # TODO: Fix this to use get_status(BudgetType.TOOL_CALLS)
            remaining: int = getattr(self._budget_manager, "get_remaining_tool_calls", lambda: max(0, self._total_budget - self._budget_used))()
            return remaining
        return max(0, self._total_budget - self._budget_used)

    def is_exhausted(self) -> bool:
        """Check if tool budget is exhausted.

        Returns:
            True if no budget remaining
        """
        return self.get_remaining_budget() <= 0

    def get_status(self) -> BudgetStatus:
        """Get comprehensive budget status.

        Returns:
            BudgetStatus with current budget state
        """
        total = self.budget
        used = self.budget_used
        remaining = self.get_remaining_budget()
        utilization = used / total if total > 0 else 0

        return BudgetStatus(
            total=total,
            used=used,
            remaining=remaining,
            is_exhausted=self.is_exhausted(),
            utilization=utilization,
        )

    # =====================================================================
    # Budget Operations
    # =====================================================================

    def consume(self, amount: int = 1) -> None:
        """Consume tool call budget.

        Args:
            amount: Number of budget units to consume
        """
        if amount <= 0:
            return

        if self._budget_manager:
            # TODO: Fix this to use consume(BudgetType.TOOL_CALLS, amount)
            if hasattr(self._budget_manager, "consume_tool_call"):
                self._budget_manager.consume_tool_call(amount)
            else:
                # Fallback to using record_tool_call
                self._budget_manager.record_tool_call("tool_call")
        else:
            self._budget_used += amount

        self._stats.total_consumed += amount

        # Check for warning threshold
        remaining = self.get_remaining_budget()
        total = self.budget

        if remaining < total * self._config.warning_threshold:
            self._stats.warning_count += 1
            if self._on_warning:
                self._on_warning(remaining, total)
            logger.debug(
                f"Budget warning: {remaining}/{total} remaining " f"({remaining/total*100:.1f}%)"
            )

    def reset(
        self,
        new_budget: Optional[int] = None,
    ) -> None:
        """Reset the tool budget.

        Args:
            new_budget: New budget to set, or use default
        """
        if new_budget is not None:
            self.budget = new_budget
        else:
            self.budget = self._config.default_budget

        if self._budget_manager:
            # TODO: Fix this to use reset(BudgetType.TOOL_CALLS)
            if hasattr(self._budget_manager, "reset_tool_calls"):
                self._budget_manager.reset_tool_calls()
            else:
                self._budget_manager.reset()

        self._budget_used = 0
        self._stats.total_reset_count += 1

        logger.debug(f"Budget reset to {self.budget}")

    def set_multiplier(self, multiplier: float) -> None:
        """Set budget multiplier for complexity adjustments.

        Args:
            multiplier: Multiplier to apply (e.g., 2.0 for complex tasks)
        """
        old_multiplier = self._config.budget_multiplier
        self._config.budget_multiplier = multiplier

        # Update effective budget
        effective_budget = int(self._config.default_budget * multiplier)
        self.budget = effective_budget

        # Track history
        self._stats.multiplier_history.append((old_multiplier, multiplier))

        logger.debug(
            f"Budget multiplier changed from {old_multiplier} to {multiplier}, "
            f"effective budget: {effective_budget}"
        )

    # =====================================================================
    # Statistics
    # =====================================================================

    def get_stats(self) -> BudgetStats:
        """Get budget usage statistics.

        Returns:
            BudgetStats with usage analytics
        """
        return BudgetStats(
            total_consumed=self._stats.total_consumed + self.budget_used,
            total_reset_count=self._stats.total_reset_count,
            multiplier_history=list(self._stats.multiplier_history),
            warning_count=self._stats.warning_count,
        )

    def clear_stats(self) -> None:
        """Clear accumulated statistics."""
        self._stats = BudgetStats()


def create_tool_budget_coordinator(
    default_budget: int = 25,
    warning_threshold: float = 0.2,
    budget_manager: Optional["BudgetManager"] = None,
) -> ToolBudgetCoordinator:
    """Factory function to create a ToolBudgetCoordinator.

    Args:
        default_budget: Default tool call budget
        warning_threshold: Percentage to trigger warnings (0-1)
        budget_manager: Optional BudgetManager for integration

    Returns:
        Configured ToolBudgetCoordinator instance
    """
    config = ToolBudgetConfig(
        default_budget=default_budget,
        warning_threshold=warning_threshold,
    )

    return ToolBudgetCoordinator(
        budget_manager=budget_manager,
        config=config,
    )


__all__ = [
    "ToolBudgetCoordinator",
    "ToolBudgetConfig",
    "BudgetStatus",
    "BudgetStats",
    "create_tool_budget_coordinator",
]
