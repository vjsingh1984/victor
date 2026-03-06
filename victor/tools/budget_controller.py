"""Budget controller extracted from ToolCoordinator.

Tracks tool execution budget consumption and enforcement.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BudgetController:
    """Controls tool execution budget.

    Tracks how many tool calls have been consumed and enforces limits.
    Extracted from ToolCoordinator for independent testing and reuse.
    """

    def __init__(self, budget: int = 10, tool_costs: Optional[dict[str, int]] = None):
        self._budget = budget
        self._consumed = 0
        self._tool_costs = tool_costs or {}
        self._tool_counts: dict[str, int] = {}

    def consume(self, tool_name: str, cost: int = 1) -> bool:
        """Consume budget for a tool call.

        Args:
            tool_name: Name of the tool being called.
            cost: Cost of this call (default 1).

        Returns:
            True if budget was available and consumed, False if exhausted.
        """
        actual_cost = self._tool_costs.get(tool_name, cost)
        if self._consumed + actual_cost > self._budget:
            logger.warning(
                f"Budget exhausted: {self._consumed}/{self._budget} "
                f"(tried to consume {actual_cost} for {tool_name})"
            )
            return False

        self._consumed += actual_cost
        self._tool_counts[tool_name] = self._tool_counts.get(tool_name, 0) + 1
        return True

    def get_remaining(self) -> int:
        """Get remaining budget."""
        return max(0, self._budget - self._consumed)

    def is_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return self._consumed >= self._budget

    @property
    def budget(self) -> int:
        """Total budget."""
        return self._budget

    @budget.setter
    def budget(self, value: int) -> None:
        """Set total budget."""
        self._budget = value

    @property
    def consumed(self) -> int:
        """Total consumed."""
        return self._consumed

    @property
    def tool_counts(self) -> dict[str, int]:
        """Per-tool call counts."""
        return dict(self._tool_counts)

    def reset(self) -> None:
        """Reset budget tracking."""
        self._consumed = 0
        self._tool_counts.clear()
