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

"""Protocols for parallel execution framework.

This module defines protocol interfaces for parallel execution strategies,
ensuring LSP compliance and extensibility.

Design Principles (SOLID):
- Interface Segregation: Lean, focused protocols
- Liskov Substitution: All implementations are interchangeable
- Dependency Inversion: High-level modules depend on these protocols
"""

from __future__ import annotations

from typing import (
    Any,
    Optional,
    Protocol,
    runtime_checkable,
)


__all__ = [
    "JoinStrategyProtocol",
    "ErrorStrategyProtocol",
    "ResultAggregatorProtocol",
    "ParallelExecutorProtocol",
]


# =============================================================================
# Join Strategy Protocol
# =============================================================================


@runtime_checkable
class JoinStrategyProtocol(Protocol):
    """Protocol for parallel task join strategies.

    A join strategy determines how results from parallel tasks are
    combined and what conditions must be met for overall success.

    Implementations must be async-compatible and thread-safe.
    """

    async def evaluate(
        self,
        results: list[Any],
        errors: list[Exception],
    ) -> tuple[bool, Any, list[Exception]]:
        """Evaluate parallel task results.

        Args:
            results: List of task results (may contain None for failed tasks)
            errors: List of errors from failed tasks

        Returns:
            Tuple of (success, aggregated_result, errors):
                - success: Whether the join criteria were met
                - aggregated_result: Combined result or single best result
                - errors: Errors to propagate (may be empty)
        """
        ...

    def should_stop_on_error(self) -> bool:
        """Return True if execution should stop on first error.

        Returns:
            True to fail fast, False to continue all tasks
        """
        ...


# =============================================================================
# Error Strategy Protocol
# =============================================================================


@runtime_checkable
class ErrorStrategyProtocol(Protocol):
    """Protocol for parallel task error handling strategies.

    An error strategy determines how errors affect parallel execution
    and what information is propagated to callers.
    """

    async def handle_error(
        self,
        error: Exception,
        task_index: int,
        total_tasks: int,
    ) -> tuple[bool, Optional[Exception]]:
        """Handle an error from a parallel task.

        Args:
            error: The error that occurred
            task_index: Index of the failed task
            total_tasks: Total number of tasks being executed

        Returns:
            Tuple of (should_stop, error_to_propagate):
                - should_stop: True to stop execution, False to continue
                - error_to_propagate: Error to raise or None to suppress
        """
        ...

    def should_cancel_on_error(self) -> bool:
        """Return True if pending tasks should be cancelled on error.

        Returns:
            True to cancel pending tasks, False to let them complete
        """
        ...


# =============================================================================
# Result Aggregator Protocol
# =============================================================================


@runtime_checkable
class ResultAggregatorProtocol(Protocol):
    """Protocol for aggregating parallel task results.

    A result aggregator combines multiple task results into a single
    output according to a specific strategy (concat, merge, select, etc.).
    """

    async def aggregate(
        self,
        results: list[Any],
    ) -> Any:
        """Aggregate multiple results into one.

        Args:
            results: List of task results to aggregate

        Returns:
            Aggregated result
        """
        ...


# =============================================================================
# Parallel Executor Protocol
# =============================================================================


@runtime_checkable
class ParallelExecutorProtocol(Protocol):
    """Protocol for parallel task executors.

    A parallel executor manages the execution of multiple tasks concurrently,
    handling resource limits, error propagation, and result aggregation.
    """

    async def execute(
        self,
        tasks: list[Any],
        config: Any,
    ) -> "ParallelExecutionResult":
        """Execute tasks in parallel according to configuration.

        Args:
            tasks: List of callables or task definitions to execute
            config: ParallelConfig instance

        Returns:
            ParallelExecutionResult with aggregated results
        """
        ...


# =============================================================================
# Result Classes (for Protocol compatibility)
# =============================================================================


class ParallelExecutionResult:
    """Result from parallel task execution.

    Attributes:
        success: Whether execution met the success criteria
        results: List or dict of individual task results
        errors: List of errors from failed tasks
        total_count: Total number of tasks executed
        success_count: Number of successful tasks
        failure_count: Number of failed tasks
        duration_seconds: Total execution time
        strategy_used: Join strategy that was applied
    """

    def __init__(
        self,
        success: bool,
        results: Any,
        errors: list[Exception],
        total_count: int,
        success_count: int,
        failure_count: int,
        duration_seconds: float = 0.0,
        strategy_used: str = "all",
    ):
        self.success = success
        self.results = results
        self.errors = errors
        self.total_count = total_count
        self.success_count = success_count
        self.failure_count = failure_count
        self.duration_seconds = duration_seconds
        self.strategy_used = strategy_used

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "results": self.results,
            "errors": [str(e) for e in self.errors],
            "total_count": self.total_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "duration_seconds": self.duration_seconds,
            "strategy_used": self.strategy_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParallelExecutionResult":
        """Create from dictionary."""
        return cls(
            success=data["success"],
            results=data["results"],
            errors=[Exception(e) for e in data.get("errors", [])],
            total_count=data["total_count"],
            success_count=data["success_count"],
            failure_count=data["failure_count"],
            duration_seconds=data.get("duration_seconds", 0.0),
            strategy_used=data.get("strategy_used", "all"),
        )
