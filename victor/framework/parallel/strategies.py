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

"""Parallel execution strategies for workflow nodes.

This module provides strategy enums and configuration classes for parallel
execution of workflow tasks, tools, and nodes. These strategies control
how parallel tasks are joined, how errors are handled, and what resource
limits are applied.

Design Principles (SOLID):
- Single Responsibility: Each strategy handles one aspect of parallel execution
- Open/Closed: Extensible via custom strategies without modifying core logic
- Liskov Substitution: All strategies implement the same protocol
- Interface Segregation: Lean interfaces for specific use cases
- Dependency Inversion: Depends on protocols, not concretions

Example:
    from victor.framework.parallel import (
        JoinStrategy,
        ErrorStrategy,
        ResourceLimit,
        ParallelConfig,
    )

    # Configure parallel execution
    config = ParallelConfig(
        join_strategy=JoinStrategy.MAJORITY,
        error_strategy=ErrorStrategy.COLLECT_ERRORS,
        resource_limit=ResourceLimit(max_concurrent=5, timeout=30.0),
    )
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from victor.framework.parallel.protocols import (
    JoinStrategyProtocol,
    ErrorStrategyProtocol,
    ResultAggregatorProtocol,
)


__all__ = [
    # Enums
    "JoinStrategy",
    "ErrorStrategy",
    # Configuration
    "ResourceLimit",
    "ParallelConfig",
    # Strategy classes
    "AllJoinStrategy",
    "AnyJoinStrategy",
    "FirstJoinStrategy",
    "NOfMJoinStrategy",
    "MajorityJoinStrategy",
    "FailFastErrorStrategy",
    "ContinueAllErrorStrategy",
    "CollectErrorsErrorStrategy",
    # Protocol validators
    "validate_join_strategy",
    "validate_error_strategy",
]


# =============================================================================
# Enums
# =============================================================================


class JoinStrategy(str, Enum):
    """Strategy for joining parallel task results.

    Determines how parallel task results are combined and what conditions
    must be met for the parallel execution to be considered successful.

    Attributes:
        ALL: All tasks must succeed (default, safest)
        ANY: At least one task must succeed (redundancy pattern)
        FIRST: First successful result wins (race pattern)
        N_OF_M: Exactly N of M tasks must succeed
        MAJORITY: More than half of tasks must succeed (consensus pattern)
    """

    ALL = "all"
    ANY = "any"
    FIRST = "first"
    N_OF_M = "n_of_m"
    MAJORITY = "majority"

    @classmethod
    def from_string(cls, value: str) -> "JoinStrategy":
        """Convert string to JoinStrategy, with validation."""
        try:
            return cls(value.lower())
        except ValueError:
            valid = [s.value for s in cls]
            raise ValueError(f"Invalid join_strategy '{value}'. " f"Must be one of: {valid}")


class ErrorStrategy(str, Enum):
    """Strategy for handling errors in parallel execution.

    Determines how errors affect the overall parallel execution and
    what information is propagated to callers.

    Attributes:
        FAIL_FAST: Stop on first error (default, fail-fast behavior)
        CONTINUE_ALL: Continue all tasks despite errors (best effort)
        COLLECT_ERRORS: Collect all errors but don't stop (comprehensive)
    """

    FAIL_FAST = "fail_fast"
    CONTINUE_ALL = "continue_all"
    COLLECT_ERRORS = "collect_errors"

    @classmethod
    def from_string(cls, value: str) -> "ErrorStrategy":
        """Convert string to ErrorStrategy, with validation."""
        try:
            return cls(value.lower())
        except ValueError:
            valid = [s.value for s in cls]
            raise ValueError(f"Invalid error_strategy '{value}'. " f"Must be one of: {valid}")


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class ResourceLimit:
    """Resource limits for parallel execution.

    Defines constraints on parallel task execution to prevent resource
    exhaustion and ensure predictable behavior.

    Attributes:
        max_concurrent: Maximum number of concurrent tasks (None = unlimited)
        timeout: Per-task timeout in seconds (None = no timeout)
        memory_limit: Optional memory limit in MB (for monitoring only)
        cpu_limit: Optional CPU limit as fraction (0.0-1.0, for monitoring)
    """

    max_concurrent: Optional[int] = None
    timeout: Optional[float] = None
    memory_limit: Optional[int] = None
    cpu_limit: Optional[float] = None

    def __post_init__(self):
        """Validate resource limits."""
        if self.max_concurrent is not None and self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive or None")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive or None")
        if self.cpu_limit is not None and not (0.0 <= self.cpu_limit <= 1.0):
            raise ValueError("cpu_limit must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "max_concurrent": self.max_concurrent,
            "timeout": self.timeout,
            "memory_limit": self.memory_limit,
            "cpu_limit": self.cpu_limit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceLimit":
        """Deserialize from dictionary."""
        return cls(
            max_concurrent=data.get("max_concurrent"),
            timeout=data.get("timeout"),
            memory_limit=data.get("memory_limit"),
            cpu_limit=data.get("cpu_limit"),
        )


@dataclass
class ParallelConfig:
    """Complete configuration for parallel execution.

    Combines join strategy, error handling, and resource limits into
    a single configuration object for parallel task execution.

    Attributes:
        join_strategy: How to combine parallel results
        error_strategy: How to handle errors
        resource_limit: Resource constraints
        n_of_m: Required count for N_OF_M join strategy
    """

    join_strategy: JoinStrategy = JoinStrategy.ALL
    error_strategy: ErrorStrategy = ErrorStrategy.FAIL_FAST
    resource_limit: ResourceLimit = field(default_factory=ResourceLimit)
    n_of_m: Optional[int] = None  # Required for N_OF_M strategy

    def __post_init__(self):
        """Validate configuration."""
        if self.join_strategy == JoinStrategy.N_OF_M and self.n_of_m is None:
            raise ValueError("n_of_m must be specified for N_OF_M join strategy")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "join_strategy": self.join_strategy.value,
            "error_strategy": self.error_strategy.value,
            "resource_limit": self.resource_limit.to_dict(),
            "n_of_m": self.n_of_m,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParallelConfig":
        """Deserialize from dictionary."""
        resource_limit_data = data.get("resource_limit", {})
        return cls(
            join_strategy=JoinStrategy.from_string(data.get("join_strategy", "all")),
            error_strategy=ErrorStrategy.from_string(data.get("error_strategy", "fail_fast")),
            resource_limit=ResourceLimit.from_dict(resource_limit_data),
            n_of_m=data.get("n_of_m"),
        )


# =============================================================================
# Join Strategy Implementations
# =============================================================================


@dataclass
class AllJoinStrategy:
    """All tasks must succeed for parallel execution to succeed.

    This is the safest and most conservative join strategy. All parallel
    tasks must complete successfully for the overall execution to succeed.
    """

    async def evaluate(
        self,
        results: List[Any],
        errors: List[Exception],
    ) -> tuple[bool, Any, List[Exception]]:
        """Evaluate results requiring all tasks to succeed.

        Args:
            results: List of task results (may contain None for failed tasks)
            errors: List of errors from failed tasks

        Returns:
            Tuple of (success, aggregated_result, errors)
        """
        if errors:
            return False, None, errors

        return True, results, []

    def should_stop_on_error(self) -> bool:
        """Return True to stop on first error."""
        return True


@dataclass
class AnyJoinStrategy:
    """At least one task must succeed for parallel execution to succeed.

    This strategy is useful for redundancy patterns where multiple
    attempts are made in parallel and any success is acceptable.
    """

    async def evaluate(
        self,
        results: List[Any],
        errors: List[Exception],
    ) -> tuple[bool, Any, List[Exception]]:
        """Evaluate results requiring at least one successful task.

        Args:
            results: List of task results (may contain None for failed tasks)
            errors: List of errors from failed tasks

        Returns:
            Tuple of (success, aggregated_result, errors)
        """
        # Filter out None results (from failed tasks)
        successful_results = [r for r in results if r is not None]

        if successful_results:
            return True, successful_results, errors

        return False, None, errors

    def should_stop_on_error(self) -> bool:
        """Return False to continue despite errors."""
        return False


@dataclass
class FirstJoinStrategy:
    """First successful result wins.

    This strategy is useful for race patterns where the first
    task to complete successfully determines the result.
    """

    async def evaluate(
        self,
        results: List[Any],
        errors: List[Exception],
    ) -> tuple[bool, Any, List[Exception]]:
        """Evaluate results returning first successful result.

        Args:
            results: List of task results (ordered by completion)
            errors: List of errors from failed tasks

        Returns:
            Tuple of (success, aggregated_result, errors)
        """
        for result in results:
            if result is not None:
                return True, result, errors

        return False, None, errors

    def should_stop_on_error(self) -> bool:
        """Return False to continue until first success."""
        return False


@dataclass
class NOfMJoinStrategy:
    """Exactly N of M tasks must succeed.

    This strategy requires a specific number of tasks to succeed.
    Useful for quorum patterns where a minimum threshold is required.

    Attributes:
        required: Number of tasks that must succeed
    """

    required: int

    def __post_init__(self):
        """Validate required count."""
        if self.required <= 0:
            raise ValueError("required must be positive")

    async def evaluate(
        self,
        results: List[Any],
        errors: List[Exception],
    ) -> tuple[bool, Any, List[Exception]]:
        """Evaluate results requiring N of M tasks to succeed.

        Args:
            results: List of task results (may contain None for failed tasks)
            errors: List of errors from failed tasks

        Returns:
            Tuple of (success, aggregated_result, errors)
        """
        len(results) + len(errors)
        successful_results = [r for r in results if r is not None]
        success_count = len(successful_results)

        if success_count >= self.required:
            return True, successful_results, errors

        return False, None, errors

    def should_stop_on_error(self) -> bool:
        """Return False - need to count all successes."""
        return False


@dataclass
class MajorityJoinStrategy:
    """More than half of tasks must succeed (consensus pattern).

    This strategy is useful for democratic decision making where
    majority agreement is required.
    """

    async def evaluate(
        self,
        results: List[Any],
        errors: List[Exception],
    ) -> tuple[bool, Any, List[Exception]]:
        """Evaluate results requiring majority of tasks to succeed.

        Args:
            results: List of task results (may contain None for failed tasks)
            errors: List of errors from failed tasks

        Returns:
            Tuple of (success, aggregated_result, errors)
        """
        total_tasks = len(results) + len(errors)
        if total_tasks == 0:
            return True, [], errors

        successful_results = [r for r in results if r is not None]
        success_count = len(successful_results)
        required = (total_tasks // 2) + 1

        if success_count >= required:
            return True, successful_results, errors

        return False, None, errors

    def should_stop_on_error(self) -> bool:
        """Return False - need to count all for majority."""
        return False


# =============================================================================
# Error Strategy Implementations
# =============================================================================


@dataclass
class FailFastErrorStrategy:
    """Stop execution on first error.

    This is the default error handling strategy. When any task fails,
    all pending tasks are cancelled and the error is propagated.
    """

    async def handle_error(
        self,
        error: Exception,
        task_index: int,
        total_tasks: int,
    ) -> tuple[bool, Optional[Exception]]:
        """Handle error by failing fast.

        Args:
            error: The error that occurred
            task_index: Index of the failed task
            total_tasks: Total number of tasks

        Returns:
            Tuple of (should_stop, error_to_raise)
        """
        return True, error

    def should_cancel_on_error(self) -> bool:
        """Return True to cancel pending tasks on error."""
        return True


@dataclass
class ContinueAllErrorStrategy:
    """Continue execution despite errors.

    This strategy allows all tasks to complete regardless of failures.
    Errors are logged but don't stop execution.
    """

    async def handle_error(
        self,
        error: Exception,
        task_index: int,
        total_tasks: int,
    ) -> tuple[bool, Optional[Exception]]:
        """Handle error by continuing execution.

        Args:
            error: The error that occurred
            task_index: Index of the failed task
            total_tasks: Total number of tasks

        Returns:
            Tuple of (should_stop, error_to_raise)
        """
        # Log and continue
        return False, None

    def should_cancel_on_error(self) -> bool:
        """Return False to continue all tasks."""
        return False


@dataclass
class CollectErrorsErrorStrategy:
    """Collect all errors but continue execution.

    This strategy allows all tasks to complete and collects all errors
    for comprehensive reporting. Execution fails only if no tasks succeed.
    """

    collected_errors: List[Exception] = field(default_factory=list)

    async def handle_error(
        self,
        error: Exception,
        task_index: int,
        total_tasks: int,
    ) -> tuple[bool, Optional[Exception]]:
        """Handle error by collecting it.

        Args:
            error: The error that occurred
            task_index: Index of the failed task
            total_tasks: Total number of tasks

        Returns:
            Tuple of (should_stop, error_to_raise)
        """
        self.collected_errors.append(error)
        return False, None

    def should_cancel_on_error(self) -> bool:
        """Return False to continue all tasks."""
        return False

    def get_errors(self) -> List[Exception]:
        """Get all collected errors."""
        return self.collected_errors.copy()

    def clear_errors(self) -> None:
        """Clear collected errors."""
        self.collected_errors.clear()


# =============================================================================
# Strategy Factory Functions
# =============================================================================


_JOIN_STRATEGY_IMPLS: Dict[JoinStrategy, type] = {
    JoinStrategy.ALL: AllJoinStrategy,
    JoinStrategy.ANY: AnyJoinStrategy,
    JoinStrategy.FIRST: FirstJoinStrategy,
    JoinStrategy.N_OF_M: NOfMJoinStrategy,
    JoinStrategy.MAJORITY: MajorityJoinStrategy,
}

_ERROR_STRATEGY_IMPLS: Dict[ErrorStrategy, type] = {
    ErrorStrategy.FAIL_FAST: FailFastErrorStrategy,
    ErrorStrategy.CONTINUE_ALL: ContinueAllErrorStrategy,
    ErrorStrategy.COLLECT_ERRORS: CollectErrorsErrorStrategy,
}


def create_join_strategy(strategy: JoinStrategy, **kwargs: Any) -> Any:
    """Create a join strategy instance from enum.

    Args:
        strategy: The join strategy enum value
        **kwargs: Additional arguments for the strategy (e.g., n_of_m)

    Returns:
        Strategy instance implementing JoinStrategyProtocol
    """
    strategy_cls = _JOIN_STRATEGY_IMPLS.get(strategy)
    if strategy_cls is None:
        raise ValueError(f"Unknown join strategy: {strategy}")

    if strategy == JoinStrategy.N_OF_M:
        required = kwargs.get("n_of_m", kwargs.get("required", 1))
        return strategy_cls(required=required)

    return strategy_cls()


def create_error_strategy(strategy: ErrorStrategy, **kwargs: Any) -> Any:
    """Create an error strategy instance from enum.

    Args:
        strategy: The error strategy enum value
        **kwargs: Additional arguments for the strategy

    Returns:
        Strategy instance implementing ErrorStrategyProtocol
    """
    strategy_cls = _ERROR_STRATEGY_IMPLS.get(strategy)
    if strategy_cls is None:
        raise ValueError(f"Unknown error strategy: {strategy}")

    return strategy_cls(**kwargs)


# =============================================================================
# Validation Functions
# =============================================================================


def validate_join_strategy(strategy: str) -> JoinStrategy:
    """Validate and convert a join strategy string.

    Args:
        strategy: Strategy name as string

    Returns:
        Validated JoinStrategy enum

    Raises:
        ValueError: If strategy name is invalid
    """
    return JoinStrategy.from_string(strategy)


def validate_error_strategy(strategy: str) -> ErrorStrategy:
    """Validate and convert an error strategy string.

    Args:
        strategy: Strategy name as string

    Returns:
        Validated ErrorStrategy enum

    Raises:
        ValueError: If strategy name is invalid
    """
    return ErrorStrategy.from_string(strategy)
