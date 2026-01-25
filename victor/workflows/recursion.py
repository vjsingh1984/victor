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

"""Recursion depth tracking for nested workflow and team execution.

This module provides unified recursion depth tracking to prevent infinite nesting
between workflows and teams. All nesting types count toward the same limit:
- Workflow invoking workflow
- Workflow spawning team
- Team spawning team
- Team spawning workflow

Example:
    from victor.workflows.recursion import RecursionContext

    recursion_ctx = RecursionContext(max_depth=3)

    # Enter a nested execution
    recursion_ctx.enter("workflow", "my_workflow")
    recursion_ctx.enter("team", "research_team")

    # Check current depth
    print(f"Current depth: {recursion_ctx.current_depth}")  # 2

    # Exit in reverse order
    recursion_ctx.exit()
    recursion_ctx.exit()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional

from victor.core.errors import RecursionDepthError

logger = logging.getLogger(__name__)


@dataclass
class RecursionContext:
    """Tracks recursion depth for nested execution with thread safety.

    This class provides unified recursion depth tracking across all nested
    execution types (workflows and teams). It maintains a stack trace
    for debugging and clear error messages when limits are exceeded.

    Thread-safe for concurrent access via reentrant locks, allowing safe
    use in multi-threaded environments where workflows may execute in parallel.

    Attributes:
        current_depth: Current nesting level (0 = top-level)
        max_depth: Maximum allowed nesting level (default: 3)
        execution_stack: Stack trace of execution entries
        _lock: Thread lock for concurrent access (reentrant for same-thread nesting)

    Example:
        >>> ctx = RecursionContext(max_depth=3)
        >>> ctx.enter("workflow", "main")
        >>> ctx.enter("team", "research")
        >>> print(ctx.current_depth)
        2
        >>> print(ctx.execution_stack)
        ['workflow:main', 'team:research']
        >>> ctx.exit()
        >>> ctx.exit()
        >>> print(ctx.current_depth)
        0
    """

    current_depth: int = 0
    max_depth: int = 3
    execution_stack: List[str] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def enter(self, operation_type: str, identifier: str) -> None:
        """Enter a nested execution level with thread-safe locking.

        Args:
            operation_type: Type of operation ("workflow" or "team")
            identifier: Unique identifier (workflow name, team name, etc.)

        Raises:
            RecursionDepthError: If max_depth would be exceeded

        Example:
            >>> ctx = RecursionContext(max_depth=3)
            >>> ctx.enter("workflow", "my_workflow")
            >>> ctx.enter("team", "research_team")
            >>> ctx.enter("workflow", "nested_workflow")
            >>> ctx.enter("team", "inner_team")  # Raises RecursionDepthError
        """
        with self._lock:
            if self.current_depth >= self.max_depth:
                raise RecursionDepthError(
                    message=f"Maximum recursion depth ({self.max_depth}) exceeded. "
                    f"Attempting to enter {operation_type}:{identifier}",
                    current_depth=self.current_depth,
                    max_depth=self.max_depth,
                    execution_stack=self.execution_stack.copy(),
                )

            self.current_depth += 1
            self.execution_stack.append(f"{operation_type}:{identifier}")

            logger.debug(
                f"Entered recursion level {self.current_depth}/{self.max_depth}: "
                f"{operation_type}:{identifier}"
            )

    def exit(self) -> None:
        """Exit a nested execution level with thread-safe locking.

        Decrements the current depth and removes the last entry from the
        execution stack. Safe to call when already at depth 0.

        Example:
            >>> ctx = RecursionContext()
            >>> ctx.enter("workflow", "test")
            >>> ctx.exit()
            >>> print(ctx.current_depth)
            0
        """
        with self._lock:
            if self.execution_stack:
                exited = self.execution_stack.pop()
                logger.debug(f"Exited recursion level: {exited}")

            self.current_depth = max(0, self.current_depth - 1)

    def can_nest(self, levels: int = 1) -> bool:
        """Check if nesting is possible without exceeding max_depth (thread-safe).

        Args:
            levels: Number of levels to check (default: 1)

        Returns:
            True if nesting is possible, False otherwise

        Example:
            >>> ctx = RecursionContext(max_depth=3)
            >>> ctx.enter("workflow", "test")
            >>> ctx.can_nest(1)  # Can go to depth 2
            True
            >>> ctx.can_nest(2)  # Can go to depth 3
            True
            >>> ctx.can_nest(3)  # Would exceed max_depth
            False
        """
        with self._lock:
            return (self.current_depth + levels) <= self.max_depth

    def get_depth_info(self) -> Dict[str, Any]:
        """Get current recursion depth information (thread-safe).

        Returns:
            Dictionary with current_depth, max_depth, remaining_depth,
            and execution_stack

        Example:
            >>> ctx = RecursionContext(max_depth=5)
            >>> ctx.enter("workflow", "main")
            >>> info = ctx.get_depth_info()
            >>> print(info['current_depth'])
            1
            >>> print(info['remaining_depth'])
            4
        """
        with self._lock:
            return {
                "current_depth": self.current_depth,
                "max_depth": self.max_depth,
                "remaining_depth": self.max_depth - self.current_depth,
                "execution_stack": self.execution_stack.copy(),
            }

    def __enter__(self) -> "RecursionContext":
        """Context manager entry - for automatic cleanup.

        Example:
            >>> ctx = RecursionContext()
            >>> with ctx:
            ...     ctx.enter("workflow", "test")
            ...     # Automatically exits when leaving context
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - reset state (thread-safe)."""
        with self._lock:
            self.current_depth = 0
            self.execution_stack.clear()

    def __repr__(self) -> str:
        """String representation (thread-safe)."""
        with self._lock:
            return (
                f"RecursionContext(depth={self.current_depth}/{self.max_depth}, "
                f"stack={self.execution_stack})"
            )

    def __deepcopy__(self, memo: Dict[int, Any]) -> "RecursionContext":
        """Custom deepcopy to support checkpointing.

        When checkpointing creates a deep copy of workflow state, we need to
        handle the _lock field properly. Locks cannot be pickled/deepcopied,
        so we create a new lock for the copied instance.

        The execution_stack and depth tracking are copied to preserve the
        current recursion state.

        Args:
            memo: Memoization dictionary for tracking already-copied objects

        Returns:
            New RecursionContext with copied state and fresh lock
        """
        # Avoid infinite recursion
        if id(self) in memo:
            return memo[id(self)]

        # Create a new instance with the same class
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy non-lock fields (need lock for thread-safe access)
        with self._lock:
            result.current_depth = self.current_depth
            result.max_depth = self.max_depth
            result.execution_stack = self.execution_stack.copy()

        # Create a fresh lock for the new instance
        result._lock = threading.RLock()

        return result


class RecursionGuard:
    """Context manager for automatic recursion tracking.

    Ensures that enter() and exit() are properly paired, even when
    exceptions occur.

    Example:
        >>> ctx = RecursionContext(max_depth=3)
        >>> with RecursionGuard(ctx, "workflow", "my_workflow"):
        ...     # Nested execution here
        ...     pass
        # Automatically exits when leaving context
    """

    def __init__(
        self,
        recursion_context: RecursionContext,
        operation_type: str,
        identifier: str,
    ):
        self._ctx = recursion_context
        self._operation_type = operation_type
        self._identifier = identifier

    def __enter__(self) -> "RecursionGuard":
        self._ctx.enter(self._operation_type, self._identifier)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit the recursion guard.

        Returns:
            False to indicate exceptions should not be suppressed
        """
        self._ctx.exit()
        return False  # Don't suppress exceptions
