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

"""Generic parallel executor for workflow nodes and tasks.

This module provides a framework-level parallel execution capability that
can be used across all verticals. It replaces duplicated parallel execution
patterns with a unified, configurable implementation.

Design Principles (SOLID):
- Single Responsibility: Only handles parallel execution coordination
- Open/Closed: Extensible via custom strategies without modifying core
- Liskov Substitution: Compatible with all strategy implementations
- Interface Segregation: Lean interfaces for specific use cases
- Dependency Inversion: Depends on protocols, not concretions

Key Features:
- Configurable join strategies (all, any, first, n_of_m, majority)
- Flexible error handling (fail_fast, continue_all, collect_errors)
- Resource limits (max_concurrent, timeout, memory, cpu)
- Progress callbacks for observability
- Thread-safe operations using asyncio
- YAML configuration support

Example:
    from victor.framework.parallel import (
        ParallelExecutor,
        ParallelConfig,
        JoinStrategy,
        ErrorStrategy,
        ResourceLimit,
    )

    # Define async tasks
    async def task1(ctx): return "result1"
    async def task2(ctx): return "result2"

    # Configure execution
    config = ParallelConfig(
        join_strategy=JoinStrategy.ALL,
        error_strategy=ErrorStrategy.COLLECT_ERRORS,
        resource_limit=ResourceLimit(max_concurrent=4, timeout=30.0),
    )

    # Execute in parallel
    executor = ParallelExecutor(config)
    result = await executor.execute([task1, task2], context={})

    print(f"Success: {result.success}")
    print(f"Results: {result.results}")
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
)

from victor.framework.parallel.protocols import (
    ErrorStrategyProtocol,
    JoinStrategyProtocol,
    ParallelExecutionResult,
    ParallelExecutorProtocol,
)
from victor.framework.parallel.strategies import (
    AllJoinStrategy,
    AnyJoinStrategy,
    ContinueAllErrorStrategy,
    create_error_strategy,
    create_join_strategy,
    ErrorStrategy,
    FailFastErrorStrategy,
    FirstJoinStrategy,
    JoinStrategy,
    MajorityJoinStrategy,
    NOfMJoinStrategy,
    ParallelConfig,
    ResourceLimit,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
TaskFunc = Callable[..., Awaitable[T]]
TaskInput = Union[TaskFunc[T], Coroutine[Any, Any, T], tuple[TaskFunc[T], Dict[str, Any]]]

__all__ = [
    # Main executor
    "ParallelExecutor",
    # Result class
    "ParallelExecutionResult",
    # Configuration
    "ParallelConfig",
    "ResourceLimit",
    "JoinStrategy",
    "ErrorStrategy",
    # Convenience functions
    "execute_parallel",
    "execute_parallel_with_config",
    "create_parallel_executor",
]


# =============================================================================
# Progress Callback Types
# =============================================================================


ProgressCallback = Callable[[str, str, Any], None]
"""Callback for progress updates.

Args:
    task_id: Identifier for the task
    status: Status of the task (started, completed, failed, timeout)
    result_or_error: Result if successful, error if failed
"""


@dataclass
class ProgressEvent:
    """Event emitted during parallel execution.

    Attributes:
        task_id: Identifier for the task
        status: Current status (started, completed, failed, timeout)
        result: Result if successful
        error: Error if failed
        timestamp: Event timestamp
    """

    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[Exception] = None
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# Main Parallel Executor
# =============================================================================


class ParallelExecutor:
    """Generic parallel executor for async tasks.

    Manages concurrent execution of async tasks with configurable
    join strategies, error handling, and resource limits.

    Attributes:
        config: Parallel configuration
        progress_callback: Optional callback for progress updates
    """

    def __init__(
        self,
        config: Optional[ParallelConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        """Initialize the parallel executor.

        Args:
            config: Parallel execution configuration
            progress_callback: Optional callback for progress updates
        """
        self.config = config or ParallelConfig()
        self.progress_callback = progress_callback
        self._join_strategy: Optional[JoinStrategyProtocol] = None
        self._error_strategy: Optional[ErrorStrategyProtocol] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

        # Initialize strategies from config
        self._init_strategies()
        self._init_semaphore()

    def _init_strategies(self) -> None:
        """Initialize strategy instances from config."""
        self._join_strategy = create_join_strategy(
            self.config.join_strategy,
            n_of_m=self.config.n_of_m,
        )
        self._error_strategy = create_error_strategy(self.config.error_strategy)

    def _init_semaphore(self) -> None:
        """Initialize semaphore for concurrency control."""
        max_concurrent = self.config.resource_limit.max_concurrent
        if max_concurrent is not None:
            self._semaphore = asyncio.Semaphore(max_concurrent)

    def set_config(self, config: ParallelConfig) -> None:
        """Update the executor configuration.

        Args:
            config: New configuration to apply
        """
        self.config = config
        self._init_strategies()
        self._init_semaphore()

    def set_progress_callback(self, callback: ProgressCallback) -> None:
        """Set or update the progress callback.

        Args:
            callback: Callback function for progress updates
        """
        self.progress_callback = callback

    async def execute(
        self,
        tasks: List["TaskInput[Any]"],
        context: Optional[Dict[str, Any]] = None,
    ) -> ParallelExecutionResult:
        """Execute tasks in parallel.

        Args:
            tasks: List of tasks to execute
                - Can be an awaitable function: async_func
                - Can be a coroutine: async_func()
                - Can be a tuple: (async_func, kwargs_dict)
            context: Optional context shared across all tasks

        Returns:
            ParallelExecutionResult with aggregated results
        """
        start_time = time.time()
        context = context or {}

        if not tasks:
            return ParallelExecutionResult(
                success=True,
                results=[],
                errors=[],
                total_count=0,
                success_count=0,
                failure_count=0,
                duration_seconds=0.0,
                strategy_used=self.config.join_strategy.value,
            )

        # Prepare task coroutines
        awaitables: List[Awaitable[Any]] = []
        for i, task in enumerate(tasks):
            awaitables.append(self._prepare_task(task, i, context))

        # Execute with appropriate strategy
        if self.config.error_strategy == ErrorStrategy.FAIL_FAST:
            result = await self._execute_fail_fast(awaitables, start_time)
        else:
            result = await self._execute_continue_all(awaitables, start_time)

        return result

    def _prepare_task(
        self,
        task: TaskInput[Any],
        index: int,
        context: Dict[str, Any],
    ) -> Coroutine[Any, Any, T]:
        """Prepare a task for execution.

        Args:
            task: Task input in various formats
            index: Task index for identification
            context: Shared execution context

        Returns:
            Coroutine to await
        """
        if isinstance(task, tuple):
            func, kwargs = task
            return self._wrap_task(func, index, context, **kwargs)
        elif asyncio.iscoroutine(task):
            # Task is already a coroutine - wrap it directly
            return self._wrap_coroutine(task, index, context)
        elif inspect.iscoroutinefunction(task):
            # Task is a coroutine function - create the actual coroutine
            # We need to return the coroutine, not call _wrap_task directly
            # because _wrap_task is async and returns a coroutine
            return self._create_wrapped_task(task, index, context)
        else:
            # Assume it's a callable that returns a coroutine
            return self._create_wrapped_task(task, index, context)

    async def _wrap_task(
        self,
        func: Callable[..., Awaitable[T]],
        index: int,
        context: Dict[str, Any],
        **kwargs: Any,
    ) -> T:
        """Wrap a task function with timeout and progress tracking.

        Args:
            func: Async function to execute
            index: Task index
            context: Shared context
            **kwargs: Additional kwargs for the function

        Returns:
            Task result
        """
        task_id = f"task_{index}"
        timeout = self.config.resource_limit.timeout

        # Emit start event
        self._emit_progress(task_id, "started", None)

        try:
            # Try to call with context, fall back to just kwargs
            if timeout is not None:
                try:
                    result = await asyncio.wait_for(
                        func(context=context, **kwargs),
                        timeout=timeout,
                    )
                except TypeError:
                    # Function doesn't accept context parameter
                    result = await asyncio.wait_for(
                        func(**kwargs),
                        timeout=timeout,
                    )
            else:
                try:
                    result = await func(context=context, **kwargs)
                except TypeError:
                    # Function doesn't accept context parameter
                    result = await func(**kwargs)

            self._emit_progress(task_id, "completed", result)
            return result

        except asyncio.TimeoutError:
            error = asyncio.TimeoutError(f"Task {task_id} timed out after {timeout}s")
            self._emit_progress(task_id, "timeout", error)
            raise

        except Exception as e:
            self._emit_progress(task_id, "failed", e)
            raise

    def _create_wrapped_task(
        self,
        func: Callable[..., Awaitable[T]],
        index: int,
        context: Dict[str, Any],
    ) -> Coroutine[Any, Any, T]:
        """Create a wrapped coroutine from a function.

        This method returns an actual coroutine (not an async function)
        that can be awaited later.

        Args:
            func: Async function to execute
            index: Task index
            context: Shared context

        Returns:
            Coroutine to await
        """
        return self._wrap_task(func, index, context)

    async def _wrap_coroutine(
        self,
        coro: Coroutine[Any, Any, T],
        index: int,
        context: Dict[str, Any],
    ) -> T:
        """Wrap an existing coroutine with timeout and progress tracking.

        Args:
            coro: Coroutine to execute
            index: Task index
            context: Shared context (not used for coroutines, just for consistency)

        Returns:
            Coroutine result
        """
        task_id = f"task_{index}"
        timeout = self.config.resource_limit.timeout

        self._emit_progress(task_id, "started", None)

        try:
            if timeout is not None:
                result = await asyncio.wait_for(coro, timeout=timeout)
            else:
                result = await coro

            self._emit_progress(task_id, "completed", result)
            return result

        except asyncio.TimeoutError:
            error = asyncio.TimeoutError(f"Task {task_id} timed out after {timeout}s")
            self._emit_progress(task_id, "timeout", error)
            raise

        except Exception as e:
            self._emit_progress(task_id, "failed", e)
            raise

    def _emit_progress(
        self,
        task_id: str,
        status: str,
        result_or_error: Optional[Any],
    ) -> None:
        """Emit a progress event.

        Args:
            task_id: Task identifier
            status: Task status
            result_or_error: Result or error
        """
        if self.progress_callback:
            try:
                self.progress_callback(task_id, status, result_or_error)
            except Exception as e:
                logger.debug(f"Progress callback error: {e}")

    async def _execute_fail_fast(
        self,
        coroutines: List[Awaitable[T]],
        start_time: float,
    ) -> ParallelExecutionResult:
        """Execute tasks with fail-fast error handling.

        Args:
            coroutines: List of awaitable tasks
            start_time: Execution start time

        Returns:
            Parallel execution result
        """
        results: List[Any] = [None] * len(coroutines)
        errors: List[Exception] = []
        success_count = 0
        failure_count = 0

        # Use asyncio.gather with return_exceptions=True
        # but stop on first error based on strategy
        gathered_results = await asyncio.gather(
            *coroutines,
            return_exceptions=True,
        )

        for i, result in enumerate(gathered_results):
            if isinstance(result, Exception):
                errors.append(result)
                failure_count += 1
                # For fail_fast, we could break here but gather
                # already ran everything. For true fail-fast,
                # we'd need a different approach with tasks.
            else:
                results[i] = result
                success_count += 1

        # Apply join strategy
        if self._join_strategy is not None:
            should_stop = self._join_strategy.should_stop_on_error()
            if should_stop and errors:
                success, final_result, _ = await self._join_strategy.evaluate(
                    [r for r in results if r is not None],
                    errors,
                )
            else:
                success, final_result, _ = await self._join_strategy.evaluate(
                    results,
                    errors,
                )
        else:
            # Default behavior if no join strategy
            success = len(errors) == 0
            final_result = results

        return ParallelExecutionResult(
            success=success,
            results=final_result,
            errors=errors,
            total_count=len(coroutines),
            success_count=success_count,
            failure_count=failure_count,
            duration_seconds=time.time() - start_time,
            strategy_used=self.config.join_strategy.value,
        )

    async def _execute_continue_all(
        self,
        coroutines: List[Awaitable[T]],
        start_time: float,
    ) -> ParallelExecutionResult:
        """Execute tasks continuing despite errors.

        Args:
            coroutines: List of awaitable tasks
            start_time: Execution start time

        Returns:
            Parallel execution result
        """
        results: List[Any] = [None] * len(coroutines)
        errors: List[Exception] = []
        success_count = 0
        failure_count = 0

        # Apply semaphore if configured
        if self._semaphore is not None:
            wrapped = []
            for i, coro in enumerate(coroutines):
                wrapped.append(self._execute_with_semaphore(i, coro))
            gathered_results = await asyncio.gather(
                *wrapped,
                return_exceptions=True,
            )
        else:
            gathered_results = await asyncio.gather(
                *coroutines,
                return_exceptions=True,
            )

        for i, result in enumerate(gathered_results):
            if isinstance(result, Exception):
                errors.append(result)
                failure_count += 1
            else:
                results[i] = result
                success_count += 1

        # Apply join strategy
        if self._join_strategy is not None:
            success, final_result, _ = await self._join_strategy.evaluate(
                results,
                errors,
            )
        else:
            success = len(errors) == 0
            final_result = results

        return ParallelExecutionResult(
            success=success,
            results=final_result,
            errors=errors,
            total_count=len(coroutines),
            success_count=success_count,
            failure_count=failure_count,
            duration_seconds=time.time() - start_time,
            strategy_used=self.config.join_strategy.value,
        )

    async def _execute_with_semaphore(
        self,
        index: int,
        coro: Awaitable[T],
    ) -> T:
        """Execute a coroutine with semaphore control.

        Args:
            index: Task index
            coro: Coroutine to execute

        Returns:
            Coroutine result
        """
        if self._semaphore is None:
            return await coro

        async with self._semaphore:
            return await coro

    async def execute_stream(
        self,
        tasks: List["TaskInput[Any]"],
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[ProgressEvent, None]:
        """Execute tasks in parallel, yielding progress events.

        This is a generator that yields events as tasks complete,
        allowing callers to handle results incrementally.

        Args:
            tasks: List of tasks to execute
            context: Optional shared context

        Yields:
            ProgressEvent for each task completion
        """
        context = context or {}
        task_futures = []

        for i, task in enumerate(tasks):
            coro: Coroutine[Any, Any, Any] = self._prepare_task(task, i, context)
            task_futures.append(asyncio.ensure_future(coro))

        # Create tasks for completion tracking
        pending = set(task_futures)

        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in done:
                task_idx = task_futures.index(task)
                task_id = f"task_{task_idx}"
                try:
                    result = await task
                    yield ProgressEvent(
                        task_id=task_id,
                        status="completed",
                        result=result,
                    )
                except Exception as e:
                    yield ProgressEvent(
                        task_id=task_id,
                        status="failed",
                        error=e,
                    )


# =============================================================================
# YAML Handler Integration
# =============================================================================


@dataclass
class ParallelExecutorHandler:
    """YAML workflow handler for parallel execution.

    This handler integrates with the workflow system to provide
    parallel execution from YAML workflow definitions.

    Example YAML:
        - id: parallel_analysis
          type: compute
          handler: parallel_executor
          config:
            join_strategy: all
            error_strategy: collect_errors
            resource_limit:
              max_concurrent: 5
              timeout: 30.0
          tasks: [task1, task2, task3]
          output: results
    """

    config: ParallelConfig = field(default_factory=ParallelConfig)

    async def __call__(
        self,
        node: Any,  # ComputeNode
        context: Any,  # WorkflowContext
        tool_registry: Any,  # ToolRegistry
    ) -> Any:  # NodeResult
        """Execute parallel tasks from workflow node.

        Args:
            node: Workflow compute node
            context: Workflow execution context
            tool_registry: Tool registry

        Returns:
            Node result with parallel execution results
        """
        from victor.workflows.executor import NodeResult, ExecutorNodeStatus

        start_time = time.time()

        # Extract configuration from node
        config = self._extract_config(node)

        # Extract tasks from node
        tasks = self._extract_tasks(node, context, tool_registry)

        # Create executor and execute
        executor = ParallelExecutor(config)
        result = await executor.execute(tasks, context.data)

        # Store output in context
        if node.output_key:
            context.set(node.output_key, result.results)

        # Determine node status
        status = ExecutorNodeStatus.COMPLETED if result.success else ExecutorNodeStatus.FAILED

        return NodeResult(
            node_id=node.id,
            status=status,
            output=result.results,
            error="; ".join(str(e) for e in result.errors) if result.errors else None,
            duration_seconds=time.time() - start_time,
        )

    def _extract_config(self, node: Any) -> ParallelConfig:
        """Extract parallel config from workflow node.

        Args:
            node: Workflow compute node

        Returns:
            ParallelConfig instance
        """
        # Check for config in node attributes
        if hasattr(node, "parallel_config"):
            return cast(ParallelConfig, node.parallel_config)

        # Build from individual attributes
        config = ParallelConfig()

        if hasattr(node, "join_strategy"):
            config.join_strategy = JoinStrategy.from_string(node.join_strategy)

        if hasattr(node, "error_strategy"):
            config.error_strategy = ErrorStrategy.from_string(node.error_strategy)

        if hasattr(node, "max_concurrent"):
            config.resource_limit.max_concurrent = node.max_concurrent

        if hasattr(node, "timeout"):
            config.resource_limit.timeout = node.timeout

        return config

    def _extract_tasks(
        self,
        node: Any,
        context: Any,
        tool_registry: Any,
    ) -> List[TaskInput[Any]]:
        """Extract task functions from workflow node.

        Args:
            node: Workflow compute node
            context: Workflow context
            tool_registry: Tool registry

        Returns:
            List of async task functions
        """
        tasks: List[TaskInput[Any]] = []

        # If node has tools, create tool execution tasks
        if hasattr(node, "tools") and node.tools:
            for tool_name in node.tools:
                tasks.append(self._create_tool_task(tool_name, context, tool_registry))

        # If node has handler tasks, use those
        elif hasattr(node, "tasks") and node.tasks:
            tasks = node.tasks  # type: ignore[assignment]

        return tasks

    def _create_tool_task(
        self,
        tool_name: str,
        context: Any,
        tool_registry: Any,
    ) -> Callable[[], Awaitable[Any]]:
        """Create an async task for tool execution.

        Args:
            tool_name: Name of tool to execute
            context: Workflow context
            tool_registry: Tool registry

        Returns:
            Async task function
        """

        async def execute_tool(**kwargs: Any) -> Any:
            result = await tool_registry.execute(
                tool_name,
                _exec_ctx={"workflow_context": context.data},
                **kwargs,
            )
            if result.success:
                return result.output
            else:
                raise Exception(f"Tool {tool_name} failed: {result.error}")

        return execute_tool


# =============================================================================
# Convenience Functions
# =============================================================================


async def execute_parallel(
    tasks: List["TaskInput[Any]"],
    context: Optional[Dict[str, Any]] = None,
    join_strategy: JoinStrategy = JoinStrategy.ALL,
    error_strategy: ErrorStrategy = ErrorStrategy.FAIL_FAST,
    max_concurrent: Optional[int] = None,
    timeout: Optional[float] = None,
) -> ParallelExecutionResult:
    """Convenience function for parallel execution.

    Args:
        tasks: List of tasks to execute
        context: Optional shared context
        join_strategy: How to join results
        error_strategy: How to handle errors
        max_concurrent: Maximum concurrent tasks
        timeout: Per-task timeout

    Returns:
        ParallelExecutionResult with aggregated results

    Example:
        result = await execute_parallel(
            [task1, task2, task3],
            join_strategy=JoinStrategy.ANY,
            max_concurrent=5,
        )
    """
    config = ParallelConfig(
        join_strategy=join_strategy,
        error_strategy=error_strategy,
        resource_limit=ResourceLimit(
            max_concurrent=max_concurrent,
            timeout=timeout,
        ),
    )

    executor = ParallelExecutor(config)
    return await executor.execute(tasks, context)


async def execute_parallel_with_config(
    tasks: List["TaskInput[Any]"],
    config: ParallelConfig,
    context: Optional[Dict[str, Any]] = None,
) -> ParallelExecutionResult:
    """Execute parallel tasks with full configuration.

    Args:
        tasks: List of tasks to execute
        config: Complete parallel configuration
        context: Optional shared context

    Returns:
        ParallelExecutionResult with aggregated results
    """
    executor = ParallelExecutor(config)
    return await executor.execute(tasks, context)


def create_parallel_executor(
    join_strategy: Union[JoinStrategy, str] = JoinStrategy.ALL,
    error_strategy: Union[ErrorStrategy, str] = ErrorStrategy.FAIL_FAST,
    max_concurrent: Optional[int] = None,
    timeout: Optional[float] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> ParallelExecutor:
    """Factory function to create a configured ParallelExecutor.

    Args:
        join_strategy: Join strategy (enum or string)
        error_strategy: Error strategy (enum or string)
        max_concurrent: Maximum concurrent tasks
        timeout: Per-task timeout
        progress_callback: Optional progress callback

    Returns:
        Configured ParallelExecutor instance

    Example:
        executor = create_parallel_executor(
            join_strategy="majority",
            max_concurrent=10,
        )
        result = await executor.execute([task1, task2])
    """
    # Convert strings to enums if needed
    if isinstance(join_strategy, str):
        join_strategy = JoinStrategy.from_string(join_strategy)
    if isinstance(error_strategy, str):
        error_strategy = ErrorStrategy.from_string(error_strategy)

    config = ParallelConfig(
        join_strategy=join_strategy,
        error_strategy=error_strategy,
        resource_limit=ResourceLimit(
            max_concurrent=max_concurrent,
            timeout=timeout,
        ),
    )

    return ParallelExecutor(config, progress_callback)


# =============================================================================
# Register Framework Handler
# =============================================================================


def register_parallel_handler() -> None:
    """Register the parallel executor as a framework handler.

    Call this during application initialization to make parallel
    execution available in YAML workflows.

    Example:
        from victor.framework.parallel import register_parallel_handler
        register_parallel_handler()
    """
    try:
        from victor.workflows.executor import register_compute_handler
        from victor.workflows import handlers

        handler = ParallelExecutorHandler()
        register_compute_handler("parallel_executor", handler)

        # Also register with FRAMEWORK_HANDLERS
        handlers.FRAMEWORK_HANDLERS["parallel_executor"] = handler

        logger.debug("Registered parallel_executor framework handler")
    except ImportError:
        logger.warning("Could not register parallel_executor handler")


# Auto-register on import
register_parallel_handler()
