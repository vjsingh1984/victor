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

"""Parallel tool execution for improved performance.

This module provides parallel execution of independent tool calls
while respecting dependencies and resource constraints.

Design Principles:
- Reuses existing ToolExecutor infrastructure
- Respects tool dependencies (write ops wait for reads)
- Configurable concurrency limits
- Progress callbacks for streaming updates
- Graceful error handling with partial results
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from victor.agent.tool_executor import ToolExecutionResult, ToolExecutor

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Tool categories for dependency analysis."""

    READ_ONLY = "read_only"  # Can run in parallel with other reads
    WRITE = "write"  # Must wait for all reads to complete
    COMPUTE = "compute"  # CPU-bound, can run in parallel
    NETWORK = "network"  # I/O-bound, can run in parallel


# Tool categorization for parallel execution decisions
TOOL_CATEGORIES: Dict[str, ToolCategory] = {
    # Read-only tools - safe to parallelize
    "read_file": ToolCategory.READ_ONLY,
    "list_directory": ToolCategory.READ_ONLY,
    "code_search": ToolCategory.READ_ONLY,
    "semantic_code_search": ToolCategory.READ_ONLY,
    "grep_search": ToolCategory.READ_ONLY,
    "plan_files": ToolCategory.READ_ONLY,
    "git": ToolCategory.READ_ONLY,  # Most git ops are read-only
    # Write tools - serialize to prevent conflicts
    "write_file": ToolCategory.WRITE,
    "edit_files": ToolCategory.WRITE,
    "execute_bash": ToolCategory.WRITE,  # Could modify filesystem
    "docker": ToolCategory.WRITE,
    # Compute tools - can parallelize
    "code_review": ToolCategory.COMPUTE,
    "refactor": ToolCategory.COMPUTE,
    "analyze_dependencies": ToolCategory.COMPUTE,
    # Network tools - I/O bound, can parallelize
    "web_search": ToolCategory.NETWORK,
    "web_fetch": ToolCategory.NETWORK,
    "http_request": ToolCategory.NETWORK,
}


@dataclass
class ParallelExecutionConfig:
    """Configuration for parallel tool execution."""

    max_concurrent: int = 5  # Max tools to run simultaneously
    enable_parallel: bool = True  # Global toggle
    parallelize_reads: bool = True  # Run read-only tools in parallel
    parallelize_network: bool = True  # Run network I/O in parallel
    timeout_per_tool: float = 60.0  # Per-tool timeout in seconds
    batch_size: int = 10  # Max tools per batch


@dataclass
class ParallelExecutionResult:
    """Result of parallel tool execution."""

    results: List[ToolExecutionResult] = field(default_factory=list)
    total_time: float = 0.0
    parallel_speedup: float = 1.0  # Estimated speedup from parallelization
    errors: List[str] = field(default_factory=list)
    completed_count: int = 0
    failed_count: int = 0


class ParallelToolExecutor:
    """Executes multiple tools in parallel when safe.

    This executor analyzes tool calls to determine which can safely
    run in parallel and which must be serialized due to dependencies.

    Example:
        executor = ParallelToolExecutor(tool_executor, config)
        results = await executor.execute_parallel(tool_calls, context)
    """

    def __init__(
        self,
        tool_executor: ToolExecutor,
        config: Optional[ParallelExecutionConfig] = None,
        progress_callback: Optional[Callable[[str, str, bool], None]] = None,
    ):
        """Initialize parallel executor.

        Args:
            tool_executor: Base tool executor for individual tool calls
            config: Parallel execution configuration
            progress_callback: Optional callback(tool_name, status, success)
        """
        self.executor = tool_executor
        self.config = config or ParallelExecutionConfig()
        self.progress_callback = progress_callback

    def _get_category(self, tool_name: str) -> ToolCategory:
        """Get category for a tool, defaulting to COMPUTE."""
        return TOOL_CATEGORIES.get(tool_name, ToolCategory.COMPUTE)

    def _can_parallelize(self, tool_calls: List[Dict[str, Any]]) -> bool:
        """Check if tool calls can be parallelized."""
        if not self.config.enable_parallel:
            return False

        if len(tool_calls) <= 1:
            return False

        categories = [self._get_category(tc.get("name", "")) for tc in tool_calls]

        # Don't parallelize if any writes present (need ordering)
        if ToolCategory.WRITE in categories:
            return False

        return True

    def _group_by_category(
        self, tool_calls: List[Dict[str, Any]]
    ) -> Dict[ToolCategory, List[Dict[str, Any]]]:
        """Group tool calls by category for batching."""
        groups: Dict[ToolCategory, List[Dict[str, Any]]] = {}
        for tc in tool_calls:
            category = self._get_category(tc.get("name", ""))
            if category not in groups:
                groups[category] = []
            groups[category].append(tc)
        return groups

    def _extract_file_dependencies(self, tool_calls: List[Dict[str, Any]]) -> Dict[int, Set[int]]:
        """Extract file-based dependencies between tool calls.

        Returns a dict mapping tool index to set of indices it depends on.
        """
        dependencies: Dict[int, Set[int]] = {i: set() for i in range(len(tool_calls))}
        path_to_writers: Dict[str, List[int]] = {}

        for i, tc in enumerate(tool_calls):
            name = tc.get("name", "")
            args = tc.get("arguments", {})

            # Track paths this tool writes to
            if self._get_category(name) == ToolCategory.WRITE:
                path = args.get("path", args.get("file_path", ""))
                if path:
                    if path not in path_to_writers:
                        path_to_writers[path] = []
                    path_to_writers[path].append(i)

            # Check if this tool reads from a path written by earlier tool
            if self._get_category(name) == ToolCategory.READ_ONLY:
                path = args.get("path", args.get("file_path", ""))
                if path and path in path_to_writers:
                    dependencies[i].update(path_to_writers[path])

        return dependencies

    async def execute_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> ParallelExecutionResult:
        """Execute tool calls with parallelization where safe.

        Args:
            tool_calls: List of tool call dicts with name/arguments
            context: Shared context passed to all tools

        Returns:
            ParallelExecutionResult with all results and metrics
        """
        import time

        start_time = time.time()
        result = ParallelExecutionResult()

        if not tool_calls:
            return result

        context = context or {}

        # Check if parallelization is beneficial
        if not self._can_parallelize(tool_calls):
            # Sequential execution
            for tc in tool_calls:
                exec_result = await self._execute_single(tc, context)
                result.results.append(exec_result)
                if exec_result.success:
                    result.completed_count += 1
                else:
                    result.failed_count += 1
                    if exec_result.error:
                        result.errors.append(exec_result.error)

            result.total_time = time.time() - start_time
            result.parallel_speedup = 1.0
            return result

        # Parallel execution with dependency analysis
        dependencies = self._extract_file_dependencies(tool_calls)

        # Execute in waves respecting dependencies
        sequential_time_estimate = 0.0
        pending = set(range(len(tool_calls)))
        completed: Set[int] = set()
        results_by_index: Dict[int, ToolExecutionResult] = {}

        while pending:
            # Find tools with all dependencies satisfied
            ready = [i for i in pending if not dependencies[i] - completed]

            if not ready:
                # Deadlock - shouldn't happen with proper dependency analysis
                logger.error("Deadlock detected in parallel execution")
                for i in pending:
                    results_by_index[i] = ToolExecutionResult(
                        tool_name=tool_calls[i].get("name", "unknown"),
                        success=False,
                        result=None,
                        error="Deadlock in parallel execution",
                    )
                    result.failed_count += 1
                    result.errors.append("Deadlock in parallel execution")
                break

            # Limit batch size
            batch = ready[: self.config.batch_size]

            # Execute batch in parallel
            tasks = [self._execute_single(tool_calls[i], context) for i in batch]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for idx, res in zip(batch, batch_results, strict=True):
                exec_result: ToolExecutionResult
                if isinstance(res, Exception):
                    exec_result = ToolExecutionResult(
                        tool_name=tool_calls[idx].get("name", "unknown"),
                        success=False,
                        result=None,
                        error=str(res),
                    )
                    result.failed_count += 1
                    result.errors.append(str(res))
                elif isinstance(res, ToolExecutionResult):
                    exec_result = res
                    if exec_result.success:
                        result.completed_count += 1
                    else:
                        result.failed_count += 1
                        if exec_result.error:
                            result.errors.append(exec_result.error)
                else:
                    # Unexpected type - treat as failure
                    exec_result = ToolExecutionResult(
                        tool_name=tool_calls[idx].get("name", "unknown"),
                        success=False,
                        result=None,
                        error=f"Unexpected result type: {type(res)}",
                    )
                    result.failed_count += 1

                results_by_index[idx] = exec_result
                completed.add(idx)
                pending.discard(idx)

                # Estimate sequential time
                sequential_time_estimate += exec_result.execution_time

        # Collect results in original order
        for i in range(len(tool_calls)):
            if i in results_by_index:
                result.results.append(results_by_index[i])

        result.total_time = time.time() - start_time
        if result.total_time > 0:
            result.parallel_speedup = sequential_time_estimate / result.total_time
        else:
            result.parallel_speedup = 1.0

        logger.info(
            f"Parallel execution: {len(tool_calls)} tools in {result.total_time:.2f}s "
            f"(speedup: {result.parallel_speedup:.2f}x)"
        )

        return result

    async def _execute_single(
        self,
        tool_call: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ToolExecutionResult:
        """Execute a single tool call with progress callback."""
        tool_name = tool_call.get("name", "unknown")
        arguments = tool_call.get("arguments", {})

        if self.progress_callback:
            self.progress_callback(tool_name, "started", True)

        try:
            result = await asyncio.wait_for(
                self.executor.execute(
                    tool_name=tool_name,
                    arguments=arguments,
                    context=context,
                ),
                timeout=self.config.timeout_per_tool,
            )

            if self.progress_callback:
                self.progress_callback(tool_name, "completed", result.success)

            return result

        except asyncio.TimeoutError:
            if self.progress_callback:
                self.progress_callback(tool_name, "timeout", False)

            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool '{tool_name}' timed out after {self.config.timeout_per_tool}s",
            )
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(tool_name, "error", False)

            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
            )


def create_parallel_executor(
    tool_executor: ToolExecutor,
    max_concurrent: int = 5,
    enable: bool = True,
    progress_callback: Optional[Callable[[str, str, bool], None]] = None,
) -> ParallelToolExecutor:
    """Factory function to create a parallel executor.

    Args:
        tool_executor: Base tool executor
        max_concurrent: Maximum concurrent tool executions
        enable: Whether to enable parallelization
        progress_callback: Optional progress callback

    Returns:
        Configured ParallelToolExecutor
    """
    config = ParallelExecutionConfig(
        max_concurrent=max_concurrent,
        enable_parallel=enable,
    )
    return ParallelToolExecutor(
        tool_executor=tool_executor,
        config=config,
        progress_callback=progress_callback,
    )
