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
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from victor.agent.tool_executor import ToolExecutionResult, ToolExecutor

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    READ_ONLY = "read_only"
    WRITE = "write"
    COMPUTE = "compute"
    NETWORK = "network"


# Optimized tool categorization using frozenset for O(1) lookups
READ_TOOLS = frozenset(
    {
        "read_file",
        "list_directory",
        "code_search",
        "semantic_code_search",
        "grep_search",
        "plan_files",
        "git",
    }
)
WRITE_TOOLS = frozenset({"write_file", "edit_files", "execute_bash", "docker"})
NETWORK_TOOLS = frozenset({"web_search", "web_fetch", "http_request"})

# TOOL_CATEGORIES dictionary for backward compatibility with tests
TOOL_CATEGORIES: Dict[str, ToolCategory] = {
    **dict.fromkeys(READ_TOOLS, ToolCategory.READ_ONLY),
    **dict.fromkeys(WRITE_TOOLS, ToolCategory.WRITE),
    **dict.fromkeys(NETWORK_TOOLS, ToolCategory.NETWORK),
}


@dataclass
class ParallelExecutionConfig:
    max_concurrent: int = 5
    enable_parallel: bool = True
    parallelize_reads: bool = True
    timeout_per_tool: float = 60.0


@dataclass
class ParallelExecutionResult:
    results: List[ToolExecutionResult] = field(default_factory=list)
    total_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    completed_count: int = 0
    failed_count: int = 0
    parallel_speedup: float = 1.0  # Speedup ratio vs sequential execution


class ParallelToolExecutor:
    def __init__(
        self,
        tool_executor: ToolExecutor,
        config: Optional[ParallelExecutionConfig] = None,
        progress_callback: Optional[Callable[[str, str, bool], None]] = None,
    ):
        self.executor = tool_executor
        self.config = config or ParallelExecutionConfig()
        self.progress_callback = progress_callback

    def _get_category(self, tool_name: str) -> ToolCategory:
        if tool_name in READ_TOOLS:
            return ToolCategory.READ_ONLY
        elif tool_name in WRITE_TOOLS:
            return ToolCategory.WRITE
        elif tool_name in NETWORK_TOOLS:
            return ToolCategory.NETWORK
        return ToolCategory.COMPUTE

    def _has_write_tools(self, tool_calls: List[Dict[str, Any]]) -> bool:
        return any(tc.get("name", "") in WRITE_TOOLS for tc in tool_calls)

    def _can_parallelize(self, tool_calls: List[Dict[str, Any]]) -> bool:
        """Check if the given tool calls can be parallelized.

        Args:
            tool_calls: List of tool call dictionaries

        Returns:
            True if calls can be parallelized, False otherwise
        """
        # Cannot parallelize if disabled
        if not self.config.enable_parallel:
            return False
        # Cannot parallelize single or empty calls
        if len(tool_calls) <= 1:
            return False
        # Cannot parallelize if writes are present
        if self._has_write_tools(tool_calls):
            return False
        return True

    def _extract_file_dependencies(self, tool_calls: List[Dict[str, Any]]) -> Dict[int, Set[int]]:
        dependencies = {i: set() for i in range(len(tool_calls))}
        path_writers = {}

        for i, tc in enumerate(tool_calls):
            name = tc.get("name", "")
            args = tc.get("arguments", {})
            path = args.get("path") or args.get("file_path")

            if not path:
                continue

            if name in WRITE_TOOLS:
                path_writers.setdefault(path, []).append(i)
            elif name in READ_TOOLS and path in path_writers:
                dependencies[i].update(path_writers[path])

        return dependencies

    async def execute_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> ParallelExecutionResult:
        start_time = time.time()
        result = ParallelExecutionResult()

        if not tool_calls:
            return result

        context = context or {}

        # Simple parallelization: skip if writes present or disabled
        if (
            not self.config.enable_parallel
            or len(tool_calls) <= 1
            or self._has_write_tools(tool_calls)
        ):
            return await self._execute_sequential(tool_calls, context, result, start_time)

        # Parallel execution with dependency handling
        dependencies = self._extract_file_dependencies(tool_calls)
        pending = set(range(len(tool_calls)))
        completed = set()
        results_by_index = {}

        while pending:
            ready = [i for i in pending if not (dependencies[i] - completed)]

            if not ready:
                # Handle deadlock
                for i in pending:
                    results_by_index[i] = self._create_error_result(
                        tool_calls[i].get("name", "unknown"), "Dependency deadlock"
                    )
                    result.failed_count += 1
                break

            # Execute batch with concurrency limit
            batch = ready[: self.config.max_concurrent]
            tasks = [self._execute_single(tool_calls[i], context) for i in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for idx, res in zip(batch, batch_results, strict=False):
                if isinstance(res, Exception):
                    exec_result = self._create_error_result(
                        tool_calls[idx].get("name", "unknown"), str(res)
                    )
                    result.failed_count += 1
                else:
                    exec_result = res
                    if exec_result.success:
                        result.completed_count += 1
                    else:
                        result.failed_count += 1
                        if exec_result.error:
                            result.errors.append(exec_result.error)

                results_by_index[idx] = exec_result
                completed.add(idx)
                pending.discard(idx)

        # Collect results in order
        result.results = [
            results_by_index[i] for i in range(len(tool_calls)) if i in results_by_index
        ]
        result.total_time = time.time() - start_time

        return result

    async def _execute_sequential(
        self,
        tool_calls: List[Dict[str, Any]],
        context: Dict[str, Any],
        result: ParallelExecutionResult,
        start_time: float,
    ) -> ParallelExecutionResult:
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
        return result

    def _create_error_result(self, tool_name: str, error: str) -> ToolExecutionResult:
        return ToolExecutionResult(
            tool_name=tool_name,
            success=False,
            result=None,
            error=error,
        )

    async def _execute_single(
        self, tool_call: Dict[str, Any], context: Dict[str, Any]
    ) -> ToolExecutionResult:
        tool_name = tool_call.get("name", "unknown")
        arguments = tool_call.get("arguments", {})

        if self.progress_callback:
            self.progress_callback(tool_name, "started", True)

        try:
            result = await asyncio.wait_for(
                self.executor.execute(tool_name=tool_name, arguments=arguments, context=context),
                timeout=self.config.timeout_per_tool,
            )

            if self.progress_callback:
                self.progress_callback(tool_name, "completed", result.success)

            return result

        except asyncio.TimeoutError:
            error_msg = f"Tool '{tool_name}' timed out after {self.config.timeout_per_tool}s"
            if self.progress_callback:
                self.progress_callback(tool_name, "timeout", False)
            return self._create_error_result(tool_name, error_msg)

        except Exception as e:
            if self.progress_callback:
                self.progress_callback(tool_name, "error", False)
            return self._create_error_result(tool_name, str(e))


def create_parallel_executor(
    tool_executor: ToolExecutor,
    max_concurrent: int = 5,
    enable: bool = True,
    progress_callback: Optional[Callable[[str, str, bool], None]] = None,
) -> ParallelToolExecutor:
    config = ParallelExecutionConfig(max_concurrent=max_concurrent, enable_parallel=enable)
    return ParallelToolExecutor(tool_executor, config, progress_callback)
