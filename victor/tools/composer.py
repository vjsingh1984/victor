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

"""Tool composition with advanced execution patterns.

This module provides sophisticated tool orchestration:
- Tool composition and chaining
- Parallel tool execution
- Tool dependency resolution
- Result aggregation strategies
- Dynamic workflow generation
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from graphlib import TopologicalSorter
from typing import Any, Dict, List, Optional, Set, Tuple, Callable

logger = logging.getLogger(__name__)


class AggregationStrategy(str, Enum):
    """Strategies for aggregating tool results."""

    FIRST = "first"  # Return first successful result
    ALL = "all"  # Return all results
    MERGE = "merge"  # Merge results into single dict
    CONCAT = "concat"  # Concatenate list results
    VOTE = "vote"  # Vote on results
    CUSTOM = "custom"  # Custom aggregation function


class ExecutionStrategy(str, Enum):
    """Tool execution strategies."""

    SEQUENTIAL = "sequential"  # Execute one by one
    PARALLEL = "parallel"  # Execute all at once
    DEPENDENCY = "dependency"  # Respect dependencies
    ADAPTIVE = "adaptive"  # Adapt based on performance


@dataclass
class ToolSpec:
    """Specification for a tool in composition.

    Attributes:
        name: Tool name
        tool: Tool instance or callable
        inputs: Input parameter mapping
        dependencies: Tool dependencies
        timeout: Execution timeout
        retry_count: Number of retries
    """

    name: str
    tool: Any
    inputs: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 30.0
    retry_count: int = 0


@dataclass
class ToolResult:
    """Result from tool execution.

    Attributes:
        tool_name: Tool that produced result
        success: Whether execution succeeded
        result: Result value
        error: Error if failed
        duration: Execution duration
        metadata: Additional metadata
    """

    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositionPlan:
    """Plan for tool composition.

    Attributes:
        tools: List of tool specifications
        execution_strategy: How to execute tools
        aggregation_strategy: How to aggregate results
        max_parallelism: Maximum parallel executions
        fail_fast: Stop on first error
    """

    tools: List[ToolSpec]
    execution_strategy: ExecutionStrategy = ExecutionStrategy.DEPENDENCY
    aggregation_strategy: AggregationStrategy = AggregationStrategy.MERGE
    max_parallelism: int = 5
    fail_fast: bool = True


class ToolComposer:
    """Compose and execute tools with advanced patterns.

    Example:
        from victor.tools.composer import ToolComposer, CompositionPlan

        composer = ToolComposer()

        # Define tools
        plan = CompositionPlan(
            tools=[
                ToolSpec(name="search", tool=search_tool, inputs={"query": "test"}),
                ToolSpec(name="analyze", tool=analyze_tool, dependencies=["search"]),
            ],
            execution_strategy=ExecutionStrategy.DEPENDENCY,
            aggregation_strategy=AggregationStrategy.MERGE
        )

        # Execute composition
        result = await composer.execute_plan(plan)

        # Or use fluent API
        result = await (
            composer
            .add_tool(search_tool, inputs={"query": "test"})
            .add_tool(analyze_tool, depends_on=["search"])
            .execute()
        )
    """

    def __init__(self, orchestrator: Optional[Any] = None):
        """Initialize tool composer.

        Args:
            orchestrator: Agent orchestrator for tool execution
        """
        self.orchestrator = orchestrator
        self._tools: List[ToolSpec] = []
        self._execution_strategy = ExecutionStrategy.DEPENDENCY
        self._aggregation_strategy = AggregationStrategy.MERGE
        self._max_parallelism = 5
        self._fail_fast = True

    def add_tool(
        self,
        tool: Any,
        name: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
        timeout: float = 30.0,
    ) -> "ToolComposer":
        """Add tool to composition (fluent API).

        Args:
            tool: Tool instance or callable
            name: Optional tool name
            inputs: Input parameters
            depends_on: List of tool names this depends on
            timeout: Execution timeout

        Returns:
            Self for chaining
        """
        tool_name = name or str(getattr(tool, "name", str(id(tool))))

        spec = ToolSpec(
            name=tool_name,
            tool=tool,
            inputs=inputs or {},
            dependencies=depends_on or [],
            timeout=timeout,
        )

        self._tools.append(spec)
        return self

    def with_strategy(self, strategy: ExecutionStrategy) -> "ToolComposer":
        """Set execution strategy (fluent API)."""
        self._execution_strategy = strategy
        return self

    def with_aggregation(self, strategy: AggregationStrategy) -> "ToolComposer":
        """Set aggregation strategy (fluent API)."""
        self._aggregation_strategy = strategy
        return self

    def with_parallelism(self, max_parallel: int) -> "ToolComposer":
        """Set max parallelism (fluent API)."""
        self._max_parallelism = max_parallel
        return self

    def with_fail_fast(self, fail_fast: bool) -> "ToolComposer":
        """Set fail-fast behavior (fluent API)."""
        self._fail_fast = fail_fast
        return self

    async def execute(self) -> Any:
        """Execute composed tools (fluent API).

        Returns:
            Aggregated result
        """
        if not self._tools:
            return None

        plan = CompositionPlan(
            tools=self._tools,
            execution_strategy=self._execution_strategy,
            aggregation_strategy=self._aggregation_strategy,
            max_parallelism=self._max_parallelism,
            fail_fast=self._fail_fast,
        )

        result = await self.execute_plan(plan)

        # Reset state
        self._tools.clear()

        return result

    async def execute_plan(self, plan: CompositionPlan) -> Any:
        """Execute composition plan.

        Args:
            plan: Composition plan

        Returns:
            Aggregated result
        """
        if plan.execution_strategy == ExecutionStrategy.SEQUENTIAL:
            results = await self._execute_sequential(plan)
        elif plan.execution_strategy == ExecutionStrategy.PARALLEL:
            results = await self._execute_parallel(plan)
        elif plan.execution_strategy == ExecutionStrategy.DEPENDENCY:
            results = await self._execute_with_dependencies(plan)
        else:  # ADAPTIVE
            results = await self._execute_adaptive(plan)

        # Aggregate results
        return await self._aggregate_results(results, plan.aggregation_strategy)

    async def _execute_sequential(self, plan: CompositionPlan) -> List[ToolResult]:
        """Execute tools sequentially."""
        results = []

        for spec in plan.tools:
            result = await self._execute_tool(spec)
            results.append(result)

            if plan.fail_fast and not result.success:
                logger.error(f"Tool {spec.name} failed, stopping execution")
                break

        return results

    async def _execute_parallel(self, plan: CompositionPlan) -> List[ToolResult]:
        """Execute tools in parallel."""
        semaphore = asyncio.Semaphore(plan.max_parallelism)

        async def execute_with_semaphore(spec: ToolSpec) -> ToolResult:
            async with semaphore:
                return await self._execute_tool(spec)

        tasks = [execute_with_semaphore(spec) for spec in plan.tools]
        results = await asyncio.gather(*tasks, return_exceptions=not plan.fail_fast)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    ToolResult(
                        tool_name=plan.tools[i].name,
                        success=False,
                        error=str(result),
                    )
                )
            else:
                processed_results.append(result)  # type: ignore[arg-type]

        return processed_results

    async def _execute_with_dependencies(self, plan: CompositionPlan) -> List[ToolResult]:
        """Execute tools respecting dependencies."""
        # Build dependency graph
        graph = {spec.name: spec.dependencies for spec in plan.tools}

        # Topological sort
        sorter = TopologicalSorter(graph)
        execution_order = list(sorter.static_order())

        # Execute in order
        results_map: Dict[str, ToolResult] = {}

        for tool_name in execution_order:
            spec = next(s for s in plan.tools if s.name == tool_name)

            # Update inputs from dependencies
            inputs = spec.inputs.copy()
            for dep_name in spec.dependencies:
                if dep_name in results_map and results_map[dep_name].success:
                    # Pass dependency result as input
                    inputs[f"_{dep_name}_result"] = results_map[dep_name].result

            # Execute
            spec_with_inputs = ToolSpec(
                name=spec.name,
                tool=spec.tool,
                inputs=inputs,
                dependencies=spec.dependencies,
                timeout=spec.timeout,
            )

            result = await self._execute_tool(spec_with_inputs)
            results_map[tool_name] = result

            if plan.fail_fast and not result.success:
                logger.error(f"Tool {spec.name} failed, stopping execution")
                break

        return list(results_map.values())

    async def _execute_adaptive(self, plan: CompositionPlan) -> List[ToolResult]:
        """Adaptive execution based on tool characteristics."""
        # For now, use dependency-based execution
        # In production, could use ML to predict optimal strategy
        return await self._execute_with_dependencies(plan)

    async def _execute_tool(self, spec: ToolSpec) -> ToolResult:
        """Execute single tool."""
        import time

        start_time = time.time()

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._run_tool(spec),
                timeout=spec.timeout,
            )

            duration = time.time() - start_time

            return ToolResult(
                tool_name=spec.name,
                success=True,
                result=result,
                duration=duration,
            )

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            logger.error(f"Tool {spec.name} timed out after {spec.timeout}s")

            return ToolResult(
                tool_name=spec.name,
                success=False,
                error=f"Timeout after {spec.timeout}s",
                duration=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Tool {spec.name} failed: {e}")

            return ToolResult(
                tool_name=spec.name,
                success=False,
                error=str(e),
                duration=duration,
            )

    async def _run_tool(self, spec: ToolSpec) -> Any:
        """Run tool with inputs."""
        tool = spec.tool
        inputs = spec.inputs

        # Check if tool is async callable
        if asyncio.iscoroutinefunction(tool):
            return await tool(**inputs)
        elif callable(tool):
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: tool(**inputs))
        else:
            # Try to use tool.execute() method (BaseTool pattern)
            if hasattr(tool, "execute") and asyncio.iscoroutinefunction(tool.execute):
                return await tool.execute(**inputs)
            elif hasattr(tool, "execute"):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: tool.execute(**inputs))
            else:
                raise ValueError(f"Cannot execute tool: {spec.name}")

    async def _aggregate_results(
        self, results: List[ToolResult], strategy: AggregationStrategy
    ) -> Any:
        """Aggregate tool results."""
        if not results:
            return None

        if strategy == AggregationStrategy.FIRST:
            return self._aggregate_first(results)
        elif strategy == AggregationStrategy.ALL:
            return self._aggregate_all(results)
        elif strategy == AggregationStrategy.MERGE:
            return self._aggregate_merge(results)
        elif strategy == AggregationStrategy.CONCAT:
            return self._aggregate_concat(results)
        elif strategy == AggregationStrategy.VOTE:
            return await self._aggregate_vote(results)
        else:
            return self._aggregate_all(results)

    def _aggregate_first(self, results: List[ToolResult]) -> Any:
        """Return first successful result."""
        for result in results:
            if result.success:
                return result.result
        return None

    def _aggregate_all(self, results: List[ToolResult]) -> List[ToolResult]:
        """Return all results."""
        return results

    def _aggregate_merge(self, results: List[ToolResult]) -> Dict[str, Any]:
        """Merge results into single dict."""
        merged = {}

        for result in results:
            if result.success:
                if isinstance(result.result, dict):
                    merged.update(result.result)
                else:
                    merged[result.tool_name] = result.result

        return merged

    def _aggregate_concat(self, results: List[ToolResult]) -> List[Any]:
        """Concatenate list results."""
        concatenated = []

        for result in results:
            if result.success:
                if isinstance(result.result, list):
                    concatenated.extend(result.result)
                else:
                    concatenated.append(result.result)

        return concatenated

    async def _aggregate_vote(self, results: List[ToolResult]) -> Any:
        """Vote on results."""
        # Count occurrences of each result
        from collections import Counter

        successful_results = [r.result for r in results if r.success]

        if not successful_results:
            return None

        # For hashable results
        try:
            counter = Counter(successful_results)
            return counter.most_common(1)[0][0]
        except TypeError:
            # For unhashable results (e.g., dicts), return first
            return successful_results[0]

    def create_workflow_from_plan(self, plan: CompositionPlan) -> Dict[str, Any]:
        """Create workflow definition from plan.

        Args:
            plan: Composition plan

        Returns:
            Workflow definition dict
        """
        nodes = []

        for spec in plan.tools:
            node = {
                "id": spec.name,
                "type": "tool",
                "tool": spec.name,
                "inputs": spec.inputs,
            }

            if spec.dependencies:
                node["depends_on"] = spec.dependencies

            nodes.append(node)

        return {
            "workflow_id": f"composed_workflow_{id(plan)}",
            "nodes": nodes,
            "execution_strategy": plan.execution_strategy.value,
            "aggregation_strategy": plan.aggregation_strategy.value,
        }


def compose_tools(
    tools: List[Any],
    strategy: ExecutionStrategy = ExecutionStrategy.DEPENDENCY,
    aggregation: AggregationStrategy = AggregationStrategy.MERGE,
) -> ToolComposer:
    """Compose tools with given strategies.

    Args:
        tools: List of tools
        strategy: Execution strategy
        aggregation: Aggregation strategy

    Returns:
        ToolComposer instance

    Example:
        result = await compose_tools(
            [search_tool, analyze_tool],
            strategy=ExecutionStrategy.SEQUENTIAL
        ).execute()
    """
    composer = ToolComposer()
    composer._execution_strategy = strategy
    composer._aggregation_strategy = aggregation

    for tool in tools:
        composer.add_tool(tool)

    return composer


def parallel_tools(
    *tools: Any,
    max_parallelism: int = 5,
) -> ToolComposer:
    """Compose tools for parallel execution.

    Args:
        *tools: Tools to execute in parallel
        max_parallelism: Maximum parallel executions

    Returns:
        ToolComposer instance

    Example:
        result = await parallel_tools(
            search_tool,
            fetch_tool,
            max_parallelism=3
        ).execute()
    """
    composer = ToolComposer()
    composer._execution_strategy = ExecutionStrategy.PARALLEL
    composer._max_parallelism = max_parallelism

    for tool in tools:
        composer.add_tool(tool)

    return composer


def chain_tools(*tools: Any) -> ToolComposer:
    """Chain tools for sequential execution with passing results.

    Args:
        *tools: Tools to chain

    Returns:
        ToolComposer instance

    Example:
        result = await chain_tools(
            search_tool,
            analyze_tool,
            summarize_tool
        ).execute()
    """
    composer = ToolComposer()
    composer._execution_strategy = ExecutionStrategy.DEPENDENCY

    prev_tool = None
    for tool in tools:
        tool_name = getattr(tool, "name", str(id(tool)))

        if prev_tool:
            composer.add_tool(tool, name=tool_name, depends_on=[prev_tool])
        else:
            composer.add_tool(tool, name=tool_name)

        prev_tool = tool_name

    return composer
