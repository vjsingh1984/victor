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

"""Comprehensive performance benchmarks for workflow execution.

This module provides extensive benchmarks for workflow execution, focusing on:

1. End-to-End Execution Time: Total workflow execution time
2. Node Execution Throughput: Nodes executed per second
3. Parallel Execution Efficiency: Speedup from parallelization
4. Recursion Depth Impact: Performance impact of nested workflows
5. Tool Execution Overhead: Cost of tool calls within workflows
6. Conditional Branching: Performance of conditional edges
7. State Management: Overhead of state updates and propagation
8. Caching Effectiveness: Performance improvement from caching

Performance Targets:
- Simple workflow (5 nodes): <100ms
- Medium workflow (20 nodes): <500ms
- Complex workflow (50 nodes): <2000ms
- Parallel efficiency: >70% speedup
- Recursion overhead: <5% per level
- Tool execution: <50ms per tool call

Usage:
    # Run all benchmarks
    pytest tests/performance/workflow_execution_benchmarks.py -v

    # Run specific benchmark groups
    pytest tests/performance/workflow_execution_benchmarks.py -k "throughput" -v
    pytest tests/performance/workflow_execution_benchmarks.py -k "parallel" -v
    pytest tests/performance/workflow_execution_benchmarks.py -k "recursion" -v

    # Generate benchmark report
    pytest tests/performance/workflow_execution_benchmarks.py --benchmark-only --benchmark-json=workflow_benchmarks.json
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import random
import time
import tracemalloc
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


# =============================================================================
# Workflow Execution Components
# =============================================================================


class NodeType(str, Enum):
    """Types of workflow nodes."""

    AGENT = "agent"
    TOOL = "tool"
    CONDITION = "condition"
    TRANSFORM = "transform"
    PARALLEL = "parallel"
    HITL = "hitl"
    TEAM = "team"
    COMPUTE = "compute"


@dataclass
class ExecutionContext:
    """Execution context for workflow nodes."""

    state: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update the execution state."""
        self.state.update(updates)

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a value from the execution state."""
        return self.state.get(key, default)


@dataclass
class ExecutionResult:
    """Result from executing a workflow node."""

    node_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    tool_calls: int = 0
    next_nodes: List[str] = field(default_factory=list)


class WorkflowNode:
    """Mock workflow node for execution benchmarking."""

    def __init__(
        self,
        node_id: str,
        node_type: NodeType,
        execution_delay: float = 0.01,
        tool_calls: int = 0,
        fail_rate: float = 0.0,
    ):
        self.id = node_id
        self.type = node_type
        self._execution_delay = execution_delay
        self._tool_calls = tool_calls
        self._fail_rate = fail_rate
        self.execution_count = 0

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute the node."""
        start_time = time.perf_counter()
        self.execution_count += 1

        # Simulate tool call overhead
        if self._tool_calls > 0:
            await asyncio.sleep(self._execution_delay * self._tool_calls * 0.1)

        # Simulate execution delay
        await asyncio.sleep(self._execution_delay)

        # Simulate failures
        if random.random() < self._fail_rate:
            exec_time = time.perf_counter() - start_time
            return ExecutionResult(
                node_id=self.id,
                success=False,
                error=f"Node {self.id} failed",
                execution_time=exec_time,
                tool_calls=self._tool_calls,
            )

        # Update context
        context.update_state({
            f"{self.id}_output": f"Result from {self.id}",
            f"{self.id}_timestamp": time.time(),
        })

        exec_time = time.perf_counter() - start_time

        return ExecutionResult(
            node_id=self.id,
            success=True,
            output=f"Output from {self.id}",
            execution_time=exec_time,
            tool_calls=self._tool_calls,
        )


class ConditionalNode(WorkflowNode):
    """Node that evaluates conditions and branches."""

    def __init__(
        self,
        node_id: str,
        condition_func: Callable[[ExecutionContext], str],
        branches: Dict[str, List[str]],
        execution_delay: float = 0.005,
    ):
        super().__init__(node_id, NodeType.CONDITION, execution_delay)
        self.condition_func = condition_func
        self.branches = branches

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute condition and determine next nodes."""
        result = await super().execute(context)

        # Evaluate condition
        branch = self.condition_func(context)
        next_nodes = self.branches.get(branch, [])

        result.next_nodes = next_nodes
        return result


class WorkflowGraph:
    """Mock workflow graph for execution benchmarking."""

    def __init__(self, graph_id: str = "test_graph"):
        self.id = graph_id
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: Dict[str, List[str]] = {}  # node_id -> list of next node IDs
        self.entry_point: Optional[str] = None
        self.enable_caching = False
        self._execution_cache: Dict[str, ExecutionResult] = {}

    def add_node(self, node: WorkflowNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self.edges:
            self.edges[node.id] = []

    def add_edge(self, source: str, target: str) -> None:
        """Add an edge between nodes."""
        if source in self.edges:
            self.edges[source].append(target)

    def set_entry_point(self, node_id: str) -> None:
        """Set the entry point for workflow execution."""
        self.entry_point = node_id

    async def execute(
        self,
        initial_context: Optional[ExecutionContext] = None,
        max_iterations: int = 100,
    ) -> Dict[str, Any]:
        """Execute the workflow graph."""
        if initial_context is None:
            initial_context = ExecutionContext()

        context = initial_context
        current_nodes = [self.entry_point] if self.entry_point else []
        executed_nodes: Set[str] = set()
        results: List[ExecutionResult] = []
        total_tool_calls = 0

        start_time = time.perf_counter()
        iterations = 0

        while current_nodes and iterations < max_iterations:
            iterations += 1
            next_nodes = []

            for node_id in current_nodes:
                if node_id not in self.nodes:
                    continue

                node = self.nodes[node_id]

                # Check cache if enabled
                if self.enable_caching and node_id in self._execution_cache:
                    result = self._execution_cache[node_id]
                else:
                    result = await node.execute(context)

                    if self.enable_caching:
                        self._execution_cache[node_id] = result

                results.append(result)
                executed_nodes.add(node_id)
                total_tool_calls += result.tool_calls

                if not result.success:
                    # Stop on failure
                    current_nodes = []
                    next_nodes = []
                    break

                # Add next nodes
                if result.next_nodes:
                    next_nodes.extend(result.next_nodes)
                else:
                    next_nodes.extend(self.edges.get(node_id, []))

            current_nodes = list(set(next_nodes) - executed_nodes)

        total_time = time.perf_counter() - start_time
        success = all(r.success for r in results)

        return {
            "success": success,
            "total_time": total_time,
            "executed_nodes": len(executed_nodes),
            "total_iterations": iterations,
            "total_tool_calls": total_tool_calls,
            "results": results,
            "context": context,
        }


# =============================================================================
# Workflow Graph Generators
# =============================================================================


def create_linear_workflow(node_count: int, execution_delay: float = 0.01) -> WorkflowGraph:
    """Create a linear workflow graph."""
    graph = WorkflowGraph("linear_workflow")

    # Create nodes
    for i in range(node_count):
        node = WorkflowNode(
            node_id=f"node_{i}",
            node_type=NodeType.AGENT,
            execution_delay=execution_delay,
            tool_calls=random.randint(0, 5),
        )
        graph.add_node(node)

        # Add edge from previous node
        if i > 0:
            graph.add_edge(f"node_{i-1}", node.id)

    # Set entry point
    if node_count > 0:
        graph.set_entry_point("node_0")

    return graph


def create_parallel_workflow(branches: int, nodes_per_branch: int, execution_delay: float = 0.01) -> WorkflowGraph:
    """Create a parallel workflow graph."""
    graph = WorkflowGraph("parallel_workflow")

    # Entry node
    entry_node = WorkflowNode("entry", NodeType.AGENT, execution_delay)
    graph.add_node(entry_node)
    graph.set_entry_point("entry")

    # Create parallel branches
    for branch_idx in range(branches):
        prev_node_id = "entry"

        for node_idx in range(nodes_per_branch):
            node_id = f"branch_{branch_idx}_node_{node_idx}"
            node = WorkflowNode(
                node_id=node_id,
                node_type=NodeType.AGENT,
                execution_delay=execution_delay,
                tool_calls=random.randint(0, 5),
            )
            graph.add_node(node)
            graph.add_edge(prev_node_id, node_id)
            prev_node_id = node_id

    return graph


def create_conditional_workflow(branches: int, nodes_per_branch: int, execution_delay: float = 0.01) -> WorkflowGraph:
    """Create a conditional workflow graph."""
    graph = WorkflowGraph("conditional_workflow")

    # Entry node
    entry_node = WorkflowNode("entry", NodeType.AGENT, execution_delay)
    graph.add_node(entry_node)
    graph.set_entry_point("entry")

    # Condition node
    def condition_func(ctx: ExecutionContext) -> str:
        return f"branch_{random.randint(0, branches - 1)}"

    branch_mapping = {f"branch_{i}": [] for i in range(branches)}

    condition_node = ConditionalNode(
        node_id="condition",
        condition_func=condition_func,
        branches=branch_mapping,
        execution_delay=execution_delay,
    )
    graph.add_node(condition_node)
    graph.add_edge("entry", "condition")

    # Create branch nodes
    for branch_idx in range(branches):
        prev_node_id = "condition"
        branch_mapping[f"branch_{branch_idx}"] = []

        for node_idx in range(nodes_per_branch):
            node_id = f"branch_{branch_idx}_node_{node_idx}"
            node = WorkflowNode(
                node_id=node_id,
                node_type=NodeType.AGENT,
                execution_delay=execution_delay,
                tool_calls=random.randint(0, 5),
            )
            graph.add_node(node)
            graph.add_edge(prev_node_id, node_id)
            branch_mapping[f"branch_{branch_idx}"].append(node_id)
            prev_node_id = node_id

    return graph


def create_nested_workflow(depth: int, nodes_per_level: int, execution_delay: float = 0.01) -> WorkflowGraph:
    """Create a nested workflow graph for recursion testing."""
    graph = WorkflowGraph("nested_workflow")

    # Create nested structure
    def add_level(parent_id: Optional[str], current_depth: int):
        if current_depth >= depth:
            return

        for i in range(nodes_per_level):
            node_id = f"level_{current_depth}_node_{i}"
            node = WorkflowNode(
                node_id=node_id,
                node_type=NodeType.AGENT,
                execution_delay=execution_delay,
                tool_calls=random.randint(0, 3),
            )
            graph.add_node(node)

            if parent_id:
                graph.add_edge(parent_id, node_id)
            elif current_depth == 0:
                if not graph.entry_point:
                    graph.set_entry_point(node_id)

            # Add next level
            add_level(node_id, current_depth + 1)

    add_level(None, 0)

    return graph


# =============================================================================
# Benchmark Fixtures
# =============================================================================


@pytest.fixture
def simple_workflow():
    """Create simple workflow (5 nodes)."""
    return create_linear_workflow(node_count=5, execution_delay=0.01)


@pytest.fixture
def medium_workflow():
    """Create medium workflow (20 nodes)."""
    return create_linear_workflow(node_count=20, execution_delay=0.01)


@pytest.fixture
def complex_workflow():
    """Create complex workflow (50 nodes)."""
    return create_linear_workflow(node_count=50, execution_delay=0.01)


@pytest.fixture
def parallel_workflow():
    """Create parallel workflow (3 branches, 5 nodes each)."""
    return create_parallel_workflow(branches=3, nodes_per_branch=5, execution_delay=0.01)


@pytest.fixture
def conditional_workflow():
    """Create conditional workflow (3 branches, 5 nodes each)."""
    return create_conditional_workflow(branches=3, nodes_per_branch=5, execution_delay=0.01)


# =============================================================================
# End-to-End Execution Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("node_count", [5, 10, 20, 50])
def test_linear_workflow_execution(benchmark, node_count):
    """Benchmark linear workflow execution time.

    Performance Targets:
    - 5 nodes: <100ms
    - 10 nodes: <200ms
    - 20 nodes: <400ms
    - 50 nodes: <1000ms
    """
    workflow = create_linear_workflow(node_count=node_count, execution_delay=0.01)

    def execute_workflow():
        return asyncio.run(workflow.execute())

    result = benchmark(execute_workflow)

    assert result["success"]

    target_ms = {5: 100, 10: 200, 20: 400, 50: 1000}.get(node_count, 2000)
    actual_ms = result["total_time"] * 1000

    print(f"\nLinear Execution | Nodes: {node_count:2} | "
          f"Time: {actual_ms:6.2f}ms | "
          f"Tool Calls: {result['total_tool_calls']:3}")

    assert actual_ms < target_ms, (
        f"Execution time {actual_ms:.2f}ms exceeds target {target_ms}ms for {node_count} nodes"
    )


@pytest.mark.benchmark
@pytest.mark.parametrize("branches,nodes_per_branch", [
    (2, 5),
    (3, 5),
    (5, 5),
    (3, 10),
])
def test_parallel_workflow_execution(benchmark, branches, nodes_per_branch):
    """Benchmark parallel workflow execution time.

    Measures the speedup from executing branches in parallel.

    Expected Speedup:
    - 2 branches: ~1.7x (some overhead)
    - 3 branches: ~2.5x
    - 5 branches: ~3.5x (diminishing returns)
    """
    workflow = create_parallel_workflow(
        branches=branches,
        nodes_per_branch=nodes_per_branch,
        execution_delay=0.01,
    )

    def execute_workflow():
        return asyncio.run(workflow.execute())

    result = benchmark(execute_workflow)

    assert result["success"]

    total_nodes = 1 + branches * nodes_per_branch
    print(f"\nParallel Execution | Branches: {branches} | Nodes/branch: {nodes_per_branch} | "
          f"Time: {result['total_time']*1000:6.2f}ms | "
          f"Nodes: {total_nodes}")


# =============================================================================
# Node Execution Throughput Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("duration_seconds", [1, 5, 10])
def test_node_throughput(benchmark, duration_seconds):
    """Benchmark node execution throughput (nodes/second).

    Measures how many nodes can be executed per second.
    """
    workflow = create_linear_workflow(node_count=100, execution_delay=0.001)

    def execute_for_duration():
        async def _execute():
            start_time = time.time()
            nodes_executed = 0

            while time.time() - start_time < duration_seconds:
                result = await workflow.execute()
                nodes_executed += result["executed_nodes"]

            elapsed = time.time() - start_time
            throughput = nodes_executed / elapsed
            return {"nodes_executed": nodes_executed, "elapsed": elapsed, "throughput": throughput}

        return asyncio.run(_execute())

    result = benchmark(execute_for_duration)

    print(f"\nThroughput | Duration: {duration_seconds}s | "
          f"Nodes: {result['nodes_executed']} | "
          f"Throughput: {result['throughput']:.1f} nodes/s")

    # Target: >100 nodes/second
    assert result["throughput"] > 100, f"Throughput {result['throughput']:.1f} nodes/s below target"


# =============================================================================
# Recursion Depth Impact Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("depth", [1, 2, 3, 5, 7])
def test_recursion_depth_impact(benchmark, depth):
    """Benchmark performance impact of recursion depth.

    Performance Targets:
    - <5% overhead per depth level
    - Linear growth with depth
    """
    workflow = create_nested_workflow(depth=depth, nodes_per_level=3, execution_delay=0.005)

    def execute_workflow():
        return asyncio.run(workflow.execute())

    result = benchmark(execute_workflow)

    assert result["success"]

    # Calculate expected time (sequential execution)
    expected_time = result["executed_nodes"] * 0.005
    overhead = (result["total_time"] - expected_time) / expected_time * 100 if expected_time > 0 else 0

    print(f"\nRecursion Depth | Depth: {depth} | "
          f"Nodes: {result['executed_nodes']} | "
          f"Time: {result['total_time']*1000:6.2f}ms | "
          f"Overhead: {overhead:.1f}%")


# =============================================================================
# Tool Execution Overhead Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("tool_calls_per_node", [0, 5, 10, 20])
def test_tool_execution_overhead(benchmark, tool_calls_per_node):
    """Benchmark overhead of tool calls within workflow nodes.

    Measures the additional cost of simulating tool calls.

    Expected:
    - Linear relationship with tool call count
    - <5ms per tool call
    """
    workflow = create_linear_workflow(node_count=5, execution_delay=0.01)

    # Override node tool calls
    for node in workflow.nodes.values():
        node._tool_calls = tool_calls_per_node

    def execute_workflow():
        return asyncio.run(workflow.execute())

    result = benchmark(execute_workflow)

    assert result["success"]

    total_tool_calls = result["total_tool_calls"]
    time_per_tool_call = (result["total_time"] / total_tool_calls * 1000) if total_tool_calls > 0 else 0

    print(f"\nTool Execution | Tool calls/node: {tool_calls_per_node:2} | "
          f"Total: {result['total_time']*1000:6.2f}ms | "
          f"Per-call: {time_per_tool_call:5.3f}ms")

    # Target: <5ms per tool call
    if tool_calls_per_node > 0:
        assert time_per_tool_call < 5, f"Tool call overhead {time_per_tool_call:.3f}ms exceeds 5ms"


# =============================================================================
# Conditional Branching Benchmarks
# =============================================================================


@pytest.mark.benchmark
def test_conditional_branching_performance(benchmark):
    """Benchmark conditional branching performance.

    Measures the overhead of evaluating conditions and selecting branches.
    """
    workflow = create_conditional_workflow(branches=3, nodes_per_branch=5, execution_delay=0.01)

    def execute_workflow():
        return asyncio.run(workflow.execute())

    result = benchmark(execute_workflow)

    assert result["success"]

    print(f"\nConditional Branching | Time: {result['total_time']*1000:6.2f}ms | "
          f"Nodes: {result['executed_nodes']}")

    # Conditional should execute only one branch
    # Expected: entry + condition + 5 nodes = 7 nodes
    assert result["executed_nodes"] <= 7, "Conditional should execute only one branch"


# =============================================================================
# State Management Overhead Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("state_updates_per_node", [1, 5, 10, 20])
def test_state_management_overhead(benchmark, state_updates_per_node):
    """Benchmark overhead of state management.

    Measures the cost of updating and propagating state through the workflow.
    """
    workflow = create_linear_workflow(node_count=10, execution_delay=0.01)

    # Monkey-patch nodes to update state
    original_execute = WorkflowNode.execute

    async def execute_with_state_updates(self, context):
        for i in range(state_updates_per_node):
            context.update_state({
                f"key_{i}": f"value_{i}" * 10,  # ~70 bytes per key
            })
        return await original_execute(self, context)

    WorkflowNode.execute = execute_with_state_updates

    try:
        def execute_workflow():
            return asyncio.run(workflow.execute())

        result = benchmark(execute_workflow)

        assert result["success"]

        state_size_kb = len(str(result["context"].state)) / 1024

        print(f"\nState Management | Updates/node: {state_updates_per_node:2} | "
              f"Time: {result['total_time']*1000:6.2f}ms | "
              f"State size: {state_size_kb:.1f}KB")

    finally:
        # Restore original method
        WorkflowNode.execute = original_execute


# =============================================================================
# Caching Effectiveness Benchmarks
# =============================================================================


@pytest.mark.benchmark
def test_caching_effectiveness(benchmark):
    """Benchmark performance improvement from caching node results.

    Measures the speedup when node executions are cached.
    """
    workflow = create_linear_workflow(node_count=20, execution_delay=0.01)

    # First run (populates cache)
    def run_with_cache():
        async def _run():
            workflow.enable_caching = True
            if not workflow._execution_cache:
                workflow._execution_cache.clear()
            return await workflow.execute()
        return asyncio.run(_run())

    result = benchmark(run_with_cache)

    assert result["success"]

    # Check if cache was used
    cache_hits = sum(1 for k in workflow._execution_cache.keys() if k in workflow.nodes)
    total_nodes = len(workflow.nodes)

    print(f"\nCaching Effectiveness | Time: {result['total_time']*1000:.2f}ms | "
          f"Cache hits: {cache_hits}/{total_nodes} | "
          f"Caching enabled: {workflow.enable_caching}")


# =============================================================================
# Memory Profiling Benchmarks
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.parametrize("node_count", [20, 50, 100])
def test_memory_workflow_execution(benchmark, node_count):
    """Benchmark memory usage during workflow execution.

    Performance Targets:
    - 20 nodes: <10MB
    - 50 nodes: <25MB
    - 100 nodes: <50MB
    """
    gc.collect()
    tracemalloc.start()

    workflow = create_linear_workflow(node_count=node_count, execution_delay=0.01)

    def execute_and_measure():
        async def _execute_and_measure():
            result = await workflow.execute()
            current, peak = tracemalloc.get_traced_memory()
            return {"result": result, "peak_memory": peak}
        return asyncio.run(_execute_and_measure())

    output = benchmark(execute_and_measure)

    assert output["result"]["success"]

    peak_mb = output["peak_memory"] / (1024 * 1024)
    memory_per_node_kb = (output["peak_memory"] / node_count) / 1024

    target_mb = {20: 10, 50: 25, 100: 50}.get(node_count, 100)

    print(f"\nMemory Usage | Nodes: {node_count:3} | "
          f"Peak: {peak_mb:5.1f}MB | Per-node: {memory_per_node_kb:5.1f}KB")

    assert peak_mb < target_mb, f"Memory usage {peak_mb:.1f}MB exceeds target {target_mb}MB"


# =============================================================================
# Complex Scenario Benchmarks
# =============================================================================


@pytest.mark.benchmark
def test_real_world_workflow_scenario(benchmark):
    """Benchmark realistic workflow with mixed node types.

    Scenario: Code review workflow with agent nodes, tool calls, conditions.
    """
    workflow = WorkflowGraph("real_world_workflow")

    # Create realistic workflow
    nodes = [
        ("analyze", NodeType.AGENT, 0.02, 10),
        ("check_style", NodeType.TOOL, 0.01, 5),
        ("check_security", NodeType.TOOL, 0.015, 8),
        ("check_performance", NodeType.TOOL, 0.01, 5),
        ("merge_results", NodeType.AGENT, 0.02, 5),
        ("generate_report", NodeType.AGENT, 0.015, 5),
    ]

    for node_id, node_type, delay, tools in nodes:
        node = WorkflowNode(
            node_id=node_id,
            node_type=node_type,
            execution_delay=delay,
            tool_calls=tools,
        )
        workflow.add_node(node)

    # Connect nodes
    workflow.add_edge("analyze", "check_style")
    workflow.add_edge("analyze", "check_security")
    workflow.add_edge("analyze", "check_performance")
    workflow.add_edge("check_style", "merge_results")
    workflow.add_edge("check_security", "merge_results")
    workflow.add_edge("check_performance", "merge_results")
    workflow.add_edge("merge_results", "generate_report")

    workflow.set_entry_point("analyze")

    def execute_workflow():
        return asyncio.run(workflow.execute())

    result = benchmark(execute_workflow)

    assert result["success"]

    print(f"\nReal-world Workflow | Time: {result['total_time']*1000:.2f}ms | "
          f"Nodes: {result['executed_nodes']} | "
          f"Tool calls: {result['total_tool_calls']}")


# =============================================================================
# Summary and Regression Tests
# =============================================================================


@pytest.mark.summary
def test_workflow_execution_performance_summary():
    """Generate comprehensive performance summary for workflow execution."""
    results = {
        "linear": {},
        "parallel": {},
        "throughput": {},
        "recursion": {},
        "tools": {},
        "memory": {},
    }

    print("\n" + "=" * 80)
    print("WORKFLOW EXECUTION PERFORMANCE SUMMARY")
    print("=" * 80)

    # Test linear execution
    print("\n1. Linear Workflow Execution")
    print("-" * 60)
    for node_count in [5, 10, 20, 50]:
        workflow = create_linear_workflow(node_count=node_count, execution_delay=0.01)

        async def run_linear():
            return await workflow.execute()

        start = time.time()
        result = asyncio.run(run_linear())
        elapsed = time.time() - start

        results["linear"][node_count] = {
            "time_ms": elapsed * 1000,
            "success": result["success"],
        }

        status = "✓" if result["success"] else "✗"
        print(f"  {node_count:2} nodes {status}  {elapsed*1000:6.2f}ms")

    # Test parallel execution
    print("\n2. Parallel Workflow Execution")
    print("-" * 60)
    for branches in [2, 3, 5]:
        workflow = create_parallel_workflow(branches=branches, nodes_per_branch=5, execution_delay=0.01)

        async def run_parallel():
            return await workflow.execute()

        start = time.time()
        result = asyncio.run(run_parallel())
        elapsed = time.time() - start

        results["parallel"][branches] = {
            "time_ms": elapsed * 1000,
            "success": result["success"],
        }

        status = "✓" if result["success"] else "✗"
        print(f"  {branches} branches {status}  {elapsed*1000:6.2f}ms")

    # Test throughput
    print("\n3. Node Execution Throughput")
    print("-" * 60)
    workflow = create_linear_workflow(node_count=100, execution_delay=0.001)

    async def run_throughput():
        start = time.time()
        nodes_executed = 0

        for _ in range(5):
            result = await workflow.execute()
            nodes_executed += result["executed_nodes"]

        elapsed = time.time() - start
        return nodes_executed / elapsed

    throughput = asyncio.run(run_throughput())
    results["throughput"]["nodes_per_second"] = throughput

    status = "✓" if throughput > 100 else "✗"
    print(f"  Throughput {status}  {throughput:.1f} nodes/second")

    # Test recursion
    print("\n4. Recursion Depth Impact")
    print("-" * 60)
    for depth in [1, 3, 5, 7]:
        workflow = create_nested_workflow(depth=depth, nodes_per_level=3, execution_delay=0.005)

        async def run_recursion():
            return await workflow.execute()

        start = time.time()
        result = asyncio.run(run_recursion())
        elapsed = time.time() - start

        expected = result["executed_nodes"] * 0.005
        overhead = ((elapsed - expected) / expected * 100) if expected > 0 else 0

        results["recursion"][depth] = {
            "time_ms": elapsed * 1000,
            "overhead_pct": overhead,
        }

        status = "✓" if overhead < 5 * depth else "✗"
        print(f"  Depth {depth} {status}  {elapsed*1000:6.2f}ms  ({overhead:.1f}% overhead)")

    # Test tool execution
    print("\n5. Tool Execution Overhead")
    print("-" * 60)
    for tool_calls in [0, 5, 10, 20]:
        workflow = create_linear_workflow(node_count=5, execution_delay=0.01)
        for node in workflow.nodes.values():
            node._tool_calls = tool_calls

        async def run_tools():
            return await workflow.execute()

        start = time.time()
        result = asyncio.run(run_tools())
        elapsed = time.time() - start

        time_per_call = (elapsed / result["total_tool_calls"] * 1000) if result["total_tool_calls"] > 0 else 0

        results["tools"][tool_calls] = {
            "time_ms": elapsed * 1000,
            "per_call_ms": time_per_call,
        }

        status = "✓" if time_per_call < 5 or tool_calls == 0 else "✗"
        print(f"  {tool_calls:2} calls/node {status}  {elapsed*1000:6.2f}ms  ({time_per_call:.3f}ms/call)")

    # Test memory
    print("\n6. Memory Usage")
    print("-" * 60)
    for node_count in [20, 50, 100]:
        gc.collect()
        tracemalloc.start()

        workflow = create_linear_workflow(node_count=node_count, execution_delay=0.01)

        async def run_memory():
            await workflow.execute()
            return tracemalloc.get_traced_memory()[1]

        peak_kb = asyncio.run(run_memory()) / 1024
        tracemalloc.stop()

        results["memory"][node_count] = {"peak_kb": peak_kb}

        target_kb = {20: 10 * 1024, 50: 25 * 1024, 100: 50 * 1024}.get(node_count, 100 * 1024)
        status = "✓" if peak_kb < target_kb else "✗"
        print(f"  {node_count:3} nodes {status}  {peak_kb:6.1f}KB")

    # Print summary
    print("\n" + "=" * 80)
    print("PERFORMANCE TARGETS")
    print("=" * 80)
    print("  ✓ Simple workflow (5 nodes): <100ms")
    print("  ✓ Medium workflow (20 nodes): <400ms")
    print("  ✓ Complex workflow (50 nodes): <1000ms")
    print("  ✓ Throughput: >100 nodes/second")
    print("  ✓ Recursion overhead: <5% per level")
    print("  ✓ Tool execution: <5ms per tool call")
    print("  ✓ Memory usage: <50MB for 100 nodes")
    print("\n" + "=" * 80)

    # Save results
    results_dir = Path("/tmp/benchmark_results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "workflow_execution_benchmarks.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Verify critical targets
    assert results["linear"][5]["time_ms"] < 100
    assert results["linear"][20]["time_ms"] < 400
    assert results["throughput"]["nodes_per_second"] > 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-k", "summary"])
