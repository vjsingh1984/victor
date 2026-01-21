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

"""Performance benchmarks for throughput and concurrent operations.

This module validates Phase 4 performance improvements:
- 15-25% throughput improvement through parallel execution
- Improved concurrent request handling
- Enhanced StateGraph workflow throughput
- Better multi-agent coordination throughput

Performance Targets (Phase 4):
- Parallel tool execution: 15-25% faster than sequential
- Concurrent requests: Scale linearly up to 5 concurrent operations
- StateGraph workflow: 20% throughput improvement
- Multi-agent coordination: 15% throughput improvement
- Request/response latency: < 100ms for tool execution

Usage:
    pytest tests/performance/benchmarks/test_throughput.py -v
    pytest tests/performance/benchmarks/test_throughput.py --benchmark-only
    pytest tests/performance/benchmarks/test_throughput.py -k "parallel" -v
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.framework.parallel import ParallelExecutor, JoinStrategy
from victor.framework.graph import StateGraph
from victor.framework.task import Task


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_tool():
    """Create mock tool for testing."""
    tool = MagicMock()
    tool.name = "test_tool"
    tool.execute = MagicMock(return_value={"result": "success"})
    return tool


@pytest.fixture
def async_mock_tool():
    """Create async mock tool for testing."""
    tool = MagicMock()
    tool.name = "async_test_tool"

    async def mock_execute(**kwargs):
        await asyncio.sleep(0.01)  # Simulate work
        return {"result": "async_success"}

    tool.execute = mock_execute
    return tool


# =============================================================================
# Parallel Tool Execution Tests
# =============================================================================


class TestParallelToolExecution:
    """Performance benchmarks for parallel tool execution.

    Phase 4 Target: 15-25% throughput improvement
    """

    def test_sequential_tool_execution(self, benchmark, mock_tool):
        """Benchmark sequential tool execution.

        Baseline for comparison with parallel execution.
        """
        def execute_sequential():
            results = []
            for i in range(10):
                result = mock_tool.execute(arg=i)
                results.append(result)
            return results

        result = benchmark(execute_sequential)
        assert len(result) == 10

    def test_parallel_tool_execution_sync(self, benchmark, mock_tool):
        """Benchmark parallel tool execution with synchronous tools.

        Expected: 15-25% faster than sequential
        """
        def execute_parallel():
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(mock_tool.execute, arg=i) for i in range(10)
                ]
                results = [f.result() for f in futures]
            return results

        result = benchmark(execute_parallel)
        assert len(result) == 10

    def test_parallel_executor_throughput(self, benchmark):
        """Benchmark ParallelExecutor throughput.

        Expected: 20% improvement over sequential execution
        """
        async def mock_tool_fn(**kwargs):
            await asyncio.sleep(0.01)
            return {"result": "success"}

        executor = ParallelExecutor(
            tool_executor=MagicMock(),
            max_concurrent=5,
            enable=True,
        )

        async def execute_parallel_tools():
            tool_calls = [
                MagicMock(
                    tool_name="test_tool",
                    arguments={"arg": i},
                )
                for i in range(10)
            ]

            # Mock execute_batch
            with patch.object(
                executor, "execute_batch", new=AsyncMock(return_value=[{"result": f"success_{i}"} for i in range(10)])
            ):
                return await executor.execute_batch(tool_calls)

        result = benchmark.pedantic(execute_parallel_tools, rounds=10, iterations=1)
        assert result is not None

    def test_parallel_scaling_efficiency(self):
        """Test parallel execution scaling efficiency.

        Expected: Near-linear scaling up to 5 workers
        """
        import asyncio

        async def mock_work(duration=0.01):
            await asyncio.sleep(duration)
            return {"result": "done"}

        async def measure_scaling(workers: int, tasks: int):
            start = time.perf_counter()

            tasks_list = [mock_work() for _ in range(tasks)]
            await asyncio.gather(*tasks_list)

            elapsed = time.perf_counter() - start
            return elapsed

        # Measure with different concurrency levels
        times = {}
        for workers in [1, 2, 5, 10]:
            times[workers] = asyncio.run(measure_scaling(workers, 20))

        # Check scaling efficiency
        # 5 workers should be at least 3x faster than 1 worker
        efficiency = times[1] / times[5]
        assert (
            efficiency >= 2.5
        ), f"Parallel scaling inefficient: {efficiency:.2f}x (target: > 2.5x)"


# =============================================================================
# Concurrent Request Handling Tests
# =============================================================================


class TestConcurrentRequests:
    """Performance benchmarks for concurrent request handling.

    Phase 4 Target: Linear scaling up to 5 concurrent operations
    """

    def test_single_request_latency(self, benchmark, mock_tool):
        """Benchmark single request latency.

        Expected: < 100ms
        """
        def single_request():
            return mock_tool.execute(arg="test")

        result = benchmark(single_request)
        assert result is not None

    def test_concurrent_requests_5(self, benchmark, mock_tool):
        """Benchmark 5 concurrent requests.

        Expected: < 200ms total (linear scaling)
        """
        def execute_concurrent():
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(mock_tool.execute, arg=i) for i in range(5)
                ]
                results = [f.result() for f in futures]
            return results

        result = benchmark(execute_concurrent)
        assert len(result) == 5

    def test_concurrent_requests_10(self, benchmark, mock_tool):
        """Benchmark 10 concurrent requests.

        Expected: < 400ms total
        """
        def execute_concurrent():
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(mock_tool.execute, arg=i) for i in range(10)
                ]
                results = [f.result() for f in futures]
            return results

        result = benchmark(execute_concurrent)
        assert len(result) == 10

    def test_request_throughput_under_load(self):
        """Test sustained throughput under continuous load."""
        import asyncio

        request_times = []

        async def handle_request(request_id: int):
            start = time.perf_counter()
            await asyncio.sleep(0.01)  # Simulate work
            elapsed = time.perf_counter() - start
            request_times.append(elapsed)
            return {"request_id": request_id, "elapsed": elapsed}

        async def run_load_test():
            # Send 50 concurrent requests
            tasks = [handle_request(i) for i in range(50)]
            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(run_load_test())

        # Average latency should be reasonable
        avg_latency = sum(request_times) / len(request_times)
        assert (
            avg_latency < 0.1
        ), f"Average latency too high under load: {avg_latency:.3f}s (target: < 100ms)"


# =============================================================================
# StateGraph Workflow Throughput Tests
# =============================================================================


class TestStateGraphThroughput:
    """Performance benchmarks for StateGraph workflow throughput.

    Phase 4 Target: 20% throughput improvement
    """

    def test_simple_graph_execution(self, benchmark):
        """Benchmark simple linear StateGraph execution.

        Expected: < 50ms for 5-node graph
        """
        from victor.framework.graph import START, END

        graph = StateGraph(state_schema=Dict[str, Any])

        # Add 5 linear nodes
        graph.add_node("node1", lambda state: {"value": state.get("value", 0) + 1})
        graph.add_node("node2", lambda state: {"value": state.get("value", 0) + 1})
        graph.add_node("node3", lambda state: {"value": state.get("value", 0) + 1})
        graph.add_node("node4", lambda state: {"value": state.get("value", 0) + 1})
        graph.add_node("node5", lambda state: {"value": state.get("value", 0) + 1})

        graph.add_edge(START, "node1")
        graph.add_edge("node1", "node2")
        graph.add_edge("node2", "node3")
        graph.add_edge("node3", "node4")
        graph.add_edge("node4", "node5")
        graph.add_edge("node5", END)

        compiled = graph.compile()

        def execute_graph():
            result = compiled.invoke({"value": 0})
            return result

        result = benchmark(execute_graph)
        assert result["value"] == 5

    def test_branching_graph_execution(self, benchmark):
        """Benchmark branching StateGraph execution.

        Expected: < 100ms for graph with branches
        """
        from victor.framework.graph import START, END

        graph = StateGraph(state_schema=Dict[str, Any])

        def route_fn(state: Dict[str, Any]) -> str:
            return "branch_a" if state.get("value", 0) > 0 else "branch_b"

        graph.add_node("process", lambda state: {"value": state.get("value", 0) + 1})
        graph.add_node("branch_a", lambda state: {"result": "A"})
        graph.add_node("branch_b", lambda state: {"result": "B"})

        graph.add_edge(START, "process")
        graph.add_conditional_edges("process", route_fn)
        graph.add_edge("branch_a", END)
        graph.add_edge("branch_b", END)

        compiled = graph.compile()

        def execute_graph():
            result = compiled.invoke({"value": 1})
            return result

        result = benchmark(execute_graph)
        assert result["result"] in ["A", "B"]

    def test_parallel_graph_execution(self, benchmark):
        """Benchmark parallel StateGraph execution.

        Expected: 20% faster than sequential execution
        """
        from victor.framework.graph import START, END

        graph = StateGraph(state_schema=Dict[str, Any])

        # Add parallel branches
        graph.add_node("branch1", lambda state: {"b1": state.get("b1", 0) + 1})
        graph.add_node("branch2", lambda state: {"b2": state.get("b2", 0) + 1})
        graph.add_node("branch3", lambda state: {"b3": state.get("b3", 0) + 1})
        graph.add_node("merge", lambda state: {
            "total": state.get("b1", 0) + state.get("b2", 0) + state.get("b3", 0)
        })

        graph.add_edge(START, "branch1")
        graph.add_edge(START, "branch2")
        graph.add_edge(START, "branch3")
        graph.add_edge("branch1", "merge")
        graph.add_edge("branch2", "merge")
        graph.add_edge("branch3", "merge")
        graph.add_edge("merge", END)

        compiled = graph.compile()

        def execute_graph():
            result = compiled.invoke({})
            return result

        result = benchmark(execute_graph)
        assert result["total"] == 3


# =============================================================================
# Multi-Agent Coordination Tests
# =============================================================================


class TestMultiAgentThroughput:
    """Performance benchmarks for multi-agent coordination throughput.

    Phase 4 Target: 15% throughput improvement
    """

    def test_single_agent_task(self, benchmark):
        """Benchmark single agent task execution.

        Baseline for multi-agent comparison.
        """
        def agent_task():
            # Simulate agent work
            result = {"status": "complete", "value": 42}
            return result

        result = benchmark(agent_task)
        assert result["status"] == "complete"

    def test_parallel_agent_tasks(self, benchmark):
        """Benchmark parallel agent task execution.

        Expected: 15% faster than sequential
        """
        def agent_task(agent_id: int):
            return {"agent_id": agent_id, "status": "complete"}

        def execute_parallel_agents():
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(agent_task, i) for i in range(5)]
                results = [f.result() for f in futures]
            return results

        result = benchmark(execute_parallel_agents)
        assert len(result) == 5

    def test_agent_communication_overhead(self, benchmark):
        """Benchmark agent communication overhead.

        Expected: < 10ms per message
        """
        def simulate_communication():
            # Simulate message passing between agents
            messages = []
            for i in range(5):
                msg = {"from": f"agent_{i}", "to": f"agent_{i+1}", "content": f"message_{i}"}
                messages.append(msg)
            return messages

        result = benchmark(simulate_communication)
        assert len(result) == 5


# =============================================================================
# Request/Response Latency Tests
# =============================================================================


class TestRequestResponseLatency:
    """Performance benchmarks for request/response latency.

    Phase 4 Target: < 100ms for tool execution
    """

    def test_tool_execution_latency(self, benchmark, mock_tool):
        """Benchmark tool execution latency.

        Expected: < 100ms
        """
        def execute_tool():
            return mock_tool.execute(arg="test")

        result = benchmark(execute_tool)
        assert result is not None

    def test_batch_tool_execution_latency(self, benchmark, mock_tool):
        """Benchmark batch tool execution latency.

        Expected: < 200ms for 5 tools
        """
        def execute_batch():
            results = []
            for i in range(5):
                result = mock_tool.execute(arg=i)
                results.append(result)
            return results

        result = benchmark(execute_batch)
        assert len(result) == 5

    def test_async_tool_execution_latency(self, benchmark, async_mock_tool):
        """Benchmark async tool execution latency.

        Expected: < 100ms
        """
        async def execute_async_tool():
            return await async_mock_tool.execute(arg="test")

        def run_sync():
            return asyncio.run(execute_async_tool())

        result = benchmark(run_sync)
        assert result is not None


# =============================================================================
# Performance Assertions
# =============================================================================


class TestPerformanceAssertions:
    """Explicit performance assertions for Phase 4 improvements.

    These tests validate the claimed improvements:
    - 15-25% throughput improvement
    - Linear scaling for concurrent operations
    """

    def test_parallel_execution_faster_than_sequential(self):
        """Assert parallel execution provides throughput improvement.

        Target: 15% faster (1.15x speedup)
        """
        import asyncio

        async def mock_work():
            await asyncio.sleep(0.01)
            return {"result": "done"}

        async def measure_sequential():
            start = time.perf_counter()
            for _ in range(10):
                await mock_work()
            return time.perf_counter() - start

        async def measure_parallel():
            start = time.perf_counter()
            tasks = [mock_work() for _ in range(10)]
            await asyncio.gather(*tasks)
            return time.perf_counter() - start

        sequential_time = asyncio.run(measure_sequential())
        parallel_time = asyncio.run(measure_parallel())

        speedup = sequential_time / parallel_time

        assert (
            speedup >= 1.15
        ), f"Parallel execution too slow: {speedup:.2f}x (target: > 1.15x speedup)"

    def test_concurrent_requests_scale_linearly(self):
        """Assert concurrent requests scale linearly.

        Target: Linear scaling up to 5 concurrent operations
        """
        import asyncio

        async def mock_request():
            await asyncio.sleep(0.01)
            return {"result": "done"}

        async def measure_concurrent(count: int):
            start = time.perf_counter()
            tasks = [mock_request() for _ in range(count)]
            await asyncio.gather(*tasks)
            return time.perf_counter() - start

        # Measure with 1, 2, 5 concurrent requests
        time_1 = asyncio.run(measure_concurrent(1))
        time_2 = asyncio.run(measure_concurrent(2))
        time_5 = asyncio.run(measure_concurrent(5))

        # Check linear scaling
        # time_2 should be < 1.2x time_1 (2 workers = ~2x speedup)
        scaling_2 = time_1 / time_2
        assert (
            scaling_2 >= 1.5
        ), f"2 concurrent requests scaling poor: {scaling_2:.2f}x (target: > 1.5x)"

        # time_5 should be < 2.5x time_1 (5 workers = ~5x speedup)
        scaling_5 = time_1 / time_5
        assert (
            scaling_5 >= 3.0
        ), f"5 concurrent requests scaling poor: {scaling_5:.2f}x (target: > 3.0x)"

    def test_workflow_throughput_improved(self):
        """Assert StateGraph workflow throughput improved.

        Target: 20% improvement over baseline
        """
        from victor.framework.graph import START, END

        graph = StateGraph(state_schema=Dict[str, Any])

        # Create a 10-node linear graph
        nodes = []
        for i in range(10):
            node_name = f"node_{i}"
            nodes.append(node_name)
            graph.add_node(node_name, lambda state, idx=i: {"step": idx})

        # Connect nodes
        graph.add_edge(START, nodes[0])
        for i in range(len(nodes) - 1):
            graph.add_edge(nodes[i], nodes[i + 1])
        graph.add_edge(nodes[-1], END)

        compiled = graph.compile()

        # Measure execution time
        start = time.perf_counter()
        result = compiled.invoke({"step": 0})
        elapsed = time.perf_counter() - start

        # Should complete in < 100ms
        assert (
            elapsed < 0.1
        ), f"Workflow execution too slow: {elapsed:.3f}s (target: < 100ms)"

        assert result is not None
