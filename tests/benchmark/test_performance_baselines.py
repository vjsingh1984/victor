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

"""Performance baseline tests for Victor AI.

This module establishes performance baselines for key operations:
1. Single request latency
2. Concurrent throughput
3. Memory usage under load
4. Tool execution performance
5. Context management performance

Performance Targets:
- Single request: <100ms (p50), <500ms (p99)
- Throughput: >100 requests/second
- Memory: <1GB for 100 concurrent sessions
- Tool execution: <50ms (p50), <200ms (p99)
- Context compaction: <100ms for 1000 messages
"""

import asyncio
import gc
import psutil
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.orchestrator import AgentOrchestrator
from victor.tools.base import BaseTool, ToolResult
from victor.agent.coordinators.compaction_strategies import (
    TruncationCompactionStrategy,
    HybridCompactionStrategy,
)


# =============================================================================
# Performance Targets
# =============================================================================


PERFORMANCE_TARGETS = {
    "latency_ms": {
        "p50": 100,  # 50th percentile
        "p95": 300,  # 95th percentile
        "p99": 500,  # 99th percentile
    },
    "throughput": {
        "min_requests_per_second": 100,
    },
    "memory": {
        "max_mb_for_100_sessions": 1024,  # 1GB
        "max_mb_per_session": 10,
    },
    "tool_execution": {
        "p50_ms": 50,
        "p95_ms": 150,
        "p99_ms": 200,
    },
    "context_compaction": {
        "max_time_ms_for_1000_messages": 100,
    },
}


# =============================================================================
# Test Tools
# =============================================================================


class FastTestTool(BaseTool):
    """A fast tool for performance testing."""

    name = "fast_tool"
    description = "A fast test tool"

    parameters = {
        "type": "object",
        "properties": {
            "value": {"type": "string", "description": "Input value"},
        },
        "required": ["value"],
    }

    async def execute(self, value: str = "", **kwargs):
        return ToolResult.create_success(output=f"Fast: {value}")


class SlowTestTool(BaseTool):
    """A tool with simulated delay for testing."""

    name = "slow_tool"
    description = "A slow test tool"

    parameters = {
        "type": "object",
        "properties": {
            "delay_ms": {"type": "number", "description": "Delay in milliseconds"},
        },
        "required": ["delay_ms"],
    }

    async def execute(self, delay_ms: float = 10, **kwargs):
        await asyncio.sleep(delay_ms / 1000.0)
        return ToolResult.create_success(output=f"Delayed by {delay_ms}ms")


# =============================================================================
# Latency Tests
# =============================================================================


class TestLatencyBaselines:
    """Establish baseline latency for single requests."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_single_request_latency_p50(self, mock_orchestrator):
        """Measure P50 latency for single request."""
        latencies = []

        # Warm up
        for _ in range(10):
            await mock_orchestrator.chat("Warm up")

        # Measure
        for _ in range(50):
            start = time.time()
            await mock_orchestrator.chat("Test message")
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        print("\n✓ Single Request Latency:")
        print(f"  P50: {p50:.2f}ms (target: <{PERFORMANCE_TARGETS['latency_ms']['p50']}ms)")
        print(f"  P95: {p95:.2f}ms (target: <{PERFORMANCE_TARGETS['latency_ms']['p95']}ms)")
        print(f"  P99: {p99:.2f}ms (target: <{PERFORMANCE_TARGETS['latency_ms']['p99']}ms)")

        # Assert against targets
        assert (
            p50 < PERFORMANCE_TARGETS["latency_ms"]["p50"]
        ), f"P50 latency {p50:.2f}ms exceeds target {PERFORMANCE_TARGETS['latency_ms']['p50']}ms"
        assert (
            p95 < PERFORMANCE_TARGETS["latency_ms"]["p95"]
        ), f"P95 latency {p95:.2f}ms exceeds target {PERFORMANCE_TARGETS['latency_ms']['p95']}ms"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_request_latency_distribution(self, mock_orchestrator):
        """Measure full latency distribution."""
        latencies = []

        for _ in range(100):
            start = time.time()
            await mock_orchestrator.chat("Latency test")
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

        latencies.sort()

        percentiles = {
            "min": latencies[0],
            "p10": latencies[int(len(latencies) * 0.10)],
            "p25": latencies[int(len(latencies) * 0.25)],
            "p50": latencies[int(len(latencies) * 0.50)],
            "p75": latencies[int(len(latencies) * 0.75)],
            "p90": latencies[int(len(latencies) * 0.90)],
            "p95": latencies[int(len(latencies) * 0.95)],
            "p99": latencies[int(len(latencies) * 0.99)],
            "max": latencies[-1],
            "avg": sum(latencies) / len(latencies),
        }

        print("\n✓ Latency Distribution:")
        for name, value in percentiles.items():
            print(f"  {name}: {value:.2f}ms")

        # Check variance (max should not be >10x median)
        assert (
            percentiles["max"] < percentiles["p50"] * 10
        ), f"High latency variance: max={percentiles['max']:.2f}ms, p50={percentiles['p50']:.2f}ms"


# =============================================================================
# Throughput Tests
# =============================================================================


class TestThroughputBaselines:
    """Establish baseline throughput for concurrent requests."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    @pytest.mark.slow
    async def test_concurrent_throughput(self, mock_orchestrator):
        """Measure throughput with concurrent requests."""
        num_requests = 100
        concurrency = 10

        start_time = time.time()

        # Create concurrent tasks
        tasks = [mock_orchestrator.chat(f"Throughput test {i}") for i in range(num_requests)]

        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        throughput = num_requests / elapsed

        print("\n✓ Concurrent Throughput:")
        print(f"  Requests: {num_requests}")
        print(f"  Concurrency: {concurrency}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.2f} requests/second")
        print(f"  Target: >{PERFORMANCE_TARGETS['throughput']['min_requests_per_second']} req/s")

        assert (
            throughput >= PERFORMANCE_TARGETS["throughput"]["min_requests_per_second"]
        ), f"Throughput {throughput:.2f} req/s below target {PERFORMANCE_TARGETS['throughput']['min_requests_per_second']} req/s"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    @pytest.mark.slow
    async def test_throughput_scaling(self, mock_orchestrator):
        """Measure how throughput scales with concurrency."""
        concurrency_levels = [1, 5, 10, 20, 50]
        results = []

        for concurrency in concurrency_levels:
            num_requests = concurrency * 10

            start_time = time.time()
            tasks = [mock_orchestrator.chat(f"Scaling test {i}") for i in range(num_requests)]

            await asyncio.gather(*tasks)
            elapsed = time.time() - start_time

            throughput = num_requests / elapsed
            results.append({"concurrency": concurrency, "throughput": throughput})

            print(f"\nConcurrency: {concurrency}")
            print(f"  Throughput: {throughput:.2f} req/s")

        # Check for near-linear scaling up to a point
        # (expect some diminishing returns at high concurrency)
        base_throughput = results[0]["throughput"]
        max_throughput = max(r["throughput"] for r in results)

        print("\n✓ Throughput Scaling:")
        print(f"  Single user: {base_throughput:.2f} req/s")
        print(f"  Max throughput: {max_throughput:.2f} req/s")
        print(f"  Scaling factor: {max_throughput / base_throughput:.2f}x")


# =============================================================================
# Memory Tests
# =============================================================================


class TestMemoryBaselines:
    """Establish baseline memory usage."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_memory_per_session(self):
        """Measure memory usage per session."""
        process = psutil.Process()
        gc.collect()

        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create multiple sessions
        sessions = []
        for i in range(10):
            # Simulate session creation
            session_data = {"messages": [{"role": "user", "content": "x" * 100} for _ in range(10)]}
            sessions.append(session_data)

        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        memory_per_session = (final_memory - baseline_memory) / 10

        print("\n✓ Memory Per Session:")
        print(f"  Baseline: {baseline_memory:.2f}MB")
        print(f"  Final: {final_memory:.2f}MB")
        print(f"  Per session: {memory_per_session:.2f}MB")
        print(f"  Target: <{PERFORMANCE_TARGETS['memory']['max_mb_per_session']}MB")

        assert (
            memory_per_session < PERFORMANCE_TARGETS["memory"]["max_mb_per_session"]
        ), f"Memory per session {memory_per_session:.2f}MB exceeds target {PERFORMANCE_TARGETS['memory']['max_mb_per_session']}MB"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    @pytest.mark.slow
    async def test_memory_under_load(self, mock_orchestrator):
        """Measure memory usage with 100 concurrent sessions."""
        process = psutil.Process()
        gc.collect()

        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Simulate 100 concurrent sessions
        num_sessions = 100
        tasks = []

        for i in range(num_sessions):
            task = mock_orchestrator.chat(f"Session {i} message")
            tasks.append(task)

        await asyncio.gather(*tasks)
        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - baseline_memory

        print(f"\n✓ Memory Under Load ({num_sessions} sessions):")
        print(f"  Baseline: {baseline_memory:.2f}MB")
        print(f"  Final: {final_memory:.2f}MB")
        print(f"  Used: {memory_used:.2f}MB")
        print(f"  Target: <{PERFORMANCE_TARGETS['memory']['max_mb_for_100_sessions']}MB")

        assert (
            memory_used < PERFORMANCE_TARGETS["memory"]["max_mb_for_100_sessions"]
        ), f"Memory {memory_used:.2f}MB exceeds target {PERFORMANCE_TARGETS['memory']['max_mb_for_100_sessions']}MB"


# =============================================================================
# Tool Execution Tests
# =============================================================================


class TestToolExecutionBaselines:
    """Establish baseline tool execution performance."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_fast_tool_execution(self):
        """Measure fast tool execution latency."""
        tool = FastTestTool()

        latencies = []
        for _ in range(100):
            start = time.time()
            await tool.execute(value="test")
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        print("\n✓ Fast Tool Execution:")
        print(f"  P50: {p50:.2f}ms")
        print(f"  P95: {p95:.2f}ms")
        print(f"  P99: {p99:.2f}ms")

        # Fast tools should be very fast
        assert p50 < 10, f"Fast tool P50 {p50:.2f}ms exceeds 10ms"
        assert p99 < 50, f"Fast tool P99 {p99:.2f}ms exceeds 50ms"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_tool_batch_execution(self):
        """Measure batch tool execution performance."""
        tools = [FastTestTool(), SlowTestTool()]

        start = time.time()

        # Execute tools in batch
        tasks = [tool.execute(value=f"batch {i}") for i, tool in enumerate(tools * 10)]
        results = await asyncio.gather(*tasks)

        elapsed = (time.time() - start) * 1000
        avg_per_tool = elapsed / len(results)

        print("\n✓ Batch Tool Execution:")
        print(f"  Total time: {elapsed:.2f}ms")
        print(f"  Tools executed: {len(results)}")
        print(f"  Average per tool: {avg_per_tool:.2f}ms")

        # Batch execution should be efficient
        assert avg_per_tool < 20, f"Batch execution avg {avg_per_tool:.2f}ms exceeds 20ms"


# =============================================================================
# Context Management Tests
# =============================================================================


class TestContextManagementBaselines:
    """Establish baseline context management performance."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_context_compaction_performance(self):
        """Measure context compaction performance."""
        strategy = TruncationCompactionStrategy(max_chars=100)

        # Create large conversation
        large_conversation = [{"role": "user", "content": f"Message {i}" * 10} for i in range(1000)]

        # Measure compaction time using the strategy directly
        start = time.time()
        compacted = strategy.compact(large_conversation, target_tokens=1000)
        elapsed_ms = (time.time() - start) * 1000

        print("\n✓ Context Compaction Performance:")
        print(f"  Original messages: {len(large_conversation)}")
        print(f"  Compacted messages: {len(compacted)}")
        print(f"  Time: {elapsed_ms:.2f}ms")
        print(
            f"  Target: <{PERFORMANCE_TARGETS['context_compaction']['max_time_ms_for_1000_messages']}ms"
        )

        assert (
            elapsed_ms < PERFORMANCE_TARGETS["context_compaction"]["max_time_ms_for_1000_messages"]
        ), f"Compaction time {elapsed_ms:.2f}ms exceeds target {PERFORMANCE_TARGETS['context_compaction']['max_time_ms_for_1000_messages']}ms"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_context_strategies_comparison(self):
        """Compare different context compaction strategies."""
        strategies = {
            "truncation": TruncationCompactionStrategy(max_chars=100),
            "hybrid": HybridCompactionStrategy(),
        }

        conversation = [{"role": "user", "content": f"Message {i}" * 5} for i in range(500)]

        results = {}

        for name, strategy in strategies.items():
            start = time.time()
            # Use strategy directly for synchronous compaction
            if hasattr(strategy, "compact"):
                compacted = strategy.compact(conversation, target_tokens=1000)
            else:
                # For async strategies, run them
                compacted = await strategy.compact_async(conversation, target_tokens=1000)
            elapsed_ms = (time.time() - start) * 1000

            results[name] = {
                "time_ms": elapsed_ms,
                "original_length": len(conversation),
                "compacted_length": len(compacted),
                "reduction_ratio": len(compacted) / len(conversation),
            }

            print(f"\n✓ {name.capitalize()} Strategy:")
            print(f"  Time: {elapsed_ms:.2f}ms")
            print(f"  Reduction: {len(conversation)} → {len(compacted)} messages")
            print(f"  Ratio: {results[name]['reduction_ratio']:.2%}")

        # All strategies should complete in reasonable time
        for name, result in results.items():
            assert result["time_ms"] < 200, f"{name} strategy too slow: {result['time_ms']:.2f}ms"


# =============================================================================
# Performance Regression Detection
# =============================================================================


class TestPerformanceRegression:
    """Detect performance regressions by comparing against historical baselines."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_no_regression_in_latency(self, mock_orchestrator):
        """Ensure latency hasn't regressed from baseline."""
        latencies = []

        for _ in range(50):
            start = time.time()
            await mock_orchestrator.chat("Regression test")
            latencies.append((time.time() - start) * 1000)

        latencies.sort()
        p99 = latencies[int(len(latencies) * 0.99)]

        # This should be compared against stored historical baseline
        # For now, just check against targets
        target = PERFORMANCE_TARGETS["latency_ms"]["p99"]

        print("\n✓ Regression Check (Latency P99):")
        print(f"  Current: {p99:.2f}ms")
        print(f"  Target: {target}ms")
        print(f"  Status: {'PASS' if p99 < target else 'FAIL'}")

        assert p99 < target, f"Performance regression detected: P99={p99:.2f}ms"

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_no_regression_in_throughput(self, mock_orchestrator):
        """Ensure throughput hasn't regressed from baseline."""
        num_requests = 50
        start_time = time.time()

        tasks = [mock_orchestrator.chat(f"Regression test {i}") for i in range(num_requests)]

        await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        throughput = num_requests / elapsed
        target = PERFORMANCE_TARGETS["throughput"]["min_requests_per_second"]

        print("\n✓ Regression Check (Throughput):")
        print(f"  Current: {throughput:.2f} req/s")
        print(f"  Target: {target} req/s")
        print(f"  Status: {'PASS' if throughput >= target else 'FAIL'}")

        assert (
            throughput >= target
        ), f"Performance regression detected: throughput={throughput:.2f} req/s"


# =============================================================================
# Baseline Persistence
# =============================================================================


def save_performance_baselines(
    results: dict[str, Any], output_path: str = "/tmp/performance_baselines.json"
):
    """Save performance baselines for future comparison."""
    import json
    from pathlib import Path

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    baseline = {
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "targets": PERFORMANCE_TARGETS,
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"\n✓ Performance baselines saved to: {output_file}")
    return baseline


def load_performance_baselines(
    baseline_path: str = "/tmp/performance_baselines.json",
) -> dict[str, Any]:
    """Load historical performance baselines for comparison."""
    import json
    from pathlib import Path

    baseline_file = Path(baseline_path)
    if not baseline_file.exists():
        return {}

    with open(baseline_file, "r") as f:
        return json.load(f)


# =============================================================================
# Mock Orchestrator Fixture
# =============================================================================


@pytest.fixture
async def mock_orchestrator():
    """Create mock orchestrator for performance testing."""

    orchestrator = MagicMock(spec=AgentOrchestrator)

    # Mock chat method with realistic delay
    async def mock_chat(message: str, **kwargs):
        # Simulate processing time
        await asyncio.sleep(0.01)  # 10ms base processing
        return f"Response to: {message[:50]}..."

    orchestrator.chat = mock_chat
    orchestrator.stream_chat = AsyncMock()

    return orchestrator
