#!/usr/bin/env python3
"""Benchmark for tool pipeline caching performance.

Measures the performance improvement from hot path optimization cache.
Target: 10-20% improvement in tool execution throughput.
"""
import asyncio
import sys
import time
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, "/Users/vijaysingh/code/codingagent")

from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig
from victor.agent.argument_normalizer import ArgumentNormalizer
from victor.agent.tool_executor import ToolExecutor
from victor.tools.base import ToolRegistry


class BenchmarkToolExecutor(ToolExecutor):
    """Mock tool executor for benchmarking."""

    def __init__(self, latency_ms: float = 1.0):
        self.call_count = 0
        self.latency_ms = latency_ms

    async def execute(self, tool_name: str, arguments: dict, context: dict = None):
        # Simulate minimal tool execution latency
        await asyncio.sleep(self.latency_ms / 1000.0)
        self.call_count += 1
        # Return a proper result object with all required attributes
        return type(
            "Result", (), {"success": True, "result": f"Executed {tool_name}", "error": None}
        )


async def benchmark_with_cache(
    tool_calls: List[Dict[str, Any]],
    iterations: int = 10,
) -> float:
    """Benchmark execution with cache enabled.

    Args:
        tool_calls: List of tool calls to execute
        iterations: Number of iterations to run

    Returns:
        Total execution time in seconds
    """
    registry = ToolRegistry()
    executor = BenchmarkToolExecutor(latency_ms=1.0)
    normalizer = ArgumentNormalizer()

    config = ToolPipelineConfig(
        tool_budget=1000,
        enable_caching=True,
        enable_analytics=False,
    )

    pipeline = ToolPipeline(
        tool_registry=registry,
        tool_executor=executor,
        config=config,
        argument_normalizer=normalizer,
    )

    # Mock registry to allow all tools
    registry.is_tool_enabled = lambda x: True
    registry.has_tool = lambda x: True

    start = time.time()
    for _ in range(iterations):
        for call in tool_calls:
            await pipeline._execute_single_call(call, {})
    duration = time.time() - start

    return duration


async def benchmark_without_cache(
    tool_calls: List[Dict[str, Any]],
    iterations: int = 10,
) -> float:
    """Benchmark execution with cache disabled.

    Args:
        tool_calls: List of tool calls to execute
        iterations: Number of iterations to run

    Returns:
        Total execution time in seconds
    """
    registry = ToolRegistry()
    executor = BenchmarkToolExecutor(latency_ms=1.0)
    normalizer = ArgumentNormalizer()

    config = ToolPipelineConfig(
        tool_budget=1000,
        enable_caching=True,  # Still enable result caching, just not decision caching
        enable_analytics=False,
    )

    pipeline = ToolPipeline(
        tool_registry=registry,
        tool_executor=executor,
        config=config,
        argument_normalizer=normalizer,
    )

    # Mock registry to allow all tools
    registry.is_tool_enabled = lambda x: True
    registry.has_tool = lambda x: True

    start = time.time()
    for _ in range(iterations):
        for call in tool_calls:
            await pipeline._execute_single_call(call, {})
            # Clear decision cache after each call to simulate no caching
            pipeline._decision_cache.clear()
    duration = time.time() - start

    return duration


async def run_benchmark():
    """Run the complete benchmark suite."""
    print("=" * 70)
    print("Tool Pipeline Cache Performance Benchmark")
    print("=" * 70)
    print()

    # Benchmark 1: Repeated tool calls with same arguments
    print("Benchmark 1: Repeated tool calls (high cache hit rate)")
    print("-" * 70)

    tool_calls_repeated = [
        {"name": "read", "arguments": {"path": "/test/file1.txt"}},
        {"name": "read", "arguments": {"path": "/test/file2.txt"}},
        {"name": "read", "arguments": {"path": "/test/file3.txt"}},
    ] * 10  # 30 calls, many duplicates

    cached_time = await benchmark_with_cache(tool_calls_repeated, iterations=10)
    uncached_time = await benchmark_without_cache(tool_calls_repeated, iterations=10)

    speedup = uncached_time / cached_time
    improvement = ((uncached_time - cached_time) / uncached_time) * 100

    print(f"Without cache: {uncached_time:.4f}s")
    print(f"With cache:    {cached_time:.4f}s")
    print(f"Speedup:       {speedup:.2f}x")
    print(f"Improvement:   {improvement:.1f}%")
    print()

    # Benchmark 2: Unique tool calls (low cache hit rate)
    print("Benchmark 2: Unique tool calls (low cache hit rate)")
    print("-" * 70)

    tool_calls_unique = [
        {"name": f"tool{i}", "arguments": {"path": f"/test/file{i}.txt"}} for i in range(100)
    ]

    cached_time = await benchmark_with_cache(tool_calls_unique, iterations=1)
    uncached_time = await benchmark_without_cache(tool_calls_unique, iterations=1)

    speedup = uncached_time / cached_time
    improvement = ((uncached_time - cached_time) / uncached_time) * 100

    print(f"Without cache: {uncached_time:.4f}s")
    print(f"With cache:    {cached_time:.4f}s")
    print(f"Speedup:       {speedup:.2f}x")
    print(f"Improvement:   {improvement:.1f}%")
    print()

    # Benchmark 3: Mixed workload
    print("Benchmark 3: Mixed workload (50% duplicates)")
    print("-" * 70)

    tool_calls_mixed = [
        {"name": "read", "arguments": {"path": f"/test/file{i % 50}.txt"}} for i in range(100)
    ]

    cached_time = await benchmark_with_cache(tool_calls_mixed, iterations=5)
    uncached_time = await benchmark_without_cache(tool_calls_mixed, iterations=5)

    speedup = uncached_time / cached_time
    improvement = ((uncached_time - cached_time) / uncached_time) * 100

    print(f"Without cache: {uncached_time:.4f}s")
    print(f"With cache:    {cached_time:.4f}s")
    print(f"Speedup:       {speedup:.2f}x")
    print(f"Improvement:   {improvement:.1f}%")
    print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("Target improvement: ≥10%")
    print()

    if improvement >= 10:
        print(f"✓ SUCCESS: Achieved {improvement:.1f}% improvement (target: 10%)")
    else:
        print(f"✗ FAIL: Only {improvement:.1f}% improvement (target: 10%)")
        print("  Note: Actual improvement depends on workload characteristics.")
        print("  The cache is most effective with repeated tool calls.")

    print()
    print("Cache Statistics:")
    registry = ToolRegistry()
    executor = BenchmarkToolExecutor()
    normalizer = ArgumentNormalizer()
    config = ToolPipelineConfig(tool_budget=1000)
    pipeline = ToolPipeline(
        tool_registry=registry,
        tool_executor=executor,
        config=config,
        argument_normalizer=normalizer,
    )

    # Run a sample workload
    registry.is_tool_enabled = lambda x: True
    registry.has_tool = lambda x: True

    for call in tool_calls_repeated:
        await pipeline._execute_single_call(call, {})

    stats = pipeline.get_cache_stats()["decision_cache"]
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Validation cache size: {stats['validation_cache_size']}")
    print(f"  Normalization cache size: {stats['normalization_cache_size']}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
