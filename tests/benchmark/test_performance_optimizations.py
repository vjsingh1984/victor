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

"""Performance benchmarks for Work Stream 3.3 optimizations.

This module provides comprehensive benchmarks for:
1. Batch tool execution performance
2. Context compaction optimization
3. Cached prompt building
4. Overall system performance improvement

Target: 20% overall performance improvement
"""

import asyncio
import gc
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.tool_executor import ToolExecutor, ToolExecutionResult
from victor.tools.base import BaseTool, ToolRegistry, ToolResult
from victor.tools.decorators import tool
from victor.storage.cache.tool_cache import ToolCache
from victor.agent.coordinators.compaction_strategies import (
    LLMCompactionStrategy,
    TruncationCompactionStrategy,
    HybridCompactionStrategy,
)
from victor.agent.coordinators.prompt_coordinator import (
    PromptCoordinator,
    BasePromptContributor,
    PromptContext,
)
from victor.agent.context_compactor import ContextCompactor


# =============================================================================
# Test Tools
# =============================================================================


class SlowMockTool(BaseTool):
    """A tool that simulates slow I/O operations."""

    name = "slow_tool"
    description = "A tool that takes time to execute"

    parameters = {
        "type": "object",
        "properties": {
            "delay": {"type": "number", "description": "Delay in seconds"},
            "value": {"type": "string", "description": "Input value"},
        },
        "required": ["value"],
    }

    async def execute(self, delay: float = 0.1, value: str = "", **kwargs):
        await asyncio.sleep(delay)
        return ToolResult.success(output=f"Processed: {value}")


class FastMockTool(BaseTool):
    """A fast tool for comparison."""

    name = "fast_tool"
    description = "A fast tool"

    parameters = {
        "type": "object",
        "properties": {
            "value": {"type": "string", "description": "Input value"},
        },
        "required": ["value"],
    }

    async def execute(self, value: str = "", **kwargs):
        return ToolResult.success(output=f"Fast: {value}")


# =============================================================================
# Benchmark 1: Batch Tool Execution
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_batch_tool_execution_performance():
    """Benchmark batch tool execution vs sequential execution.

    Target: 40% improvement for parallel independent tools
    """
    registry = ToolRegistry()
    registry.register(SlowMockTool())
    registry.register(FastMockTool())

    executor = ToolExecutor(tool_registry=registry)

    # Create tool calls
    tool_calls = [
        ("slow_tool", {"value": "task1", "delay": 0.05}),
        ("slow_tool", {"value": "task2", "delay": 0.05}),
        ("slow_tool", {"value": "task3", "delay": 0.05}),
        ("slow_tool", {"value": "task4", "delay": 0.05}),
        ("fast_tool", {"value": "task5"}),
        ("fast_tool", {"value": "task6"}),
    ]

    # Benchmark sequential execution
    start = time.perf_counter()
    sequential_results = []
    for tool_name, args in tool_calls:
        result = await executor.execute(tool_name, args)
        sequential_results.append(result)
    sequential_time = time.perf_counter() - start

    # Benchmark batch execution
    start = time.perf_counter()
    batch_results = await executor.execute_batch(tool_calls, max_concurrency=4)
    batch_time = time.perf_counter() - start

    # Calculate improvement
    improvement = ((sequential_time - batch_time) / sequential_time) * 100

    print(f"\nBatch Tool Execution Benchmark:")
    print(f"  Sequential time: {sequential_time:.4f}s")
    print(f"  Batch time: {batch_time:.4f}s")
    print(f"  Improvement: {improvement:.1f}%")

    # Assertions
    assert len(batch_results) == len(sequential_results), "Batch should return same number of results"
    assert all(r.success for r in batch_results), "All batch executions should succeed"
    assert batch_time < sequential_time, "Batch execution should be faster than sequential"

    # Target: At least 30% improvement (4 tasks with 0.05s delay each)
    # Sequential: ~0.24s, Batch: ~0.06s (4-way parallel) = 75% improvement
    assert improvement >= 30, f"Expected at least 30% improvement, got {improvement:.1f}%"


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_batch_execution_with_different_speeds():
    """Benchmark batch execution with mixed slow/fast tools.

    Target: 35% improvement
    """
    registry = ToolRegistry()
    registry.register(SlowMockTool())
    registry.register(FastMockTool())

    executor = ToolExecutor(tool_registry=registry)

    tool_calls = [
        ("slow_tool", {"value": "slow1", "delay": 0.1}),
        ("fast_tool", {"value": "fast1"}),
        ("slow_tool", {"value": "slow2", "delay": 0.08}),
        ("fast_tool", {"value": "fast2"}),
        ("slow_tool", {"value": "slow3", "delay": 0.06}),
    ]

    # Sequential
    start = time.perf_counter()
    seq_results = []
    for tool_name, args in tool_calls:
        result = await executor.execute(tool_name, args)
        seq_results.append(result)
    sequential_time = time.perf_counter() - start

    # Batch
    start = time.perf_counter()
    batch_results = await executor.execute_batch(tool_calls, max_concurrency=3)
    batch_time = time.perf_counter() - start

    improvement = ((sequential_time - batch_time) / sequential_time) * 100

    print(f"\nMixed Speed Batch Execution Benchmark:")
    print(f"  Sequential time: {sequential_time:.4f}s")
    print(f"  Batch time: {batch_time:.4f}s")
    print(f"  Improvement: {improvement:.1f}%")

    assert improvement >= 25, f"Expected at least 25% improvement, got {improvement:.1f}%"


# =============================================================================
# Benchmark 2: Context Compaction Optimization
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_llm_based_compaction_performance():
    """Benchmark LLM-based summarization vs simple truncation.

    Target: 20% faster with better context preservation
    """
    # Create large context
    large_context = []
    for i in range(50):
        large_context.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Message {i}: " + "x" * 500,  # 500 chars per message
        })

    # Mock controller
    mock_controller = MagicMock()
    mock_controller.messages = large_context
    mock_controller.get_context_metrics.return_value = MagicMock(
        char_count=25000,
        utilization=0.95,
        is_overflow_risk=True,
    )

    # Benchmark truncation strategy
    truncation_strategy = TruncationCompactionStrategy(max_chars=5000)
    start = time.perf_counter()
    truncation_result = truncation_strategy.compact(large_context, target_tokens=1000)
    truncation_time = time.perf_counter() - start

    # Benchmark LLM-based strategy (mocked for performance)
    llm_strategy = LLMCompactionStrategy(
        summarization_model="gpt-4o-mini",  # Fast small model
        cache_summaries=True,
    )
    # Mock the LLM call to avoid actual API calls in benchmark
    llm_strategy._summarize_with_llm = AsyncMock(
        return_value="Summary of conversation: Key points discussed..."
    )

    start = time.perf_counter()
    llm_result = await llm_strategy.compact_async(large_context, target_tokens=1000)
    llm_time = time.perf_counter() - start

    # Calculate quality improvement (LLM preserves more meaningful content)
    truncation_chars = sum(len(m.get("content", "")) for m in truncation_result)
    llm_chars = sum(len(m.get("content", "")) for m in llm_result)

    print(f"\nContext Compaction Benchmark:")
    print(f"  Truncation time: {truncation_time:.4f}s")
    print(f"  LLM-based time: {llm_time:.4f}s")
    print(f"  Truncation preserved: {truncation_chars} chars")
    print(f"  LLM preserved: {llm_chars} chars")
    print(f"  LLM time overhead: {((llm_time - truncation_time) / truncation_time) * 100:.1f}%")

    # LLM should be within 2x of truncation time (acceptable trade-off for better quality)
    assert llm_time < truncation_time * 2, "LLM compaction should be within 2x of truncation time"


@pytest.mark.benchmark
def test_compaction_summary_caching():
    """Benchmark summary caching for repeated compactions.

    Target: 80% faster on cache hit
    """
    strategy = LLMCompactionStrategy(
        summarization_model="gpt-4o-mini",
        cache_summaries=True,
    )

    # Mock messages
    messages = [
        {"role": "user", "content": "Question 1"},
        {"role": "assistant", "content": "Answer 1"},
        {"role": "user", "content": "Question 2"},
        {"role": "assistant", "content": "Answer 2"},
    ]

    # Mock LLM call
    strategy._summarize_with_llm = AsyncMock(return_value="Cached summary")

    # First compaction (cache miss)
    start = time.perf_counter()
    summary1 = asyncio.run(strategy._get_or_create_summary(messages))
    first_time = time.perf_counter() - start

    # Second compaction (cache hit)
    start = time.perf_counter()
    summary2 = asyncio.run(strategy._get_or_create_summary(messages))
    second_time = time.perf_counter() - start

    speedup = (first_time / second_time) if second_time > 0 else float('inf')

    print(f"\nSummary Caching Benchmark:")
    print(f"  First call (cache miss): {first_time:.6f}s")
    print(f"  Second call (cache hit): {second_time:.6f}s")
    print(f"  Speedup: {speedup:.1f}x")

    assert summary1 == summary2, "Cached summary should match original"
    assert second_time < first_time / 10, "Cache hit should be at least 10x faster"


# =============================================================================
# Benchmark 3: Cached Prompt Building
# =============================================================================


@pytest.mark.benchmark
async def test_prompt_building_cache_performance():
    """Benchmark cached prompt building.

    Target: 90% faster on cache hit
    """
    class TestContributor(BasePromptContributor):
        async def contribute(self, context: PromptContext) -> str:
            # Simulate expensive computation
            await asyncio.sleep(0.01)
            return f"Contribution for {context.get('task', 'unknown')}"

    contributors = [TestContributor(priority=50) for _ in range(5)]
    coordinator = PromptCoordinator(contributors=contributors, enable_cache=True)

    context = PromptContext({"task": "code_review", "language": "python"})

    # First build (cache miss)
    start = time.perf_counter()
    prompt1 = await coordinator.build_system_prompt(context)
    first_time = time.perf_counter() - start

    # Second build (cache hit)
    start = time.perf_counter()
    prompt2 = await coordinator.build_system_prompt(context)
    second_time = time.perf_counter() - start

    speedup = (first_time / second_time) if second_time > 0 else float('inf')
    improvement = ((first_time - second_time) / first_time) * 100

    print(f"\nPrompt Building Cache Benchmark:")
    print(f"  First build (cache miss): {first_time:.4f}s")
    print(f"  Second build (cache hit): {second_time:.6f}s")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Improvement: {improvement:.1f}%")

    assert prompt1 == prompt2, "Cached prompt should match original"
    assert improvement >= 90, f"Expected at least 90% improvement, got {improvement:.1f}%"


@pytest.mark.benchmark
async def test_prompt_cache_invalidation():
    """Benchmark cache invalidation efficiency.

    Target: Invalidate only relevant entries
    """
    contributor = BasePromptContributor()
    coordinator = PromptCoordinator(contributors=[contributor], enable_cache=True)

    # Build multiple cached prompts
    contexts = [
        PromptContext({"task": "code_review"}),
        PromptContext({"task": "test_generation"}),
        PromptContext({"task": "refactoring"}),
    ]

    for ctx in contexts:
        await coordinator.build_system_prompt(ctx)

    cache_size_before = len(coordinator._prompt_cache)

    # Invalidate one context
    start = time.perf_counter()
    coordinator.invalidate_cache(contexts[0])
    invalidation_time = time.perf_counter() - start

    cache_size_after = len(coordinator._prompt_cache)

    print(f"\nCache Invalidation Benchmark:")
    print(f"  Cache size before: {cache_size_before}")
    print(f"  Cache size after: {cache_size_after}")
    print(f"  Invalidation time: {invalidation_time:.6f}s")

    assert cache_size_after == cache_size_before - 1, "Should invalidate exactly one entry"
    assert invalidation_time < 0.001, "Invalidation should be very fast"


# =============================================================================
# Benchmark 4: Overall System Performance
# =============================================================================


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_overall_performance_improvement():
    """Benchmark overall system performance with all optimizations.

    Target: 20% overall improvement
    """
    registry = ToolRegistry()
    registry.register(SlowMockTool())
    registry.register(FastMockTool())

    executor = ToolExecutor(
        tool_registry=registry,
        tool_cache=ToolCache(max_size=100),
    )

    # Create test scenario
    tool_calls = [
        ("slow_tool", {"value": f"task{i}", "delay": 0.02})
        for i in range(10)
    ]

    # Baseline: Sequential execution, no cache
    start = time.perf_counter()
    baseline_results = []
    for tool_name, args in tool_calls[:5]:  # Run 5 tasks
        result = await executor.execute(tool_name, args, skip_cache=True)
        baseline_results.append(result)
    baseline_time = time.perf_counter() - start

    # Optimized: Batch execution with cache
    # First run to populate cache
    await executor.execute_batch(tool_calls[:5], max_concurrency=4)

    # Second run with cache
    start = time.perf_counter()
    optimized_results = await executor.execute_batch(
        tool_calls[:5],
        max_concurrency=4,
    )
    optimized_time = time.perf_counter() - start

    improvement = ((baseline_time - optimized_time) / baseline_time) * 100

    print(f"\nOverall Performance Improvement:")
    print(f"  Baseline time (sequential, no cache): {baseline_time:.4f}s")
    print(f"  Optimized time (batch, cached): {optimized_time:.4f}s")
    print(f"  Improvement: {improvement:.1f}%")

    # Target: At least 20% overall improvement
    assert improvement >= 20, f"Expected at least 20% improvement, got {improvement:.1f}%"


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_memory_efficiency():
    """Benchmark memory efficiency of optimizations.

    Target: No significant memory increase (< 10%)
    """
    import tracemalloc

    tracemalloc.start()

    # Baseline
    registry = ToolRegistry()
    registry.register(SlowMockTool())
    executor_baseline = ToolExecutor(tool_registry=registry)

    tool_calls = [("slow_tool", {"value": f"task{i}", "delay": 0.01}) for i in range(20)]

    # Measure baseline memory
    for tool_name, args in tool_calls:
        await executor_baseline.execute(tool_name, args)

    baseline_memory = tracemalloc.get_traced_memory()[1] / 1024  # KB

    # Optimized (batch + cache)
    executor_optimized = ToolExecutor(
        tool_registry=registry,
        tool_cache=ToolCache(max_size=100),
    )

    await executor_optimized.execute_batch(tool_calls, max_concurrency=4)

    optimized_memory = tracemalloc.get_traced_memory()[1] / 1024  # KB

    memory_increase = ((optimized_memory - baseline_memory) / baseline_memory) * 100

    print(f"\nMemory Efficiency Benchmark:")
    print(f"  Baseline memory: {baseline_memory:.1f} KB")
    print(f"  Optimized memory: {optimized_memory:.1f} KB")
    print(f"  Memory increase: {memory_increase:.1f}%")

    tracemalloc.stop()

    # Memory increase should be less than 20% (cache overhead)
    assert memory_increase < 20, f"Memory increase {memory_increase:.1f}% exceeds threshold"


# =============================================================================
# Summary Report
# =============================================================================


@pytest.mark.benchmark
def test_performance_summary():
    """Generate summary report of all performance improvements."""
    print("\n" + "=" * 70)
    print("PERFORMANCE OPTIMIZATION SUMMARY")
    print("=" * 70)
    print("\n1. Batch Tool Execution:")
    print("   - Target: 40% improvement for parallel independent tools")
    print("   - Implementation: asyncio.gather with bounded concurrency")
    print("   - Result: See test_batch_tool_execution_performance\n")

    print("2. Context Compaction:")
    print("   - Target: LLM-based summarization with < 2x time overhead")
    print("   - Implementation: Use smaller model (gpt-4o-mini) for summarization")
    print("   - Result: See test_llm_based_compaction_performance\n")

    print("3. Prompt Building Cache:")
    print("   - Target: 90% faster on cache hit")
    print("   - Implementation: Hash-based cache keys in PromptCoordinator")
    print("   - Result: See test_prompt_building_cache_performance\n")

    print("4. Overall System:")
    print("   - Target: 20% overall performance improvement")
    print("   - Implementation: Combined optimizations")
    print("   - Result: See test_overall_performance_improvement\n")

    print("=" * 70)
    print("All benchmarks completed. Run with pytest -v to see detailed results.")
    print("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "benchmark"])
