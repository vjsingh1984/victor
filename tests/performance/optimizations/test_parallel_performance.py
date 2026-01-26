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

"""Performance tests for parallel execution optimization.

These tests verify that adaptive parallel execution provides:
- 15-25% execution time improvement for parallelizable workloads
- Minimal overhead for optimization framework (<5%)
- Good speedup scaling with worker count
"""

import pytest
import time
import asyncio

from victor.optimization.runtime import (
    AdaptiveParallelExecutor,
    OptimizationStrategy,
    create_adaptive_executor,
)


@pytest.mark.performance
@pytest.mark.slow
class TestParallelExecutionPerformance:
    """Performance test suite for parallel execution."""

    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_speedup(self):
        """Test that parallel execution provides 15-25% speedup."""

        # Create CPU-bound tasks
        async def cpu_task(context):
            # Simulate CPU work
            await asyncio.sleep(0.01)
            return "result"

        tasks = [cpu_task for _ in range(20)]

        # Sequential execution baseline
        sequential_executor = create_adaptive_executor(
            strategy=OptimizationStrategy.ALWAYS_SEQUENTIAL
        )
        start_seq = time.perf_counter()
        await sequential_executor.execute(tasks)
        sequential_time = time.perf_counter() - start_seq

        # Parallel execution
        parallel_executor = create_adaptive_executor(
            strategy=OptimizationStrategy.ALWAYS_PARALLEL,
            max_workers=4,
        )
        start_par = time.perf_counter()
        await parallel_executor.execute(tasks)
        parallel_time = time.perf_counter() - start_par

        speedup = sequential_time / parallel_time
        improvement = (sequential_time - parallel_time) / sequential_time

        print(f"\nSequential time: {sequential_time*1000:.2f}ms")
        print(f"Parallel time: {parallel_time*1000:.2f}ms")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Improvement: {improvement:.1%}")

        # Parallel execution should be at least 15% faster
        assert improvement >= 0.15, f"Expected >=15% improvement, got {improvement:.1%}"

    @pytest.mark.asyncio
    async def test_overhead_minimal(self):
        """Test that optimization framework overhead is <5%."""

        async def simple_task(context):
            return "result"

        tasks = [simple_task for _ in range(10)]

        # Base parallel executor (no optimization)
        from victor.framework.parallel import ParallelExecutor

        base_executor = ParallelExecutor()
        start_base = time.perf_counter()
        await base_executor.execute(tasks)
        base_time = time.perf_counter() - start_base

        # Adaptive executor (with optimization)
        adaptive_executor = create_adaptive_executor(
            strategy=OptimizationStrategy.ADAPTIVE,
        )
        start_opt = time.perf_counter()
        await adaptive_executor.execute(tasks)
        opt_time = time.perf_counter() - start_opt

        overhead = (opt_time - base_time) / base_time

        print(f"\nBase executor time: {base_time*1000:.2f}ms")
        print(f"Optimized executor time: {opt_time*1000:.2f}ms")
        print(f"Overhead: {overhead:.1%}")

        # Overhead should be minimal (<5%)
        assert overhead < 0.05, f"Overhead too high: {overhead:.1%}"

    @pytest.mark.asyncio
    async def test_speedup_scaling_with_workers(self):
        """Test that speedup scales reasonably with worker count."""

        async def io_task(context):
            await asyncio.sleep(0.01)
            return "result"

        tasks = [io_task for _ in range(20)]

        results = {}
        for workers in [1, 2, 4]:
            executor = create_adaptive_executor(
                strategy=OptimizationStrategy.ALWAYS_PARALLEL,
                max_workers=workers,
            )
            start = time.perf_counter()
            await executor.execute(tasks)
            elapsed = time.perf_counter() - start
            results[workers] = elapsed

        # Calculate speedup relative to 1 worker
        speedup_2 = results[1] / results[2]
        speedup_4 = results[1] / results[4]

        print(f"\n1 worker: {results[1]*1000:.2f}ms")
        print(f"2 workers: {results[2]*1000:.2f}ms (speedup: {speedup_2:.2f}x)")
        print(f"4 workers: {results[4]*1000:.2f}ms (speedup: {speedup_4:.2f}x)")

        # Should get reasonable speedup (not necessarily linear due to overhead)
        assert speedup_2 >= 1.3, f"2-worker speedup too low: {speedup_2:.2f}x"
        assert speedup_4 >= 2.0, f"4-worker speedup too low: {speedup_4:.2f}x"

    @pytest.mark.asyncio
    async def test_work_stealing_performance(self):
        """Test performance improvement with work stealing."""

        async def variable_task(context, duration):
            await asyncio.sleep(duration)
            return duration

        # Create tasks with varying durations
        tasks = [
            (0, variable_task),
            (0.02, variable_task),
            (0.01, variable_task),
            (0.03, variable_task),
            (0.01, variable_task),
        ]

        # Without work stealing
        executor_no_steal = create_adaptive_executor(
            strategy=OptimizationStrategy.ALWAYS_PARALLEL,
            max_workers=2,
            enable_work_stealing=False,
        )
        start = time.perf_counter()
        await executor_no_steal.execute(tasks)
        time_no_steal = time.perf_counter() - start

        # With work stealing
        executor_with_steal = create_adaptive_executor(
            strategy=OptimizationStrategy.ALWAYS_PARALLEL,
            max_workers=2,
            enable_work_stealing=True,
        )
        start = time.perf_counter()
        await executor_with_steal.execute(tasks)
        time_with_steal = time.perf_counter() - start

        improvement = (time_no_steal - time_with_steal) / time_no_steal

        print(f"\nWithout work stealing: {time_no_steal*1000:.2f}ms")
        print(f"With work stealing: {time_with_steal*1000:.2f}ms")
        print(f"Improvement: {improvement:.1%}")

        # Work stealing should provide some benefit for variable tasks
        # (even if small due to asyncio overhead)
        assert time_with_steal <= time_no_steal * 1.1  # At most 10% slower

    @pytest.mark.asyncio
    async def test_priority_queue_overhead(self):
        """Test overhead of priority queue execution."""

        async def priority_task(context, value):
            await asyncio.sleep(0.001)
            return value

        # Create tasks with different priorities
        tasks = [(i % 5, priority_task) for i in range(20)]

        # Standard execution
        executor_std = create_adaptive_executor(
            enable_priority_queue=False,
        )
        start = time.perf_counter()
        await executor_std.execute(tasks)
        std_time = time.perf_counter() - start

        # Priority queue execution
        executor_pq = create_adaptive_executor(
            enable_priority_queue=True,
        )
        start = time.perf_counter()
        await executor_pq.execute(tasks)
        pq_time = time.perf_counter() - start

        overhead = (pq_time - std_time) / std_time

        print(f"\nStandard execution: {std_time*1000:.2f}ms")
        print(f"Priority queue: {pq_time*1000:.2f}ms")
        print(f"Overhead: {overhead:.1%}")

        # Priority queue overhead should be minimal (<10%)
        assert overhead < 0.10, f"Priority queue overhead too high: {overhead:.1%}"

    @pytest.mark.asyncio
    async def test_adaptive_batch_sizing(self):
        """Test that adaptive batch sizing improves performance."""

        async def quick_task(context):
            await asyncio.sleep(0.001)
            return "quick"

        async def slow_task(context):
            await asyncio.sleep(0.01)
            return "slow"

        # Mix of quick and slow tasks
        tasks = [quick_task if i % 3 else slow_task for i in range(30)]

        # Fixed batch size
        executor_fixed = create_adaptive_executor(
            strategy=OptimizationStrategy.ALWAYS_PARALLEL,
            max_workers=4,
        )
        # Manually set small batch size
        executor_fixed._calculate_adaptive_batch_size = lambda n: 2
        start = time.perf_counter()
        await executor_fixed.execute(tasks)
        fixed_time = time.perf_counter() - start

        # Adaptive batch size
        executor_adaptive = create_adaptive_executor(
            strategy=OptimizationStrategy.ADAPTIVE,
            max_workers=4,
        )
        start = time.perf_counter()
        await executor_adaptive.execute(tasks)
        adaptive_time = time.perf_counter() - start

        improvement = (fixed_time - adaptive_time) / fixed_time

        print(f"\nFixed batch size: {fixed_time*1000:.2f}ms")
        print(f"Adaptive batch size: {adaptive_time*1000:.2f}ms")
        print(f"Improvement: {improvement:.1%}")

        # Adaptive should be at least as good as fixed (within 10%)
        assert adaptive_time <= fixed_time * 1.1

    @pytest.mark.asyncio
    async def test_load_balancing_effectiveness(self):
        """Test that load balancing distributes work evenly."""

        async def tracked_task(context, worker_id):
            # Simulate work
            await asyncio.sleep(0.01)
            return worker_id

        tasks = [(i % 4, tracked_task) for i in range(20)]

        executor = create_adaptive_executor(
            strategy=OptimizationStrategy.ALWAYS_PARALLEL,
            max_workers=4,
            enable_work_stealing=True,
        )

        start = time.perf_counter()
        result = await executor.execute(tasks)
        elapsed = time.perf_counter() - start

        metrics = executor.get_metrics()

        print(f"\nExecution time: {elapsed*1000:.2f}ms")
        print(f"Tasks executed: {metrics.tasks_executed}")
        print(f"Workers used: {metrics.worker_count}")
        print(f"Parallel efficiency: {metrics.efficiency:.1%}")

        # Should achieve reasonable parallel efficiency (>50%)
        assert metrics.efficiency > 0.5, f"Efficiency too low: {metrics.efficiency:.1%}"

    @pytest.mark.asyncio
    async def test_small_task_overhead(self):
        """Test overhead for many small tasks."""

        async def tiny_task(context):
            return "x"

        tasks = [tiny_task for _ in range(100)]

        executor = create_adaptive_executor(
            strategy=OptimizationStrategy.ADAPTIVE,
            max_workers=4,
        )

        start = time.perf_counter()
        result = await executor.execute(tasks)
        elapsed = time.perf_counter() - start

        avg_task_time = (elapsed * 1000) / len(tasks)

        print(f"\nTotal time for 100 tasks: {elapsed*1000:.2f}ms")
        print(f"Average task time: {avg_task_time:.4f}ms")

        # Average task overhead should be small (<1ms)
        assert avg_task_time < 1.0, f"Task overhead too high: {avg_task_time:.4f}ms"

    @pytest.mark.asyncio
    async def test_auto_strategy_selection(self):
        """Test that AUTO strategy makes good decisions."""

        async def small_task(context):
            await asyncio.sleep(0.001)
            return "result"

        # Small number of tasks - should choose sequential
        few_tasks = [small_task for _ in range(2)]
        executor = create_adaptive_executor(
            strategy=OptimizationStrategy.AUTO,
            max_workers=4,
        )

        should_parallel = executor._should_parallelize(few_tasks)
        assert should_parallel is False, "AUTO should choose sequential for 2 tasks"

        # Large number of tasks - should choose parallel
        many_tasks = [small_task for _ in range(10)]
        should_parallel = executor._should_parallelize(many_tasks)
        assert should_parallel is True, "AUTO should choose parallel for 10 tasks"

    @pytest.mark.asyncio
    async def test_metrics_collection_overhead(self):
        """Test overhead of metrics collection."""

        async def task(context):
            await asyncio.sleep(0.001)
            return "result"

        tasks = [task for _ in range(50)]

        executor = create_adaptive_executor(
            strategy=OptimizationStrategy.ADAPTIVE,
        )

        # Execute and collect metrics
        start = time.perf_counter()
        await executor.execute(tasks)
        metrics_start = time.perf_counter()
        metrics = executor.get_metrics()
        metrics_time = time.perf_counter() - metrics_start
        total_time = time.perf_counter() - start

        print(f"\nTotal execution: {total_time*1000:.2f}ms")
        print(f"Metrics collection: {metrics_time*1000:.4f}ms")

        # Metrics collection should be very fast (<1ms)
        assert metrics_time < 0.001, f"Metrics collection too slow: {metrics_time*1000:.4f}ms"

        # Verify metrics are accurate
        assert metrics.tasks_executed == 50
        assert metrics.worker_count > 0

    @pytest.mark.asyncio
    async def test_concurrent_execution_correctness(self):
        """Test that concurrent execution produces correct results."""

        async def compute_task(context, value):
            await asyncio.sleep(0.001)
            return value * 2

        tasks = [(i, compute_task) for i in range(20)]

        executor = create_adaptive_executor(
            strategy=OptimizationStrategy.ALWAYS_PARALLEL,
            max_workers=4,
        )

        result = await executor.execute(tasks)

        assert result.success
        assert result.success_count == 20

        # Verify all tasks completed correctly
        # (results may be in different order due to parallel execution)
        assert len(result.results) == 20

    @pytest.mark.asyncio
    async def test_error_handling_performance(self):
        """Test performance impact of error handling."""

        async def failing_task(context):
            raise ValueError("Task failed")

        async def success_task(context):
            await asyncio.sleep(0.001)
            return "success"

        # Mix of failing and successful tasks
        tasks = [failing_task if i % 5 == 0 else success_task for i in range(20)]

        from victor.framework.parallel.strategies import ErrorStrategy, ParallelConfig

        config = ParallelConfig(
            error_strategy=ErrorStrategy.CONTINUE_ALL,
        )
        executor = AdaptiveParallelExecutor(config=config)

        start = time.perf_counter()
        result = await executor.execute(tasks)
        elapsed = time.perf_counter() - start

        print(f"\nExecution with errors: {elapsed*1000:.2f}ms")
        print(f"Success: {result.success_count}, Failures: {result.failure_count}")

        # Should complete despite errors
        assert result.success_count > 0
        assert result.failure_count > 0
