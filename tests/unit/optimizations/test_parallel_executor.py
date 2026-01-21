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

"""Unit tests for AdaptiveParallelExecutor."""

import pytest

from victor.optimizations.parallel_executor import (
    AdaptiveParallelExecutor,
    OptimizationStrategy,
    PerformanceMetrics,
    TaskWithPriority,
    create_adaptive_executor,
)


class TestAdaptiveParallelExecutor:
    """Test suite for AdaptiveParallelExecutor."""

    @pytest.mark.asyncio
    async def test_execute_empty_tasks(self):
        """Test executing empty task list."""
        executor = AdaptiveParallelExecutor()

        result = await executor.execute([])

        assert result.success
        assert result.total_count == 0
        assert result.results == []

    @pytest.mark.asyncio
    async def test_execute_single_task(self):
        """Test executing a single task."""
        executor = AdaptiveParallelExecutor()

        async def task1(context):
            return "result1"

        result = await executor.execute([task1])

        assert result.success
        assert result.total_count == 1
        assert result.success_count == 1

    @pytest.mark.asyncio
    async def test_execute_multiple_tasks_always_sequential(self):
        """Test sequential execution with ALWAYS_SEQUENTIAL strategy."""
        executor = AdaptiveParallelExecutor(
            strategy=OptimizationStrategy.ALWAYS_SEQUENTIAL
        )

        execution_order = []

        async def task1(context):
            execution_order.append(1)
            return "result1"

        async def task2(context):
            execution_order.append(2)
            return "result2"

        await executor.execute([task1, task2])

        assert execution_order == [1, 2]

    @pytest.mark.asyncio
    async def test_execute_multiple_tasks_always_parallel(self):
        """Test parallel execution with ALWAYS_PARALLEL strategy."""
        executor = AdaptiveParallelExecutor(
            strategy=OptimizationStrategy.ALWAYS_PARALLEL,
            max_workers=2,
        )

        async def task1(context):
            return "result1"

        async def task2(context):
            return "result2"

        result = await executor.execute([task1, task2])

        assert result.success
        assert result.total_count == 2
        assert result.success_count == 2

    @pytest.mark.asyncio
    async def test_execute_with_auto_strategy_small_task_count(self):
        """Test AUTO strategy with small task count."""
        executor = AdaptiveParallelExecutor(
            strategy=OptimizationStrategy.AUTO,
            max_workers=4,
        )

        async def task1(context):
            return "result1"

        async def task2(context):
            return "result2"

        result = await executor.execute([task1, task2])

        assert result.success

    @pytest.mark.asyncio
    async def test_execute_with_auto_strategy_large_task_count(self):
        """Test AUTO strategy with large task count (should parallelize)."""
        executor = AdaptiveParallelExecutor(
            strategy=OptimizationStrategy.AUTO,
            max_workers=2,
        )

        async def make_task(i):
            async def task(context):
                return f"result{i}"
            return task

        tasks = [make_task(i) for i in range(5)]
        result = await executor.execute(tasks)

        assert result.success
        assert result.total_count == 5

    @pytest.mark.asyncio
    async def test_execute_with_fail_fast_error(self):
        """Test execution with FAIL_FAST error strategy."""
        from victor.framework.parallel.strategies import ErrorStrategy

        config = pytest.importorskip("victor.framework.parallel.strategies").ParallelConfig(
            error_strategy=ErrorStrategy.FAIL_FAST
        )
        executor = AdaptiveParallelExecutor(config=config)

        async def failing_task(context):
            raise ValueError("Task failed")

        async def success_task(context):
            return "success"

        result = await executor.execute([failing_task, success_task])

        assert not result.success
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_get_metrics(self):
        """Test getting performance metrics."""
        executor = AdaptiveParallelExecutor()

        async def task1(context):
            return "result1"

        await executor.execute([task1])
        metrics = executor.get_metrics()

        assert metrics.tasks_executed == 1
        assert metrics.total_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_reset_metrics(self):
        """Test resetting metrics."""
        executor = AdaptiveParallelExecutor()

        async def task1(context):
            return "result1"

        await executor.execute([task1])
        metrics_before = executor.get_metrics()
        assert metrics_before.tasks_executed > 0

        executor.reset_metrics()
        metrics_after = executor.get_metrics()
        assert metrics_after.tasks_executed == 0

    @pytest.mark.asyncio
    async def test_execute_with_priority_queue(self):
        """Test execution with priority queue enabled."""
        executor = AdaptiveParallelExecutor(
            enable_priority_queue=True,
        )

        async def priority_task(context):
            return "result"

        # Create tasks with priorities (priority, task) tuples
        tasks = [
            (2, priority_task),
            (1, priority_task),
            (3, priority_task),
        ]

        result = await executor.execute(tasks)

        # Should execute successfully with priority queue
        assert result.success
        assert result.total_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_work_stealing(self):
        """Test execution with work stealing enabled."""
        executor = AdaptiveParallelExecutor(
            enable_work_stealing=True,
            max_workers=2,
        )

        async def task1(context):
            return "result1"

        async def task2(context):
            return "result2"

        async def task3(context):
            return "result3"

        result = await executor.execute([task1, task2, task3])

        assert result.success
        assert result.total_count == 3

    @pytest.mark.asyncio
    async def test_should_parallelize_with_always_parallel(self):
        """Test _should_parallelize with ALWAYS_PARALLEL."""
        executor = AdaptiveParallelExecutor(
            strategy=OptimizationStrategy.ALWAYS_PARALLEL
        )

        assert executor._should_parallelize([]) is True
        assert executor._should_parallelize([None]) is True

    @pytest.mark.asyncio
    async def test_should_parallelize_with_always_sequential(self):
        """Test _should_parallelize with ALWAYS_SEQUENTIAL."""
        executor = AdaptiveParallelExecutor(
            strategy=OptimizationStrategy.ALWAYS_SEQUENTIAL
        )

        assert executor._should_parallelize([]) is False
        assert executor._should_parallelize([None, None, None]) is False

    @pytest.mark.asyncio
    async def test_should_parallelize_with_auto_single_task(self):
        """Test _should_parallelize with AUTO and single task."""
        executor = AdaptiveParallelExecutor(
            strategy=OptimizationStrategy.AUTO
        )

        assert executor._should_parallelize([None]) is False

    @pytest.mark.asyncio
    async def test_should_parallelize_with_auto_multiple_tasks(self):
        """Test _should_parallelize with AUTO and multiple tasks."""
        executor = AdaptiveParallelExecutor(
            strategy=OptimizationStrategy.AUTO,
            max_workers=2,
        )

        # With 3 tasks, should parallelize
        assert executor._should_parallelize([None, None, None]) is True

    @pytest.mark.asyncio
    async def test_calculate_adaptive_batch_size(self):
        """Test adaptive batch size calculation."""
        executor = AdaptiveParallelExecutor(max_workers=4)

        # For small number of tasks
        batch_size = executor._calculate_adaptive_batch_size(2)
        assert batch_size == 1

        # For many tasks
        batch_size = executor._calculate_adaptive_batch_size(100)
        assert 1 <= batch_size <= 25

    @pytest.mark.asyncio
    async def test_create_batches(self):
        """Test batch creation."""
        executor = AdaptiveParallelExecutor()

        tasks = [f"task{i}" for i in range(10)]
        batches = executor._create_batches(tasks, 3)

        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[3]) == 1

    @pytest.mark.asyncio
    async def test_performance_metrics_parallel_ratio(self):
        """Test PerformanceMetrics parallel_ratio calculation."""
        metrics = PerformanceMetrics(
            total_duration_ms=100.0,
            parallel_duration_ms=80.0,
        )

        assert metrics.parallel_ratio == 0.8

    @pytest.mark.asyncio
    async def test_performance_metrics_efficiency(self):
        """Test PerformanceMetrics efficiency calculation."""
        metrics = PerformanceMetrics(
            speedup_factor=3.0,
            worker_count=4,
        )

        assert metrics.efficiency == 0.75


class TestTaskWithPriority:
    """Test suite for TaskWithPriority."""

    def test_task_with_priority_comparison(self):
        """Test TaskWithPriority comparison for priority queue."""
        task1 = TaskWithPriority(priority=2, task_id=1, task="task1")
        task2 = TaskWithPriority(priority=1, task_id=2, task="task2")

        # Lower priority value = higher priority
        assert task2 < task1

    def test_task_with_priority_attributes(self):
        """Test TaskWithPriority attributes."""
        task = TaskWithPriority(
            priority=1,
            task_id=42,
            task="my_task",
        )

        assert task.priority == 1
        assert task.task_id == 42
        assert task.task == "my_task"


class TestCreateAdaptiveExecutor:
    """Test suite for create_adaptive_executor factory function."""

    def test_create_with_string_strategy(self):
        """Test creating executor with string strategy."""
        executor = create_adaptive_executor(
            strategy="adaptive",
            max_workers=4,
        )

        assert executor.strategy == OptimizationStrategy.ADAPTIVE
        assert executor.max_workers == 4

    def test_create_with_enum_strategy(self):
        """Test creating executor with enum strategy."""
        executor = create_adaptive_executor(
            strategy=OptimizationStrategy.ALWAYS_PARALLEL,
            max_workers=2,
        )

        assert executor.strategy == OptimizationStrategy.ALWAYS_PARALLEL

    def test_create_with_work_stealing(self):
        """Test creating executor with work stealing enabled."""
        executor = create_adaptive_executor(
            enable_work_stealing=True,
        )

        assert executor.enable_work_stealing is True

    def test_create_with_priority_queue(self):
        """Test creating executor with priority queue enabled."""
        executor = create_adaptive_executor(
            enable_priority_queue=True,
        )

        assert executor.enable_priority_queue is True

    def test_create_with_string_join_strategy(self):
        """Test creating executor with string join strategy."""
        executor = create_adaptive_executor(
            join_strategy="majority",
        )

        assert executor.config.join_strategy.value == "majority"

    def test_create_with_string_error_strategy(self):
        """Test creating executor with string error strategy."""
        executor = create_adaptive_executor(
            error_strategy="collect_errors",
        )

        assert executor.config.error_strategy.value == "collect_errors"
