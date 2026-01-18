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

"""Tests for parallel executor."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.framework.parallel import (
    ParallelExecutor,
    ParallelConfig,
    ParallelExecutionResult,
    ProgressEvent,
    ResourceLimit,
    JoinStrategy,
    ErrorStrategy,
    execute_parallel,
    execute_parallel_with_config,
    create_parallel_executor,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def async_tasks():
    """Fixture providing sample async tasks."""

    async def task1(**kwargs):
        await asyncio.sleep(0.01)
        return "result1"

    async def task2(**kwargs):
        await asyncio.sleep(0.01)
        return "result2"

    async def task3(**kwargs):
        await asyncio.sleep(0.01)
        return "result3"

    return [task1, task2, task3]


@pytest.fixture
def failing_tasks():
    """Fixture providing tasks that fail."""

    async def success_task(**kwargs):
        await asyncio.sleep(0.01)
        return "success"

    async def fail_task(**kwargs):
        await asyncio.sleep(0.01)
        raise ValueError("Task failed")

    return [success_task, fail_task, success_task]


@pytest.fixture
def timeout_task():
    """Fixture providing a task that times out."""

    async def slow_task(**kwargs):
        await asyncio.sleep(10)  # Longer than test timeout
        return "should not complete"

    return slow_task


# =============================================================================
# Test ParallelExecutor
# =============================================================================


class TestParallelExecutor:
    """Tests for ParallelExecutor class."""

    @pytest.mark.asyncio
    async def test_execute_all_success(self, async_tasks):
        """Test executing all tasks successfully."""
        executor = ParallelExecutor()
        result = await executor.execute(async_tasks)

        assert result.success is True
        assert result.success_count == 3
        assert result.failure_count == 0
        assert len(result.results) == 3

    @pytest.mark.asyncio
    async def test_execute_empty_tasks(self):
        """Test executing empty task list."""
        executor = ParallelExecutor()
        result = await executor.execute([])

        assert result.success is True
        assert result.total_count == 0
        assert result.success_count == 0

    @pytest.mark.asyncio
    async def test_execute_with_fail_fast(self, failing_tasks):
        """Test fail-fast error handling."""
        config = ParallelConfig(
            error_strategy=ErrorStrategy.FAIL_FAST,
        )
        executor = ParallelExecutor(config)
        result = await executor.execute(failing_tasks)

        # With fail-fast and ALL join strategy, errors cause failure
        assert result.failure_count > 0
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_execute_with_continue_all(self, failing_tasks):
        """Test continue-all error handling."""
        config = ParallelConfig(
            join_strategy=JoinStrategy.ANY,
            error_strategy=ErrorStrategy.CONTINUE_ALL,
        )
        executor = ParallelExecutor(config)
        result = await executor.execute(failing_tasks)

        # ANY strategy should succeed with at least one success
        assert result.success is True
        assert result.success_count == 2
        assert result.failure_count == 1

    @pytest.mark.asyncio
    async def test_execute_with_collect_errors(self, failing_tasks):
        """Test collect-errors error handling."""
        config = ParallelConfig(
            join_strategy=JoinStrategy.ANY,
            error_strategy=ErrorStrategy.COLLECT_ERRORS,
        )
        executor = ParallelExecutor(config)
        result = await executor.execute(failing_tasks)

        assert result.success is True
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_execute_with_max_concurrent(self, async_tasks):
        """Test concurrency limit with semaphore."""
        config = ParallelConfig(
            resource_limit=ResourceLimit(max_concurrent=2),
        )
        executor = ParallelExecutor(config)
        result = await executor.execute(async_tasks)

        assert result.success is True
        assert result.success_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_any_join_strategy(self, failing_tasks):
        """Test ANY join strategy."""
        config = ParallelConfig(join_strategy=JoinStrategy.ANY)
        executor = ParallelExecutor(config)
        result = await executor.execute(failing_tasks)

        # ANY should succeed with at least one success
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_majority_join_strategy(self, async_tasks):
        """Test MAJORITY join strategy."""
        config = ParallelConfig(join_strategy=JoinStrategy.MAJORITY)
        executor = ParallelExecutor(config)
        result = await executor.execute(async_tasks)

        # All succeed, so majority passes
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_first_join_strategy(self, async_tasks):
        """Test FIRST join strategy."""
        config = ParallelConfig(join_strategy=JoinStrategy.FIRST)
        executor = ParallelExecutor(config)
        result = await executor.execute(async_tasks)

        assert result.success is True
        # FIRST strategy returns first successful result
        assert result.results == "result1"

    @pytest.mark.asyncio
    async def test_execute_with_n_of_m_join_strategy(self, async_tasks):
        """Test N_OF_M join strategy."""
        config = ParallelConfig(
            join_strategy=JoinStrategy.N_OF_M,
            n_of_m=2,
        )
        executor = ParallelExecutor(config)
        result = await executor.execute(async_tasks)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_context(self):
        """Test passing context to tasks."""
        context_passed = []

        async def context_task(**kwargs):
            context_passed.append(kwargs.get("context"))
            return "done"

        executor = ParallelExecutor()
        test_context = {"key": "value"}
        await executor.execute([context_task], context=test_context)

        assert len(context_passed) == 1
        assert context_passed[0] == test_context

    @pytest.mark.asyncio
    async def test_execute_with_tuple_tasks(self):
        """Test tasks specified as tuples with kwargs."""

        async def kwargs_task(**kwargs):
            return kwargs.get("value", "default")

        executor = ParallelExecutor()
        result = await executor.execute(
            [
                (kwargs_task, {"value": "custom"}),
            ]
        )

        assert result.success is True
        assert result.results == ["custom"]

    @pytest.mark.asyncio
    async def test_progress_callback(self, async_tasks):
        """Test progress callback is invoked."""
        events = []

        def callback(task_id, status, result_or_error):
            events.append((task_id, status, result_or_error))

        executor = ParallelExecutor(progress_callback=callback)
        result = await executor.execute(async_tasks)

        assert result.success is True
        # Should have events for started and completed for each task
        assert len(events) >= 3  # At least 3 started events

    @pytest.mark.asyncio
    async def test_set_config(self, async_tasks):
        """Test updating configuration."""
        executor = ParallelExecutor()
        new_config = ParallelConfig(join_strategy=JoinStrategy.ANY)
        executor.set_config(new_config)

        result = await executor.execute(async_tasks)

        assert result.success is True
        assert result.strategy_used == "any"

    @pytest.mark.asyncio
    async def test_set_progress_callback(self, async_tasks):
        """Test updating progress callback."""
        events = []

        def callback(task_id, status, result_or_error):
            events.append(task_id)

        executor = ParallelExecutor()
        executor.set_progress_callback(callback)
        await executor.execute(async_tasks)

        assert len(events) >= 3

    @pytest.mark.asyncio
    async def test_execute_stream(self, async_tasks):
        """Test streaming execution with progress events."""
        executor = ParallelExecutor()
        events = []

        async for event in executor.execute_stream(async_tasks):
            events.append(event)

        assert len(events) == 3
        assert all(isinstance(e, ProgressEvent) for e in events)
        assert all(e.status in ["completed", "failed"] for e in events)

    @pytest.mark.asyncio
    async def test_execute_stream_with_failures(self, failing_tasks):
        """Test streaming execution with task failures."""
        executor = ParallelExecutor()
        events = []

        async for event in executor.execute_stream(failing_tasks):
            events.append(event)

        assert len(events) == 3
        completed = [e for e in events if e.status == "completed"]
        failed = [e for e in events if e.status == "failed"]
        assert len(completed) == 2
        assert len(failed) == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_execute_parallel(self, async_tasks):
        """Test execute_parallel convenience function."""
        result = await execute_parallel(
            async_tasks,
            join_strategy=JoinStrategy.ALL,
        )

        assert result.success is True
        assert result.success_count == 3

    @pytest.mark.asyncio
    async def test_execute_parallel_with_limits(self, async_tasks):
        """Test execute_parallel with resource limits."""
        result = await execute_parallel(
            async_tasks,
            max_concurrent=2,
            timeout=30.0,
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_parallel_with_config(self, async_tasks):
        """Test execute_parallel_with_config function."""
        config = ParallelConfig(
            join_strategy=JoinStrategy.MAJORITY,
            error_strategy=ErrorStrategy.COLLECT_ERRORS,
        )

        result = await execute_parallel_with_config(async_tasks, config)

        assert result.success is True

    def test_create_parallel_executor(self):
        """Test create_parallel_executor factory function."""
        executor = create_parallel_executor(
            join_strategy="any",
            error_strategy="continue_all",
            max_concurrent=5,
        )

        assert isinstance(executor, ParallelExecutor)
        assert executor.config.join_strategy == JoinStrategy.ANY
        assert executor.config.error_strategy == ErrorStrategy.CONTINUE_ALL
        assert executor.config.resource_limit.max_concurrent == 5

    def test_create_parallel_executor_with_enums(self):
        """Test factory with enum arguments."""
        executor = create_parallel_executor(
            join_strategy=JoinStrategy.FIRST,
            error_strategy=ErrorStrategy.FAIL_FAST,
        )

        assert executor.config.join_strategy == JoinStrategy.FIRST


class TestParallelExecutionResult:
    """Tests for ParallelExecutionResult."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = ParallelExecutionResult(
            success=True,
            results=["a", "b", "c"],
            errors=[],
            total_count=3,
            success_count=3,
            failure_count=0,
            duration_seconds=1.5,
            strategy_used="all",
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["results"] == ["a", "b", "c"]
        assert data["total_count"] == 3
        assert data["strategy_used"] == "all"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "success": False,
            "results": ["a"],
            "errors": ["Error 1", "Error 2"],
            "total_count": 3,
            "success_count": 1,
            "failure_count": 2,
            "duration_seconds": 2.0,
            "strategy_used": "any",
        }

        result = ParallelExecutionResult.from_dict(data)

        assert result.success is False
        assert result.results == ["a"]
        assert len(result.errors) == 2
        assert result.total_count == 3


class TestYAMLHandler:
    """Tests for YAML workflow handler integration."""

    @pytest.mark.asyncio
    async def test_parallel_executor_handler(self):
        """Test ParallelExecutorHandler as workflow handler."""
        from victor.framework.parallel.executor import ParallelExecutorHandler

        # Create a simple mock node class
        class MockNode:
            def __init__(self):
                self.id = "test_parallel"
                self.output_key = "test_output"
                self.join_strategy = "all"
                self.error_strategy = "collect_errors"
                self.max_concurrent = 3
                self.timeout = 30.0
                self.tools = []
                self.input_mapping = {}

            def has_config(self):
                return False

        node = MockNode()

        # Mock context
        context = MagicMock()
        context.data = {}
        context.set = MagicMock()

        # Mock tool registry
        tool_registry = MagicMock()

        # Create tasks that succeed
        async def mock_task(**kwargs):
            return f"result_{kwargs.get('index', 0)}"

        node.tasks = [mock_task, mock_task, mock_task]

        handler = ParallelExecutorHandler()
        result = await handler(node, context, tool_registry)

        assert result.status.value == "completed"
        assert context.set.called

    @pytest.mark.asyncio
    async def test_extract_config(self):
        """Test config extraction from node."""
        from victor.framework.parallel.executor import ParallelExecutorHandler

        handler = ParallelExecutorHandler()

        # Node with parallel_config attribute
        class MockNode1:
            def __init__(self, config):
                self.parallel_config = config

        node1 = MockNode1(ParallelConfig(join_strategy=JoinStrategy.MAJORITY))

        config1 = handler._extract_config(node1)
        assert config1.join_strategy == JoinStrategy.MAJORITY

        # Node with individual attributes
        class MockNode2:
            def __init__(self):
                self.join_strategy = "any"
                self.error_strategy = "fail_fast"
                self.max_concurrent = 5
                self.timeout = 60.0

        node2 = MockNode2()

        config2 = handler._extract_config(node2)
        assert config2.join_strategy == JoinStrategy.ANY
        assert config2.resource_limit.max_concurrent == 5


class TestTimeout:
    """Tests for timeout handling."""

    @pytest.mark.asyncio
    async def test_task_timeout(self):
        """Test that tasks timeout as configured."""

        async def slow_task(**kwargs):
            await asyncio.sleep(2)  # Sleep longer than timeout
            return "should not happen"

        config = ParallelConfig(
            resource_limit=ResourceLimit(timeout=0.1),
        )
        executor = ParallelExecutor(config)
        result = await executor.execute([slow_task])

        assert result.failure_count == 1
        assert len(result.errors) > 0
        # TimeoutError is raised - it may be empty or have a message
        assert isinstance(result.errors[0], (asyncio.TimeoutError, TimeoutError))


class TestResourceLimitValidation:
    """Tests for resource limit validation in executor."""

    @pytest.mark.asyncio
    async def test_semaphore_is_created(self):
        """Test that semaphore is created when max_concurrent is set."""
        config = ParallelConfig(
            resource_limit=ResourceLimit(max_concurrent=3),
        )
        executor = ParallelExecutor(config)

        assert executor._semaphore is not None
        assert executor._semaphore._value == 3

    @pytest.mark.asyncio
    async def test_no_semaphore_without_limit(self):
        """Test that no semaphore is created when no limit."""
        config = ParallelConfig()
        executor = ParallelExecutor(config)

        assert executor._semaphore is None
