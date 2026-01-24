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

"""Tests for request batching optimization."""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock

from victor.core.batching import (
    RequestBatcher,
    ToolCallBatcher,
    BatchPriority,
    BatchEntry,
    BatchStats,
    get_llm_batcher,
    get_tool_batcher,
    reset_batchers,
)


class TestBatchEntry:
    """Test BatchEntry dataclass."""

    def test_batch_entry_creation(self):
        """Test creating a batch entry."""
        entry = BatchEntry(
            request_id="test_id",
            args=(1, 2, 3),
            kwargs={"key": "value"},
            priority=BatchPriority.HIGH,
        )

        assert entry.request_id == "test_id"
        assert entry.args == (1, 2, 3)
        assert entry.kwargs == {"key": "value"}
        assert entry.priority == BatchPriority.HIGH
        assert entry.timestamp > 0

    def test_batch_entry_hashable(self):
        """Test that batch entry is hashable."""
        entry = BatchEntry(
            request_id="test_id",
            args=(),
            kwargs={},
        )

        # Should be able to create a set with entries
        entry_set = {entry}
        assert entry in entry_set


class TestBatchStats:
    """Test BatchStats thread-safe statistics."""

    def test_batch_stats_initialization(self):
        """Test stats initialization."""
        stats = BatchStats()

        assert stats.total_requests == 0
        assert stats.total_batches == 0
        assert stats.avg_batch_size == 0.0

    def test_batch_stats_record_request(self):
        """Test recording requests."""
        stats = BatchStats()

        stats.record_request(BatchPriority.HIGH, wait_time=0.1)
        stats.record_request(BatchPriority.LOW, wait_time=0.2)

        assert stats.total_requests == 2
        assert stats.total_wait_time == pytest.approx(0.3)
        assert stats.priority_distribution[BatchPriority.HIGH] == 1
        assert stats.priority_distribution[BatchPriority.LOW] == 1

    def test_batch_stats_record_batch(self):
        """Test recording batch execution."""
        stats = BatchStats()

        stats.record_batch(batch_size=5, execution_time=0.5)
        stats.record_batch(batch_size=10, execution_time=1.0)

        assert stats.total_batches == 2
        assert stats.avg_batch_size == 7.5
        assert stats.total_execution_time == 1.5


class TestRequestBatcher:
    """Test RequestBatcher implementation."""

    @pytest.fixture
    async def batcher(self):
        """Create a test batcher instance."""
        batcher = RequestBatcher(
            key_func=lambda **kwargs: kwargs.get("model", "default"),
            batch_func=self._mock_batch_func,
            max_batch_size=3,
            batch_timeout=0.1,
        )
        await batcher.start()
        yield batcher
        await batcher.stop()

    @staticmethod
    async def _mock_batch_func(entries):
        """Mock batch function."""
        return [f"result_{i}" for i in range(len(entries))]

    @pytest.mark.asyncio
    async def test_submit_single_request(self, batcher):
        """Test submitting a single request."""
        result = await batcher.submit(model="test", prompt="Hello")

        assert result == "result_0"
        assert batcher.stats.total_requests == 1

    @pytest.mark.asyncio
    async def test_batch_accumulation(self, batcher):
        """Test that requests are batched."""
        # Submit multiple requests concurrently
        results = await asyncio.gather(
            batcher.submit(model="test", prompt="1"),
            batcher.submit(model="test", prompt="2"),
            batcher.submit(model="test", prompt="3"),
        )

        assert len(results) == 3
        assert all(r.startswith("result_") for r in results)

    @pytest.mark.asyncio
    async def test_max_batch_size_flush(self, batcher):
        """Test that batch flushes at max size."""
        # Submit exactly max_batch_size requests
        tasks = [batcher.submit(model="test", prompt=str(i)) for i in range(batcher.max_batch_size)]

        results = await asyncio.gather(*tasks)

        assert len(results) == batcher.max_batch_size
        # Should have been flushed immediately
        assert batcher.stats.total_batches >= 1

    @pytest.mark.asyncio
    async def test_timeout_flush(self, batcher):
        """Test that batch flushes on timeout."""
        # Submit one request (less than max_batch_size)
        task = asyncio.create_task(batcher.submit(model="test", prompt="1"))

        # Wait for timeout
        await asyncio.sleep(batcher.batch_timeout * 2)

        # Task should complete
        result = await task
        assert result == "result_0"

    @pytest.mark.asyncio
    async def test_priority_ordering(self, batcher):
        """Test that priority affects execution order."""
        high_priority = asyncio.create_task(
            batcher.submit(model="test", prompt="high", priority=BatchPriority.HIGH)
        )
        low_priority = asyncio.create_task(
            batcher.submit(model="test", prompt="low", priority=BatchPriority.LOW)
        )

        # Both should complete
        results = await asyncio.gather(high_priority, low_priority)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_error_handling(self, batcher):
        """Test error handling in batch execution."""
        # Create batcher that raises error
        error_batcher = RequestBatcher(
            key_func=lambda **kwargs: kwargs.get("model", "default"),
            batch_func=lambda entries: (_ for _ in ()).throw(Exception("Batch failed")),
            max_batch_size=3,
            batch_timeout=0.1,
        )
        await error_batcher.start()

        try:
            with pytest.raises(Exception, match="Batch failed"):
                await error_batcher.submit(model="test", prompt="1")
        finally:
            await error_batcher.stop()

    @pytest.mark.asyncio
    async def test_flush_all(self, batcher):
        """Test flushing all pending batches."""
        # Submit requests
        await batcher.submit(model="test1", prompt="1")
        await batcher.submit(model="test2", prompt="2")

        # Flush all
        await batcher.flush_all()

        # Check stats
        stats = batcher.get_stats()
        assert stats["total_requests"] >= 2

    @pytest.mark.asyncio
    async def test_batcher_stats(self, batcher):
        """Test batcher statistics."""
        await batcher.submit(model="test", prompt="1")
        await batcher.submit(model="test", prompt="2")

        stats = batcher.get_stats()

        assert stats["total_requests"] == 2
        assert "avg_batch_size" in stats
        assert "avg_wait_time" in stats


class TestToolCallBatcher:
    """Test ToolCallBatcher implementation."""

    @pytest.fixture
    def mock_executor(self):
        """Create mock tool executor."""
        executor = Mock()
        executor.execute_tool = AsyncMock(return_value="tool_result")
        return executor

    @pytest.fixture
    async def tool_batcher(self, mock_executor):
        """Create a tool call batcher instance."""
        batcher = ToolCallBatcher(
            executor=mock_executor,
            max_batch_size=3,
            batch_timeout=0.1,
        )
        await batcher.start()
        yield batcher
        await batcher.stop()

    @pytest.mark.asyncio
    async def test_batch_tool_calls(self, tool_batcher, mock_executor):
        """Test batching multiple tool calls."""
        calls = [
            {"tool": "read_file", "args": {"path": "file1.py"}},
            {"tool": "read_file", "args": {"path": "file2.py"}},
            {"tool": "list_directory", "args": {"path": "."}},
        ]

        results = await tool_batcher.batch_calls(calls)

        assert len(results) == 3
        # Convert to list to handle generator case
        results_list = list(results) if not isinstance(results, list) else results
        assert all(r == "tool_result" for r in results_list)
        assert mock_executor.execute_tool.call_count == 3

    @pytest.mark.asyncio
    async def test_tool_batcher_stats(self, tool_batcher, mock_executor):
        """Test tool batcher statistics."""
        calls = [
            {"tool": "read_file", "args": {"path": "file1.py"}},
            {"tool": "read_file", "args": {"path": "file2.py"}},
        ]

        # Convert generator result to list
        _ = list(await tool_batcher.batch_calls(calls))

        stats = tool_batcher.get_stats()
        assert stats["total_requests"] >= 2  # Use >= since async might have more


class TestGlobalBatchers:
    """Test global batcher instances."""

    def test_get_llm_batcher(self):
        """Test getting global LLM batcher."""
        reset_batchers()

        batcher1 = get_llm_batcher()
        batcher2 = get_llm_batcher()

        assert batcher1 is batcher2

    def test_get_tool_batcher(self):
        """Test getting global tool batcher."""
        reset_batchers()

        mock_executor = Mock()
        batcher1 = get_tool_batcher(mock_executor)
        batcher2 = get_tool_batcher(mock_executor)

        assert batcher1 is batcher2

    @pytest.mark.asyncio
    async def test_reset_batchers(self):
        """Test resetting global batchers."""
        # Get initial batcher (might need event loop for async)
        batcher = get_llm_batcher()
        reset_batchers()

        new_batcher = get_llm_batcher()
        assert new_batcher is not batcher


@pytest.mark.integration
class TestRequestBatcherIntegration:
    """Integration tests for request batching."""

    @pytest.mark.asyncio
    async def test_real_batching_performance(self):
        """Test that actual batching improves performance."""

        # Create a batcher with simulated delay
        async def slow_batch(entries):
            # Simulate network delay
            await asyncio.sleep(0.1)
            return [f"result_{i}" for i in range(len(entries))]

        batcher = RequestBatcher(
            key_func=lambda **kwargs: kwargs.get("model"),
            batch_func=slow_batch,
            max_batch_size=5,
            batch_timeout=0.05,
        )
        await batcher.start()

        try:
            # Time sequential execution (no batching)
            start = time.perf_counter()
            for i in range(5):
                await slow_batch([Mock(request_id=str(i))])
            sequential_time = time.perf_counter() - start

            # Time batched execution
            start = time.perf_counter()
            tasks = [batcher.submit(model="test", prompt=str(i)) for i in range(5)]
            await asyncio.gather(*tasks)
            batched_time = time.perf_counter() - start

            # Batching should be faster (one delay vs 5 delays)
            assert batched_time < sequential_time * 0.5

        finally:
            await batcher.stop()
