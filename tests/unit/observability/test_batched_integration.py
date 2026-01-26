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

"""Tests for batched observability integration (Phase 2.1).

Tests that BatchedObservabilityIntegration:
1. Batches events by size
2. Batches events by time
3. Uses hybrid strategy correctly
4. Flushes pending events on shutdown
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from victor.observability.batching_integration import (
    BatchConfig,
    BatchedObservabilityIntegration,
    BatchStats,
    BatchStrategy,
)


class TestBatchedObservabilityIntegration:
    """Test BatchedObservabilityIntegration class."""

    @pytest.fixture
    def emitted_batches(self) -> List[List[Dict[str, Any]]]:
        """Fixture to track emitted batches."""
        return []

    @pytest.fixture
    def mock_emitter(self, emitted_batches):
        """Create a mock emitter that tracks batches."""

        async def emitter(events: List[Dict[str, Any]]) -> None:
            emitted_batches.append(events)

        return emitter

    @pytest.mark.asyncio
    async def test_events_are_batched_by_size(self, mock_emitter, emitted_batches):
        """Events should be batched when batch size is reached."""
        config = BatchConfig(
            strategy=BatchStrategy.SIZE,
            max_batch_size=3,
            enabled=True,
        )
        integration = BatchedObservabilityIntegration(config)
        integration.set_emitter(mock_emitter)

        # Emit 2 events (should not flush yet)
        await integration.emit_event({"type": "event1", "data": "test1"})
        await integration.emit_event({"type": "event2", "data": "test2"})

        # No batches flushed yet (batch size not reached)
        assert len(emitted_batches) == 0

        # Emit 3rd event (should trigger flush)
        await integration.emit_event({"type": "event3", "data": "test3"})

        # Batch should have been flushed
        assert len(emitted_batches) == 1
        assert len(emitted_batches[0]) == 3

    @pytest.mark.asyncio
    async def test_events_are_batched_by_time(self, mock_emitter, emitted_batches):
        """Events should be batched when max wait time is reached."""
        config = BatchConfig(
            strategy=BatchStrategy.TIME,
            max_wait_time=0.1,  # 100ms
            enabled=True,
        )
        integration = BatchedObservabilityIntegration(config)
        integration.set_emitter(mock_emitter)

        # Start the flush loop
        await integration.start()

        try:
            # Emit event
            await integration.emit_event({"type": "event1", "data": "test1"})

            # Wait for time-based flush
            await asyncio.sleep(0.2)

            # Should have flushed by time
            assert len(emitted_batches) >= 1
            assert any(len(batch) > 0 for batch in emitted_batches)

        finally:
            await integration.stop()

    @pytest.mark.asyncio
    async def test_hybrid_strategy_uses_both(self, mock_emitter, emitted_batches):
        """Hybrid strategy should use both size and time triggers."""
        config = BatchConfig(
            strategy=BatchStrategy.HYBRID,
            max_batch_size=5,
            max_wait_time=0.5,
            enabled=True,
        )
        integration = BatchedObservabilityIntegration(config)
        integration.set_emitter(mock_emitter)

        # Emit 5 events (should trigger size-based flush)
        for i in range(5):
            await integration.emit_event({"type": f"event{i}", "data": f"test{i}"})

        # Should have flushed by size
        assert len(emitted_batches) == 1
        assert len(emitted_batches[0]) == 5

    @pytest.mark.asyncio
    async def test_graceful_shutdown_flushes_pending(self, mock_emitter, emitted_batches):
        """Graceful shutdown should flush pending events."""
        config = BatchConfig(
            strategy=BatchStrategy.SIZE,
            max_batch_size=10,  # Large batch size
            flush_on_shutdown=True,
            enabled=True,
        )
        integration = BatchedObservabilityIntegration(config)
        integration.set_emitter(mock_emitter)

        # Emit events (not enough to trigger size-based flush)
        await integration.emit_event({"type": "event1", "data": "test1"})
        await integration.emit_event({"type": "event2", "data": "test2"})

        # No flush yet
        assert len(emitted_batches) == 0

        # Shutdown should flush pending
        await integration.shutdown()

        # All pending events should be flushed
        assert len(emitted_batches) == 1
        assert len(emitted_batches[0]) == 2


class TestBatchStats:
    """Test BatchStats class."""

    def test_record_flush_updates_stats(self):
        """record_flush should update statistics."""
        stats = BatchStats()

        # First flush
        stats.record_flush(batch_size=10, wait_time_ms=50.0)
        assert stats.batches_flushed == 1
        assert stats.events_flushed == 10
        assert stats.avg_batch_size == 10.0
        assert stats.avg_wait_time_ms == 50.0

        # Second flush
        stats.record_flush(batch_size=20, wait_time_ms=100.0)
        assert stats.batches_flushed == 2
        assert stats.events_flushed == 30
        assert stats.avg_batch_size == 15.0  # (10 + 20) / 2
        assert stats.avg_wait_time_ms == 75.0  # (50 + 100) / 2

    def test_get_summary(self):
        """get_summary should return all stats."""
        stats = BatchStats()
        stats.events_queued = 100
        stats.events_flushed = 80
        stats.events_dropped = 20
        stats.batches_flushed = 8

        summary = stats.get_summary()

        assert summary["events_queued"] == 100
        assert summary["events_flushed"] == 80
        assert summary["events_dropped"] == 20
        assert summary["batches_flushed"] == 8


class TestBatchConfig:
    """Test BatchConfig class."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        config = BatchConfig()

        assert config.strategy == BatchStrategy.HYBRID
        assert config.max_batch_size == 100
        assert config.max_wait_time == 0.5
        assert config.enabled is True
        assert config.flush_on_shutdown is True

    def test_custom_config(self):
        """Custom config should override defaults."""
        config = BatchConfig(
            strategy=BatchStrategy.SIZE,
            max_batch_size=50,
            max_wait_time=1.0,
            enabled=False,
        )

        assert config.strategy == BatchStrategy.SIZE
        assert config.max_batch_size == 50
        assert config.max_wait_time == 1.0
        assert config.enabled is False


class TestImmediateStrategy:
    """Test immediate (non-batched) strategy."""

    @pytest.fixture
    def emitted_batches(self) -> List[List[Dict[str, Any]]]:
        """Fixture to track emitted batches."""
        return []

    @pytest.fixture
    def mock_emitter(self, emitted_batches):
        """Create a mock emitter that tracks batches."""

        async def emitter(events: List[Dict[str, Any]]) -> None:
            emitted_batches.append(events)

        return emitter

    @pytest.mark.asyncio
    async def test_immediate_strategy_emits_immediately(self, mock_emitter, emitted_batches):
        """Immediate strategy should emit each event immediately."""
        config = BatchConfig(
            strategy=BatchStrategy.IMMEDIATE,
            enabled=True,
        )
        integration = BatchedObservabilityIntegration(config)
        integration.set_emitter(mock_emitter)

        # Each event should be emitted immediately
        await integration.emit_event({"type": "event1"})
        assert len(emitted_batches) == 1
        assert len(emitted_batches[0]) == 1

        await integration.emit_event({"type": "event2"})
        assert len(emitted_batches) == 2
        assert len(emitted_batches[1]) == 1
