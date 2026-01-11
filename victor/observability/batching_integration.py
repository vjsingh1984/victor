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

"""Batched observability integration for async event emission.

This module provides efficient batching for observability events to reduce
I/O overhead and improve performance. Events are collected in batches and
flushed based on size, time, or explicit triggers.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


class BatchStrategy(str, Enum):
    """Batching strategies for event emission."""
    TIME = "time"
    SIZE = "size"
    HYBRID = "hybrid"
    IMMEDIATE = "immediate"
    MANUAL = "manual"


@dataclass
class BatchConfig:
    """Configuration for batched observability."""
    strategy: BatchStrategy = BatchStrategy.HYBRID
    max_batch_size: int = 100
    max_wait_time: float = 0.5  # 500ms
    enabled: bool = True
    flush_on_shutdown: bool = True
    max_queue_size: int = 0  # 0 = unlimited
    drop_handler: Optional[Callable[[Dict[str, Any]], None]] = None


@dataclass
class BatchStats:
    """Statistics for batched observability."""
    events_queued: int = 0
    events_flushed: int = 0
    events_dropped: int = 0
    batches_flushed: int = 0
    avg_batch_size: float = 0.0
    avg_wait_time_ms: float = 0.0
    current_queue_size: int = 0
    last_flush_time: float = 0.0

    def record_flush(self, batch_size: int, wait_time_ms: float) -> None:
        """Record a flush operation."""
        self.batches_flushed += 1
        self.events_flushed += batch_size
        self.avg_batch_size = (
            (self.avg_batch_size * (self.batches_flushed - 1) + batch_size) / self.batches_flushed
        )
        self.avg_wait_time_ms = (
            (self.avg_wait_time_ms * (self.batches_flushed - 1) + wait_time_ms) / self.batches_flushed
        )
        self.last_flush_time = time.time()
        self.current_queue_size = 0

    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary."""
        return {
            "events_queued": self.events_queued,
            "events_flushed": self.events_flushed,
            "events_dropped": self.events_dropped,
            "batches_flushed": self.batches_flushed,
            "avg_batch_size": self.avg_batch_size,
            "avg_wait_time_ms": self.avg_wait_time_ms,
            "current_queue_size": self.current_queue_size,
            "last_flush_time": self.last_flush_time,
        }


class BatchedObservabilityIntegration:
    """Observability integration with async batching.

    This integration collects observability events in batches and flushes
    them based on the configured strategy.
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        """Initialize batched observability integration."""
        self._config = config or BatchConfig()
        self._queue: Deque[Dict[str, Any]] = deque()
        self._event_times: Deque[float] = deque()
        self._stats = BatchStats()
        self._emitter: Optional[Callable[[List[Dict[str, Any]]], Awaitable[None]]] = None
        self._flush_task: Optional[asyncio.Task] = None
        self._flush_lock = asyncio.Lock()
        self._shutdown = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue_lock = threading.Lock()
        self._flush_event = asyncio.Event()

    def set_emitter(
        self,
        emitter: Callable[[List[Dict[str, Any]]], Awaitable[None]]
    ) -> None:
        """Set the event emitter backend."""
        self._emitter = emitter

    async def emit_event(self, event: Dict[str, Any]) -> None:
        """Queue an event for batched emission."""
        if self._shutdown:
            logger.warning("Cannot emit event: integration is shut down")
            return

        if not self._config.enabled or self._config.strategy == BatchStrategy.IMMEDIATE:
            await self._emit_batch([event])
            return

        with self._queue_lock:
            if (
                self._config.max_queue_size > 0
                and len(self._queue) >= self._config.max_queue_size
            ):
                self._stats.events_dropped += 1
                if self._config.drop_handler:
                    try:
                        self._config.drop_handler(event)
                    except Exception as e:
                        logger.error(f"Drop handler failed: {e}")
                logger.warning("Event queue full, dropping event")
                return

            self._queue.append(event)
            self._event_times.append(time.time())
            self._stats.events_queued += 1
            self._stats.current_queue_size = len(self._queue)

        await self._check_and_flush()

    async def _check_and_flush(self) -> None:
        """Check if we should flush based on strategy and flush if needed."""
        with self._queue_lock:
            queue_size = len(self._queue)

        if queue_size == 0:
            return

        should_flush = False

        if self._config.strategy == BatchStrategy.SIZE:
            should_flush = queue_size >= self._config.max_batch_size
        elif self._config.strategy == BatchStrategy.HYBRID:
            should_flush = (
                queue_size >= self._config.max_batch_size
                or self._should_flush_by_time()
            )

        if should_flush:
            await self.flush()

    def _should_flush_by_time(self) -> bool:
        """Check if we should flush based on time."""
        with self._queue_lock:
            if not self._event_times:
                return False
            oldest_time = self._event_times[0]
            return (time.time() - oldest_time) >= self._config.max_wait_time

    async def flush(self) -> int:
        """Flush all pending events immediately."""
        async with self._flush_lock:
            with self._queue_lock:
                if not self._queue:
                    return 0

                events = list(self._queue)
                event_times = list(self._event_times)

                self._queue.clear()
                self._event_times.clear()
                self._stats.current_queue_size = 0

            if event_times:
                oldest_time = event_times[0]
                wait_time_ms = (time.time() - oldest_time) * 1000
            else:
                wait_time_ms = 0.0

            try:
                await self._emit_batch(events)
                self._stats.record_flush(len(events), wait_time_ms)
                return len(events)
            except Exception as e:
                logger.error(f"Failed to flush batch: {e}")
                with self._queue_lock:
                    self._queue.extendleft(reversed(events))
                    self._event_times.extendleft(reversed(event_times))
                raise

    async def _emit_batch(self, events: List[Dict[str, Any]]) -> None:
        """Emit a batch of events."""
        if not self._emitter:
            for event in events:
                logger.debug(f"Observability event: {event}")
        else:
            await self._emitter(events)

    async def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Start the background flush task."""
        if self._flush_task is not None:
            logger.warning("Flush task already started")
            return

        self._loop = loop or asyncio.get_event_loop()

        if self._config.strategy in (BatchStrategy.TIME, BatchStrategy.HYBRID):
            self._flush_task = asyncio.create_task(
                self._flush_loop(), name="batched_observability_flush"
            )
            logger.debug("Started batched observability flush task")

    async def _flush_loop(self) -> None:
        """Background flush loop for time-based flushing."""
        while not self._shutdown:
            try:
                timeout = self._config.max_wait_time
                await asyncio.wait_for(self._flush_event.wait(), timeout=timeout)
                self._flush_event.clear()
            except asyncio.TimeoutError:
                pass

            if self._should_flush_by_time():
                try:
                    await self.flush()
                except Exception as e:
                    logger.error(f"Flush loop error: {e}")

    async def stop(self) -> None:
        """Stop the background flush task."""
        self._shutdown = True

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

    async def shutdown(self) -> int:
        """Shutdown the integration and flush pending events."""
        self._shutdown = True

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        flushed = 0
        if self._config.flush_on_shutdown:
            flushed = await self.flush()

        return flushed

    def get_stats(self) -> BatchStats:
        """Get current statistics."""
        with self._queue_lock:
            self._stats.current_queue_size = len(self._queue)
        return self._stats

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = BatchStats()


__all__ = [
    "BatchedObservabilityIntegration",
    "BatchConfig",
    "BatchStrategy",
    "BatchStats",
]
