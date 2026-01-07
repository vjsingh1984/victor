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

"""Synchronous wrapper for async event backend.

This module provides synchronous wrappers around the async IEventBackend,
allowing gradual migration from sync to async APIs.

Usage:
    from victor.core.events.sync_wrapper import SyncEventWrapper
    from victor.core.events import create_event_backend

    # Create async backend
    backend = create_event_backend()
    await backend.connect()

    # Wrap for sync usage
    sync_wrapper = SyncEventWrapper(backend)

    # Use sync API
    event = Event(topic="test", data={})
    sync_wrapper.publish(event)  # Sync call, runs async internally
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable

from victor.core.events.protocols import (
    EventHandler,
    Event,
    IEventBackend,
    SubscriptionHandle,
)

if TYPE_CHECKING:
    from victor.core.events import ObservabilityBus

logger = logging.getLogger(__name__)

# Type alias for sync event handlers
SyncEventHandler = Callable[[Event], None]


class SyncEventWrapper:
    """Synchronous wrapper around IEventBackend.

    This wrapper allows synchronous code to use the async event backend
    by running async operations in an event loop via asyncio.run().

    This is intended for gradual migration from sync to async APIs.
    New code should use async/await directly.

    Example:
        >>> from victor.core.events import create_event_backend, Event
        >>> from victor.core.events.sync_wrapper import SyncEventWrapper
        >>>
        >>> # Create async backend
        >>> backend = create_event_backend()
        >>> await backend.connect()
        >>>
        >>> # Wrap for sync usage
        >>> sync_wrapper = SyncEventWrapper(backend)
        >>>
        >>> # Use sync API
        >>> event = Event(topic="tool.start", data={"tool": "read_file"})
        >>> sync_wrapper.publish(event)
        True

    Thread Safety:
        - Each call creates a new event loop via asyncio.run()
        - Not safe to share SyncEventWrapper across threads
        - For thread safety, use backend directly with proper async/await
    """

    def __init__(self, backend: IEventBackend) -> None:
        """Initialize the sync wrapper.

        Args:
            backend: Async event backend to wrap
        """
        self._backend = backend
        logger.debug(f"[SyncEventWrapper] Created wrapper for {type(backend).__name__}")

    def publish(self, event: Event) -> bool:
        """Publish an event synchronously.

        This method runs the async publish operation in a new event loop.

        Args:
            event: Event to publish

        Returns:
            True if event was published successfully

        Example:
            >>> wrapper = SyncEventWrapper(backend)
            >>> event = Event(topic="test", data={})
            >>> wrapper.publish(event)
            True
        """
        try:
            return asyncio.run(self._backend.publish(event))
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                # Already in async context, get running loop
                loop = asyncio.get_event_loop()
                task = loop.create_task(self._backend.publish(event))
                return asyncio.run(task)  # Wait for task completion
            raise

    def publish_batch(self, events: list[Event]) -> int:
        """Publish multiple events synchronously.

        Args:
            events: List of events to publish

        Returns:
            Number of events successfully published
        """
        try:
            return asyncio.run(self._backend.publish_batch(events))
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                loop = asyncio.get_event_loop()
                task = loop.create_task(self._backend.publish_batch(events))
                return asyncio.run(task)
            raise

    def subscribe(
        self,
        pattern: str,
        handler: SyncEventHandler,
    ) -> SubscriptionHandle:
        """Subscribe to events synchronously.

        Note: The handler will be called from async context. If the handler
        is CPU-intensive or blocking, consider running it in a thread pool.

        Args:
            pattern: Topic pattern to subscribe to (e.g., "tool.*")
            handler: Synchronous event handler

        Returns:
            Subscription handle for unsubscription

        Example:
            >>> def handler(event):
            ...     print(f"Received: {event.topic}")
            >>>
            >>> wrapper = SyncEventWrapper(backend)
            >>> handle = wrapper.subscribe("tool.*", handler)
            >>>
            >>> # Later: unsubscribe
            >>> handle.unsubscribe()
        """

        async def async_wrapper(event: Event) -> None:
            """Wrap sync handler for async backend."""
            try:
                handler(event)
            except Exception as e:
                logger.error(f"[SyncEventWrapper] Handler error for {event.topic}: {e}")

        try:
            return asyncio.run(self._backend.subscribe(pattern, async_wrapper))
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                loop = asyncio.get_event_loop()
                task = loop.create_task(self._backend.subscribe(pattern, async_wrapper))
                return asyncio.run(task)
            raise

    def unsubscribe(self, handle: SubscriptionHandle) -> bool:
        """Unsubscribe from events synchronously.

        Args:
            handle: Subscription handle from subscribe()

        Returns:
            True if unsubscribed successfully
        """
        try:
            return asyncio.run(self._backend.unsubscribe(handle))
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                loop = asyncio.get_event_loop()
                task = loop.create_task(self._backend.unsubscribe(handle))
                return asyncio.run(task)
            raise

    def health_check(self) -> bool:
        """Check backend health synchronously.

        Returns:
            True if backend is healthy
        """
        try:
            return asyncio.run(self._backend.health_check())
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                loop = asyncio.get_event_loop()
                task = loop.create_task(self._backend.health_check())
                return asyncio.run(task)
            raise


class SyncObservabilityBus:
    """Synchronous wrapper for ObservabilityBus.

    This provides a drop-in replacement for the legacy EventBus with
    the same sync API, while internally using the canonical async system.

    Example:
        >>> from victor.core.events.sync_wrapper import SyncObservabilityBus
        >>>
        >>> bus = SyncObservabilityBus()
        >>>
        >>> # Sync API like legacy EventBus
        >>> bus.emit("tool.start", {"tool": "read_file"})
        >>>
        >>> # Subscribe
        >>> bus.subscribe("tool.*", lambda e: print(e.topic))
    """

    def __init__(self, backend: IEventBackend | None = None) -> None:
        """Initialize the sync observability bus.

        Args:
            backend: Optional pre-configured async backend
        """
        from victor.core.events.backends import ObservabilityBus

        if backend:
            self._async_bus = ObservabilityBus(backend=backend)
        else:
            self._async_bus = ObservabilityBus()

        # Create sync wrapper
        self._sync_wrapper = SyncEventWrapper(self._async_bus.backend)

        logger.debug("[SyncObservabilityBus] Created sync wrapper")

    def emit(
        self,
        topic: str,
        data: dict,
        *,
        source: str = "victor",
        correlation_id: str | None = None,
    ) -> bool:
        """Emit an observability event synchronously.

        Args:
            topic: Event topic (e.g., "tool.start", "metric.latency")
            data: Event payload
            source: Event source identifier
            correlation_id: Optional correlation ID

        Returns:
            True if event was emitted

        Example:
            >>> bus = SyncObservabilityBus()
            >>> bus.emit("tool.start", {"tool": "read_file", "path": "test.txt"})
            True
        """
        event = Event(
            topic=topic,
            data=data,
            source=source,
            correlation_id=correlation_id,
        )
        return self._sync_wrapper.publish(event)

    def subscribe(self, pattern: str, handler: SyncEventHandler) -> SubscriptionHandle:
        """Subscribe to observability events synchronously.

        Args:
            pattern: Topic pattern (e.g., "tool.*", "metric.*")
            handler: Synchronous event handler

        Returns:
            Subscription handle

        Example:
            >>> def handler(event):
            ...     print(f"{event.topic}: {event.data}")
            >>>
            >>> bus = SyncObservabilityBus()
            >>> handle = bus.subscribe("tool.*", handler)
        """
        return self._sync_wrapper.subscribe(pattern, handler)

    @property
    def backend(self) -> IEventBackend:
        """Get the underlying async backend."""
        return self._async_bus.backend

    @property
    def async_bus(self) -> "ObservabilityBus":
        """Get the underlying async ObservabilityBus."""
        return self._async_bus
