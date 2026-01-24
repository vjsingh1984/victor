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
    event = MessagingEvent(topic="test", data={})
    sync_wrapper.publish(event)  # Sync call, runs async internally
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING, Awaitable, Callable, cast

from victor.core.events.protocols import (
    EventHandler,
    MessagingEvent,
    IEventBackend,
    SubscriptionHandle,
)

if TYPE_CHECKING:
    from victor.core.events import ObservabilityBus

logger = logging.getLogger(__name__)

# Type alias for sync event handlers
SyncEventHandler = Callable[[MessagingEvent], None]


class SyncEventWrapper:
    """Synchronous wrapper around IEventBackend.

    This wrapper allows synchronous code to use the async event backend
    by running async operations in an event loop via asyncio.run().

    This is intended for gradual migration from sync to async APIs.
    New code should use async/await directly.

    Example:
        >>> from victor.core.events import create_event_backend, MessagingEvent
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
        >>> event = MessagingEvent(topic="tool.start", data={"tool": "read_file"})
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

    def _run_async_synchronous(self, coro: Awaitable[Any]) -> Any:
        """Run an async coroutine in a synchronous context.

        This method handles both cases:
        1. Called from sync context - uses asyncio.run()
        2. Called from async context - runs in a separate thread with its own event loop

        Args:
            coro: Awaitable coroutine to execute

        Returns:
            Result of the coroutine

        Raises:
            TimeoutError: If operation takes longer than 5 seconds
            Exception: If coroutine raises an exception
        """
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
            # If we get here, we're in an async context
            # This shouldn't happen for a sync wrapper - warn and run in thread pool
            logger.warning(
                "[SyncEventWrapper] Method called from async context. "
                "This indicates incorrect usage. Use async backend directly."
            )
            # Run in a new thread with its own event loop to avoid blocking
            import threading

            result = [None]
            exception: list[Exception | None] = [None]

            def run_in_thread() -> None:
                try:
                    result[0] = asyncio.run(cast(Any, coro))
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join(timeout=5.0)  # 5 second timeout

            if exception[0]:
                raise exception[0]
            if thread.is_alive():
                raise TimeoutError("Async operation timed out")
            return result[0]
        except RuntimeError:
            # No running loop, safe to use asyncio.run()
            return asyncio.run(cast(Any, coro))

    def publish(self, event: MessagingEvent) -> bool:
        """Publish an event synchronously.

        This method runs the async publish operation in a new event loop.

        Args:
            event: MessagingEvent to publish

        Returns:
            True if event was published successfully

        Example:
            >>> wrapper = SyncEventWrapper(backend)
            >>> event = MessagingEvent(topic="test", data={})
            >>> wrapper.publish(event)
            True
        """
        return cast(bool, self._run_async_synchronous(self._backend.publish(event)))

    def publish_batch(self, events: list[MessagingEvent]) -> int:
        """Publish multiple events synchronously.

        Args:
            events: List of events to publish

        Returns:
            Number of events successfully published
        """
        return cast(int, self._run_async_synchronous(self._backend.publish_batch(events)))

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

        async def async_wrapper(event: MessagingEvent) -> None:
            """Wrap sync handler for async backend."""
            try:
                handler(event)
            except Exception as e:
                logger.error(f"[SyncEventWrapper] Handler error for {event.topic}: {e}")

        return cast(SubscriptionHandle, self._run_async_synchronous(self._backend.subscribe(pattern, async_wrapper)))

    def unsubscribe(self, handle: SubscriptionHandle) -> bool:
        """Unsubscribe from events synchronously.

        Args:
            handle: Subscription handle from subscribe()

        Returns:
            True if unsubscribed successfully
        """
        return cast(bool, self._run_async_synchronous(self._backend.unsubscribe(handle)))

    def health_check(self) -> bool:
        """Check backend health synchronously.

        Returns:
            True if backend is healthy
        """
        return cast(bool, self._run_async_synchronous(self._backend.health_check()))


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
        data: dict[str, Any],
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
        event = MessagingEvent(
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
