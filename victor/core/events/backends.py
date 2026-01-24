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

"""Event backend implementations.

This module provides concrete implementations of IEventBackend:
- InMemoryEventBackend: Default in-process implementation
- (Future) KafkaEventBackend, SQSEventBackend, RabbitMQEventBackend, RedisEventBackend

The factory function create_event_backend() creates the appropriate backend
based on configuration.

Example:
    from victor.core.events.backends import (
        create_event_backend,
        InMemoryEventBackend,
        BackendConfig,
        BackendType,
    )

    # Use factory (recommended)
    backend = create_event_backend()
    await backend.connect()
    await backend.publish(MessagingEvent(topic="test", data={}))

    # Or create directly
    backend = InMemoryEventBackend()
    await backend.connect()
"""

from __future__ import annotations

import asyncio
import logging
import queue
import random
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from victor.core.events.protocols import (
    BackendConfig,
    BackendType,
    DeliveryGuarantee,
    MessagingEvent,
    EventHandler,
    EventPublishError,
    EventSubscriptionError,
    IEventBackend,
    SubscriptionHandle,
)

logger = logging.getLogger(__name__)


# =============================================================================
# In-Memory Event Backend
# =============================================================================


@dataclass
class _Subscription:
    """Internal subscription tracking."""

    id: str
    pattern: str
    handler: EventHandler
    is_active: bool = True


class InMemoryEventBackend:
    """In-memory event backend for single-instance deployments.

    This is the default backend that provides fast, in-process pub/sub.
    It's suitable for:
    - Single-instance deployments
    - Development and testing
    - Low-latency observability
    - Multi-threaded environments (thread-safe)

    Limitations:
    - Events are lost on process restart
    - Cannot scale horizontally
    - Only AT_MOST_ONCE delivery (no persistence)

    Thread Safety:
    - Thread-safe implementation using queue.Queue and threading.Lock
    - Supports publishing from any thread
    - Supports subscribing from any thread
    - Handlers run in the dispatcher's event loop (non-blocking)
    - Safe for cross-event-loop usage

    Example:
        backend = InMemoryEventBackend()
        await backend.connect()

        async def handler(event):
            print(f"Received: {event.topic}")

        handle = await backend.subscribe("tool.*", handler)
        await backend.publish(MessagingEvent(topic="tool.call", data={"name": "read"}))

        await backend.disconnect()
    """

    def __init__(
        self,
        config: Optional[BackendConfig] = None,
        *,
        queue_maxsize: int = 10000,
        critical_reserve: int = 0,
        auto_start_dispatcher: bool = True,
        sampling_rate: float = 1.0,
        sampling_whitelist: Optional[Set[str]] = None,
    ) -> None:
        """Initialize the in-memory backend.

        Args:
            config: Optional backend configuration
            queue_maxsize: Maximum queue size (0 for unbounded)
            critical_reserve: Reserved slots for critical events (future feature)
            auto_start_dispatcher: If False, don't start dispatcher on connect (for testing)
            sampling_rate: Probability of accepting events (0.0-1.0, default 1.0 = all)
            sampling_whitelist: Event topics that are never sampled (always accepted)
        """
        self._config = config or BackendConfig()
        self._queue_maxsize = queue_maxsize
        self._critical_reserve = critical_reserve
        self._subscriptions: Dict[str, _Subscription] = {}
        # Use thread-safe queue.Queue for cross-thread publishing
        self._event_queue: queue.Queue[MessagingEvent] = queue.Queue(maxsize=queue_maxsize)
        self._is_connected = False
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._auto_start_dispatcher = auto_start_dispatcher
        # Use threading.Lock for thread-safe subscription management
        self._lock: threading.Lock = threading.Lock()
        self._pending_tasks: Set[asyncio.Task] = set()
        self._dropped_event_count = 0
        # AT_LEAST_ONCE delivery tracking
        self._pending_events: Dict[str, MessagingEvent] = {}  # event_id -> event
        self._delivery_guarantee = self._config.delivery_guarantee

        # Event sampling (Phase 4: Observability Backpressure)
        self._sampling_rate = max(0.0, min(1.0, sampling_rate))
        self._sampling_whitelist = sampling_whitelist or set()
        self._sampled_event_count = 0
        self._total_event_count = 0

    @property
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        return BackendType.IN_MEMORY

    @property
    def is_connected(self) -> bool:
        """Check if backend is connected and ready."""
        return self._is_connected

    async def connect(self) -> None:
        """Connect and start the event dispatcher.

        The dispatcher runs in the background, delivering events to handlers.
        Can be disabled with auto_start_dispatcher=False for testing passive queue behavior.
        """
        if self._is_connected:
            return

        self._is_connected = True
        if self._auto_start_dispatcher:
            self._dispatcher_task = asyncio.create_task(self._dispatch_loop())
        logger.debug("InMemoryEventBackend connected")

    async def disconnect(self) -> None:
        """Disconnect and stop the dispatcher.

        Waits for pending events to be delivered (with timeout).
        """
        if not self._is_connected:
            return

        self._is_connected = False

        # Cancel dispatcher
        if self._dispatcher_task and not self._dispatcher_task.done():
            self._dispatcher_task.cancel()
            try:
                await asyncio.wait_for(self._dispatcher_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Cancel pending handler tasks
        for task in list(self._pending_tasks):
            if not task.done():
                task.cancel()

        self._pending_tasks.clear()
        logger.debug("InMemoryEventBackend disconnected")

    async def health_check(self) -> bool:
        """Check backend health.

        Returns:
            True if connected and dispatcher is running
        """
        return (
            self._is_connected
            and self._dispatcher_task is not None
            and not self._dispatcher_task.done()
        )

    async def publish(self, event: MessagingEvent) -> bool:
        """Publish an event to all matching subscribers.

        Thread-safe: Can be called from any thread.

        Args:
            event: Event to publish

        Returns:
            True if event was queued for delivery

        Raises:
            EventPublishError: If queue is full and cannot accept event
        """
        if not self._is_connected:
            raise EventPublishError(event, "Backend not connected", retryable=True)

        # Phase 4: Event sampling (observability backpressure)
        # Apply sampling to reduce event volume under load
        self._total_event_count += 1

        # Check if event is whitelisted (never sampled)
        is_whitelisted = any(
            event.topic.startswith(pattern) for pattern in self._sampling_whitelist
        )

        # Apply sampling rate (unless whitelisted or sampling is disabled)
        if not is_whitelisted and self._sampling_rate < 1.0:
            if random.random() > self._sampling_rate:
                # Event sampled out (dropped)
                self._sampled_event_count += 1
                logger.debug(
                    f"Event sampled out: {event.topic} "
                    f"(sampling_rate={self._sampling_rate:.2%}, "
                    f"total={self._total_event_count}, "
                    f"sampled={self._sampled_event_count})"
                )
                return True  # Return True to indicate no error (event was processed by sampling)

        try:
            # Non-blocking put (thread-safe)
            self._event_queue.put_nowait(event)
            return True
        except queue.Full:
            # Queue full - drop event (AT_MOST_ONCE semantics)
            self._dropped_event_count += 1
            logger.warning(
                f"Event queue full, dropping event: {event.topic} "
                f"(queue_size={self._queue_maxsize}, "
                f"dropped={self._dropped_event_count}, "
                f"sampled={self._sampled_event_count})"
            )
            return False

    async def publish_batch(self, events: List[MessagingEvent]) -> int:
        """Publish multiple events.

        Args:
            events: List of events to publish

        Returns:
            Number of events successfully queued
        """
        success_count = 0
        for event in events:
            try:
                if await self.publish(event):
                    success_count += 1
            except EventPublishError:
                continue
        return success_count

    async def subscribe(
        self,
        pattern: str,
        handler: EventHandler,
    ) -> SubscriptionHandle:
        """Subscribe to events matching a pattern.

        Thread-safe: Can be called from any thread.

        Args:
            pattern: Topic pattern (e.g., "tool.*", "agent.message")
            handler: Async callback for events

        Returns:
            Handle for managing the subscription
        """
        subscription_id = uuid.uuid4().hex[:12]

        subscription = _Subscription(
            id=subscription_id,
            pattern=pattern,
            handler=handler,
            is_active=True,
        )

        # Use threading.Lock for thread-safe subscription management
        with self._lock:
            self._subscriptions[subscription_id] = subscription

        # Create handle with unsubscribe capability
        handle = SubscriptionHandle(
            subscription_id=subscription_id,
            pattern=pattern,
            is_active=True,
        )

        # Bind unsubscribe method

        async def bound_unsubscribe() -> None:
            await self.unsubscribe(handle)

        handle.unsubscribe = bound_unsubscribe

        logger.debug(f"Subscribed to pattern '{pattern}' (id: {subscription_id})")
        return handle

    async def unsubscribe(self, handle: SubscriptionHandle) -> bool:
        """Unsubscribe using a handle.

        Thread-safe: Can be called from any thread.

        Args:
            handle: Subscription handle from subscribe()

        Returns:
            True if unsubscribed successfully
        """
        # Use threading.Lock for thread-safe subscription management
        with self._lock:
            subscription = self._subscriptions.get(handle.subscription_id)
            if subscription:
                subscription.is_active = False
                del self._subscriptions[handle.subscription_id]
                handle.is_active = False
                logger.debug(f"Unsubscribed from pattern '{handle.pattern}'")
                return True
        return False

    async def _dispatch_loop(self) -> None:
        """Background loop that dispatches events to handlers.

        Thread-safe: Reads from thread-safe queue.Queue and dispatches
        to subscriptions in the event loop's context.

        For AT_LEAST_ONCE delivery:
        - Tracks pending events until ACK'd
        - Re-queues events on NACK with requeue=True
        - Removes events from pending after successful ACK
        """
        import concurrent.futures

        while self._is_connected:
            try:
                # Wait for event with timeout to allow checking is_connected
                # Use run_in_executor to run blocking queue.get() in a thread pool
                loop = asyncio.get_event_loop()
                try:
                    # Use lambda to properly pass arguments to queue.get
                    event = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, lambda: self._event_queue.get(block=True, timeout=0.1)
                        ),
                        timeout=0.2,
                    )
                except (asyncio.TimeoutError, concurrent.futures.TimeoutError):
                    continue
                except queue.Empty:
                    continue

                # Check if this event needs AT_LEAST_ONCE tracking
                needs_ack = (
                    event.delivery_guarantee == DeliveryGuarantee.AT_LEAST_ONCE
                    and not event.is_acknowledged()
                )

                if needs_ack:
                    # Add to pending events for tracking
                    self._pending_events[event.id] = event
                    event.increment_delivery_count()

                # Get matching subscriptions (thread-safe snapshot)
                with self._lock:
                    subscriptions = list(self._subscriptions.values())
                    matching = [
                        s for s in subscriptions if s.is_active and event.matches_pattern(s.pattern)
                    ]

                # Dispatch to handlers and wait for completion
                handler_tasks = []
                for subscription in matching:
                    try:
                        task = asyncio.create_task(
                            self._safe_call_handler(subscription.handler, event)
                        )
                        self._pending_tasks.add(task)
                        handler_tasks.append(task)
                        task.add_done_callback(self._pending_tasks.discard)
                    except Exception:
                        # Silently skip all errors - event loop issues, closed loops, etc.
                        pass

                # Wait for all handlers to complete
                if handler_tasks:
                    results = await asyncio.gather(*handler_tasks, return_exceptions=True)

                # Handle ACK/NACK for AT_LEAST_ONCE delivery
                if needs_ack:
                    if event.is_acknowledged():
                        # Explicitly ACK'd or NACK'd without requeue
                        # Successfully processed, remove from pending
                        self._pending_events.pop(event.id, None)
                    elif event._nack_requeue:
                        # Explicit NACK with requeue=True - don't auto-ACK
                        if event.should_retry():
                            # Reset flag before re-queuing for next delivery
                            event.reset_for_redelivery()
                            if event.id in self._pending_events:
                                del self._pending_events[event.id]
                            try:
                                self._event_queue.put_nowait(event)
                            except queue.Full:
                                # Queue full - drop and count as lost
                                self._dropped_event_count += 1
                                logger.warning(
                                    f"AT_LEAST_ONCE event {event.id} dropped on retry - queue full"
                                )
                        else:
                            # Max retries reached
                            self._pending_events.pop(event.id, None)
                            logger.error(
                                f"AT_LEAST_ONCE event {event.id} failed after "
                                f"{event._delivery_count} delivery attempts"
                            )
                    elif not any(isinstance(r, Exception) for r in results if r is not None):
                        # No exceptions in handlers and not explicitly NACK'd
                        # Treat successful completion as implicit ACK
                        # This maintains backward compatibility with handlers that don't call ack()
                        await event.ack()
                        self._pending_events.pop(event.id, None)
                    else:
                        # Handler raised exception - treat as NACK with requeue
                        if event.should_retry():
                            event.increment_delivery_count()
                            if event.id in self._pending_events:
                                del self._pending_events[event.id]
                            try:
                                self._event_queue.put_nowait(event)
                            except queue.Full:
                                self._dropped_event_count += 1
                                logger.warning(
                                    f"AT_LEAST_ONCE event {event.id} dropped on retry - queue full"
                                )
                        else:
                            # Max retries reached
                            self._pending_events.pop(event.id, None)
                            logger.error(
                                f"AT_LEAST_ONCE event {event.id} failed after "
                                f"{event._delivery_count} delivery attempts"
                            )

            except asyncio.CancelledError:
                break
            except Exception:
                # Silently suppress all dispatch loop errors
                pass

    async def _safe_call_handler(self, handler: EventHandler, event: MessagingEvent) -> None:
        """Safely call a handler, catching exceptions.

        Handles both async and sync handlers gracefully.
        """
        try:
            # Check if handler is a coroutine function
            import inspect

            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                # Sync handler - call it directly
                handler(event)
        except Exception as e:
            logger.warning(f"Event handler error for {event.topic}: {e}")

    def get_subscription_count(self) -> int:
        """Get number of active subscriptions."""
        # Note: This is called from sync context in tests, so we can't use async with
        # For thread safety in tests that create multiple event loops, we'll count without lock
        # In production, this should be called from the same event loop as connect()
        return sum(1 for s in self._subscriptions.values() if s.is_active)

    def get_queue_depth(self) -> int:
        """Get current event queue depth."""
        return self._event_queue.qsize()

    def get_queue_capacity(self) -> int:
        """Get maximum queue size."""
        return self._queue_maxsize

    def get_dropped_event_count(self) -> int:
        """Get number of events dropped due to queue overflow."""
        return self._dropped_event_count

    def get_sampling_rate(self) -> float:
        """Get the current sampling rate (0.0 to 1.0)."""
        return self._sampling_rate

    def get_sampled_event_count(self) -> int:
        """Get number of events sampled out (dropped due to sampling)."""
        return self._sampled_event_count

    def get_total_event_count(self) -> int:
        """Get total number of events published (including sampled)."""
        return self._total_event_count

    def get_sampling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive sampling statistics.

        Returns:
            Dict with sampling_rate, sampled_count, total_count, and effective_rate
        """
        if self._total_event_count == 0:
            effective_rate = 1.0
        else:
            effective_rate = 1.0 - (self._sampled_event_count / self._total_event_count)

        return {
            "sampling_rate": self._sampling_rate,
            "sampled_count": self._sampled_event_count,
            "total_count": self._total_event_count,
            "effective_rate": effective_rate,
            "whitelisted_patterns": list(self._sampling_whitelist),
        }

    def get_queue_utilization(self) -> float:
        """Get queue utilization as a percentage (0.0 to 100.0)."""
        if self._queue_maxsize == 0:
            return 0.0  # Unbounded queue
        return (self.get_queue_depth() / self._queue_maxsize) * 100.0

    def is_overflowing(self) -> bool:
        """Check if queue is currently full."""
        return self.get_queue_depth() >= self._queue_maxsize if self._queue_maxsize > 0 else False


# =============================================================================
# Backend Factory
# =============================================================================

# Registry of backend factories
_backend_factories: Dict[BackendType, Callable[[BackendConfig], IEventBackend]] = {}


def register_backend_factory(
    backend_type: BackendType,
    factory: Callable[[BackendConfig], IEventBackend],
) -> None:
    """Register a backend factory.

    External packages can register their own backend implementations:

    Example:
        from victor.core.events.backends import register_backend_factory
        from my_package import KafkaEventBackend

        register_backend_factory(
            BackendType.KAFKA,
            lambda config: KafkaEventBackend(config)
        )
    """
    _backend_factories[backend_type] = factory


def create_event_backend(
    config: Optional[BackendConfig] = None,
    *,
    backend_type: Optional[BackendType] = None,
) -> IEventBackend:
    """Factory function to create event backends.

    Creates the appropriate backend based on configuration or explicit type.
    If no backend is registered for the requested type, falls back to InMemory.

    Args:
        config: Optional backend configuration
        backend_type: Override backend type (uses config.backend_type if not set)

    Returns:
        IEventBackend implementation

    Example:
        # Default in-memory backend
        backend = create_event_backend()

        # With config
        config = BackendConfig(
            backend_type=BackendType.KAFKA,
            extra={"bootstrap_servers": "localhost:9092"}
        )
        backend = create_event_backend(config)

        # Override type
        backend = create_event_backend(backend_type=BackendType.REDIS)
    """
    config = config or BackendConfig()
    selected_type = backend_type or config.backend_type

    # Check for registered factory
    factory = _backend_factories.get(selected_type)
    if factory:
        return factory(config)

    # Default to InMemory for unregistered types
    if selected_type != BackendType.IN_MEMORY:
        logger.warning(
            f"Backend type '{selected_type.value}' not registered, " f"falling back to IN_MEMORY"
        )

    return InMemoryEventBackend(config)


# Register built-in backend
def _create_in_memory_backend(config: BackendConfig) -> InMemoryEventBackend:
    """Create in-memory backend with optional sampling configuration."""
    extra = config.extra or {}
    sampling_rate = extra.get("sampling_rate", 1.0)
    sampling_whitelist = extra.get("sampling_whitelist")
    if sampling_whitelist is not None:
        sampling_whitelist = set(sampling_whitelist)
    return InMemoryEventBackend(
        config,
        sampling_rate=sampling_rate,
        sampling_whitelist=sampling_whitelist,
    )


register_backend_factory(
    BackendType.IN_MEMORY,
    _create_in_memory_backend,
)


# Register distributed backends if available
def _register_distributed_backends() -> None:
    """Register Redis and Kafka backends if their dependencies are available."""
    # Redis backend
    try:
        from victor.core.events.redis_backend import RedisEventBackend

        register_backend_factory(
            BackendType.REDIS,
            lambda config: RedisEventBackend(config),
        )
        logger.debug("Registered RedisEventBackend")
    except ImportError:
        pass

    # Kafka backend
    try:
        from victor.core.events.kafka_backend import KafkaEventBackend

        register_backend_factory(
            BackendType.KAFKA,
            lambda config: KafkaEventBackend(config),
        )
        logger.debug("Registered KafkaEventBackend")
    except ImportError:
        pass


# Auto-register distributed backends on module load
_register_distributed_backends()


# =============================================================================
# Specialized Bus Classes
# =============================================================================


class ObservabilityBus:
    """Specialized event bus for observability (metrics, tracing, logging).

    Optimized for:
    - High throughput
    - Lossy delivery OK (AT_MOST_ONCE)
    - Batching and sampling
    - Low latency

    Example:
        bus = ObservabilityBus()
        await bus.connect()

        # Emit metrics
        await bus.emit("metric.latency", {"value": 42.5, "unit": "ms"})

        # Subscribe to all metrics
        await bus.subscribe("metric.*", lambda e: print(e.data))
    """

    def __init__(
        self,
        backend: Optional[IEventBackend] = None,
        config: Optional[BackendConfig] = None,
    ) -> None:
        """Initialize observability bus.

        Args:
            backend: Optional pre-configured backend
            config: Configuration (ignored if backend provided)
        """
        if backend:
            self._backend = backend
        else:
            cfg = config or BackendConfig.for_observability()
            self._backend = create_event_backend(cfg)

        # Track exporters (for compatibility with old EventBus API)
        self._exporters: List[Any] = []
        self._exporter_handles: List[SubscriptionHandle] = []
        self._pending_exporters: List[tuple[Any, EventHandler]] = (
            []
        )  # (exporter, handler) awaiting subscription

    @property
    def backend(self) -> IEventBackend:
        """Get underlying backend."""
        return self._backend

    async def connect(self) -> None:
        """Connect the backend."""
        await self._backend.connect()

    async def disconnect(self) -> None:
        """Disconnect the backend."""
        await self._backend.disconnect()

    def add_exporter(self, exporter: Any) -> None:
        """Add an event exporter to the bus.

        Exporters are handlers that write events to external systems
        (files, databases, APIs). They receive all events emitted through the bus.

        Note: The exporter will be subscribed on the next call to emit().

        Args:
            exporter: Exporter object with export(event) method
        """

        # Create handler for this exporter
        async def _export_handler(event):
            try:
                if hasattr(exporter, "export"):
                    await exporter.export(event)
                elif callable(exporter):
                    # Exporter is a callable function
                    await exporter(event)
            except Exception as e:
                logger.debug(f"Exporter error: {e}")

        # Store exporter and handler for subscription on next emit
        self._exporters.append(exporter)
        self._pending_exporters.append((exporter, _export_handler))

    async def _subscribe_exporter(self, exporter: Any, handler: EventHandler) -> None:
        """Subscribe an exporter to the event bus.

        This is called asynchronously to handle subscription when event loop is available.

        Args:
            exporter: The exporter object
            handler: The event handler for the exporter
        """
        # Subscribe to all events ("*" pattern)
        try:
            handle = await self.subscribe("*", handler)
            # Store the handle so we can unsubscribe later
            self._exporter_handles.append((exporter, handle))

            # Remove from pending list
            self._pending_exporters = [
                (exp, h) for exp, h in self._pending_exporters if exp != exporter
            ]
        except Exception as e:
            logger.debug(f"Failed to subscribe exporter: {e}")

    def remove_exporter(self, exporter: Any) -> None:
        """Remove an event exporter from the bus.

        Args:
            exporter: Exporter object to remove
        """
        # Remove from list
        if exporter in self._exporters:
            self._exporters.remove(exporter)

        # Remove from pending
        self._pending_exporters = [
            (exp, h) for exp, h in self._pending_exporters if exp != exporter
        ]

        # Unsubscribe from backend (sync - deactivates subscription immediately)
        for i, (exp, handle) in enumerate(self._exporter_handles):
            if exp == exporter:
                # Deactivate subscription directly (sync operation)
                handle.is_active = False
                # Remove from backend's subscription dict
                if hasattr(self._backend, "_subscriptions"):
                    # Remove without lock since this is sync context
                    # In production, should call await handle.unsubscribe() instead
                    if handle.subscription_id in self._backend._subscriptions:
                        del self._backend._subscriptions[handle.subscription_id]
                # Remove from handles list
                del self._exporter_handles[i]
                break

    async def emit(
        self,
        topic: str,
        data: Dict[str, Any],
        *,
        source: str = "victor",
        correlation_id: Optional[str] = None,
    ) -> bool:
        """Emit an observability event.

        Args:
            topic: Event topic (e.g., "metric.latency", "trace.span")
            data: Event payload
            source: Event source identifier
            correlation_id: Optional correlation ID

        Returns:
            True if event was emitted
        """
        # Process pending exporter subscriptions
        if self._pending_exporters:
            pending = list(self._pending_exporters)  # Copy to avoid modification during iteration
            for exporter, handler in pending:
                await self._subscribe_exporter(exporter, handler)

        event = MessagingEvent(
            topic=topic,
            data=data,
            source=source,
            correlation_id=correlation_id,
            delivery_guarantee=DeliveryGuarantee.AT_MOST_ONCE,
        )

        try:
            # Auto-connect backend if not connected (lazy initialization)
            if not self._backend._is_connected:
                await self._backend.connect()

            return await self._backend.publish(event)
        except EventPublishError as e:
            # Log but don't propagate - observability errors shouldn't crash the app
            logger.debug(f"Failed to emit observability event: {e}")
            return False

    async def subscribe(
        self,
        pattern: str,
        handler: EventHandler,
    ) -> SubscriptionHandle:
        """Subscribe to observability events."""
        return await self._backend.subscribe(pattern, handler)

    async def unsubscribe(self, handle: SubscriptionHandle) -> bool:
        """Unsubscribe from observability events.

        Args:
            handle: Subscription handle returned by subscribe()

        Returns:
            True if unsubscribed successfully
        """
        return await self._backend.unsubscribe(handle)

    def emit_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
    ) -> None:
        """Emit an error event (fire-and-forget).

        This is a compatibility method for code that expects EventBus-like interface.
        Events are emitted asynchronously without waiting for completion.

        Args:
            error: The exception.
            context: Optional error context.
            recoverable: Whether error is recoverable.
        """
        # Schedule emit as fire-and-forget task
        import asyncio

        asyncio.create_task(
            self.emit(
                topic="error",
                data={
                    "message": str(error),
                    "type": type(error).__name__,
                    "recoverable": recoverable,
                    **(context or {}),
                },
                source="observability",
            )
        )

    def emit_metric(
        self,
        metric: str,
        value: float,
        **labels: Any,
    ) -> None:
        """Emit a metric event (fire-and-forget).

        This is a convenience method for emitting metric events.
        Events are emitted asynchronously without waiting for completion.

        Args:
            metric: Metric name (e.g., "latency", "request_count")
            value: Metric value
            **labels: Additional metric labels/dimensions
        """
        # Schedule emit as fire-and-forget task
        import asyncio

        asyncio.create_task(
            self.emit(
                topic="metric",
                data={
                    "name": metric,
                    "value": value,
                    **labels,
                },
                source="observability",
            )
        )


class AgentMessageBus:
    """Specialized event bus for cross-agent communication.

    Optimized for:
    - Reliable delivery (AT_LEAST_ONCE)
    - Point-to-point and broadcast messaging
    - Request-response patterns
    - Agent coordination

    Example:
        bus = AgentMessageBus()
        await bus.connect()

        # Send to specific agent
        await bus.send("agent.task", {"action": "analyze"}, to_agent="researcher")

        # Broadcast to all agents
        await bus.broadcast("agent.status", {"phase": "planning"})

        # Subscribe to messages for this agent
        await bus.subscribe_agent("executor", handler)
    """

    def __init__(
        self,
        backend: Optional[IEventBackend] = None,
        config: Optional[BackendConfig] = None,
    ) -> None:
        """Initialize agent message bus.

        Args:
            backend: Optional pre-configured backend
            config: Configuration (ignored if backend provided)
        """
        if backend:
            self._backend = backend
        else:
            cfg = config or BackendConfig.for_agent_messaging()
            self._backend = create_event_backend(cfg)

    @property
    def backend(self) -> IEventBackend:
        """Get underlying backend."""
        return self._backend

    async def connect(self) -> None:
        """Connect the backend."""
        await self._backend.connect()

    async def disconnect(self) -> None:
        """Disconnect the backend."""
        await self._backend.disconnect()

    async def send(
        self,
        topic: str,
        data: Dict[str, Any],
        *,
        to_agent: str,
        from_agent: str = "coordinator",
        correlation_id: Optional[str] = None,
    ) -> bool:
        """Send a message to a specific agent.

        Args:
            topic: Message topic
            data: Message payload
            to_agent: Target agent ID
            from_agent: Sender agent ID
            correlation_id: Optional correlation ID

        Returns:
            True if message was sent
        """
        event = MessagingEvent(
            topic=f"agent.{to_agent}.{topic}",
            data={
                **data,
                "_from_agent": from_agent,
                "_to_agent": to_agent,
            },
            source=from_agent,
            correlation_id=correlation_id,
            partition_key=to_agent,
            delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
        )

        # Auto-connect backend if not connected (lazy initialization)
        if not self._backend._is_connected:
            await self._backend.connect()

        return await self._backend.publish(event)

    async def broadcast(
        self,
        topic: str,
        data: Dict[str, Any],
        *,
        from_agent: str = "coordinator",
        correlation_id: Optional[str] = None,
    ) -> bool:
        """Broadcast a message to all agents.

        Args:
            topic: Message topic
            data: Message payload
            from_agent: Sender agent ID
            correlation_id: Optional correlation ID

        Returns:
            True if broadcast was sent
        """
        event = MessagingEvent(
            topic=f"agent.broadcast.{topic}",
            data={
                **data,
                "_from_agent": from_agent,
                "_broadcast": True,
            },
            source=from_agent,
            correlation_id=correlation_id,
            delivery_guarantee=DeliveryGuarantee.AT_LEAST_ONCE,
        )

        # Auto-connect backend if not connected (lazy initialization)
        if not self._backend._is_connected:
            await self._backend.connect()

        return await self._backend.publish(event)

    async def subscribe_agent(
        self,
        agent_id: str,
        handler: EventHandler,
    ) -> SubscriptionHandle:
        """Subscribe to messages for a specific agent.

        Args:
            agent_id: Agent ID to subscribe for
            handler: Message handler

        Returns:
            Subscription handle
        """
        # Subscribe to directed messages and broadcasts
        directed_handle = await self._backend.subscribe(
            f"agent.{agent_id}.*",
            handler,
        )
        broadcast_handle = await self._backend.subscribe(
            "agent.broadcast.*",
            handler,
        )

        # Create combined handle
        combined = SubscriptionHandle(
            subscription_id=f"{directed_handle.subscription_id}+{broadcast_handle.subscription_id}",
            pattern=f"agent.{agent_id}.* + agent.broadcast.*",
            is_active=True,
        )

        async def unsubscribe_both() -> None:
            await directed_handle.unsubscribe()
            await broadcast_handle.unsubscribe()
            combined.is_active = False

        combined.unsubscribe = unsubscribe_both
        return combined


# =============================================================================
# Convenience Functions for DI Integration
# =============================================================================


def get_observability_bus() -> ObservabilityBus:
    """Get the ObservabilityBus instance from DI container.

    This is a convenience function that retrieves the ObservabilityBus
    from the global service container. If not registered, it will
    automatically register and return a new instance using backend from settings.

    The backend type is determined by settings.event_backend_type, which defaults
    to "in_memory" if not configured.

    Returns:
        ObservabilityBus instance

    Example:
        >>> from victor.core.events.backends import get_observability_bus
        >>>
        >>> bus = get_observability_bus()
        >>> await bus.emit("tool.start", {"tool": "read_file"})
    """
    from victor.core.container import get_container
    from victor.config.settings import get_settings

    container = get_container()

    # Auto-register if not exists
    if not container.is_registered(ObservabilityBus):

        def create_bus(container):
            # Read settings to determine backend
            settings = get_settings()
            backend_type_str = settings.event_backend_type.lower()

            # Map string to BackendType enum
            from victor.core.events.protocols import BackendType

            backend_type_map = {
                "in_memory": BackendType.IN_MEMORY,
                "sqlite": BackendType.DATABASE,
                "redis": BackendType.REDIS,
                "kafka": BackendType.KAFKA,
                "sqs": BackendType.SQS,
                "rabbitmq": BackendType.RABBITMQ,
            }

            backend_type = backend_type_map.get(backend_type_str, BackendType.IN_MEMORY)

            # Create backend from settings
            cfg = BackendConfig.for_observability()
            if settings.eventbus_sampling_enabled:
                sampling_rate = max(0.0, min(1.0, settings.eventbus_sampling_default_rate))
                cfg.extra["sampling_rate"] = sampling_rate
                # Always keep critical events
                cfg.extra["sampling_whitelist"] = {
                    "error.*",
                    "lifecycle.*",
                }
            backend = create_event_backend(cfg, backend_type=backend_type)

            return ObservabilityBus(backend=backend)

        from victor.core.container import ServiceLifetime

        container.register(ObservabilityBus, create_bus, ServiceLifetime.SINGLETON)

    return container.get(ObservabilityBus)


def get_agent_message_bus() -> AgentMessageBus:
    """Get the AgentMessageBus instance from DI container.

    This is a convenience function that retrieves the AgentMessageBus
    from the global service container. If not registered, it will
    automatically register and return a new instance using backend from settings.

    The backend type is determined by settings.event_backend_type, which defaults
    to "in_memory" if not configured.

    Returns:
        AgentMessageBus instance

    Example:
        >>> from victor.core.events.backends import get_agent_message_bus
        >>>
        >>> bus = get_agent_message_bus()
        >>> await bus.send("task", {"action": "analyze"}, to_agent="researcher")
    """
    from victor.core.container import get_container
    from victor.config.settings import get_settings

    container = get_container()

    # Auto-register if not exists
    if not container.is_registered(AgentMessageBus):

        def create_bus(container):
            # Read settings to determine backend
            settings = get_settings()
            backend_type_str = settings.event_backend_type.lower()

            # Map string to BackendType enum
            from victor.core.events.protocols import BackendType

            backend_type_map = {
                "in_memory": BackendType.IN_MEMORY,
                "sqlite": BackendType.DATABASE,
                "redis": BackendType.REDIS,
                "kafka": BackendType.KAFKA,
                "sqs": BackendType.SQS,
                "rabbitmq": BackendType.RABBITMQ,
            }

            backend_type = backend_type_map.get(backend_type_str, BackendType.IN_MEMORY)

            # Create backend from settings
            backend = create_event_backend(backend_type=backend_type)

            return AgentMessageBus(backend=backend)

        from victor.core.container import ServiceLifetime

        container.register(AgentMessageBus, create_bus, ServiceLifetime.SINGLETON)

    return container.get(AgentMessageBus)


def get_event_backend() -> IEventBackend:
    """Get the IEventBackend instance from DI container.

    This is a convenience function that retrieves the event backend
    from the global service container.

    Returns:
        IEventBackend instance

    Example:
        >>> from victor.core.events.backends import get_event_backend
        >>>
        >>> backend = get_event_backend()
        >>> await backend.publish(MessagingEvent(topic="test", data={}))
    """
    from victor.core.container import get_container

    container = get_container()
    return container.get(IEventBackend)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Backends
    "InMemoryEventBackend",
    # Specialized buses
    "ObservabilityBus",
    "AgentMessageBus",
    # Factory
    "create_event_backend",
    "register_backend_factory",
    # Convenience functions for DI integration
    "get_observability_bus",
    "get_agent_message_bus",
    "get_event_backend",
]
