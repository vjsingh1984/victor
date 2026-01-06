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
    await backend.publish(Event(topic="test", data={}))

    # Or create directly
    backend = InMemoryEventBackend()
    await backend.connect()
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from victor.core.events.protocols import (
    BackendConfig,
    BackendType,
    DeliveryGuarantee,
    Event,
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

    Limitations:
    - Events are lost on process restart
    - Cannot scale horizontally
    - Only AT_MOST_ONCE delivery (no persistence)

    Thread Safety:
    - Subscriptions are thread-safe (uses lock)
    - Publish is thread-safe (uses asyncio.Queue)
    - Handlers run in asyncio tasks (non-blocking)

    Example:
        backend = InMemoryEventBackend()
        await backend.connect()

        async def handler(event):
            print(f"Received: {event.topic}")

        handle = await backend.subscribe("tool.*", handler)
        await backend.publish(Event(topic="tool.call", data={"name": "read"}))

        await backend.disconnect()
    """

    def __init__(
        self,
        config: Optional[BackendConfig] = None,
        *,
        queue_maxsize: int = 10000,
    ) -> None:
        """Initialize the in-memory backend.

        Args:
            config: Optional backend configuration
            queue_maxsize: Maximum queue size (0 for unbounded)
        """
        self._config = config or BackendConfig()
        self._queue_maxsize = queue_maxsize
        self._subscriptions: Dict[str, _Subscription] = {}
        self._event_queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=queue_maxsize)
        self._is_connected = False
        self._dispatcher_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        self._pending_tasks: Set[asyncio.Task] = set()

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
        """
        if self._is_connected:
            return

        self._is_connected = True
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

    async def publish(self, event: Event) -> bool:
        """Publish an event to all matching subscribers.

        Args:
            event: Event to publish

        Returns:
            True if event was queued for delivery

        Raises:
            EventPublishError: If queue is full and cannot accept event
        """
        if not self._is_connected:
            raise EventPublishError(event, "Backend not connected", retryable=True)

        try:
            # Non-blocking put
            self._event_queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            # Queue full - drop event (AT_MOST_ONCE semantics)
            logger.warning(f"Event queue full, dropping event: {event.topic}")
            return False

    async def publish_batch(self, events: List[Event]) -> int:
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

        Args:
            handle: Subscription handle from subscribe()

        Returns:
            True if unsubscribed successfully
        """
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
        """Background loop that dispatches events to handlers."""
        while self._is_connected:
            try:
                # Wait for event with timeout to allow checking is_connected
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=0.1,
                    )
                except asyncio.TimeoutError:
                    continue

                # Get matching subscriptions
                with self._lock:
                    subscriptions = [
                        s
                        for s in self._subscriptions.values()
                        if s.is_active and event.matches_pattern(s.pattern)
                    ]

                # Dispatch to handlers concurrently
                for subscription in subscriptions:
                    task = asyncio.create_task(self._safe_call_handler(subscription.handler, event))
                    self._pending_tasks.add(task)
                    task.add_done_callback(self._pending_tasks.discard)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Dispatch loop error: {e}")

    async def _safe_call_handler(self, handler: EventHandler, event: Event) -> None:
        """Safely call a handler, catching exceptions."""
        try:
            await handler(event)
        except Exception as e:
            logger.warning(f"Event handler error for {event.topic}: {e}")

    def get_subscription_count(self) -> int:
        """Get number of active subscriptions."""
        with self._lock:
            return sum(1 for s in self._subscriptions.values() if s.is_active)

    def get_queue_depth(self) -> int:
        """Get current event queue depth."""
        return self._event_queue.qsize()


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
register_backend_factory(
    BackendType.IN_MEMORY,
    lambda config: InMemoryEventBackend(config),
)


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
        event = Event(
            topic=topic,
            data=data,
            source=source,
            correlation_id=correlation_id,
            delivery_guarantee=DeliveryGuarantee.AT_MOST_ONCE,
        )
        return await self._backend.publish(event)

    async def subscribe(
        self,
        pattern: str,
        handler: EventHandler,
    ) -> SubscriptionHandle:
        """Subscribe to observability events."""
        return await self._backend.subscribe(pattern, handler)


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
        event = Event(
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
        event = Event(
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
]
