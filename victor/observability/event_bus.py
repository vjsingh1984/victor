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

"""EventBus - Central pub/sub event system for Victor.

Implements the Pub/Sub pattern with:
- Thread-safe singleton access
- Category-based subscription
- Priority ordering
- Async support
- Lazy evaluation for performance
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Union,
)
from weakref import WeakSet

if TYPE_CHECKING:
    from victor.observability.exporters import BaseExporter

logger = logging.getLogger(__name__)


class EventCategory(str, Enum):
    """Categories of events in Victor.

    Events are categorized to allow selective subscription and filtering.
    """

    TOOL = "tool"  # Tool invocation and results
    STATE = "state"  # State machine transitions
    MODEL = "model"  # LLM interactions
    ERROR = "error"  # Errors and exceptions
    AUDIT = "audit"  # Security and compliance
    METRIC = "metric"  # Performance metrics
    LIFECYCLE = "lifecycle"  # Session start/end
    CUSTOM = "custom"  # User-defined events


class EventPriority(int, Enum):
    """Priority levels for event handlers.

    Higher priority handlers execute first.
    """

    LOW = 0
    NORMAL = 50
    HIGH = 100
    CRITICAL = 200


class BackpressureStrategy(str, Enum):
    """Strategy for handling queue overflow.

    When the event queue is full, this determines behavior:
    - DROP_OLDEST: Remove oldest event to make room (lossy but non-blocking)
    - DROP_NEWEST: Discard incoming event (lossy but preserves order)
    - BLOCK: Wait for space (can cause backpressure upstream)
    - REJECT: Raise exception (caller handles backpressure)
    """

    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"
    REJECT = "reject"


class EventQueueFullError(Exception):
    """Raised when event queue is full and backpressure strategy is REJECT."""

    def __init__(self, queue_size: int, event_name: str):
        self.queue_size = queue_size
        self.event_name = event_name
        super().__init__(f"Event queue full ({queue_size} events). Cannot enqueue '{event_name}'")


@dataclass
class BackpressureMetrics:
    """Metrics for monitoring event queue backpressure.

    Attributes:
        events_dropped: Total events dropped due to backpressure
        events_rejected: Total events rejected (REJECT strategy)
        peak_queue_depth: Maximum queue depth observed
        current_queue_depth: Current queue depth
        backpressure_events: Number of times backpressure was applied
    """

    events_dropped: int = 0
    events_rejected: int = 0
    peak_queue_depth: int = 0
    current_queue_depth: int = 0
    backpressure_events: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to dictionary."""
        return {
            "events_dropped": self.events_dropped,
            "events_rejected": self.events_rejected,
            "peak_queue_depth": self.peak_queue_depth,
            "current_queue_depth": self.current_queue_depth,
            "backpressure_events": self.backpressure_events,
        }


@dataclass(frozen=True)
class VictorEvent:
    """Canonical event format for all Victor observations.

    Immutable event object that can be serialized for persistence
    or transmitted to external systems.

    Attributes:
        id: Unique event identifier
        timestamp: When the event occurred
        category: Event category for routing
        name: Specific event name within category
        data: Event payload
        trace_id: Optional distributed trace ID
        session_id: Optional session identifier
        priority: Event priority for ordering
        source: Component that generated the event
    """

    category: EventCategory
    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    trace_id: Optional[str] = None
    session_id: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
    source: str = "victor"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary.

        Returns:
            JSON-serializable dictionary representation.
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "name": self.name,
            "data": self.data,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "priority": self.priority.value,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VictorEvent":
        """Deserialize event from dictionary.

        Args:
            data: Dictionary with event data.

        Returns:
            VictorEvent instance.
        """
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if "timestamp" in data
                else datetime.now(timezone.utc)
            ),
            category=EventCategory(data["category"]),
            name=data["name"],
            data=data.get("data", {}),
            trace_id=data.get("trace_id"),
            session_id=data.get("session_id"),
            priority=EventPriority(data.get("priority", EventPriority.NORMAL.value)),
            source=data.get("source", "victor"),
        )


# Type aliases
EventHandler = Callable[[VictorEvent], None]
AsyncEventHandler = Callable[[VictorEvent], Any]


@dataclass
class Subscription:
    """Represents a subscription to an event category.

    Attributes:
        handler: Callback function
        category: Event category to subscribe to
        priority: Handler priority
        is_async: Whether handler is async
        filter_fn: Optional filter function
    """

    handler: Union[EventHandler, AsyncEventHandler]
    category: EventCategory
    priority: EventPriority = EventPriority.NORMAL
    is_async: bool = False
    filter_fn: Optional[Callable[[VictorEvent], bool]] = None

    def matches(self, event: VictorEvent) -> bool:
        """Check if this subscription matches an event.

        Args:
            event: Event to check.

        Returns:
            True if subscription should receive this event.
        """
        if event.category != self.category:
            return False
        if self.filter_fn and not self.filter_fn(event):
            return False
        return True


class EventBus:
    """Central event bus implementing Pub/Sub pattern.

    Thread-safe singleton that manages event subscriptions and dispatch.
    Supports both sync and async handlers with priority ordering.

    Design Patterns:
        - Singleton: Single instance per process
        - Observer: Subscribers notified of events
        - Strategy: Pluggable exporters

    Example:
        bus = EventBus.get_instance()

        # Subscribe
        bus.subscribe(EventCategory.TOOL, my_handler)

        # Publish
        bus.publish(VictorEvent(category=EventCategory.TOOL, name="read"))
    """

    _instance: Optional["EventBus"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "EventBus":
        """Create or return singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    # Default configuration for backpressure
    DEFAULT_QUEUE_MAXSIZE = 10000
    DEFAULT_BACKPRESSURE_STRATEGY = BackpressureStrategy.DROP_OLDEST

    def __init__(
        self,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        backpressure_strategy: BackpressureStrategy = DEFAULT_BACKPRESSURE_STRATEGY,
    ) -> None:
        """Initialize the event bus.

        Args:
            queue_maxsize: Maximum queue size (0 for unbounded, not recommended).
            backpressure_strategy: Strategy when queue is full.
        """
        if getattr(self, "_initialized", False):
            return

        self._subscriptions: Dict[EventCategory, List[Subscription]] = {
            cat: [] for cat in EventCategory
        }
        self._exporters: List["BaseExporter"] = []
        self._queue_maxsize = queue_maxsize
        self._backpressure_strategy = backpressure_strategy
        self._event_queue: asyncio.Queue[VictorEvent] = asyncio.Queue(maxsize=queue_maxsize)
        self._is_processing = False
        self._session_id: Optional[str] = None
        self._trace_context: Dict[str, str] = {}
        self._backpressure_metrics = BackpressureMetrics()
        self._initialized = True
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "EventBus":
        """Get the singleton EventBus instance.

        Returns:
            EventBus singleton.
        """
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing).

        Warning: Only use in tests!
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance._subscriptions = {cat: [] for cat in EventCategory}
                cls._instance._exporters = []
                cls._instance._session_id = None
                cls._instance._trace_context = {}
                cls._instance._backpressure_metrics = BackpressureMetrics()
                # Recreate queue with same maxsize
                maxsize = getattr(cls._instance, "_queue_maxsize", 10000)
                cls._instance._event_queue = asyncio.Queue(maxsize=maxsize)

    def set_session_id(self, session_id: str) -> None:
        """Set the current session ID for event correlation.

        Args:
            session_id: Session identifier.
        """
        self._session_id = session_id

    def set_trace_context(self, trace_id: str, span_id: Optional[str] = None) -> None:
        """Set distributed tracing context.

        Args:
            trace_id: Distributed trace ID.
            span_id: Optional span ID.
        """
        self._trace_context = {"trace_id": trace_id}
        if span_id:
            self._trace_context["span_id"] = span_id

    def subscribe(
        self,
        category: EventCategory,
        handler: Union[EventHandler, AsyncEventHandler],
        *,
        priority: EventPriority = EventPriority.NORMAL,
        filter_fn: Optional[Callable[[VictorEvent], bool]] = None,
    ) -> Callable[[], None]:
        """Subscribe to events in a category.

        Args:
            category: Event category to subscribe to.
            handler: Callback function for events.
            priority: Handler priority (higher executes first).
            filter_fn: Optional function to filter events.

        Returns:
            Unsubscribe function.

        Example:
            def on_tool(event):
                print(f"Tool: {event.name}")

            unsubscribe = bus.subscribe(EventCategory.TOOL, on_tool)
            # Later: unsubscribe()
        """
        is_async = asyncio.iscoroutinefunction(handler)

        subscription = Subscription(
            handler=handler,
            category=category,
            priority=priority,
            is_async=is_async,
            filter_fn=filter_fn,
        )

        with self._lock:
            self._subscriptions[category].append(subscription)
            # Sort by priority (highest first)
            self._subscriptions[category].sort(key=lambda s: s.priority, reverse=True)

        def unsubscribe() -> None:
            with self._lock:
                if subscription in self._subscriptions[category]:
                    self._subscriptions[category].remove(subscription)

        return unsubscribe

    def subscribe_all(
        self,
        handler: Union[EventHandler, AsyncEventHandler],
        *,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> Callable[[], None]:
        """Subscribe to all event categories.

        Args:
            handler: Callback function.
            priority: Handler priority.

        Returns:
            Unsubscribe function that removes all subscriptions.
        """
        unsubscribers = []
        for category in EventCategory:
            unsub = self.subscribe(category, handler, priority=priority)
            unsubscribers.append(unsub)

        def unsubscribe_all() -> None:
            for unsub in unsubscribers:
                unsub()

        return unsubscribe_all

    def publish(self, event: VictorEvent) -> None:
        """Publish an event synchronously.

        Notifies all matching subscribers and exporters.

        Args:
            event: Event to publish.
        """
        # Enrich event with context
        if self._session_id and not event.session_id:
            # Create new event with session_id (frozen dataclass)
            event = VictorEvent(
                id=event.id,
                timestamp=event.timestamp,
                category=event.category,
                name=event.name,
                data=event.data,
                trace_id=event.trace_id or self._trace_context.get("trace_id"),
                session_id=self._session_id,
                priority=event.priority,
                source=event.source,
            )

        # Notify subscribers
        with self._lock:
            subscriptions = list(self._subscriptions.get(event.category, []))

        for subscription in subscriptions:
            if subscription.matches(event):
                try:
                    if subscription.is_async:
                        # Queue async handler
                        asyncio.create_task(subscription.handler(event))  # type: ignore
                    else:
                        subscription.handler(event)
                except Exception as e:
                    logger.warning(f"Event handler error: {e}")

        # Send to exporters
        for exporter in self._exporters:
            try:
                exporter.export(event)
            except Exception as e:
                logger.warning(f"Exporter error: {e}")

    async def publish_async(self, event: VictorEvent) -> None:
        """Publish an event asynchronously.

        Args:
            event: Event to publish.
        """
        # Enrich event with context
        if self._session_id and not event.session_id:
            event = VictorEvent(
                id=event.id,
                timestamp=event.timestamp,
                category=event.category,
                name=event.name,
                data=event.data,
                trace_id=event.trace_id or self._trace_context.get("trace_id"),
                session_id=self._session_id,
                priority=event.priority,
                source=event.source,
            )

        # Notify subscribers
        with self._lock:
            subscriptions = list(self._subscriptions.get(event.category, []))

        for subscription in subscriptions:
            if subscription.matches(event):
                try:
                    if subscription.is_async:
                        await subscription.handler(event)  # type: ignore
                    else:
                        subscription.handler(event)
                except Exception as e:
                    logger.warning(f"Event handler error: {e}")

        # Send to exporters asynchronously
        for exporter in self._exporters:
            try:
                if hasattr(exporter, "export_async"):
                    await exporter.export_async(event)
                else:
                    exporter.export(event)
            except Exception as e:
                logger.warning(f"Exporter error: {e}")

    def add_exporter(self, exporter: "BaseExporter") -> None:
        """Add an event exporter.

        Args:
            exporter: Exporter instance.
        """
        self._exporters.append(exporter)

    def remove_exporter(self, exporter: "BaseExporter") -> None:
        """Remove an event exporter.

        Args:
            exporter: Exporter to remove.
        """
        if exporter in self._exporters:
            self._exporters.remove(exporter)

    def get_subscription_count(self, category: Optional[EventCategory] = None) -> int:
        """Get number of subscriptions.

        Args:
            category: Optional category to count.

        Returns:
            Number of subscriptions.
        """
        if category:
            return len(self._subscriptions.get(category, []))
        return sum(len(subs) for subs in self._subscriptions.values())

    # =========================================================================
    # Backpressure Management
    # =========================================================================

    def configure_backpressure(
        self,
        strategy: BackpressureStrategy,
        queue_maxsize: Optional[int] = None,
    ) -> None:
        """Configure backpressure handling at runtime.

        Note: Changing queue_maxsize requires recreating the queue,
        which may lose pending events. Use with caution.

        Args:
            strategy: New backpressure strategy.
            queue_maxsize: Optional new queue size (recreates queue).
        """
        with self._lock:
            self._backpressure_strategy = strategy
            if queue_maxsize is not None and queue_maxsize != self._queue_maxsize:
                self._queue_maxsize = queue_maxsize
                # Drain existing queue and recreate
                old_events = []
                while not self._event_queue.empty():
                    try:
                        old_events.append(self._event_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                self._event_queue = asyncio.Queue(maxsize=queue_maxsize)
                # Re-add events up to new capacity
                for event in old_events[:queue_maxsize] if queue_maxsize > 0 else old_events:
                    try:
                        self._event_queue.put_nowait(event)
                    except asyncio.QueueFull:
                        break

    def get_backpressure_metrics(self) -> BackpressureMetrics:
        """Get current backpressure metrics.

        Returns:
            BackpressureMetrics with current stats.
        """
        # Update current queue depth
        self._backpressure_metrics.current_queue_depth = self._event_queue.qsize()
        return self._backpressure_metrics

    def get_queue_depth(self) -> int:
        """Get current queue depth.

        Returns:
            Number of events in queue.
        """
        return self._event_queue.qsize()

    def get_queue_capacity(self) -> int:
        """Get queue capacity (maxsize).

        Returns:
            Maximum queue size (0 means unbounded).
        """
        return self._queue_maxsize

    def is_queue_full(self) -> bool:
        """Check if queue is at capacity.

        Returns:
            True if queue is full (or unbounded queue always returns False).
        """
        return self._event_queue.full()

    async def queue_event_async(
        self,
        event: VictorEvent,
        timeout: Optional[float] = None,
    ) -> bool:
        """Queue an event with backpressure handling.

        This method respects the configured backpressure strategy when
        the queue is full.

        Args:
            event: Event to queue.
            timeout: Optional timeout for BLOCK strategy (seconds).

        Returns:
            True if event was queued, False if dropped.

        Raises:
            EventQueueFullError: If strategy is REJECT and queue is full.
        """
        # Track queue depth metrics
        current_depth = self._event_queue.qsize()
        if current_depth > self._backpressure_metrics.peak_queue_depth:
            self._backpressure_metrics.peak_queue_depth = current_depth

        # Try non-blocking put first
        try:
            self._event_queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            pass

        # Queue is full - apply backpressure strategy
        self._backpressure_metrics.backpressure_events += 1

        if self._backpressure_strategy == BackpressureStrategy.DROP_NEWEST:
            self._backpressure_metrics.events_dropped += 1
            logger.debug(f"Event dropped (DROP_NEWEST): {event.name}")
            return False

        elif self._backpressure_strategy == BackpressureStrategy.DROP_OLDEST:
            try:
                # Remove oldest event
                dropped = self._event_queue.get_nowait()
                logger.debug(f"Event dropped (DROP_OLDEST): {dropped.name}")
                self._backpressure_metrics.events_dropped += 1
                # Add new event
                self._event_queue.put_nowait(event)
                return True
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                # Race condition - just drop new event
                self._backpressure_metrics.events_dropped += 1
                return False

        elif self._backpressure_strategy == BackpressureStrategy.BLOCK:
            try:
                await asyncio.wait_for(
                    self._event_queue.put(event),
                    timeout=timeout,
                )
                return True
            except asyncio.TimeoutError:
                logger.warning(f"Event queue blocked timeout: {event.name}")
                self._backpressure_metrics.events_dropped += 1
                return False

        elif self._backpressure_strategy == BackpressureStrategy.REJECT:
            self._backpressure_metrics.events_rejected += 1
            raise EventQueueFullError(self._queue_maxsize, event.name)

        return False

    def queue_event_sync(self, event: VictorEvent) -> bool:
        """Queue an event synchronously with backpressure handling.

        Non-blocking version for sync code paths.

        Args:
            event: Event to queue.

        Returns:
            True if event was queued, False if dropped.

        Raises:
            EventQueueFullError: If strategy is REJECT and queue is full.
        """
        # Track queue depth metrics
        current_depth = self._event_queue.qsize()
        if current_depth > self._backpressure_metrics.peak_queue_depth:
            self._backpressure_metrics.peak_queue_depth = current_depth

        # Try non-blocking put first
        try:
            self._event_queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            pass

        # Queue is full - apply backpressure strategy
        self._backpressure_metrics.backpressure_events += 1

        if self._backpressure_strategy == BackpressureStrategy.DROP_NEWEST:
            self._backpressure_metrics.events_dropped += 1
            logger.debug(f"Event dropped (DROP_NEWEST): {event.name}")
            return False

        elif self._backpressure_strategy == BackpressureStrategy.DROP_OLDEST:
            try:
                dropped = self._event_queue.get_nowait()
                logger.debug(f"Event dropped (DROP_OLDEST): {dropped.name}")
                self._backpressure_metrics.events_dropped += 1
                self._event_queue.put_nowait(event)
                return True
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                self._backpressure_metrics.events_dropped += 1
                return False

        elif self._backpressure_strategy == BackpressureStrategy.BLOCK:
            # In sync context, BLOCK behaves like DROP_NEWEST
            self._backpressure_metrics.events_dropped += 1
            logger.warning(f"Event dropped (BLOCK in sync context): {event.name}")
            return False

        elif self._backpressure_strategy == BackpressureStrategy.REJECT:
            self._backpressure_metrics.events_rejected += 1
            raise EventQueueFullError(self._queue_maxsize, event.name)

        return False

    # =========================================================================
    # Convenience Factory Methods
    # =========================================================================

    def emit_tool_start(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_id: Optional[str] = None,
    ) -> None:
        """Emit a tool start event.

        Args:
            tool_name: Name of the tool.
            arguments: Tool arguments.
            tool_id: Optional tool call ID.
        """
        self.publish(
            VictorEvent(
                category=EventCategory.TOOL,
                name=f"{tool_name}.start",
                data={
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "tool_id": tool_id,
                },
            )
        )

    def emit_tool_end(
        self,
        tool_name: str,
        result: Any,
        success: bool = True,
        tool_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Emit a tool end event.

        Args:
            tool_name: Name of the tool.
            result: Tool result.
            success: Whether tool succeeded.
            tool_id: Optional tool call ID.
            duration_ms: Optional duration in milliseconds.
        """
        self.publish(
            VictorEvent(
                category=EventCategory.TOOL,
                name=f"{tool_name}.end",
                data={
                    "tool_name": tool_name,
                    "result": str(result)[:1000],  # Truncate large results
                    "success": success,
                    "tool_id": tool_id,
                    "duration_ms": duration_ms,
                },
            )
        )

    def emit_state_change(
        self,
        old_stage: str,
        new_stage: str,
        confidence: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a state change event.

        Args:
            old_stage: Previous stage name.
            new_stage: New stage name.
            confidence: Transition confidence.
            context: Optional additional context.
        """
        self.publish(
            VictorEvent(
                category=EventCategory.STATE,
                name="stage_transition",
                data={
                    "old_stage": old_stage,
                    "new_stage": new_stage,
                    "confidence": confidence,
                    **(context or {}),
                },
            )
        )

    def emit_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
    ) -> None:
        """Emit an error event.

        Args:
            error: The exception.
            context: Optional error context.
            recoverable: Whether error is recoverable.
        """
        self.publish(
            VictorEvent(
                category=EventCategory.ERROR,
                name=type(error).__name__,
                priority=EventPriority.HIGH if not recoverable else EventPriority.NORMAL,
                data={
                    "message": str(error),
                    "type": type(error).__name__,
                    "recoverable": recoverable,
                    **(context or {}),
                },
            )
        )

    def emit_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Emit a metric event.

        Args:
            metric_name: Name of the metric.
            value: Metric value.
            unit: Unit of measurement.
            tags: Optional metric tags.
        """
        self.publish(
            VictorEvent(
                category=EventCategory.METRIC,
                name=metric_name,
                data={
                    "value": value,
                    "unit": unit,
                    "tags": tags or {},
                },
            )
        )
