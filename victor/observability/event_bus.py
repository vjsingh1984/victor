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
import time
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
    VERTICAL = "vertical"  # Vertical integration events
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


# =============================================================================
# Event Sampling Configuration (Phase 3 - Scalability)
# =============================================================================


@dataclass
class SamplingConfig:
    """Configuration for event sampling per category.

    Sampling reduces high-volume event load while preserving important events.
    A sampling rate of 1.0 means all events pass through; 0.1 means 10% pass.

    Attributes:
        rates: Sampling rates per category (0.0 to 1.0)
        default_rate: Default rate for unconfigured categories
        preserve_errors: Always emit ERROR category events (sampling=1.0)
        preserve_critical: Always emit CRITICAL priority events
    """

    rates: Dict[EventCategory, float] = field(default_factory=dict)
    default_rate: float = 1.0
    preserve_errors: bool = True
    preserve_critical: bool = True

    def __post_init__(self) -> None:
        """Validate sampling rates."""
        for category, rate in self.rates.items():
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"Sampling rate for {category} must be between 0.0 and 1.0")
        if not 0.0 <= self.default_rate <= 1.0:
            raise ValueError("Default sampling rate must be between 0.0 and 1.0")

    def get_rate(self, category: EventCategory) -> float:
        """Get sampling rate for a category.

        Args:
            category: Event category

        Returns:
            Sampling rate (0.0 to 1.0)
        """
        if self.preserve_errors and category == EventCategory.ERROR:
            return 1.0
        return self.rates.get(category, self.default_rate)

    def should_sample(self, event: VictorEvent) -> bool:
        """Determine if an event should be sampled (emitted).

        Uses deterministic sampling based on event ID for consistency.

        Args:
            event: Event to check

        Returns:
            True if event should be emitted
        """
        # Always emit critical priority events
        if self.preserve_critical and event.priority == EventPriority.CRITICAL:
            return True

        rate = self.get_rate(event.category)
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False

        # Deterministic sampling based on event ID hash
        event_hash = hash(event.id) & 0xFFFFFFFF
        threshold = int(rate * 0xFFFFFFFF)
        return event_hash < threshold


@dataclass
class SamplingMetrics:
    """Metrics for event sampling.

    Attributes:
        events_sampled: Events that passed sampling
        events_dropped: Events dropped by sampling
        by_category: Breakdown by category
    """

    events_sampled: int = 0
    events_dropped: int = 0
    by_category: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def record(self, category: EventCategory, sampled: bool) -> None:
        """Record a sampling decision.

        Args:
            category: Event category
            sampled: Whether event was sampled (emitted)
        """
        if sampled:
            self.events_sampled += 1
        else:
            self.events_dropped += 1

        cat_name = category.value
        if cat_name not in self.by_category:
            self.by_category[cat_name] = {"sampled": 0, "dropped": 0}
        if sampled:
            self.by_category[cat_name]["sampled"] += 1
        else:
            self.by_category[cat_name]["dropped"] += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to dictionary."""
        return {
            "events_sampled": self.events_sampled,
            "events_dropped": self.events_dropped,
            "by_category": self.by_category,
        }


# =============================================================================
# Event Batching Configuration (Phase 3 - Scalability)
# =============================================================================


@dataclass
class BatchConfig:
    """Configuration for event batching.

    Batching groups events before sending to exporters, reducing overhead
    for high-volume event streams.

    Attributes:
        enabled: Whether batching is enabled
        batch_size: Maximum events per batch
        flush_interval_ms: Maximum time to hold events (milliseconds)
        categories: Categories to batch (empty means all)
    """

    enabled: bool = False
    batch_size: int = 100
    flush_interval_ms: float = 1000.0
    categories: Set[EventCategory] = field(default_factory=set)

    def should_batch(self, event: VictorEvent) -> bool:
        """Check if an event should be batched.

        Args:
            event: Event to check

        Returns:
            True if event should be batched
        """
        if not self.enabled:
            return False
        # Empty categories means batch all
        if not self.categories:
            return True
        return event.category in self.categories


class EventBatcher:
    """Batches events for efficient bulk export.

    Groups events by category and flushes when batch size or time
    threshold is reached.

    Thread-safe for concurrent event submission.
    """

    def __init__(self, config: BatchConfig) -> None:
        """Initialize the batcher.

        Args:
            config: Batch configuration
        """
        self._config = config
        self._batches: Dict[EventCategory, List[VictorEvent]] = {}
        self._last_flush: Dict[EventCategory, float] = {}
        self._lock = threading.Lock()
        self._flush_callbacks: List[Callable[[List[VictorEvent]], None]] = []

    def add_event(self, event: VictorEvent) -> Optional[List[VictorEvent]]:
        """Add an event to the batch.

        Args:
            event: Event to batch

        Returns:
            List of events if batch is ready to flush, None otherwise
        """
        with self._lock:
            category = event.category

            if category not in self._batches:
                self._batches[category] = []
                self._last_flush[category] = time.time() * 1000

            self._batches[category].append(event)

            # Check if batch should flush
            if self._should_flush(category):
                return self._flush_category(category)

        return None

    def _should_flush(self, category: EventCategory) -> bool:
        """Check if a category's batch should flush.

        Args:
            category: Category to check

        Returns:
            True if batch should flush
        """
        batch = self._batches.get(category, [])

        # Size threshold
        if len(batch) >= self._config.batch_size:
            return True

        # Time threshold
        now_ms = time.time() * 1000
        last_flush = self._last_flush.get(category, now_ms)
        if now_ms - last_flush >= self._config.flush_interval_ms:
            return True

        return False

    def _flush_category(self, category: EventCategory) -> List[VictorEvent]:
        """Flush a category's batch.

        Must be called with lock held.

        Args:
            category: Category to flush

        Returns:
            List of flushed events
        """
        events = self._batches.get(category, [])
        self._batches[category] = []
        self._last_flush[category] = time.time() * 1000
        return events

    def flush_all(self) -> Dict[EventCategory, List[VictorEvent]]:
        """Flush all batches.

        Returns:
            Dictionary mapping categories to their flushed events
        """
        with self._lock:
            result: Dict[EventCategory, List[VictorEvent]] = {}
            for category in list(self._batches.keys()):
                events = self._flush_category(category)
                if events:
                    result[category] = events
            return result

    def get_pending_count(self) -> int:
        """Get total pending events across all batches.

        Returns:
            Number of pending events
        """
        with self._lock:
            return sum(len(batch) for batch in self._batches.values())


# =============================================================================
# Exporter Priority Filter (Phase 3 - Scalability)
# =============================================================================


@dataclass
class ExporterConfig:
    """Configuration for an exporter's event filtering.

    Allows per-exporter control over which events are received.

    Attributes:
        min_priority: Minimum priority to export
        categories: Categories to export (empty means all)
        exclude_categories: Categories to exclude
        sampling_override: Optional per-exporter sampling config
    """

    min_priority: EventPriority = EventPriority.LOW
    categories: Set[EventCategory] = field(default_factory=set)
    exclude_categories: Set[EventCategory] = field(default_factory=set)
    sampling_override: Optional[SamplingConfig] = None

    def should_export(self, event: VictorEvent) -> bool:
        """Check if event should be sent to this exporter.

        Args:
            event: Event to check

        Returns:
            True if event should be exported
        """
        # Check priority
        if event.priority.value < self.min_priority.value:
            return False

        # Check category inclusion
        if self.categories and event.category not in self.categories:
            return False

        # Check category exclusion
        if event.category in self.exclude_categories:
            return False

        # Check sampling override
        if self.sampling_override and not self.sampling_override.should_sample(event):
            return False

        return True


# =============================================================================
# Centralized EventBus Configuration (P1 Scalability)
# =============================================================================


@dataclass
class EventBusConfig:
    """Centralized configuration for EventBus.

    Combines backpressure, sampling, and batching configuration into a single
    object that can be created from Settings for consistent initialization.

    Attributes:
        queue_maxsize: Maximum queue size for backpressure
        backpressure_strategy: Strategy when queue is full
        sampling_config: Optional sampling configuration
        batch_config: Optional batching configuration
    """

    queue_maxsize: int = 10000
    backpressure_strategy: BackpressureStrategy = BackpressureStrategy.DROP_OLDEST
    sampling_config: Optional[SamplingConfig] = None
    batch_config: Optional[BatchConfig] = None

    @classmethod
    def from_settings(cls, settings: Any) -> "EventBusConfig":
        """Create EventBusConfig from a Settings instance.

        This factory method extracts all EventBus-related settings and
        constructs a unified configuration object.

        Args:
            settings: Victor Settings instance

        Returns:
            Configured EventBusConfig

        Example:
            from victor.config.settings import load_settings

            settings = load_settings()
            config = EventBusConfig.from_settings(settings)
            bus = EventBus(
                queue_maxsize=config.queue_maxsize,
                backpressure_strategy=config.backpressure_strategy,
                sampling_config=config.sampling_config,
                batch_config=config.batch_config,
            )
        """
        # Parse backpressure strategy
        strategy_str = getattr(settings, "eventbus_backpressure_strategy", "drop_oldest")
        try:
            strategy = BackpressureStrategy(strategy_str)
        except ValueError:
            logger.warning(f"Invalid backpressure strategy '{strategy_str}', using DROP_OLDEST")
            strategy = BackpressureStrategy.DROP_OLDEST

        # Build sampling config if enabled
        sampling_config = None
        if getattr(settings, "eventbus_sampling_enabled", False):
            default_rate = getattr(settings, "eventbus_sampling_default_rate", 1.0)
            sampling_config = SamplingConfig(
                default_rate=default_rate,
                preserve_errors=True,
                preserve_critical=True,
            )

        # Build batch config if enabled
        batch_config = None
        if getattr(settings, "eventbus_batching_enabled", False):
            batch_config = BatchConfig(
                enabled=True,
                batch_size=getattr(settings, "eventbus_batch_size", 100),
                flush_interval_ms=getattr(settings, "eventbus_batch_flush_interval_ms", 1000.0),
            )

        return cls(
            queue_maxsize=getattr(settings, "eventbus_queue_maxsize", 10000),
            backpressure_strategy=strategy,
            sampling_config=sampling_config,
            batch_config=batch_config,
        )

    def apply_to_bus(self, bus: "EventBus") -> None:
        """Apply this configuration to an EventBus instance.

        Useful for reconfiguring an existing EventBus at runtime.

        Args:
            bus: EventBus instance to configure
        """
        bus.configure_backpressure(
            strategy=self.backpressure_strategy,
            queue_maxsize=self.queue_maxsize,
        )

        if self.sampling_config:
            bus.configure_sampling(self.sampling_config)
        else:
            bus.disable_sampling()

        if self.batch_config:
            bus.configure_batching(self.batch_config)
        else:
            bus.disable_batching()


def create_eventbus_from_settings(settings: Any) -> "EventBus":
    """Create and configure an EventBus from Settings.

    This is the recommended way to create an EventBus with consistent
    configuration from the application settings.

    Args:
        settings: Victor Settings instance

    Returns:
        Configured EventBus instance

    Example:
        from victor.config.settings import load_settings
        from victor.observability.event_bus import create_eventbus_from_settings

        settings = load_settings()
        bus = create_eventbus_from_settings(settings)
    """
    config = EventBusConfig.from_settings(settings)
    return EventBus(
        queue_maxsize=config.queue_maxsize,
        backpressure_strategy=config.backpressure_strategy,
        sampling_config=config.sampling_config,
        batch_config=config.batch_config,
    )


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
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
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
        sampling_config: Optional[SamplingConfig] = None,
        batch_config: Optional[BatchConfig] = None,
    ) -> None:
        """Initialize the event bus.

        Args:
            queue_maxsize: Maximum queue size (0 for unbounded, not recommended).
            backpressure_strategy: Strategy when queue is full.
            sampling_config: Optional sampling configuration.
            batch_config: Optional batching configuration.
        """
        if getattr(self, "_initialized", False):
            return

        self._subscriptions: Dict[EventCategory, List[Subscription]] = {
            cat: [] for cat in EventCategory
        }
        self._exporters: List["BaseExporter"] = []
        self._exporter_configs: Dict[int, ExporterConfig] = {}  # id(exporter) -> config
        self._queue_maxsize = queue_maxsize
        self._backpressure_strategy = backpressure_strategy
        self._event_queue: asyncio.Queue[VictorEvent] = asyncio.Queue(maxsize=queue_maxsize)
        self._is_processing = False
        self._session_id: Optional[str] = None
        self._trace_context: Dict[str, str] = {}
        self._backpressure_metrics = BackpressureMetrics()
        self._initialized = True
        self._lock = threading.Lock()

        # Track pending async tasks to prevent leaks under stress
        self._pending_tasks: set[asyncio.Task[None]] = set()

        # Rate-limited warning for DROP_OLDEST/DROP_NEWEST (Fix: Workstream E)
        self._last_drop_warning: float = 0
        self._drop_warning_interval: float = 60.0  # seconds - max 1 warning per minute

        # Phase 3: Sampling and batching for scalability
        self._sampling_config: Optional[SamplingConfig] = sampling_config
        self._sampling_metrics = SamplingMetrics()
        self._batch_config: Optional[BatchConfig] = batch_config
        self._batcher: Optional[EventBatcher] = (
            EventBatcher(batch_config) if batch_config and batch_config.enabled else None
        )

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
                # Cancel all pending async tasks to prevent leaks
                pending = getattr(cls._instance, "_pending_tasks", set())
                for task in list(pending):
                    if not task.done():
                        task.cancel()
                pending.clear()

                cls._instance._subscriptions = {cat: [] for cat in EventCategory}
                cls._instance._exporters = []
                cls._instance._exporter_configs = {}
                cls._instance._session_id = None
                cls._instance._trace_context = {}
                cls._instance._backpressure_metrics = BackpressureMetrics()
                cls._instance._pending_tasks = set()
                # Recreate queue with same maxsize
                maxsize = getattr(cls._instance, "_queue_maxsize", 10000)
                cls._instance._event_queue = asyncio.Queue(maxsize=maxsize)
                # Reset sampling/batching (Phase 3)
                cls._instance._sampling_config = None
                cls._instance._sampling_metrics = SamplingMetrics()
                cls._instance._batch_config = None
                cls._instance._batcher = None

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
        Applies sampling if configured.

        Args:
            event: Event to publish.
        """
        # Apply sampling if configured
        if self._sampling_config:
            sampled = self._sampling_config.should_sample(event)
            self._sampling_metrics.record(event.category, sampled)
            if not sampled:
                return  # Event dropped by sampling

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
                        # Queue async handler with tracking to prevent leaks
                        task = asyncio.create_task(subscription.handler(event))  # type: ignore
                        self._pending_tasks.add(task)
                        task.add_done_callback(self._pending_tasks.discard)
                    else:
                        subscription.handler(event)
                except Exception as e:
                    logger.warning(f"Event handler error: {e}")

        # Send to exporters (with per-exporter filtering)
        for exporter in self._exporters:
            try:
                # Check exporter-specific config
                exporter_config = self._exporter_configs.get(id(exporter))
                if exporter_config and not exporter_config.should_export(event):
                    continue
                exporter.export(event)
            except Exception as e:
                logger.warning(f"Exporter error: {e}")

    async def publish_async(self, event: VictorEvent) -> None:
        """Publish an event asynchronously.

        Applies sampling if configured.

        Args:
            event: Event to publish.
        """
        # Apply sampling if configured
        if self._sampling_config:
            sampled = self._sampling_config.should_sample(event)
            self._sampling_metrics.record(event.category, sampled)
            if not sampled:
                return  # Event dropped by sampling

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

        # Send to exporters asynchronously (with per-exporter filtering)
        for exporter in self._exporters:
            try:
                # Check exporter-specific config
                exporter_config = self._exporter_configs.get(id(exporter))
                if exporter_config and not exporter_config.should_export(event):
                    continue
                if hasattr(exporter, "export_async"):
                    await exporter.export_async(event)
                else:
                    exporter.export(event)
            except Exception as e:
                logger.warning(f"Exporter error: {e}")

    def add_exporter(
        self,
        exporter: "BaseExporter",
        config: Optional[ExporterConfig] = None,
    ) -> None:
        """Add an event exporter with optional filtering config.

        Args:
            exporter: Exporter instance.
            config: Optional exporter-specific filtering configuration.
        """
        self._exporters.append(exporter)
        if config:
            self._exporter_configs[id(exporter)] = config

    def remove_exporter(self, exporter: "BaseExporter") -> None:
        """Remove an event exporter.

        Args:
            exporter: Exporter to remove.
        """
        if exporter in self._exporters:
            self._exporters.remove(exporter)
            # Clean up config
            self._exporter_configs.pop(id(exporter), None)

    async def shutdown(self, timeout: float = 5.0) -> int:
        """Gracefully shutdown the event bus.

        Cancels all pending async tasks and waits for them to complete.
        Use this before application exit to prevent task leaks.

        Args:
            timeout: Maximum seconds to wait for tasks to complete.

        Returns:
            Number of tasks that were cancelled.
        """
        cancelled_count = 0
        pending = list(self._pending_tasks)

        if not pending:
            return 0

        # Cancel all pending tasks
        for task in pending:
            if not task.done():
                task.cancel()
                cancelled_count += 1

        # Wait for tasks to complete (with timeout)
        if pending:
            try:
                await asyncio.wait(pending, timeout=timeout)
            except Exception as e:
                logger.debug(f"Error waiting for tasks during shutdown: {e}")

        self._pending_tasks.clear()
        logger.debug(f"EventBus shutdown: cancelled {cancelled_count} pending tasks")
        return cancelled_count

    def get_pending_task_count(self) -> int:
        """Get count of pending async tasks.

        Useful for debugging and monitoring task leaks.

        Returns:
            Number of pending tasks.
        """
        return len(self._pending_tasks)

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
            # Rate-limited warning for DROP_NEWEST (Workstream E fix)
            now = time.time()
            if now - self._last_drop_warning >= self._drop_warning_interval:
                logger.warning("Event bus dropping events (queue full, DROP_NEWEST strategy)")
                self._last_drop_warning = now
            else:
                logger.debug(f"Event dropped (DROP_NEWEST): {event.name}")
            return False

        elif self._backpressure_strategy == BackpressureStrategy.DROP_OLDEST:
            try:
                # Remove oldest event
                dropped = self._event_queue.get_nowait()
                self._backpressure_metrics.events_dropped += 1
                # Rate-limited warning for DROP_OLDEST (Workstream E fix)
                now = time.time()
                if now - self._last_drop_warning >= self._drop_warning_interval:
                    logger.warning("Event bus dropping events (queue full, DROP_OLDEST strategy)")
                    self._last_drop_warning = now
                else:
                    logger.debug(f"Event dropped (DROP_OLDEST): {dropped.name}")
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
            # Rate-limited warning for DROP_NEWEST (Workstream E fix)
            now = time.time()
            if now - self._last_drop_warning >= self._drop_warning_interval:
                logger.warning("Event bus dropping events (queue full, DROP_NEWEST strategy)")
                self._last_drop_warning = now
            else:
                logger.debug(f"Event dropped (DROP_NEWEST): {event.name}")
            return False

        elif self._backpressure_strategy == BackpressureStrategy.DROP_OLDEST:
            try:
                dropped = self._event_queue.get_nowait()
                self._backpressure_metrics.events_dropped += 1
                # Rate-limited warning for DROP_OLDEST (Workstream E fix)
                now = time.time()
                if now - self._last_drop_warning >= self._drop_warning_interval:
                    logger.warning("Event bus dropping events (queue full, DROP_OLDEST strategy)")
                    self._last_drop_warning = now
                else:
                    logger.debug(f"Event dropped (DROP_OLDEST): {dropped.name}")
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

    # =========================================================================
    # Sampling & Batching Configuration (Phase 3 - Scalability)
    # =========================================================================

    def configure_sampling(self, config: SamplingConfig) -> None:
        """Configure event sampling.

        Sampling reduces high-volume event load while preserving important
        events. Use this to control throughput in high-traffic scenarios.

        Args:
            config: Sampling configuration

        Example:
            bus.configure_sampling(SamplingConfig(
                rates={
                    EventCategory.METRIC: 0.1,  # 10% of metrics
                    EventCategory.TOOL: 0.5,    # 50% of tool events
                },
                default_rate=1.0,  # All other categories pass through
                preserve_errors=True,  # Always emit errors
            ))
        """
        self._sampling_config = config
        logger.debug(f"EventBus sampling configured: default_rate={config.default_rate}")

    def disable_sampling(self) -> None:
        """Disable event sampling.

        All events will pass through without sampling.
        """
        self._sampling_config = None
        logger.debug("EventBus sampling disabled")

    def get_sampling_metrics(self) -> SamplingMetrics:
        """Get current sampling metrics.

        Returns:
            SamplingMetrics with counts by category
        """
        return self._sampling_metrics

    def configure_batching(self, config: BatchConfig) -> None:
        """Configure event batching.

        Batching groups events for efficient bulk export, reducing
        overhead for high-volume event streams.

        Args:
            config: Batch configuration

        Example:
            bus.configure_batching(BatchConfig(
                enabled=True,
                batch_size=100,
                flush_interval_ms=1000,
                categories={EventCategory.METRIC},  # Only batch metrics
            ))
        """
        self._batch_config = config
        if config.enabled:
            self._batcher = EventBatcher(config)
        else:
            self._batcher = None
        logger.debug(f"EventBus batching configured: enabled={config.enabled}")

    def disable_batching(self) -> None:
        """Disable event batching.

        Events will be sent immediately without batching.
        """
        self._batch_config = None
        self._batcher = None
        logger.debug("EventBus batching disabled")

    def flush_batches(self) -> Dict[EventCategory, List[VictorEvent]]:
        """Flush all pending batches.

        Use this before shutdown or when immediate delivery is needed.

        Returns:
            Dictionary mapping categories to flushed events
        """
        if self._batcher:
            return self._batcher.flush_all()
        return {}

    def get_batch_pending_count(self) -> int:
        """Get number of events pending in batches.

        Returns:
            Number of pending events (0 if batching disabled)
        """
        if self._batcher:
            return self._batcher.get_pending_count()
        return 0

    def set_exporter_config(
        self,
        exporter: "BaseExporter",
        config: ExporterConfig,
    ) -> None:
        """Set filtering configuration for an exporter.

        Args:
            exporter: The exporter to configure
            config: Filtering configuration

        Example:
            # Only send HIGH+ priority TOOL events to this exporter
            bus.set_exporter_config(my_exporter, ExporterConfig(
                min_priority=EventPriority.HIGH,
                categories={EventCategory.TOOL},
            ))
        """
        if exporter not in self._exporters:
            raise ValueError("Exporter not registered with EventBus")
        self._exporter_configs[id(exporter)] = config

    def get_exporter_config(
        self,
        exporter: "BaseExporter",
    ) -> Optional[ExporterConfig]:
        """Get filtering configuration for an exporter.

        Args:
            exporter: The exporter to query

        Returns:
            ExporterConfig or None if not configured
        """
        return self._exporter_configs.get(id(exporter))

    def emit_lifecycle_event(
        self,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> None:
        """Emit lifecycle events (graph_started, workflow_completed, etc.).

        Lifecycle events track the start/end of major operations like
        workflow executions, graph traversals, and session management.

        Args:
            event_type: Type of lifecycle event (e.g., "graph_started",
                "workflow_completed", "session_started").
            data: Optional event payload with additional context.
            priority: Event priority (default NORMAL).

        Example:
            bus.emit_lifecycle_event("workflow_started", {
                "workflow_id": "deep_research",
                "inputs": {"query": "AI trends"},
            })
        """
        self.publish(
            VictorEvent(
                category=EventCategory.LIFECYCLE,
                name=event_type,
                priority=priority,
                data=data or {},
            )
        )


def get_event_bus() -> EventBus:
    """Factory function for DIP-compliant EventBus access.

    This function provides a Dependency Inversion Principle (DIP) compliant
    way to access the EventBus singleton. Use this instead of directly
    calling EventBus.get_instance() for better testability and loose coupling.

    Returns:
        The singleton EventBus instance.

    Example:
        from victor.observability.event_bus import get_event_bus

        bus = get_event_bus()
        bus.publish(VictorEvent(...))
    """
    return EventBus.get_instance()
