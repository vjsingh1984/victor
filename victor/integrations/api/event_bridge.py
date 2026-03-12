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

"""Event Bridge for Real-Time Updates.

Bridges the internal EventBus to WebSocket clients for real-time updates.
Enables VS Code extension and other clients to receive live updates for:
- Tool execution progress
- File changes
- Provider switching
- Error notifications
- Metric updates

Architecture:
    EventBus (internal) → EventBridge → WebSocket → Clients
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from victor.core.events import ObservabilityBus as EventBus, MessagingEvent

logger = logging.getLogger(__name__)


class BridgeEventType(str, Enum):
    """Types of events that can be bridged to clients."""

    # Tool events
    TOOL_START = "tool.start"
    TOOL_PROGRESS = "tool.progress"
    TOOL_COMPLETE = "tool.complete"
    TOOL_ERROR = "tool.error"

    # File events
    FILE_CREATED = "file.created"
    FILE_MODIFIED = "file.modified"
    FILE_DELETED = "file.deleted"

    # Provider events
    PROVIDER_SWITCH = "provider.switch"
    PROVIDER_ERROR = "provider.error"
    PROVIDER_RECOVERY = "provider.recovery"

    # Session events
    SESSION_START = "session.start"
    SESSION_END = "session.end"
    SESSION_ERROR = "session.error"

    # Metrics events
    METRICS_UPDATE = "metrics.update"
    BUDGET_WARNING = "budget.warning"
    BUDGET_EXHAUSTED = "budget.exhausted"

    # General events
    NOTIFICATION = "notification"
    ERROR = "error"


@dataclass
class BridgeEvent:
    """Event to be sent to clients."""

    type: BridgeEventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class ClientConnection:
    """Represents a connected WebSocket client."""

    id: str
    send: Callable[[str], None]  # Async send function
    subscriptions: Set[str] = field(default_factory=lambda: {"*"})  # Event types subscribed to
    correlation_id: Optional[str] = None
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def is_subscribed(self, event_type: str) -> bool:
        """Check if client is subscribed to event type."""
        return event_type in self.subscriptions or "*" in self.subscriptions

    def accepts(self, event: BridgeEvent) -> bool:
        """Check whether the client should receive this event."""
        if not self.is_subscribed(event.type.value):
            return False

        if not self.correlation_id:
            return True

        request_id = event.data.get("request_id")
        if isinstance(request_id, str) and request_id == self.correlation_id:
            return True

        correlation_id = event.data.get("correlation_id")
        return isinstance(correlation_id, str) and correlation_id == self.correlation_id


class EventBroadcaster:
    """Broadcasts events to connected WebSocket clients.

    Singleton that manages client connections and event distribution.
    """

    _instance: Optional["EventBroadcaster"] = None

    def __new__(cls) -> "EventBroadcaster":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the broadcaster."""
        if self._initialized:
            return

        self._clients: Dict[str, ClientConnection] = {}
        self._event_queue: asyncio.Queue[BridgeEvent] = asyncio.Queue()
        self._recent_events: deque[BridgeEvent] = deque(maxlen=500)
        self._broadcast_task: Optional[asyncio.Task] = None
        self._dispatch_latency_ms_window: deque[float] = deque(maxlen=2000)
        self._client_send_success_count = 0
        self._client_send_failure_count = 0
        self._events_dispatched_count = 0
        self._delivery_success_slo = 0.999
        self._dispatch_latency_p95_slo_ms = 200.0
        self._last_slo_breach_log_ts = 0.0
        self._slo_breach_log_interval_sec = 30.0
        self._running = False
        self._initialized = True

    async def start(self) -> None:
        """Start the broadcast loop."""
        if self._running:
            return

        self._running = True
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        logger.info("EventBroadcaster started")

    async def stop(self) -> None:
        """Stop the broadcast loop."""
        self._running = False
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        logger.info("EventBroadcaster stopped")

    def add_client(
        self,
        client_id: str,
        send_func: Callable[[str], None],
        subscriptions: Optional[Set[str]] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Add a connected client."""
        self._clients[client_id] = ClientConnection(
            id=client_id,
            send=send_func,
            subscriptions=subscriptions or {"*"},
            correlation_id=correlation_id,
        )
        logger.info(f"Client connected: {client_id}")

    def remove_client(self, client_id: str) -> None:
        """Remove a disconnected client."""
        if client_id in self._clients:
            del self._clients[client_id]
            logger.info(f"Client disconnected: {client_id}")

    def update_subscriptions(
        self,
        client_id: str,
        subscriptions: Set[str],
        correlation_id: Optional[str] = None,
    ) -> None:
        """Update client's event subscriptions."""
        if client_id in self._clients:
            self._clients[client_id].subscriptions = subscriptions
            self._clients[client_id].correlation_id = correlation_id

    @staticmethod
    def normalize_subscriptions(values: Optional[List[str] | Set[str]]) -> Set[str]:
        """Normalize incoming subscription names for internal matching."""
        if not values:
            return {"*"}

        normalized = set()
        for value in values:
            if value in ("all", "*"):
                normalized.add("*")
            elif value:
                normalized.add(value)

        return normalized or {"*"}

    @property
    def client_count(self) -> int:
        """Get number of connected clients."""
        return len(self._clients)

    async def broadcast(self, event: BridgeEvent) -> None:
        """Queue an event for broadcast."""
        await self._event_queue.put(event)
        self._recent_events.append(event)

    def broadcast_sync(self, event: BridgeEvent) -> None:
        """Queue an event for broadcast (sync version)."""
        try:
            self._event_queue.put_nowait(event)
            self._recent_events.append(event)
        except asyncio.QueueFull:
            logger.warning("Event queue full, dropping event")

    def get_recent_events(
        self,
        *,
        limit: int = 20,
        subscriptions: Optional[Set[str]] = None,
        correlation_id: Optional[str] = None,
    ) -> List[BridgeEvent]:
        """Return recent events, newest first, optionally filtered."""
        normalized = self.normalize_subscriptions(subscriptions)
        probe = ClientConnection(
            id="snapshot",
            send=lambda _message: None,
            subscriptions=normalized,
            correlation_id=correlation_id,
        )
        matched = [event for event in reversed(self._recent_events) if probe.accepts(event)]
        return matched[: max(1, limit)]

    async def _broadcast_loop(self) -> None:
        """Main broadcast loop."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._send_to_clients(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Broadcast error: {e}")

    async def _send_to_clients(self, event: BridgeEvent) -> None:
        """Send event to all subscribed clients."""
        event_json = event.to_json()
        disconnected = []

        for client_id, client in self._clients.items():
            if client.accepts(event):
                try:
                    send_result = client.send(event_json)
                    if inspect.isawaitable(send_result):
                        await asyncio.wait_for(send_result, timeout=5.0)
                    client.last_activity = time.time()
                    self._client_send_success_count += 1
                except Exception as e:
                    logger.warning(f"Failed to send to {client_id}: {e}")
                    self._client_send_failure_count += 1
                    disconnected.append(client_id)

        # Remove disconnected clients
        for client_id in disconnected:
            self.remove_client(client_id)

        self._events_dispatched_count += 1
        dispatch_latency_ms = max(0.0, (time.time() - event.timestamp) * 1000.0)
        self._dispatch_latency_ms_window.append(dispatch_latency_ms)
        self._maybe_log_slo_breaches()

    def get_reliability_dashboard(self) -> Dict[str, Any]:
        """Get event-bridge reliability metrics and SLO status."""
        total_send_attempts = self._client_send_success_count + self._client_send_failure_count
        delivery_success_rate = (
            self._client_send_success_count / total_send_attempts if total_send_attempts else 1.0
        )
        dispatch_latency_p95_ms = self._percentile(self._dispatch_latency_ms_window, 95.0)

        return {
            "events_dispatched": self._events_dispatched_count,
            "total_send_attempts": total_send_attempts,
            "send_successes": self._client_send_success_count,
            "send_failures": self._client_send_failure_count,
            "delivery_success_rate": delivery_success_rate,
            "dispatch_latency_p95_ms": dispatch_latency_p95_ms,
            "slo_thresholds": {
                "delivery_success_rate_min": self._delivery_success_slo,
                "dispatch_latency_p95_ms_max": self._dispatch_latency_p95_slo_ms,
            },
            "slo_status": {
                "delivery_success_rate": delivery_success_rate >= self._delivery_success_slo,
                "dispatch_latency_p95_ms": dispatch_latency_p95_ms
                <= self._dispatch_latency_p95_slo_ms,
            },
        }

    @staticmethod
    def _percentile(values: Any, percentile: float) -> float:
        """Compute percentile using linear interpolation."""
        values_list = list(values)
        if not values_list:
            return 0.0
        ordered = sorted(values_list)
        if len(ordered) == 1:
            return float(ordered[0])

        rank = (percentile / 100.0) * (len(ordered) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(ordered) - 1)
        weight = rank - lower
        return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)

    def _maybe_log_slo_breaches(self) -> None:
        """Log reliability SLO breaches with basic throttling."""
        snapshot = self.get_reliability_dashboard()
        if snapshot["total_send_attempts"] == 0:
            return

        breaches = []
        if not snapshot["slo_status"]["delivery_success_rate"]:
            breaches.append(
                "delivery_success_rate "
                f"{snapshot['delivery_success_rate']:.4f} < "
                f"{snapshot['slo_thresholds']['delivery_success_rate_min']:.4f}"
            )
        if not snapshot["slo_status"]["dispatch_latency_p95_ms"]:
            breaches.append(
                "dispatch_latency_p95_ms "
                f"{snapshot['dispatch_latency_p95_ms']:.2f} > "
                f"{snapshot['slo_thresholds']['dispatch_latency_p95_ms_max']:.2f}"
            )
        if not breaches:
            return

        now = time.time()
        if now - self._last_slo_breach_log_ts < self._slo_breach_log_interval_sec:
            return
        self._last_slo_breach_log_ts = now
        logger.warning("EventBridge SLO breach: %s", "; ".join(breaches))


class WebSocketEventHandler:
    """Handles WebSocket event subscriptions and messaging.

    Provides a simple interface for WebSocket handlers to integrate
    with the EventBroadcaster.
    """

    def __init__(self, broadcaster: Optional[EventBroadcaster] = None):
        """Initialize handler."""
        self._broadcaster = broadcaster or EventBroadcaster()

    async def handle_connection(
        self,
        websocket,  # WebSocket connection object
        client_id: Optional[str] = None,
    ) -> None:
        """Handle a new WebSocket connection.

        Args:
            websocket: WebSocket connection object with send/recv methods
            client_id: Optional client identifier
        """
        client_id = client_id or uuid.uuid4().hex[:12]

        # Add client
        self._broadcaster.add_client(
            client_id,
            websocket.send,
        )

        try:
            # Handle incoming messages
            async for message in websocket:
                await self._handle_message(client_id, message)
        finally:
            self._broadcaster.remove_client(client_id)

    async def _handle_message(self, client_id: str, message: str) -> None:
        """Handle an incoming message from a client."""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type == "subscribe":
                # Update subscriptions
                requested = data.get("categories") or data.get("events") or ["all"]
                subscriptions = self._broadcaster.normalize_subscriptions(requested)
                correlation_id = data.get("correlation_id")
                self._broadcaster.update_subscriptions(
                    client_id,
                    subscriptions,
                    correlation_id=correlation_id if isinstance(correlation_id, str) else None,
                )

            elif msg_type == "unsubscribe":
                # Remove subscriptions
                self._broadcaster.update_subscriptions(client_id, set(), correlation_id=None)

            elif msg_type == "ping":
                # Respond with pong
                await self._broadcaster._clients[client_id].send(
                    json.dumps({"type": "pong", "timestamp": time.time()})
                )

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from {client_id}")
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")


class EventBusAdapter:
    """Adapter to bridge internal EventBus events to the broadcaster.

    Subscribes to EventBus events and converts them to BridgeEvents
    for broadcasting to WebSocket clients.

    M1 Async Path:
    - Always uses async subscribe path
    - Async methods (connect_async, disconnect_async) are the primary APIs
    - Sync methods kept for backward compatibility but delegate to async
    """

    # Mapping of internal event topics to bridge event types
    # Maps topic prefixes from canonical event system to bridge events
    EVENT_MAPPING = {
        "tool.start": BridgeEventType.TOOL_START,
        "tool.progress": BridgeEventType.TOOL_PROGRESS,
        "tool.complete": BridgeEventType.TOOL_COMPLETE,
        "tool.error": BridgeEventType.TOOL_ERROR,
        "file.modified": BridgeEventType.FILE_MODIFIED,
        "provider.switch": BridgeEventType.PROVIDER_SWITCH,
        "provider.error": BridgeEventType.PROVIDER_ERROR,
        "lifecycle.session.start": BridgeEventType.SESSION_START,
        "lifecycle.session.end": BridgeEventType.SESSION_END,
        "metrics.update": BridgeEventType.METRICS_UPDATE,
        "budget.warning": BridgeEventType.BUDGET_WARNING,
    }

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        broadcaster: Optional[EventBroadcaster] = None,
    ):
        """Initialize adapter."""
        self._event_bus = event_bus
        self._broadcaster = broadcaster or EventBroadcaster()
        self._subscriptions: List[str] = []
        self._subscription_handles: List[Any] = []
        self._pending_async_tasks: Set[asyncio.Task[Any]] = set()
        self._disconnect_requested = False

    async def connect_async(self, event_bus: EventBus) -> None:
        """Connect to an EventBus and subscribe to events (async path).

        This is the primary connection method. All subscriptions use
        the async subscribe API and are properly awaited.

        Args:
            event_bus: The EventBus to connect to (must support async subscribe)

        Raises:
            RuntimeError: If the event bus doesn't have an async subscribe method
        """
        self._event_bus = event_bus
        self._disconnect_requested = False

        subscribe = getattr(event_bus, "subscribe", None)
        if not callable(subscribe):
            logger.warning("EventBusAdapter connect failed: event bus has no subscribe() method")
            raise RuntimeError("EventBus has no subscribe() method")

        # Verify subscribe is async (M1 requirement)
        if not inspect.iscoroutinefunction(subscribe):
            logger.warning(
                "EventBusAdapter connect failed: event bus does not support async subscribe. "
                "Use ObservabilityBus or other async-compatible event bus."
            )
            raise RuntimeError("EventBus must support async subscribe()")

        subscriptions_created = 0
        for pattern in self.EVENT_MAPPING.keys():
            try:
                # Always use async handler for modern event bus
                handle = await subscribe(pattern, self._on_event_async)
                self._subscriptions.append(pattern)
                self._subscription_handles.append(handle)
                subscriptions_created += 1
            except Exception as e:
                logger.warning(f"Failed to subscribe to {pattern}: {e}")
                # Continue with other subscriptions

        logger.info(
            f"EventBusAdapter connected via async path "
            f"({subscriptions_created}/{len(self.EVENT_MAPPING)} subscriptions)"
        )

    def connect(self, event_bus: EventBus) -> None:
        """Connect to an EventBus (sync wrapper for backward compatibility).

        Deprecated: Use connect_async() instead. This method is kept
        for backward compatibility and may fire-and-forget the async operation.

        Args:
            event_bus: The EventBus to connect to
        """
        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, schedule the connection
            self._run_async_operation(
                self.connect_async(event_bus),
                description="connect to event bus",
            )
        except RuntimeError:
            # No running loop, try to run it
            try:
                asyncio.run(self.connect_async(event_bus))
            except Exception as e:
                logger.warning(f"EventBusAdapter sync connect failed: {e}")

    async def disconnect_async(self) -> None:
        """Disconnect from the EventBus (async path).

        This is the primary disconnect method. All unsubscribes use
        the async API and are properly awaited.

        Ensures all subscription handles are properly cleaned up
        before returning.
        """
        if not self._event_bus:
            return

        self._disconnect_requested = True
        unsubscribed_count = 0

        # Unsubscribe from all handles (async path)
        for handle in list(self._subscription_handles):
            unsubscribe = getattr(handle, "unsubscribe", None)
            if callable(unsubscribe):
                try:
                    result = unsubscribe()
                    if inspect.isawaitable(result):
                        await result
                    unsubscribed_count += 1
                except Exception as e:
                    logger.debug(f"Failed to unsubscribe from handle: {e}")

        self._subscription_handles.clear()
        self._subscriptions.clear()

        # Wait for any pending async operations to complete
        if self._pending_async_tasks:
            pending = list(self._pending_async_tasks)
            if pending:
                # Wait up to 5 seconds for pending tasks
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timeout waiting for {len(pending)} pending async tasks "
                        "during disconnect"
                    )

        logger.info(
            f"EventBusAdapter disconnected via async path "
            f"({unsubscribed_count} handles cleaned up)"
        )

    def disconnect(self) -> None:
        """Disconnect from the EventBus (sync wrapper for backward compatibility).

        Deprecated: Use disconnect_async() instead. This method is kept
        for backward compatibility and may fire-and-forget the async operation.
        """
        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, schedule the disconnection
            self._run_async_operation(
                self.disconnect_async(),
                description="disconnect from event bus",
            )
        except RuntimeError:
            # No running loop, try to run it
            try:
                asyncio.run(self.disconnect_async())
            except Exception as e:
                logger.warning(f"EventBusAdapter sync disconnect failed: {e}")

    def _on_event(self, event: MessagingEvent) -> None:
        """Handle an internal EventBus event."""
        # Map event topic to bridge event type
        bridge_type = self.EVENT_MAPPING.get(event.topic)
        if not bridge_type:
            return

        bridge_data = dict(event.data)
        if event.correlation_id and "correlation_id" not in bridge_data:
            bridge_data["correlation_id"] = event.correlation_id

        bridge_event = BridgeEvent(
            type=bridge_type,
            data=bridge_data,
        )

        self._broadcaster.broadcast_sync(bridge_event)

    async def _on_event_async(self, event: MessagingEvent) -> None:
        """Async wrapper for event backends that require awaitable handlers."""
        self._on_event(event)

    def _track_subscription(self, topic: str, handle: Any = None) -> None:
        """Track subscribed topics and optional async subscription handles."""
        self._subscriptions.append(topic)
        if handle is not None:
            self._subscription_handles.append(handle)

    def _track_subscription_handle(self, handle: Any) -> None:
        """Track an async subscription handle once subscribe() resolves."""
        if handle is None:
            return

        if self._disconnect_requested:
            unsubscribe = getattr(handle, "unsubscribe", None)
            if callable(unsubscribe):
                try:
                    result = unsubscribe()
                    if inspect.isawaitable(result):
                        self._run_async_operation(
                            result,
                            description="unsubscribe handle after disconnect",
                        )
                except Exception:
                    pass
            return

        self._subscription_handles.append(handle)

    def _run_async_operation(
        self,
        awaitable: Any,
        *,
        description: str,
        on_success: Optional[Callable[[Any], None]] = None,
    ) -> None:
        """Run an async operation from a sync code path."""
        try:
            asyncio.get_running_loop()  # Check if loop is running
        except RuntimeError:
            try:
                result = asyncio.run(awaitable)
                if on_success:
                    on_success(result)
            except Exception as e:
                logger.debug(f"Failed to {description}: {e}")
            return

        task = asyncio.ensure_future(awaitable)
        self._pending_async_tasks.add(task)

        def _on_done(done_task: asyncio.Task[Any]) -> None:
            self._pending_async_tasks.discard(done_task)
            try:
                result = done_task.result()
                if on_success:
                    on_success(result)
            except asyncio.CancelledError:
                logger.debug(f"Cancelled task while trying to {description}")
            except Exception as e:
                logger.debug(f"Failed to {description}: {e}")

        task.add_done_callback(_on_done)


# Convenience functions for common events
def emit_tool_start(
    tool_name: str,
    arguments: Dict[str, Any],
    tool_id: Optional[str] = None,
) -> None:
    """Emit a tool start event."""
    broadcaster = EventBroadcaster()
    broadcaster.broadcast_sync(
        BridgeEvent(
            type=BridgeEventType.TOOL_START,
            data={
                "tool_id": tool_id or uuid.uuid4().hex[:12],
                "name": tool_name,
                "arguments": arguments,
            },
        )
    )


def emit_tool_complete(
    tool_id: str,
    result: str,
    duration_ms: Optional[int] = None,
    follow_up_suggestions: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Emit a tool complete event."""
    data: Dict[str, Any] = {
        "tool_id": tool_id,
        "result": result,
        "duration_ms": duration_ms,
    }
    if follow_up_suggestions:
        data["follow_up_suggestions"] = follow_up_suggestions

    broadcaster = EventBroadcaster()
    broadcaster.broadcast_sync(
        BridgeEvent(
            type=BridgeEventType.TOOL_COMPLETE,
            data=data,
        )
    )


def emit_tool_error(
    tool_id: str,
    error: str,
) -> None:
    """Emit a tool error event."""
    broadcaster = EventBroadcaster()
    broadcaster.broadcast_sync(
        BridgeEvent(
            type=BridgeEventType.TOOL_ERROR,
            data={
                "tool_id": tool_id,
                "error": error,
            },
        )
    )


def emit_notification(
    message: str,
    level: str = "info",
) -> None:
    """Emit a notification event."""
    broadcaster = EventBroadcaster()
    broadcaster.broadcast_sync(
        BridgeEvent(
            type=BridgeEventType.NOTIFICATION,
            data={
                "message": message,
                "level": level,
            },
        )
    )


class EventBridge:
    """Main EventBridge interface for bridging EventBus to WebSocket clients.

    Provides a unified interface for:
    - Starting/stopping the bridge
    - Connecting to EventBus
    - Broadcasting events to WebSocket clients

    M1 Async Path:
    - async_start() and async_stop() are the primary APIs
    - start() and stop() are sync wrappers for backward compatibility

    Example:
        bus = get_event_bus()
        bridge = EventBridge(bus)
        await bridge.async_start()

        # ... later
        await bridge.async_stop()
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        """Initialize the EventBridge.

        Args:
            event_bus: Optional EventBus to connect to immediately
        """
        self._event_bus = event_bus
        self._broadcaster = EventBroadcaster()
        self._adapter = EventBusAdapter(event_bus, self._broadcaster)
        self._running = False

    async def async_start(self) -> None:
        """Start the EventBridge (async path).

        This is the primary start method. Uses async connect and properly
        awaits all operations.

        Connects to the EventBus and begins broadcasting events.
        """
        if self._running:
            return

        if self._event_bus:
            await self._adapter.connect_async(self._event_bus)

        await self._broadcaster.start()
        self._running = True
        logger.info("EventBridge started via async path")

    def start(self) -> None:
        """Start the EventBridge (sync wrapper for backward compatibility).

        Deprecated: Use async_start() instead. This method is kept
        for backward compatibility and may fire-and-forget the async operation.
        """
        if self._running:
            return

        # Optimistically set running flag for immediate feedback
        # (backward compatibility: old code expects start() to set _running)
        self._running = True

        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, schedule the start
            self._run_async_operation(
                self._start_and_set_flag(),
                description="start event bridge",
            )
        except RuntimeError:
            # No running loop, try to run it
            try:
                asyncio.run(self._start_and_set_flag())
            except Exception as e:
                self._running = False  # Reset on failure
                logger.warning(f"EventBridge sync start failed: {e}")

    async def _start_and_set_flag(self) -> None:
        """Internal helper that performs async start without early return check.

        This is called by the sync start() method after setting _running=True,
        so it needs to do the actual work without the early return.
        """
        try:
            if self._event_bus:
                await self._adapter.connect_async(self._event_bus)
            await self._broadcaster.start()
            logger.info("EventBridge started via sync wrapper")
        except Exception:
            self._running = False
            raise

    async def async_stop(self) -> None:
        """Stop the EventBridge (async path).

        This is the primary stop method. Uses async disconnect and properly
        awaits all cleanup operations.

        Disconnects from EventBus and stops broadcasting.
        """
        if not self._running:
            return

        await self._adapter.disconnect_async()
        await self._broadcaster.stop()
        self._running = False
        logger.info("EventBridge stopped via async path")

    def stop(self) -> None:
        """Stop the EventBridge (sync wrapper for backward compatibility).

        Deprecated: Use async_stop() instead. This method is kept
        for backward compatibility and may fire-and-forget the async operation.
        """
        if not self._running:
            return

        # Optimistically clear running flag for immediate feedback
        self._running = False

        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, schedule the stop
            self._run_async_operation(
                self._stop_and_cleanup(),
                description="stop event bridge",
            )
        except RuntimeError:
            # No running loop, try to run it
            try:
                asyncio.run(self._stop_and_cleanup())
            except Exception as e:
                logger.warning(f"EventBridge sync stop failed: {e}")

    async def _stop_and_cleanup(self) -> None:
        """Internal helper that performs async cleanup without early return check.

        This is called by the sync stop() method after setting _running=False,
        so it needs to do the actual cleanup without the early return.
        """
        try:
            await self._adapter.disconnect_async()
            await self._broadcaster.stop()
            logger.info("EventBridge stopped via sync wrapper")
        except Exception:
            raise

    async def handle_connection(
        self,
        websocket,
        client_id: Optional[str] = None,
    ) -> None:
        """Handle a new WebSocket connection.

        Args:
            websocket: WebSocket connection object
            client_id: Optional client identifier
        """
        handler = WebSocketEventHandler(self._broadcaster)
        await handler.handle_connection(websocket, client_id)

    def get_reliability_dashboard_data(self) -> Dict[str, Any]:
        """Expose reliability metrics and SLO status for dashboards."""
        return self._broadcaster.get_reliability_dashboard()

    def _run_async_operation(self, awaitable: Any, *, description: str) -> None:
        """Run async broadcaster lifecycle operations from sync APIs."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(awaitable)
            except Exception as e:
                logger.debug(f"Failed to {description}: {e}")
            return

        task = asyncio.ensure_future(awaitable)

        def _on_done(done_task: asyncio.Task[Any]) -> None:
            try:
                done_task.result()
            except asyncio.CancelledError:
                logger.debug(f"Cancelled task while trying to {description}")
            except Exception as e:
                logger.debug(f"Failed to {description}: {e}")

        task.add_done_callback(_on_done)
