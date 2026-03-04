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
    subscriptions: Set[str] = field(default_factory=set)  # Event types subscribed to
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def is_subscribed(self, event_type: str) -> bool:
        """Check if client is subscribed to event type."""
        if not self.subscriptions:
            return True  # No subscriptions = all events
        return event_type in self.subscriptions or "*" in self.subscriptions


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
        self._broadcast_task: Optional[asyncio.Task] = None
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
    ) -> None:
        """Add a connected client."""
        self._clients[client_id] = ClientConnection(
            id=client_id,
            send=send_func,
            subscriptions=subscriptions or set(),
        )
        logger.info(f"Client connected: {client_id}")

    def remove_client(self, client_id: str) -> None:
        """Remove a disconnected client."""
        if client_id in self._clients:
            del self._clients[client_id]
            logger.info(f"Client disconnected: {client_id}")

    def update_subscriptions(self, client_id: str, subscriptions: Set[str]) -> None:
        """Update client's event subscriptions."""
        if client_id in self._clients:
            self._clients[client_id].subscriptions = subscriptions

    @property
    def client_count(self) -> int:
        """Get number of connected clients."""
        return len(self._clients)

    async def broadcast(self, event: BridgeEvent) -> None:
        """Queue an event for broadcast."""
        await self._event_queue.put(event)

    def broadcast_sync(self, event: BridgeEvent) -> None:
        """Queue an event for broadcast (sync version)."""
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Event queue full, dropping event")

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
            if client.is_subscribed(event.type.value):
                try:
                    send_result = client.send(event_json)
                    if inspect.isawaitable(send_result):
                        await asyncio.wait_for(send_result, timeout=5.0)
                    client.last_activity = time.time()
                except Exception as e:
                    logger.warning(f"Failed to send to {client_id}: {e}")
                    disconnected.append(client_id)

        # Remove disconnected clients
        for client_id in disconnected:
            self.remove_client(client_id)


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
                subscriptions = set(data.get("events", ["*"]))
                self._broadcaster.update_subscriptions(client_id, subscriptions)

            elif msg_type == "unsubscribe":
                # Remove subscriptions
                self._broadcaster.update_subscriptions(client_id, set())

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

    def connect(self, event_bus: EventBus) -> None:
        """Connect to an EventBus and subscribe to events."""
        self._event_bus = event_bus
        self._disconnect_requested = False

        subscribe = getattr(event_bus, "subscribe", None)
        if not callable(subscribe):
            logger.warning("EventBusAdapter connect failed: event bus has no subscribe() method")
            return

        try:
            for internal_type in self.EVENT_MAPPING.keys():
                try:
                    # New async APIs require async handlers; legacy APIs expect sync handlers.
                    use_async_handler = inspect.iscoroutinefunction(subscribe)
                    handler: Callable[[MessagingEvent], Any] = (
                        self._on_event_async if use_async_handler else self._on_event
                    )
                    result = subscribe(internal_type, handler)
                    if inspect.isawaitable(result):
                        # If API reports awaitable despite sync detection, retry with async handler.
                        if not use_async_handler:
                            close_result = getattr(result, "close", None)
                            if callable(close_result):
                                close_result()
                            result = subscribe(internal_type, self._on_event_async)
                        self._subscriptions.append(internal_type)
                        self._run_async_operation(
                            result,
                            description=f"subscribe to {internal_type}",
                            on_success=self._track_subscription_handle,
                        )
                    else:
                        self._track_subscription(internal_type)
                except Exception as e:
                    logger.debug(f"Failed to subscribe to {internal_type}: {e}")
        except Exception as e:
            logger.warning(f"EventBusAdapter connect failed: {e}")

        logger.info(f"EventBusAdapter connected (subscriptions: {len(self._subscriptions)})")

    def disconnect(self) -> None:
        """Disconnect from the EventBus."""
        if self._event_bus:
            self._disconnect_requested = True

            had_async_handles = bool(self._subscription_handles)
            for handle in list(self._subscription_handles):
                unsubscribe = getattr(handle, "unsubscribe", None)
                if not callable(unsubscribe):
                    continue
                try:
                    result = unsubscribe()
                    if inspect.isawaitable(result):
                        pattern = getattr(handle, "pattern", "unknown")
                        self._run_async_operation(
                            result,
                            description=f"unsubscribe handle for {pattern}",
                        )
                except Exception:
                    pass
            self._subscription_handles.clear()

            # Legacy sync API fallback: unsubscribe(topic, handler)
            subscribe_fn = getattr(self._event_bus, "subscribe", None)
            uses_async_subscribe = callable(subscribe_fn) and inspect.iscoroutinefunction(subscribe_fn)
            has_pending_subscribes = bool(self._pending_async_tasks)
            if not had_async_handles and not uses_async_subscribe and not has_pending_subscribes:
                unsubscribe_fn = getattr(self._event_bus, "unsubscribe", None)
                if callable(unsubscribe_fn):
                    for event_type in self._subscriptions:
                        try:
                            unsubscribe_fn(event_type, self._on_event)
                        except Exception:
                            pass
            self._subscriptions.clear()

    def _on_event(self, event: MessagingEvent) -> None:
        """Handle an internal EventBus event."""
        # Map event topic to bridge event type
        bridge_type = self.EVENT_MAPPING.get(event.topic)
        if not bridge_type:
            return

        bridge_event = BridgeEvent(
            type=bridge_type,
            data=event.data,
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
            loop = asyncio.get_running_loop()
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
) -> None:
    """Emit a tool complete event."""
    broadcaster = EventBroadcaster()
    broadcaster.broadcast_sync(
        BridgeEvent(
            type=BridgeEventType.TOOL_COMPLETE,
            data={
                "tool_id": tool_id,
                "result": result,
                "duration_ms": duration_ms,
            },
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

    Example:
        bus = get_event_bus()
        bridge = EventBridge(bus)
        bridge.start()

        # ... later
        bridge.stop()
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

    def start(self) -> None:
        """Start the EventBridge.

        Connects to the EventBus and begins broadcasting events.
        """
        if self._running:
            return

        if self._event_bus:
            self._adapter.connect(self._event_bus)

        self._run_async_operation(
            self._broadcaster.start(),
            description="start broadcaster",
        )
        self._running = True
        logger.info("EventBridge started")

    def stop(self) -> None:
        """Stop the EventBridge.

        Disconnects from EventBus and stops broadcasting.
        """
        if not self._running:
            return

        self._adapter.disconnect()
        self._run_async_operation(
            self._broadcaster.stop(),
            description="stop broadcaster",
        )
        self._running = False
        logger.info("EventBridge stopped")

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
