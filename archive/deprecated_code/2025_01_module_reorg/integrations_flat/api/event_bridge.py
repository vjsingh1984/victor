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
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from victor.observability.event_bus import EventBus, VictorEvent

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
                    await asyncio.wait_for(
                        asyncio.coroutine(client.send)(event_json),
                        timeout=5.0,
                    )
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

    # Mapping of internal event types to bridge event types
    EVENT_MAPPING = {
        "tool_start": BridgeEventType.TOOL_START,
        "tool_progress": BridgeEventType.TOOL_PROGRESS,
        "tool_complete": BridgeEventType.TOOL_COMPLETE,
        "tool_error": BridgeEventType.TOOL_ERROR,
        "file_change": BridgeEventType.FILE_MODIFIED,
        "provider_switch": BridgeEventType.PROVIDER_SWITCH,
        "provider_error": BridgeEventType.PROVIDER_ERROR,
        "session_start": BridgeEventType.SESSION_START,
        "session_end": BridgeEventType.SESSION_END,
        "metrics_update": BridgeEventType.METRICS_UPDATE,
        "budget_warning": BridgeEventType.BUDGET_WARNING,
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

    def connect(self, event_bus: EventBus) -> None:
        """Connect to an EventBus and subscribe to events."""
        self._event_bus = event_bus

        # Subscribe to all mapped events
        for internal_type in self.EVENT_MAPPING.keys():
            try:
                event_bus.subscribe(internal_type, self._on_event)
                self._subscriptions.append(internal_type)
            except Exception as e:
                logger.warning(f"Failed to subscribe to {internal_type}: {e}")

        logger.info(f"EventBusAdapter connected, subscribed to {len(self._subscriptions)} events")

    def disconnect(self) -> None:
        """Disconnect from the EventBus."""
        if self._event_bus:
            for event_type in self._subscriptions:
                try:
                    self._event_bus.unsubscribe(event_type, self._on_event)
                except Exception:
                    pass
            self._subscriptions.clear()

    def _on_event(self, event: VictorEvent) -> None:
        """Handle an internal EventBus event."""
        bridge_type = self.EVENT_MAPPING.get(event.event_type)
        if not bridge_type:
            return

        bridge_event = BridgeEvent(
            type=bridge_type,
            data=event.data if hasattr(event, "data") else {},
        )

        self._broadcaster.broadcast_sync(bridge_event)


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
