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

"""WebSocket server for real-time team collaboration dashboard.

This module provides a FastAPI-based WebSocket server that streams real-time
team execution events to connected dashboard clients. It subscribes to team
events from the event bus and broadcasts them to connected clients.

Key Features:
- WebSocket endpoint for real-time event streaming
- Team execution state tracking
- Member status updates
- Communication flow visualization
- Shared context changes
- Negotiation progress tracking
- Support for 10+ concurrent team executions

Architecture:
- FastAPI with WebSocket support
- Event bus subscription for team events
- Connection management per team execution
- JSON message serialization
- Automatic reconnection handling
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
)

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore
    WebSocket = None  # type: ignore

from victor.workflows.team_collaboration import (
    TeamCommunicationProtocol,
    SharedTeamContext,
    NegotiationFramework,
    CommunicationLog,
    ContextUpdate,
    NegotiationResult,
)
from victor.workflows.team_metrics import (
    TeamMetricsCollector,
    MemberExecutionMetrics,
    TeamExecutionMetrics,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


class MemberStatus(str, Enum):
    """Status of a team member."""

    IDLE = "idle"
    """Member is not currently executing."""

    RUNNING = "running"
    """Member is actively executing."""

    COMPLETED = "completed"
    """Member completed successfully."""

    FAILED = "failed"
    """Member execution failed."""

    WAITING = "waiting"
    """Member is waiting for input/resources."""


@dataclass
class MemberState:
    """Real-time state of a team member.

    Attributes:
        member_id: Unique member identifier
        role: Member role (e.g., "security_reviewer")
        status: Current status
        start_time: Execution start time
        end_time: Execution end time
        duration_seconds: Execution duration
        tool_calls_used: Number of tool calls made
        tools_used: Set of tools used
        error_message: Error message if failed
        last_activity: Last activity timestamp
    """

    member_id: str
    role: str = "assistant"
    status: MemberStatus = MemberStatus.IDLE
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    tool_calls_used: int = 0
    tools_used: Set[str] = field(default_factory=set)
    error_message: Optional[str] = None
    last_activity: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "member_id": self.member_id,
            "role": self.role,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "tool_calls_used": self.tool_calls_used,
            "tools_used": list(self.tools_used),
            "error_message": self.error_message,
            "last_activity": self.last_activity,
        }


@dataclass
class TeamExecutionState:
    """Real-time state of a team execution.

    Attributes:
        execution_id: Unique execution identifier
        team_id: Team identifier
        formation: Team formation type
        member_states: States of all members
        shared_context: Shared context snapshot
        communication_logs: Communication history
        negotiation_status: Current negotiation status
        start_time: Execution start time
        end_time: Execution end time
        duration_seconds: Total execution duration
        success: Whether execution succeeded
        recursion_depth: Recursion depth
        consensus_achieved: Whether consensus was achieved
    """

    execution_id: str
    team_id: str
    formation: str
    member_states: Dict[str, MemberState] = field(default_factory=dict)
    shared_context: Dict[str, Any] = field(default_factory=dict)
    communication_logs: List[CommunicationLog] = field(default_factory=list)
    negotiation_status: Optional[NegotiationResult] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    success: Optional[bool] = None
    recursion_depth: int = 0
    consensus_achieved: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "execution_id": self.execution_id,
            "team_id": self.team_id,
            "formation": self.formation,
            "member_states": {k: v.to_dict() for k, v in self.member_states.items()},
            "shared_context": self.shared_context,
            "communication_logs": [log.to_dict() for log in self.communication_logs],
            "negotiation_status": (
                self.negotiation_status.__dict__ if self.negotiation_status else None
            ),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "recursion_depth": self.recursion_depth,
            "consensus_achieved": self.consensus_achieved,
        }


@dataclass
class DashboardEvent:
    """Event broadcast to dashboard clients.

    Attributes:
        event_type: Type of event
        execution_id: Execution identifier
        timestamp: Event timestamp
        data: Event payload
    """

    event_type: str
    execution_id: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type,
            "execution_id": self.execution_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }


# =============================================================================
# Event Types
# =============================================================================


class DashboardEventType(str, Enum):
    """Event types broadcast to dashboard clients."""

    # Team lifecycle events
    TEAM_STARTED = "team.started"
    """Team execution started."""

    TEAM_COMPLETED = "team.completed"
    """Team execution completed."""

    # Member events
    MEMBER_STARTED = "member.started"
    """Team member started."""

    MEMBER_UPDATED = "member.updated"
    """Team member state updated."""

    MEMBER_COMPLETED = "member.completed"
    """Team member completed."""

    MEMBER_FAILED = "member.failed"
    """Team member failed."""

    # Communication events
    MESSAGE_SENT = "message.sent"
    """Message sent between members."""

    MESSAGE_RECEIVED = "message.received"
    """Message received by member."""

    # Context events
    CONTEXT_UPDATED = "context.updated"
    """Shared context updated."""

    CONTEXT_MERGED = "context.merged"
    """Context merged."""

    # Negotiation events
    NEGOTIATION_STARTED = "negotiation.started"
    """Negotiation started."""

    NEGOTIATION_VOTE = "negotiation.vote"
    """Vote cast."""

    NEGOTIATION_COMPLETED = "negotiation.completed"
    """Negotiation completed."""

    # Progress events
    PROGRESS_UPDATE = "progress.update"
    """Progress update."""

    METRICS_UPDATE = "metrics.update"
    """Metrics update."""


# =============================================================================
# Connection Manager
# =============================================================================


class ConnectionManager:
    """Manages WebSocket connections for team dashboard.

    This class handles WebSocket connection lifecycle, message broadcasting,
    and connection cleanup. It supports multiple concurrent connections per
    team execution.

    Attributes:
        _active_connections: Mapping of execution_id to set of WebSocket connections
        _connection_metadata: Mapping of connection to metadata
        _lock: Async lock for thread safety
    """

    def __init__(self) -> None:
        """Initialize connection manager."""
        self._active_connections: Dict[str, Set[WebSocket]] = defaultdict(set)  # type: ignore[arg-type]
        self._connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, execution_id: str) -> None:
        """Connect a WebSocket to a team execution.

        Args:
            websocket: WebSocket connection
            execution_id: Team execution identifier
        """
        await self._lock.acquire()
        try:
            self._active_connections[execution_id].add(websocket)
            self._connection_metadata[websocket] = {
                "execution_id": execution_id,
                "connected_at": time.time(),
                "client_id": uuid.uuid4().hex[:8],
            }
            logger.info(
                f"WebSocket connected: execution_id={execution_id}, "
                f"client_id={self._connection_metadata[websocket]['client_id']}"
            )
        finally:
            self._lock.release()

    async def disconnect(self, websocket: WebSocket) -> None:
        """Disconnect a WebSocket.

        Args:
            websocket: WebSocket connection
        """
        await self._lock.acquire()
        try:
            if websocket in self._connection_metadata:
                metadata = self._connection_metadata[websocket]
                execution_id = metadata["execution_id"]
                client_id = metadata.get("client_id", "unknown")

                # Remove from active connections
                if execution_id in self._active_connections:
                    self._active_connections[execution_id].discard(websocket)
                    if not self._active_connections[execution_id]:
                        del self._active_connections[execution_id]

                # Remove metadata
                del self._connection_metadata[websocket]

                logger.info(
                    f"WebSocket disconnected: execution_id={execution_id}, "
                    f"client_id={client_id}"
                )
        finally:
            self._lock.release()

    async def broadcast_to_execution(
        self,
        execution_id: str,
        event: DashboardEvent,
    ) -> None:
        """Broadcast event to all connections for a team execution.

        Args:
            execution_id: Team execution identifier
            event: Event to broadcast
        """
        await self._lock.acquire()
        try:
            connections = self._active_connections.get(execution_id, set()).copy()
        finally:
            self._lock.release()

        if not connections:
            return

        # Serialize event once
        message = json.dumps(event.to_dict())

        # Broadcast to all connections
        disconnected = []
        for websocket in connections:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.append(websocket)

        # Clean up disconnected WebSockets
        for websocket in disconnected:
            await self.disconnect(websocket)

    async def broadcast_to_all(self, event: DashboardEvent) -> None:
        """Broadcast event to all active connections.

        Args:
            event: Event to broadcast
        """
        await self._lock.acquire()
        try:
            all_connections = set()
            for connections in self._active_connections.values():
                all_connections.update(connections)
        finally:
            self._lock.release()

        message = json.dumps(event.to_dict())
        disconnected = []

        for websocket in all_connections:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to broadcast: {e}")
                disconnected.append(websocket)

        for websocket in disconnected:
            await self.disconnect(websocket)

    async def send_personal_message(
        self,
        message: Dict[str, Any],
        websocket: WebSocket,
    ) -> None:
        """Send a message to a specific WebSocket.

        Args:
            message: Message to send
            websocket: WebSocket connection
        """
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.warning(f"Failed to send personal message: {e}")
            await self.disconnect(websocket)

    def get_connection_count(self, execution_id: str) -> int:
        """Get number of active connections for an execution.

        Args:
            execution_id: Team execution identifier

        Returns:
            Number of active connections
        """
        return len(self._active_connections.get(execution_id, set()))

    def get_all_execution_ids(self) -> List[str]:
        """Get all execution IDs with active connections.

        Returns:
            List of execution IDs
        """
        return list(self._active_connections.keys())


# =============================================================================
# Dashboard Server
# =============================================================================


class TeamDashboardServer:
    """WebSocket server for team collaboration dashboard.

    This server provides real-time updates for team execution by subscribing
    to team events from the event bus and broadcasting them to connected
    WebSocket clients.

    Attributes:
        _app: FastAPI application
        _connection_manager: WebSocket connection manager
        _execution_states: Team execution states
        _metrics_collector: Team metrics collector
        _event_bus_subscriptions: Event bus subscriptions
    """

    def __init__(
        self,
        metrics_collector: Optional[TeamMetricsCollector] = None,
        cors_origins: Optional[List[str]] = None,
    ) -> None:
        """Initialize dashboard server.

        Args:
            metrics_collector: Team metrics collector
            cors_origins: CORS allowed origins
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for the dashboard server. "
                "Install it with: pip install victor-ai[api]"
            )

        self._app = FastAPI(
            title="Victor Team Dashboard",
            description="Real-time team collaboration dashboard",
            version="0.5.0",
        )

        # Setup CORS
        cors_origins = cors_origins or ["http://localhost:3000", "http://localhost:8000"]
        self._app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Components
        self._connection_manager = ConnectionManager()
        self._execution_states: Dict[str, TeamExecutionState] = {}
        self._metrics_collector = metrics_collector or TeamMetricsCollector.get_instance()
        self._event_bus_subscriptions: List[Any] = []

        # Setup routes
        self._setup_routes()

        logger.info("Team Dashboard Server initialized")

    @property
    def app(self) -> FastAPI:
        """Get FastAPI application."""
        return self._app

    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""

        @self._app.get("/health")
        async def health_check() -> Dict[str, Any]:
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "active_connections": sum(
                    len(conns) for conns in self._connection_manager._active_connections.values()
                ),
                "tracked_executions": len(self._execution_states),
            }

        @self._app.websocket("/ws/team/{execution_id}")
        async def websocket_endpoint(websocket: WebSocket, execution_id: str) -> None:
            """WebSocket endpoint for team execution updates.

            Args:
                websocket: WebSocket connection
                execution_id: Team execution identifier
            """
            await websocket.accept()

            # Connect to execution
            await self._connection_manager.connect(websocket, execution_id)

            try:
                # Send initial state
                if execution_id in self._execution_states:
                    initial_event = DashboardEvent(
                        event_type=DashboardEventType.TEAM_STARTED.value,
                        execution_id=execution_id,
                        timestamp=time.time(),
                        data=self._execution_states[execution_id].to_dict(),
                    )
                    await self._connection_manager.send_personal_message(
                        initial_event.to_dict(),
                        websocket,
                    )

                # Keep connection alive and handle incoming messages
                while True:
                    data = await websocket.receive_text()

                    # Handle incoming messages (e.g., subscriptions, queries)
                    try:
                        message = json.loads(data)
                        await self._handle_client_message(execution_id, websocket, message)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from client: {data}")

            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected: {execution_id}")
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
            finally:
                await self._connection_manager.disconnect(websocket)

    async def _handle_client_message(
        self,
        execution_id: str,
        websocket: WebSocket,
        message: Dict[str, Any],
    ) -> None:
        """Handle incoming message from client.

        Args:
            execution_id: Execution identifier
            websocket: WebSocket connection
            message: Client message
        """
        action = message.get("action")

        if action == "subscribe":
            # Subscribe to specific event types
            event_types = message.get("event_types", [])
            # NOTE: Per-connection filtering requires event type registry and filter middleware
            # Deferred: Low priority - current broadcast model works for small teams
            logger.debug(f"Client subscribed to events: {event_types}")

        elif action == "query_state":
            # Send current state
            if execution_id in self._execution_states:
                state = self._execution_states[execution_id].to_dict()
                await self._connection_manager.send_personal_message(
                    {
                        "action": "state_snapshot",
                        "execution_id": execution_id,
                        "state": state,
                    },
                    websocket,
                )

        elif action == "ping":
            # Respond to ping
            await self._connection_manager.send_personal_message(
                {"action": "pong", "timestamp": time.time()},
                websocket,
            )

    # =========================================================================
    # Event Broadcasting
    # =========================================================================

    async def broadcast_team_started(
        self,
        execution_id: str,
        team_id: str,
        formation: str,
        member_count: int,
        recursion_depth: int,
    ) -> None:
        """Broadcast team started event.

        Args:
            execution_id: Execution identifier
            team_id: Team identifier
            formation: Formation type
            member_count: Number of members
            recursion_depth: Recursion depth
        """
        # Initialize execution state
        self._execution_states[execution_id] = TeamExecutionState(
            execution_id=execution_id,
            team_id=team_id,
            formation=formation,
            start_time=datetime.now(timezone.utc),
            recursion_depth=recursion_depth,
        )

        event = DashboardEvent(
            event_type=DashboardEventType.TEAM_STARTED.value,
            execution_id=execution_id,
            timestamp=time.time(),
            data={
                "team_id": team_id,
                "formation": formation,
                "member_count": member_count,
                "recursion_depth": recursion_depth,
                "start_time": (
                    self._execution_states[execution_id].start_time.isoformat()
                    if self._execution_states[execution_id].start_time
                    else None
                ),
            },
        )

        await self._connection_manager.broadcast_to_execution(execution_id, event)

    async def broadcast_member_started(
        self,
        execution_id: str,
        member_id: str,
        role: str,
    ) -> None:
        """Broadcast member started event.

        Args:
            execution_id: Execution identifier
            member_id: Member identifier
            role: Member role
        """
        if execution_id not in self._execution_states:
            logger.warning(f"Execution {execution_id} not found")
            return

        # Update member state
        state = MemberState(
            member_id=member_id,
            role=role,
            status=MemberStatus.RUNNING,
            start_time=datetime.now(timezone.utc),
        )
        self._execution_states[execution_id].member_states[member_id] = state

        event = DashboardEvent(
            event_type=DashboardEventType.MEMBER_STARTED.value,
            execution_id=execution_id,
            timestamp=time.time(),
            data={
                "member_id": member_id,
                "role": role,
                "status": state.status.value,
                "start_time": state.start_time.isoformat() if state.start_time else None,
            },
        )

        await self._connection_manager.broadcast_to_execution(execution_id, event)

    async def broadcast_member_updated(
        self,
        execution_id: str,
        member_id: str,
        tool_calls_used: int,
        tools_used: Set[str],
    ) -> None:
        """Broadcast member updated event.

        Args:
            execution_id: Execution identifier
            member_id: Member identifier
            tool_calls_used: Number of tool calls
            tools_used: Tools used
        """
        if execution_id not in self._execution_states:
            return

        member_state = self._execution_states[execution_id].member_states.get(member_id)
        if not member_state:
            return

        # Update state
        member_state.tool_calls_used = tool_calls_used
        member_state.tools_used.update(tools_used)
        member_state.last_activity = time.time()

        event = DashboardEvent(
            event_type=DashboardEventType.MEMBER_UPDATED.value,
            execution_id=execution_id,
            timestamp=time.time(),
            data={
                "member_id": member_id,
                "tool_calls_used": tool_calls_used,
                "tools_used": list(tools_used),
            },
        )

        await self._connection_manager.broadcast_to_execution(execution_id, event)

    async def broadcast_member_completed(
        self,
        execution_id: str,
        member_id: str,
        success: bool,
        duration_seconds: float,
        error_message: Optional[str] = None,
    ) -> None:
        """Broadcast member completed event.

        Args:
            execution_id: Execution identifier
            member_id: Member identifier
            success: Whether succeeded
            duration_seconds: Duration
            error_message: Error message if failed
        """
        if execution_id not in self._execution_states:
            return

        member_state = self._execution_states[execution_id].member_states.get(member_id)
        if not member_state:
            return

        # Update state
        member_state.status = MemberStatus.COMPLETED if success else MemberStatus.FAILED
        member_state.end_time = datetime.now(timezone.utc)
        member_state.duration_seconds = duration_seconds
        member_state.error_message = error_message
        member_state.last_activity = time.time()

        event_type = (
            DashboardEventType.MEMBER_COMPLETED.value
            if success
            else DashboardEventType.MEMBER_FAILED.value
        )

        event = DashboardEvent(
            event_type=event_type,
            execution_id=execution_id,
            timestamp=time.time(),
            data={
                "member_id": member_id,
                "success": success,
                "duration_seconds": duration_seconds,
                "error_message": error_message,
                "end_time": member_state.end_time.isoformat(),
            },
        )

        await self._connection_manager.broadcast_to_execution(execution_id, event)

    async def broadcast_team_completed(
        self,
        execution_id: str,
        success: bool,
        duration_seconds: float,
        consensus_achieved: Optional[bool] = None,
    ) -> None:
        """Broadcast team completed event.

        Args:
            execution_id: Execution identifier
            success: Whether succeeded
            duration_seconds: Duration
            consensus_achieved: Whether consensus achieved
        """
        if execution_id not in self._execution_states:
            return

        state = self._execution_states[execution_id]
        state.end_time = datetime.now(timezone.utc)
        state.duration_seconds = duration_seconds
        state.success = success
        state.consensus_achieved = consensus_achieved

        event = DashboardEvent(
            event_type=DashboardEventType.TEAM_COMPLETED.value,
            execution_id=execution_id,
            timestamp=time.time(),
            data={
                "success": success,
                "duration_seconds": duration_seconds,
                "consensus_achieved": consensus_achieved,
                "end_time": state.end_time.isoformat(),
                "final_state": state.to_dict(),
            },
        )

        await self._connection_manager.broadcast_to_execution(execution_id, event)

    async def broadcast_communication(
        self,
        execution_id: str,
        log: CommunicationLog,
    ) -> None:
        """Broadcast communication event.

        Args:
            execution_id: Execution identifier
            log: Communication log
        """
        if execution_id not in self._execution_states:
            return

        # Add to communication logs
        self._execution_states[execution_id].communication_logs.append(log)

        event = DashboardEvent(
            event_type=DashboardEventType.MESSAGE_SENT.value,
            execution_id=execution_id,
            timestamp=time.time(),
            data={
                "log": log.to_dict(),
            },
        )

        await self._connection_manager.broadcast_to_execution(execution_id, event)

    async def broadcast_context_update(
        self,
        execution_id: str,
        update: ContextUpdate,
    ) -> None:
        """Broadcast context update event.

        Args:
            execution_id: Execution identifier
            update: Context update
        """
        if execution_id not in self._execution_states:
            return

        # Update shared context
        state = self._execution_states[execution_id]
        state.shared_context[update.key] = update.value

        event = DashboardEvent(
            event_type=DashboardEventType.CONTEXT_UPDATED.value,
            execution_id=execution_id,
            timestamp=time.time(),
            data={
                "key": update.key,
                "value": update.value,
                "member_id": update.member_id,
                "operation": update.operation,
                "timestamp": update.timestamp,
            },
        )

        await self._connection_manager.broadcast_to_execution(execution_id, event)

    async def broadcast_negotiation_update(
        self,
        execution_id: str,
        result: NegotiationResult,
    ) -> None:
        """Broadcast negotiation update event.

        Args:
            execution_id: Execution identifier
            result: Negotiation result
        """
        if execution_id not in self._execution_states:
            return

        self._execution_states[execution_id].negotiation_status = result

        event_type = (
            DashboardEventType.NEGOTIATION_COMPLETED.value
            if result.success
            else DashboardEventType.NEGOTIATION_STARTED.value
        )

        event = DashboardEvent(
            event_type=event_type,
            execution_id=execution_id,
            timestamp=time.time(),
            data={
                "success": result.success,
                "rounds": result.rounds,
                "consensus_achieved": result.consensus_achieved,
                "votes": result.votes,
                "agreed_proposal": (
                    result.agreed_proposal.__dict__ if result.agreed_proposal else None
                ),
            },
        )

        await self._connection_manager.broadcast_to_execution(execution_id, event)

    # =========================================================================
    # State Queries
    # =========================================================================

    def get_execution_state(self, execution_id: str) -> Optional[TeamExecutionState]:
        """Get execution state.

        Args:
            execution_id: Execution identifier

        Returns:
            Execution state or None
        """
        return self._execution_states.get(execution_id)

    def get_all_execution_states(self) -> Dict[str, TeamExecutionState]:
        """Get all execution states.

        Returns:
            Dictionary of execution states
        """
        return self._execution_states.copy()

    def get_active_executions(self) -> List[str]:
        """Get list of active execution IDs.

        Returns:
            List of execution IDs
        """
        return [eid for eid, state in self._execution_states.items() if state.end_time is None]


# =============================================================================
# Singleton Instance
# =============================================================================

_dashboard_server_instance: Optional[TeamDashboardServer] = None


def get_dashboard_server(
    metrics_collector: Optional[TeamMetricsCollector] = None,
    cors_origins: Optional[List[str]] = None,
) -> TeamDashboardServer:
    """Get singleton dashboard server instance.

    Args:
        metrics_collector: Team metrics collector
        cors_origins: CORS allowed origins

    Returns:
        Dashboard server instance
    """
    global _dashboard_server_instance

    if _dashboard_server_instance is None:
        _dashboard_server_instance = TeamDashboardServer(
            metrics_collector=metrics_collector,
            cors_origins=cors_origins,
        )

    return _dashboard_server_instance


__all__ = [
    # Data classes
    "MemberStatus",
    "MemberState",
    "TeamExecutionState",
    "DashboardEvent",
    "DashboardEventType",
    # Server
    "TeamDashboardServer",
    "ConnectionManager",
    "get_dashboard_server",
]
