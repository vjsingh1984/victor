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

"""CQRS Bridge - Connects framework events with CQRS event sourcing.

This module provides adapters that bridge:
1. Framework Events (victor/framework/events.py) ↔ CQRS Events (victor/core/event_sourcing.py)
2. Observability EventBus ↔ CQRS EventDispatcher
3. Framework Agent commands ↔ CQRS Command/Query buses

Design Patterns:
- Adapter: Converts between framework and CQRS event formats
- Bridge: Decouples abstraction from implementation
- Mediator: Coordinates event flow between subsystems

Architecture:
    Framework Layer                    CQRS Layer
    ┌─────────────┐                   ┌──────────────┐
    │   Event     │   CQRSBridge      │    Event     │
    │  (simple)   │◄─────────────────►│  (sourced)   │
    └─────────────┘                   └──────────────┘
           │                                 │
           ▼                                 ▼
    ┌─────────────┐                   ┌──────────────┐
    │  EventBus   │   EventAdapter    │   Event      │
    │ (pub/sub)   │◄─────────────────►│  Dispatcher  │
    └─────────────┘                   └──────────────┘

Example:
    from victor.framework.cqrs_bridge import CQRSBridge, FrameworkEventAdapter

    # Create bridge
    bridge = CQRSBridge()

    # Connect to agent
    bridge.connect_agent(agent)

    # Framework events are now automatically forwarded to CQRS layer
    async for event in agent.stream("Analyze code"):
        # Events are being sourced for replay/projection
        print(event.content)

    # Query session history via CQRS
    history = await bridge.get_session_history(session_id)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Union
from uuid import uuid4

from victor.framework.events import AgentExecutionEvent, EventType

if TYPE_CHECKING:
    from victor.core.agent_commands import (
        AgentCommandBus,
        SessionProjection,
    )
    from victor.core.cqrs import CommandBus, Mediator, QueryBus
    from victor.core.event_sourcing import DomainEvent as CQRSEvent
    from victor.core.event_sourcing import EventDispatcher, EventStore
    from victor.framework.agent import Agent
    from victor.core.events import ObservabilityBus

logger = logging.getLogger(__name__)


# =============================================================================
# Event Conversion - Delegates to EventRegistry (Single Source of Truth)
# =============================================================================

# Import conversion functions from EventRegistry - the canonical implementation
from victor.framework.event_registry import (
    EventTarget,
    convert_from_cqrs,
    convert_from_observability,
    convert_to_cqrs,
    convert_to_observability,
    get_event_registry,
)


# Wrapper functions - delegate to EventRegistry (single source of truth)
# These are kept for backward compatibility with existing callers.


def framework_event_to_cqrs(event: AgentExecutionEvent) -> Dict[str, Any]:
    """Convert framework AgentExecutionEvent to CQRS data. Delegates to EventRegistry."""
    return convert_to_cqrs(event)


def cqrs_event_to_framework(cqrs_event: "CQRSEvent") -> AgentExecutionEvent:
    """Convert a CQRS DomainEvent to a framework AgentExecutionEvent.

    Delegates to EventRegistry for the actual conversion.

    Args:
        cqrs_event: CQRS DomainEvent instance.

    Returns:
        Framework AgentExecutionEvent instance.
    """
    # Extract event type and data from CQRS event
    event_type = cqrs_event.event_type

    # Get event data using _get_data() or from metadata
    if hasattr(cqrs_event, "_get_data"):
        data = cqrs_event._get_data()
    else:
        data = getattr(cqrs_event, "metadata", {}).get("data", {})

    metadata = getattr(cqrs_event, "metadata", {})

    # For class-based events, extract attributes into data dict
    if event_type in (
        "ToolCalledEvent",
        "ToolResultEvent",
        "StateChangedEvent",
        "TaskStartedEvent",
        "TaskCompletedEvent",
        "TaskFailedEvent",
    ):
        # These are class-based events with attributes
        if event_type == "ToolCalledEvent":
            data = {
                "tool_name": getattr(cqrs_event, "tool_name", ""),
                "arguments": getattr(cqrs_event, "arguments", {}),
                "tool_id": metadata.get("tool_id"),
            }
            event_type = "tool_called"
        elif event_type == "ToolResultEvent":
            data = {
                "tool_name": getattr(cqrs_event, "tool_name", ""),
                "result": getattr(cqrs_event, "result", ""),
                "success": getattr(cqrs_event, "success", True),
                "tool_id": metadata.get("tool_id"),
            }
            event_type = "tool_result"
        elif event_type == "StateChangedEvent":
            data = {
                "old_stage": getattr(cqrs_event, "from_state", ""),
                "new_stage": getattr(cqrs_event, "to_state", ""),
            }
            event_type = "stage_changed"
        elif event_type == "TaskStartedEvent":
            data = {
                "task_id": getattr(cqrs_event, "task_id", ""),
                "provider": getattr(cqrs_event, "provider", ""),
                "model": getattr(cqrs_event, "model", ""),
            }
            event_type = "stream_started"
        elif event_type == "TaskCompletedEvent":
            data = {"success": True}
            event_type = "stream_ended"
        elif event_type == "TaskFailedEvent":
            data = {
                "success": False,
                "error": getattr(cqrs_event, "error", "Unknown error"),
            }
            event_type = "stream_ended"

    # Handle events with event_type in metadata (generic events from forwarding)
    if event_type == "Event" and "event_type" in metadata:
        event_type = metadata["event_type"]
        data = metadata.get("data", data)

    return convert_from_cqrs(data, event_type, metadata)


def observability_event_to_framework(
    topic: str, data: Dict[str, Any], **metadata
) -> AgentExecutionEvent:
    """Convert an observability event (topic-based) to a framework AgentExecutionEvent.

    Uses topic-based mapping since observability events now use topics
    (e.g., "tool.start" instead of category-based "tool.start").

    Args:
        topic: Event topic (e.g., "tool.start", "state.transition").
        data: Event data dictionary.
        **metadata: Additional metadata.

    Returns:
        Framework AgentExecutionEvent instance.
    """
    from victor.framework.events import (
        content_event,
        error_event,
        stage_change_event,
        stream_end_event,
        stream_start_event,
        tool_call_event,
        tool_result_event,
    )

    event_data = data

    # Extract category from topic prefix
    topic_prefix = topic.split(".", 1)[0] if "." in topic else topic

    # Use topic-based conversion
    if topic_prefix == "tool":
        if ".start" in topic:
            return tool_call_event(
                tool_name=event_data.get("tool_name", topic.replace(".start", "")),
                tool_id=event_data.get("tool_id"),
                arguments=event_data.get("arguments", {}),
            )
        elif ".end" in topic:
            return tool_result_event(
                tool_name=event_data.get("tool_name", topic.replace(".end", "")),
                tool_id=event_data.get("tool_id"),
                result=str(event_data.get("result", "")),
                success=event_data.get("success", True),
            )
        else:
            return tool_call_event(
                tool_name=event_data.get("tool_name", topic),
                tool_id=event_data.get("tool_id"),
                arguments=event_data.get("arguments", {}),
            )

    elif topic_prefix == "state":
        return stage_change_event(
            old_stage=event_data.get("old_stage", "unknown"),
            new_stage=event_data.get("new_stage", "unknown"),
            metadata=event_data,
        )

    elif topic_prefix == "error":
        return error_event(
            error=event_data.get("message", topic),
            recoverable=event_data.get("recoverable", True),
        )

    elif topic_prefix == "lifecycle":
        if "start" in topic:
            return stream_start_event(metadata=event_data)
        else:
            return stream_end_event(
                success=event_data.get("success", True),
                error=event_data.get("error"),
            )

    else:
        # Generic fallback
        return content_event(
            content=f"[{topic_prefix.upper()}] {topic}",
            metadata=event_data,
        )


def framework_event_to_observability(event: AgentExecutionEvent) -> Dict[str, Any]:
    """Convert framework AgentExecutionEvent to observability data. Delegates to EventRegistry."""
    return convert_to_observability(event)


# =============================================================================
# Event Adapters
# =============================================================================


@dataclass
class FrameworkEventAdapter:
    """Adapter that forwards framework events to CQRS and observability.

    This adapter sits in the event flow and ensures events are propagated
    to all interested subsystems.

    Attributes:
        event_dispatcher: CQRS EventDispatcher for sourcing.
        event_bus: Observability EventBus for pub/sub.
        session_id: Current session ID for correlation.
        aggregate_id: CQRS aggregate ID for event sourcing.

    Example:
        adapter = FrameworkEventAdapter(
            event_dispatcher=dispatcher,
            event_bus=event_bus,
            session_id="session-123"
        )

        # Forward framework events
        adapter.forward(event)
    """

    event_dispatcher: Optional["EventDispatcher"] = None
    event_bus: Optional["ObservabilityBus"] = None
    session_id: Optional[str] = None
    aggregate_id: Optional[str] = None
    _forwarded_count: int = field(default=0, init=False)

    def forward(self, event: AgentExecutionEvent) -> None:
        """Forward a framework event to CQRS and observability.

        Args:
            event: Framework AgentExecutionEvent to forward.
        """
        # Forward to CQRS EventDispatcher
        if self.event_dispatcher:
            self._forward_to_cqrs(event)

        # Forward to observability EventBus
        if self.event_bus:
            self._forward_to_observability(event)

        self._forwarded_count += 1

    def _forward_to_cqrs(self, event: AgentExecutionEvent) -> None:
        """Forward event to CQRS layer."""
        try:
            from victor.core.event_sourcing import (
                DomainEvent as CQRSEvent,
                StateChangedEvent,
                ToolCalledEvent,
                ToolResultEvent,
            )

            cqrs_data = framework_event_to_cqrs(event)
            event_type = cqrs_data.pop("event_type", "unknown")
            aggregate_id = self.aggregate_id or self.session_id or "unknown"

            # Map framework event types to CQRS event classes
            if event_type == "tool_called":
                cqrs_event = ToolCalledEvent(
                    task_id=aggregate_id,
                    tool_name=cqrs_data.get("tool_name", ""),
                    arguments=cqrs_data.get("arguments", {}),
                    metadata=cqrs_data.get("metadata", {}),
                )
            elif event_type == "tool_result":
                cqrs_event = ToolResultEvent(
                    task_id=aggregate_id,
                    tool_name=cqrs_data.get("tool_name", ""),
                    success=cqrs_data.get("success", True),
                    result=str(cqrs_data.get("result", "")),
                    metadata=cqrs_data.get("metadata", {}),
                )
            elif event_type == "stage_changed":
                cqrs_event = StateChangedEvent(
                    task_id=aggregate_id,
                    from_state=cqrs_data.get("old_stage", ""),
                    to_state=cqrs_data.get("new_stage", ""),
                    reason="framework_event",
                    metadata=cqrs_data.get("metadata", {}),
                )
            else:
                # For other event types, use Event.from_dict for dynamic creation
                cqrs_event = CQRSEvent(
                    metadata={
                        "event_type": event_type,
                        "aggregate_id": aggregate_id,
                        **cqrs_data.get("metadata", {}),
                        "data": cqrs_data,
                    }
                )

            # Dispatch to handlers (async dispatch, handle sync/async context)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.event_dispatcher.dispatch(cqrs_event))
                else:
                    loop.run_until_complete(self.event_dispatcher.dispatch(cqrs_event))
            except RuntimeError:
                # No event loop available, create a new one
                asyncio.run(self.event_dispatcher.dispatch(cqrs_event))

        except Exception as e:
            logger.warning(f"Failed to forward event to CQRS: {e}")

    def _forward_to_observability(self, event: AgentExecutionEvent) -> None:
        """Forward event to observability layer."""
        try:
            from victor.core.events import get_observability_bus

            # Convert framework AgentExecutionEvent to topic and data for ObservabilityBus
            obs_data = framework_event_to_observability(event)

            # Extract topic from observability data format
            # The framework_event_to_observability should return {"category": ..., "name": ..., "data": ...}
            # Convert to topic: "category.name" -> "category.name"
            topic = obs_data.get("topic") or f"{obs_data['category']}.{obs_data['name']}"

            bus = get_observability_bus()
            if bus:
                import asyncio

                asyncio.run(
                    bus.emit(
                        topic=topic,
                        data={
                            **obs_data["data"],
                            "category": obs_data["category"],  # Preserve for observability
                        },
                    )
                )

        except Exception as e:
            logger.warning(f"Failed to forward event to observability: {e}")

    @property
    def forwarded_count(self) -> int:
        """Get number of events forwarded."""
        return self._forwarded_count


class ObservabilityToCQRSBridge:
    """Bridge that connects observability EventBus to CQRS EventDispatcher.

    This enables existing observability events to be sourced in CQRS
    for event replay and projections.

    Example:
        bridge = ObservabilityToCQRSBridge(event_bus, dispatcher)
        bridge.start()  # Begin forwarding

        # Later
        bridge.stop()
    """

    def __init__(
        self,
        event_bus: "ObservabilityBus",
        event_dispatcher: "EventDispatcher",
        aggregate_id: str = "observability",
    ) -> None:
        """Initialize bridge.

        Args:
            event_bus: Observability EventBus.
            event_dispatcher: CQRS EventDispatcher.
            aggregate_id: Aggregate ID for sourced events.
        """
        self._event_bus = event_bus
        self._dispatcher = event_dispatcher
        self._aggregate_id = aggregate_id
        self._unsubscribe: Optional[Callable[[], None]] = None
        self._is_running = False
        self._event_count = 0

    def start(self) -> None:
        """Start forwarding events."""
        if self._is_running:
            return

        # Subscribe to all observability events
        self._unsubscribe = self._event_bus.subscribe_all(self._handle_event)
        self._is_running = True
        logger.debug("ObservabilityToCQRSBridge started")

    def stop(self) -> None:
        """Stop forwarding events."""
        if not self._is_running:
            return

        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None

        self._is_running = False
        logger.debug(f"ObservabilityToCQRSBridge stopped ({self._event_count} events)")

    def _handle_event(self, topic: str, data: Dict[str, Any], **metadata) -> None:
        """Handle an observability event (topic-based)."""
        try:
            from victor.core.event_sourcing import (
                DomainEvent as CQRSEvent,
                StateChangedEvent,
                ToolCalledEvent,
                ToolResultEvent,
            )

            # Convert to framework event first
            framework_event = observability_event_to_framework(topic, data, **metadata)

            # Then to CQRS event data
            cqrs_data = framework_event_to_cqrs(framework_event)
            event_type = cqrs_data.pop("event_type", "unknown")

            # Map to concrete CQRS event classes
            # Extract topic prefix for metadata
            topic_prefix = topic.split(".", 1)[0] if "." in topic else topic

            if event_type == "tool_called":
                cqrs_event = ToolCalledEvent(
                    task_id=self._aggregate_id,
                    tool_name=cqrs_data.get("tool_name", ""),
                    arguments=cqrs_data.get("arguments", {}),
                    metadata={
                        "original_category": topic_prefix,
                        "original_name": topic,
                        **cqrs_data.get("metadata", {}),
                    },
                )
            elif event_type == "tool_result":
                cqrs_event = ToolResultEvent(
                    task_id=self._aggregate_id,
                    tool_name=cqrs_data.get("tool_name", ""),
                    success=cqrs_data.get("success", True),
                    result=str(cqrs_data.get("result", "")),
                    metadata={
                        "original_category": topic_prefix,
                        "original_name": topic,
                        **cqrs_data.get("metadata", {}),
                    },
                )
            elif event_type == "stage_changed":
                cqrs_event = StateChangedEvent(
                    task_id=self._aggregate_id,
                    from_state=cqrs_data.get("old_stage", ""),
                    to_state=cqrs_data.get("new_stage", ""),
                    reason="observability_event",
                    metadata={
                        "original_category": topic_prefix,
                        "original_name": topic,
                        **cqrs_data.get("metadata", {}),
                    },
                )
            else:
                # For other event types, create a base Event with metadata
                cqrs_event = CQRSEvent(
                    metadata={
                        "event_type": event_type,
                        "original_category": topic_prefix,
                        "original_name": topic,
                        **cqrs_data.get("metadata", {}),
                        "data": cqrs_data,
                    }
                )

            # Dispatch to handlers (async dispatch, handle sync/async context)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._dispatcher.dispatch(cqrs_event))
                else:
                    loop.run_until_complete(self._dispatcher.dispatch(cqrs_event))
            except RuntimeError:
                # No event loop available, create a new one
                asyncio.run(self._dispatcher.dispatch(cqrs_event))

            self._event_count += 1

        except Exception as e:
            logger.warning(f"ObservabilityToCQRSBridge event handling error: {e}")

    @property
    def is_running(self) -> bool:
        """Check if bridge is running."""
        return self._is_running

    @property
    def event_count(self) -> int:
        """Get number of events processed."""
        return self._event_count


# =============================================================================
# CQRS Bridge - Main Integration Class
# =============================================================================


class CQRSBridge:
    """Main bridge connecting framework API with CQRS architecture.

    This class provides:
    1. Event forwarding from framework to CQRS
    2. Command execution via CQRS CommandBus
    3. Query execution via CQRS QueryBus
    4. Session management with event sourcing
    5. Integration with observability

    Design Pattern: Facade
    Simplifies interaction with the complex CQRS subsystem.

    Example:
        # Create bridge with automatic setup
        bridge = await CQRSBridge.create()

        # Connect to agent
        bridge.connect_agent(agent)

        # Execute commands
        result = await bridge.start_session()

        # Query state
        session = await bridge.get_session(result["session_id"])

        # Access projections
        history = bridge.get_conversation_history(session_id)
    """

    def __init__(
        self,
        command_bus: Optional["AgentCommandBus"] = None,
        event_dispatcher: Optional["EventDispatcher"] = None,
        projection: Optional["SessionProjection"] = None,
        event_bus: Optional["ObservabilityBus"] = None,
    ) -> None:
        """Initialize CQRS bridge.

        Args:
            command_bus: AgentCommandBus for commands/queries.
            event_dispatcher: CQRS EventDispatcher.
            projection: SessionProjection for read model.
            event_bus: Observability EventBus.
        """
        self._command_bus = command_bus
        self._event_dispatcher = event_dispatcher
        self._projection = projection
        self._event_bus = event_bus
        self._adapters: Dict[str, FrameworkEventAdapter] = {}
        self._obs_bridge: Optional[ObservabilityToCQRSBridge] = None
        self._connected_agents: Set[str] = set()

    @classmethod
    async def create(
        cls,
        enable_event_sourcing: bool = True,
        enable_observability: bool = True,
    ) -> "CQRSBridge":
        """Create a fully configured CQRS bridge.

        Args:
            enable_event_sourcing: Enable CQRS event sourcing.
            enable_observability: Enable observability integration.

        Returns:
            Configured CQRSBridge instance.
        """
        from victor.core.agent_commands import AgentCommandBus

        # Create the pre-configured command bus which includes
        # its own event_store, dispatcher, and projection
        command_bus = AgentCommandBus()

        # Extract components from the command bus
        event_dispatcher = command_bus.dispatcher
        projection = command_bus.projection

        # Get observability bus
        event_bus = None
        if enable_observability:
            try:
                from victor.core.events import get_observability_bus

                event_bus = get_observability_bus()
            except ImportError:
                logger.debug("Observability not available")

        bridge = cls(
            command_bus=command_bus,
            event_dispatcher=event_dispatcher,
            projection=projection,
            event_bus=event_bus,
        )

        # Set up observability-to-CQRS forwarding
        if event_bus and enable_event_sourcing:
            bridge._obs_bridge = ObservabilityToCQRSBridge(
                event_bus=event_bus,
                event_dispatcher=event_dispatcher,
            )
            bridge._obs_bridge.start()

        return bridge

    # =========================================================================
    # Agent Connection
    # =========================================================================

    def connect_agent(
        self,
        agent: "Agent",
        session_id: Optional[str] = None,
    ) -> str:
        """Connect an agent to CQRS event flow.

        Creates an event adapter that forwards framework events
        to CQRS and observability subsystems.

        Args:
            agent: Framework Agent instance.
            session_id: Optional session ID (auto-generated if not provided).

        Returns:
            Session ID for this connection.
        """
        session_id = session_id or f"session-{uuid4().hex[:8]}"

        # Create adapter for this agent
        adapter = FrameworkEventAdapter(
            event_dispatcher=self._event_dispatcher,
            event_bus=self._event_bus,
            session_id=session_id,
            aggregate_id=session_id,
        )

        self._adapters[session_id] = adapter
        self._connected_agents.add(session_id)

        # Store bridge reference on agent
        agent._cqrs_bridge = self
        agent._cqrs_session_id = session_id
        agent._cqrs_adapter = adapter

        logger.debug(f"Connected agent to CQRS bridge (session: {session_id})")

        return session_id

    def disconnect_agent(self, session_id: str) -> None:
        """Disconnect an agent from CQRS event flow.

        Args:
            session_id: Session ID to disconnect.
        """
        if session_id in self._adapters:
            del self._adapters[session_id]

        if session_id in self._connected_agents:
            self._connected_agents.remove(session_id)

        logger.debug(f"Disconnected agent from CQRS bridge (session: {session_id})")

    def get_adapter(self, session_id: str) -> Optional[FrameworkEventAdapter]:
        """Get the event adapter for a session.

        Args:
            session_id: Session ID.

        Returns:
            FrameworkEventAdapter or None.
        """
        return self._adapters.get(session_id)

    # =========================================================================
    # Command Execution
    # =========================================================================

    async def start_session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Start a new session via CQRS command.

        Args:
            session_id: Optional session ID.
            metadata: Optional session metadata.

        Returns:
            Command result with session_id.
        """
        from victor.core.agent_commands import StartSessionCommand

        session_id = session_id or f"session-{uuid4().hex[:8]}"
        command = StartSessionCommand(
            session_id=session_id,
            metadata=metadata or {},
        )

        result = await self._command_bus.execute(command)
        return result.result if result.success else {"error": result.error}

    async def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a session via CQRS command.

        Args:
            session_id: Session to end.

        Returns:
            Command result.
        """
        from victor.core.agent_commands import EndSessionCommand

        command = EndSessionCommand(session_id=session_id)
        result = await self._command_bus.execute(command)
        return result.result if result.success else {"error": result.error}

    async def execute_tool(
        self,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a tool via CQRS command.

        Args:
            session_id: Session ID.
            tool_name: Tool to execute.
            arguments: Tool arguments.

        Returns:
            Command result with tool output.
        """
        from victor.core.agent_commands import ExecuteToolCommand

        command = ExecuteToolCommand(
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
        )

        result = await self._command_bus.execute(command)
        return result.result if result.success else {"error": result.error}

    async def send_chat(
        self,
        session_id: str,
        message: str,
        role: str = "user",
    ) -> Dict[str, Any]:
        """Send a chat message via CQRS command.

        Args:
            session_id: Session ID.
            message: Message content.
            role: Message role (user/assistant).

        Returns:
            Command result.
        """
        from victor.core.agent_commands import ChatCommand

        # Note: ChatCommand doesn't have a role field - it's derived from the message flow
        command = ChatCommand(
            session_id=session_id,
            message=message,
        )

        result = await self._command_bus.execute(command)
        return result.result if result.success else {"error": result.error}

    # =========================================================================
    # Queries
    # =========================================================================

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session details via CQRS query.

        Args:
            session_id: Session ID to retrieve.

        Returns:
            Session details.
        """
        from victor.core.agent_commands import GetSessionQuery

        query = GetSessionQuery(session_id=session_id)
        result = await self._command_bus.execute(query)
        return result.data if result.success else {"error": result.error}

    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get conversation history via CQRS query.

        Args:
            session_id: Session ID.
            limit: Maximum messages to retrieve.

        Returns:
            Conversation history.
        """
        from victor.core.agent_commands import GetConversationHistoryQuery

        query = GetConversationHistoryQuery(session_id=session_id, limit=limit)
        result = await self._command_bus.execute(query)
        return result.data if result.success else {"error": result.error}

    async def get_tools(
        self,
        session_id: str,
        filter_category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get available tools via CQRS query.

        Args:
            session_id: Session ID.
            filter_category: Optional category filter.

        Returns:
            Tool list.
        """
        from victor.core.agent_commands import GetToolsQuery

        query = GetToolsQuery(session_id=session_id, category=filter_category)
        result = await self._command_bus.execute(query)
        return result.data if result.success else {"error": result.error}

    async def get_metrics(self, session_id: str) -> Dict[str, Any]:
        """Get session metrics via CQRS query.

        Args:
            session_id: Session ID.

        Returns:
            Session metrics.
        """
        from victor.core.agent_commands import GetSessionMetricsQuery

        query = GetSessionMetricsQuery(session_id=session_id)
        result = await self._command_bus.execute(query)
        return result.data if result.success else {"error": result.error}

    # =========================================================================
    # Direct Projection Access
    # =========================================================================

    @property
    def projection(self) -> Optional["SessionProjection"]:
        """Get the session projection for direct access."""
        return self._projection

    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions from projection.

        Returns:
            List of session dictionaries.
        """
        if self._projection:
            return [
                {
                    "session_id": s.get("id", session_id),
                    "status": s.get("status", "unknown"),
                    "started_at": s.get("started_at"),
                    "message_count": len(s.get("messages", [])),
                    "tool_count": len(s.get("tool_calls", [])),
                }
                for session_id, s in self._projection.sessions.items()
            ]
        return []

    # =========================================================================
    # Event Subscription
    # =========================================================================

    def subscribe_to_events(
        self,
        handler: Callable[["CQRSEvent"], None],
        event_types: Optional[List[str]] = None,
    ) -> Callable[[], None]:
        """Subscribe to CQRS events.

        Args:
            handler: Event handler callback.
            event_types: Optional list of event types to filter.

        Returns:
            Unsubscribe function.
        """
        if not self._event_dispatcher:
            return lambda: None

        if event_types:
            # Filtered subscription
            def filtered_handler(event: "CQRSEvent") -> None:
                if event.event_type in event_types:
                    handler(event)

            self._event_dispatcher.subscribe_all(filtered_handler)
            # Return an unsubscribe function
            return lambda: self._event_dispatcher._all_handlers.remove(filtered_handler)
        else:
            self._event_dispatcher.subscribe_all(handler)
            # Return an unsubscribe function
            return lambda: self._event_dispatcher._all_handlers.remove(handler)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        """Clean up resources."""
        # Stop observability bridge
        if self._obs_bridge:
            self._obs_bridge.stop()

        # Clear adapters
        self._adapters.clear()
        self._connected_agents.clear()

        logger.debug("CQRSBridge closed")

    async def __aenter__(self) -> "CQRSBridge":
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"CQRSBridge(connected_agents={len(self._connected_agents)}, "
            f"adapters={len(self._adapters)})"
        )


# =============================================================================
# Factory Functions
# =============================================================================


async def create_cqrs_bridge(
    enable_event_sourcing: bool = True,
    enable_observability: bool = True,
) -> CQRSBridge:
    """Create a configured CQRS bridge.

    Convenience factory function.

    Args:
        enable_event_sourcing: Enable CQRS event sourcing.
        enable_observability: Enable observability integration.

    Returns:
        Configured CQRSBridge instance.
    """
    return await CQRSBridge.create(
        enable_event_sourcing=enable_event_sourcing,
        enable_observability=enable_observability,
    )


def create_event_adapter(
    session_id: str,
    event_dispatcher: Optional["EventDispatcher"] = None,
    event_bus: Optional["ObservabilityBus"] = None,
) -> FrameworkEventAdapter:
    """Create a framework event adapter.

    Args:
        session_id: Session ID for event correlation.
        event_dispatcher: CQRS EventDispatcher.
        event_bus: Observability EventBus.

    Returns:
        Configured FrameworkEventAdapter.
    """
    return FrameworkEventAdapter(
        event_dispatcher=event_dispatcher,
        event_bus=event_bus,
        session_id=session_id,
        aggregate_id=session_id,
    )
