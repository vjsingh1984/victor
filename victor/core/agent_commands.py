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

"""CQRS Commands and Queries for Agent Operations.

This module provides CQRS-based command/query definitions for agent operations,
enabling clean separation of read and write operations and integration with
the event sourcing infrastructure.

Design Principles:
- Commands represent intent to change state (chat, tool execution, config changes)
- Queries represent read operations (session info, history, metrics)
- Events capture what happened (for audit trail and replay)

Example:
    from victor.core.agent_commands import (
        ChatCommand,
        ExecuteToolCommand,
        GetSessionQuery,
        AgentCommandBus,
    )

    # Setup
    bus = AgentCommandBus()

    # Send a chat command
    result = await bus.execute(ChatCommand(
        session_id="session-1",
        message="Write a hello world function",
        provider="anthropic",
    ))

    # Query session state
    session = await bus.execute(GetSessionQuery(session_id="session-1"))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type
from uuid import uuid4

from victor.core.cqrs import (
    Command,
    CommandBus,
    CommandHandler,
    CommandMiddleware,
    CommandResult,
    Query,
    QueryBus,
    QueryHandler,
    QueryResult,
    Mediator,
    LoggingCommandMiddleware,
    LoggingQueryMiddleware,
)
from victor.core.event_sourcing import (
    Event,
    EventDispatcher,
    InMemoryEventStore,
    Projection,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Agent Commands
# =============================================================================


@dataclass
class ChatCommand(Command):
    """Command to send a chat message to the agent.

    Attributes:
        session_id: Session identifier
        message: User message content
        provider: LLM provider name
        model: Optional model override
        tools: Optional list of enabled tools
        thinking: Enable extended thinking mode
    """

    session_id: str = ""
    message: str = ""
    provider: str = "anthropic"
    model: Optional[str] = None
    tools: Optional[List[str]] = None
    thinking: bool = False


@dataclass
class ExecuteToolCommand(Command):
    """Command to execute a specific tool.

    Attributes:
        session_id: Session identifier
        tool_name: Name of the tool to execute
        arguments: Tool arguments
        dry_run: If True, validate but don't execute
    """

    session_id: str = ""
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False


@dataclass
class CancelOperationCommand(Command):
    """Command to cancel an ongoing operation.

    Attributes:
        session_id: Session identifier
        operation_id: Optional specific operation to cancel
        reason: Cancellation reason
    """

    session_id: str = ""
    operation_id: Optional[str] = None
    reason: str = "User requested cancellation"


@dataclass
class SwitchProviderCommand(Command):
    """Command to switch the LLM provider.

    Attributes:
        session_id: Session identifier
        provider: New provider name
        model: Optional model for the provider
    """

    session_id: str = ""
    provider: str = ""
    model: Optional[str] = None


@dataclass
class UpdateConfigCommand(Command):
    """Command to update session configuration.

    Attributes:
        session_id: Session identifier
        config_updates: Dictionary of configuration updates
    """

    session_id: str = ""
    config_updates: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StartSessionCommand(Command):
    """Command to start a new agent session.

    Attributes:
        session_id: Optional session ID (auto-generated if not provided)
        working_directory: Working directory for the session
        provider: Initial provider
        model: Initial model
        tools: List of enabled tools
        metadata: Optional session metadata
    """

    session_id: Optional[str] = None
    working_directory: str = "."
    provider: str = "anthropic"
    model: Optional[str] = None
    tools: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EndSessionCommand(Command):
    """Command to end an agent session.

    Attributes:
        session_id: Session to end
        save_history: Whether to save conversation history
    """

    session_id: str = ""
    save_history: bool = True


# =============================================================================
# Agent Queries
# =============================================================================


@dataclass
class GetSessionQuery(Query[Dict[str, Any]]):
    """Query to get session information.

    Attributes:
        session_id: Session identifier
        include_history: Include message history
        include_metrics: Include session metrics
    """

    session_id: str = ""
    include_history: bool = False
    include_metrics: bool = True


@dataclass
class GetConversationHistoryQuery(Query[List[Dict[str, Any]]]):
    """Query to get conversation history.

    Attributes:
        session_id: Session identifier
        limit: Maximum messages to return
        offset: Messages to skip
    """

    session_id: str = ""
    limit: int = 100
    offset: int = 0


@dataclass
class GetToolsQuery(Query[List[Dict[str, Any]]]):
    """Query to get available tools.

    Attributes:
        session_id: Session identifier (for context-aware filtering)
        category: Optional tool category filter
        include_disabled: Include disabled tools
    """

    session_id: str = ""
    category: Optional[str] = None
    include_disabled: bool = False


@dataclass
class GetProvidersQuery(Query[List[Dict[str, Any]]]):
    """Query to get available providers.

    Attributes:
        include_unavailable: Include providers without API keys
    """

    include_unavailable: bool = False


@dataclass
class GetSessionMetricsQuery(Query[Dict[str, Any]]):
    """Query to get session metrics.

    Attributes:
        session_id: Session identifier
        metric_types: Specific metrics to retrieve
    """

    session_id: str = ""
    metric_types: Optional[List[str]] = None


@dataclass
class SearchCodeQuery(Query[List[Dict[str, Any]]]):
    """Query to search code in the workspace.

    Attributes:
        session_id: Session context
        query: Search query
        semantic: Use semantic search
        file_pattern: Glob pattern for files
        limit: Maximum results
    """

    session_id: str = ""
    query: str = ""
    semantic: bool = True
    file_pattern: Optional[str] = None
    limit: int = 20


# =============================================================================
# Agent Events (for Event Sourcing)
# =============================================================================


@dataclass
class SessionStartedEvent(Event):
    """Event when a session starts."""

    session_id: str = ""
    provider: str = ""
    working_directory: str = ""


@dataclass
class ChatMessageSentEvent(Event):
    """Event when a chat message is sent."""

    session_id: str = ""
    message: str = ""
    role: str = "user"


@dataclass
class ChatResponseReceivedEvent(Event):
    """Event when a chat response is received."""

    session_id: str = ""
    response: str = ""
    tokens_used: int = 0
    model: str = ""


@dataclass
class ToolExecutedEvent(Event):
    """Event when a tool is executed."""

    session_id: str = ""
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    result_summary: str = ""
    execution_time_ms: int = 0


@dataclass
class ProviderSwitchedEvent(Event):
    """Event when provider is switched."""

    session_id: str = ""
    old_provider: str = ""
    new_provider: str = ""


@dataclass
class SessionEndedEvent(Event):
    """Event when a session ends."""

    session_id: str = ""
    total_messages: int = 0
    total_tool_calls: int = 0
    duration_seconds: float = 0.0


@dataclass
class ErrorOccurredEvent(Event):
    """Event when an error occurs."""

    session_id: str = ""
    error_type: str = ""
    error_message: str = ""
    recoverable: bool = True


# =============================================================================
# Session Projection (Read Model)
# =============================================================================


class SessionProjection(Projection):
    """Projection that builds session state from events.

    Provides optimized read access to session data.
    """

    def __init__(self):
        """Initialize session projection."""
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.message_counts: Dict[str, int] = {}
        self.tool_counts: Dict[str, int] = {}

    async def handle_SessionStartedEvent(self, event: SessionStartedEvent) -> None:
        """Handle session start."""
        self.sessions[event.session_id] = {
            "id": event.session_id,
            "provider": event.provider,
            "working_directory": event.working_directory,
            "started_at": event.timestamp,
            "status": "active",
            "messages": [],
            "tool_calls": [],
        }
        self.message_counts[event.session_id] = 0
        self.tool_counts[event.session_id] = 0

    async def handle_ChatMessageSentEvent(self, event: ChatMessageSentEvent) -> None:
        """Handle chat message."""
        if event.session_id in self.sessions:
            self.sessions[event.session_id]["messages"].append(
                {
                    "role": event.role,
                    "content": event.message,
                    "timestamp": event.timestamp,
                }
            )
            self.message_counts[event.session_id] += 1

    async def handle_ChatResponseReceivedEvent(self, event: ChatResponseReceivedEvent) -> None:
        """Handle chat response."""
        if event.session_id in self.sessions:
            self.sessions[event.session_id]["messages"].append(
                {
                    "role": "assistant",
                    "content": event.response,
                    "timestamp": event.timestamp,
                    "model": event.model,
                    "tokens": event.tokens_used,
                }
            )
            self.message_counts[event.session_id] += 1

    async def handle_ToolExecutedEvent(self, event: ToolExecutedEvent) -> None:
        """Handle tool execution."""
        if event.session_id in self.sessions:
            self.sessions[event.session_id]["tool_calls"].append(
                {
                    "tool": event.tool_name,
                    "arguments": event.arguments,
                    "success": event.success,
                    "result": event.result_summary,
                    "execution_time_ms": event.execution_time_ms,
                    "timestamp": event.timestamp,
                }
            )
            self.tool_counts[event.session_id] += 1

    async def handle_ProviderSwitchedEvent(self, event: ProviderSwitchedEvent) -> None:
        """Handle provider switch."""
        if event.session_id in self.sessions:
            self.sessions[event.session_id]["provider"] = event.new_provider

    async def handle_SessionEndedEvent(self, event: SessionEndedEvent) -> None:
        """Handle session end."""
        if event.session_id in self.sessions:
            self.sessions[event.session_id]["status"] = "ended"
            self.sessions[event.session_id]["ended_at"] = event.timestamp
            self.sessions[event.session_id]["duration_seconds"] = event.duration_seconds

    async def handle_ErrorOccurredEvent(self, event: ErrorOccurredEvent) -> None:
        """Handle error event."""
        if event.session_id in self.sessions:
            if "errors" not in self.sessions[event.session_id]:
                self.sessions[event.session_id]["errors"] = []
            self.sessions[event.session_id]["errors"].append(
                {
                    "type": event.error_type,
                    "message": event.error_message,
                    "recoverable": event.recoverable,
                    "timestamp": event.timestamp,
                }
            )

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        return self.sessions.get(session_id)

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all active sessions."""
        return [s for s in self.sessions.values() if s.get("status") == "active"]

    def get_session_metrics(self, session_id: str) -> Dict[str, Any]:
        """Get metrics for a session."""
        session = self.sessions.get(session_id)
        if not session:
            return {}

        return {
            "message_count": self.message_counts.get(session_id, 0),
            "tool_call_count": self.tool_counts.get(session_id, 0),
            "status": session.get("status", "unknown"),
            "provider": session.get("provider", "unknown"),
            "error_count": len(session.get("errors", [])),
        }

    async def handle(self, event: Event) -> None:
        """Handle an event and update projection.

        Routes events to their specific handlers based on event type.

        Args:
            event: Event to handle.
        """
        handler_name = f"handle_{type(event).__name__}"
        handler = getattr(self, handler_name, None)
        if handler:
            await handler(event)

    async def rebuild(self, events: List[Event]) -> None:
        """Rebuild projection from event history.

        Args:
            events: All events to replay.
        """
        # Clear current state
        self.sessions.clear()
        self.message_counts.clear()
        self.tool_counts.clear()

        # Replay all events
        for event in events:
            await self.handle(event)


# =============================================================================
# Command Handlers
# =============================================================================


class StartSessionHandler(CommandHandler[StartSessionCommand]):
    """Handler for starting agent sessions."""

    def __init__(
        self,
        event_store: InMemoryEventStore,
        dispatcher: EventDispatcher,
    ):
        self.event_store = event_store
        self.dispatcher = dispatcher

    async def handle(self, command: StartSessionCommand) -> Dict[str, Any]:
        """Handle start session command.

        Returns:
            Dict with session_id. The CommandBus wraps this in CommandResult.
        """
        # Use provided session_id or generate one
        session_id = command.session_id or f"session-{uuid4().hex[:8]}"

        event = SessionStartedEvent(
            session_id=session_id,
            provider=command.provider,
            working_directory=command.working_directory,
        )

        await self.event_store.append(session_id, [event])
        await self.dispatcher.dispatch(event)

        logger.info(f"Session started: {session_id}")

        return {"session_id": session_id}


class ChatHandler(CommandHandler[ChatCommand]):
    """Handler for chat commands.

    This is a placeholder that demonstrates the pattern.
    In production, this would integrate with AgentOrchestrator.
    """

    def __init__(
        self,
        event_store: InMemoryEventStore,
        dispatcher: EventDispatcher,
    ):
        self.event_store = event_store
        self.dispatcher = dispatcher

    async def handle(self, command: ChatCommand) -> Dict[str, Any]:
        """Handle chat command.

        Returns:
            Dict with chat result. The CommandBus wraps this in CommandResult.
        """
        # Record the user message event
        user_event = ChatMessageSentEvent(
            session_id=command.session_id,
            message=command.message,
            role="user",
        )
        await self.event_store.append(command.session_id, [user_event])
        await self.dispatcher.dispatch(user_event)

        # In production, this would call the actual orchestrator
        # For now, return a placeholder result
        return {
            "session_id": command.session_id,
            "message_received": True,
            "provider": command.provider,
        }


class ExecuteToolHandler(CommandHandler[ExecuteToolCommand]):
    """Handler for tool execution commands."""

    def __init__(
        self,
        event_store: InMemoryEventStore,
        dispatcher: EventDispatcher,
    ):
        self.event_store = event_store
        self.dispatcher = dispatcher

    async def handle(self, command: ExecuteToolCommand) -> Dict[str, Any]:
        """Handle tool execution command.

        Returns:
            Dict with tool execution result. The CommandBus wraps this in CommandResult.
        """
        import time

        start_time = time.time()

        # In production, this would execute the actual tool
        # For demonstration, simulate execution
        success = not command.dry_run

        execution_time_ms = int((time.time() - start_time) * 1000)

        event = ToolExecutedEvent(
            session_id=command.session_id,
            tool_name=command.tool_name,
            arguments=command.arguments,
            success=success,
            result_summary="Tool executed successfully" if success else "Dry run",
            execution_time_ms=execution_time_ms,
        )

        await self.event_store.append(command.session_id, [event])
        await self.dispatcher.dispatch(event)

        return {
            "tool": command.tool_name,
            "execution_time_ms": execution_time_ms,
            "dry_run": command.dry_run,
            "success": success,
        }


class EndSessionHandler(CommandHandler[EndSessionCommand]):
    """Handler for ending sessions."""

    def __init__(
        self,
        event_store: InMemoryEventStore,
        dispatcher: EventDispatcher,
        projection: SessionProjection,
    ):
        self.event_store = event_store
        self.dispatcher = dispatcher
        self.projection = projection

    async def handle(self, command: EndSessionCommand) -> Dict[str, Any]:
        """Handle end session command.

        Returns:
            Dict with session end result. The CommandBus wraps this in CommandResult.

        Raises:
            ValueError: If session not found.
        """
        session = self.projection.get_session(command.session_id)
        if not session:
            raise ValueError(f"Session {command.session_id} not found")

        metrics = self.projection.get_session_metrics(command.session_id)

        # Calculate duration
        started_at = session.get("started_at")
        duration = 0.0
        if started_at:
            from datetime import datetime as dt, timezone

            duration = (dt.now(timezone.utc) - started_at).total_seconds()

        event = SessionEndedEvent(
            session_id=command.session_id,
            total_messages=metrics.get("message_count", 0),
            total_tool_calls=metrics.get("tool_call_count", 0),
            duration_seconds=duration,
        )

        await self.event_store.append(command.session_id, [event])
        await self.dispatcher.dispatch(event)

        logger.info(f"Session ended: {command.session_id}")

        return {
            "session_id": command.session_id,
            "metrics": metrics,
            "duration_seconds": duration,
        }


# =============================================================================
# Query Handlers
# =============================================================================


class GetSessionHandler(QueryHandler[Dict[str, Any]]):
    """Handler for session queries.

    Note: Returns raw dict data. The QueryBus wraps this in QueryResult.
    """

    def __init__(self, projection: SessionProjection):
        self.projection = projection

    async def handle(self, query: GetSessionQuery) -> Dict[str, Any]:
        """Handle get session query.

        Returns:
            Dict with session data. The QueryBus wraps this in QueryResult.

        Raises:
            ValueError: If session not found.
        """
        session = self.projection.get_session(query.session_id)

        if not session:
            raise ValueError(f"Session {query.session_id} not found")

        result = {
            "id": session["id"],
            "provider": session["provider"],
            "status": session["status"],
            "working_directory": session["working_directory"],
        }

        if query.include_history:
            result["messages"] = session.get("messages", [])
            result["tool_calls"] = session.get("tool_calls", [])

        if query.include_metrics:
            result["metrics"] = self.projection.get_session_metrics(query.session_id)

        return result


class GetConversationHistoryHandler(QueryHandler[Dict[str, Any]]):
    """Handler for conversation history queries.

    Note: Returns raw dict data. The QueryBus wraps this in QueryResult.
    """

    def __init__(self, projection: SessionProjection):
        self.projection = projection

    async def handle(self, query: GetConversationHistoryQuery) -> Dict[str, Any]:
        """Handle get conversation history query.

        Returns:
            Dict with messages and pagination metadata. The QueryBus wraps this in QueryResult.

        Raises:
            ValueError: If session not found.
        """
        session = self.projection.get_session(query.session_id)

        if not session:
            raise ValueError(f"Session {query.session_id} not found")

        messages = session.get("messages", [])

        # Apply pagination
        start = query.offset
        end = query.offset + query.limit
        paginated = messages[start:end]

        return {
            "messages": paginated,
            "total": len(messages),
            "offset": query.offset,
            "limit": query.limit,
        }


class GetSessionMetricsHandler(QueryHandler[Dict[str, Any]]):
    """Handler for session metrics queries.

    Note: Returns raw dict data. The QueryBus wraps this in QueryResult.
    """

    def __init__(self, projection: SessionProjection):
        self.projection = projection

    async def handle(self, query: GetSessionMetricsQuery) -> Dict[str, Any]:
        """Handle get session metrics query.

        Returns:
            Dict with session metrics. The QueryBus wraps this in QueryResult.

        Raises:
            ValueError: If session not found.
        """
        metrics = self.projection.get_session_metrics(query.session_id)

        if not metrics:
            raise ValueError(f"Session {query.session_id} not found")

        # Filter metrics if specific types requested
        if query.metric_types:
            metrics = {k: v for k, v in metrics.items() if k in query.metric_types}

        return metrics


# =============================================================================
# Agent Command Bus (Pre-configured)
# =============================================================================


class AgentCommandBus:
    """Pre-configured command/query bus for agent operations.

    Provides a unified interface for sending commands and queries
    related to agent operations, with built-in event sourcing.

    Example:
        bus = AgentCommandBus()

        # Start a session
        result = await bus.execute(StartSessionCommand(
            provider="anthropic",
            working_directory="/path/to/project",
        ))
        session_id = result.data["session_id"]

        # Send a chat
        await bus.execute(ChatCommand(
            session_id=session_id,
            message="Hello, world!",
        ))

        # Query session
        session = await bus.execute(GetSessionQuery(session_id=session_id))
    """

    def __init__(self):
        """Initialize the agent command bus with all infrastructure."""
        # Event infrastructure
        self.event_store = InMemoryEventStore()
        self.dispatcher = EventDispatcher()
        self.projection = SessionProjection()

        # Register projection with dispatcher
        self.dispatcher.subscribe_all(self.projection.handle)

        # Create buses
        self.command_bus = CommandBus()
        self.query_bus = QueryBus()

        # Add middleware
        self.command_bus.use(LoggingCommandMiddleware())
        self.query_bus.use(LoggingQueryMiddleware())

        # Register command handlers
        self._register_command_handlers()

        # Register query handlers
        self._register_query_handlers()

        # Create mediator
        self.mediator = Mediator(self.command_bus, self.query_bus)

    def _register_command_handlers(self) -> None:
        """Register all command handlers."""
        self.command_bus.register(
            StartSessionCommand,
            StartSessionHandler(self.event_store, self.dispatcher),
        )
        self.command_bus.register(
            ChatCommand,
            ChatHandler(self.event_store, self.dispatcher),
        )
        self.command_bus.register(
            ExecuteToolCommand,
            ExecuteToolHandler(self.event_store, self.dispatcher),
        )
        self.command_bus.register(
            EndSessionCommand,
            EndSessionHandler(self.event_store, self.dispatcher, self.projection),
        )

    def _register_query_handlers(self) -> None:
        """Register all query handlers."""
        self.query_bus.register(
            GetSessionQuery,
            GetSessionHandler(self.projection),
        )
        self.query_bus.register(
            GetConversationHistoryQuery,
            GetConversationHistoryHandler(self.projection),
        )
        self.query_bus.register(
            GetSessionMetricsQuery,
            GetSessionMetricsHandler(self.projection),
        )

    async def execute(self, message: Command | Query) -> CommandResult | QueryResult:
        """Execute a command or query.

        Args:
            message: Command or Query to execute

        Returns:
            CommandResult or QueryResult
        """
        return await self.mediator.send(message)

    async def get_events(self, session_id: str) -> List[Event]:
        """Get all events for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of events
        """
        envelopes = await self.event_store.read(session_id)
        return [env.event for env in envelopes]


# =============================================================================
# Factory Functions
# =============================================================================


def create_agent_command_bus() -> AgentCommandBus:
    """Create a fully configured agent command bus.

    Returns:
        Configured AgentCommandBus instance
    """
    return AgentCommandBus()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Commands
    "ChatCommand",
    "ExecuteToolCommand",
    "CancelOperationCommand",
    "SwitchProviderCommand",
    "UpdateConfigCommand",
    "StartSessionCommand",
    "EndSessionCommand",
    # Queries
    "GetSessionQuery",
    "GetConversationHistoryQuery",
    "GetToolsQuery",
    "GetProvidersQuery",
    "GetSessionMetricsQuery",
    "SearchCodeQuery",
    # Events
    "SessionStartedEvent",
    "ChatMessageSentEvent",
    "ChatResponseReceivedEvent",
    "ToolExecutedEvent",
    "ProviderSwitchedEvent",
    "SessionEndedEvent",
    "ErrorOccurredEvent",
    # Projection
    "SessionProjection",
    # Handlers
    "StartSessionHandler",
    "ChatHandler",
    "ExecuteToolHandler",
    "EndSessionHandler",
    "GetSessionHandler",
    "GetConversationHistoryHandler",
    "GetSessionMetricsHandler",
    # Bus
    "AgentCommandBus",
    # Factory
    "create_agent_command_bus",
]
