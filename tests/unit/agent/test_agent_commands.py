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

"""Tests for Agent Commands CQRS integration."""

import pytest

from victor.core.agent_commands import (
    # Commands
    ChatCommand,
    ExecuteToolCommand,
    StartSessionCommand,
    EndSessionCommand,
    GetSessionQuery,
    GetConversationHistoryQuery,
    GetSessionMetricsQuery,
    # Events
    SessionStartedEvent,
    ChatMessageSentEvent,
    ChatResponseReceivedEvent,
    ToolExecutedEvent,
    SessionEndedEvent,
    SessionProjection,
    # Handlers
    StartSessionHandler,
    ChatHandler,
    ExecuteToolHandler,
    EndSessionHandler,
    GetSessionHandler,
    GetConversationHistoryHandler,
    create_agent_command_bus,
)
from victor.core.event_sourcing import InMemoryEventStore, EventDispatcher


# =============================================================================
# Command Tests
# =============================================================================


class TestChatCommand:
    """Tests for ChatCommand."""

    def test_creation_with_defaults(self):
        """Test creating command with defaults."""
        cmd = ChatCommand()

        assert cmd.session_id == ""
        assert cmd.message == ""
        assert cmd.provider == "anthropic"
        assert cmd.model is None
        assert cmd.tools is None
        assert cmd.thinking is False
        assert cmd.command_id is not None
        assert cmd.timestamp is not None

    def test_creation_with_values(self):
        """Test creating command with values."""
        cmd = ChatCommand(
            session_id="session-1",
            message="Hello, world!",
            provider="openai",
            model="gpt-4",
            tools=["read", "write"],
            thinking=True,
        )

        assert cmd.session_id == "session-1"
        assert cmd.message == "Hello, world!"
        assert cmd.provider == "openai"
        assert cmd.model == "gpt-4"
        assert cmd.tools == ["read", "write"]
        assert cmd.thinking is True


class TestExecuteToolCommand:
    """Tests for ExecuteToolCommand."""

    def test_creation(self):
        """Test creating command."""
        cmd = ExecuteToolCommand(
            session_id="session-1",
            tool_name="read_file",
            arguments={"path": "/tmp/test.txt"},
            dry_run=True,
        )

        assert cmd.session_id == "session-1"
        assert cmd.tool_name == "read_file"
        assert cmd.arguments == {"path": "/tmp/test.txt"}
        assert cmd.dry_run is True


class TestStartSessionCommand:
    """Tests for StartSessionCommand."""

    def test_creation(self):
        """Test creating command."""
        cmd = StartSessionCommand(
            working_directory="/home/user/project",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            tools=["read", "write", "shell"],
        )

        assert cmd.working_directory == "/home/user/project"
        assert cmd.provider == "anthropic"
        assert cmd.model == "claude-sonnet-4-20250514"
        assert cmd.tools == ["read", "write", "shell"]


class TestEndSessionCommand:
    """Tests for EndSessionCommand."""

    def test_creation(self):
        """Test creating command."""
        cmd = EndSessionCommand(
            session_id="session-1",
            save_history=False,
        )

        assert cmd.session_id == "session-1"
        assert cmd.save_history is False


# =============================================================================
# Query Tests
# =============================================================================


class TestGetSessionQuery:
    """Tests for GetSessionQuery."""

    def test_creation_with_defaults(self):
        """Test creating query with defaults."""
        query = GetSessionQuery(session_id="session-1")

        assert query.session_id == "session-1"
        assert query.include_history is False
        assert query.include_metrics is True
        assert query.query_id is not None

    def test_creation_with_values(self):
        """Test creating query with values."""
        query = GetSessionQuery(
            session_id="session-1",
            include_history=True,
            include_metrics=False,
        )

        assert query.include_history is True
        assert query.include_metrics is False


class TestGetConversationHistoryQuery:
    """Tests for GetConversationHistoryQuery."""

    def test_creation(self):
        """Test creating query."""
        query = GetConversationHistoryQuery(
            session_id="session-1",
            limit=50,
            offset=10,
        )

        assert query.session_id == "session-1"
        assert query.limit == 50
        assert query.offset == 10


class TestGetSessionMetricsQuery:
    """Tests for GetSessionMetricsQuery."""

    def test_creation(self):
        """Test creating query."""
        query = GetSessionMetricsQuery(
            session_id="session-1",
            metric_types=["message_count", "tool_call_count"],
        )

        assert query.session_id == "session-1"
        assert query.metric_types == ["message_count", "tool_call_count"]


# =============================================================================
# Event Tests
# =============================================================================


class TestSessionEvents:
    """Tests for session events."""

    def test_session_started_event(self):
        """Test SessionStartedEvent."""
        event = SessionStartedEvent(
            session_id="session-1",
            provider="anthropic",
            working_directory="/home/user",
        )

        assert event.session_id == "session-1"
        assert event.provider == "anthropic"
        assert event.working_directory == "/home/user"
        assert event.event_id is not None
        assert event.timestamp is not None

    def test_chat_message_sent_event(self):
        """Test ChatMessageSentEvent."""
        event = ChatMessageSentEvent(
            session_id="session-1",
            message="Hello!",
            role="user",
        )

        assert event.session_id == "session-1"
        assert event.message == "Hello!"
        assert event.role == "user"

    def test_chat_response_received_event(self):
        """Test ChatResponseReceivedEvent."""
        event = ChatResponseReceivedEvent(
            session_id="session-1",
            response="Hi there!",
            tokens_used=50,
            model="claude-sonnet-4-20250514",
        )

        assert event.session_id == "session-1"
        assert event.response == "Hi there!"
        assert event.tokens_used == 50
        assert event.model == "claude-sonnet-4-20250514"

    def test_tool_executed_event(self):
        """Test ToolExecutedEvent."""
        event = ToolExecutedEvent(
            session_id="session-1",
            tool_name="read_file",
            arguments={"path": "/tmp/test.txt"},
            success=True,
            result_summary="File read successfully",
            execution_time_ms=45,
        )

        assert event.tool_name == "read_file"
        assert event.arguments == {"path": "/tmp/test.txt"}
        assert event.success is True
        assert event.execution_time_ms == 45

    def test_session_ended_event(self):
        """Test SessionEndedEvent."""
        event = SessionEndedEvent(
            session_id="session-1",
            total_messages=10,
            total_tool_calls=5,
            duration_seconds=300.5,
        )

        assert event.total_messages == 10
        assert event.total_tool_calls == 5
        assert event.duration_seconds == 300.5


# =============================================================================
# Projection Tests
# =============================================================================


class TestSessionProjection:
    """Tests for SessionProjection."""

    @pytest.fixture
    def projection(self):
        """Create projection for tests."""
        return SessionProjection()

    @pytest.mark.asyncio
    async def test_handle_session_started(self, projection):
        """Test handling session started event."""
        event = SessionStartedEvent(
            session_id="session-1",
            provider="anthropic",
            working_directory="/home/user",
        )

        await projection.handle_SessionStartedEvent(event)

        session = projection.get_session("session-1")
        assert session is not None
        assert session["id"] == "session-1"
        assert session["provider"] == "anthropic"
        assert session["status"] == "active"

    @pytest.mark.asyncio
    async def test_handle_chat_messages(self, projection):
        """Test handling chat messages."""
        # Start session first
        await projection.handle_SessionStartedEvent(
            SessionStartedEvent(session_id="session-1", provider="anthropic")
        )

        # Add message
        await projection.handle_ChatMessageSentEvent(
            ChatMessageSentEvent(session_id="session-1", message="Hello!", role="user")
        )

        session = projection.get_session("session-1")
        assert len(session["messages"]) == 1
        assert session["messages"][0]["content"] == "Hello!"
        assert projection.message_counts["session-1"] == 1

    @pytest.mark.asyncio
    async def test_handle_tool_executed(self, projection):
        """Test handling tool execution."""
        await projection.handle_SessionStartedEvent(
            SessionStartedEvent(session_id="session-1", provider="anthropic")
        )

        await projection.handle_ToolExecutedEvent(
            ToolExecutedEvent(
                session_id="session-1",
                tool_name="read_file",
                success=True,
                result_summary="Success",
                execution_time_ms=50,
            )
        )

        session = projection.get_session("session-1")
        assert len(session["tool_calls"]) == 1
        assert session["tool_calls"][0]["tool"] == "read_file"
        assert projection.tool_counts["session-1"] == 1

    @pytest.mark.asyncio
    async def test_handle_session_ended(self, projection):
        """Test handling session ended."""
        await projection.handle_SessionStartedEvent(
            SessionStartedEvent(session_id="session-1", provider="anthropic")
        )

        await projection.handle_SessionEndedEvent(
            SessionEndedEvent(
                session_id="session-1",
                total_messages=5,
                total_tool_calls=3,
                duration_seconds=120.0,
            )
        )

        session = projection.get_session("session-1")
        assert session["status"] == "ended"
        assert session["duration_seconds"] == 120.0

    @pytest.mark.asyncio
    async def test_get_active_sessions(self, projection):
        """Test getting active sessions."""
        await projection.handle_SessionStartedEvent(
            SessionStartedEvent(session_id="session-1", provider="anthropic")
        )
        await projection.handle_SessionStartedEvent(
            SessionStartedEvent(session_id="session-2", provider="openai")
        )
        await projection.handle_SessionEndedEvent(SessionEndedEvent(session_id="session-1"))

        active = projection.get_active_sessions()
        assert len(active) == 1
        assert active[0]["id"] == "session-2"

    @pytest.mark.asyncio
    async def test_get_session_metrics(self, projection):
        """Test getting session metrics."""
        await projection.handle_SessionStartedEvent(
            SessionStartedEvent(session_id="session-1", provider="anthropic")
        )
        await projection.handle_ChatMessageSentEvent(
            ChatMessageSentEvent(session_id="session-1", message="Hello")
        )
        await projection.handle_ToolExecutedEvent(
            ToolExecutedEvent(session_id="session-1", tool_name="read_file")
        )

        metrics = projection.get_session_metrics("session-1")
        assert metrics["message_count"] == 1
        assert metrics["tool_call_count"] == 1
        assert metrics["status"] == "active"


# =============================================================================
# Command Handler Tests
# =============================================================================


class TestStartSessionHandler:
    """Tests for StartSessionHandler.

    Note: Handlers return raw dicts. The CommandBus wraps these in CommandResult.
    """

    @pytest.fixture
    def handler(self):
        """Create handler for tests."""
        store = InMemoryEventStore()
        dispatcher = EventDispatcher()
        projection = SessionProjection()
        dispatcher.subscribe_all(projection.handle)
        return StartSessionHandler(store, dispatcher), store, projection

    @pytest.mark.asyncio
    async def test_start_session(self, handler):
        """Test starting a session."""
        handler_instance, store, projection = handler

        cmd = StartSessionCommand(
            working_directory="/home/user",
            provider="anthropic",
        )

        # Handler returns raw dict, CommandBus wraps in CommandResult
        result = await handler_instance.handle(cmd)

        assert isinstance(result, dict)
        assert "session_id" in result
        assert result["session_id"].startswith("session-")

        # Check event was stored
        envelopes = await store.read(result["session_id"])
        assert len(envelopes) == 1
        assert isinstance(envelopes[0].event, SessionStartedEvent)

        # Check projection was updated
        session = projection.get_session(result["session_id"])
        assert session is not None


class TestChatHandler:
    """Tests for ChatHandler.

    Note: Handlers return raw dicts. The CommandBus wraps these in CommandResult.
    """

    @pytest.fixture
    def handler(self):
        """Create handler for tests."""
        store = InMemoryEventStore()
        dispatcher = EventDispatcher()
        projection = SessionProjection()
        dispatcher.subscribe_all(projection.handle)
        return ChatHandler(store, dispatcher), store, projection

    @pytest.mark.asyncio
    async def test_chat_message(self, handler):
        """Test sending a chat message."""
        handler_instance, store, projection = handler

        # First start a session
        await projection.handle_SessionStartedEvent(
            SessionStartedEvent(session_id="session-1", provider="anthropic")
        )

        cmd = ChatCommand(
            session_id="session-1",
            message="Hello, world!",
            provider="anthropic",
        )

        # Handler returns raw dict
        result = await handler_instance.handle(cmd)

        assert isinstance(result, dict)
        assert result["message_received"] is True

        # Check event was stored
        envelopes = await store.read("session-1")
        assert len(envelopes) == 1
        assert isinstance(envelopes[0].event, ChatMessageSentEvent)


class TestExecuteToolHandler:
    """Tests for ExecuteToolHandler.

    Note: Handlers return raw dicts. The CommandBus wraps these in CommandResult.
    """

    @pytest.fixture
    def handler(self):
        """Create handler for tests."""
        store = InMemoryEventStore()
        dispatcher = EventDispatcher()
        projection = SessionProjection()
        dispatcher.subscribe_all(projection.handle)
        return ExecuteToolHandler(store, dispatcher), store, projection

    @pytest.mark.asyncio
    async def test_execute_tool(self, handler):
        """Test executing a tool."""
        handler_instance, store, projection = handler

        # Start session
        await projection.handle_SessionStartedEvent(
            SessionStartedEvent(session_id="session-1", provider="anthropic")
        )

        cmd = ExecuteToolCommand(
            session_id="session-1",
            tool_name="read_file",
            arguments={"path": "/tmp/test.txt"},
        )

        # Handler returns raw dict
        result = await handler_instance.handle(cmd)

        assert isinstance(result, dict)
        assert result["tool"] == "read_file"
        assert "execution_time_ms" in result

    @pytest.mark.asyncio
    async def test_execute_tool_dry_run(self, handler):
        """Test tool dry run."""
        handler_instance, store, projection = handler

        await projection.handle_SessionStartedEvent(
            SessionStartedEvent(session_id="session-1", provider="anthropic")
        )

        cmd = ExecuteToolCommand(
            session_id="session-1",
            tool_name="write_file",
            arguments={"path": "/tmp/test.txt", "content": "test"},
            dry_run=True,
        )

        # Handler returns raw dict
        result = await handler_instance.handle(cmd)

        assert isinstance(result, dict)
        assert result["dry_run"] is True


class TestEndSessionHandler:
    """Tests for EndSessionHandler.

    Note: Handlers return raw dicts. The CommandBus wraps these in CommandResult.
    Errors are raised as exceptions, which CommandBus catches and wraps.
    """

    @pytest.fixture
    def handler(self):
        """Create handler for tests."""
        store = InMemoryEventStore()
        dispatcher = EventDispatcher()
        projection = SessionProjection()
        dispatcher.subscribe_all(projection.handle)
        return EndSessionHandler(store, dispatcher, projection), store, projection

    @pytest.mark.asyncio
    async def test_end_session(self, handler):
        """Test ending a session."""
        handler_instance, store, projection = handler

        # Start session first
        await projection.handle_SessionStartedEvent(
            SessionStartedEvent(session_id="session-1", provider="anthropic")
        )

        cmd = EndSessionCommand(session_id="session-1")

        # Handler returns raw dict
        result = await handler_instance.handle(cmd)

        assert isinstance(result, dict)
        assert result["session_id"] == "session-1"
        assert "metrics" in result
        assert "duration_seconds" in result

    @pytest.mark.asyncio
    async def test_end_nonexistent_session(self, handler):
        """Test ending non-existent session raises error."""
        handler_instance, store, projection = handler

        cmd = EndSessionCommand(session_id="nonexistent")

        # Handler raises ValueError for non-existent session
        with pytest.raises(ValueError) as exc_info:
            await handler_instance.handle(cmd)

        assert "not found" in str(exc_info.value)


# =============================================================================
# Query Handler Tests
# =============================================================================


class TestGetSessionHandler:
    """Tests for GetSessionHandler.

    Note: Query handlers return raw data. The QueryBus wraps this in QueryResult.
    """

    @pytest.fixture
    def handler(self):
        """Create handler for tests."""
        projection = SessionProjection()
        return GetSessionHandler(projection), projection

    @pytest.mark.asyncio
    async def test_get_session(self, handler):
        """Test getting session."""
        handler_instance, projection = handler

        # Add a session
        await projection.handle_SessionStartedEvent(
            SessionStartedEvent(
                session_id="session-1",
                provider="anthropic",
                working_directory="/home/user",
            )
        )

        query = GetSessionQuery(session_id="session-1")
        result = await handler_instance.handle(query)

        # Handler returns raw dict
        assert isinstance(result, dict)
        assert result["id"] == "session-1"
        assert result["provider"] == "anthropic"

    @pytest.mark.asyncio
    async def test_get_session_with_history(self, handler):
        """Test getting session with history."""
        handler_instance, projection = handler

        await projection.handle_SessionStartedEvent(
            SessionStartedEvent(session_id="session-1", provider="anthropic")
        )
        await projection.handle_ChatMessageSentEvent(
            ChatMessageSentEvent(session_id="session-1", message="Hello")
        )

        query = GetSessionQuery(session_id="session-1", include_history=True)
        result = await handler_instance.handle(query)

        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, handler):
        """Test getting non-existent session raises error."""
        handler_instance, projection = handler

        query = GetSessionQuery(session_id="nonexistent")

        # Handler raises ValueError for non-existent session
        with pytest.raises(ValueError) as exc_info:
            await handler_instance.handle(query)

        assert "not found" in str(exc_info.value)


class TestGetConversationHistoryHandler:
    """Tests for GetConversationHistoryHandler.

    Note: Query handlers return raw data. The QueryBus wraps this in QueryResult.
    """

    @pytest.fixture
    def handler(self):
        """Create handler for tests."""
        projection = SessionProjection()
        return GetConversationHistoryHandler(projection), projection

    @pytest.mark.asyncio
    async def test_get_history(self, handler):
        """Test getting conversation history."""
        handler_instance, projection = handler

        await projection.handle_SessionStartedEvent(
            SessionStartedEvent(session_id="session-1", provider="anthropic")
        )

        # Add multiple messages
        for i in range(5):
            await projection.handle_ChatMessageSentEvent(
                ChatMessageSentEvent(session_id="session-1", message=f"Message {i}", role="user")
            )

        query = GetConversationHistoryQuery(session_id="session-1", limit=3, offset=1)
        result = await handler_instance.handle(query)

        # Handler returns raw dict
        assert isinstance(result, dict)
        assert len(result["messages"]) == 3
        assert result["messages"][0]["content"] == "Message 1"
        assert result["total"] == 5


# =============================================================================
# AgentCommandBus Tests
# =============================================================================


class TestAgentCommandBus:
    """Tests for AgentCommandBus."""

    @pytest.fixture
    def bus(self):
        """Create bus for tests."""
        return create_agent_command_bus()

    @pytest.mark.asyncio
    async def test_start_and_end_session(self, bus):
        """Test starting and ending a session."""
        # Start session
        start_result = await bus.execute(
            StartSessionCommand(
                working_directory="/home/user",
                provider="anthropic",
            )
        )

        assert start_result.success is True
        session_id = start_result.result["session_id"]

        # Query session
        session_result = await bus.execute(GetSessionQuery(session_id=session_id))

        assert session_result.success is True
        assert session_result.data["status"] == "active"

        # End session
        end_result = await bus.execute(EndSessionCommand(session_id=session_id))

        assert end_result.success is True

        # Verify ended
        final_result = await bus.execute(GetSessionQuery(session_id=session_id))
        assert final_result.data["status"] == "ended"

    @pytest.mark.asyncio
    async def test_chat_flow(self, bus):
        """Test chat message flow."""
        # Start session
        start_result = await bus.execute(StartSessionCommand(provider="anthropic"))
        session_id = start_result.result["session_id"]

        # Send chat
        chat_result = await bus.execute(
            ChatCommand(
                session_id=session_id,
                message="Hello, world!",
            )
        )

        assert chat_result.success is True

        # Check history
        history_result = await bus.execute(GetConversationHistoryQuery(session_id=session_id))

        assert history_result.success is True
        assert len(history_result.data["messages"]) == 1
        assert history_result.data["messages"][0]["content"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_tool_execution_flow(self, bus):
        """Test tool execution flow."""
        # Start session
        start_result = await bus.execute(StartSessionCommand(provider="anthropic"))
        session_id = start_result.result["session_id"]

        # Execute tool
        tool_result = await bus.execute(
            ExecuteToolCommand(
                session_id=session_id,
                tool_name="read_file",
                arguments={"path": "/tmp/test.txt"},
            )
        )

        assert tool_result.success is True

        # Check metrics
        metrics_result = await bus.execute(GetSessionMetricsQuery(session_id=session_id))

        assert metrics_result.success is True
        assert metrics_result.data["tool_call_count"] == 1

    @pytest.mark.asyncio
    async def test_get_events(self, bus):
        """Test getting events for a session."""
        # Start session
        start_result = await bus.execute(StartSessionCommand(provider="anthropic"))
        session_id = start_result.result["session_id"]

        # Send chat
        await bus.execute(ChatCommand(session_id=session_id, message="Test"))

        # Get events
        events = await bus.get_events(session_id)

        assert len(events) == 2  # SessionStarted + ChatMessageSent
        assert isinstance(events[0], SessionStartedEvent)
        assert isinstance(events[1], ChatMessageSentEvent)

    @pytest.mark.asyncio
    async def test_full_workflow(self, bus):
        """Test complete workflow with all operations."""
        # Start session
        start_result = await bus.execute(
            StartSessionCommand(
                working_directory="/home/user/project",
                provider="anthropic",
                tools=["read", "write"],
            )
        )
        session_id = start_result.result["session_id"]

        # Send multiple messages
        await bus.execute(ChatCommand(session_id=session_id, message="Read the README"))
        await bus.execute(ChatCommand(session_id=session_id, message="Now write a test"))

        # Execute tools
        await bus.execute(
            ExecuteToolCommand(
                session_id=session_id,
                tool_name="read_file",
                arguments={"path": "README.md"},
            )
        )
        await bus.execute(
            ExecuteToolCommand(
                session_id=session_id,
                tool_name="write_file",
                arguments={"path": "test.py", "content": "# Test"},
            )
        )

        # Check final metrics
        metrics_result = await bus.execute(GetSessionMetricsQuery(session_id=session_id))

        assert metrics_result.data["message_count"] == 2
        assert metrics_result.data["tool_call_count"] == 2

        # End session
        end_result = await bus.execute(EndSessionCommand(session_id=session_id))

        assert end_result.success is True
        assert end_result.result["metrics"]["message_count"] == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestCQRSEventSourcingIntegration:
    """Tests for CQRS + Event Sourcing integration."""

    @pytest.mark.asyncio
    async def test_event_replay_rebuilds_state(self):
        """Test that events can rebuild projection state."""
        bus = create_agent_command_bus()

        # Create some state
        start_result = await bus.execute(StartSessionCommand(provider="anthropic"))
        session_id = start_result.result["session_id"]

        await bus.execute(ChatCommand(session_id=session_id, message="Hello"))
        await bus.execute(ExecuteToolCommand(session_id=session_id, tool_name="read_file"))

        # Get events
        events = await bus.get_events(session_id)
        assert len(events) == 3

        # Create new projection and replay
        new_projection = SessionProjection()
        dispatcher = EventDispatcher()
        dispatcher.subscribe_all(new_projection.handle)

        for event in events:
            await dispatcher.dispatch(event)

        # Verify rebuilt state matches
        rebuilt_session = new_projection.get_session(session_id)
        original_session = bus.projection.get_session(session_id)

        assert rebuilt_session["id"] == original_session["id"]
        assert rebuilt_session["provider"] == original_session["provider"]
        assert rebuilt_session["status"] == original_session["status"]

    @pytest.mark.asyncio
    async def test_concurrent_sessions(self):
        """Test handling multiple concurrent sessions."""
        bus = create_agent_command_bus()

        # Start multiple sessions
        sessions = []
        for i in range(3):
            result = await bus.execute(
                StartSessionCommand(
                    provider="anthropic",
                    working_directory=f"/project{i}",
                )
            )
            sessions.append(result.result["session_id"])

        # Send messages to each
        for i, session_id in enumerate(sessions):
            await bus.execute(
                ChatCommand(session_id=session_id, message=f"Message for session {i}")
            )

        # Verify isolation
        for i, session_id in enumerate(sessions):
            history = await bus.execute(GetConversationHistoryQuery(session_id=session_id))
            # history.data is a dict with "messages", "total", "offset", "limit"
            assert len(history.data["messages"]) == 1
            assert f"session {i}" in history.data["messages"][0]["content"]
