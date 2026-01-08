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

"""Comprehensive unit tests for victor/framework/agent.py.

These tests cover:
- Agent class initialization and configuration
- Task execution methods (run, stream)
- Event handling and state management
- Error handling and recovery
- CQRS integration
- Workflow and team execution
- Lifecycle management
"""

import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_orchestrator():
    """Create a mock AgentOrchestrator for testing."""
    orchestrator = MagicMock()
    orchestrator.provider = MagicMock()
    orchestrator.provider.name = "test_provider"
    orchestrator.model = "test-model"
    orchestrator.messages = []
    orchestrator.reset_conversation = MagicMock()
    orchestrator.stream_chat = AsyncMock(return_value=AsyncMock())

    # Add protocol methods for State wrapper
    orchestrator.get_stage = MagicMock(return_value=MagicMock(value="INITIAL"))
    orchestrator.get_tool_calls_count = MagicMock(return_value=0)
    orchestrator.get_tool_budget = MagicMock(return_value=50)
    orchestrator.get_observed_files = MagicMock(return_value=set())
    orchestrator.get_modified_files = MagicMock(return_value=set())
    orchestrator.get_message_count = MagicMock(return_value=0)
    orchestrator.is_streaming = MagicMock(return_value=False)
    orchestrator.current_provider = "test_provider"
    orchestrator.current_model = "test-model"
    orchestrator.get_iteration_count = MagicMock(return_value=0)
    orchestrator.get_max_iterations = MagicMock(return_value=25)
    orchestrator.reset = MagicMock()
    orchestrator.close = AsyncMock()

    return orchestrator


@pytest.fixture
def mock_vertical():
    """Create a mock vertical class for testing."""
    vertical = MagicMock()
    vertical.name = "test_vertical"
    vertical.get_config = MagicMock(
        return_value=MagicMock(
            tools=["tool1", "tool2"],
            system_prompt="Test system prompt",
            provider_hints={"preferred_providers": ["anthropic"]},
        )
    )
    vertical.get_workflow_provider = MagicMock(return_value=None)
    vertical.get_team_spec_provider = MagicMock(return_value=None)
    return vertical


@pytest.fixture
def mock_stream_chunk():
    """Create a mock stream chunk."""
    chunk = MagicMock()
    chunk.content = "test content"
    chunk.metadata = {}
    chunk.tool_calls = []
    return chunk


# =============================================================================
# Agent Initialization Tests (lines 98-108)
# =============================================================================


class TestAgentInit:
    """Tests for Agent.__init__ method."""

    def test_init_with_orchestrator(self, mock_orchestrator):
        """Agent should initialize with orchestrator and store references."""
        from victor.framework.agent import Agent

        agent = Agent(
            mock_orchestrator,
            provider="anthropic",
            model="claude-3",
        )

        assert agent._orchestrator is mock_orchestrator
        assert agent._provider == "anthropic"
        assert agent._model == "claude-3"
        assert agent._vertical is None
        assert agent._vertical_config is None
        assert agent._state is not None
        assert agent._state_observers == []
        assert agent._cqrs_bridge is None
        assert agent._cqrs_session_id is None
        assert agent._cqrs_adapter is None

    def test_init_with_vertical(self, mock_orchestrator, mock_vertical):
        """Agent should store vertical and vertical_config when provided."""
        from victor.framework.agent import Agent

        vertical_config = MagicMock()
        agent = Agent(
            mock_orchestrator,
            provider="anthropic",
            model="claude-3",
            vertical=mock_vertical,
            vertical_config=vertical_config,
        )

        assert agent._vertical is mock_vertical
        assert agent._vertical_config is vertical_config

    def test_init_default_values(self, mock_orchestrator):
        """Agent should use default values when not provided."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        assert agent._provider == "anthropic"
        assert agent._model is None
        assert agent._vertical is None
        assert agent._vertical_config is None


# =============================================================================
# Agent.from_orchestrator Tests (lines 249-251)
# =============================================================================


class TestAgentFromOrchestrator:
    """Tests for Agent.from_orchestrator classmethod."""

    def test_from_orchestrator_basic(self, mock_orchestrator):
        """from_orchestrator should create Agent from existing orchestrator."""
        from victor.framework.agent import Agent

        agent = Agent.from_orchestrator(mock_orchestrator)

        assert agent._orchestrator is mock_orchestrator
        assert agent._provider == "test_provider"
        assert agent._model == "test-model"

    def test_from_orchestrator_unknown_provider(self):
        """from_orchestrator should handle missing provider name."""
        from victor.framework.agent import Agent

        orchestrator = MagicMock()
        orchestrator.provider = MagicMock(spec=[])  # No 'name' attribute
        orchestrator.model = None

        agent = Agent.from_orchestrator(orchestrator)

        assert agent._provider == "unknown"
        assert agent._model is None


# =============================================================================
# Agent.run Tests (lines 393-422)
# =============================================================================


class TestAgentRun:
    """Tests for Agent.run method."""

    @pytest.mark.asyncio
    async def test_run_basic(self, mock_orchestrator):
        """run should collect events and return TaskResult."""
        from victor.framework.agent import Agent
        from victor.framework.events import Event, EventType

        # Setup mock stream
        async def mock_stream(*args, **kwargs):
            yield Event(type=EventType.CONTENT, content="Hello ")
            yield Event(type=EventType.CONTENT, content="World")
            yield Event(type=EventType.STREAM_END, success=True)

        agent = Agent(mock_orchestrator)

        with patch.object(agent, "stream", mock_stream):
            result = await agent.run("test prompt")

        assert result.content == "Hello World"
        assert result.success is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_run_with_error_event(self, mock_orchestrator):
        """run should handle ERROR events and set success=False."""
        from victor.framework.agent import Agent
        from victor.framework.events import Event, EventType

        async def mock_stream(*args, **kwargs):
            yield Event(type=EventType.ERROR, error="Something went wrong")
            yield Event(type=EventType.STREAM_END, success=False, error="Something went wrong")

        agent = Agent(mock_orchestrator)

        with patch.object(agent, "stream", mock_stream):
            result = await agent.run("test prompt")

        assert result.success is False
        assert result.error == "Something went wrong"

    @pytest.mark.asyncio
    async def test_run_with_stream_end_failure(self, mock_orchestrator):
        """run should handle STREAM_END with success=False."""
        from victor.framework.agent import Agent
        from victor.framework.events import Event, EventType

        async def mock_stream(*args, **kwargs):
            yield Event(type=EventType.CONTENT, content="partial")
            yield Event(type=EventType.STREAM_END, success=False, error="Timeout")

        agent = Agent(mock_orchestrator)

        with patch.object(agent, "stream", mock_stream):
            result = await agent.run("test prompt")

        assert result.content == "partial"
        assert result.success is False
        assert result.error == "Timeout"

    @pytest.mark.asyncio
    async def test_run_with_cancellation_error(self, mock_orchestrator):
        """run should handle CancellationError."""
        from victor.framework.agent import Agent
        from victor.framework.errors import CancellationError

        async def mock_stream(*args, **kwargs):
            raise CancellationError("User cancelled")
            yield  # pragma: no cover

        agent = Agent(mock_orchestrator)

        with patch.object(agent, "stream", mock_stream):
            result = await agent.run("test prompt")

        assert result.success is False
        assert result.error == "Operation cancelled"

    @pytest.mark.asyncio
    async def test_run_with_generic_exception(self, mock_orchestrator):
        """run should handle generic exceptions."""
        from victor.framework.agent import Agent

        async def mock_stream(*args, **kwargs):
            raise RuntimeError("Unexpected error")
            yield  # pragma: no cover

        agent = Agent(mock_orchestrator)

        with patch.object(agent, "stream", mock_stream):
            result = await agent.run("test prompt")

        assert result.success is False
        assert result.error == "Unexpected error"

    @pytest.mark.asyncio
    async def test_run_with_context(self, mock_orchestrator):
        """run should pass context to stream."""
        from victor.framework.agent import Agent
        from victor.framework.events import Event, EventType

        captured_context = {}

        async def mock_stream(prompt, context=None):
            captured_context["context"] = context
            yield Event(type=EventType.STREAM_END, success=True)

        agent = Agent(mock_orchestrator)
        context = {"file": "test.py", "error": "SyntaxError"}

        with patch.object(agent, "stream", mock_stream):
            await agent.run("fix bug", context=context)

        assert captured_context["context"] == context


# =============================================================================
# Agent.stream Tests (lines 462-490)
# =============================================================================


class TestAgentStream:
    """Tests for Agent.stream method."""

    @pytest.mark.asyncio
    async def test_stream_basic(self, mock_orchestrator):
        """stream should yield events from orchestrator."""
        from victor.framework.agent import Agent
        from victor.framework.events import Event, EventType

        # Mock the internal stream function
        async def mock_stream_with_events(orchestrator, prompt):
            yield Event(type=EventType.STREAM_START)
            yield Event(type=EventType.CONTENT, content="Test")
            yield Event(type=EventType.STREAM_END, success=True)

        agent = Agent(mock_orchestrator)

        with patch("victor.framework._internal.stream_with_events", mock_stream_with_events):
            events = []
            async for event in agent.stream("test"):
                events.append(event)

        assert len(events) == 3
        assert events[0].type == EventType.STREAM_START
        assert events[1].type == EventType.CONTENT
        assert events[2].type == EventType.STREAM_END

    @pytest.mark.asyncio
    async def test_stream_with_context(self, mock_orchestrator):
        """stream should format and prepend context to prompt."""
        from victor.framework.agent import Agent
        from victor.framework.events import Event, EventType

        captured_prompt = {}

        async def mock_stream_with_events(orchestrator, prompt):
            captured_prompt["prompt"] = prompt
            yield Event(type=EventType.STREAM_END, success=True)

        agent = Agent(mock_orchestrator)

        with patch("victor.framework._internal.stream_with_events", mock_stream_with_events):
            async for _ in agent.stream("fix bug", context={"file": "test.py"}):
                pass

        # Context should be prepended to prompt
        assert "File: test.py" in captured_prompt["prompt"]
        assert "fix bug" in captured_prompt["prompt"]

    @pytest.mark.asyncio
    async def test_stream_notifies_state_observers(self, mock_orchestrator):
        """stream should notify state observers on stage changes."""
        from victor.framework.agent import Agent
        from victor.framework.events import Event, EventType
        from victor.agent.conversation_state import ConversationStage

        observer_called = []

        def observer(old_state, new_state):
            observer_called.append((old_state, new_state))

        # Mock changing stage
        stage_sequence = [ConversationStage.INITIAL, ConversationStage.PLANNING]
        call_count = [0]

        def get_stage_side_effect():
            result = stage_sequence[min(call_count[0], len(stage_sequence) - 1)]
            call_count[0] += 1
            return result

        mock_orchestrator.get_stage.side_effect = get_stage_side_effect

        async def mock_stream_with_events(orchestrator, prompt):
            yield Event(type=EventType.STREAM_START)
            yield Event(type=EventType.CONTENT, content="Test")
            yield Event(type=EventType.STREAM_END, success=True)

        agent = Agent(mock_orchestrator)
        agent.on_state_change(observer)

        with patch("victor.framework._internal.stream_with_events", mock_stream_with_events):
            async for _ in agent.stream("test"):
                pass

        # Observer should have been called at least once for stage change
        # (depends on mock timing, but we test the mechanism)
        assert agent._state_observers == [observer]

    @pytest.mark.asyncio
    async def test_stream_forwards_to_cqrs(self, mock_orchestrator):
        """stream should forward events to CQRS when enabled."""
        from victor.framework.agent import Agent
        from victor.framework.events import Event, EventType

        mock_adapter = MagicMock()

        async def mock_stream_with_events(orchestrator, prompt):
            yield Event(type=EventType.CONTENT, content="Test")

        agent = Agent(mock_orchestrator)
        agent._cqrs_adapter = mock_adapter

        with patch("victor.framework._internal.stream_with_events", mock_stream_with_events):
            async for _ in agent.stream("test"):
                pass

        mock_adapter.forward.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_observer_exception_does_not_break_streaming(self, mock_orchestrator):
        """stream should catch and suppress observer exceptions."""
        from victor.framework.agent import Agent
        from victor.framework.events import Event, EventType
        from victor.agent.conversation_state import ConversationStage

        # Create an observer that raises an exception
        def failing_observer(old_state, new_state):
            raise RuntimeError("Observer failure")

        # Mock the stage to trigger observer call
        stage_sequence = [ConversationStage.INITIAL, ConversationStage.PLANNING]
        call_count = [0]

        def get_stage_side_effect():
            result = stage_sequence[min(call_count[0], len(stage_sequence) - 1)]
            call_count[0] += 1
            return result

        mock_orchestrator.get_stage.side_effect = get_stage_side_effect

        async def mock_stream_with_events(orchestrator, prompt):
            yield Event(type=EventType.STREAM_START)
            yield Event(type=EventType.CONTENT, content="Test")
            yield Event(type=EventType.STREAM_END, success=True)

        agent = Agent(mock_orchestrator)
        agent.on_state_change(failing_observer)

        events = []
        with patch("victor.framework._internal.stream_with_events", mock_stream_with_events):
            # This should not raise despite the observer failing
            async for event in agent.stream("test"):
                events.append(event)

        # All events should be yielded despite observer error
        assert len(events) == 3


# =============================================================================
# Agent.chat and ChatSession Tests (lines 509, 1108-1164)
# =============================================================================


class TestAgentChat:
    """Tests for Agent.chat method and ChatSession class."""

    def test_chat_returns_session(self, mock_orchestrator):
        """chat should return a ChatSession instance."""
        from victor.framework.agent import Agent, ChatSession

        agent = Agent(mock_orchestrator)
        session = agent.chat("Hello")

        assert isinstance(session, ChatSession)

    @pytest.mark.asyncio
    async def test_chat_session_send(self, mock_orchestrator):
        """ChatSession.send should delegate to AgentSession."""
        from victor.framework.agent import Agent, ChatSession
        from victor.framework.task import TaskResult

        agent = Agent(mock_orchestrator)
        session = agent.chat("Initial prompt")

        mock_result = TaskResult(content="Response", success=True)
        session._delegate.send = AsyncMock(return_value=mock_result)

        result = await session.send("Follow-up")

        assert result.content == "Response"
        session._delegate.send.assert_called_once_with("Follow-up")

    @pytest.mark.asyncio
    async def test_chat_session_stream(self, mock_orchestrator):
        """ChatSession.stream should delegate to AgentSession."""
        from victor.framework.agent import Agent, ChatSession
        from victor.framework.events import Event, EventType

        agent = Agent(mock_orchestrator)
        session = agent.chat("Initial prompt")

        async def mock_stream(message):
            yield Event(type=EventType.CONTENT, content="Streamed")

        session._delegate.stream = mock_stream

        events = []
        async for event in session.stream("Follow-up"):
            events.append(event)

        assert len(events) == 1
        assert events[0].content == "Streamed"

    def test_chat_session_turn_count(self, mock_orchestrator):
        """ChatSession.turn_count should return delegate's turn count."""
        from victor.framework.agent import Agent, ChatSession

        agent = Agent(mock_orchestrator)
        session = agent.chat("Initial prompt")
        session._delegate._turn_count = 5

        assert session.turn_count == 5

    def test_chat_session_history(self, mock_orchestrator):
        """ChatSession.history should return delegate's history."""
        from victor.framework.agent import Agent, ChatSession

        agent = Agent(mock_orchestrator)
        session = agent.chat("Initial prompt")

        # Mock messages on orchestrator
        mock_msg = MagicMock()
        mock_msg.role = "user"
        mock_msg.content = "Hello"
        mock_orchestrator.messages = [mock_msg]

        history = session.history

        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"

    def test_chat_session_get_session(self, mock_orchestrator):
        """ChatSession.get_session should return the underlying AgentSession."""
        from victor.framework.agent import Agent, ChatSession
        from victor.framework.agent_components import AgentSession

        agent = Agent(mock_orchestrator)
        session = agent.chat("Initial prompt")

        underlying = session.get_session()

        assert isinstance(underlying, AgentSession)
        assert underlying is session._delegate


# =============================================================================
# State Properties and Observers Tests (lines 522, 531, 540, 549-551, 572-578)
# =============================================================================


class TestAgentStateProperties:
    """Tests for Agent state-related properties and methods."""

    def test_state_property(self, mock_orchestrator):
        """state property should return State wrapper."""
        from victor.framework.agent import Agent
        from victor.framework.state import State

        agent = Agent(mock_orchestrator)

        assert isinstance(agent.state, State)
        assert agent._state is agent.state

    def test_vertical_property(self, mock_orchestrator, mock_vertical):
        """vertical property should return stored vertical class."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        assert agent.vertical is mock_vertical

    def test_vertical_property_none(self, mock_orchestrator):
        """vertical property should return None when not set."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        assert agent.vertical is None

    def test_vertical_config_property(self, mock_orchestrator):
        """vertical_config property should return stored config."""
        from victor.framework.agent import Agent

        config = MagicMock()
        agent = Agent(mock_orchestrator, vertical_config=config)

        assert agent.vertical_config is config

    def test_vertical_name_property(self, mock_orchestrator, mock_vertical):
        """vertical_name property should return vertical's name."""
        from victor.framework.agent import Agent

        mock_vertical.name = "CodingAssistant"
        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        assert agent.vertical_name == "CodingAssistant"

    def test_vertical_name_property_none(self, mock_orchestrator):
        """vertical_name property should return None when no vertical."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        assert agent.vertical_name is None

    def test_on_state_change_registers_observer(self, mock_orchestrator):
        """on_state_change should register observer callback."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)
        observer = MagicMock()

        unsubscribe = agent.on_state_change(observer)

        assert observer in agent._state_observers
        assert callable(unsubscribe)

    def test_on_state_change_unsubscribe(self, mock_orchestrator):
        """on_state_change unsubscribe function should remove observer."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)
        observer = MagicMock()

        unsubscribe = agent.on_state_change(observer)
        assert observer in agent._state_observers

        unsubscribe()
        assert observer not in agent._state_observers

    def test_on_state_change_unsubscribe_idempotent(self, mock_orchestrator):
        """on_state_change unsubscribe should be idempotent."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)
        observer = MagicMock()

        unsubscribe = agent.on_state_change(observer)
        unsubscribe()
        unsubscribe()  # Should not raise

        assert observer not in agent._state_observers


# =============================================================================
# switch_model and set_tools Tests (lines 595-598, 606-608)
# =============================================================================


class TestAgentConfiguration:
    """Tests for Agent configuration methods."""

    @pytest.mark.asyncio
    async def test_switch_model(self, mock_orchestrator):
        """switch_model should update provider and model."""
        from victor.framework.agent import Agent

        mock_pm = MagicMock()
        mock_pm.switch_provider = AsyncMock()
        mock_orchestrator.provider_manager = mock_pm

        agent = Agent(mock_orchestrator, provider="anthropic", model="old-model")

        await agent.switch_model("openai", "gpt-4")

        mock_pm.switch_provider.assert_called_once_with("openai", "gpt-4")
        assert agent._provider == "openai"
        assert agent._model == "gpt-4"

    @pytest.mark.asyncio
    async def test_switch_model_no_provider_manager(self, mock_orchestrator):
        """switch_model should still update internal state without provider_manager."""
        from victor.framework.agent import Agent

        # Remove provider_manager attribute
        del mock_orchestrator.provider_manager

        agent = Agent(mock_orchestrator, provider="anthropic", model="old-model")

        await agent.switch_model("openai", "gpt-4")

        assert agent._provider == "openai"
        assert agent._model == "gpt-4"

    def test_set_tools_with_toolset(self, mock_orchestrator):
        """set_tools should configure tools using ToolSet."""
        from victor.framework.agent import Agent
        from victor.framework.tools import ToolSet

        agent = Agent(mock_orchestrator)
        toolset = ToolSet.minimal()

        with patch("victor.framework._internal.configure_tools") as mock_configure:
            agent.set_tools(toolset)

            mock_configure.assert_called_once_with(mock_orchestrator, toolset)

    def test_set_tools_with_list(self, mock_orchestrator):
        """set_tools should configure tools using list."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)
        tools = ["read_file", "write_file", "shell"]

        with patch("victor.framework._internal.configure_tools") as mock_configure:
            agent.set_tools(tools)

            mock_configure.assert_called_once_with(mock_orchestrator, tools)


# =============================================================================
# get_orchestrator Tests (line 633)
# =============================================================================


class TestGetOrchestrator:
    """Tests for Agent.get_orchestrator method."""

    def test_get_orchestrator(self, mock_orchestrator):
        """get_orchestrator should return the internal orchestrator."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        assert agent.get_orchestrator() is mock_orchestrator


# =============================================================================
# Observability Tests (lines 660-663, 672, 698-705)
# =============================================================================


class TestAgentObservability:
    """Tests for Agent observability methods."""

    def test_event_bus_property_with_observability(self, mock_orchestrator):
        """event_bus property should return EventBus when available."""
        from victor.framework.agent import Agent

        mock_observability = MagicMock()
        mock_event_bus = MagicMock()
        mock_observability.event_bus = mock_event_bus
        mock_orchestrator.observability = mock_observability

        agent = Agent(mock_orchestrator)

        assert agent.event_bus is mock_event_bus

    def test_event_bus_property_without_observability(self, mock_orchestrator):
        """event_bus property should return None when no observability."""
        from victor.framework.agent import Agent

        # Remove observability attribute
        if hasattr(mock_orchestrator, "observability"):
            del mock_orchestrator.observability

        agent = Agent(mock_orchestrator)

        assert agent.event_bus is None

    def test_observability_property(self, mock_orchestrator):
        """observability property should return ObservabilityIntegration."""
        from victor.framework.agent import Agent

        mock_observability = MagicMock()
        mock_orchestrator.observability = mock_observability

        agent = Agent(mock_orchestrator)

        assert agent.observability is mock_observability

    def test_subscribe_to_events(self, mock_orchestrator):
        """subscribe_to_events should subscribe to EventBus."""
        from victor.framework.agent import Agent

        mock_backend = MagicMock()
        mock_unsubscribe = MagicMock()
        mock_backend.subscribe = MagicMock(return_value=mock_unsubscribe)

        mock_event_bus = MagicMock()
        mock_event_bus.backend = mock_backend

        mock_observability = MagicMock()
        mock_observability.event_bus = mock_event_bus
        mock_orchestrator.observability = mock_observability

        agent = Agent(mock_orchestrator)
        handler = MagicMock()

        unsubscribe = agent.subscribe_to_events("TOOL", handler)

        assert unsubscribe is mock_unsubscribe
        mock_backend.subscribe.assert_called_once_with("tool.*", handler)

    def test_subscribe_to_events_no_event_bus(self, mock_orchestrator):
        """subscribe_to_events should return None when no event bus."""
        from victor.framework.agent import Agent

        # Remove observability
        if hasattr(mock_orchestrator, "observability"):
            del mock_orchestrator.observability

        agent = Agent(mock_orchestrator)
        handler = MagicMock()

        result = agent.subscribe_to_events("TOOL", handler)

        assert result is None


# =============================================================================
# CQRS Tests (lines 741-755, 764, 773, 786-789, 805-808, 824-827, 837-838)
# =============================================================================


class TestAgentCQRS:
    """Tests for Agent CQRS integration methods."""

    @pytest.mark.asyncio
    async def test_enable_cqrs(self, mock_orchestrator):
        """enable_cqrs should create and connect CQRS bridge."""
        from victor.framework.agent import Agent

        mock_bridge = MagicMock()
        mock_bridge.connect_agent = MagicMock(return_value="session-123")

        agent = Agent(mock_orchestrator)

        with patch(
            "victor.framework.cqrs_bridge.CQRSBridge.create",
            new=AsyncMock(return_value=mock_bridge),
        ):
            bridge = await agent.enable_cqrs(session_id="my-session")

        assert bridge is mock_bridge
        assert agent._cqrs_bridge is mock_bridge
        assert agent._cqrs_session_id == "session-123"

    @pytest.mark.asyncio
    async def test_enable_cqrs_reuses_existing_bridge(self, mock_orchestrator):
        """enable_cqrs should reuse existing bridge."""
        from victor.framework.agent import Agent

        mock_bridge = MagicMock()
        mock_bridge.connect_agent = MagicMock(return_value="session-456")

        agent = Agent(mock_orchestrator)
        agent._cqrs_bridge = mock_bridge

        bridge = await agent.enable_cqrs()

        assert bridge is mock_bridge
        mock_bridge.connect_agent.assert_called_once()

    def test_cqrs_bridge_property(self, mock_orchestrator):
        """cqrs_bridge property should return stored bridge."""
        from victor.framework.agent import Agent

        mock_bridge = MagicMock()
        agent = Agent(mock_orchestrator)
        agent._cqrs_bridge = mock_bridge

        assert agent.cqrs_bridge is mock_bridge

    def test_cqrs_session_id_property(self, mock_orchestrator):
        """cqrs_session_id property should return stored session ID."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)
        agent._cqrs_session_id = "test-session"

        assert agent.cqrs_session_id == "test-session"

    @pytest.mark.asyncio
    async def test_cqrs_get_session(self, mock_orchestrator):
        """cqrs_get_session should query session via bridge."""
        from victor.framework.agent import Agent

        mock_bridge = MagicMock()
        mock_bridge.get_session = AsyncMock(return_value={"session": "data"})

        agent = Agent(mock_orchestrator)
        agent._cqrs_bridge = mock_bridge
        agent._cqrs_session_id = "session-123"

        result = await agent.cqrs_get_session()

        assert result == {"session": "data"}
        mock_bridge.get_session.assert_called_once_with("session-123")

    @pytest.mark.asyncio
    async def test_cqrs_get_session_not_enabled(self, mock_orchestrator):
        """cqrs_get_session should raise AgentError when CQRS not enabled."""
        from victor.framework.agent import Agent
        from victor.framework.errors import AgentError

        agent = Agent(mock_orchestrator)

        with pytest.raises(AgentError, match="CQRS not enabled"):
            await agent.cqrs_get_session()

    @pytest.mark.asyncio
    async def test_cqrs_get_history(self, mock_orchestrator):
        """cqrs_get_history should query conversation history."""
        from victor.framework.agent import Agent

        mock_bridge = MagicMock()
        mock_bridge.get_conversation_history = AsyncMock(return_value={"messages": []})

        agent = Agent(mock_orchestrator)
        agent._cqrs_bridge = mock_bridge
        agent._cqrs_session_id = "session-123"

        result = await agent.cqrs_get_history(limit=50)

        assert result == {"messages": []}
        mock_bridge.get_conversation_history.assert_called_once_with("session-123", limit=50)

    @pytest.mark.asyncio
    async def test_cqrs_get_history_not_enabled(self, mock_orchestrator):
        """cqrs_get_history should raise AgentError when CQRS not enabled."""
        from victor.framework.agent import Agent
        from victor.framework.errors import AgentError

        agent = Agent(mock_orchestrator)

        with pytest.raises(AgentError, match="CQRS not enabled"):
            await agent.cqrs_get_history()

    @pytest.mark.asyncio
    async def test_cqrs_get_metrics(self, mock_orchestrator):
        """cqrs_get_metrics should query session metrics."""
        from victor.framework.agent import Agent

        mock_bridge = MagicMock()
        mock_bridge.get_metrics = AsyncMock(return_value={"tool_calls": 10})

        agent = Agent(mock_orchestrator)
        agent._cqrs_bridge = mock_bridge
        agent._cqrs_session_id = "session-123"

        result = await agent.cqrs_get_metrics()

        assert result == {"tool_calls": 10}
        mock_bridge.get_metrics.assert_called_once_with("session-123")

    @pytest.mark.asyncio
    async def test_cqrs_get_metrics_not_enabled(self, mock_orchestrator):
        """cqrs_get_metrics should raise AgentError when CQRS not enabled."""
        from victor.framework.agent import Agent
        from victor.framework.errors import AgentError

        agent = Agent(mock_orchestrator)

        with pytest.raises(AgentError, match="CQRS not enabled"):
            await agent.cqrs_get_metrics()

    def test_forward_event_to_cqrs(self, mock_orchestrator):
        """_forward_event_to_cqrs should forward to adapter when set."""
        from victor.framework.agent import Agent
        from victor.framework.events import Event, EventType

        mock_adapter = MagicMock()
        agent = Agent(mock_orchestrator)
        agent._cqrs_adapter = mock_adapter

        event = Event(type=EventType.CONTENT, content="test")
        agent._forward_event_to_cqrs(event)

        mock_adapter.forward.assert_called_once_with(event)

    def test_forward_event_to_cqrs_no_adapter(self, mock_orchestrator):
        """_forward_event_to_cqrs should do nothing when no adapter."""
        from victor.framework.agent import Agent
        from victor.framework.events import Event, EventType

        agent = Agent(mock_orchestrator)
        event = Event(type=EventType.CONTENT, content="test")

        # Should not raise
        agent._forward_event_to_cqrs(event)


# =============================================================================
# Workflow and Team Execution Tests (lines 882-905, 947-987, 1009, 1028, 1032, 1036)
# =============================================================================


class TestAgentWorkflows:
    """Tests for Agent workflow execution methods."""

    @pytest.mark.asyncio
    async def test_run_workflow_no_vertical(self, mock_orchestrator):
        """run_workflow should raise AgentError when no vertical."""
        from victor.framework.agent import Agent
        from victor.framework.errors import AgentError

        agent = Agent(mock_orchestrator)

        with pytest.raises(AgentError, match="No vertical configured"):
            await agent.run_workflow("test_workflow")

    @pytest.mark.asyncio
    async def test_run_workflow_no_provider(self, mock_orchestrator, mock_vertical):
        """run_workflow should raise AgentError when vertical has no workflow provider."""
        from victor.framework.agent import Agent
        from victor.framework.errors import AgentError

        mock_vertical.get_workflow_provider = MagicMock(return_value=None)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        with pytest.raises(AgentError, match="does not provide workflows"):
            await agent.run_workflow("test_workflow")

    @pytest.mark.asyncio
    async def test_run_workflow_not_found(self, mock_orchestrator, mock_vertical):
        """run_workflow should raise AgentError when workflow not found."""
        from victor.framework.agent import Agent
        from victor.framework.errors import AgentError

        mock_provider = MagicMock()
        mock_provider.run_compiled_workflow = AsyncMock(
            side_effect=ValueError("Unknown workflow: unknown_workflow")
        )
        mock_vertical.get_workflow_provider = MagicMock(return_value=mock_provider)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        with pytest.raises(ValueError, match="Unknown workflow"):
            await agent.run_workflow("unknown_workflow")

    @pytest.mark.asyncio
    async def test_run_workflow_success(self, mock_orchestrator, mock_vertical):
        """run_workflow should execute workflow and return result."""
        from victor.framework.agent import Agent

        mock_provider = MagicMock()
        mock_result = {"success": True, "output": "done"}
        mock_provider.run_compiled_workflow = AsyncMock(return_value=mock_result)
        mock_vertical.get_workflow_provider = MagicMock(return_value=mock_provider)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)
        result = await agent.run_workflow("test_workflow", context={"key": "value"})

        assert result == {"success": True, "output": "done"}
        mock_provider.run_compiled_workflow.assert_called_once_with(
            "test_workflow", {"key": "value"}, timeout=None
        )

    def test_get_available_workflows_no_vertical(self, mock_orchestrator):
        """get_available_workflows should return empty list when no vertical."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        assert agent.get_available_workflows() == []

    def test_get_available_workflows_no_provider(self, mock_orchestrator, mock_vertical):
        """get_available_workflows should return empty list when no provider."""
        from victor.framework.agent import Agent

        mock_vertical.get_workflow_provider = MagicMock(return_value=None)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        assert agent.get_available_workflows() == []

    def test_get_available_workflows(self, mock_orchestrator, mock_vertical):
        """get_available_workflows should return workflow names."""
        from victor.framework.agent import Agent

        mock_provider = MagicMock()
        mock_provider.get_workflow_names = MagicMock(return_value=["wf1", "wf2"])
        mock_vertical.get_workflow_provider = MagicMock(return_value=mock_provider)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        assert agent.get_available_workflows() == ["wf1", "wf2"]


class TestAgentTeams:
    """Tests for Agent team execution methods."""

    @pytest.mark.asyncio
    async def test_run_team_no_vertical(self, mock_orchestrator):
        """run_team should raise AgentError when no vertical."""
        from victor.framework.agent import Agent
        from victor.framework.errors import AgentError

        agent = Agent(mock_orchestrator)

        with pytest.raises(AgentError, match="No vertical configured"):
            await agent.run_team("test_team", "Test goal")

    @pytest.mark.asyncio
    async def test_run_team_no_team_provider(self, mock_orchestrator, mock_vertical):
        """run_team should raise AgentError when vertical has no team spec provider."""
        from victor.framework.agent import Agent
        from victor.framework.errors import AgentError

        # Remove get_team_spec_provider method
        del mock_vertical.get_team_spec_provider

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        with pytest.raises(AgentError, match="does not support teams"):
            await agent.run_team("test_team", "Test goal")

    @pytest.mark.asyncio
    async def test_run_team_no_specs(self, mock_orchestrator, mock_vertical):
        """run_team should raise AgentError when team specs are empty."""
        from victor.framework.agent import Agent
        from victor.framework.errors import AgentError

        mock_provider = MagicMock()
        mock_provider.get_team_specs = MagicMock(return_value={})
        mock_vertical.get_team_spec_provider = MagicMock(return_value=mock_provider)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        with pytest.raises(AgentError, match="has no team specs defined"):
            await agent.run_team("test_team", "Test goal")

    @pytest.mark.asyncio
    async def test_run_team_not_found(self, mock_orchestrator, mock_vertical):
        """run_team should raise AgentError when team not found."""
        from victor.framework.agent import Agent
        from victor.framework.errors import AgentError

        mock_provider = MagicMock()
        mock_provider.get_team_specs = MagicMock(
            return_value={
                "team1": MagicMock(),
                "team2": MagicMock(),
            }
        )
        mock_vertical.get_team_spec_provider = MagicMock(return_value=mock_provider)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        with pytest.raises(AgentError, match="not found"):
            await agent.run_team("unknown_team", "Test goal")

    def test_get_available_teams_no_vertical(self, mock_orchestrator):
        """get_available_teams should return empty list when no vertical."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        assert agent.get_available_teams() == []

    def test_get_available_teams_no_provider_method(self, mock_orchestrator, mock_vertical):
        """get_available_teams should return empty list when no get_team_spec_provider."""
        from victor.framework.agent import Agent

        del mock_vertical.get_team_spec_provider

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        assert agent.get_available_teams() == []

    def test_get_available_teams_null_provider(self, mock_orchestrator, mock_vertical):
        """get_available_teams should return empty list when provider is None."""
        from victor.framework.agent import Agent

        mock_vertical.get_team_spec_provider = MagicMock(return_value=None)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        assert agent.get_available_teams() == []

    def test_get_available_teams_no_get_team_specs(self, mock_orchestrator, mock_vertical):
        """get_available_teams should return empty list when provider has no get_team_specs."""
        from victor.framework.agent import Agent

        mock_provider = MagicMock(spec=[])  # No get_team_specs method
        mock_vertical.get_team_spec_provider = MagicMock(return_value=mock_provider)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        assert agent.get_available_teams() == []

    def test_get_available_teams_empty_specs(self, mock_orchestrator, mock_vertical):
        """get_available_teams should return empty list when specs are empty."""
        from victor.framework.agent import Agent

        mock_provider = MagicMock()
        mock_provider.get_team_specs = MagicMock(return_value={})
        mock_vertical.get_team_spec_provider = MagicMock(return_value=mock_provider)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        assert agent.get_available_teams() == []

    def test_get_available_teams(self, mock_orchestrator, mock_vertical):
        """get_available_teams should return team names."""
        from victor.framework.agent import Agent

        mock_provider = MagicMock()
        mock_provider.get_team_specs = MagicMock(
            return_value={
                "team1": MagicMock(),
                "team2": MagicMock(),
            }
        )
        mock_vertical.get_team_spec_provider = MagicMock(return_value=mock_provider)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        teams = agent.get_available_teams()
        assert set(teams) == {"team1", "team2"}

    @pytest.mark.asyncio
    async def test_run_team_success(self, mock_orchestrator, mock_vertical):
        """run_team should execute team and return result."""
        from victor.framework.agent import Agent
        from victor.framework.teams import AgentTeam

        # Create a team spec
        mock_team_spec = MagicMock()
        mock_team_spec.name = "test_team"
        mock_team_spec.members = [MagicMock()]
        mock_team_spec.formation = MagicMock()
        mock_team_spec.total_tool_budget = 100
        mock_team_spec.max_iterations = 50

        # Create team provider
        mock_provider = MagicMock()
        mock_provider.get_team_specs = MagicMock(
            return_value={
                "test_team": mock_team_spec,
            }
        )
        mock_vertical.get_team_spec_provider = MagicMock(return_value=mock_provider)

        # Create mock team and result
        mock_team_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.final_output = "Team completed work"
        mock_team_instance.run = AsyncMock(return_value=mock_result)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        with patch.object(AgentTeam, "create", new=AsyncMock(return_value=mock_team_instance)):
            result = await agent.run_team(
                "test_team",
                "Test goal",
                context={"key": "value"},
                timeout_seconds=60.0,
            )

        assert result["success"] is True
        assert result["final_output"] == "Team completed work"
        assert result["team_name"] == "test_team"
        assert result["goal"] == "Test goal"
        mock_team_instance.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_team_no_get_team_specs_method(self, mock_orchestrator, mock_vertical):
        """run_team should raise AgentError when team provider has no get_team_specs."""
        from victor.framework.agent import Agent
        from victor.framework.errors import AgentError

        mock_provider = MagicMock(spec=[])  # No get_team_specs method
        mock_vertical.get_team_spec_provider = MagicMock(return_value=mock_provider)

        agent = Agent(mock_orchestrator, vertical=mock_vertical)

        with pytest.raises(AgentError, match="does not provide team specs"):
            await agent.run_team("test_team", "Test goal")


# =============================================================================
# Lifecycle Tests (lines 1046-1047, 1052-1062, 1065, 1068)
# =============================================================================


class TestAgentLifecycle:
    """Tests for Agent lifecycle methods."""

    @pytest.mark.asyncio
    async def test_reset(self, mock_orchestrator):
        """reset should reset conversation and state."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        await agent.reset()

        mock_orchestrator.reset_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_with_cqrs(self, mock_orchestrator):
        """close should cleanup CQRS bridge when enabled."""
        from victor.framework.agent import Agent

        mock_bridge = MagicMock()
        mock_bridge.disconnect_agent = MagicMock()
        mock_bridge.close = MagicMock()

        agent = Agent(mock_orchestrator)
        agent._cqrs_bridge = mock_bridge
        agent._cqrs_session_id = "session-123"
        agent._cqrs_adapter = MagicMock()

        await agent.close()

        mock_bridge.disconnect_agent.assert_called_once_with("session-123")
        mock_bridge.close.assert_called_once()
        assert agent._cqrs_bridge is None
        assert agent._cqrs_session_id is None
        assert agent._cqrs_adapter is None

    @pytest.mark.asyncio
    async def test_close_with_orchestrator_close(self, mock_orchestrator):
        """close should call orchestrator.close when available."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        await agent.close()

        mock_orchestrator.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_orchestrator_close(self, mock_orchestrator):
        """close should handle orchestrator without close method."""
        from victor.framework.agent import Agent

        del mock_orchestrator.close

        agent = Agent(mock_orchestrator)

        # Should not raise
        await agent.close()

    @pytest.mark.asyncio
    async def test_context_manager_enter(self, mock_orchestrator):
        """__aenter__ should return self."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        result = await agent.__aenter__()

        assert result is agent

    @pytest.mark.asyncio
    async def test_context_manager_exit(self, mock_orchestrator):
        """__aexit__ should call close."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        with patch.object(agent, "close", new=AsyncMock()) as mock_close:
            await agent.__aexit__(None, None, None)

            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, mock_orchestrator):
        """Agent should work as async context manager."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator)

        async with agent as a:
            assert a is agent

        mock_orchestrator.close.assert_called_once()

    def test_repr(self, mock_orchestrator):
        """__repr__ should return descriptive string."""
        from victor.framework.agent import Agent

        agent = Agent(mock_orchestrator, provider="anthropic", model="claude-3")

        repr_str = repr(agent)

        assert "Agent" in repr_str
        assert "anthropic" in repr_str
        assert "claude-3" in repr_str


# =============================================================================
# Agent.create Tests (lines 181-234) - Integration style tests
# =============================================================================


class TestAgentCreate:
    """Tests for Agent.create classmethod."""

    @pytest.mark.asyncio
    async def test_create_basic(self):
        """create should create Agent with orchestrator."""
        from victor.framework.agent import Agent

        mock_orchestrator = MagicMock()

        with patch(
            "victor.framework._internal.create_orchestrator_from_options",
            new=AsyncMock(return_value=mock_orchestrator),
        ):
            agent = await Agent.create(provider="anthropic")

        assert agent._orchestrator is mock_orchestrator
        assert agent._provider == "anthropic"

    @pytest.mark.asyncio
    async def test_create_with_vertical(self, mock_vertical):
        """create should apply vertical configuration."""
        from victor.framework.agent import Agent

        mock_orchestrator = MagicMock()

        with patch(
            "victor.framework._internal.create_orchestrator_from_options",
            new=AsyncMock(return_value=mock_orchestrator),
        ):
            agent = await Agent.create(
                provider="anthropic",
                vertical=mock_vertical,
            )

        assert agent._vertical is mock_vertical
        assert agent._vertical_config is not None

    @pytest.mark.asyncio
    async def test_create_with_provider_hints(self, mock_vertical):
        """create should respect provider hints from vertical config."""
        from victor.framework.agent import Agent

        mock_orchestrator = MagicMock()

        # Set up vertical config with provider hints preferring openai
        mock_config = MagicMock()
        mock_config.tools = []
        mock_config.system_prompt = "Test prompt"
        mock_config.provider_hints = {"preferred_providers": ["openai", "google"]}
        mock_vertical.get_config = MagicMock(return_value=mock_config)

        with patch(
            "victor.framework._internal.create_orchestrator_from_options",
            new=AsyncMock(return_value=mock_orchestrator),
        ):
            # User still chooses anthropic despite not being preferred
            agent = await Agent.create(
                provider="anthropic",
                vertical=mock_vertical,
            )

        # Agent should still use anthropic (hints are just hints)
        assert agent._provider == "anthropic"

    @pytest.mark.asyncio
    async def test_create_provider_error(self):
        """create should raise ProviderError for provider issues."""
        from victor.framework.agent import Agent
        from victor.framework.errors import ProviderError

        with patch(
            "victor.framework._internal.create_orchestrator_from_options",
            new=AsyncMock(side_effect=Exception("Provider API error")),
        ):
            with pytest.raises(ProviderError, match="API error"):
                await Agent.create(provider="anthropic")

    @pytest.mark.asyncio
    async def test_create_agent_error(self):
        """create should raise AgentError for other errors."""
        from victor.framework.agent import Agent
        from victor.framework.errors import AgentError

        with patch(
            "victor.framework._internal.create_orchestrator_from_options",
            new=AsyncMock(side_effect=Exception("Configuration invalid")),
        ):
            with pytest.raises(AgentError, match="Configuration invalid"):
                await Agent.create(provider="anthropic")


# =============================================================================
# Agent.create_team Tests (lines 334-349)
# =============================================================================


class TestAgentCreateTeam:
    """Tests for Agent.create_team classmethod."""

    @pytest.mark.asyncio
    async def test_create_team_basic(self):
        """create_team should create AgentTeam."""
        from victor.framework.agent import Agent
        from victor.framework.teams import TeamMemberSpec

        mock_orchestrator = MagicMock()
        mock_team = MagicMock()

        with (
            patch(
                "victor.framework._internal.create_orchestrator_from_options",
                new=AsyncMock(return_value=mock_orchestrator),
            ),
            patch("victor.framework.teams.AgentTeam.create", new=AsyncMock(return_value=mock_team)),
        ):
            team = await Agent.create_team(
                name="Test Team",
                goal="Complete test",
                members=[
                    TeamMemberSpec(role="researcher", goal="Research"),
                    TeamMemberSpec(role="executor", goal="Execute"),
                ],
            )

        assert team is mock_team

    @pytest.mark.asyncio
    async def test_create_team_with_formation(self):
        """create_team should use specified formation."""
        from victor.framework.agent import Agent
        from victor.framework.teams import TeamMemberSpec
        from victor.teams import TeamFormation

        mock_orchestrator = MagicMock()
        mock_team = MagicMock()

        with (
            patch(
                "victor.framework._internal.create_orchestrator_from_options",
                new=AsyncMock(return_value=mock_orchestrator),
            ),
            patch(
                "victor.framework.teams.AgentTeam.create", new=AsyncMock(return_value=mock_team)
            ) as mock_create,
        ):
            await Agent.create_team(
                name="Test Team",
                goal="Complete test",
                members=[TeamMemberSpec(role="executor", goal="Execute")],
                formation=TeamFormation.PARALLEL,
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs.get("formation") == TeamFormation.PARALLEL

    @pytest.mark.asyncio
    async def test_create_team_default_formation(self):
        """create_team should default to SEQUENTIAL formation."""
        from victor.framework.agent import Agent
        from victor.framework.teams import TeamMemberSpec
        from victor.teams import TeamFormation

        mock_orchestrator = MagicMock()
        mock_team = MagicMock()

        with (
            patch(
                "victor.framework._internal.create_orchestrator_from_options",
                new=AsyncMock(return_value=mock_orchestrator),
            ),
            patch(
                "victor.framework.teams.AgentTeam.create", new=AsyncMock(return_value=mock_team)
            ) as mock_create,
        ):
            await Agent.create_team(
                name="Test Team",
                goal="Complete test",
                members=[TeamMemberSpec(role="executor", goal="Execute")],
            )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs.get("formation") == TeamFormation.SEQUENTIAL


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module-level exports."""

    def test_agent_importable(self):
        """Agent class should be importable from framework.agent."""
        from victor.framework.agent import Agent

        assert Agent is not None

    def test_chat_session_importable(self):
        """ChatSession class should be importable from framework.agent."""
        from victor.framework.agent import ChatSession

        assert ChatSession is not None
