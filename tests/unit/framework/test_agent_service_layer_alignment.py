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

"""Tests for Phase 3: Service Layer Alignment in Agent.

These tests verify that Agent.run() and Agent.stream() correctly use
ChatService instead of accessing orchestrator directly when the
USE_SERVICE_LAYER_FOR_AGENT feature flag is enabled.

This aligns with the service+state-pass architecture and ensures
Phase 2 coordinator batching works consistently.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import AsyncIterator

from victor.core.feature_flags import FeatureFlag, reset_feature_flag_manager
from victor.framework.agent import Agent
from victor.framework.events import EventType
from victor.providers.base import CompletionResponse, StreamChunk


# Agent.__init__ validates type(orchestrator).__name__ == "AgentOrchestrator".
class AgentOrchestrator(MagicMock):
    """MagicMock subclass whose type name satisfies Agent.__init__ validation."""


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_feature_flags():
    """Reset feature flag manager before and after each test."""
    reset_feature_flag_manager()
    yield
    reset_feature_flag_manager()


@pytest.fixture
def mock_orchestrator():
    """Create a mock AgentOrchestrator for testing."""
    orchestrator = AgentOrchestrator()
    orchestrator.provider = MagicMock()
    orchestrator.provider.name = "test_provider"
    orchestrator.model = "test-model"
    orchestrator.messages = []

    # Mock chat method for direct sync execution
    def mock_chat_response(content="Hello World", tool_calls=None):
        """Create a mock CompletionResponse."""
        return CompletionResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls,
            stop_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            model="test-model",
        )

    orchestrator.chat = AsyncMock(return_value=mock_chat_response())

    # Mock stream_chat for streaming
    async def mock_stream_chat(message, **kwargs):
        """Mock streaming response."""
        chunks = [
            StreamChunk(content="Hello ", finish_reason=None, is_complete=False),
            StreamChunk(content="World", finish_reason=None, is_complete=False),
            StreamChunk(content="", finish_reason="stop", is_complete=True),
        ]
        for chunk in chunks:
            yield chunk

    orchestrator.stream_chat = mock_stream_chat

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

    # Add _container for service access
    orchestrator._container = MagicMock()

    return orchestrator


@pytest.fixture
def mock_chat_service():
    """Create a mock ChatService."""
    chat_service = MagicMock()

    # Mock chat() method
    def mock_chat_response(content="Hello from ChatService", tool_calls=None):
        return CompletionResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls,
            stop_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            model="test-model",
        )

    chat_service.chat = AsyncMock(return_value=mock_chat_response())

    # Mock stream_chat() method as async generator
    async def mock_stream_chat(message, **kwargs):
        """Mock streaming response from ChatService."""
        chunks = [
            StreamChunk(content="Hello ", finish_reason=None, is_complete=False),
            StreamChunk(content="from ", finish_reason=None, is_complete=False),
            StreamChunk(content="ChatService", finish_reason=None, is_complete=False),
            StreamChunk(content="", finish_reason="stop", is_complete=True),
        ]
        for chunk in chunks:
            yield chunk

    chat_service.stream_chat = mock_stream_chat

    return chat_service


# =============================================================================
# Tests: Agent.run() with Service Layer
# =============================================================================


class TestAgentRunWithServiceLayer:
    """Tests for Agent.run() using ChatService instead of orchestrator."""

    @pytest.mark.asyncio
    async def test_run_uses_chat_service_when_flag_enabled(
        self, mock_orchestrator, mock_chat_service, reset_feature_flags
    ):
        """Verify Agent.run() uses ChatService when USE_SERVICE_LAYER_FOR_AGENT is enabled."""
        from victor.core.feature_flags import get_feature_flag_manager
        from victor.runtime.context import ServiceAccessor

        # Enable the feature flag
        manager = get_feature_flag_manager()
        manager.enable(FeatureFlag.USE_SERVICE_LAYER_FOR_AGENT)

        # Add _container to orchestrator
        mock_orchestrator._container = MagicMock()

        # Create agent
        agent = Agent(orchestrator=mock_orchestrator)

        # Mock ServiceAccessor to return our mock ChatService
        with patch.object(ServiceAccessor, "chat", mock_chat_service):
            result = await agent.run("Hello")

        # Verify ChatService.chat() was called
        mock_chat_service.chat.assert_called_once_with("Hello")

        # Verify orchestrator.chat() was NOT called (bypassed)
        mock_orchestrator.chat.assert_not_called()

        # Verify result content
        assert result.content == "Hello from ChatService"

    @pytest.mark.asyncio
    async def test_run_fallback_to_orchestrator_when_service_unavailable(
        self, mock_orchestrator, reset_feature_flags
    ):
        """Verify Agent.run() falls back to orchestrator when ChatService is unavailable."""
        from victor.core.feature_flags import get_feature_flag_manager
        from victor.runtime.context import ServiceAccessor

        # Enable the feature flag
        manager = get_feature_flag_manager()
        manager.enable(FeatureFlag.USE_SERVICE_LAYER_FOR_AGENT)

        # Create agent
        agent = Agent(orchestrator=mock_orchestrator)

        # Mock ServiceAccessor to return None (service unavailable)
        with patch.object(ServiceAccessor, "chat", None):
            result = await agent.run("Hello")

        # Verify orchestrator.chat() was called (fallback)
        mock_orchestrator.chat.assert_called_once_with("Hello")

        # Verify result content
        assert result.content == "Hello World"

    @pytest.mark.asyncio
    async def test_run_uses_orchestrator_when_flag_disabled(
        self, mock_orchestrator, mock_chat_service, reset_feature_flags
    ):
        """Verify Agent.run() uses orchestrator when USE_SERVICE_LAYER_FOR_AGENT is disabled."""
        from victor.core.feature_flags import get_feature_flag_manager
        from victor.runtime.context import ServiceAccessor

        # Explicitly disable feature flag (now enabled by default)
        manager = get_feature_flag_manager()
        manager.disable(FeatureFlag.USE_SERVICE_LAYER_FOR_AGENT)

        # Create agent
        agent = Agent(orchestrator=mock_orchestrator)

        # Mock ServiceAccessor (should not be used)
        with patch.object(ServiceAccessor, "chat", mock_chat_service):
            result = await agent.run("Hello")

        # Verify orchestrator.chat() was called (legacy path)
        mock_orchestrator.chat.assert_called_once_with("Hello")

        # Verify ChatService was NOT called
        mock_chat_service.chat.assert_not_called()

        # Verify result content
        assert result.content == "Hello World"


# =============================================================================
# Tests: Agent.stream() with Service Layer
# =============================================================================


class TestAgentStreamWithServiceLayer:
    """Tests for Agent.stream() using ChatService instead of orchestrator."""

    @pytest.mark.asyncio
    async def test_stream_uses_chat_service_when_flag_enabled(
        self, mock_orchestrator, mock_chat_service, reset_feature_flags
    ):
        """Verify Agent.stream() uses ChatService when USE_SERVICE_LAYER_FOR_AGENT is enabled."""
        from victor.core.feature_flags import get_feature_flag_manager
        from victor.runtime.context import ServiceAccessor

        # Enable the feature flag
        manager = get_feature_flag_manager()
        manager.enable(FeatureFlag.USE_SERVICE_LAYER_FOR_AGENT)

        # Create agent
        agent = Agent(orchestrator=mock_orchestrator)

        # Mock ServiceAccessor to return our mock ChatService
        with patch.object(ServiceAccessor, "chat", mock_chat_service):
            events = []
            async for event in agent.stream("Hello"):
                events.append(event)

        # Verify we got events (this proves ChatService.stream_chat was called)
        assert len(events) > 0
        assert all(event.type == EventType.CONTENT for event in events)
        contents = [e.content for e in events if e.content]
        assert "Hello" in "".join(contents)
        assert "from" in "".join(contents)
        assert "ChatService" in "".join(contents)

    @pytest.mark.asyncio
    async def test_stream_fallback_to_orchestrator_when_service_unavailable(
        self, mock_orchestrator, reset_feature_flags
    ):
        """Verify Agent.stream() falls back to orchestrator when ChatService is unavailable."""
        from victor.core.feature_flags import get_feature_flag_manager
        from victor.runtime.context import ServiceAccessor
        from victor.framework._internal import stream_with_events

        # Enable the feature flag
        manager = get_feature_flag_manager()
        manager.enable(FeatureFlag.USE_SERVICE_LAYER_FOR_AGENT)

        # Create agent
        agent = Agent(orchestrator=mock_orchestrator)

        # Mock ServiceAccessor to return None (service unavailable)
        # Mock stream_with_events to return events
        async def mock_stream_with_events(orch, prompt):
            from victor.framework.agent import AgentExecutionEvent

            yield AgentExecutionEvent(
                type=EventType.CONTENT,
                content="Hello from orchestrator",
            )

        with patch.object(ServiceAccessor, "chat", None):
            with patch("victor.framework._internal.stream_with_events", mock_stream_with_events):
                events = []
                async for event in agent.stream("Hello"):
                    events.append(event)

        # Verify we got events from fallback
        assert len(events) > 0
        assert events[0].type == EventType.CONTENT
        assert events[0].content == "Hello from orchestrator"

    @pytest.mark.asyncio
    async def test_stream_uses_orchestrator_when_flag_disabled(
        self, mock_orchestrator, mock_chat_service, reset_feature_flags
    ):
        """Verify Agent.stream() uses orchestrator when USE_SERVICE_LAYER_FOR_AGENT is disabled."""
        from victor.core.feature_flags import get_feature_flag_manager
        from victor.runtime.context import ServiceAccessor
        from victor.framework._internal import stream_with_events

        # Explicitly disable feature flag (now enabled by default)
        manager = get_feature_flag_manager()
        manager.disable(FeatureFlag.USE_SERVICE_LAYER_FOR_AGENT)

        # Create agent
        agent = Agent(orchestrator=mock_orchestrator)

        # Mock stream_with_events to return events
        async def mock_stream_with_events(orch, prompt):
            from victor.framework.agent import AgentExecutionEvent

            yield AgentExecutionEvent(
                type=EventType.CONTENT,
                content="Hello from orchestrator",
            )

        # Mock ServiceAccessor (should not be used)
        with patch.object(ServiceAccessor, "chat", mock_chat_service):
            with patch("victor.framework._internal.stream_with_events", mock_stream_with_events):
                events = []
                async for event in agent.stream("Hello"):
                    events.append(event)

        # Verify we got events from orchestrator path
        assert len(events) > 0
        assert events[0].type == EventType.CONTENT
        assert events[0].content == "Hello from orchestrator"


# =============================================================================
# Tests: Feature Flag Control
# =============================================================================


class TestFeatureFlagControl:
    """Tests for feature flag control of service layer alignment."""

    @pytest.mark.asyncio
    async def test_feature_flag_default_is_enabled(self, reset_feature_flags):
        """Verify USE_SERVICE_LAYER_FOR_AGENT is enabled by default (opt-out)."""
        from victor.core.feature_flags import get_feature_flag_manager

        # Check default state
        manager = get_feature_flag_manager()
        assert manager.is_enabled(FeatureFlag.USE_SERVICE_LAYER_FOR_AGENT)

        # Verify it's NOT in the opt-in list
        assert not FeatureFlag.USE_SERVICE_LAYER_FOR_AGENT.is_opt_in_by_default()

    @pytest.mark.asyncio
    async def test_feature_flag_environment_variable(self, mock_orchestrator, reset_feature_flags):
        """Verify feature flag can be enabled via environment variable."""
        import os

        # Set environment variable
        os.environ["VICTOR_USE_SERVICE_LAYER_FOR_AGENT"] = "true"

        try:
            # Reset manager to pick up env var
            reset_feature_flag_manager()
            from victor.core.feature_flags import get_feature_flag_manager

            manager = get_feature_flag_manager()
            assert manager.is_enabled(FeatureFlag.USE_SERVICE_LAYER_FOR_AGENT)
        finally:
            # Clean up
            os.environ.pop("VICTOR_USE_SERVICE_LAYER_FOR_AGENT", None)
            reset_feature_flag_manager()


# =============================================================================
# Tests: Phase 2 Coordinator Integration
# =============================================================================


class TestPhase2CoordinatorIntegration:
    """Tests to verify Phase 2 coordinator still works with service layer alignment."""

    @pytest.mark.asyncio
    async def test_coordinator_batching_works_with_service_layer(
        self, mock_orchestrator, mock_chat_service, reset_feature_flags
    ):
        """Verify Phase 2 coordinator batching works when using ChatService."""
        from victor.core.feature_flags import get_feature_flag_manager
        from victor.runtime.context import ServiceAccessor

        # Enable both feature flags
        manager = get_feature_flag_manager()
        manager.enable(FeatureFlag.USE_SERVICE_LAYER_FOR_AGENT)
        manager.enable(FeatureFlag.USE_STAGE_TRANSITION_COORDINATOR)

        # Create agent
        agent = Agent(orchestrator=mock_orchestrator)

        # Mock ServiceAccessor to return our mock ChatService
        with patch.object(ServiceAccessor, "chat", mock_chat_service):
            result = await agent.run("Hello")

        # Verify ChatService.chat() was called
        mock_chat_service.chat.assert_called_once()

        # ChatService internally uses TurnExecutor, which has Phase 2 coordinator
        # integration (begin/end turn calls). This test verifies the path works.
        assert result.content == "Hello from ChatService"


# =============================================================================
# Tests: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in service layer alignment."""

    @pytest.mark.asyncio
    async def test_service_layer_error_falls_back_gracefully(
        self, mock_orchestrator, reset_feature_flags
    ):
        """Verify errors in ChatService are handled gracefully."""
        from victor.core.feature_flags import get_feature_flag_manager
        from victor.runtime.context import ServiceAccessor

        # Enable the feature flag
        manager = get_feature_flag_manager()
        manager.enable(FeatureFlag.USE_SERVICE_LAYER_FOR_AGENT)

        # Create agent
        agent = Agent(orchestrator=mock_orchestrator)

        # Mock ServiceAccessor with None to simulate service unavailable
        with patch.object(ServiceAccessor, "chat", None):
            # Should fall back to orchestrator without crashing
            result = await agent.run("Hello")

        # Verify fallback to orchestrator worked
        mock_orchestrator.chat.assert_called_once()
        assert result.content == "Hello World"
