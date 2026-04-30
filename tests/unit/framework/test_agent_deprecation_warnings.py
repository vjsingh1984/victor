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

"""Tests for deprecation warnings during service layer alignment."""

import pytest
import warnings
from unittest.mock import AsyncMock, MagicMock

from victor.core.feature_flags import FeatureFlag, reset_feature_flag_manager
from victor.framework.agent import Agent
from victor.providers.base import CompletionResponse


# Agent.__init__ validates type(orchestrator).__name__ == "AgentOrchestrator".
class AgentOrchestrator(MagicMock):
    """MagicMock subclass whose type name satisfies Agent.__init__ validation."""


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

    def mock_chat_response(content="Hello World", tool_calls=None):
        return CompletionResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls,
            stop_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            model="test-model",
        )

    orchestrator.chat = AsyncMock(return_value=mock_chat_response())
    orchestrator._container = MagicMock()

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


class TestAgentDeprecationWarnings:
    """Tests for deprecation warnings when using legacy paths."""

    @pytest.mark.asyncio
    async def test_agent_run_legacy_path_shows_deprecation_warning(
        self, mock_orchestrator, reset_feature_flags
    ):
        """Verify Agent.run() shows deprecation warning when using legacy path."""
        from victor.core.feature_flags import get_feature_flag_manager

        # Ensure feature flag is disabled (default)
        manager = get_feature_flag_manager()
        manager.disable(FeatureFlag.USE_SERVICE_LAYER_FOR_AGENT)

        # Create agent
        agent = Agent(orchestrator=mock_orchestrator)

        # Call run() and expect deprecation warning
        with pytest.warns(
            DeprecationWarning, match="Agent.run\\(\\) is using direct orchestrator access"
        ):
            result = await agent.run("Hello")

        # Verify result is still returned correctly
        assert result.content == "Hello World"

    @pytest.mark.asyncio
    async def test_agent_stream_legacy_path_shows_deprecation_warning(
        self, mock_orchestrator, reset_feature_flags
    ):
        """Verify Agent.stream() shows deprecation warning when using legacy path."""
        from victor.core.feature_flags import get_feature_flag_manager
        from victor.framework._internal import stream_with_events

        # Ensure feature flag is disabled (default)
        manager = get_feature_flag_manager()
        manager.disable(FeatureFlag.USE_SERVICE_LAYER_FOR_AGENT)

        # Create agent
        agent = Agent(orchestrator=mock_orchestrator)

        # Mock stream_with_events to return events
        async def mock_stream_with_events(orch, prompt):
            from victor.framework.agent import AgentExecutionEvent
            from victor.framework.events import EventType

            yield AgentExecutionEvent(
                type=EventType.CONTENT,
                content="Hello from orchestrator",
            )

        # Call stream() and check that warnings are emitted
        # Note: pytest.warns doesn't work well with async generators,
        # so we use warnings.catch_warnings instead
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)

            from unittest.mock import patch

            with patch("victor.framework._internal.stream_with_events", mock_stream_with_events):
                events = []
                async for event in agent.stream("Hello"):
                    events.append(event)

            # Check that deprecation warning was issued
            deprecation_warnings = [
                warning for warning in w if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) > 0
            assert any("Agent.stream()" in str(warning.message) for warning in deprecation_warnings)

        # Verify events are still returned correctly
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_agent_run_service_layer_no_warning(self, mock_orchestrator, reset_feature_flags):
        """Verify Agent.run() does NOT show deprecation warning when using service layer."""
        from victor.core.feature_flags import get_feature_flag_manager
        from victor.runtime.context import ServiceAccessor

        # Enable the feature flag
        manager = get_feature_flag_manager()
        manager.enable(FeatureFlag.USE_SERVICE_LAYER_FOR_AGENT)

        # Create agent
        agent = Agent(orchestrator=mock_orchestrator)

        # Mock ChatService
        mock_chat_service = MagicMock()

        def mock_chat_response(content="Hello from ChatService", tool_calls=None):
            return CompletionResponse(
                content=content,
                role="assistant",
                tool_calls=tool_calls,
                stop_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                model="test-model",
            )

        mock_chat_service.chat = AsyncMock(return_value=mock_chat_response())

        # Call run() with service layer - should NOT show deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DeprecationWarning)

            from unittest.mock import patch

            with patch.object(ServiceAccessor, "chat", mock_chat_service):
                result = await agent.run("Hello")

            # Check that NO deprecation warning about legacy path was issued
            legacy_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and "direct orchestrator access" in str(warning.message)
            ]
            assert len(legacy_warnings) == 0, f"Unexpected legacy path warning: {legacy_warnings}"

        # Verify result is returned correctly
        assert result.content == "Hello from ChatService"


# Note: Orchestrator deprecation warnings are tested indirectly through Agent tests.
# Direct testing requires integration tests with real orchestrator instances,
# which is covered by the integration test suite.
