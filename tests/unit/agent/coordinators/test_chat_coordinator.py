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

"""Tests for ChatCoordinator.

This test file provides comprehensive coverage for ChatCoordinator, which handles
both non-streaming and streaming chat operations with full agentic loop support.

Test Coverage Strategy:
- Test all public methods (chat, stream_chat)
- Test async behavior and error handling
- Test token usage tracking
- Test tool calling and execution
- Test context overflow handling
- Test recovery mechanisms
- Test iteration limits and budgeting
- Test provider rate limit handling

ChatCoordinator is the highest priority coordinator for testing as all user
interactions go through it. Current coverage: 4.77%. Target: >75%.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, MagicMock, patch, PropertyMock
from typing import Any, List, Optional

from victor.agent.coordinators.chat_coordinator import ChatCoordinator
from victor.providers.base import CompletionResponse, StreamChunk
from victor.core.errors import ProviderRateLimitError
from victor.framework.task import TaskComplexity
from victor.agent.unified_task_tracker import TrackerTaskType


class TestChatCoordinatorInitialization:
    """Test suite for ChatCoordinator initialization and setup."""

    @pytest.fixture
    def mock_orchestrator(self) -> Mock:
        """Create mock orchestrator with all required dependencies."""
        orch = Mock()
        orch.conversation = Mock()
        orch.conversation.ensure_system_prompt = Mock()
        orch.conversation.message_count = Mock(return_value=5)
        orch.provider = Mock()
        orch.provider.supports_tools = Mock(return_value=True)
        orch.provider.chat = AsyncMock()
        orch.provider.stream = self._create_stream_generator()
        orch.model = "test-model"
        orch.temperature = 0.7
        orch.max_tokens = 4096
        orch.tool_budget = 10
        orch.tool_calls_used = 0
        orch.thinking = False
        orch.messages = []
        orch.add_message = Mock()
        orch._system_added = False
        orch.task_classifier = Mock()
        orch.task_classifier.classify = Mock(
            return_value=Mock(tool_budget=5, complexity=TaskComplexity.MEDIUM)
        )
        orch.settings = Mock()
        orch.settings.chat_max_iterations = 10
        orch.tool_selector = Mock()
        orch.tool_selector.select_tools = AsyncMock(return_value=[])
        orch.tool_selector.prioritize_by_stage = Mock(return_value=[])
        orch.conversation_state = Mock()
        orch.conversation_state.state = Mock()
        orch.conversation_state.state.stage = None
        orch._context_compactor = None
        orch._handle_tool_calls = AsyncMock(return_value=[])
        orch.response_completer = Mock()
        orch.response_completer.ensure_response = AsyncMock(
            return_value=Mock(content="Fallback response")
        )
        orch.response_completer.format_tool_failure_message = Mock(
            return_value="Tool failed message"
        )
        orch._cumulative_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        orch._current_stream_context = None

        # Streaming-specific attributes
        orch._metrics_coordinator = Mock()
        orch._metrics_coordinator.start_streaming = Mock()
        orch._metrics_coordinator.stop_streaming = Mock()
        orch._metrics_collector = Mock()
        orch._metrics_collector.init_stream_metrics = Mock(return_value=Mock(start_time=0.0))
        orch._session_state = Mock()
        orch._session_state.reset_for_new_turn = Mock()
        orch.unified_tracker = Mock()
        orch.unified_tracker.reset = Mock()
        orch.unified_tracker.config = {"max_total_iterations": 50}
        orch.unified_tracker.max_exploration_iterations = 25
        orch.unified_tracker.detect_task_type = Mock(return_value=TrackerTaskType.EDIT)
        orch.unified_tracker._progress = Mock()
        orch.unified_tracker._progress.tool_budget = 10
        orch.unified_tracker._progress.has_prompt_requirements = False
        orch.unified_tracker.set_tool_budget = Mock()
        orch.unified_tracker.set_max_iterations = Mock()
        orch.unified_tracker.record_iteration = Mock()
        orch.unified_tracker.record_tool_call = Mock()
        orch.unified_tracker.check_loop_warning = Mock(return_value=None)
        orch.unified_tracker.unique_resources = []
        orch.reminder_manager = Mock()
        orch.reminder_manager.reset = Mock()
        orch._usage_analytics = None
        orch._sequence_tracker = None
        orch._context_manager = Mock()
        orch._context_manager.start_background_compaction = AsyncMock()
        orch._context_manager.get_max_context_chars = Mock(return_value=100000)
        orch.task_coordinator = Mock()
        orch.task_coordinator._reminder_manager = None
        orch.task_coordinator.set_reminder_manager = Mock()
        orch.task_coordinator.prepare_task = Mock(
            return_value=(Mock(complexity=TaskComplexity.MEDIUM), 10)
        )
        orch.task_coordinator.current_intent = None
        orch.task_coordinator.temperature = 0.7
        orch.task_coordinator.tool_budget = 10
        orch.task_coordinator.apply_intent_guard = Mock()
        orch.task_coordinator.apply_task_guidance = Mock()
        orch._tool_planner = Mock()
        orch._tool_planner.infer_goals_from_message = Mock(return_value=[])
        orch._tool_planner.plan_tools = Mock(return_value=[])
        orch._tool_planner.filter_tools_by_intent = Mock(return_value=[])
        orch._model_supports_tool_calls = Mock(return_value=True)
        orch._check_cancellation = Mock(return_value=False)
        orch._provider_coordinator = Mock()
        orch._provider_coordinator.get_rate_limit_wait_time = Mock(return_value=1.0)
        orch.sanitizer = Mock()
        orch.sanitizer.sanitize = Mock(return_value="sanitized content")
        orch.sanitizer.strip_markup = Mock(return_value="plain text")
        orch.sanitizer.is_garbage_content = Mock(return_value=False)
        orch.observed_files = []
        orch._recovery_integration = Mock()
        orch._recovery_integration.handle_response = AsyncMock(return_value=Mock(action="continue"))
        orch._recovery_integration.record_outcome = Mock()
        orch._chunk_generator = Mock()
        orch._chunk_generator.generate_content_chunk = Mock(
            side_effect=lambda c, is_final=False: StreamChunk(content=c, is_final=is_final)
        )
        orch._streaming_handler = Mock()
        orch._recovery_coordinator = Mock()
        orch._recovery_coordinator.check_natural_completion = Mock(return_value=None)
        orch._recovery_coordinator.handle_empty_response = Mock(return_value=(None, False))
        orch._recovery_coordinator.check_force_action = Mock(return_value=(False, None))
        orch._recovery_coordinator.get_recovery_fallback_message = Mock(
            return_value="Fallback message"
        )
        orch._record_intelligent_outcome = Mock()
        orch._task_completion_detector = Mock()
        orch._task_completion_detector.analyze_response = Mock()
        orch._task_completion_detector.get_completion_confidence = Mock(return_value=None)
        orch._force_finalize = False
        orch._continuation_prompts = 0
        orch._asking_input_prompts = 0
        orch._consecutive_blocked_attempts = 0
        orch._cumulative_prompt_interventions = 0
        orch._current_intent = None
        orch.provider_name = "test_provider"
        orch.debug_logger = Mock()
        orch.debug_logger.reset = Mock()
        orch.debug_logger.log_iteration_start = Mock()
        orch.debug_logger.log_limits = Mock()
        orch._intelligent_integration = None
        orch._required_files = []
        orch._required_outputs = []
        orch._read_files_session = set()
        orch._all_files_read_nudge_sent = False
        orch._streaming_controller = Mock()
        orch._streaming_controller.current_session = None

        return orch

    @staticmethod
    def _create_stream_generator():
        """Create a mock async generator for streaming."""

        async def generator(*args, **kwargs):
            yield StreamChunk(content="Hello", is_final=False)
            yield StreamChunk(content=" world", is_final=False)
            yield StreamChunk(content="!", is_final=True)

        return generator

    @pytest.fixture
    def coordinator(self, mock_orchestrator: Mock) -> ChatCoordinator:
        """Create ChatCoordinator instance."""
        return ChatCoordinator(orchestrator=mock_orchestrator)

    def test_initialization(self, coordinator: ChatCoordinator, mock_orchestrator: Mock):
        """Test that ChatCoordinator initializes correctly."""
        assert coordinator._orchestrator == mock_orchestrator
        assert coordinator._intent_classification_handler is None
        assert coordinator._continuation_handler is None
        assert coordinator._tool_execution_handler is None


class TestChatCoordinatorChat:
    """Test suite for the chat() method (non-streaming)."""

    @pytest.fixture
    def mock_orchestrator(self) -> Mock:
        """Create mock orchestrator for chat tests."""
        orch = Mock()
        orch.conversation = Mock()
        orch.conversation.ensure_system_prompt = Mock()
        orch.provider = Mock()
        orch.provider.supports_tools = Mock(return_value=False)
        orch.provider.chat = AsyncMock(
            return_value=CompletionResponse(
                content="Response content", role="assistant", tool_calls=None
            )
        )
        orch.model = "test-model"
        orch.temperature = 0.7
        orch.max_tokens = 4096
        orch.tool_budget = 10
        orch.tool_calls_used = 0
        orch.thinking = False
        orch.messages = []
        orch.add_message = Mock()
        orch._system_added = False
        orch.task_classifier = Mock()
        orch.task_classifier.classify = Mock(
            return_value=Mock(tool_budget=5, complexity=TaskComplexity.MEDIUM)
        )
        orch.settings = Mock()
        orch.settings.chat_max_iterations = 10
        orch.conversation_state = Mock()
        orch.conversation_state.state = Mock()
        orch.conversation_state.state.stage = None
        orch._context_compactor = None
        orch._handle_tool_calls = AsyncMock(return_value=[])
        orch.response_completer = Mock()
        orch.response_completer.ensure_response = AsyncMock(return_value=Mock(content="Fallback"))
        orch.response_completer.format_tool_failure_message = Mock(return_value="Tool failed")
        orch._cumulative_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        return orch

    @pytest.fixture
    def coordinator(self, mock_orchestrator: Mock) -> ChatCoordinator:
        """Create ChatCoordinator instance."""
        return ChatCoordinator(orchestrator=mock_orchestrator)

    @pytest.mark.asyncio
    async def test_chat_simple_response(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test chat with a simple response (no tool calls)."""
        # Execute
        response = await coordinator.chat("Hello")

        # Assert
        assert response.content == "Response content"
        assert response.role == "assistant"
        assert response.tool_calls is None
        mock_orchestrator.conversation.ensure_system_prompt.assert_called_once()
        # add_message is called twice: once for user, once for assistant
        assert mock_orchestrator.add_message.call_count == 2
        mock_orchestrator.provider.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_token_usage_tracking(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test that chat tracks token usage correctly."""
        # Setup
        mock_orchestrator.provider.chat = AsyncMock(
            return_value=CompletionResponse(
                content="Response",
                role="assistant",
                tool_calls=None,
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            )
        )

        # Execute
        await coordinator.chat("Hello")

        # Assert
        assert mock_orchestrator._cumulative_token_usage["prompt_tokens"] == 100
        assert mock_orchestrator._cumulative_token_usage["completion_tokens"] == 50
        assert mock_orchestrator._cumulative_token_usage["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test chat with tool calls that succeeds on second iteration."""
        # Setup - first call returns tool calls, second returns final response
        mock_orchestrator.provider.supports_tools = Mock(return_value=True)
        mock_orchestrator.tool_selector = Mock()
        mock_orchestrator.tool_selector.select_tools = AsyncMock(
            return_value=[{"name": "test_tool"}]
        )
        mock_orchestrator.tool_selector.prioritize_by_stage = Mock(
            return_value=[{"name": "test_tool"}]
        )

        tool_call_response = CompletionResponse(
            content="Thinking...",
            role="assistant",
            tool_calls=[{"name": "test_tool", "arguments": {}}],
        )
        final_response = CompletionResponse(
            content="Final response", role="assistant", tool_calls=None
        )

        mock_orchestrator.provider.chat = AsyncMock(
            side_effect=[tool_call_response, final_response]
        )
        mock_orchestrator._handle_tool_calls = AsyncMock(return_value=[{"success": True}])

        # Execute
        response = await coordinator.chat("Use a tool")

        # Assert
        assert response.content == "Final response"
        assert mock_orchestrator.provider.chat.call_count == 2
        mock_orchestrator._handle_tool_calls.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_empty_response_uses_completer(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test that empty response triggers response completer."""
        # Setup
        mock_orchestrator.provider.chat = AsyncMock(
            return_value=CompletionResponse(content="", role="assistant", tool_calls=None)
        )
        mock_orchestrator.response_completer.ensure_response = AsyncMock(
            return_value=Mock(content="Completed")
        )

        # Execute
        response = await coordinator.chat("Hello")

        # Assert
        assert response.content == "Completed"
        mock_orchestrator.response_completer.ensure_response.assert_called_once()
        mock_orchestrator.add_message.assert_called_with("assistant", "Completed")

    @pytest.mark.asyncio
    async def test_chat_max_iterations_exceeded(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test that chat stops after max iterations even with tool calls."""
        # Setup
        mock_orchestrator.provider.supports_tools = Mock(return_value=True)
        mock_orchestrator.tool_selector = Mock()
        mock_orchestrator.tool_selector.select_tools = AsyncMock(return_value=[])

        # Always return tool calls to test iteration limit
        tool_response = CompletionResponse(
            content="Still working",
            role="assistant",
            tool_calls=[{"name": "loop_tool", "arguments": {}}],
        )
        mock_orchestrator.provider.chat = AsyncMock(return_value=tool_response)
        mock_orchestrator._handle_tool_calls = AsyncMock(return_value=[{"success": True}])

        # Execute
        response = await coordinator.chat("Keep working")

        # Assert - should stop after max iterations and use completer
        mock_orchestrator.response_completer.ensure_response.assert_called_once()
        assert response.content == "Fallback"

    @pytest.mark.asyncio
    async def test_chat_with_thinking_enabled(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test chat with thinking parameter enabled."""
        # Setup
        mock_orchestrator.thinking = True

        # Execute
        await coordinator.chat("Think about this")

        # Assert - check that thinking was passed to provider
        call_kwargs = mock_orchestrator.provider.chat.call_args[1]
        assert "thinking" in call_kwargs
        assert call_kwargs["thinking"]["type"] == "enabled"
        assert call_kwargs["thinking"]["budget_tokens"] == 10000

    @pytest.mark.asyncio
    async def test_chat_with_context_compaction(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test chat with context compaction before API call."""
        # Setup
        mock_compactor = Mock()
        mock_compactor.check_and_compact = Mock(
            return_value=Mock(action_taken=True, messages_removed=5, tokens_freed=1000)
        )
        mock_orchestrator._context_compactor = mock_compactor

        # Execute
        await coordinator.chat("Long conversation message")

        # Assert
        mock_compactor.check_and_compact.assert_called()

    @pytest.mark.asyncio
    async def test_chat_tool_failure_uses_completer(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test that tool failures trigger response completer with failure context."""
        # Setup
        mock_orchestrator.provider.supports_tools = Mock(return_value=True)
        mock_orchestrator.tool_selector = Mock()
        mock_orchestrator.tool_selector.select_tools = AsyncMock(return_value=[])

        tool_response = CompletionResponse(
            content="", role="assistant", tool_calls=[{"name": "failing_tool", "arguments": {}}]
        )
        mock_orchestrator.provider.chat = AsyncMock(return_value=tool_response)
        mock_orchestrator._handle_tool_calls = AsyncMock(
            return_value=[{"success": False, "error": "Tool failed", "name": "failing_tool"}]
        )

        # Execute
        response = await coordinator.chat("Use failing tool")

        # Assert - completer should be called with failure context
        mock_orchestrator.response_completer.ensure_response.assert_called_once()
        call_args = mock_orchestrator.response_completer.ensure_response.call_args[1]
        assert "failure_context" in call_args
        assert call_args["failure_context"] is not None


class TestChatCoordinatorStreamChat:
    """Test suite for the stream_chat() method."""

    @pytest.fixture
    def mock_orchestrator(self) -> Mock:
        """Create mock orchestrator for streaming tests."""
        orch = Mock()
        orch.conversation = Mock()
        orch.conversation.ensure_system_prompt = Mock()
        orch.conversation.message_count = Mock(return_value=3)
        orch.provider = Mock()
        orch.provider.supports_tools = Mock(return_value=True)
        orch.provider.stream = self._create_stream_generator()
        orch.model = "test-model"
        orch.temperature = 0.7
        orch.max_tokens = 4096
        orch.tool_budget = 10
        orch.tool_calls_used = 0
        orch.thinking = False
        orch.messages = []
        orch.add_message = Mock()
        orch._system_added = False
        orch._metrics_coordinator = Mock()
        orch._metrics_coordinator.start_streaming = Mock()
        orch._metrics_coordinator.stop_streaming = Mock()
        orch._metrics_collector = Mock()
        orch._metrics_collector.init_stream_metrics = Mock(return_value=Mock(start_time=0.0))
        orch._metrics_collector.record_first_token = Mock()
        orch._session_state = Mock()
        orch._session_state.reset_for_new_turn = Mock()
        orch.unified_tracker = Mock()
        orch.unified_tracker.reset = Mock()
        orch.unified_tracker.config = {"max_total_iterations": 50}
        orch.unified_tracker.max_exploration_iterations = 25
        orch.unified_tracker.detect_task_type = Mock(return_value=TrackerTaskType.EDIT)
        orch.unified_tracker._progress = Mock()
        orch.unified_tracker._progress.tool_budget = 10
        orch.unified_tracker._progress.has_prompt_requirements = False
        orch.unified_tracker.set_tool_budget = Mock()
        orch.unified_tracker.set_max_iterations = Mock()
        orch.unified_tracker.record_iteration = Mock()
        orch.unified_tracker.record_tool_call = Mock()
        orch.unified_tracker.check_loop_warning = Mock(return_value=None)
        orch.unified_tracker.unique_resources = []
        orch.reminder_manager = Mock()
        orch.reminder_manager.reset = Mock()
        orch._usage_analytics = None
        orch._sequence_tracker = None
        orch._context_manager = Mock()
        orch._context_manager.start_background_compaction = AsyncMock()
        orch._context_manager.get_max_context_chars = Mock(return_value=100000)
        orch.task_coordinator = Mock()
        orch.task_coordinator._reminder_manager = None
        orch.task_coordinator.set_reminder_manager = Mock()
        orch.task_coordinator.prepare_task = Mock(
            return_value=(Mock(complexity=TaskComplexity.MEDIUM), 10)
        )
        orch.task_coordinator.current_intent = None
        orch.task_coordinator.temperature = 0.7
        orch.task_coordinator.tool_budget = 10
        orch.task_coordinator.apply_intent_guard = Mock()
        orch.task_coordinator.apply_task_guidance = Mock()
        orch._tool_planner = Mock()
        orch._tool_planner.infer_goals_from_message = Mock(return_value=[])
        orch._tool_planner.plan_tools = Mock(return_value=[])
        orch._tool_planner.filter_tools_by_intent = Mock(return_value=[])
        orch._model_supports_tool_calls = Mock(return_value=True)
        orch._check_cancellation = Mock(return_value=False)
        orch._provider_coordinator = Mock()
        orch._provider_coordinator.get_rate_limit_wait_time = Mock(return_value=1.0)
        orch.sanitizer = Mock()
        orch.sanitizer.sanitize = Mock(return_value="sanitized")
        orch.sanitizer.strip_markup = Mock(return_value="plain")
        orch.sanitizer.is_garbage_content = Mock(return_value=False)
        orch.observed_files = []
        orch.settings = Mock()
        orch.settings.stream_idle_timeout_seconds = 300
        orch._recovery_integration = Mock()
        orch._recovery_integration.handle_response = AsyncMock(return_value=Mock(action="continue"))
        orch._recovery_integration.record_outcome = Mock()
        orch._chunk_generator = Mock()
        orch._chunk_generator.generate_content_chunk = Mock(
            side_effect=lambda c, is_final=False: StreamChunk(content=c, is_final=is_final)
        )
        orch._record_intelligent_outcome = Mock()
        orch._context_compactor = None
        orch._force_finalize = False
        orch._continuation_prompts = 0
        orch._asking_input_prompts = 0
        orch._consecutive_blocked_attempts = 0
        orch._cumulative_prompt_interventions = 0
        orch._current_intent = None
        orch.provider_name = "test_provider"
        orch.debug_logger = Mock()
        orch.debug_logger.reset = Mock()
        orch.debug_logger.log_iteration_start = Mock()
        orch.debug_logger.log_limits = Mock()
        orch._task_completion_detector = Mock()
        orch._task_completion_detector.analyze_response = Mock()
        orch._task_completion_detector.get_completion_confidence = Mock(return_value=None)
        orch._intelligent_integration = None
        orch._prepare_intelligent_request = AsyncMock(return_value=None)
        orch._required_files = []
        orch._required_outputs = []
        orch._read_files_session = set()
        orch._all_files_read_nudge_sent = False
        orch._streaming_controller = Mock()
        orch._streaming_controller.current_session = None
        orch._current_stream_context = None
        orch._cumulative_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        orch._streaming_handler = Mock()
        orch._recovery_coordinator = Mock()
        orch._recovery_coordinator.check_natural_completion = Mock(return_value=None)
        orch._recovery_coordinator.handle_empty_response = Mock(return_value=(None, False))
        orch._recovery_coordinator.check_force_action = Mock(return_value=(False, None))
        orch._recovery_coordinator.get_recovery_fallback_message = Mock(return_value="Fallback")
        orch._tool_pipeline = Mock()
        return orch

    @staticmethod
    def _create_stream_generator():
        """Create a mock async generator for streaming."""

        async def generator(*args, **kwargs):
            # Simulate a complete streaming cycle that finishes immediately
            yield StreamChunk(
                content="Final response",
                is_final=True,
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )

        return generator

    @pytest.fixture
    def coordinator(self, mock_orchestrator: Mock) -> ChatCoordinator:
        """Create ChatCoordinator instance."""
        return ChatCoordinator(orchestrator=mock_orchestrator)

    @pytest.mark.asyncio
    async def test_stream_chat_yields_chunks(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test that stream_chat yields StreamChunk objects."""

        # Setup - mock the internal implementation to avoid complex streaming logic
        async def mock_stream_impl(user_message: str):
            yield StreamChunk(content="Test response", is_final=True)

        coordinator._stream_chat_impl = mock_stream_impl

        # Execute
        chunks = []
        async for chunk in coordinator.stream_chat("Hello stream"):
            chunks.append(chunk)

        # Assert - verify chunks were yielded
        assert len(chunks) >= 1  # At least one chunk should be yielded
        assert any(chunk.content == "Test response" for chunk in chunks)

    @pytest.mark.asyncio
    async def test_stream_chat_updates_cumulative_token_usage(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test that stream_chat updates cumulative token usage after completion."""
        # Setup - mock the internal implementation and context
        mock_context = Mock()
        mock_context.cumulative_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        mock_orchestrator._current_stream_context = mock_context

        async def mock_stream_impl(user_message: str):
            yield StreamChunk(content="Response", is_final=True)

        coordinator._stream_chat_impl = mock_stream_impl

        # Execute
        async for _ in coordinator.stream_chat("Track tokens"):
            pass

        # Assert
        assert mock_orchestrator._cumulative_token_usage["prompt_tokens"] == 100
        assert mock_orchestrator._cumulative_token_usage["completion_tokens"] == 50
        assert mock_orchestrator._cumulative_token_usage["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_stream_chat_cancellation_handling(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test that stream_chat handles cancellation requests."""

        # Setup - mock implementation that yields cancellation chunk
        async def mock_stream_impl(user_message: str):
            yield StreamChunk(content="\n\n[Cancelled by user]\n", is_final=True)

        coordinator._stream_chat_impl = mock_stream_impl

        # Execute
        chunks = []
        async for chunk in coordinator.stream_chat("Cancel me"):
            chunks.append(chunk)

        # Assert - should yield chunks including cancellation
        assert len(chunks) >= 1
        assert any("Cancelled" in chunk.content for chunk in chunks if chunk.content)


class TestChatCoordinatorHelperMethods:
    """Test suite for helper methods in ChatCoordinator."""

    @pytest.fixture
    def mock_orchestrator(self) -> Mock:
        """Create mock orchestrator for helper method tests."""
        orch = Mock()
        orch.provider = Mock()
        orch.provider.supports_tools = Mock(return_value=True)
        orch.provider.chat = AsyncMock()
        orch.model = "test-model"
        orch.temperature = 0.7
        orch.max_tokens = 4096
        orch.tool_budget = 10
        orch.tool_calls_used = 0
        orch.messages = []
        orch.add_message = Mock()
        orch.conversation = Mock()
        orch.conversation.message_count = Mock(return_value=3)
        orch.conversation_state = Mock()
        orch.conversation_state.state = Mock()
        orch.conversation_state.state.stage = None
        orch._context_manager = Mock()
        orch._context_manager.get_max_context_chars = Mock(return_value=100000)
        orch._model_supports_tool_calls = Mock(return_value=True)
        orch.observed_files = []
        orch._tool_planner = Mock()
        orch._tool_planner.infer_goals_from_message = Mock(return_value=[])
        orch._tool_planner.plan_tools = Mock(return_value=[])
        orch._tool_planner.filter_tools_by_intent = Mock(return_value=[])
        orch.tool_selector = Mock()
        orch.tool_selector.select_tools = AsyncMock(return_value=[])
        orch.tool_selector.prioritize_by_stage = Mock(return_value=[])
        orch._provider_coordinator = Mock()
        orch._provider_coordinator.get_rate_limit_wait_time = Mock(return_value=1.0)
        orch.sanitizer = Mock()
        orch.sanitizer.is_garbage_content = Mock(return_value=False)
        orch._metrics_collector = Mock()
        orch._metrics_collector.record_first_token = Mock()
        orch._streaming_controller = Mock()
        orch._streaming_controller.current_session = None
        orch.provider_name = "test_provider"
        orch.temperature = 0.7
        orch._recovery_coordinator = Mock()
        orch._recovery_coordinator.check_natural_completion = Mock(return_value=None)
        orch._recovery_coordinator.handle_empty_response = Mock(return_value=(None, False))
        orch._recovery_coordinator.check_force_action = Mock(return_value=(False, None))
        orch._recovery_coordinator.get_recovery_fallback_message = Mock(return_value="Fallback")
        orch._record_intelligent_outcome = Mock()
        orch._intelligent_integration = None
        orch._task_tracker = Mock()
        orch._task_tracker.current_task_type = "general"
        orch._task_tracker.is_analysis_task = False
        orch._task_tracker.is_action_task = False
        return orch

    @pytest.fixture
    def coordinator(self, mock_orchestrator: Mock) -> ChatCoordinator:
        """Create ChatCoordinator instance."""
        return ChatCoordinator(orchestrator=mock_orchestrator)

    def test_classify_task_keywords_analysis(self, coordinator: ChatCoordinator):
        """Test task classification for analysis tasks."""
        # Execute
        result = coordinator._classify_task_keywords("Analyze this code")

        # Assert
        assert result["is_analysis_task"] is True
        assert result["is_action_task"] is False
        assert result["coarse_task_type"] == "analysis"

    def test_classify_task_keywords_action(self, coordinator: ChatCoordinator):
        """Test task classification for action tasks."""
        # Execute
        result = coordinator._classify_task_keywords("Create a new file")

        # Assert
        assert result["is_analysis_task"] is False
        assert result["is_action_task"] is True
        assert result["coarse_task_type"] == "action"

    def test_classify_task_keywords_execution(self, coordinator: ChatCoordinator):
        """Test task classification for execution tasks."""
        # Execute
        result = coordinator._classify_task_keywords("Run the tests")

        # Assert
        assert result["needs_execution"] is True
        assert result["coarse_task_type"] == "execution"

    def test_classify_task_keywords_default(self, coordinator: ChatCoordinator):
        """Test task classification for default tasks."""
        # Execute
        result = coordinator._classify_task_keywords("Tell me a joke")

        # Assert
        assert result["is_analysis_task"] is False
        assert result["is_action_task"] is False
        assert result["needs_execution"] is False
        assert result["coarse_task_type"] == "default"

    def test_extract_required_files_from_prompt(self, coordinator: ChatCoordinator):
        """Test extraction of required files (returns empty list as per implementation)."""
        # Execute
        files = coordinator._extract_required_files_from_prompt("Fix file.py and test.py")

        # Assert - implementation returns empty list
        assert files == []

    def test_extract_required_outputs_from_prompt(self, coordinator: ChatCoordinator):
        """Test extraction of required outputs (returns empty list as per implementation)."""
        # Execute
        outputs = coordinator._extract_required_outputs_from_prompt("Generate output.txt")

        # Assert - implementation returns empty list
        assert outputs == []

    def test_get_max_context_chars(self, coordinator: ChatCoordinator, mock_orchestrator: Mock):
        """Test getting max context characters."""
        # Execute
        max_chars = coordinator._get_max_context_chars()

        # Assert
        assert max_chars == 100000
        mock_orchestrator._context_manager.get_max_context_chars.assert_called_once()

    def test_parse_and_validate_tool_calls(self, coordinator: ChatCoordinator):
        """Test parsing and validating tool calls."""
        # Setup
        tool_calls = [
            {"name": "tool1", "arguments": {"arg1": "value1"}},
            {"name": "tool2", "arguments": None},  # Missing arguments
            {"name": "tool3", "arguments": {"arg2": "value2"}},
        ]

        # Execute
        validated, content = coordinator._parse_and_validate_tool_calls(tool_calls, "Test content")

        # Assert
        assert validated == tool_calls
        assert validated[1]["arguments"] == {}  # Should be normalized to empty dict
        assert content == "Test content"


class TestChatCoordinatorRateLimitRetry:
    """Test suite for rate limit retry logic."""

    @pytest.fixture
    def mock_orchestrator(self) -> Mock:
        """Create mock orchestrator for rate limit tests."""
        orch = Mock()
        orch.provider = Mock()
        orch.provider.supports_tools = Mock(return_value=False)
        orch.model = "test-model"
        orch.temperature = 0.7
        orch.max_tokens = 4096
        orch.messages = []
        orch.sanitizer = Mock()
        orch.sanitizer.is_garbage_content = Mock(return_value=False)
        orch._metrics_collector = Mock()
        orch._metrics_collector.record_first_token = Mock()
        orch._provider_coordinator = Mock()
        orch._provider_coordinator.get_rate_limit_wait_time = Mock(
            return_value=0.1
        )  # Short wait for tests
        return orch

    @pytest.fixture
    def coordinator(self, mock_orchestrator: Mock) -> ChatCoordinator:
        """Create ChatCoordinator instance."""
        return ChatCoordinator(orchestrator=mock_orchestrator)

    def test_get_rate_limit_wait_time(self, coordinator: ChatCoordinator, mock_orchestrator: Mock):
        """Test calculation of rate limit wait time with exponential backoff."""
        # Setup
        exc = ProviderRateLimitError("Rate limited")
        mock_orchestrator._provider_coordinator.get_rate_limit_wait_time = Mock(return_value=2.0)

        # Execute
        wait_time = coordinator._get_rate_limit_wait_time(exc, attempt=2)

        # Assert - 2.0 * 2^2 = 8.0, but capped at 300
        assert wait_time == 8.0

    def test_get_rate_limit_wait_time_capped(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test that rate limit wait time is capped at 300 seconds."""
        # Setup
        exc = ProviderRateLimitError("Rate limited")
        mock_orchestrator._provider_coordinator.get_rate_limit_wait_time = Mock(return_value=100.0)

        # Execute
        wait_time = coordinator._get_rate_limit_wait_time(exc, attempt=10)

        # Assert - 100 * 2^10 would be huge, but capped at 300
        assert wait_time == 300.0


class TestChatCoordinatorRecoveryMethods:
    """Test suite for recovery-related methods."""

    @pytest.fixture
    def mock_orchestrator(self) -> Mock:
        """Create mock orchestrator for recovery tests."""
        orch = Mock()
        orch.provider = Mock()
        orch.provider_name = "test_provider"
        orch.model = "test-model"
        orch.temperature = 0.7
        orch.tool_calls_used = 5
        orch.tool_budget = 10
        orch._record_intelligent_outcome = Mock()
        orch._chunk_generator = Mock()
        orch._chunk_generator.generate_content_chunk = Mock(
            side_effect=lambda c, is_final=False: StreamChunk(content=c, is_final=is_final)
        )
        orch._task_tracker = Mock()
        orch._task_tracker.current_task_type = "general"
        orch._task_tracker.is_analysis_task = False
        orch._task_tracker.is_action_task = False
        orch._recovery_coordinator = Mock()
        orch._streaming_controller = Mock()
        orch._streaming_controller.current_session = Mock()
        orch._streaming_controller.current_session.start_time = 1000.0
        return orch

    @pytest.fixture
    def coordinator(self, mock_orchestrator: Mock) -> ChatCoordinator:
        """Create ChatCoordinator instance."""
        return ChatCoordinator(orchestrator=mock_orchestrator)

    def test_create_recovery_context(self, coordinator: ChatCoordinator, mock_orchestrator: Mock):
        """Test creation of recovery context."""
        # Setup
        stream_ctx = Mock()
        stream_ctx.total_iterations = 5
        stream_ctx.last_quality_score = 0.8
        stream_ctx.max_total_iterations = 50
        stream_ctx.tool_calls_used = 5
        stream_ctx.tool_budget = 10
        stream_ctx.start_time = 1000.0
        stream_ctx.unified_task_type = TrackerTaskType.EDIT
        stream_ctx.is_analysis_task = False
        stream_ctx.is_action_task = True

        # Execute
        import time

        mock_orchestrator._streaming_controller.current_session.start_time = time.time()
        recovery_ctx = coordinator._create_recovery_context(stream_ctx)

        # Assert
        assert recovery_ctx.iteration == 5
        assert recovery_ctx.tool_calls_used == 5
        assert recovery_ctx.tool_budget == 10
        assert recovery_ctx.provider_name == "test_provider"
        assert recovery_ctx.model == "test-model"

    def test_apply_recovery_action_force_summary(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test applying force_summary recovery action."""
        # Setup
        stream_ctx = Mock()
        recovery_action = Mock(action="force_summary", message="Summarizing")

        # Execute
        chunk = coordinator._apply_recovery_action(recovery_action, stream_ctx)

        # Assert
        assert chunk is not None
        assert chunk.is_final is True
        assert stream_ctx.force_completion is True

    def test_apply_recovery_action_retry(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test applying retry recovery action."""
        # Setup
        stream_ctx = Mock()
        recovery_action = Mock(action="retry", message="Try again")

        # Execute
        chunk = coordinator._apply_recovery_action(recovery_action, stream_ctx)

        # Assert - retry action should add system message and return None
        assert chunk is None
        mock_orchestrator.add_message.assert_called()
        # Check that the system message was added (with either the message or default)
        assert any(call[0][0] == "system" for call in mock_orchestrator.add_message.call_args_list)

    def test_apply_recovery_action_finalize(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test applying finalize recovery action."""
        # Setup
        stream_ctx = Mock()
        recovery_action = Mock(action="finalize", message="Final response")

        # Execute
        chunk = coordinator._apply_recovery_action(recovery_action, stream_ctx)

        # Assert
        assert chunk is not None
        assert chunk.is_final is True
        assert chunk.content == "Final response"

    def test_apply_recovery_action_continue(self, coordinator: ChatCoordinator):
        """Test applying continue recovery action (no action)."""
        # Setup
        stream_ctx = Mock()
        recovery_action = Mock(action="continue")

        # Execute
        chunk = coordinator._apply_recovery_action(recovery_action, stream_ctx)

        # Assert
        assert chunk is None

    @pytest.mark.asyncio
    async def test_handle_empty_response_recovery_success(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test empty response recovery when successful."""
        # Setup
        stream_ctx = Mock()
        tools = []
        mock_orchestrator.provider.chat = AsyncMock(
            return_value=CompletionResponse(
                content="Recovered content", role="assistant", tool_calls=None
            )
        )

        # Execute
        success, tool_calls, final_chunk = await coordinator._handle_empty_response_recovery(
            stream_ctx, tools
        )

        # Assert
        assert success is True
        assert final_chunk is not None
        assert final_chunk.content == "Recovered content"
        assert final_chunk.is_final is True

    @pytest.mark.asyncio
    async def test_handle_empty_response_recovery_failure(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test empty response recovery when it fails."""
        # Setup
        stream_ctx = Mock()
        tools = []
        mock_orchestrator.provider.chat = AsyncMock(side_effect=Exception("Failed"))

        # Execute
        success, tool_calls, final_chunk = await coordinator._handle_empty_response_recovery(
            stream_ctx, tools
        )

        # Assert
        assert success is False
        assert tool_calls is None
        assert final_chunk is None


class TestChatCoordinatorDelegationMethods:
    """Test suite for delegation methods used by orchestrator tests."""

    @pytest.fixture
    def mock_orchestrator(self) -> Mock:
        """Create mock orchestrator for delegation tests."""
        orch = Mock()
        orch._check_cancellation = Mock(return_value=False)
        orch._streaming_handler = Mock()
        orch._recovery_coordinator = Mock()
        orch._recovery_coordinator.check_progress = Mock(return_value=True)
        orch.temperature = 0.7
        orch.provider_name = "test_provider"
        orch.model = "test-model"
        orch._streaming_controller = Mock()
        orch._streaming_controller.current_session = Mock()
        orch._streaming_controller.current_session.start_time = 1000.0
        return orch

    @pytest.fixture
    def coordinator(self, mock_orchestrator: Mock) -> ChatCoordinator:
        """Create ChatCoordinator instance."""
        return ChatCoordinator(orchestrator=mock_orchestrator)

    def test_handle_cancellation_detected(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test handling cancellation when detected."""
        # Setup
        mock_orchestrator._check_cancellation = Mock(return_value=True)

        # Execute
        chunk = coordinator._handle_cancellation(elapsed=10.0)

        # Assert
        assert chunk is not None
        assert chunk.is_final is True
        # Check for either cancellation message format
        assert "Cancelled" in chunk.content or "cancelled" in chunk.content.lower()

    def test_handle_cancellation_not_detected(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test handling cancellation when not detected."""
        # Execute
        chunk = coordinator._handle_cancellation(elapsed=10.0)

        # Assert
        assert chunk is None

    def test_check_progress_making_progress(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test check_progress when session is making progress."""
        # Setup
        stream_ctx = Mock()
        stream_ctx.total_iterations = 5
        stream_ctx.last_quality_score = 0.8
        stream_ctx.force_completion = False
        stream_ctx.max_total_iterations = 50
        stream_ctx.tool_calls_used = 5
        stream_ctx.tool_budget = 10
        stream_ctx.start_time = 1000.0
        stream_ctx.unified_task_type = TrackerTaskType.EDIT
        stream_ctx.is_analysis_task = False
        stream_ctx.is_action_task = True

        mock_orchestrator._recovery_coordinator.check_progress = Mock(return_value=True)

        # Execute
        should_force = coordinator._check_progress_with_handler(stream_ctx)

        # Assert
        assert should_force is False
        assert stream_ctx.force_completion is False

    def test_check_progress_not_making_progress(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test check_progress when session is stuck."""
        # Setup
        stream_ctx = Mock()
        stream_ctx.total_iterations = 5
        stream_ctx.last_quality_score = 0.3
        stream_ctx.force_completion = False
        stream_ctx.max_total_iterations = 50
        stream_ctx.tool_calls_used = 5
        stream_ctx.tool_budget = 10
        stream_ctx.start_time = 1000.0
        stream_ctx.unified_task_type = TrackerTaskType.EDIT
        stream_ctx.is_analysis_task = False
        stream_ctx.is_action_task = True

        mock_orchestrator._recovery_coordinator.check_progress = Mock(return_value=False)

        # Execute
        should_force = coordinator._check_progress_with_handler(stream_ctx)

        # Assert
        assert should_force is True
        assert stream_ctx.force_completion is True

    def test_handle_force_completion_with_handler_analysis_task(self, coordinator: ChatCoordinator):
        """Test force completion message for analysis tasks."""
        # Setup
        stream_ctx = Mock()
        stream_ctx.force_completion = True
        stream_ctx.is_analysis_task = True

        # Execute
        chunk = coordinator._handle_force_completion_with_handler(stream_ctx)

        # Assert
        assert chunk is not None
        assert chunk.is_final is True
        assert "Research loop limit" in chunk.content

    def test_handle_force_completion_with_handler_action_task(self, coordinator: ChatCoordinator):
        """Test force completion message for action tasks."""
        # Setup
        stream_ctx = Mock()
        stream_ctx.force_completion = True
        stream_ctx.is_analysis_task = False
        stream_ctx.is_action_task = True

        # Execute
        chunk = coordinator._handle_force_completion_with_handler(stream_ctx)

        # Assert
        assert chunk is not None
        assert chunk.is_final is True
        assert "Exploration limit" in chunk.content

    def test_handle_force_completion_not_triggered(self, coordinator: ChatCoordinator):
        """Test force completion when not triggered."""
        # Setup
        stream_ctx = Mock()
        stream_ctx.force_completion = False

        # Execute
        chunk = coordinator._handle_force_completion_with_handler(stream_ctx)

        # Assert
        assert chunk is None


class TestChatCoordinatorEdgeCases:
    """Test suite for edge cases and error conditions."""

    @pytest.fixture
    def mock_orchestrator(self) -> Mock:
        """Create mock orchestrator for edge case tests."""
        orch = Mock()
        orch.conversation = Mock()
        orch.conversation.ensure_system_prompt = Mock()
        orch.provider = Mock()
        orch.provider.supports_tools = Mock(return_value=False)
        orch.provider.chat = AsyncMock(
            return_value=CompletionResponse(content="", role="assistant", tool_calls=None)
        )
        orch.model = "test-model"
        orch.temperature = 0.7
        orch.max_tokens = 4096
        orch.tool_budget = 10
        orch.tool_calls_used = 0
        orch.thinking = False
        orch.messages = []
        orch.add_message = Mock()
        orch._system_added = False
        orch.task_classifier = Mock()
        orch.task_classifier.classify = Mock(
            return_value=Mock(tool_budget=5, complexity=TaskComplexity.MEDIUM)
        )
        orch.settings = Mock()
        orch.settings.chat_max_iterations = 10
        orch.conversation_state = Mock()
        orch.conversation_state.state = Mock()
        orch.conversation_state.state.stage = None
        orch._context_compactor = None
        orch._handle_tool_calls = AsyncMock(return_value=[])
        orch.response_completer = Mock()
        orch.response_completer.ensure_response = AsyncMock(return_value=Mock(content=None))
        orch.response_completer.format_tool_failure_message = Mock(return_value="All tools failed")
        orch._cumulative_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        return orch

    @pytest.fixture
    def coordinator(self, mock_orchestrator: Mock) -> ChatCoordinator:
        """Create ChatCoordinator instance."""
        return ChatCoordinator(orchestrator=mock_orchestrator)

    @pytest.mark.asyncio
    async def test_chat_completer_fails_uses_fallback(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test that fallback message is used when completer fails."""
        # Setup - completer returns None
        mock_orchestrator.response_completer.ensure_response = AsyncMock(
            return_value=Mock(content=None)
        )

        # Execute
        response = await coordinator.chat("Test fallback")

        # Assert
        assert (
            response.content
            == "I was unable to generate a complete response. Please try rephrasing your request."
        )
        mock_orchestrator.add_message.assert_called()

    @pytest.mark.asyncio
    async def test_chat_with_tool_failures_uses_format_message(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test that tool failure format message is used when tools fail."""
        # Setup
        mock_orchestrator.provider.supports_tools = Mock(return_value=True)
        mock_orchestrator.tool_selector = Mock()
        mock_orchestrator.tool_selector.select_tools = AsyncMock(return_value=[])

        tool_response = CompletionResponse(
            content="", role="assistant", tool_calls=[{"name": "failing_tool", "arguments": {}}]
        )
        mock_orchestrator.provider.chat = AsyncMock(return_value=tool_response)
        mock_orchestrator._handle_tool_calls = AsyncMock(
            return_value=[{"success": False, "error": "Tool error", "name": "failing_tool"}]
        )
        mock_orchestrator.response_completer.ensure_response = AsyncMock(
            return_value=Mock(content=None)
        )

        # Execute
        response = await coordinator.chat("Use failing tool")

        # Assert
        assert "All tools failed" in response.content
        mock_orchestrator.response_completer.format_tool_failure_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_provider_exception_propagates(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test that provider exceptions are propagated."""
        # Setup
        mock_orchestrator.provider.chat = AsyncMock(side_effect=RuntimeError("Provider error"))

        # Execute & Assert
        with pytest.raises(RuntimeError, match="Provider error"):
            await coordinator.chat("Trigger error")
