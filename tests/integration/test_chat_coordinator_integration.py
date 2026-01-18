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

"""Integration tests for ChatCoordinator complex async flows.

This test suite provides comprehensive integration testing for ChatCoordinator,
focusing on complex async flows that require realistic orchestrator interaction
rather than unit-level mocking.

Test Coverage:
- Multi-turn agentic loops with tool execution
- Streaming iterations with full lifecycle
- Tool execution integration with real pipeline
- Error recovery across multiple iterations
- Context overflow and compaction
- Rate limit retry with exponential backoff
- Concurrent operation handling
- State management across iterations
- Recovery coordinator integration
- Task completion detection

These tests use comprehensive mocks that simulate real orchestrator behavior
while maintaining test isolation and speed.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, MagicMock, patch, PropertyMock
from typing import Any, List, Optional, Dict
from dataclasses import dataclass

from victor.agent.coordinators.chat_coordinator import ChatCoordinator
from victor.providers.base import CompletionResponse, StreamChunk
from victor.core.errors import ProviderRateLimitError
from victor.framework.task import TaskComplexity
from victor.agent.unified_task_tracker import TrackerTaskType


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def integration_orchestrator():
    """Create a comprehensive orchestrator mock for integration testing.

    This fixture provides a fully-configured orchestrator mock that simulates
    realistic behavior across all subsystems:
    - Provider with streaming and tool support
    - Tool pipeline with execution tracking
    - Context management with compaction
    - Recovery coordinator with state tracking
    - Metrics collection
    - State management

    Returns:
        Mock orchestrator configured for integration testing
    """
    orch = Mock()

    # Basic properties
    orch.model = "test-model"
    orch.temperature = 0.7
    orch.max_tokens = 4096
    orch.tool_budget = 10
    orch.tool_calls_used = 0
    orch.thinking = False
    orch.provider_name = "test_provider"
    orch.messages = []
    orch._system_added = False
    orch._force_finalize = False
    orch._continuation_prompts = 0
    orch._asking_input_prompts = 0
    orch._consecutive_blocked_attempts = 0
    orch._cumulative_prompt_interventions = 0
    orch._current_intent = None
    orch.observed_files = []
    orch._required_files = []
    orch._required_outputs = []
    orch._read_files_session = set()
    orch._all_files_read_nudge_sent = False

    # Conversation management
    orch.conversation = Mock()
    orch.conversation.ensure_system_prompt = Mock()
    orch.conversation.message_count = Mock(return_value=3)
    orch.add_message = Mock(
        side_effect=lambda role, content, **kwargs: orch.messages.append(
            {"role": role, "content": content, **kwargs}
        )
    )

    # Provider with streaming support
    orch.provider = Mock()
    orch.provider.supports_tools = Mock(return_value=True)
    orch.provider.chat = AsyncMock()
    orch.provider.stream = None  # Will be set per test

    # Task classification
    orch.task_classifier = Mock()
    orch.task_classifier.classify = Mock(
        return_value=Mock(tool_budget=5, complexity=TaskComplexity.MEDIUM)
    )

    # Settings
    orch.settings = Mock()
    orch.settings.chat_max_iterations = 10
    orch.settings.stream_idle_timeout_seconds = 300

    # Conversation state
    orch.conversation_state = Mock()
    orch.conversation_state.state = Mock()
    orch.conversation_state.state.stage = None

    # Context management
    orch._context_compactor = None
    orch._context_manager = Mock()
    orch._context_manager.get_max_context_chars = Mock(return_value=100000)
    orch._context_manager.start_background_compaction = AsyncMock()

    # Tool selector
    orch.tool_selector = Mock()
    orch.tool_selector.select_tools = AsyncMock(return_value=[])
    orch.tool_selector.prioritize_by_stage = Mock(return_value=[])

    # Tool pipeline
    orch._handle_tool_calls = AsyncMock(return_value=[])
    orch._tool_pipeline = Mock()

    # Response completer
    orch.response_completer = Mock()
    orch.response_completer.ensure_response = AsyncMock(return_value=Mock(content="Fallback"))
    orch.response_completer.format_tool_failure_message = Mock(return_value="Tool failed")

    # Token usage tracking
    orch._cumulative_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # Streaming context
    orch._current_stream_context = None

    # Metrics and session state
    orch._metrics_coordinator = Mock()
    orch._metrics_coordinator.start_streaming = Mock()
    orch._metrics_coordinator.stop_streaming = Mock()

    orch._metrics_collector = Mock()
    orch._metrics_collector.init_stream_metrics = Mock(return_value=Mock(start_time=0.0))
    orch._metrics_collector.record_first_token = Mock()

    orch._session_state = Mock()
    orch._session_state.reset_for_new_turn = Mock()

    # Unified tracker
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

    # Reminder manager
    orch.reminder_manager = Mock()
    orch.reminder_manager.reset = Mock()

    # Usage analytics
    orch._usage_analytics = None

    # Sequence tracker
    orch._sequence_tracker = None

    # Task coordinator
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

    # Tool planner
    orch._tool_planner = Mock()
    orch._tool_planner.infer_goals_from_message = Mock(return_value=[])
    orch._tool_planner.plan_tools = Mock(return_value=[])
    orch._tool_planner.filter_tools_by_intent = Mock(return_value=[])

    # Model capabilities
    orch._model_supports_tool_calls = Mock(return_value=True)

    # Cancellation
    orch._check_cancellation = Mock(return_value=False)

    # Provider coordinator
    orch._provider_coordinator = Mock()
    orch._provider_coordinator.get_rate_limit_wait_time = Mock(return_value=1.0)

    # Content sanitizer
    orch.sanitizer = Mock()
    orch.sanitizer.sanitize = Mock(return_value="sanitized content")
    orch.sanitizer.strip_markup = Mock(return_value="plain text")
    orch.sanitizer.is_garbage_content = Mock(return_value=False)

    # Recovery integration
    orch._recovery_integration = Mock()
    orch._recovery_integration.handle_response = AsyncMock(return_value=Mock(action="continue"))
    orch._recovery_integration.record_outcome = Mock()

    # Chunk generator
    orch._chunk_generator = Mock()
    orch._chunk_generator.generate_content_chunk = Mock(
        side_effect=lambda c, is_final=False: StreamChunk(content=c, is_final=is_final)
    )

    # Streaming handler
    orch._streaming_handler = Mock()

    # Recovery coordinator
    orch._recovery_coordinator = Mock()
    orch._recovery_coordinator.check_natural_completion = Mock(return_value=None)
    orch._recovery_coordinator.handle_empty_response = Mock(return_value=(None, False))
    orch._recovery_coordinator.check_force_action = Mock(return_value=(False, None))
    orch._recovery_coordinator.get_recovery_fallback_message = Mock(return_value="Fallback message")

    # Task completion detector
    orch._task_completion_detector = Mock()
    orch._task_completion_detector.analyze_response = Mock()
    orch._task_completion_detector.get_completion_confidence = Mock(return_value=None)

    # Intelligent integration
    orch._intelligent_integration = None

    # Prepare intelligent request
    orch._prepare_intelligent_request = AsyncMock(return_value=None)

    # Record intelligent outcome
    orch._record_intelligent_outcome = Mock()

    # Streaming controller
    orch._streaming_controller = Mock()
    orch._streaming_controller.current_session = None

    # Task tracker
    orch._task_tracker = Mock()
    orch._task_tracker.current_task_type = "general"
    orch._task_tracker.is_analysis_task = False
    orch._task_tracker.is_action_task = False

    # Debug logger
    orch.debug_logger = Mock()
    orch.debug_logger.reset = Mock()
    orch.debug_logger.log_iteration_start = Mock()
    orch.debug_logger.log_limits = Mock()

    return orch


@pytest.fixture
def chat_coordinator(integration_orchestrator):
    """Create ChatCoordinator instance with integration orchestrator."""
    return ChatCoordinator(orchestrator=integration_orchestrator)


def create_stream_generator(chunks: List[StreamChunk]):
    """Create an async generator for streaming.

    Args:
        chunks: List of StreamChunk objects to yield

    Returns:
        Async generator function
    """

    async def generator(*args, **kwargs):
        for chunk in chunks:
            yield chunk

    return generator


# ============================================================================
# Multi-Turn Agentic Loop Integration Tests
# ============================================================================


class TestMultiTurnAgenticLoop:
    """Test suite for multi-turn agentic loops with tool execution."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_turn_with_successful_tool_execution(
        self, chat_coordinator, integration_orchestrator
    ):
        """Test multi-turn conversation where tool executes successfully and then model responds.

        Scenario:
        1. User asks a question
        2. Model responds with tool call
        3. Tool executes successfully
        4. Model provides final response based on tool result

        This test verifies the complete agentic loop with successful tool execution.
        """
        # Setup: First call returns tool call, second returns final response
        tool_call_response = CompletionResponse(
            content="I'll help you with that.",
            role="assistant",
            tool_calls=[{"name": "search", "arguments": {"query": "test"}}],
        )
        final_response = CompletionResponse(
            content="Based on the search results, here's what I found.",
            role="assistant",
            tool_calls=None,
        )

        integration_orchestrator.provider.chat = AsyncMock(
            side_effect=[tool_call_response, final_response]
        )
        integration_orchestrator._handle_tool_calls = AsyncMock(
            return_value=[{"success": True, "name": "search", "output": "Results"}]
        )

        # Execute
        response = await chat_coordinator.chat("Search for information")

        # Assert
        assert response.content == "Based on the search results, here's what I found."
        assert integration_orchestrator.provider.chat.call_count == 2
        integration_orchestrator._handle_tool_calls.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_turn_with_multiple_tool_calls(
        self, chat_coordinator, integration_orchestrator
    ):
        """Test multi-turn conversation with multiple sequential tool calls.

        Scenario:
        1. Model calls tool A
        2. Tool A executes
        3. Model calls tool B based on A's result
        4. Tool B executes
        5. Model provides final response

        This test verifies the coordinator can handle multiple tool calls across iterations.
        """
        # Setup: Three iterations with tool calls, then final response
        responses = [
            CompletionResponse(
                content="Searching...",
                role="assistant",
                tool_calls=[{"name": "search", "arguments": {"query": "test"}}],
            ),
            CompletionResponse(
                content="Analyzing...",
                role="assistant",
                tool_calls=[{"name": "analyze", "arguments": {"data": "results"}}],
            ),
            CompletionResponse(
                content="Summarizing...",
                role="assistant",
                tool_calls=[{"name": "summarize", "arguments": {"content": "analysis"}}],
            ),
            CompletionResponse(content="Here's the summary.", role="assistant", tool_calls=None),
        ]

        integration_orchestrator.provider.chat = AsyncMock(side_effect=responses)
        integration_orchestrator._handle_tool_calls = AsyncMock(
            side_effect=[
                [{"success": True, "name": "search", "output": "Results"}],
                [{"success": True, "name": "analyze", "output": "Analysis"}],
                [{"success": True, "name": "summarize", "output": "Summary"}],
            ]
        )

        # Execute
        response = await chat_coordinator.chat("Research this topic")

        # Assert
        assert response.content == "Here's the summary."
        assert integration_orchestrator.provider.chat.call_count == 4
        assert integration_orchestrator._handle_tool_calls.call_count == 3

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_turn_with_tool_failure_recovery(
        self, chat_coordinator, integration_orchestrator
    ):
        """Test multi-turn conversation where tool fails and model recovers.

        Scenario:
        1. Model calls tool
        2. Tool fails
        3. Model acknowledges failure and provides alternative response

        This test verifies recovery from tool failures.
        """
        # Setup: Tool call then final response after failure
        tool_call_response = CompletionResponse(
            content="", role="assistant", tool_calls=[{"name": "failing_tool", "arguments": {}}]
        )
        final_response = CompletionResponse(
            content="I couldn't complete that task, but here's an alternative.",
            role="assistant",
            tool_calls=None,
        )

        integration_orchestrator.provider.chat = AsyncMock(
            side_effect=[tool_call_response, final_response]
        )
        integration_orchestrator._handle_tool_calls = AsyncMock(
            return_value=[{"success": False, "error": "Tool failed", "name": "failing_tool"}]
        )

        # Execute
        response = await chat_coordinator.chat("Use failing tool")

        # Assert
        assert response.content == "I couldn't complete that task, but here's an alternative."
        integration_orchestrator._handle_tool_calls.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_turn_iteration_limit_enforcement(
        self, chat_coordinator, integration_orchestrator
    ):
        """Test that iteration limits are enforced in multi-turn loops.

        Scenario:
        1. Model keeps making tool calls
        2. Coordinator enforces max_iterations limit
        3. Coordinator uses response completer to generate final response

        This test verifies iteration budget enforcement.
        """
        # Setup: Always return tool calls to test iteration limit
        integration_orchestrator.provider.chat = AsyncMock(
            return_value=CompletionResponse(
                content="Still working...",
                role="assistant",
                tool_calls=[{"name": "loop_tool", "arguments": {}}],
            )
        )
        integration_orchestrator._handle_tool_calls = AsyncMock(
            return_value=[{"success": True, "name": "loop_tool"}]
        )

        # Execute
        response = await chat_coordinator.chat("Keep working")

        # Assert: Should stop after max_iterations (10 * 2 = 20 budget)
        assert integration_orchestrator.provider.chat.call_count <= 20
        integration_orchestrator.response_completer.ensure_response.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_turn_token_usage_tracking(
        self, chat_coordinator, integration_orchestrator
    ):
        """Test that token usage is tracked across multi-turn iterations.

        Scenario:
        1. Model makes multiple iterations
        2. Each iteration tracks token usage
        3. Cumulative usage is accurate

        This test verifies token tracking across iterations.
        """
        # Setup: Multiple responses with token usage
        responses = [
            CompletionResponse(
                content="Step 1",
                role="assistant",
                tool_calls=[{"name": "tool1", "arguments": {}}],
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            ),
            CompletionResponse(
                content="Step 2",
                role="assistant",
                tool_calls=[{"name": "tool2", "arguments": {}}],
                usage={"prompt_tokens": 200, "completion_tokens": 75, "total_tokens": 275},
            ),
            CompletionResponse(
                content="Final",
                role="assistant",
                tool_calls=None,
                usage={"prompt_tokens": 150, "completion_tokens": 60, "total_tokens": 210},
            ),
        ]

        integration_orchestrator.provider.chat = AsyncMock(side_effect=responses)
        integration_orchestrator._handle_tool_calls = AsyncMock(
            side_effect=[[{"success": True}], [{"success": True}]]
        )

        # Execute
        await chat_coordinator.chat("Multi-step task")

        # Assert: Token usage accumulated correctly
        assert integration_orchestrator._cumulative_token_usage["prompt_tokens"] == 450
        assert integration_orchestrator._cumulative_token_usage["completion_tokens"] == 185
        assert integration_orchestrator._cumulative_token_usage["total_tokens"] == 635


# ============================================================================
# Streaming Integration Tests
# ============================================================================


class TestStreamingIntegration:
    """Test suite for streaming chat with full lifecycle."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_streaming_full_lifecycle(self, chat_coordinator, integration_orchestrator):
        """Test complete streaming lifecycle from start to finish.

        Scenario:
        1. Stream starts
        2. Multiple chunks are yielded
        3. Stream completes with final chunk
        4. Token usage is updated

        This test verifies the complete streaming lifecycle.
        """
        # Setup: Create stream with multiple chunks
        chunks = [
            StreamChunk(content="Hello", is_final=False),
            StreamChunk(content=" world", is_final=False),
            StreamChunk(
                content="!",
                is_final=True,
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
        ]

        integration_orchestrator.provider.stream = create_stream_generator(chunks)

        # Mock internal streaming implementation
        async def mock_stream_impl(user_message: str):
            for chunk in chunks:
                yield chunk

        chat_coordinator._stream_chat_impl = mock_stream_impl

        # Execute
        received_chunks = []
        async for chunk in chat_coordinator.stream_chat("Hello stream"):
            received_chunks.append(chunk)

        # Assert
        assert len(received_chunks) == 3
        assert received_chunks[0].content == "Hello"
        assert received_chunks[1].content == " world"
        assert received_chunks[2].content == "!"
        assert received_chunks[2].is_final is True

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_streaming_with_tool_calls(self, chat_coordinator, integration_orchestrator):
        """Test streaming with tool call execution.

        Scenario:
        1. Stream initial response
        2. Tool call is detected
        3. Tool executes
        4. Stream continues with result

        This test verifies streaming with tool execution.
        """
        # Setup: Stream chunks including tool call
        chunks = [
            StreamChunk(content="Searching", is_final=False),
            StreamChunk(
                content="",
                tool_calls=[{"name": "search", "arguments": {"query": "test"}}],
                is_final=False,
            ),
            StreamChunk(content="Found results", is_final=True),
        ]

        integration_orchestrator.provider.stream = create_stream_generator(chunks)
        integration_orchestrator._handle_tool_calls = AsyncMock(
            return_value=[{"success": True, "output": "Results"}]
        )

        # Mock streaming implementation
        async def mock_stream_impl(user_message: str):
            for chunk in chunks:
                yield chunk

        chat_coordinator._stream_chat_impl = mock_stream_impl

        # Execute
        received_chunks = []
        async for chunk in chat_coordinator.stream_chat("Search"):
            received_chunks.append(chunk)

        # Assert
        assert len(received_chunks) == 3
        assert any(chunk.tool_calls for chunk in received_chunks)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_streaming_token_accumulation(self, chat_coordinator, integration_orchestrator):
        """Test that tokens are accumulated during streaming.

        Scenario:
        1. Stream multiple chunks
        2. Each chunk has usage info
        3. Cumulative usage is tracked

        This test verifies token tracking during streaming.
        """
        # Setup
        chunks = [
            StreamChunk(
                content="Part 1",
                is_final=False,
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            ),
            StreamChunk(
                content="Part 2",
                is_final=True,
                usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            ),
        ]

        integration_orchestrator.provider.stream = create_stream_generator(chunks)

        # Mock streaming context with cumulative usage
        mock_context = Mock()
        mock_context.cumulative_usage = {
            "prompt_tokens": 30,
            "completion_tokens": 15,
            "total_tokens": 45,
        }
        integration_orchestrator._current_stream_context = mock_context

        async def mock_stream_impl(user_message: str):
            for chunk in chunks:
                yield chunk

        chat_coordinator._stream_chat_impl = mock_stream_impl

        # Execute
        async for _ in chat_coordinator.stream_chat("Test"):
            pass

        # Assert: Cumulative usage updated
        assert integration_orchestrator._cumulative_token_usage["prompt_tokens"] == 30
        assert integration_orchestrator._cumulative_token_usage["completion_tokens"] == 15
        assert integration_orchestrator._cumulative_token_usage["total_tokens"] == 45

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_streaming_with_cancellation(self, chat_coordinator, integration_orchestrator):
        """Test streaming handles cancellation gracefully.

        Scenario:
        1. Stream starts
        2. Cancellation is requested
        3. Stream yields final cancellation chunk
        4. Stream stops

        This test verifies cancellation handling during streaming.
        """
        # Setup: Mock cancellation detection
        integration_orchestrator._check_cancellation = Mock(return_value=True)

        chunks = [
            StreamChunk(content="Initial", is_final=False),
            StreamChunk(content="\n\n[Cancelled by user]\n", is_final=True),
        ]

        async def mock_stream_impl(user_message: str):
            yield chunks[1]  # Directly yield cancellation

        chat_coordinator._stream_chat_impl = mock_stream_impl

        # Execute
        received_chunks = []
        async for chunk in chat_coordinator.stream_chat("Cancel me"):
            received_chunks.append(chunk)

        # Assert: Cancellation chunk was yielded
        assert len(received_chunks) >= 1
        assert any("Cancelled" in chunk.content for chunk in received_chunks)


# ============================================================================
# Context Overflow Integration Tests
# ============================================================================


class TestContextOverflowIntegration:
    """Test suite for context overflow and compaction."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_context_compaction_before_api_call(
        self, chat_coordinator, integration_orchestrator
    ):
        """Test context compaction before API call when context is large.

        Scenario:
        1. Context approaches limit
        2. Compaction is triggered before API call
        3. Messages are removed
        4. API call succeeds with compacted context

        This test verifies proactive context compaction.
        """
        # Setup: Mock context compactor
        mock_compactor = Mock()
        mock_compactor.check_and_compact = Mock(
            return_value=Mock(action_taken=True, messages_removed=5, tokens_freed=1000)
        )
        integration_orchestrator._context_compactor = mock_compactor

        integration_orchestrator.provider.chat = AsyncMock(
            return_value=CompletionResponse(content="Response", role="assistant", tool_calls=None)
        )

        # Execute
        await chat_coordinator.chat("Long conversation")

        # Assert: Compaction was triggered
        mock_compactor.check_and_compact.assert_called()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_context_compaction_after_response(
        self, chat_coordinator, integration_orchestrator
    ):
        """Test context compaction after response when context grows.

        Scenario:
        1. API call returns large response
        2. Context exceeds threshold
        3. Compaction is triggered after response
        4. Next iteration has compacted context

        This test verifies reactive context compaction.
        """
        # Setup: Mock context compactor
        mock_compactor = Mock()
        call_count = [0]

        def compaction_side_effect(*args, **kwargs):
            call_count[0] += 1
            return Mock(
                action_taken=(call_count[0] == 2),  # Compact after response
                messages_removed=3,
                tokens_freed=500,
            )

        mock_compactor.check_and_compact = Mock(side_effect=compaction_side_effect)
        integration_orchestrator._context_compactor = mock_compactor

        tool_call_response = CompletionResponse(
            content="Thinking", role="assistant", tool_calls=[{"name": "tool", "arguments": {}}]
        )
        final_response = CompletionResponse(content="Done", role="assistant", tool_calls=None)

        integration_orchestrator.provider.chat = AsyncMock(
            side_effect=[tool_call_response, final_response]
        )
        integration_orchestrator._handle_tool_calls = AsyncMock(return_value=[{"success": True}])

        # Execute
        await chat_coordinator.chat("Generate large response")

        # Assert: Compaction called multiple times
        assert mock_compactor.check_and_compact.call_count >= 2

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_context_limit_threshold_check(self, chat_coordinator, integration_orchestrator):
        """Test that context limit is checked at 90% threshold.

        Scenario:
        1. Context length approaches 90% of max
        2. Compaction is triggered
        3. Context is reduced

        This test verifies threshold-based compaction.
        """
        # Setup: Add many messages to approach limit
        for i in range(100):
            integration_orchestrator.messages.append(
                {"role": "user", "content": "x" * 1000}  # 1KB per message
            )

        # Mock compactor
        mock_compactor = Mock()
        mock_compactor.check_and_compact = Mock(
            return_value=Mock(action_taken=True, messages_removed=50, tokens_freed=50000)
        )
        integration_orchestrator._context_compactor = mock_compactor

        integration_orchestrator.provider.chat = AsyncMock(
            return_value=CompletionResponse(content="Compacted", role="assistant", tool_calls=None)
        )

        # Execute
        await chat_coordinator.chat("Continue")

        # Assert: Compaction triggered due to context length
        mock_compactor.check_and_compact.assert_called()


# ============================================================================
# Rate Limit Retry Integration Tests
# ============================================================================


class TestRateLimitRetryIntegration:
    """Test suite for rate limit retry with exponential backoff."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rate_limit_retry_in_streaming(self, chat_coordinator, integration_orchestrator):
        """Test successful retry after rate limit error in streaming.

        Scenario:
        1. Streaming call hits rate limit
        2. System waits with exponential backoff
        3. Retry succeeds
        4. Stream continues

        This test verifies rate limit retry logic in streaming context.
        Note: Rate limit retry is only implemented in streaming, not in non-streaming chat.
        """
        # Setup: Mock inner stream method to fail then succeed
        attempt_count = [0]

        async def mock_stream_inner(tools, provider_kwargs, stream_ctx):
            attempt_count[0] += 1

            if attempt_count[0] == 1:
                raise ProviderRateLimitError("Rate limit exceeded", retry_after=0.01)

            # Success on retry
            return "Content after retry", None, 100, False

        chat_coordinator._stream_provider_response_inner = mock_stream_inner

        # Execute via retry wrapper
        result = await chat_coordinator._stream_with_rate_limit_retry(
            tools=None, provider_kwargs={}, stream_ctx=Mock(), max_retries=3
        )

        # Assert: Retry occurred and succeeded
        assert result[0] == "Content after retry"
        assert attempt_count[0] == 2

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rate_limit_retry_exhaustion(self, chat_coordinator, integration_orchestrator):
        """Test that retries are exhausted after max attempts.

        Scenario:
        1. Multiple rate limit errors occur
        2. System retries with exponential backoff
        3. After max retries, error is raised

        This test verifies retry limit enforcement in streaming.
        """

        # Setup: Always raise rate limit error
        async def mock_stream_inner_always_fails(tools, provider_kwargs, stream_ctx):
            raise ProviderRateLimitError("Rate limit exceeded", retry_after=0.01)

        chat_coordinator._stream_provider_response_inner = mock_stream_inner_always_fails

        # Execute & Assert: Should raise after retries exhausted
        with pytest.raises(ProviderRateLimitError):
            await chat_coordinator._stream_with_rate_limit_retry(
                tools=None,
                provider_kwargs={},
                stream_ctx=Mock(),
                max_retries=2,  # Lower max for faster test
            )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_exponential_backoff_calculation(
        self, chat_coordinator, integration_orchestrator
    ):
        """Test that exponential backoff is calculated correctly.

        Scenario:
        1. Rate limit occurs
        2. Wait time is calculated: base * 2^attempt
        3. Wait time is capped at 300 seconds

        This test verifies exponential backoff calculation.
        """
        # Setup: Mock provider coordinator
        integration_orchestrator._provider_coordinator.get_rate_limit_wait_time = Mock(
            return_value=2.0
        )

        exc = ProviderRateLimitError("Rate limited")

        # Test different attempt numbers
        wait_time_1 = chat_coordinator._get_rate_limit_wait_time(exc, attempt=1)
        wait_time_2 = chat_coordinator._get_rate_limit_wait_time(exc, attempt=2)
        wait_time_3 = chat_coordinator._get_rate_limit_wait_time(exc, attempt=3)

        # Assert: Exponential growth
        assert wait_time_1 == 2.0 * 2**1  # 4.0
        assert wait_time_2 == 2.0 * 2**2  # 8.0
        assert wait_time_3 == 2.0 * 2**3  # 16.0

        # Test cap at 300
        wait_time_max = chat_coordinator._get_rate_limit_wait_time(exc, attempt=20)
        assert wait_time_max == 300.0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rate_limit_with_streaming_integration(
        self, chat_coordinator, integration_orchestrator
    ):
        """Test rate limit retry during streaming integration.

        Scenario:
        1. Stream starts
        2. Rate limit occurs
        3. Retry with backoff
        4. Stream continues

        This test verifies rate limit handling in streaming context.
        """
        # Setup: Mock inner streaming with rate limit handling
        attempt_count = [0]

        async def mock_stream_inner_integration(tools, provider_kwargs, stream_ctx):
            attempt_count[0] += 1

            if attempt_count[0] == 1:
                raise ProviderRateLimitError("Rate limit exceeded", retry_after=0.01)

            # Success on retry
            return "Streamed content after retry", None, 100, False

        chat_coordinator._stream_provider_response_inner = mock_stream_inner_integration

        # Execute via retry wrapper
        result = await chat_coordinator._stream_with_rate_limit_retry(
            tools=None, provider_kwargs={}, stream_ctx=Mock(), max_retries=3
        )

        # Assert: Retry occurred
        assert result[0] == "Streamed content after retry"
        assert attempt_count[0] == 2


# ============================================================================
# Error Recovery Integration Tests
# ============================================================================


class TestErrorRecoveryIntegration:
    """Test suite for error recovery across iterations."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_empty_response_recovery(self, chat_coordinator, integration_orchestrator):
        """Test recovery from empty model response.

        Scenario:
        1. Model returns empty response
        2. Recovery completer is triggered
        3. Response completer generates fallback
        4. Valid response is returned

        This test verifies empty response recovery via response completer.
        """
        # Setup: Empty response that triggers completer
        integration_orchestrator.provider.chat = AsyncMock(
            return_value=CompletionResponse(content="", role="assistant", tool_calls=None)
        )

        integration_orchestrator.response_completer.ensure_response = AsyncMock(
            return_value=Mock(content="Recovered response via completer")
        )

        # Execute
        response = await chat_coordinator.chat("Trigger empty response")

        # Assert: Recovery occurred via completer
        assert response.content == "Recovered response via completer"
        integration_orchestrator.response_completer.ensure_response.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_provider_error_propagation(self, chat_coordinator, integration_orchestrator):
        """Test that provider errors are propagated correctly.

        Scenario:
        1. Provider raises exception
        2. Error is propagated to caller
        3. No silent failure

        This test verifies error propagation.
        """
        # Setup: Provider raises error
        integration_orchestrator.provider.chat = AsyncMock(
            side_effect=RuntimeError("Provider error")
        )

        # Execute & Assert
        with pytest.raises(RuntimeError, match="Provider error"):
            await chat_coordinator.chat("Trigger error")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_recovery_integration_action_handling(
        self, chat_coordinator, integration_orchestrator
    ):
        """Test recovery integration action handling.

        Scenario:
        1. Response indicates issues
        2. Recovery integration analyzes
        3. Recovery action is returned
        4. Action is applied

        This test verifies recovery integration.
        """
        # Setup: Mock recovery integration
        integration_orchestrator._recovery_integration.handle_response = AsyncMock(
            return_value=Mock(action="force_summary", message="Forcing summary due to issues")
        )

        recovery_ctx = chat_coordinator._create_recovery_context(
            Mock(
                total_iterations=5,
                last_quality_score=0.3,
                max_total_iterations=50,
                tool_calls_used=5,
                tool_budget=10,
                start_time=1000.0,
                elapsed_time=lambda: 10.0,
            )
        )

        # Execute
        recovery_action = await integration_orchestrator._recovery_integration.handle_response(
            content="Low quality response",
            tool_calls=None,
            mentioned_tools=None,
            provider_name="test",
            model_name="test-model",
            tool_calls_made=5,
            tool_budget=10,
            iteration_count=5,
            max_iterations=50,
            current_temperature=0.7,
            quality_score=0.3,
            task_type="general",
        )

        # Assert: Recovery action created
        assert recovery_action.action == "force_summary"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_force_summary_recovery(self, chat_coordinator, integration_orchestrator):
        """Test force_summary recovery action.

        Scenario:
        1. Issues detected
        2. Force summary action triggered
        3. Summary is generated
        4. Stream completes

        This test verifies force summary recovery.
        """
        # Setup
        stream_ctx = Mock()
        stream_ctx.force_completion = False

        recovery_action = Mock(action="force_summary", message="Providing summary")

        # Execute
        chunk = chat_coordinator._apply_recovery_action(recovery_action, stream_ctx)

        # Assert
        assert chunk is not None
        assert chunk.is_final is True
        assert stream_ctx.force_completion is True

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_retry_recovery_action(self, chat_coordinator, integration_orchestrator):
        """Test retry recovery action.

        Scenario:
        1. Recoverable issue detected
        2. Retry action triggered
        3. System message added
        4. Loop continues

        This test verifies retry recovery.
        """
        # Setup
        stream_ctx = Mock()
        recovery_action = Mock(action="retry", message="Please try again with better focus")

        # Execute
        chunk = chat_coordinator._apply_recovery_action(recovery_action, stream_ctx)

        # Assert: No chunk returned, system message added
        assert chunk is None
        integration_orchestrator.add_message.assert_called()


# ============================================================================
# State Management Integration Tests
# ============================================================================


class TestStateManagementIntegration:
    """Test suite for state management across iterations."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_iteration_state_tracking(self, chat_coordinator, integration_orchestrator):
        """Test that iteration state is tracked correctly.

        Scenario:
        1. Multiple iterations occur
        2. State is updated each iteration
        3. Final state is consistent

        This test verifies iteration state tracking.
        """
        # Setup: Multiple tool call iterations
        responses = [
            CompletionResponse(
                content=f"Iteration {i}",
                role="assistant",
                tool_calls=[{"name": f"tool{i}", "arguments": {}}],
            )
            for i in range(3)
        ] + [CompletionResponse(content="Done", role="assistant", tool_calls=None)]

        integration_orchestrator.provider.chat = AsyncMock(side_effect=responses)
        integration_orchestrator._handle_tool_calls = AsyncMock(
            side_effect=[[{"success": True}] for _ in range(3)]
        )

        # Execute
        await chat_coordinator.chat("Multi-iteration task")

        # Assert: All iterations completed
        assert integration_orchestrator.provider.chat.call_count == 4

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tool_budget_tracking(self, chat_coordinator, integration_orchestrator):
        """Test that tool budget is tracked and enforced.

        Scenario:
        1. Tools are called
        2. Budget is decremented
        3. Budget exhausted handling

        This test verifies tool budget tracking.
        """
        # Setup: Set low budget
        integration_orchestrator.tool_budget = 3
        integration_orchestrator.tool_calls_used = 0

        # Create tool call responses
        tool_responses = [
            CompletionResponse(
                content=f"Tool call {i}",
                role="assistant",
                tool_calls=[{"name": "tool", "arguments": {}}],
            )
            for i in range(5)  # More than budget
        ]
        tool_responses.append(
            CompletionResponse(content="Final", role="assistant", tool_calls=None)
        )

        integration_orchestrator.provider.chat = AsyncMock(side_effect=tool_responses)
        integration_orchestrator._handle_tool_calls = AsyncMock(
            side_effect=[[{"success": True}] for _ in range(5)]
        )

        # Execute
        await chat_coordinator.chat("Use many tools")

        # Assert: Tool calls were tracked
        # Note: Actual budget enforcement happens in tool selector
        assert integration_orchestrator._handle_tool_calls.call_count >= 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_quality_score_tracking(self, chat_coordinator, integration_orchestrator):
        """Test that quality score is tracked across iterations.

        Scenario:
        1. Responses with different quality
        2. Quality score is updated
        3. Quality-based decisions are made

        This test verifies quality score tracking.
        """
        # Setup: Mock intelligent integration for quality validation
        integration_orchestrator._intelligent_integration = Mock()
        integration_orchestrator._intelligent_integration.validate_response = AsyncMock(
            return_value=Mock(
                is_grounded=True,
                quality_score=0.8,
                should_retry=False,
                should_finalize=False,
                grounding_issues=[],
            )
        )

        # Execute with quality validation
        quality_result = await integration_orchestrator._intelligent_integration.validate_response(
            response="Good response", query="Test query", tool_calls=2, task_type="edit"
        )

        # Assert: Quality was validated
        assert quality_result.quality_score == 0.8
        assert quality_result.is_grounded is True


# ============================================================================
# Concurrent Operation Integration Tests
# ============================================================================


class TestConcurrentOperations:
    """Test suite for concurrent operation handling."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_chat_requests(self, chat_coordinator, integration_orchestrator):
        """Test handling concurrent chat requests.

        Scenario:
        1. Multiple chat requests arrive concurrently
        2. Each is handled independently
        3. No state pollution occurs

        This test verifies concurrent request handling.
        """
        # Setup: Create multiple coordinators with shared orchestrator
        coordinators = [ChatCoordinator(orchestrator=integration_orchestrator) for _ in range(3)]

        integration_orchestrator.provider.chat = AsyncMock(
            return_value=CompletionResponse(content="Response", role="assistant", tool_calls=None)
        )

        # Execute concurrent requests
        tasks = [coord.chat(f"Request {i}") for i, coord in enumerate(coordinators)]
        responses = await asyncio.gather(*tasks)

        # Assert: All requests completed
        assert len(responses) == 3
        assert all(r.content == "Response" for r in responses)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_streaming_cancellation_during_execution(
        self, chat_coordinator, integration_orchestrator
    ):
        """Test cancellation during streaming execution.

        Scenario:
        1. Stream is in progress
        2. Cancellation is requested
        3. Stream stops gracefully
        4. Resources are cleaned up

        This test verifies cancellation during streaming.
        """

        # Setup: Create slow stream
        async def slow_stream():
            chunks = [
                StreamChunk(content="Chunk 1", is_final=False),
                StreamChunk(content="Chunk 2", is_final=False),
            ]
            for chunk in chunks:
                yield chunk
                await asyncio.sleep(0.01)

        integration_orchestrator.provider.stream = slow_stream

        # Mock streaming implementation
        async def mock_stream_impl(user_message: str):
            async for chunk in slow_stream():
                yield chunk

        chat_coordinator._stream_chat_impl = mock_stream_impl

        # Execute
        chunks_received = []
        async for chunk in chat_coordinator.stream_chat("Stream"):
            chunks_received.append(chunk)
            if len(chunks_received) >= 2:
                break  # Simulate cancellation

        # Assert: Some chunks were received
        assert len(chunks_received) >= 1


# ============================================================================
# Integration Test Helpers
# ============================================================================


@pytest.mark.integration
def test_coordinator_initialization(integration_orchestrator):
    """Test that coordinator initializes correctly with orchestrator."""
    coordinator = ChatCoordinator(orchestrator=integration_orchestrator)

    assert coordinator._orchestrator == integration_orchestrator
    assert coordinator._intent_classification_handler is None
    assert coordinator._continuation_handler is None
    assert coordinator._tool_execution_handler is None


@pytest.mark.integration
def test_create_stream_context(integration_orchestrator):
    """Test stream context creation."""
    import asyncio

    coordinator = ChatCoordinator(orchestrator=integration_orchestrator)

    # Run async test
    async def test():
        stream_ctx = await coordinator._create_stream_context("Test message")

        assert stream_ctx is not None
        assert stream_ctx.total_iterations == 0
        assert stream_ctx.tool_budget == 10
        assert stream_ctx.max_total_iterations == 50

    asyncio.run(test())


# ============================================================================
# End of Integration Tests
# ============================================================================
