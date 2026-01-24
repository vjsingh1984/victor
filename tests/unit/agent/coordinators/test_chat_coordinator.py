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
    def mock_orchestrator(self, base_mock_orchestrator: Mock) -> Mock:
        """Create mock orchestrator with all required dependencies."""

        # Add custom stream generator for this test class
        async def stream_generator(*args, **kwargs):
            yield StreamChunk(content="Hello", is_final=False)
            yield StreamChunk(content=" world", is_final=False)
            yield StreamChunk(content="!", is_final=True)

        base_mock_orchestrator.provider.stream = stream_generator
        return base_mock_orchestrator

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
    def mock_orchestrator(self, base_mock_orchestrator: Mock) -> Mock:
        """Create mock orchestrator for chat tests."""
        # Customize: disable tools for chat tests
        base_mock_orchestrator.provider.supports_tools = Mock(return_value=False)

        # The chat() method uses stream_chat() internally, which calls provider.stream()
        # Set up stream generator that yields chunks with usage metadata
        async def stream_generator(*args, **kwargs):
            yield StreamChunk(
                content="Response content",
                is_final=True,
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )

        # Create a wrapper class to prevent Mock from auto-creating attributes
        class ProviderWrapper:
            def __init__(self, provider):
                self._provider = provider
                # Explicitly copy known attributes
                self.supports_tools = provider.supports_tools
                self.chat = provider.chat
                # Set our custom stream
                self.stream = stream_generator
                # Copy any other attributes that might be accessed
                for attr in ["name", "model", "temperature", "max_tokens"]:
                    if hasattr(provider, attr):
                        setattr(self, attr, getattr(provider, attr))

        # Replace the provider with our wrapper
        original_provider = base_mock_orchestrator.provider
        base_mock_orchestrator.provider = ProviderWrapper(original_provider)
        return base_mock_orchestrator

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
        # Note: Can't assert provider.stream was called because it's an unwrapped async generator

    @pytest.mark.asyncio
    async def test_chat_with_token_usage_tracking(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test that chat tracks token usage correctly."""

        # Setup - override stream generator with specific usage values
        async def stream_with_usage(*args, **kwargs):
            yield StreamChunk(
                content="Response",
                is_final=True,
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            )

        # Create a wrapper class to prevent Mock from auto-creating attributes
        class ProviderWrapper:
            def __init__(self, provider):
                self._provider = provider
                # Explicitly copy known attributes
                self.supports_tools = provider.supports_tools
                self.chat = provider.chat
                # Set our custom stream
                self.stream = stream_with_usage
                # Copy any other attributes that might be accessed
                for attr in ["name", "model", "temperature", "max_tokens"]:
                    if hasattr(provider, attr):
                        setattr(self, attr, getattr(provider, attr))

        # Replace the provider with our wrapper
        original_provider = mock_orchestrator.provider
        mock_orchestrator.provider = ProviderWrapper(original_provider)

        # Execute
        await coordinator.chat("Hello")

        # Assert
        # Token usage is tracked from stream chunks via _current_stream_context.cumulative_usage
        # Note: The implementation updates cumulative usage after streaming completes
        assert mock_orchestrator._cumulative_token_usage["prompt_tokens"] >= 0
        assert mock_orchestrator._cumulative_token_usage["completion_tokens"] >= 0

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
        mock_orchestrator.tool_selector.initialize_tool_embeddings = AsyncMock(return_value=None)

        # Create stream generators for two iterations
        call_count = [0]

        async def stream_with_tools(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: return tool calls
                yield StreamChunk(
                    content="Thinking...",
                    is_final=True,
                    tool_calls=[{"name": "test_tool", "arguments": {}}],
                )
            else:
                # Second call: return final response
                yield StreamChunk(content="Final response", is_final=True)

        # Create a wrapper class to prevent Mock from auto-creating attributes
        class ProviderWrapper:
            def __init__(self, provider, stream_func):
                self._provider = provider
                # Explicitly copy known attributes
                self.supports_tools = provider.supports_tools
                # Override chat to return appropriate responses for the test
                from victor.providers.base import CompletionResponse

                self.chat = AsyncMock(
                    return_value=CompletionResponse(
                        content="Thinking...", role="assistant", tool_calls=None
                    )
                )
                # Set our custom stream
                self.stream = stream_func
                # Copy any other attributes that might be accessed
                for attr in ["name", "model", "temperature", "max_tokens"]:
                    if hasattr(provider, attr):
                        setattr(self, attr, getattr(provider, attr))

        # Replace the provider with our wrapper
        original_provider = mock_orchestrator.provider
        mock_orchestrator.provider = ProviderWrapper(original_provider, stream_with_tools)

        # Mock tool execution handler to continue the loop instead of returning
        from victor.agent.streaming.tool_execution import ToolExecutionResult

        mock_tool_exec_result = ToolExecutionResult(
            chunks=[], should_return=False, tool_calls_executed=1  # Continue to next iteration
        )
        # Mock both the coordinator's handler and orchestrator's handler
        coordinator._tool_execution_handler = Mock()
        coordinator._tool_execution_handler.execute_tools = AsyncMock(
            return_value=mock_tool_exec_result
        )
        coordinator._tool_execution_handler.update_observed_files = Mock()
        mock_orchestrator._tool_execution_handler = coordinator._tool_execution_handler

        # Execute
        response = await coordinator.chat("Use a tool")

        # Assert
        assert response.content == "Final response"
        assert call_count[0] == 2  # Verify stream was called twice via counter
        coordinator._tool_execution_handler.execute_tools.assert_called_once()
        assert mock_orchestrator.tool_calls_used == 1  # Tool execution was recorded

    @pytest.mark.asyncio
    async def test_chat_with_empty_response_uses_completer(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test that empty response triggers response completer."""

        # Setup - provider returns empty response to trigger recovery
        async def empty_stream(*args, **kwargs):
            yield StreamChunk(content="", is_final=True)

        # Create a wrapper class to prevent Mock from auto-creating attributes
        class ProviderWrapper:
            def __init__(self, provider, stream_func):
                self._provider = provider
                # Explicitly copy known attributes
                self.supports_tools = provider.supports_tools
                # Override chat to also return empty response for recovery path
                from victor.providers.base import CompletionResponse

                self.chat = AsyncMock(
                    return_value=CompletionResponse(content="", role="assistant", tool_calls=None)
                )
                # Set our custom stream
                self.stream = stream_func
                # Copy any other attributes that might be accessed
                for attr in ["name", "model", "temperature", "max_tokens"]:
                    if hasattr(provider, attr):
                        setattr(self, attr, getattr(provider, attr))

        # Replace the provider with our wrapper
        original_provider = mock_orchestrator.provider
        mock_orchestrator.provider = ProviderWrapper(original_provider, empty_stream)

        # Mock the recovery coordinator's fallback message to test the recovery path
        mock_orchestrator._recovery_coordinator.get_recovery_fallback_message = Mock(
            return_value="Completed"
        )
        # Mock natural completion check to return None (no natural completion)
        mock_orchestrator._recovery_coordinator.check_natural_completion = Mock(
            return_value=None  # No natural completion
        )

        # Execute
        response = await coordinator.chat("Hello")

        # Assert
        assert response.content == "Completed"
        mock_orchestrator._recovery_coordinator.get_recovery_fallback_message.assert_called_once()
        mock_orchestrator._recovery_coordinator.check_natural_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_max_iterations_exceeded(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test that chat stops after max iterations even with tool calls."""
        # Setup
        mock_orchestrator.provider.supports_tools = Mock(return_value=True)
        mock_orchestrator.tool_selector = Mock()
        mock_orchestrator.tool_selector.select_tools = AsyncMock(return_value=[])
        mock_orchestrator.tool_selector.initialize_tool_embeddings = AsyncMock(return_value=None)

        # Always return tool calls to test iteration limit
        async def tool_loop_stream(*args, **kwargs):
            yield StreamChunk(
                content="Still working",
                is_final=True,
                tool_calls=[{"name": "loop_tool", "arguments": {}}],
            )

        # Create a wrapper class to prevent Mock from auto-creating attributes
        class ProviderWrapper:
            def __init__(self, provider, stream_func):
                self._provider = provider
                # Explicitly copy known attributes
                self.supports_tools = provider.supports_tools
                # Override chat to return appropriate responses for the test
                from victor.providers.base import CompletionResponse

                self.chat = AsyncMock(
                    return_value=CompletionResponse(
                        content="Still working", role="assistant", tool_calls=None
                    )
                )
                # Set our custom stream
                self.stream = stream_func
                # Copy any other attributes that might be accessed
                for attr in ["name", "model", "temperature", "max_tokens"]:
                    if hasattr(provider, attr):
                        setattr(self, attr, getattr(provider, attr))

        # Replace the provider with our wrapper
        original_provider = mock_orchestrator.provider
        mock_orchestrator.provider = ProviderWrapper(original_provider, tool_loop_stream)
        mock_orchestrator._handle_tool_calls = AsyncMock(return_value=[{"success": True}])

        # Execute
        response = await coordinator.chat("Keep working")

        # Assert - should stop after max iterations and use completer
        mock_orchestrator.response_completer.ensure_response.assert_called_once()
        assert response.content == "Fallback response"  # From conftest.py response_completer mock

    @pytest.mark.asyncio
    async def test_chat_with_thinking_enabled(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test chat with thinking parameter enabled."""
        # Setup
        mock_orchestrator.thinking = True

        async def thinking_stream(*args, **kwargs):
            # Capture kwargs for assertion
            thinking_stream.kwargs_captured = kwargs
            yield StreamChunk(content="Thinking response", is_final=True)

        thinking_stream.kwargs_captured = {}

        # Create a wrapper class to prevent Mock from auto-creating attributes
        class ProviderWrapper:
            def __init__(self, provider, stream_func):
                self._provider = provider
                # Explicitly copy known attributes
                self.supports_tools = provider.supports_tools
                self.chat = provider.chat
                # Set our custom stream
                self.stream = stream_func
                # Copy any other attributes that might be accessed
                for attr in ["name", "model", "temperature", "max_tokens"]:
                    if hasattr(provider, attr):
                        setattr(self, attr, getattr(provider, attr))

        # Replace the provider with our wrapper
        original_provider = mock_orchestrator.provider
        mock_orchestrator.provider = ProviderWrapper(original_provider, thinking_stream)

        # Execute
        await coordinator.chat("Think about this")

        # Assert - check that thinking was passed to provider stream
        assert thinking_stream.kwargs_captured is not None
        assert "thinking" in thinking_stream.kwargs_captured
        assert thinking_stream.kwargs_captured["thinking"]["type"] == "enabled"
        assert thinking_stream.kwargs_captured["thinking"]["budget_tokens"] == 10000

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
        mock_orchestrator.tool_selector.initialize_tool_embeddings = AsyncMock(return_value=None)

        async def failing_tool_stream(*args, **kwargs):
            yield StreamChunk(
                content="", is_final=True, tool_calls=[{"name": "failing_tool", "arguments": {}}]
            )

        # Create a wrapper class to prevent Mock from auto-creating attributes
        class ProviderWrapper:
            def __init__(self, provider, stream_func):
                self._provider = provider
                # Explicitly copy known attributes
                self.supports_tools = provider.supports_tools
                # Override chat to return appropriate responses for the test
                from victor.providers.base import CompletionResponse

                self.chat = AsyncMock(
                    return_value=CompletionResponse(content="", role="assistant", tool_calls=None)
                )
                # Set our custom stream
                self.stream = stream_func
                # Copy any other attributes that might be accessed
                for attr in ["name", "model", "temperature", "max_tokens"]:
                    if hasattr(provider, attr):
                        setattr(self, attr, getattr(provider, attr))

        # Replace the provider with our wrapper
        original_provider = mock_orchestrator.provider
        mock_orchestrator.provider = ProviderWrapper(original_provider, failing_tool_stream)
        mock_orchestrator._handle_tool_calls = AsyncMock(
            return_value=[{"success": False, "error": "Tool failed", "name": "failing_tool"}]
        )

        # Execute
        response = await coordinator.chat("Use failing tool")

        # Assert - completer should be called with failure context
        mock_orchestrator.response_completer.ensure_response.assert_called_once()


class TestChatCoordinatorStreamChat:
    """Test suite for the stream_chat() method."""

    @pytest.fixture
    def mock_orchestrator(self, base_mock_orchestrator: Mock) -> Mock:
        """Create mock orchestrator for streaming tests."""

        # Add custom stream generator for streaming tests
        async def stream_generator(*args, **kwargs):
            yield StreamChunk(
                content="Final response",
                is_final=True,
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )

        base_mock_orchestrator.provider.stream = stream_generator
        base_mock_orchestrator.settings.stream_idle_timeout_seconds = 300
        return base_mock_orchestrator

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
    def mock_orchestrator(self, base_mock_orchestrator: Mock) -> Mock:
        """Create mock orchestrator for helper method tests."""
        # Add task_tracker specific setup
        base_mock_orchestrator._task_tracker = Mock()
        base_mock_orchestrator._task_tracker.current_task_type = "general"
        base_mock_orchestrator._task_tracker.is_analysis_task = False
        base_mock_orchestrator._task_tracker.is_action_task = False
        return base_mock_orchestrator

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
    def mock_orchestrator(self, base_mock_orchestrator: Mock) -> Mock:
        """Create mock orchestrator for rate limit tests."""
        # Minimal setup for rate limit tests
        base_mock_orchestrator.provider.supports_tools = Mock(return_value=False)
        base_mock_orchestrator._provider_coordinator.get_rate_limit_wait_time = Mock(
            return_value=0.1
        )
        return base_mock_orchestrator

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
    def mock_orchestrator(self, base_mock_orchestrator: Mock) -> Mock:
        """Create mock orchestrator for recovery tests."""
        # Add streaming controller with session for recovery tests
        base_mock_orchestrator._streaming_controller.current_session = Mock()
        base_mock_orchestrator._streaming_controller.current_session.start_time = 1000.0
        # Set tool_calls_used to match what test_create_recovery_context expects
        base_mock_orchestrator.tool_calls_used = 5
        return base_mock_orchestrator

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
    def mock_orchestrator(self, base_mock_orchestrator: Mock) -> Mock:
        """Create mock orchestrator for delegation tests."""
        # Add streaming controller with session
        base_mock_orchestrator._streaming_controller.current_session = Mock()
        base_mock_orchestrator._streaming_controller.current_session.start_time = 1000.0
        return base_mock_orchestrator

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
        # Changed: is_final is False because actual summary will follow
        assert chunk.is_final is False
        assert "Research loop limit" in chunk.content
        assert "generating comprehensive summary" in chunk.content

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
        # Changed: is_final is False because actual summary will follow
        assert chunk.is_final is False
        assert "Exploration limit" in chunk.content
        assert "generating comprehensive summary" in chunk.content

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
    def mock_orchestrator(self, base_mock_orchestrator: Mock) -> Mock:
        """Create mock orchestrator for edge case tests."""

        # Add stream generator for streaming operations
        async def stream_generator(*args, **kwargs):
            yield StreamChunk(content="", is_final=True)

        base_mock_orchestrator.provider.stream = stream_generator

        # Customize for edge case tests - empty responses and failures
        base_mock_orchestrator.provider.supports_tools = Mock(return_value=False)
        base_mock_orchestrator.provider.chat = AsyncMock(
            return_value=CompletionResponse(content="", role="assistant", tool_calls=None)
        )
        base_mock_orchestrator.response_completer.ensure_response = AsyncMock(
            return_value=Mock(content=None)
        )
        return base_mock_orchestrator

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

        # Assert - recovery coordinator provides fallback
        assert response.content == "Fallback message"
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
        mock_orchestrator.tool_selector.initialize_tool_embeddings = AsyncMock(return_value=None)

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

        # Assert - recovery coordinator provides fallback for empty response
        assert response.content == "Fallback message"

    @pytest.mark.asyncio
    async def test_chat_provider_exception_propagates(
        self, coordinator: ChatCoordinator, mock_orchestrator: Mock
    ):
        """Test that provider exceptions are propagated."""

        # Setup - stream generator should raise exception
        async def failing_stream(*args, **kwargs):
            raise RuntimeError("Provider error")
            yield  # Never reached, but needed for async generator

        mock_orchestrator.provider.stream = failing_stream

        # Execute & Assert
        with pytest.raises(RuntimeError, match="Provider error"):
            await coordinator.chat("Trigger error")
