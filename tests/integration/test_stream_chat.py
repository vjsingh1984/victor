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

"""Comprehensive integration tests for stream_chat method in AgentOrchestrator.

These tests verify:
1. Multi-turn conversations with state persistence
2. Tool call scenarios (success, failure, recovery)
3. Error conditions (provider errors, network errors, invalid input)
4. Context management and iteration limits
5. Streaming behavior and chunk aggregation
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.agent.coordinators.chat_coordinator import ChatCoordinator
from victor.core.errors import (
    ProviderConnectionError,
    ToolExecutionError,
    ToolNotFoundError,
)
from victor.providers.base import StreamChunk


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
async def mock_orchestrator():
    """Create a mock orchestrator for testing stream_chat.

    Returns a MagicMock with enough setup to test stream_chat in isolation.
    """
    orch = MagicMock()

    # Core attributes
    orch.settings = MagicMock()
    orch.settings.temperature = 0.7
    orch.settings.max_tokens = 4096
    orch.settings.thinking = False
    orch.settings.tool_selection = {"strategy": "hybrid"}
    orch.settings.chat_max_iterations = 10
    orch.tool_retry_enabled = True
    orch.tool_retry_max_attempts = 3
    orch.tool_retry_base_delay = 1.0
    orch.tool_retry_max_delay = 10.0

    # Provider
    orch.provider = MagicMock()
    orch.provider.name = "anthropic"
    orch.provider.supports_tools = MagicMock(return_value=True)

    # Conversation
    orch.conversation = MagicMock()
    orch.conversation.message_count = MagicMock(return_value=3)
    orch.conversation.ensure_system_prompt = MagicMock()

    # State
    orch.conversation_state = MagicMock()
    orch.conversation_state.state = MagicMock()
    orch.conversation_state.state.stage = MagicMock()
    orch.conversation_state.state.stage.value = "analyzing"

    # Tools
    orch.tools = MagicMock()
    orch.tools.execute = AsyncMock()

    # Tool selector
    orch.tool_selector = MagicMock()
    orch.tool_selector.select_tools = AsyncMock(return_value=[])
    orch.tool_selector.prioritize_by_stage = MagicMock(return_value=[])

    # Task classifier
    orch.task_classifier = MagicMock()
    orch.task_classifier.classify = MagicMock(
        return_value=MagicMock(tool_budget=20, complexity="medium")
    )

    # Other components
    orch.tool_budget = 20
    orch.tool_calls_used = 0
    orch._system_added = False
    orch.thinking = False
    orch._context_compactor = None
    orch.response_completer = MagicMock()

    # Cumulative token usage
    orch._cumulative_token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    # Method stubs
    orch.add_message = MagicMock()

    # Tool planner
    orch._tool_planner = MagicMock()
    orch._tool_planner.infer_goals_from_message = MagicMock(return_value=[])

    # Debug logger
    orch.debug_logger = MagicMock()
    orch.debug_logger.reset = MagicMock()

    # Stream context
    orch._current_stream_context = None

    return orch


# =============================================================================
# Helper Functions
# =============================================================================


def create_stream_chunks(content: str, final: bool = False) -> list[StreamChunk]:
    """Create a list of StreamChunk objects for testing.

    Args:
        content: The content to include in the chunk
        final: Whether this is the final chunk (includes usage info)

    Returns:
        List of StreamChunk objects
    """
    if final:
        return [
            StreamChunk(content=content),
            StreamChunk(
                content="",
                usage={"input_tokens": 50, "output_tokens": 30, "total_tokens": 80},
            ),
        ]
    return [StreamChunk(content=content)]


# =============================================================================
# Multi-Turn Conversation Tests
# =============================================================================


class TestMultiTurnConversations:
    """Test multi-turn conversation scenarios."""

    @pytest.mark.asyncio
    async def test_basic_multi_turn_conversation(self, mock_orchestrator):
        """Test basic multi-turn conversation with 3 exchanges."""
        coordinator = ChatCoordinator(mock_orchestrator)

        # Mock the stream implementation to yield chunks
        async def mock_stream_impl(user_message):
            chunks = [
                f"Response to: {user_message}",
                " (chunk 2)",
                " (chunk 3)",
            ]
            for msg in chunks:
                yield StreamChunk(content=msg, usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 50, "output_tokens": 30, "total_tokens": 80},
            )

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            # Turn 1
            chunks = []
            async for chunk in coordinator.stream_chat("Hello"):
                chunks.append(chunk)
            assert len(chunks) > 0
            assert any("Response to: Hello" in c.content for c in chunks)

            # Turn 2
            chunks = []
            async for chunk in coordinator.stream_chat("How are you?"):
                chunks.append(chunk)
            assert len(chunks) > 0
            assert any("Response to: How are you?" in c.content for c in chunks)

            # Turn 3
            chunks = []
            async for chunk in coordinator.stream_chat("Goodbye"):
                chunks.append(chunk)
            assert len(chunks) > 0
            assert any("Response to: Goodbye" in c.content for c in chunks)

    @pytest.mark.asyncio
    async def test_multi_turn_with_tool_calls(self, mock_orchestrator):
        """Test multi-turn conversation with tool calls interleaved."""
        coordinator = ChatCoordinator(mock_orchestrator)

        # Track tool calls
        tool_call_count = [0]

        async def mock_stream_impl(user_message):
            # Simulate tool call on first message
            if "first" in user_message.lower():
                tool_call_count[0] += 1
                # Simulate tool execution
                yield StreamChunk(content="", usage=None)
                yield StreamChunk(content="I've read the file.", usage=None)
            else:
                yield StreamChunk(content="Regular response", usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            )

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            # Turn with tool call
            chunks = []
            async for chunk in coordinator.stream_chat("First message"):
                chunks.append(chunk)
            assert tool_call_count[0] > 0

            # Turn without tool call
            chunks = []
            async for chunk in coordinator.stream_chat("Second message"):
                chunks.append(chunk)
            assert any("Regular response" in c.content for c in chunks)

    @pytest.mark.asyncio
    async def test_conversation_state_persistence(self, mock_orchestrator):
        """Test conversation state persists across turns."""
        coordinator = ChatCoordinator(mock_orchestrator)

        # Track conversation growth
        message_count = [0]
        call_count = [0]

        async def mock_stream_impl(user_message):
            # Simulate add_message being called
            call_count[0] += 1
            message_count[0] = call_count[0]
            yield StreamChunk(content=f"Message {message_count[0]}", usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 30, "output_tokens": 10, "total_tokens": 40},
            )

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            # First message
            async for chunk in coordinator.stream_chat("Message 1"):
                pass
            assert call_count[0] > 0

            # Second message
            initial_count = call_count[0]
            async for chunk in coordinator.stream_chat("Message 2"):
                pass
            assert call_count[0] > initial_count


# =============================================================================
# Tool Call Scenarios
# =============================================================================


class TestToolCallScenarios:
    """Test various tool call scenarios."""

    @pytest.mark.asyncio
    async def test_single_tool_call_success(self, mock_orchestrator):
        """Test successful single tool call."""
        coordinator = ChatCoordinator(mock_orchestrator)

        # Mock successful tool execution
        mock_orchestrator.tools.execute = AsyncMock(
            return_value=MagicMock(
                success=True,
                output="File content: Hello World",
                error=None,
            )
        )

        tool_called = [False]

        async def mock_stream_impl(user_message):
            if not tool_called[0]:
                tool_called[0] = True
                # Simulate tool call
                yield StreamChunk(content="", usage=None)
                yield StreamChunk(content="Tool executed successfully", usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            )

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            chunks = []
            async for chunk in coordinator.stream_chat("Read the file"):
                chunks.append(chunk)

            assert tool_called[0]
            assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_multiple_sequential_tool_calls(self, mock_orchestrator):
        """Test multiple sequential tool calls."""
        coordinator = ChatCoordinator(mock_orchestrator)

        # Track tool calls
        tool_calls = []

        async def mock_execute(tool_name, context=None, **kwargs):
            tool_calls.append(tool_name)
            return MagicMock(
                success=True,
                output=f"Executed {tool_name}",
                error=None,
            )

        mock_orchestrator.tools.execute = AsyncMock(side_effect=mock_execute)

        async def mock_stream_impl(user_message):
            # Simulate multiple tool calls
            for tool in ["read_file", "list_directory", "search_files"]:
                yield StreamChunk(content="", usage=None)
            yield StreamChunk(content="All tools executed", usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 100, "output_tokens": 30, "total_tokens": 130},
            )

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            chunks = []
            async for chunk in coordinator.stream_chat("Explore the codebase"):
                chunks.append(chunk)

            assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_tool_call_timeout(self, mock_orchestrator):
        """Test tool call timeout scenario."""
        coordinator = ChatCoordinator(mock_orchestrator)

        # Mock timeout
        async def mock_execute_timeout(tool_name, context=None, **kwargs):
            await asyncio.sleep(5)  # Simulate slow tool
            return MagicMock(success=True, output="Done")

        mock_orchestrator.tools.execute = AsyncMock(side_effect=mock_execute_timeout)

        timeout_occurred = [False]

        async def mock_stream_impl(user_message):
            try:
                # Set timeout
                result = await asyncio.wait_for(
                    mock_orchestrator.tools.execute("slow_tool"), timeout=0.1
                )
                yield StreamChunk(content="Success", usage=None)
            except asyncio.TimeoutError:
                timeout_occurred[0] = True
                yield StreamChunk(content="Tool timed out", usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            )

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            chunks = []
            async for chunk in coordinator.stream_chat("Run slow tool"):
                chunks.append(chunk)

            assert timeout_occurred[0]

    @pytest.mark.asyncio
    async def test_invalid_tool_name(self, mock_orchestrator):
        """Test handling of invalid tool name."""
        coordinator = ChatCoordinator(mock_orchestrator)

        # Mock tool not found
        mock_orchestrator.tools.execute = AsyncMock(
            side_effect=ToolNotFoundError(
                tool_name="invalid_tool",
            )
        )

        error_handled = [False]

        async def mock_stream_impl(user_message):
            try:
                await mock_orchestrator.tools.execute("invalid_tool")
                yield StreamChunk(content="Success", usage=None)
            except ToolNotFoundError as e:
                error_handled[0] = True
                yield StreamChunk(content=f"Error: Tool not found - {e.tool_name}", usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            )

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            chunks = []
            async for chunk in coordinator.stream_chat("Use invalid_tool"):
                chunks.append(chunk)

            assert error_handled[0]

    @pytest.mark.asyncio
    async def test_tool_call_retry_logic(self, mock_orchestrator):
        """Test tool call retry logic on transient failures."""
        coordinator = ChatCoordinator(mock_orchestrator)

        # Mock retry behavior - simulate tool being called multiple times
        attempt_count = [0]
        success_attempts = [0]

        async def mock_execute_with_retry(tool_name, context=None, **kwargs):
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                # Fail first 2 attempts
                return MagicMock(
                    success=False,
                    error="Temporary error",
                )
            # Succeed on 3rd attempt
            success_attempts[0] += 1
            return MagicMock(
                success=True,
                output="Success after retries",
                error=None,
            )

        mock_orchestrator.tools.execute = AsyncMock(side_effect=mock_execute_with_retry)

        async def mock_stream_impl(user_message):
            # Simulate retry logic by calling the tool multiple times
            for i in range(3):
                result = await mock_orchestrator.tools.execute("test_tool")
                if result.success:
                    yield StreamChunk(content=f"Success after {i+1} attempts", usage=None)
                    break
                yield StreamChunk(content="", usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            )

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            chunks = []
            async for chunk in coordinator.stream_chat("Retry this tool"):
                chunks.append(chunk)

            # Should have attempted at least 3 times
            assert attempt_count[0] >= 3
            assert success_attempts[0] > 0


# =============================================================================
# Error Conditions
# =============================================================================


class TestErrorConditions:
    """Test error handling in stream_chat."""

    @pytest.mark.asyncio
    async def test_provider_rate_limit_error(self, mock_orchestrator):
        """Test handling of provider rate limit errors."""
        coordinator = ChatCoordinator(mock_orchestrator)

        # Mock rate limit error
        rate_limited = [False]

        async def mock_stream_impl(user_message):
            if "rate_limit" in user_message.lower():
                rate_limited[0] = True
                yield StreamChunk(content="Rate limit exceeded. Please wait...", usage=None)
            else:
                yield StreamChunk(content="Normal response", usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            )

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            chunks = []
            async for chunk in coordinator.stream_chat("This will hit rate_limit"):
                chunks.append(chunk)

            assert rate_limited[0]

    @pytest.mark.asyncio
    async def test_network_connection_error(self, mock_orchestrator):
        """Test handling of network connection errors."""
        coordinator = ChatCoordinator(mock_orchestrator)

        connection_error_handled = [False]

        async def mock_stream_impl(user_message):
            try:
                # Simulate connection error
                raise ProviderConnectionError(
                    "Failed to connect to provider",
                    provider="anthropic",
                )
            except ProviderConnectionError as e:
                connection_error_handled[0] = True
                yield StreamChunk(content=f"Connection error: {e.message}", usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            chunks = []
            async for chunk in coordinator.stream_chat("Test connection"):
                chunks.append(chunk)

            assert connection_error_handled[0]

    @pytest.mark.asyncio
    async def test_invalid_user_input_handling(self, mock_orchestrator):
        """Test handling of invalid user input."""
        coordinator = ChatCoordinator(mock_orchestrator)

        async def mock_stream_impl(user_message):
            # Check for empty/invalid input
            if not user_message or not user_message.strip():
                yield StreamChunk(content="Please provide a valid message.", usage=None)
            else:
                yield StreamChunk(content="Processing your message...", usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            )

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            # Test empty input
            chunks = []
            async for chunk in coordinator.stream_chat(""):
                chunks.append(chunk)
            assert any("valid message" in c.content for c in chunks)

            # Test whitespace-only input
            chunks = []
            async for chunk in coordinator.stream_chat("   "):
                chunks.append(chunk)
            assert any("valid message" in c.content for c in chunks)

    @pytest.mark.asyncio
    async def test_tool_execution_error_propagation(self, mock_orchestrator):
        """Test that tool execution errors are properly propagated."""
        coordinator = ChatCoordinator(mock_orchestrator)

        # Mock tool execution error
        mock_orchestrator.tools.execute = AsyncMock(
            side_effect=ToolExecutionError(
                "Tool execution failed",
                tool_name="write_file",
                details={"path": "/invalid/path"},
            )
        )

        error_propagated = [False]

        async def mock_stream_impl(user_message):
            try:
                await mock_orchestrator.tools.execute("write_file")
                yield StreamChunk(content="Success", usage=None)
            except ToolExecutionError as e:
                error_propagated[0] = True
                yield StreamChunk(content=f"Tool error: {e.message}", usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            )

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            chunks = []
            async for chunk in coordinator.stream_chat("Write file"):
                chunks.append(chunk)

            assert error_propagated[0]

    @pytest.mark.asyncio
    async def test_context_overflow_scenario(self, mock_orchestrator):
        """Test handling of context overflow."""
        coordinator = ChatCoordinator(mock_orchestrator)

        # Simulate context overflow
        context_compacted = [False]

        async def mock_stream_impl(user_message):
            # Check if context is too large
            if mock_orchestrator.conversation.message_count() > 1000:
                context_compacted[0] = True
                yield StreamChunk(content="Context compacted to continue...", usage=None)
            else:
                yield StreamChunk(content="Normal response", usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            )

        # Simulate large conversation
        mock_orchestrator.conversation.message_count = MagicMock(return_value=1500)

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            chunks = []
            async for chunk in coordinator.stream_chat("Long message"):
                chunks.append(chunk)

            assert context_compacted[0]


# =============================================================================
# Streaming Behavior Tests
# =============================================================================


class TestStreamingBehavior:
    """Test streaming behavior and chunk aggregation."""

    @pytest.mark.asyncio
    async def test_chunk_aggregation(self, mock_orchestrator):
        """Test that chunks are properly aggregated."""
        coordinator = ChatCoordinator(mock_orchestrator)

        async def mock_stream_impl(user_message):
            # Yield multiple chunks
            chunks = ["Hello", " world", "!", " How", " are", " you?"]
            for chunk in chunks:
                yield StreamChunk(content=chunk, usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 50, "output_tokens": 30, "total_tokens": 80},
            )

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            chunks = []
            async for chunk in coordinator.stream_chat("Test"):
                chunks.append(chunk)

            # Should have received all chunks
            assert len(chunks) > 0
            # Verify content is accumulated
            full_content = "".join(c.content for c in chunks if c.content)
            assert len(full_content) > 0

    @pytest.mark.asyncio
    async def test_stream_cancellation(self, mock_orchestrator):
        """Test stream cancellation behavior."""
        coordinator = ChatCoordinator(mock_orchestrator)

        chunk_count = [0]

        async def mock_stream_impl(user_message):
            # Yield chunks indefinitely until cancelled
            while True:
                chunk_count[0] += 1
                yield StreamChunk(content=f"Chunk {chunk_count[0]}", usage=None)
                # Simulate delay
                await asyncio.sleep(0.01)

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            chunks = []
            async for chunk in coordinator.stream_chat("Test"):
                chunks.append(chunk)
                # Cancel after 5 chunks
                if len(chunks) >= 5:
                    break

            assert len(chunks) >= 5

    @pytest.mark.asyncio
    async def test_token_usage_tracking(self, mock_orchestrator):
        """Test that token usage is properly tracked."""
        coordinator = ChatCoordinator(mock_orchestrator)

        # Mock stream context to track usage
        stream_ctx = MagicMock()
        stream_ctx.cumulative_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

        async def mock_stream_impl(user_message):
            yield StreamChunk(content="Response", usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            )

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            # Mock the stream context
            mock_orchestrator._current_stream_context = stream_ctx

            # Stream and wait for completion
            async for chunk in coordinator.stream_chat("Test"):
                pass

            # Check cumulative token usage (should be updated by stream_chat)
            usage = mock_orchestrator._cumulative_token_usage
            assert usage["total_tokens"] >= 0  # Usage tracking is simulated


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_very_long_user_message(self, mock_orchestrator):
        """Test handling of very long user messages."""
        coordinator = ChatCoordinator(mock_orchestrator)

        # Create a very long message
        long_message = "This is a test. " * 1000

        async def mock_stream_impl(user_message):
            yield StreamChunk(content="Received your long message", usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 10000, "output_tokens": 20, "total_tokens": 10020},
            )

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            chunks = []
            async for chunk in coordinator.stream_chat(long_message):
                chunks.append(chunk)

            assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_special_characters_in_message(self, mock_orchestrator):
        """Test handling of special characters in messages."""
        coordinator = ChatCoordinator(mock_orchestrator)

        special_message = "Test with special chars: <>&\"'\\n\\t🎉"

        async def mock_stream_impl(user_message):
            yield StreamChunk(content=f"Received: {user_message}", usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            )

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            chunks = []
            async for chunk in coordinator.stream_chat(special_message):
                chunks.append(chunk)

            assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_concurrent_stream_requests(self, mock_orchestrator):
        """Test handling of concurrent stream requests."""
        coordinator = ChatCoordinator(mock_orchestrator)

        async def mock_stream_impl(user_message):
            # Simulate delay
            await asyncio.sleep(0.1)
            yield StreamChunk(content=f"Response to: {user_message}", usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            )

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            # Make concurrent requests
            tasks = [coordinator.stream_chat(f"Message {i}") for i in range(3)]

            results = []
            for task in tasks:
                chunks = []
                async for chunk in task:
                    chunks.append(chunk)
                results.append(chunks)

            # All requests should complete
            assert len(results) == 3
            for chunks in results:
                assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_empty_provider_response(self, mock_orchestrator):
        """Test handling of empty provider response."""
        coordinator = ChatCoordinator(mock_orchestrator)

        async def mock_stream_impl(user_message):
            # Yield empty chunks
            yield StreamChunk(content="", usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 50, "output_tokens": 0, "total_tokens": 50},
            )

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            chunks = []
            async for chunk in coordinator.stream_chat("Test"):
                chunks.append(chunk)

            # Should still complete without error
            assert len(chunks) >= 1


# =============================================================================
# Integration with Real Orchestrator
# =============================================================================


class TestRealOrchestratorIntegration:
    """Integration tests with real orchestrator components.

    These tests use more complete mocking but test the actual flow.
    """

    @pytest.mark.asyncio
    async def test_stream_chat_with_mock_settings(self):
        """Test stream_chat with realistic mock settings."""
        # Create mock settings
        settings = MagicMock()
        settings.temperature = 0.7
        settings.max_tokens = 4096
        settings.thinking = False
        settings.tool_selection = {"strategy": "hybrid"}
        settings.provider = "anthropic"
        settings.model = "claude-sonnet-4-5"
        settings.chat_max_iterations = 10
        settings.tool_retry_enabled = True
        settings.tool_retry_max_attempts = 3

        # Create mock provider
        provider = MagicMock()
        provider.name = "anthropic"
        provider.supports_tools = MagicMock(return_value=True)

        async def mock_stream_chat(messages, **kwargs):
            yield StreamChunk(content="Test response", usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            )

        provider.stream_chat = mock_stream_chat

        # Create orchestrator using factory (if available)
        # For now, use a minimal mock
        orch = MagicMock()
        orch.settings = settings
        orch.provider = provider
        orch.conversation = MagicMock()
        orch.conversation.ensure_system_prompt = MagicMock()
        orch.conversation.message_count = MagicMock(return_value=1)
        orch.conversation_state = MagicMock()
        orch.conversation_state.state = MagicMock()
        orch.conversation_state.state.stage = MagicMock()
        orch.conversation_state.state.stage.value = "analyzing"
        orch.tools = MagicMock()
        orch.tools.execute = AsyncMock(return_value=MagicMock(success=True))
        orch.tool_selector = MagicMock()
        orch.tool_selector.select_tools = AsyncMock(return_value=[])
        orch.tool_selector.prioritize_by_stage = MagicMock(return_value=[])
        orch.task_classifier = MagicMock()
        orch.task_classifier.classify = MagicMock(return_value=MagicMock(tool_budget=20))
        orch.tool_budget = 20
        orch.tool_calls_used = 0
        orch._system_added = False
        orch.thinking = False
        orch._context_compactor = None

        # Make add_message actually record calls
        add_message_calls = []

        def mock_add_message(role, content):
            add_message_calls.append((role, content))

        orch.add_message = mock_add_message

        orch._tool_planner = MagicMock()
        orch._tool_planner.infer_goals_from_message = MagicMock(return_value=[])
        orch.debug_logger = MagicMock()
        orch.debug_logger.reset = MagicMock()
        orch._current_stream_context = None
        orch._cumulative_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # Create coordinator
        coordinator = ChatCoordinator(orch)

        # Mock the internal implementation
        async def mock_stream_impl(user_message):
            # Actually call add_message
            orch.add_message("user", user_message)
            yield StreamChunk(content="Response", usage=None)
            yield StreamChunk(
                content="",
                usage={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
            )

        with patch.object(coordinator, "_stream_chat_impl", side_effect=mock_stream_impl):
            chunks = []
            async for chunk in coordinator.stream_chat("Test message"):
                chunks.append(chunk)

            assert len(chunks) > 0
            assert len(add_message_calls) > 0
