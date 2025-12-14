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

"""Unit tests for streaming handler module."""

import time
import pytest
from unittest.mock import MagicMock

from victor.agent.streaming.context import StreamingChatContext
from victor.agent.streaming.handler import StreamingChatHandler
from victor.agent.streaming.iteration import (
    IterationAction,
    IterationResult,
    ProviderResponseResult,
    ToolExecutionResult,
)
from victor.config.settings import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    return Settings(
        analytics_enabled=False,
        use_semantic_tool_selection=False,
    )


@pytest.fixture
def mock_message_adder():
    """Create mock message adder."""
    adder = MagicMock()
    adder.add_message = MagicMock()
    return adder


@pytest.fixture
def handler(mock_settings, mock_message_adder):
    """Create a StreamingChatHandler for testing."""
    return StreamingChatHandler(
        settings=mock_settings,
        message_adder=mock_message_adder,
        session_time_limit=60.0,
    )


@pytest.fixture
def basic_context():
    """Create a basic StreamingChatContext."""
    return StreamingChatContext(user_message="test message")


class TestStreamingChatHandlerCreation:
    """Tests for StreamingChatHandler creation."""

    def test_handler_creation(self, mock_settings, mock_message_adder):
        """Handler can be created with required args."""
        handler = StreamingChatHandler(
            settings=mock_settings,
            message_adder=mock_message_adder,
        )
        assert handler.settings == mock_settings
        assert handler.message_adder == mock_message_adder

    def test_handler_custom_time_limit(self, mock_settings, mock_message_adder):
        """Handler accepts custom time limit."""
        handler = StreamingChatHandler(
            settings=mock_settings,
            message_adder=mock_message_adder,
            session_time_limit=120.0,
        )
        assert handler.session_time_limit == 120.0


class TestCheckTimeLimit:
    """Tests for check_time_limit method."""

    def test_within_limit_returns_none(self, handler, basic_context):
        """Returns None when within time limit."""
        result = handler.check_time_limit(basic_context)
        assert result is None

    def test_over_limit_returns_result(self, handler, mock_message_adder):
        """Returns result when over time limit."""
        ctx = StreamingChatContext(user_message="test")
        ctx.start_time = time.time() - 120  # 2 minutes ago
        handler.session_time_limit = 60.0  # 1 minute limit

        result = handler.check_time_limit(ctx)

        assert result is not None
        assert result.action == IterationAction.FORCE_COMPLETION
        assert ctx.force_completion is True
        mock_message_adder.add_message.assert_called_once()


class TestCheckIterationLimit:
    """Tests for check_iteration_limit method."""

    def test_within_limit_returns_none(self, handler):
        """Returns None when within iteration limit."""
        ctx = StreamingChatContext(
            user_message="test",
            max_total_iterations=30,
            total_iterations=10,
        )
        result = handler.check_iteration_limit(ctx)
        assert result is None

    def test_at_limit_returns_result(self, handler):
        """Returns result when at iteration limit."""
        ctx = StreamingChatContext(
            user_message="test",
            max_total_iterations=30,
            total_iterations=30,
        )
        result = handler.check_iteration_limit(ctx)
        assert result is not None
        assert result.action == IterationAction.BREAK


class TestCheckForceCompletion:
    """Tests for check_force_completion method."""

    def test_not_forced_returns_none(self, handler, basic_context):
        """Returns None when no force condition."""
        result = handler.check_force_completion(basic_context)
        assert result is None

    def test_forced_returns_result(self, handler):
        """Returns result when force_completion is True."""
        ctx = StreamingChatContext(user_message="test", force_completion=True)
        result = handler.check_force_completion(ctx)
        assert result is not None
        assert result.action == IterationAction.FORCE_COMPLETION

    def test_blocked_attempts_triggers_force(self, handler):
        """Returns result when blocked attempts exceed threshold."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=3,
            max_blocked_before_force=3,
        )
        result = handler.check_force_completion(ctx)
        assert result is not None


class TestHandleBlockedAttempts:
    """Tests for handle_blocked_attempts method."""

    def test_below_threshold_returns_none(self, handler):
        """Returns None when below threshold."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=1,
            max_blocked_before_force=3,
        )
        result = handler.handle_blocked_attempts(ctx)
        assert result is None
        assert ctx.consecutive_blocked_attempts == 2

    def test_at_threshold_returns_result(self, handler, mock_message_adder):
        """Returns result and resets counter at threshold."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=2,
            max_blocked_before_force=3,
        )
        result = handler.handle_blocked_attempts(ctx)
        assert result is not None
        assert result.action == IterationAction.YIELD_AND_CONTINUE
        assert ctx.consecutive_blocked_attempts == 0
        mock_message_adder.add_message.assert_called_once()


class TestProcessProviderResponse:
    """Tests for process_provider_response method."""

    def test_response_with_tool_calls(self, handler, basic_context):
        """Processes response with tool calls."""
        response = ProviderResponseResult(
            content="Let me check that",
            tool_calls=[{"name": "read_file", "arguments": {"path": "/test"}}],
            tokens_used=50.0,
        )
        result = handler.process_provider_response(response, basic_context)

        assert result.action == IterationAction.CONTINUE
        assert result.has_tool_calls is True
        assert basic_context.total_accumulated_chars > 0

    def test_response_with_content_only(self, handler, basic_context):
        """Processes response with content but no tool calls."""
        response = ProviderResponseResult(
            content="Here is the answer",
            tokens_used=25.0,
        )
        result = handler.process_provider_response(response, basic_context)

        assert result.action == IterationAction.YIELD_AND_CONTINUE
        assert result.has_content is True
        assert len(result.chunks) == 1

    def test_response_with_garbage(self, handler, basic_context):
        """Processes response with garbage detection."""
        response = ProviderResponseResult(
            content="some content",
            garbage_detected=True,
        )
        result = handler.process_provider_response(response, basic_context)

        assert basic_context.force_completion is True

    def test_empty_response(self, handler, basic_context):
        """Processes empty response."""
        response = ProviderResponseResult()
        result = handler.process_provider_response(response, basic_context)

        assert result.action == IterationAction.CONTINUE


class TestProcessToolResults:
    """Tests for process_tool_results method."""

    def test_successful_tool_results(self, handler, basic_context):
        """Processes successful tool results."""
        execution = ToolExecutionResult()
        execution.add_result(
            "read_file",
            success=True,
            args={"path": "/test.txt"},
            elapsed=0.5,
        )

        chunks = handler.process_tool_results(execution, basic_context)

        assert len(chunks) >= 1
        # Find the tool result chunk
        tool_result_chunks = [
            c for c in chunks if c.metadata and "tool_result" in c.metadata
        ]
        assert len(tool_result_chunks) == 1
        assert tool_result_chunks[0].metadata["tool_result"]["success"] is True

    def test_failed_tool_results(self, handler, basic_context):
        """Processes failed tool results."""
        execution = ToolExecutionResult()
        execution.add_result(
            "read_file",
            success=False,
            error="file not found",
            args={"path": "/missing.txt"},
        )

        chunks = handler.process_tool_results(execution, basic_context)

        tool_result_chunks = [
            c for c in chunks if c.metadata and "tool_result" in c.metadata
        ]
        assert len(tool_result_chunks) == 1
        assert tool_result_chunks[0].metadata["tool_result"]["success"] is False
        assert tool_result_chunks[0].metadata["tool_result"]["error"] == "file not found"

    def test_includes_thinking_status(self, handler, basic_context):
        """Includes thinking status chunk."""
        execution = ToolExecutionResult()
        execution.add_result("test_tool", success=True)

        chunks = handler.process_tool_results(execution, basic_context)

        status_chunks = [
            c for c in chunks if c.metadata and c.metadata.get("status")
        ]
        assert len(status_chunks) == 1
        assert "Thinking" in status_chunks[0].metadata["status"]


class TestGenerateToolStartChunk:
    """Tests for generate_tool_start_chunk method."""

    def test_generates_correct_chunk(self, handler):
        """Generates chunk with tool start metadata."""
        chunk = handler.generate_tool_start_chunk(
            tool_name="read_file",
            tool_args={"path": "/test.txt"},
            status_msg="Reading file...",
        )

        assert chunk.content == ""
        assert chunk.metadata is not None
        assert "tool_start" in chunk.metadata
        assert chunk.metadata["tool_start"]["name"] == "read_file"
        assert chunk.metadata["tool_start"]["arguments"]["path"] == "/test.txt"
        assert chunk.metadata["tool_start"]["status_msg"] == "Reading file..."


class TestShouldContinueLoop:
    """Tests for should_continue_loop method."""

    def test_continues_on_continue_action(self, handler, basic_context):
        """Returns True for CONTINUE action."""
        result = IterationResult(action=IterationAction.CONTINUE)
        assert handler.should_continue_loop(result, basic_context) is True

    def test_stops_on_break_action(self, handler, basic_context):
        """Returns False for BREAK action."""
        result = IterationResult(action=IterationAction.BREAK)
        assert handler.should_continue_loop(result, basic_context) is False

    def test_stops_on_iteration_limit(self, handler):
        """Returns False when iteration limit reached."""
        ctx = StreamingChatContext(
            user_message="test",
            max_total_iterations=10,
            total_iterations=10,
        )
        result = IterationResult(action=IterationAction.CONTINUE)
        assert handler.should_continue_loop(result, ctx) is False

    def test_stops_on_force_completion_without_tools(self, handler):
        """Returns False when force completion and no tool calls."""
        ctx = StreamingChatContext(user_message="test", force_completion=True)
        result = IterationResult(action=IterationAction.CONTINUE)
        assert handler.should_continue_loop(result, ctx) is False

    def test_continues_on_force_completion_with_tools(self, handler):
        """Returns True when force completion but has tool calls."""
        ctx = StreamingChatContext(user_message="test", force_completion=True)
        result = IterationResult(
            action=IterationAction.CONTINUE,
            tool_calls=[{"name": "test", "arguments": {}}],
        )
        # Force completion with tool calls should still process tools
        # Implementation may vary, but typically we let tool calls complete
        # This test documents expected behavior
        assert handler.should_continue_loop(result, ctx) is True


class TestCheckNaturalCompletion:
    """Tests for check_natural_completion method."""

    def test_returns_none_with_tool_calls(self, handler):
        """Returns None when there are tool calls."""
        ctx = StreamingChatContext(
            user_message="test",
            total_accumulated_chars=1000,
            substantial_content_threshold=500,
        )
        result = handler.check_natural_completion(ctx, has_tool_calls=True, content_length=0)
        assert result is None

    def test_returns_none_without_substantial_content(self, handler):
        """Returns None when content below threshold."""
        ctx = StreamingChatContext(
            user_message="test",
            total_accumulated_chars=100,
            substantial_content_threshold=500,
        )
        result = handler.check_natural_completion(ctx, has_tool_calls=False, content_length=0)
        assert result is None

    def test_returns_break_with_substantial_content(self, handler):
        """Returns break result when substantial content and no tools."""
        ctx = StreamingChatContext(
            user_message="test",
            total_accumulated_chars=600,
            substantial_content_threshold=500,
        )
        result = handler.check_natural_completion(ctx, has_tool_calls=False, content_length=0)
        assert result is not None
        assert result.action == IterationAction.BREAK


class TestHandleEmptyResponse:
    """Tests for handle_empty_response method."""

    def test_returns_none_below_threshold(self, handler):
        """Returns None when empty responses below threshold."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_empty_responses=1,
        )
        result = handler.handle_empty_response(ctx)
        assert result is None
        assert ctx.consecutive_empty_responses == 2

    def test_returns_result_at_threshold(self, handler, mock_message_adder):
        """Returns result and forces completion at threshold."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_empty_responses=2,  # Will become 3 which is threshold
        )
        result = handler.handle_empty_response(ctx)
        assert result is not None
        assert result.action == IterationAction.YIELD_AND_CONTINUE
        assert ctx.force_completion is True
        assert ctx.consecutive_empty_responses == 0
        mock_message_adder.add_message.assert_called_once()


class TestHandleBlockedToolCall:
    """Tests for handle_blocked_tool_call method."""

    def test_records_blocked_and_returns_chunk(self, handler, mock_message_adder):
        """Records blocked attempt and returns notification chunk."""
        ctx = StreamingChatContext(user_message="test")
        initial_blocked = ctx.total_blocked_attempts

        chunk = handler.handle_blocked_tool_call(
            ctx,
            tool_name="read_file",
            tool_args={"path": "/blocked"},
            block_reason="Already tried this 3 times",
        )

        assert ctx.total_blocked_attempts == initial_blocked + 1
        assert "⛔" in chunk.content
        mock_message_adder.add_message.assert_called_once()
        assert "TOOL BLOCKED" in mock_message_adder.add_message.call_args[0][1]


class TestCheckBlockedThreshold:
    """Tests for check_blocked_threshold method."""

    def test_returns_none_below_threshold(self, handler):
        """Returns None when below thresholds."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=1,
            total_blocked_attempts=2,
        )
        result = handler.check_blocked_threshold(
            ctx, all_blocked=False, consecutive_limit=4, total_limit=6
        )
        assert result is None

    def test_returns_result_at_consecutive_threshold(self, handler, mock_message_adder):
        """Returns result when consecutive blocked reaches limit."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=3,
            total_blocked_attempts=3,
        )
        result = handler.check_blocked_threshold(
            ctx, all_blocked=True, consecutive_limit=4, total_limit=10
        )
        assert result is not None
        assert result.clear_tool_calls is True
        mock_message_adder.add_message.assert_called()

    def test_returns_result_at_total_threshold(self, handler, mock_message_adder):
        """Returns result when total blocked reaches limit."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=0,
            total_blocked_attempts=6,
        )
        result = handler.check_blocked_threshold(
            ctx, all_blocked=False, consecutive_limit=4, total_limit=6
        )
        assert result is not None
        assert result.clear_tool_calls is True

    def test_resets_consecutive_when_not_all_blocked(self, handler):
        """Resets consecutive counter when some calls succeed."""
        ctx = StreamingChatContext(
            user_message="test",
            consecutive_blocked_attempts=2,
            total_blocked_attempts=2,
        )
        result = handler.check_blocked_threshold(
            ctx, all_blocked=False, consecutive_limit=4, total_limit=10
        )
        assert result is None
        assert ctx.consecutive_blocked_attempts == 0


class TestHandleForceToolExecution:
    """Tests for handle_force_tool_execution method."""

    def test_first_attempt_prompts_tool_call(self, handler, mock_message_adder):
        """First attempt prompts for actual tool call."""
        ctx = StreamingChatContext(user_message="test")
        result = handler.handle_force_tool_execution(ctx, mentioned_tools=["read_file"])

        assert result is not None
        assert ctx.force_tool_execution_attempts == 1
        mock_message_adder.add_message.assert_called_once()
        call_content = mock_message_adder.add_message.call_args[0][1]
        assert "read_file" in call_content

    def test_third_attempt_gives_up(self, handler, mock_message_adder):
        """After 3 attempts, gives up and requests summary."""
        ctx = StreamingChatContext(
            user_message="test",
            force_tool_execution_attempts=2,  # Will become 3
        )
        result = handler.handle_force_tool_execution(ctx, mentioned_tools=["search"])

        assert result is not None
        assert ctx.force_tool_execution_attempts == 0  # Reset
        # Should have called add_message for "unable to make tool calls"
        call_content = mock_message_adder.add_message.call_args[0][1]
        assert "unable to make tool calls" in call_content


class TestToolBudgetMethods:
    """Tests for tool budget and progress checking methods."""

    def test_check_tool_budget_below_threshold(self, handler):
        """No warning when below threshold."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_budget=300,
            tool_calls_used=100,
        )
        result = handler.check_tool_budget(ctx, warning_threshold=250)
        assert result is None

    def test_check_tool_budget_at_threshold(self, handler):
        """Warning when at threshold."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_budget=300,
            tool_calls_used=250,
        )
        result = handler.check_tool_budget(ctx, warning_threshold=250)
        assert result is not None
        assert result.chunks
        assert "Approaching tool budget limit" in result.chunks[0].content

    def test_check_tool_budget_exhausted_no_warning(self, handler):
        """No warning when budget exhausted (0 remaining)."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_budget=200,
            tool_calls_used=200,
        )
        result = handler.check_tool_budget(ctx, warning_threshold=250)
        assert result is None  # No warning because remaining is 0

    def test_check_budget_exhausted_true(self, handler):
        """Budget exhausted returns True when no budget left."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_budget=100,
            tool_calls_used=100,
        )
        assert handler.check_budget_exhausted(ctx) is True

    def test_check_budget_exhausted_false(self, handler):
        """Budget exhausted returns False when budget remains."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_budget=100,
            tool_calls_used=50,
        )
        assert handler.check_budget_exhausted(ctx) is False

    def test_get_budget_exhausted_chunks(self, handler):
        """Budget exhausted generates correct chunks."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_budget=200,
        )
        chunks = handler.get_budget_exhausted_chunks(ctx)
        assert len(chunks) == 2
        assert "Tool budget reached (200)" in chunks[0].content
        assert "Generating final summary" in chunks[1].content

    def test_truncate_tool_calls_within_budget(self, handler):
        """Tool calls not truncated when within budget."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_budget=10,
            tool_calls_used=5,
        )
        tool_calls = [{"name": "tool1"}, {"name": "tool2"}, {"name": "tool3"}]
        result = handler.truncate_tool_calls(tool_calls, ctx)
        assert len(result) == 3

    def test_truncate_tool_calls_over_budget(self, handler):
        """Tool calls truncated when over budget."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_budget=10,
            tool_calls_used=8,  # Only 2 remaining
        )
        tool_calls = [{"name": "tool1"}, {"name": "tool2"}, {"name": "tool3"}]
        result = handler.truncate_tool_calls(tool_calls, ctx)
        assert len(result) == 2

    def test_truncate_empty_tool_calls(self, handler):
        """Empty tool calls returns empty."""
        ctx = StreamingChatContext(user_message="test")
        result = handler.truncate_tool_calls([], ctx)
        assert result == []


class TestProgressChecking:
    """Tests for progress checking methods."""

    def test_check_progress_below_limit(self, handler):
        """No force when below consecutive limit."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_calls_used=5,  # Below default 8
        )
        result = handler.check_progress_and_force(ctx, base_max_consecutive=8)
        assert result is False
        assert ctx.force_completion is False

    def test_check_progress_good_progress(self, handler):
        """No force when making good progress."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_calls_used=10,
        )
        # Add enough unique resources (10/2 = 5 needed)
        for i in range(5):
            ctx.add_unique_resource(f"file{i}.py")

        result = handler.check_progress_and_force(ctx, base_max_consecutive=8)
        assert result is False
        assert ctx.force_completion is False

    def test_check_progress_not_enough_progress(self, handler):
        """Force completion when not enough progress."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_calls_used=10,
        )
        # Only 2 resources (need 5 = 10/2)
        ctx.add_unique_resource("file1.py")
        ctx.add_unique_resource("file2.py")

        result = handler.check_progress_and_force(ctx, base_max_consecutive=8)
        assert result is True
        assert ctx.force_completion is True

    def test_check_progress_analysis_task_lenient(self, handler):
        """Analysis tasks use lenient threshold (1/4 instead of 1/2)."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_calls_used=20,
            is_analysis_task=True,
        )
        # Need 20/4 = 5 resources for analysis task
        for i in range(5):
            ctx.add_unique_resource(f"file{i}.py")

        result = handler.check_progress_and_force(ctx, base_max_consecutive=8)
        # Analysis task limit is 50, so 20 calls won't trigger
        assert result is False

    def test_check_progress_already_forcing(self, handler):
        """No change when already forcing completion."""
        ctx = StreamingChatContext(
            user_message="test",
            tool_calls_used=10,
            force_completion=True,  # Already set
        )
        result = handler.check_progress_and_force(ctx, base_max_consecutive=8)
        assert result is False  # Returns False because it didn't set it


class TestResearchLoopDetection:
    """Tests for research loop detection methods."""

    def test_is_research_loop_true(self, handler):
        """Detects research loop correctly."""
        assert handler.is_research_loop("loop_detected", "research pattern detected") is True
        assert handler.is_research_loop("loop_detected", "Research loop: web_search") is True
        assert handler.is_research_loop("loop_detected", "RESEARCH pattern") is True

    def test_is_research_loop_false_wrong_reason(self, handler):
        """Not a research loop with different reason."""
        assert handler.is_research_loop("tool_budget", "research loop") is False
        assert handler.is_research_loop("max_iterations", "research") is False
        assert handler.is_research_loop("none", "research") is False

    def test_is_research_loop_false_no_research(self, handler):
        """Not a research loop without research in hint."""
        assert handler.is_research_loop("loop_detected", "tool budget exceeded") is False
        assert handler.is_research_loop("loop_detected", "max iterations") is False
        assert handler.is_research_loop("loop_detected", "") is False


class TestForceCompletionMessages:
    """Tests for force completion message generation."""

    def test_get_force_completion_chunks_research_loop(self, handler):
        """Research loop generates research-specific message."""
        ctx = StreamingChatContext(user_message="test")
        chunk, message = handler.get_force_completion_chunks(ctx, is_research_loop=True)

        assert "Research loop detected" in chunk.content
        assert "SYNTHESIZE" in message
        assert "search results" in message

    def test_get_force_completion_chunks_exploration_limit(self, handler):
        """Non-research generates exploration limit message."""
        ctx = StreamingChatContext(user_message="test")
        chunk, message = handler.get_force_completion_chunks(ctx, is_research_loop=False)

        assert "exploration limit" in chunk.content
        assert "FINAL COMPREHENSIVE ANSWER" in message
        assert "STOP using tools" in message

    def test_handle_force_completion_not_forcing(self, handler):
        """Returns None when force_completion is False."""
        ctx = StreamingChatContext(
            user_message="test",
            force_completion=False,
        )
        result = handler.handle_force_completion(ctx, "loop_detected", "research")
        assert result is None

    def test_handle_force_completion_research_loop(self, handler, mock_message_adder):
        """Handles research loop force completion."""
        ctx = StreamingChatContext(
            user_message="test",
            force_completion=True,
        )
        result = handler.handle_force_completion(ctx, "loop_detected", "research loop detected")

        assert result is not None
        assert len(result.chunks) == 1
        assert "Research loop detected" in result.chunks[0].content
        # Check system message was added
        mock_message_adder.add_message.assert_called_once()
        call_args = mock_message_adder.add_message.call_args
        assert call_args[0][0] == "system"
        assert "SYNTHESIZE" in call_args[0][1]

    def test_handle_force_completion_exploration_limit(self, handler, mock_message_adder):
        """Handles exploration limit force completion."""
        ctx = StreamingChatContext(
            user_message="test",
            force_completion=True,
        )
        result = handler.handle_force_completion(ctx, "max_iterations", "too many iterations")

        assert result is not None
        assert len(result.chunks) == 1
        assert "exploration limit" in result.chunks[0].content
        # Check system message was added
        mock_message_adder.add_message.assert_called_once()
        call_args = mock_message_adder.add_message.call_args
        assert call_args[0][0] == "system"
        assert "FINAL COMPREHENSIVE ANSWER" in call_args[0][1]


class TestRecoveryPrompts:
    """Tests for recovery prompt generation methods."""

    def test_get_recovery_prompts_analysis_task_with_budget(self, handler):
        """Analysis task with budget returns task continuation prompts."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=True,
            tool_calls_used=10,
            tool_budget=100,  # 80% threshold = 80, so 10 < 80 = has budget
        )
        prompts = handler.get_recovery_prompts(ctx, 0.5, has_thinking_mode=False)

        assert len(prompts) == 3
        # First prompt should be about continuing exploration
        assert "discovery tools" in prompts[0][0].lower() or "tool call" in prompts[0][0].lower()
        # Temperatures should increase
        assert prompts[1][1] > prompts[0][1]
        assert prompts[2][1] > prompts[1][1]

    def test_get_recovery_prompts_analysis_task_exhausted(self, handler):
        """Analysis task with exhausted budget returns summary prompts."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=True,
            tool_calls_used=90,  # Above 80% of 100
            tool_budget=100,
        )
        prompts = handler.get_recovery_prompts(ctx, 0.5, has_thinking_mode=False)

        assert len(prompts) == 3
        # Should be summary prompts since budget exhausted
        assert "summarize" in prompts[0][0].lower() or "findings" in prompts[0][0].lower()

    def test_get_recovery_prompts_thinking_mode(self, handler):
        """Thinking mode returns simpler prompts with lower temps."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=False,
            tool_calls_used=90,
            tool_budget=100,
        )
        prompts = handler.get_recovery_prompts(ctx, 0.5, has_thinking_mode=True)

        assert len(prompts) == 3
        # Should have lower temperature caps
        for _, temp in prompts:
            assert temp <= 0.9

    def test_get_recovery_prompts_with_thinking_prefix(self, handler):
        """Thinking prefix is added when provided."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=True,
            tool_calls_used=10,
            tool_budget=100,
        )
        prompts = handler.get_recovery_prompts(
            ctx, 0.5, has_thinking_mode=True, thinking_disable_prefix="/no_think"
        )

        assert len(prompts) == 3
        # First prompt should have the prefix
        assert "/no_think" in prompts[0][0]

    def test_get_recovery_prompts_action_task(self, handler):
        """Action task with budget returns task continuation prompts."""
        ctx = StreamingChatContext(
            user_message="test",
            is_action_task=True,
            is_analysis_task=False,
            tool_calls_used=10,
            tool_budget=100,
        )
        prompts = handler.get_recovery_prompts(ctx, 0.5, has_thinking_mode=False)

        assert len(prompts) == 3
        # Should be task continuation prompts
        assert "discovery tools" in prompts[0][0].lower() or "tool call" in prompts[0][0].lower()

    def test_get_recovery_prompts_standard_task(self, handler):
        """Standard task returns summary prompts."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=False,
            is_action_task=False,
            tool_calls_used=10,
            tool_budget=100,
        )
        prompts = handler.get_recovery_prompts(ctx, 0.5, has_thinking_mode=False)

        assert len(prompts) == 3
        # Should be summary prompts
        assert "summarize" in prompts[0][0].lower() or "findings" in prompts[0][0].lower()


class TestShouldUseToolsForRecovery:
    """Tests for should_use_tools_for_recovery method."""

    def test_uses_tools_for_analysis_task_first_attempts(self, handler):
        """Analysis task enables tools for first 2 attempts."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=True,
            tool_calls_used=10,
            tool_budget=100,
        )

        assert handler.should_use_tools_for_recovery(ctx, 1) is True
        assert handler.should_use_tools_for_recovery(ctx, 2) is True
        assert handler.should_use_tools_for_recovery(ctx, 3) is False

    def test_no_tools_for_exhausted_budget(self, handler):
        """No tools when budget is exhausted."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=True,
            tool_calls_used=90,  # Above 80% threshold
            tool_budget=100,
        )

        assert handler.should_use_tools_for_recovery(ctx, 1) is False

    def test_no_tools_for_standard_task(self, handler):
        """Standard tasks don't use tools for recovery."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=False,
            is_action_task=False,
            tool_calls_used=10,
            tool_budget=100,
        )

        assert handler.should_use_tools_for_recovery(ctx, 1) is False


class TestGetRecoveryFallbackMessage:
    """Tests for get_recovery_fallback_message method."""

    def test_analysis_task_with_tool_calls(self, handler):
        """Analysis task with tool calls returns file summary."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=True,
            tool_calls_used=5,
        )
        unique_resources = ["file1.py", "file2.py", "file3.py"]

        message = handler.get_recovery_fallback_message(ctx, unique_resources)

        assert "Analysis Summary" in message
        assert "3 files" in message
        assert "file1.py" in message

    def test_analysis_task_truncates_files(self, handler):
        """Analysis task truncates file list to 10."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=True,
            tool_calls_used=5,
        )
        unique_resources = [f"file{i}.py" for i in range(20)]

        message = handler.get_recovery_fallback_message(ctx, unique_resources)

        assert "20 files" in message
        # Should only show first 10
        assert "file9.py" in message
        assert "file10.py" not in message

    def test_non_analysis_task(self, handler):
        """Non-analysis task returns generic message."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=False,
            tool_calls_used=0,
        )
        unique_resources = []

        message = handler.get_recovery_fallback_message(ctx, unique_resources)

        assert "No tool calls were returned" in message
        assert "retry" in message.lower()

    def test_analysis_task_no_tool_calls(self, handler):
        """Analysis task with no tool calls returns generic message."""
        ctx = StreamingChatContext(
            user_message="test",
            is_analysis_task=True,
            tool_calls_used=0,
        )
        unique_resources = []

        message = handler.get_recovery_fallback_message(ctx, unique_resources)

        assert "No tool calls were returned" in message
