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
# See the License for the specific language governing permissions
# limitations under the License.

"""Unit tests for RecoveryCoordinator.

Tests all 23 methods of RecoveryCoordinator with mocked dependencies.
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from typing import Any, Dict, List, Optional

from victor.agent.recovery_coordinator import RecoveryCoordinator, RecoveryContext
from victor.providers.base import StreamChunk


@pytest.fixture
def mock_recovery_handler():
    """Mock RecoveryHandler."""
    return Mock()


@pytest.fixture
def mock_recovery_integration():
    """Mock OrchestratorRecoveryIntegration."""
    mock = Mock()
    mock.enabled = True
    mock.handle_response = AsyncMock()
    return mock


@pytest.fixture
def mock_streaming_handler():
    """Mock StreamingChatHandler."""
    mock = Mock()
    # Set up common return values
    mock.check_time_limit = Mock(return_value=None)
    mock.check_iteration_limit = Mock(return_value=None)
    mock.check_natural_completion = Mock(return_value=False)
    mock.handle_empty_response = Mock(return_value=None)
    mock.handle_blocked_tool_call = Mock(return_value=StreamChunk(content="blocked", is_final=False))
    mock.check_blocked_threshold = Mock(return_value=None)
    mock.filter_blocked_tool_calls = Mock(return_value=([], [], 0))
    mock.check_force_action = Mock(return_value=(False, None))
    mock.handle_force_tool_execution = Mock()
    mock.check_tool_budget = Mock(return_value=None)
    mock.check_progress_and_force = Mock(return_value=False)
    mock.truncate_tool_calls = Mock(side_effect=lambda calls, ctx: calls)
    mock.handle_force_completion = Mock(return_value=None)
    mock.format_completion_metrics = Mock(return_value="metrics")
    mock.format_budget_exhausted_metrics = Mock(return_value="budget metrics")
    mock.generate_tool_result_chunks = Mock(return_value=[])
    mock.get_recovery_prompts = Mock(return_value=[])
    mock.should_use_tools_for_recovery = Mock(return_value=True)
    mock.get_recovery_fallback_message = Mock(return_value="fallback")
    mock.handle_loop_warning = Mock(return_value=None)
    return mock


@pytest.fixture
def mock_context_compactor():
    """Mock ContextCompactor."""
    mock = Mock()
    mock.get_statistics = Mock(return_value={"current_utilization": 0.5})
    return mock


@pytest.fixture
def mock_unified_tracker():
    """Mock UnifiedTaskTracker."""
    mock = Mock()
    mock.is_blocked_after_warning = Mock(return_value=False)
    mock.should_force_action = Mock(return_value=(False, None))
    mock.increment_turn = Mock()
    mock.should_stop = Mock(return_value=Mock(should_stop=False, reason="none", hint=""))
    mock.unique_resources = set()
    return mock


@pytest.fixture
def mock_settings():
    """Mock Settings."""
    mock = Mock()
    mock.recovery_blocked_consecutive_threshold = 4
    mock.recovery_blocked_total_threshold = 6
    mock.tool_call_budget_warning_threshold = 250
    mock.max_consecutive_tool_calls = 8
    return mock


@pytest.fixture
def recovery_coordinator(
    mock_recovery_handler,
    mock_recovery_integration,
    mock_streaming_handler,
    mock_context_compactor,
    mock_unified_tracker,
    mock_settings,
):
    """Create RecoveryCoordinator with mocked dependencies."""
    return RecoveryCoordinator(
        recovery_handler=mock_recovery_handler,
        recovery_integration=mock_recovery_integration,
        streaming_handler=mock_streaming_handler,
        context_compactor=mock_context_compactor,
        unified_tracker=mock_unified_tracker,
        settings=mock_settings,
    )


@pytest.fixture
def mock_streaming_context():
    """Mock StreamingChatContext."""
    mock = Mock()
    mock.total_iterations = 5
    mock.max_total_iterations = 10
    mock.total_accumulated_chars = 500
    mock.last_quality_score = 0.8
    mock.force_completion = False
    mock.unified_task_type = Mock(value="analysis")
    mock.is_analysis_task = True
    mock.is_action_task = False
    mock.get_remaining_budget = Mock(return_value=10)
    return mock


@pytest.fixture
def recovery_context(mock_streaming_context):
    """Create RecoveryContext for testing."""
    return RecoveryContext(
        iteration=5,
        elapsed_time=10.0,
        tool_calls_used=3,
        tool_budget=15,
        max_iterations=10,
        session_start_time=1000.0,
        last_quality_score=0.8,
        streaming_context=mock_streaming_context,
        provider_name="anthropic",
        model="claude-3-opus",
        temperature=0.7,
        unified_task_type=Mock(value="analysis"),
        is_analysis_task=True,
        is_action_task=False,
    )


# =====================================================================
# Condition Checking Methods Tests
# =====================================================================


class TestConditionChecking:
    """Tests for condition checking methods."""

    def test_check_time_limit_no_limit_reached(
        self, recovery_coordinator, recovery_context, mock_streaming_handler
    ):
        """Test check_time_limit when no limit reached."""
        mock_streaming_handler.check_time_limit.return_value = None

        result = recovery_coordinator.check_time_limit(recovery_context)

        assert result is None
        mock_streaming_handler.check_time_limit.assert_called_once()

    def test_check_time_limit_limit_reached(
        self, recovery_coordinator, recovery_context, mock_streaming_handler
    ):
        """Test check_time_limit when limit reached."""
        mock_result = Mock()
        mock_result.chunks = [StreamChunk(content="time limit", is_final=False)]
        mock_streaming_handler.check_time_limit.return_value = mock_result

        result = recovery_coordinator.check_time_limit(recovery_context)

        assert result is not None
        assert result.content == "time limit"

    def test_check_iteration_limit_no_limit_reached(
        self, recovery_coordinator, recovery_context, mock_streaming_handler
    ):
        """Test check_iteration_limit when no limit reached."""
        mock_streaming_handler.check_iteration_limit.return_value = None

        result = recovery_coordinator.check_iteration_limit(recovery_context)

        assert result is None

    def test_check_iteration_limit_limit_reached(
        self, recovery_coordinator, recovery_context, mock_streaming_handler
    ):
        """Test check_iteration_limit when limit reached."""
        mock_result = Mock()
        mock_result.chunks = [StreamChunk(content="iteration limit", is_final=False)]
        mock_streaming_handler.check_iteration_limit.return_value = mock_result

        result = recovery_coordinator.check_iteration_limit(recovery_context)

        assert result is not None
        assert result.content == "iteration limit"

    def test_check_natural_completion_not_complete(
        self, recovery_coordinator, recovery_context, mock_streaming_handler
    ):
        """Test check_natural_completion when not complete."""
        mock_streaming_handler.check_natural_completion.return_value = False

        result = recovery_coordinator.check_natural_completion(
            recovery_context, has_tool_calls=False, content_length=100
        )

        assert result is None

    def test_check_natural_completion_is_complete(
        self, recovery_coordinator, recovery_context, mock_streaming_handler
    ):
        """Test check_natural_completion when complete."""
        mock_streaming_handler.check_natural_completion.return_value = True

        result = recovery_coordinator.check_natural_completion(
            recovery_context, has_tool_calls=False, content_length=100
        )

        assert result is not None
        assert result.is_final is True

    def test_check_tool_budget_not_exhausted(
        self, recovery_coordinator, recovery_context, mock_streaming_handler
    ):
        """Test check_tool_budget when budget not exhausted."""
        mock_streaming_handler.check_tool_budget.return_value = None

        result = recovery_coordinator.check_tool_budget(recovery_context, warning_threshold=250)

        assert result is None

    def test_check_tool_budget_approaching_limit(
        self, recovery_coordinator, recovery_context, mock_streaming_handler
    ):
        """Test check_tool_budget when approaching limit."""
        mock_result = Mock()
        mock_result.chunks = [StreamChunk(content="budget warning", is_final=False)]
        mock_streaming_handler.check_tool_budget.return_value = mock_result

        result = recovery_coordinator.check_tool_budget(recovery_context, warning_threshold=250)

        assert result is not None
        assert result.content == "budget warning"

    def test_check_progress_making_progress(
        self, recovery_coordinator, recovery_context, mock_streaming_handler
    ):
        """Test check_progress when making progress."""
        mock_streaming_handler.check_progress_and_force.return_value = False

        result = recovery_coordinator.check_progress(recovery_context, base_max=8)

        assert result is False

    def test_check_progress_stuck(
        self, recovery_coordinator, recovery_context, mock_streaming_handler
    ):
        """Test check_progress when stuck."""
        mock_streaming_handler.check_progress_and_force.return_value = True

        result = recovery_coordinator.check_progress(recovery_context, base_max=8)

        assert result is True

    def test_check_blocked_threshold_not_exceeded(
        self, recovery_coordinator, recovery_context, mock_streaming_handler
    ):
        """Test check_blocked_threshold when threshold not exceeded."""
        mock_streaming_handler.check_blocked_threshold.return_value = None

        result = recovery_coordinator.check_blocked_threshold(
            recovery_context, all_blocked=False, consecutive_limit=4, total_limit=6
        )

        assert result is None

    def test_check_blocked_threshold_exceeded(
        self, recovery_coordinator, recovery_context, mock_streaming_handler
    ):
        """Test check_blocked_threshold when threshold exceeded."""
        mock_result = Mock()
        mock_result.chunks = [StreamChunk(content="blocked threshold", is_final=False)]
        mock_result.clear_tool_calls = True
        mock_streaming_handler.check_blocked_threshold.return_value = mock_result

        result = recovery_coordinator.check_blocked_threshold(
            recovery_context, all_blocked=True, consecutive_limit=4, total_limit=6
        )

        assert result is not None
        chunk, clear_tools = result
        assert chunk.content == "blocked threshold"
        assert clear_tools is True

    def test_check_force_action_no_recovery_handler(self, recovery_context):
        """Test check_force_action when no recovery handler."""
        coordinator = RecoveryCoordinator(
            recovery_handler=None,
            recovery_integration=None,
            streaming_handler=Mock(),
            context_compactor=None,
            unified_tracker=Mock(),
            settings=Mock(),
        )

        should_force, action_type = coordinator.check_force_action(recovery_context)

        assert should_force is False
        assert action_type is None


# =====================================================================
# Action Handling Methods Tests
# =====================================================================


class TestActionHandling:
    """Tests for action handling methods."""

    def test_handle_empty_response_below_threshold(
        self, recovery_coordinator, recovery_context, mock_streaming_handler
    ):
        """Test handle_empty_response below threshold."""
        mock_streaming_handler.handle_empty_response.return_value = None

        chunk, should_force = recovery_coordinator.handle_empty_response(recovery_context)

        assert chunk is None
        assert should_force is False

    def test_handle_empty_response_threshold_exceeded(
        self, recovery_coordinator, recovery_context, mock_streaming_handler, mock_streaming_context
    ):
        """Test handle_empty_response when threshold exceeded."""
        mock_result = Mock()
        mock_result.chunks = [StreamChunk(content="empty threshold", is_final=False)]
        mock_streaming_handler.handle_empty_response.return_value = mock_result
        mock_streaming_context.force_completion = True

        chunk, should_force = recovery_coordinator.handle_empty_response(recovery_context)

        assert chunk is not None
        assert chunk.content == "empty threshold"
        assert should_force is True

    def test_handle_blocked_tool(
        self, recovery_coordinator, recovery_context, mock_streaming_handler
    ):
        """Test handle_blocked_tool."""
        mock_streaming_handler.handle_blocked_tool_call.return_value = StreamChunk(
            content="Tool blocked", is_final=False
        )

        result = recovery_coordinator.handle_blocked_tool(
            recovery_context,
            tool_name="dangerous_tool",
            tool_args={"arg": "value"},
            block_reason="Security",
        )

        assert result is not None
        assert result.content == "Tool blocked"
        mock_streaming_handler.handle_blocked_tool_call.assert_called_once()

    def test_handle_force_tool_execution_no_recovery_handler(self, recovery_context):
        """Test handle_force_tool_execution with no recovery handler."""
        coordinator = RecoveryCoordinator(
            recovery_handler=None,
            recovery_integration=None,
            streaming_handler=Mock(),
            context_compactor=None,
            unified_tracker=Mock(),
            settings=Mock(),
        )

        should_execute, chunks = coordinator.handle_force_tool_execution(recovery_context)

        assert should_execute is False
        assert chunks is None

    def test_handle_force_completion_not_forced(
        self, recovery_coordinator, recovery_context
    ):
        """Test handle_force_completion when not forced."""
        result = recovery_coordinator.handle_force_completion(recovery_context)

        assert result is None

    def test_handle_force_completion_forced(
        self, recovery_coordinator, recovery_context, mock_streaming_context
    ):
        """Test handle_force_completion when forced."""
        mock_streaming_context.force_completion = True

        result = recovery_coordinator.handle_force_completion(recovery_context)

        assert result is not None
        assert len(result) > 0
        assert "summary" in result[0].content.lower()

    def test_handle_loop_warning_no_loop(
        self, recovery_coordinator, recovery_context, mock_unified_tracker
    ):
        """Test handle_loop_warning when no loop detected."""
        mock_unified_tracker.should_stop.return_value = Mock(should_stop=False)

        result = recovery_coordinator.handle_loop_warning(recovery_context)

        assert result is None

    def test_handle_loop_warning_loop_detected(
        self, recovery_coordinator, recovery_context, mock_unified_tracker
    ):
        """Test handle_loop_warning when loop detected."""
        mock_unified_tracker.should_stop.return_value = Mock(
            should_stop=True, reason="Loop detected"
        )

        result = recovery_coordinator.handle_loop_warning(recovery_context)

        assert result is not None
        assert len(result) > 0
        assert "loop" in result[0].content.lower()

    @pytest.mark.asyncio
    async def test_handle_recovery_with_integration_disabled(
        self, recovery_context
    ):
        """Test handle_recovery_with_integration when disabled."""
        from victor.agent.orchestrator_recovery import OrchestratorRecoveryAction

        mock_integration = Mock()
        mock_integration.enabled = False

        coordinator = RecoveryCoordinator(
            recovery_handler=None,
            recovery_integration=mock_integration,
            streaming_handler=Mock(),
            context_compactor=None,
            unified_tracker=Mock(),
            settings=Mock(),
        )

        result = await coordinator.handle_recovery_with_integration(
            recovery_context, "content", None, None
        )

        assert result.action == "continue"
        assert result.reason == "Recovery disabled"

    @pytest.mark.asyncio
    async def test_handle_recovery_with_integration_enabled(
        self, recovery_coordinator, recovery_context, mock_recovery_integration
    ):
        """Test handle_recovery_with_integration when enabled."""
        from victor.agent.orchestrator_recovery import OrchestratorRecoveryAction

        mock_action = OrchestratorRecoveryAction(action="retry", reason="Test")
        mock_recovery_integration.handle_response.return_value = mock_action

        result = await recovery_coordinator.handle_recovery_with_integration(
            recovery_context, "test content", [], None
        )

        assert result.action == "retry"
        mock_recovery_integration.handle_response.assert_called_once()

    def test_apply_recovery_action_continue(self, recovery_coordinator, recovery_context):
        """Test apply_recovery_action with continue action."""
        from victor.agent.orchestrator_recovery import OrchestratorRecoveryAction

        action = OrchestratorRecoveryAction(action="continue", reason="OK")

        result = recovery_coordinator.apply_recovery_action(action, recovery_context)

        assert result is None

    def test_apply_recovery_action_retry(self, recovery_coordinator, recovery_context):
        """Test apply_recovery_action with retry action."""
        from victor.agent.orchestrator_recovery import OrchestratorRecoveryAction

        action = OrchestratorRecoveryAction(
            action="retry", reason="Retry needed", message="Please retry"
        )
        message_adder = Mock()

        result = recovery_coordinator.apply_recovery_action(
            action, recovery_context, message_adder=message_adder
        )

        assert result is None
        message_adder.assert_called_once_with("user", "Please retry")

    def test_apply_recovery_action_abort(self, recovery_coordinator, recovery_context):
        """Test apply_recovery_action with abort action."""
        from victor.agent.orchestrator_recovery import OrchestratorRecoveryAction

        action = OrchestratorRecoveryAction(action="abort", reason="Fatal error")

        result = recovery_coordinator.apply_recovery_action(action, recovery_context)

        assert result is not None
        assert "abort" in result.content.lower()
        assert result.is_final is True


# =====================================================================
# Filtering and Truncation Methods Tests
# =====================================================================


class TestFilteringAndTruncation:
    """Tests for filtering and truncation methods."""

    def test_filter_blocked_tool_calls_none_blocked(
        self, recovery_coordinator, recovery_context, mock_streaming_handler
    ):
        """Test filter_blocked_tool_calls when none blocked."""
        tool_calls = [{"name": "tool1", "arguments": {}}, {"name": "tool2", "arguments": {}}]
        mock_streaming_handler.filter_blocked_tool_calls.return_value = (tool_calls, [], 0)

        filtered, chunks, count = recovery_coordinator.filter_blocked_tool_calls(
            recovery_context, tool_calls
        )

        assert len(filtered) == 2
        assert len(chunks) == 0
        assert count == 0

    def test_filter_blocked_tool_calls_some_blocked(
        self, recovery_coordinator, recovery_context, mock_streaming_handler
    ):
        """Test filter_blocked_tool_calls when some blocked."""
        tool_calls = [{"name": "tool1", "arguments": {}}, {"name": "blocked_tool", "arguments": {}}]
        mock_streaming_handler.filter_blocked_tool_calls.return_value = (
            [tool_calls[0]],
            [StreamChunk(content="blocked", is_final=False)],
            1,
        )

        filtered, chunks, count = recovery_coordinator.filter_blocked_tool_calls(
            recovery_context, tool_calls
        )

        assert len(filtered) == 1
        assert len(chunks) == 1
        assert count == 1

    def test_truncate_tool_calls_within_budget(
        self, recovery_coordinator, recovery_context
    ):
        """Test truncate_tool_calls when within budget."""
        tool_calls = [{"name": "tool1"}, {"name": "tool2"}]

        truncated, was_truncated = recovery_coordinator.truncate_tool_calls(
            recovery_context, tool_calls, max_calls=5
        )

        assert len(truncated) == 2
        assert was_truncated is False

    def test_truncate_tool_calls_exceeds_budget(
        self, recovery_coordinator, recovery_context
    ):
        """Test truncate_tool_calls when exceeds budget."""
        tool_calls = [{"name": f"tool{i}"} for i in range(10)]

        truncated, was_truncated = recovery_coordinator.truncate_tool_calls(
            recovery_context, tool_calls, max_calls=3
        )

        assert len(truncated) == 3
        assert was_truncated is True


# =====================================================================
# Prompt and Message Generation Tests
# =====================================================================


class TestPromptAndMessageGeneration:
    """Tests for prompt and message generation methods."""

    def test_get_recovery_prompts_no_handler(self, recovery_context):
        """Test get_recovery_prompts with no recovery handler."""
        coordinator = RecoveryCoordinator(
            recovery_handler=None,
            recovery_integration=None,
            streaming_handler=Mock(),
            context_compactor=None,
            unified_tracker=Mock(),
            settings=Mock(),
        )

        result = coordinator.get_recovery_prompts(recovery_context)

        assert result == []

    def test_get_recovery_fallback_message(
        self, recovery_coordinator, recovery_context, mock_streaming_handler
    ):
        """Test get_recovery_fallback_message."""
        mock_streaming_handler.get_recovery_fallback_message.return_value = "Fallback message"

        result = recovery_coordinator.get_recovery_fallback_message(recovery_context)

        assert result == "Fallback message"

    def test_should_use_tools_for_recovery_budget_exhausted(
        self, recovery_coordinator, recovery_context
    ):
        """Test should_use_tools_for_recovery when budget exhausted."""
        recovery_context.tool_calls_used = 15
        recovery_context.tool_budget = 15

        result = recovery_coordinator.should_use_tools_for_recovery(recovery_context)

        assert result is False

    def test_should_use_tools_for_recovery_budget_available(
        self, recovery_coordinator, recovery_context
    ):
        """Test should_use_tools_for_recovery when budget available."""
        recovery_context.tool_calls_used = 5
        recovery_context.tool_budget = 15

        result = recovery_coordinator.should_use_tools_for_recovery(recovery_context)

        assert result is True


# =====================================================================
# Metrics and Formatting Tests
# =====================================================================


class TestMetricsAndFormatting:
    """Tests for metrics and formatting methods."""

    def test_format_completion_metrics(
        self, recovery_coordinator, recovery_context, mock_streaming_handler
    ):
        """Test format_completion_metrics."""
        mock_streaming_handler.format_completion_metrics.return_value = "Metrics: 5 iterations"

        result = recovery_coordinator.format_completion_metrics(recovery_context, elapsed_time=10.5)

        assert result == "Metrics: 5 iterations"
        mock_streaming_handler.format_completion_metrics.assert_called_once()

    def test_format_budget_exhausted_metrics(
        self, recovery_coordinator, recovery_context, mock_streaming_handler
    ):
        """Test format_budget_exhausted_metrics."""
        mock_streaming_handler.format_budget_exhausted_metrics.return_value = "Budget exhausted"

        result = recovery_coordinator.format_budget_exhausted_metrics(
            recovery_context, elapsed_time=10.5, time_to_first_token=0.5
        )

        assert result == "Budget exhausted"
        mock_streaming_handler.format_budget_exhausted_metrics.assert_called_once()

    def test_generate_tool_result_chunks(
        self, recovery_coordinator, mock_streaming_handler
    ):
        """Test generate_tool_result_chunks."""
        result_dict = {"status": "success", "output": "result"}
        mock_chunks = [StreamChunk(content="result", is_final=False)]
        mock_streaming_handler.generate_tool_result_chunks.return_value = mock_chunks

        result = recovery_coordinator.generate_tool_result_chunks(result_dict)

        assert result == mock_chunks
        mock_streaming_handler.generate_tool_result_chunks.assert_called_once_with(result_dict)


# =====================================================================
# Edge Cases and Error Handling Tests
# =====================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_recovery_coordinator_with_none_dependencies(self):
        """Test RecoveryCoordinator with None optional dependencies."""
        mock_streaming_handler = Mock()
        mock_unified_tracker = Mock()
        mock_settings = Mock()

        coordinator = RecoveryCoordinator(
            recovery_handler=None,
            recovery_integration=None,
            streaming_handler=mock_streaming_handler,
            context_compactor=None,
            unified_tracker=mock_unified_tracker,
            settings=mock_settings,
        )

        assert coordinator.recovery_handler is None
        assert coordinator.recovery_integration is None
        assert coordinator.context_compactor is None
        assert coordinator.streaming_handler is mock_streaming_handler
        assert coordinator.unified_tracker is mock_unified_tracker
        assert coordinator.settings is mock_settings

    def test_recovery_context_fields(self, recovery_context):
        """Test RecoveryContext has all required fields."""
        assert recovery_context.iteration == 5
        assert recovery_context.elapsed_time == 10.0
        assert recovery_context.tool_calls_used == 3
        assert recovery_context.tool_budget == 15
        assert recovery_context.max_iterations == 10
        assert recovery_context.session_start_time == 1000.0
        assert recovery_context.last_quality_score == 0.8
        assert recovery_context.provider_name == "anthropic"
        assert recovery_context.model == "claude-3-opus"
        assert recovery_context.temperature == 0.7
        assert recovery_context.is_analysis_task is True
        assert recovery_context.is_action_task is False

    def test_empty_tool_calls_list(self, recovery_coordinator, recovery_context):
        """Test handling of empty tool calls list."""
        truncated, was_truncated = recovery_coordinator.truncate_tool_calls(
            recovery_context, [], max_calls=5
        )

        assert truncated == []
        assert was_truncated is False
