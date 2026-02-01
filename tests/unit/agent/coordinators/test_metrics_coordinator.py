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

"""Tests for MetricsCoordinator.

This test file provides comprehensive coverage for MetricsCoordinator which handles:
- Stream metrics collection and reporting
- Tool usage statistics
- Session cost tracking
- Token usage for evaluation/benchmarking
- Streaming state management

Test Pattern:
1. Mock coordinator dependencies (MetricsCollector, SessionCostTracker)
2. Test coordinator methods in isolation
3. Verify delegation to dependencies
4. Test state management (streaming, cancellation)
"""

import asyncio
import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from dataclasses import dataclass

from victor.agent.coordinators.metrics_coordinator import (
    MetricsCoordinator,
    StreamingState,
    create_metrics_coordinator,
)


# Test dataclasses
@dataclass
class MockStreamMetrics:
    """Mock StreamMetrics for testing."""

    prompt_tokens: int = 100
    completion_tokens: int = 50
    total_tokens: int = 150
    cache_read_tokens: int = 10
    cache_write_tokens: int = 5
    total_duration: float = 1.5
    tool_calls_count: int = 2


class TestStreamingState:
    """Test suite for StreamingState class."""

    def test_initial_state(self):
        """Test that StreamingState initializes with default values."""
        # Execute
        state = StreamingState()

        # Assert
        assert state.is_streaming is False
        assert state.cancel_event is None
        assert state.session_start_time is None

    def test_reset_clears_all_state(self):
        """Test that reset() clears all streaming state."""
        # Setup
        state = StreamingState()
        state.is_streaming = True
        state.cancel_event = asyncio.Event()
        state.session_start_time = 123.45

        # Execute
        state.reset()

        # Assert
        assert state.is_streaming is False
        assert state.cancel_event is None
        assert state.session_start_time is None


class TestMetricsCoordinator:
    """Test suite for MetricsCoordinator core functionality."""

    @pytest.fixture
    def mock_metrics_collector(self) -> Mock:
        """Create mock metrics collector."""
        collector = Mock()
        collector.finalize_stream_metrics = Mock(return_value=None)
        collector.get_last_stream_metrics = Mock(return_value=None)
        collector.get_streaming_metrics_summary = Mock(return_value=None)
        collector.get_streaming_metrics_history = Mock(return_value=[])
        collector.get_tool_usage_stats = Mock(return_value={})
        collector.record_tool_selection = Mock()
        collector.record_tool_execution = Mock()
        collector.on_tool_start = Mock()
        collector.on_tool_complete = Mock()
        collector.on_streaming_session_complete = Mock()
        collector.record_first_token = Mock()
        collector.init_stream_metrics = Mock()
        collector.update_model_info = Mock()
        collector.reset_stats = Mock()
        return collector

    @pytest.fixture
    def mock_session_cost_tracker(self) -> Mock:
        """Create mock session cost tracker."""
        tracker = Mock()
        tracker.record_request = Mock()
        tracker.get_summary = Mock(return_value={})
        tracker.format_inline_cost = Mock(return_value="cost n/a")
        tracker.export_json = Mock()
        tracker.export_csv = Mock()
        return tracker

    @pytest.fixture
    def cumulative_token_usage(self) -> dict[str, int]:
        """Create cumulative token usage dict."""
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }

    @pytest.fixture
    def coordinator(
        self,
        mock_metrics_collector: Mock,
        mock_session_cost_tracker: Mock,
        cumulative_token_usage: dict[str, int],
    ) -> MetricsCoordinator:
        """Create metrics coordinator with default mocks."""
        return MetricsCoordinator(
            metrics_collector=mock_metrics_collector,
            session_cost_tracker=mock_session_cost_tracker,
            cumulative_token_usage=cumulative_token_usage,
        )

    # Test initialization

    def test_initialization(self, coordinator: MetricsCoordinator):
        """Test that coordinator initializes correctly."""
        # Assert
        assert coordinator.metrics_collector is not None
        assert coordinator.session_cost_tracker is not None
        assert coordinator.is_streaming() is False
        assert coordinator.cancel_event is None

    def test_factory_function_creates_coordinator(self):
        """Test create_metrics_coordinator factory function."""
        # Execute
        coordinator = create_metrics_coordinator(
            metrics_collector=Mock(),
            session_cost_tracker=Mock(),
        )

        # Assert
        assert isinstance(coordinator, MetricsCoordinator)
        assert coordinator._cumulative_token_usage is not None
        assert "prompt_tokens" in coordinator._cumulative_token_usage
        assert "completion_tokens" in coordinator._cumulative_token_usage

    def test_factory_function_with_custom_token_usage(self):
        """Test factory function with custom token usage dict."""
        # Setup
        custom_tokens = {"prompt_tokens": 100, "completion_tokens": 50}

        # Execute
        coordinator = create_metrics_coordinator(
            metrics_collector=Mock(),
            session_cost_tracker=Mock(),
            cumulative_token_usage=custom_tokens,
        )

        # Assert
        assert coordinator._cumulative_token_usage == custom_tokens


class TestStreamMetrics:
    """Test suite for stream metrics functionality."""

    @pytest.fixture
    def coordinator(self) -> MetricsCoordinator:
        """Create coordinator with mocked dependencies."""
        return MetricsCoordinator(
            metrics_collector=Mock(),
            session_cost_tracker=Mock(),
            cumulative_token_usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
        )

    def test_finalize_stream_metrics_with_valid_metrics(self, coordinator: MetricsCoordinator):
        """Test finalize_stream_metrics records to cost tracker."""
        # Setup
        mock_metrics = MockStreamMetrics()
        coordinator._metrics_collector.finalize_stream_metrics = Mock(return_value=mock_metrics)

        # Execute
        result = coordinator.finalize_stream_metrics()

        # Assert
        assert result == mock_metrics
        coordinator._session_cost_tracker.record_request.assert_called_once_with(
            prompt_tokens=100,
            completion_tokens=50,
            cache_read_tokens=10,
            cache_write_tokens=5,
            duration_seconds=1.5,
            tool_calls=2,
        )

    def test_finalize_stream_metrics_with_usage_data(self, coordinator: MetricsCoordinator):
        """Test finalize_stream_metrics with usage data parameter."""
        # Setup
        mock_metrics = MockStreamMetrics()
        coordinator._metrics_collector.finalize_stream_metrics = Mock(return_value=mock_metrics)
        usage_data = {"prompt_tokens": 200, "completion_tokens": 100}

        # Execute
        result = coordinator.finalize_stream_metrics(usage_data)

        # Assert
        assert result == mock_metrics
        coordinator._metrics_collector.finalize_stream_metrics.assert_called_once_with(usage_data)

    def test_finalize_stream_metrics_with_none_result(self, coordinator: MetricsCoordinator):
        """Test finalize_stream_metrics when collector returns None."""
        # Setup
        coordinator._metrics_collector.finalize_stream_metrics = Mock(return_value=None)

        # Execute
        result = coordinator.finalize_stream_metrics()

        # Assert
        assert result is None
        coordinator._session_cost_tracker.record_request.assert_not_called()

    def test_get_last_stream_metrics(self, coordinator: MetricsCoordinator):
        """Test get_last_stream_metrics delegates to collector."""
        # Setup
        mock_metrics = MockStreamMetrics()
        coordinator._metrics_collector.get_last_stream_metrics = Mock(return_value=mock_metrics)

        # Execute
        result = coordinator.get_last_stream_metrics()

        # Assert
        assert result == mock_metrics
        coordinator._metrics_collector.get_last_stream_metrics.assert_called_once()

    def test_get_streaming_metrics_summary(self, coordinator: MetricsCoordinator):
        """Test get_streaming_metrics_summary delegates to collector."""
        # Setup
        summary = {"total_duration": 10.5, "tokens_per_second": 15.3}
        coordinator._metrics_collector.get_streaming_metrics_summary = Mock(return_value=summary)

        # Execute
        result = coordinator.get_streaming_metrics_summary()

        # Assert
        assert result == summary
        coordinator._metrics_collector.get_streaming_metrics_summary.assert_called_once()

    def test_get_streaming_metrics_history(self, coordinator: MetricsCoordinator):
        """Test get_streaming_metrics_history with default limit."""
        # Setup
        history = [
            {"request_id": "1", "duration": 1.0},
            {"request_id": "2", "duration": 1.5},
        ]
        coordinator._metrics_collector.get_streaming_metrics_history = Mock(return_value=history)

        # Execute
        result = coordinator.get_streaming_metrics_history()

        # Assert
        assert result == history
        coordinator._metrics_collector.get_streaming_metrics_history.assert_called_once_with(10)

    def test_get_streaming_metrics_history_with_custom_limit(self, coordinator: MetricsCoordinator):
        """Test get_streaming_metrics_history with custom limit."""
        # Setup
        history = [{"request_id": "1", "duration": 1.0}]
        coordinator._metrics_collector.get_streaming_metrics_history = Mock(return_value=history)

        # Execute
        result = coordinator.get_streaming_metrics_history(limit=5)

        # Assert
        assert result == history
        coordinator._metrics_collector.get_streaming_metrics_history.assert_called_once_with(5)


class TestCostTracking:
    """Test suite for cost tracking functionality."""

    @pytest.fixture
    def coordinator(self) -> MetricsCoordinator:
        """Create coordinator with mocked dependencies."""
        return MetricsCoordinator(
            metrics_collector=Mock(),
            session_cost_tracker=Mock(),
            cumulative_token_usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
        )

    def test_get_session_cost_summary(self, coordinator: MetricsCoordinator):
        """Test get_session_cost_summary delegates to tracker."""
        # Setup
        summary = {"total_cost": 0.0123, "total_tokens": 1500}
        coordinator._session_cost_tracker.get_summary = Mock(return_value=summary)

        # Execute
        result = coordinator.get_session_cost_summary()

        # Assert
        assert result == summary
        coordinator._session_cost_tracker.get_summary.assert_called_once()

    def test_get_session_cost_formatted(self, coordinator: MetricsCoordinator):
        """Test get_session_cost_formatted returns formatted string."""
        # Setup
        coordinator._session_cost_tracker.format_inline_cost = Mock(return_value="$0.0123")

        # Execute
        result = coordinator.get_session_cost_formatted()

        # Assert
        assert result == "$0.0123"
        coordinator._session_cost_tracker.format_inline_cost.assert_called_once()

    def test_export_session_costs_json(self, coordinator: MetricsCoordinator, tmp_path):
        """Test export_session_costs with JSON format."""
        # Setup
        output_path = tmp_path / "costs.json"

        # Execute
        coordinator.export_session_costs(str(output_path), format="json")

        # Assert
        coordinator._session_cost_tracker.export_json.assert_called_once()
        args = coordinator._session_cost_tracker.export_json.call_args[0]
        assert isinstance(args[0], Path)
        assert str(args[0]) == str(output_path)

    def test_export_session_costs_csv(self, coordinator: MetricsCoordinator, tmp_path):
        """Test export_session_costs with CSV format."""
        # Setup
        output_path = tmp_path / "costs.csv"

        # Execute
        coordinator.export_session_costs(str(output_path), format="csv")

        # Assert
        coordinator._session_cost_tracker.export_csv.assert_called_once()
        args = coordinator._session_cost_tracker.export_csv.call_args[0]
        assert isinstance(args[0], Path)

    def test_export_session_costs_default_format(self, coordinator: MetricsCoordinator, tmp_path):
        """Test export_session_costs defaults to JSON format."""
        # Setup
        output_path = tmp_path / "costs.txt"

        # Execute
        coordinator.export_session_costs(str(output_path))

        # Assert
        coordinator._session_cost_tracker.export_json.assert_called_once()


class TestTokenUsage:
    """Test suite for token usage tracking functionality."""

    @pytest.fixture
    def coordinator(self) -> MetricsCoordinator:
        """Create coordinator with mocked dependencies."""
        return MetricsCoordinator(
            metrics_collector=Mock(),
            session_cost_tracker=Mock(),
            cumulative_token_usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cache_read_tokens": 10,
                "cache_write_tokens": 5,
            },
        )

    def test_get_token_usage(self, coordinator: MetricsCoordinator):
        """Test get_token_usage returns TokenUsage dataclass."""
        # Execute
        result = coordinator.get_token_usage()

        # Assert
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150

    def test_get_token_usage_with_zero_tokens(self):
        """Test get_token_usage when no tokens have been used."""
        # Setup
        coordinator = MetricsCoordinator(
            metrics_collector=Mock(),
            session_cost_tracker=Mock(),
            cumulative_token_usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
        )

        # Execute
        result = coordinator.get_token_usage()

        # Assert
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.total_tokens == 0

    def test_get_token_usage_with_large_counts(self):
        """Test get_token_usage with large token counts."""
        # Setup
        coordinator = MetricsCoordinator(
            metrics_collector=Mock(),
            session_cost_tracker=Mock(),
            cumulative_token_usage={
                "prompt_tokens": 1000000,
                "completion_tokens": 500000,
                "total_tokens": 1500000,
                "cache_read_tokens": 100000,
                "cache_write_tokens": 50000,
            },
        )

        # Execute
        result = coordinator.get_token_usage()

        # Assert
        assert result.input_tokens == 1000000
        assert result.output_tokens == 500000
        assert result.total_tokens == 1500000

    def test_reset_token_usage(self, coordinator: MetricsCoordinator):
        """Test reset_token_usage clears all token counts."""
        # Execute
        coordinator.reset_token_usage()

        # Assert
        assert coordinator._cumulative_token_usage["prompt_tokens"] == 0
        assert coordinator._cumulative_token_usage["completion_tokens"] == 0
        assert coordinator._cumulative_token_usage["total_tokens"] == 0
        assert coordinator._cumulative_token_usage["cache_read_tokens"] == 0
        assert coordinator._cumulative_token_usage["cache_write_tokens"] == 0

    def test_update_cumulative_token_usage(self, coordinator: MetricsCoordinator):
        """Test update_cumulative_token_usage adds new tokens."""
        # Setup
        initial_prompt = coordinator._cumulative_token_usage["prompt_tokens"]
        usage_data = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }

        # Execute
        coordinator.update_cumulative_token_usage(usage_data)

        # Assert
        assert coordinator._cumulative_token_usage["prompt_tokens"] == initial_prompt + 100
        assert coordinator._cumulative_token_usage["completion_tokens"] == 100
        assert coordinator._cumulative_token_usage["total_tokens"] == 300

    def test_update_cumulative_token_usage_partial_data(self):
        """Test update_cumulative_token_usage with partial data."""
        # Setup
        coordinator = MetricsCoordinator(
            metrics_collector=Mock(),
            session_cost_tracker=Mock(),
            cumulative_token_usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cache_read_tokens": 10,
                "cache_write_tokens": 5,
            },
        )
        usage_data = {"prompt_tokens": 25}

        # Execute
        coordinator.update_cumulative_token_usage(usage_data)

        # Assert - only prompt_tokens should be updated
        assert coordinator._cumulative_token_usage["prompt_tokens"] == 125
        assert coordinator._cumulative_token_usage["completion_tokens"] == 50

    def test_update_cumulative_token_usage_with_cache_tokens(self, coordinator: MetricsCoordinator):
        """Test update_cumulative_token_usage with cache tokens."""
        # Setup
        usage_data = {
            "cache_read_tokens": 100,
            "cache_write_tokens": 50,
        }

        # Execute
        coordinator.update_cumulative_token_usage(usage_data)

        # Assert
        assert coordinator._cumulative_token_usage["cache_read_tokens"] == 110
        assert coordinator._cumulative_token_usage["cache_write_tokens"] == 55

    def test_token_usage_tracking_flow(self):
        """Test complete token usage tracking workflow."""
        # Setup
        coordinator = MetricsCoordinator(
            metrics_collector=Mock(),
            session_cost_tracker=Mock(),
            cumulative_token_usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
        )

        # Initial state
        usage = coordinator.get_token_usage()
        assert usage.total_tokens == 0

        # Update tokens
        coordinator.update_cumulative_token_usage(
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        )

        # Check updated
        usage = coordinator.get_token_usage()
        assert usage.total_tokens == 150
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50

        # Reset
        coordinator.reset_token_usage()
        usage = coordinator.get_token_usage()
        assert usage.total_tokens == 0


class TestToolUsageStats:
    """Test suite for tool usage statistics functionality."""

    @pytest.fixture
    def coordinator(self) -> MetricsCoordinator:
        """Create coordinator with mocked dependencies."""
        return MetricsCoordinator(
            metrics_collector=Mock(),
            session_cost_tracker=Mock(),
            cumulative_token_usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
        )

    def test_get_tool_usage_stats(self, coordinator: MetricsCoordinator):
        """Test get_tool_usage_stats delegates to collector."""
        # Setup
        stats = {
            "semantic_selections": 10,
            "keyword_selections": 5,
            "tool_executions": {"read_file": 3},
        }
        coordinator._metrics_collector.get_tool_usage_stats = Mock(return_value=stats)

        # Execute
        result = coordinator.get_tool_usage_stats()

        # Assert
        assert result == stats
        coordinator._metrics_collector.get_tool_usage_stats.assert_called_once_with(
            conversation_state_summary=None
        )

    def test_get_tool_usage_stats_with_conversation_summary(self, coordinator: MetricsCoordinator):
        """Test get_tool_usage_stats with conversation state."""
        # Setup
        conversation_summary = {"total_messages": 10, "iterations": 3}
        stats = {"total_tools": 5}
        coordinator._metrics_collector.get_tool_usage_stats = Mock(return_value=stats)

        # Execute
        result = coordinator.get_tool_usage_stats(conversation_summary)

        # Assert
        assert result == stats
        coordinator._metrics_collector.get_tool_usage_stats.assert_called_once_with(
            conversation_state_summary=conversation_summary
        )


class TestMetricsCollectorDelegation:
    """Test suite for metrics collector delegation methods."""

    @pytest.fixture
    def coordinator(self) -> MetricsCoordinator:
        """Create coordinator with mocked dependencies."""
        return MetricsCoordinator(
            metrics_collector=Mock(),
            session_cost_tracker=Mock(),
            cumulative_token_usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
        )

    def test_record_tool_selection(self, coordinator: MetricsCoordinator):
        """Test record_tool_selection delegates to collector."""
        # Execute
        coordinator.record_tool_selection(method="semantic", num_tools=5)

        # Assert
        coordinator._metrics_collector.record_tool_selection.assert_called_once_with("semantic", 5)

    def test_record_tool_execution(self, coordinator: MetricsCoordinator):
        """Test record_tool_execution delegates to collector."""
        # Execute
        coordinator.record_tool_execution(tool_name="read_file", success=True, elapsed_ms=123.45)

        # Assert
        coordinator._metrics_collector.record_tool_execution.assert_called_once_with(
            "read_file", True, 123.45
        )

    def test_on_tool_start(self, coordinator: MetricsCoordinator):
        """Test on_tool_start delegates to collector."""
        # Setup
        arguments = {"path": "/test/file.txt"}

        # Execute
        coordinator.on_tool_start(tool_name="read_file", arguments=arguments, iteration=1)

        # Assert
        coordinator._metrics_collector.on_tool_start.assert_called_once_with(
            "read_file", arguments, 1
        )

    def test_on_tool_complete(self, coordinator: MetricsCoordinator):
        """Test on_tool_complete delegates to collector."""
        # Setup
        result = Mock(output="test output")

        # Execute
        coordinator.on_tool_complete(result)

        # Assert
        coordinator._metrics_collector.on_tool_complete.assert_called_once_with(result)

    def test_on_streaming_session_complete(self, coordinator: MetricsCoordinator):
        """Test on_streaming_session_complete delegates to collector."""
        # Setup
        session = Mock()

        # Execute
        coordinator.on_streaming_session_complete(session)

        # Assert
        coordinator._metrics_collector.on_streaming_session_complete.assert_called_once_with(
            session
        )

    def test_record_first_token(self, coordinator: MetricsCoordinator):
        """Test record_first_token delegates to collector."""
        # Execute
        coordinator.record_first_token()

        # Assert
        coordinator._metrics_collector.record_first_token.assert_called_once()

    def test_init_stream_metrics(self, coordinator: MetricsCoordinator):
        """Test init_stream_metrics delegates to collector."""
        # Setup
        mock_metrics = MockStreamMetrics()
        coordinator._metrics_collector.init_stream_metrics = Mock(return_value=mock_metrics)

        # Execute
        result = coordinator.init_stream_metrics()

        # Assert
        assert result == mock_metrics
        coordinator._metrics_collector.init_stream_metrics.assert_called_once()

    def test_update_model_info(self, coordinator: MetricsCoordinator):
        """Test update_model_info delegates to collector."""
        # Execute
        coordinator.update_model_info(model="claude-3-5-sonnet", provider="anthropic")

        # Assert
        coordinator._metrics_collector.update_model_info.assert_called_once_with(
            "claude-3-5-sonnet", "anthropic"
        )

    def test_reset_stats(self, coordinator: MetricsCoordinator):
        """Test reset_stats delegates to collector."""
        # Execute
        coordinator.reset_stats()

        # Assert
        coordinator._metrics_collector.reset_stats.assert_called_once()

    def test_metrics_collector_property(self, coordinator: MetricsCoordinator):
        """Test metrics_collector property returns collector."""
        # Execute
        result = coordinator.metrics_collector

        # Assert
        assert result == coordinator._metrics_collector

    def test_session_cost_tracker_property(self, coordinator: MetricsCoordinator):
        """Test session_cost_tracker property returns tracker."""
        # Execute
        result = coordinator.session_cost_tracker

        # Assert
        assert result == coordinator._session_cost_tracker


class TestStreamingStateManagement:
    """Test suite for streaming state management."""

    @pytest.fixture
    def coordinator(self) -> MetricsCoordinator:
        """Create coordinator with mocked dependencies."""
        return MetricsCoordinator(
            metrics_collector=Mock(),
            session_cost_tracker=Mock(),
            cumulative_token_usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
        )

    def test_start_streaming_initializes_state(self, coordinator: MetricsCoordinator):
        """Test start_streaming initializes streaming state."""
        # Execute
        coordinator.start_streaming()

        # Assert
        assert coordinator.is_streaming() is True
        assert coordinator.cancel_event is not None
        assert coordinator._streaming_state.session_start_time is not None

    def test_start_streaming_creates_new_cancel_event(self, coordinator: MetricsCoordinator):
        """Test that start_streaming creates a new cancellation event."""
        # Execute
        coordinator.start_streaming()
        first_event = coordinator.cancel_event

        # Start again
        coordinator.start_streaming()
        second_event = coordinator.cancel_event

        # Assert - should create new event
        assert second_event is not first_event
        assert coordinator.is_streaming() is True

    def test_stop_streaming_clears_state(self, coordinator: MetricsCoordinator):
        """Test stop_streaming clears streaming state."""
        # Setup
        coordinator.start_streaming()

        # Execute
        coordinator.stop_streaming()

        # Assert
        assert coordinator.is_streaming() is False
        assert coordinator.cancel_event is None
        assert coordinator._streaming_state.session_start_time is None

    def test_is_streaming_returns_false_initially(self, coordinator: MetricsCoordinator):
        """Test is_streaming returns False before starting."""
        # Execute
        result = coordinator.is_streaming()

        # Assert
        assert result is False

    def test_is_streaming_returns_true_when_active(self, coordinator: MetricsCoordinator):
        """Test is_streaming returns True when streaming."""
        # Setup
        coordinator.start_streaming()

        # Execute
        result = coordinator.is_streaming()

        # Assert
        assert result is True

    def test_is_streaming_returns_false_after_stopping(self, coordinator: MetricsCoordinator):
        """Test is_streaming returns False after stopping."""
        # Setup
        coordinator.start_streaming()
        coordinator.stop_streaming()

        # Execute
        result = coordinator.is_streaming()

        # Assert
        assert result is False

    def test_request_cancellation_when_streaming(self, coordinator: MetricsCoordinator):
        """Test request_cancellation sets cancel event when streaming."""
        # Setup
        coordinator.start_streaming()

        # Execute
        coordinator.request_cancellation()

        # Assert
        assert coordinator.is_cancellation_requested() is True

    def test_request_cancellation_when_not_streaming(self, coordinator: MetricsCoordinator):
        """Test request_cancellation is safe when not streaming."""
        # Execute - should not raise
        coordinator.request_cancellation()

        # Assert
        assert coordinator.is_cancellation_requested() is False

    def test_request_cancellation_with_no_event(self):
        """Test request_cancellation when cancel_event is None."""
        # Setup - create coordinator and don't start streaming
        coordinator = MetricsCoordinator(
            metrics_collector=Mock(),
            session_cost_tracker=Mock(),
            cumulative_token_usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
        )

        # Execute - should not raise
        coordinator.request_cancellation()

        # Assert
        assert coordinator.is_cancellation_requested() is False

    def test_is_cancellation_requested_returns_false_initially(
        self, coordinator: MetricsCoordinator
    ):
        """Test is_cancellation_requested returns False initially."""
        # Execute
        result = coordinator.is_cancellation_requested()

        # Assert
        assert result is False

    def test_is_cancellation_requested_returns_true_after_request(
        self, coordinator: MetricsCoordinator
    ):
        """Test is_cancellation_requested returns True after request."""
        # Setup
        coordinator.start_streaming()
        coordinator.request_cancellation()

        # Execute
        result = coordinator.is_cancellation_requested()

        # Assert
        assert result is True

    def test_is_cancellation_requested_returns_false_after_stopping(
        self, coordinator: MetricsCoordinator
    ):
        """Test is_cancellation_requested returns False after stopping."""
        # Setup
        coordinator.start_streaming()
        coordinator.request_cancellation()
        coordinator.stop_streaming()

        # Execute
        result = coordinator.is_cancellation_requested()

        # Assert - no event to check after stopping
        assert result is False

    def test_get_session_elapsed_time_returns_zero_when_not_streaming(
        self, coordinator: MetricsCoordinator
    ):
        """Test get_session_elapsed_time returns 0 when not streaming."""
        # Execute
        elapsed = coordinator.get_session_elapsed_time()

        # Assert
        assert elapsed == 0.0

    @patch("time.time", return_value=100.0)
    def test_get_session_elapsed_time_returns_elapsed_seconds(
        self, mock_time, coordinator: MetricsCoordinator
    ):
        """Test get_session_elapsed_time returns correct elapsed time."""
        # Setup - start streaming at time 100
        coordinator.start_streaming()

        # Move time forward
        mock_time.return_value = 105.5

        # Execute
        elapsed = coordinator.get_session_elapsed_time()

        # Assert
        assert elapsed == 5.5

    @patch("time.time", return_value=100.0)
    def test_get_session_elapsed_time_after_stopping(
        self, mock_time, coordinator: MetricsCoordinator
    ):
        """Test get_session_elapsed_time after stopping returns 0."""
        # Setup
        coordinator.start_streaming()
        coordinator.stop_streaming()

        # Execute
        elapsed = coordinator.get_session_elapsed_time()

        # Assert - session_start_time is cleared after stopping
        assert elapsed == 0.0

    def test_cancel_event_property_returns_none_initially(self, coordinator: MetricsCoordinator):
        """Test cancel_event property returns None initially."""
        # Execute
        result = coordinator.cancel_event

        # Assert
        assert result is None

    def test_cancel_event_property_returns_event_when_streaming(
        self, coordinator: MetricsCoordinator
    ):
        """Test cancel_event property returns event when streaming."""
        # Setup
        coordinator.start_streaming()

        # Execute
        result = coordinator.cancel_event

        # Assert
        assert isinstance(result, asyncio.Event)

    def test_cancel_event_property_returns_none_after_stopping(
        self, coordinator: MetricsCoordinator
    ):
        """Test cancel_event property returns None after stopping."""
        # Setup
        coordinator.start_streaming()
        coordinator.stop_streaming()

        # Execute
        result = coordinator.cancel_event

        # Assert
        assert result is None

    def test_complete_streaming_lifecycle(self, coordinator: MetricsCoordinator):
        """Test complete streaming lifecycle: start, check, cancel, stop."""
        # Initial state
        assert coordinator.is_streaming() is False
        assert coordinator.get_session_elapsed_time() == 0.0

        # Start streaming
        coordinator.start_streaming()
        assert coordinator.is_streaming() is True
        assert coordinator.cancel_event is not None

        # Check elapsed time
        elapsed = coordinator.get_session_elapsed_time()
        assert elapsed >= 0.0

        # Request cancellation
        coordinator.request_cancellation()
        assert coordinator.is_cancellation_requested() is True

        # Stop streaming
        coordinator.stop_streaming()
        assert coordinator.is_streaming() is False
        assert coordinator.cancel_event is None
        assert coordinator.get_session_elapsed_time() == 0.0


class TestMetricsCoordinatorIntegration:
    """Integration tests for MetricsCoordinator workflows."""

    @pytest.fixture
    def coordinator(self) -> MetricsCoordinator:
        """Create coordinator with mocked dependencies."""
        return MetricsCoordinator(
            metrics_collector=Mock(),
            session_cost_tracker=Mock(),
            cumulative_token_usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
        )

    def test_complete_streaming_session_workflow(self, coordinator: MetricsCoordinator):
        """Test complete workflow: start stream, record metrics, finalize."""
        # Start streaming
        coordinator.start_streaming()
        assert coordinator.is_streaming() is True

        # Record tool execution
        coordinator.record_tool_selection("semantic", 3)
        coordinator.record_tool_execution("read_file", True, 50.0)

        # Finalize metrics
        mock_metrics = MockStreamMetrics()
        coordinator._metrics_collector.finalize_stream_metrics = Mock(return_value=mock_metrics)

        result = coordinator.finalize_stream_metrics()

        # Verify
        assert result == mock_metrics
        coordinator._session_cost_tracker.record_request.assert_called_once()

    def test_token_usage_tracking_across_session(self):
        """Test token usage tracking throughout a session."""
        # Setup
        coordinator = MetricsCoordinator(
            metrics_collector=Mock(),
            session_cost_tracker=Mock(),
            cumulative_token_usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
        )

        # First request
        coordinator.update_cumulative_token_usage(
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        )
        usage1 = coordinator.get_token_usage()
        assert usage1.total_tokens == 150

        # Second request
        coordinator.update_cumulative_token_usage(
            {"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300}
        )
        usage2 = coordinator.get_token_usage()
        assert usage2.total_tokens == 450

        # Reset for new session
        coordinator.reset_token_usage()
        usage3 = coordinator.get_token_usage()
        assert usage3.total_tokens == 0

    def test_metrics_collection_with_cancellation(self, coordinator: MetricsCoordinator):
        """Test metrics collection when operation is cancelled."""
        # Start streaming
        coordinator.start_streaming()

        # Record some metrics
        coordinator.record_first_token()
        coordinator.record_tool_execution("test_tool", True, 100.0)

        # Cancel operation
        coordinator.request_cancellation()
        assert coordinator.is_cancellation_requested() is True

        # Stop streaming
        coordinator.stop_streaming()
        assert coordinator.is_streaming() is False

    def test_export_and_format_cost_workflow(self, coordinator: MetricsCoordinator, tmp_path):
        """Test workflow of formatting and exporting costs."""
        # Setup mock cost tracker
        coordinator._session_cost_tracker.get_summary = Mock(
            return_value={
                "total_cost": 0.0250,
                "total_tokens": 500,
                "requests": 2,
            }
        )
        coordinator._session_cost_tracker.format_inline_cost = Mock(return_value="$0.0250")

        # Get summary
        summary = coordinator.get_session_cost_summary()
        assert summary["total_cost"] == 0.0250

        # Get formatted
        formatted = coordinator.get_session_cost_formatted()
        assert formatted == "$0.0250"

        # Export
        output_path = tmp_path / "costs.json"
        coordinator.export_session_costs(str(output_path))
        coordinator._session_cost_tracker.export_json.assert_called_once()
