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

"""Tests for EvaluationCoordinator.

This test file demonstrates comprehensive testing for the EvaluationCoordinator,
which handles evaluation and analytics operations including:
- RL feedback signal collection
- Usage analytics tracking
- Intelligent outcome recording
- Analytics flushing and persistence

Test Strategy:
1. Test all public methods with various input combinations
2. Test error handling and edge cases
3. Test event publishing (event-driven architecture)
4. Test with all dependencies mocked
5. Test with missing/None dependencies (graceful degradation)
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from victor.agent.coordinators.evaluation_coordinator import EvaluationCoordinator


class TestEvaluationCoordinatorInit:
    """Test EvaluationCoordinator initialization and configuration."""

    def test_init_with_all_dependencies(self):
        """Test initialization with all dependencies provided."""
        # Setup
        usage_analytics = Mock()
        sequence_tracker = Mock()
        get_rl_coordinator_fn = Mock(return_value=Mock())
        get_vertical_context_fn = Mock(return_value=Mock())
        get_stream_context_fn = Mock(return_value=Mock())
        get_provider_fn = Mock(return_value=Mock())
        get_model_fn = Mock(return_value="gpt-4")
        get_tool_calls_used_fn = Mock(return_value=5)
        get_intelligent_integration_fn = Mock(return_value=Mock())

        # Execute
        coordinator = EvaluationCoordinator(
            usage_analytics=usage_analytics,
            sequence_tracker=sequence_tracker,
            get_rl_coordinator_fn=get_rl_coordinator_fn,
            get_vertical_context_fn=get_vertical_context_fn,
            get_stream_context_fn=get_stream_context_fn,
            get_provider_fn=get_provider_fn,
            get_model_fn=get_model_fn,
            get_tool_calls_used_fn=get_tool_calls_used_fn,
            get_intelligent_integration_fn=get_intelligent_integration_fn,
            enable_event_publishing=True,
        )

        # Assert
        assert coordinator._usage_analytics == usage_analytics
        assert coordinator._sequence_tracker == sequence_tracker
        assert coordinator._get_rl_coordinator_fn == get_rl_coordinator_fn
        assert coordinator._get_vertical_context_fn == get_vertical_context_fn
        assert coordinator._get_stream_context_fn == get_stream_context_fn
        assert coordinator._get_provider_fn == get_provider_fn
        assert coordinator._get_model_fn == get_model_fn
        assert coordinator._get_tool_calls_used_fn == get_tool_calls_used_fn
        assert coordinator._get_intelligent_integration_fn == get_intelligent_integration_fn
        assert coordinator._enable_event_publishing is True
        assert coordinator._event_bus is None

    def test_init_with_minimal_dependencies(self):
        """Test initialization with minimal dependencies (None values)."""
        # Execute
        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        # Assert
        assert coordinator._usage_analytics is None
        assert coordinator._sequence_tracker is None
        assert coordinator._enable_event_publishing is False

    def test_init_event_publishing_disabled(self):
        """Test initialization with event publishing disabled."""
        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        assert coordinator._enable_event_publishing is False
        assert coordinator._event_bus is None


class TestEvaluationCoordinatorEventBus:
    """Test event bus integration and lazy loading."""

    def test_get_event_bus_when_disabled(self):
        """Test _get_event_bus returns None when publishing disabled."""
        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        bus = coordinator._get_event_bus()
        assert bus is None

    @patch("victor.core.events.backends.get_observability_bus")
    def test_get_event_bus_lazy_loading(self, mock_get_bus):
        """Test _get_event_bus lazy-loads the event bus."""
        mock_bus = Mock()
        mock_get_bus.return_value = mock_bus

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=True,
        )

        # First call should load
        bus = coordinator._get_event_bus()
        assert bus == mock_bus
        mock_get_bus.assert_called_once()

        # Second call should use cached value
        bus2 = coordinator._get_event_bus()
        assert bus2 == mock_bus
        assert mock_get_bus.call_count == 1

    @patch("victor.core.events.backends.get_observability_bus")
    def test_get_event_bus_import_error_disables_publishing(self, mock_get_bus):
        """Test _get_event_bus handles ImportError gracefully."""
        mock_get_bus.side_effect = ImportError("No module named 'events'")

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=True,
        )

        bus = coordinator._get_event_bus()
        assert bus is None
        assert coordinator._enable_event_publishing is False

    @pytest.mark.asyncio
    async def test_publish_event_when_disabled(self):
        """Test _publish_event does nothing when publishing disabled."""
        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        # Should not raise
        await coordinator._publish_event("test.topic", {"data": "test"})

    @pytest.mark.asyncio
    async def test_publish_event_with_no_event_bus(self):
        """Test _publish_event when event bus is None."""
        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=True,
        )

        # Force event bus to None
        coordinator._event_bus = None

        # Should not raise
        await coordinator._publish_event("test.topic", {"data": "test"})

    @pytest.mark.asyncio
    @patch("victor.core.events.backends.get_observability_bus")
    async def test_publish_event_success(self, mock_get_bus):
        """Test _publish_event successfully publishes event."""
        mock_bus = Mock()
        mock_bus.emit = AsyncMock()
        mock_get_bus.return_value = mock_bus

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=True,
        )

        await coordinator._publish_event("test.topic", {"key": "value"})

        mock_bus.emit.assert_called_once_with(
            topic="test.topic",
            data={"key": "value"},
            source="evaluation_coordinator",
        )

    @pytest.mark.asyncio
    @patch("victor.core.events.backends.get_observability_bus")
    async def test_publish_event_handles_exceptions(self, mock_get_bus):
        """Test _publish_event handles exceptions gracefully."""
        mock_bus = Mock()
        mock_bus.emit = AsyncMock(side_effect=RuntimeError("Bus error"))
        mock_get_bus.return_value = mock_bus

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=True,
        )

        # Should not raise despite error
        await coordinator._publish_event("test.topic", {"key": "value"})


class TestEvaluationCoordinatorUsageAnalytics:
    """Test usage_analytics property."""

    def test_usage_analytics_property_returns_analytics(self):
        """Test usage_analytics property returns the analytics instance."""
        mock_analytics = Mock()
        coordinator = EvaluationCoordinator(
            usage_analytics=mock_analytics,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
        )

        assert coordinator.usage_analytics == mock_analytics

    def test_usage_analytics_property_returns_none(self):
        """Test usage_analytics property returns None when not set."""
        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
        )

        assert coordinator.usage_analytics is None


class TestEvaluationCoordinatorRecordIntelligentOutcome:
    """Test record_intelligent_outcome method."""

    @pytest.mark.asyncio
    async def test_record_intelligent_outcome_success(self):
        """Test recording successful outcome."""
        # Setup
        mock_integration = Mock()
        mock_integration.record_intelligent_outcome = Mock()

        mock_provider = Mock()
        mock_provider.name = "anthropic"

        mock_stream_context = Mock()
        mock_stream_context._continuation_prompts = 2
        mock_stream_context._max_continuation_prompts_used = 6
        mock_stream_context._stuck_loop_detected = False

        mock_vertical_context = Mock()
        mock_vertical_context.vertical_name = "coding"

        mock_rl_coordinator = Mock()

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: mock_rl_coordinator,
            get_vertical_context_fn=lambda: mock_vertical_context,
            get_stream_context_fn=lambda: mock_stream_context,
            get_provider_fn=lambda: mock_provider,
            get_model_fn=lambda: "claude-3-opus",
            get_tool_calls_used_fn=lambda: 5,
            get_intelligent_integration_fn=lambda: mock_integration,
            enable_event_publishing=False,
        )

        # Execute
        await coordinator.record_intelligent_outcome(
            success=True,
            quality_score=0.9,
            user_satisfied=True,
            completed=True,
        )

        # Assert
        mock_integration.record_intelligent_outcome.assert_called_once_with(
            success=True,
            quality_score=0.9,
            user_satisfied=True,
            completed=True,
            rl_coordinator=mock_rl_coordinator,
            stream_context=mock_stream_context,
            vertical_context=mock_vertical_context,
            provider_name="anthropic",
            model="claude-3-opus",
            tool_calls_used=5,
            continuation_prompts=2,
            max_continuation_prompts_used=6,
            stuck_loop_detected=False,
        )

    @pytest.mark.asyncio
    async def test_record_intelligent_outcome_with_no_integration(self):
        """Test recording outcome when integration is None."""
        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        # Should not raise
        await coordinator.record_intelligent_outcome(success=True, quality_score=0.8)

    @pytest.mark.asyncio
    async def test_record_intelligent_outcome_with_default_parameters(self):
        """Test recording outcome with default parameter values."""
        mock_integration = Mock()
        mock_integration.record_intelligent_outcome = Mock()

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: mock_integration,
            enable_event_publishing=False,
        )

        await coordinator.record_intelligent_outcome(success=True)

        mock_integration.record_intelligent_outcome.assert_called_once_with(
            success=True,
            quality_score=0.5,
            user_satisfied=True,
            completed=True,
            rl_coordinator=None,
            stream_context=None,
            vertical_context=None,
            provider_name="unknown",
            model="unknown",
            tool_calls_used=0,
            continuation_prompts=0,
            max_continuation_prompts_used=6,
            stuck_loop_detected=False,
        )

    @pytest.mark.asyncio
    async def test_record_intelligent_outcome_handles_exception(self):
        """Test recording outcome handles integration exceptions."""
        mock_integration = Mock()
        mock_integration.record_intelligent_outcome = Mock(
            side_effect=RuntimeError("Integration failed")
        )

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: mock_integration,
            enable_event_publishing=False,
        )

        # Should not raise despite exception
        await coordinator.record_intelligent_outcome(success=True)

    @pytest.mark.asyncio
    @patch("victor.core.events.backends.get_observability_bus")
    async def test_record_intelligent_outcome_publishes_event(self, mock_get_bus):
        """Test recording outcome publishes event."""
        # Setup
        mock_bus = Mock()
        mock_bus.emit = AsyncMock()
        mock_get_bus.return_value = mock_bus

        mock_integration = Mock()
        mock_integration.record_intelligent_outcome = Mock()

        mock_provider = Mock()
        mock_provider.name = "openai"

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: mock_provider,
            get_model_fn=lambda: "gpt-4",
            get_tool_calls_used_fn=lambda: 3,
            get_intelligent_integration_fn=lambda: mock_integration,
            enable_event_publishing=True,
        )

        # Execute
        await coordinator.record_intelligent_outcome(
            success=False,
            quality_score=0.3,
            user_satisfied=False,
            completed=False,
        )

        # Assert event was published
        mock_bus.emit.assert_called_once()
        call_args = mock_bus.emit.call_args
        assert call_args[1]["topic"] == "evaluation.outcome_recorded"
        assert call_args[1]["source"] == "evaluation_coordinator"
        assert call_args[1]["data"]["success"] is False
        assert call_args[1]["data"]["quality_score"] == 0.3
        assert call_args[1]["data"]["provider"] == "openai"
        assert call_args[1]["data"]["model"] == "gpt-4"
        assert call_args[1]["data"]["tool_calls_used"] == 3

    @pytest.mark.asyncio
    async def test_record_intelligent_outcome_with_stuck_loop_detected(self):
        """Test recording outcome with stuck loop detected."""
        mock_integration = Mock()

        mock_stream_context = Mock()
        mock_stream_context._continuation_prompts = 6
        mock_stream_context._max_continuation_prompts_used = 6
        mock_stream_context._stuck_loop_detected = True

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: mock_stream_context,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: mock_integration,
            enable_event_publishing=False,
        )

        await coordinator.record_intelligent_outcome(success=False, quality_score=0.2)

        mock_integration.record_intelligent_outcome.assert_called_once()
        call_kwargs = mock_integration.record_intelligent_outcome.call_args[1]
        assert call_kwargs["stuck_loop_detected"] is True
        assert call_kwargs["continuation_prompts"] == 6


class TestEvaluationCoordinatorSendRLRewardSignal:
    """Test send_rl_reward_signal method."""

    @pytest.mark.asyncio
    async def test_send_rl_reward_signal_success(self):
        """Test sending RL reward signal for successful session."""
        # Setup
        mock_rl_coordinator = Mock()
        mock_rl_coordinator.record_outcome = Mock()

        mock_session = Mock()
        mock_session.provider = "anthropic"
        mock_session.model = "claude-3-opus"
        mock_session.task_type = "coding"
        mock_session.error = None
        mock_session.cancelled = False
        mock_session.duration = 5.0
        mock_session.session_id = "test123"
        mock_session.metrics = Mock()
        mock_session.metrics.total_chunks = 100

        mock_vertical_context = Mock()
        mock_vertical_context.vertical_name = "coding"

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: mock_rl_coordinator,
            get_vertical_context_fn=lambda: mock_vertical_context,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        # Execute
        await coordinator.send_rl_reward_signal(mock_session)

        # Assert
        mock_rl_coordinator.record_outcome.assert_called_once()
        call_args = mock_rl_coordinator.record_outcome.call_args[0]
        assert call_args[0] == "model_selector"
        assert call_args[2] == "coding"

        outcome = call_args[1]
        assert outcome.provider == "anthropic"
        assert outcome.model == "claude-3-opus"
        assert outcome.task_type == "coding"
        assert outcome.success is True
        assert outcome.quality_score > 0.8  # Should be high for success

    @pytest.mark.asyncio
    async def test_send_rl_reward_signal_with_error(self):
        """Test sending RL reward signal for failed session."""
        mock_rl_coordinator = Mock()
        mock_rl_coordinator.record_outcome = Mock()

        mock_session = Mock()
        mock_session.provider = "openai"
        mock_session.model = "gpt-4"
        mock_session.task_type = "analysis"
        mock_session.error = "API error"
        mock_session.duration = 2.0
        mock_session.session_id = "failed456"
        mock_session.metrics = None

        mock_vertical_context = Mock()
        mock_vertical_context.vertical_name = "coding"

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: mock_rl_coordinator,
            get_vertical_context_fn=lambda: mock_vertical_context,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        await coordinator.send_rl_reward_signal(mock_session)

        outcome = mock_rl_coordinator.record_outcome.call_args[0][1]
        assert outcome.success is False

    @pytest.mark.asyncio
    async def test_send_rl_reward_signal_with_cancelled_session(self):
        """Test sending RL reward signal for cancelled session."""
        mock_rl_coordinator = Mock()
        mock_rl_coordinator.record_outcome = Mock()

        mock_session = Mock()
        mock_session.provider = "anthropic"
        mock_session.model = "claude-3-opus"
        mock_session.error = None
        mock_session.cancelled = True
        mock_session.duration = 1.0
        mock_session.session_id = "cancelled789"
        mock_session.metrics = None

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: mock_rl_coordinator,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        await coordinator.send_rl_reward_signal(mock_session)

        outcome = mock_rl_coordinator.record_outcome.call_args[0][1]
        assert outcome.success is False

    @pytest.mark.asyncio
    async def test_send_rl_reward_signal_with_no_rl_coordinator(self):
        """Test sending RL reward signal when RL coordinator is None."""
        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        mock_session = Mock()
        mock_session.provider = "test"
        mock_session.model = "test-model"

        # Should not raise
        await coordinator.send_rl_reward_signal(mock_session)

    @pytest.mark.asyncio
    async def test_send_rl_reward_signal_quality_score_fast_response(self):
        """Test quality score calculation for fast response."""
        mock_rl_coordinator = Mock()
        mock_rl_coordinator.record_outcome = Mock()

        mock_session = Mock()
        mock_session.provider = "anthropic"
        mock_session.model = "claude-3-opus"
        mock_session.error = None
        mock_session.cancelled = False
        mock_session.duration = 5.0  # Fast response (< 10s)
        mock_session.session_id = "fast123"
        mock_session.metrics = None

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: mock_rl_coordinator,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        await coordinator.send_rl_reward_signal(mock_session)

        outcome = mock_rl_coordinator.record_outcome.call_args[0][1]
        # Fast response should get bonus (0.8 + 0.1 = 0.9)
        assert outcome.quality_score >= 0.85

    @pytest.mark.asyncio
    async def test_send_rl_reward_signal_quality_score_slow_response(self):
        """Test quality score calculation for slow response."""
        mock_rl_coordinator = Mock()
        mock_rl_coordinator.record_outcome = Mock()

        mock_session = Mock()
        mock_session.provider = "anthropic"
        mock_session.model = "claude-3-opus"
        mock_session.error = None
        mock_session.cancelled = False
        mock_session.duration = 15.0  # Slow response (> 10s)
        mock_session.session_id = "slow123"
        mock_session.metrics = None

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: mock_rl_coordinator,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        await coordinator.send_rl_reward_signal(mock_session)

        outcome = mock_rl_coordinator.record_outcome.call_args[0][1]
        # Slow response should not get bonus (0.8 only)
        assert outcome.quality_score < 0.85

    @pytest.mark.asyncio
    async def test_send_rl_reward_signal_with_vertical_context(self):
        """Test sending RL reward signal with vertical context."""
        mock_rl_coordinator = Mock()
        mock_rl_coordinator.record_outcome = Mock()

        mock_session = Mock()
        mock_session.provider = "anthropic"
        mock_session.model = "claude-3-opus"
        mock_session.error = None
        mock_session.cancelled = False
        mock_session.duration = 5.0
        mock_session.session_id = "vertical123"
        mock_session.metrics = None

        mock_vertical_context = Mock()
        mock_vertical_context.vertical_name = "research"

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: mock_rl_coordinator,
            get_vertical_context_fn=lambda: mock_vertical_context,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        await coordinator.send_rl_reward_signal(mock_session)

        call_args = mock_rl_coordinator.record_outcome.call_args[0]
        assert call_args[2] == "research"  # vertical parameter
        assert call_args[1].vertical == "research"

    @pytest.mark.asyncio
    @patch("victor.core.events.backends.get_observability_bus")
    async def test_send_rl_reward_signal_publishes_event(self, mock_get_bus):
        """Test sending RL reward signal publishes event."""
        mock_bus = Mock()
        mock_bus.emit = AsyncMock()
        mock_get_bus.return_value = mock_bus

        mock_rl_coordinator = Mock()
        mock_rl_coordinator.record_outcome = Mock()

        mock_session = Mock()
        mock_session.provider = "anthropic"
        mock_session.model = "claude-3-opus"
        mock_session.error = None
        mock_session.cancelled = False
        mock_session.duration = 5.0
        mock_session.session_id = "event123"
        mock_session.metrics = None

        mock_vertical_context = Mock()
        mock_vertical_context.vertical_name = "coding"

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: mock_rl_coordinator,
            get_vertical_context_fn=lambda: mock_vertical_context,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=True,
        )

        await coordinator.send_rl_reward_signal(mock_session)

        # Assert event was published
        mock_bus.emit.assert_called_once()
        call_args = mock_bus.emit.call_args
        assert call_args[1]["topic"] == "evaluation.rl_feedback"
        assert call_args[1]["source"] == "evaluation_coordinator"
        assert call_args[1]["data"]["session_id"] == "event123"
        assert call_args[1]["data"]["provider"] == "anthropic"
        assert call_args[1]["data"]["model"] == "claude-3-opus"
        assert "reward" in call_args[1]["data"]

    @pytest.mark.asyncio
    async def test_send_rl_reward_signal_handles_import_error(self):
        """Test sending RL reward signal handles ImportError."""
        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: Mock(),
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        mock_session = Mock()

        # Mock ImportError by patching the import
        with patch("victor.framework.rl.base.RLOutcome", side_effect=ImportError):
            # Should not raise
            await coordinator.send_rl_reward_signal(mock_session)

    @pytest.mark.asyncio
    async def test_send_rl_reward_signal_handles_attribute_error(self):
        """Test sending RL reward signal handles AttributeError."""
        mock_rl_coordinator = Mock()
        # Missing record_outcome method
        del mock_rl_coordinator.record_outcome

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: mock_rl_coordinator,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        mock_session = Mock()
        mock_session.provider = "test"
        mock_session.model = "test-model"
        mock_session.error = None
        mock_session.cancelled = False
        mock_session.duration = 5.0
        mock_session.session_id = "attr_error123"
        mock_session.metrics = None

        # Should not raise
        await coordinator.send_rl_reward_signal(mock_session)


class TestEvaluationCoordinatorFlushAnalytics:
    """Test flush_analytics method."""

    @pytest.mark.asyncio
    async def test_flush_analytics_with_all_components(self):
        """Test flushing all analytics components."""
        # Setup
        mock_usage_analytics = Mock()
        mock_usage_analytics.flush = Mock()

        mock_sequence_tracker = Mock()
        mock_sequence_tracker.get_statistics = Mock(return_value={"unique_transitions": 42})

        coordinator = EvaluationCoordinator(
            usage_analytics=mock_usage_analytics,
            sequence_tracker=mock_sequence_tracker,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        # Execute
        results = await coordinator.flush_analytics()

        # Assert
        assert results["usage_analytics"] is True
        assert results["sequence_tracker"] is True
        assert results["tool_cache"] is False

        mock_usage_analytics.flush.assert_called_once()
        mock_sequence_tracker.get_statistics.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_analytics_with_no_components(self):
        """Test flushing when all components are None."""
        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        results = await coordinator.flush_analytics()

        assert results["usage_analytics"] is False
        assert results["sequence_tracker"] is False
        assert results["tool_cache"] is False

    @pytest.mark.asyncio
    async def test_flush_analytics_with_usage_analytics_failure(self):
        """Test flushing when usage analytics raises exception."""
        mock_usage_analytics = Mock()
        mock_usage_analytics.flush = Mock(side_effect=IOError("Disk full"))

        coordinator = EvaluationCoordinator(
            usage_analytics=mock_usage_analytics,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        results = await coordinator.flush_analytics()

        assert results["usage_analytics"] is False

    @pytest.mark.asyncio
    async def test_flush_analytics_with_sequence_tracker_failure(self):
        """Test flushing when sequence tracker raises exception."""
        mock_sequence_tracker = Mock()
        mock_sequence_tracker.get_statistics = Mock(side_effect=RuntimeError("Tracker error"))

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=mock_sequence_tracker,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        results = await coordinator.flush_analytics()

        assert results["sequence_tracker"] is False

    @pytest.mark.asyncio
    @patch("victor.core.events.backends.get_observability_bus")
    async def test_flush_analytics_publishes_event(self, mock_get_bus):
        """Test flushing analytics publishes event."""
        mock_bus = Mock()
        mock_bus.emit = AsyncMock()
        mock_get_bus.return_value = mock_bus

        mock_usage_analytics = Mock()
        mock_usage_analytics.flush = Mock()

        mock_sequence_tracker = Mock()
        mock_sequence_tracker.get_statistics = Mock(return_value={"unique_transitions": 10})

        coordinator = EvaluationCoordinator(
            usage_analytics=mock_usage_analytics,
            sequence_tracker=mock_sequence_tracker,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=True,
        )

        results = await coordinator.flush_analytics()

        # Assert event was published
        mock_bus.emit.assert_called_once()
        call_args = mock_bus.emit.call_args
        assert call_args[1]["topic"] == "evaluation.analytics_flushed"
        assert call_args[1]["source"] == "evaluation_coordinator"
        assert call_args[1]["data"]["records_written"] == 2  # analytics + tracker
        assert call_args[1]["data"]["components"]["usage_analytics"] is True
        assert call_args[1]["data"]["components"]["sequence_tracker"] is True
        assert "flush_duration_ms" in call_args[1]["data"]

    @pytest.mark.asyncio
    async def test_flush_analytics_records_duration(self):
        """Test that flush_analytics records duration."""
        mock_usage_analytics = Mock()
        mock_usage_analytics.flush = Mock()

        coordinator = EvaluationCoordinator(
            usage_analytics=mock_usage_analytics,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )


        results = await coordinator.flush_analytics()

        # Should complete quickly (< 1 second)
        # Just verify the structure, not exact timing
        assert "usage_analytics" in results

    @pytest.mark.asyncio
    async def test_flush_analytics_partial_success(self):
        """Test flushing with partial success (one component fails)."""
        mock_usage_analytics = Mock()
        mock_usage_analytics.flush = Mock()  # Success

        mock_sequence_tracker = Mock()
        mock_sequence_tracker.get_statistics = Mock(
            side_effect=Exception("Tracker failed")
        )  # Failure

        coordinator = EvaluationCoordinator(
            usage_analytics=mock_usage_analytics,
            sequence_tracker=mock_sequence_tracker,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        results = await coordinator.flush_analytics()

        assert results["usage_analytics"] is True
        assert results["sequence_tracker"] is False
        assert results["tool_cache"] is False


class TestEvaluationCoordinatorIntegrationScenarios:
    """Test integration scenarios and complex workflows."""

    @pytest.mark.asyncio
    @patch("victor.core.events.backends.get_observability_bus")
    async def test_complete_evaluation_workflow(self, mock_get_bus):
        """Test complete workflow: record outcome, send reward, flush."""
        # Setup event bus
        mock_bus = Mock()
        mock_bus.emit = AsyncMock()
        mock_get_bus.return_value = mock_bus

        # Setup dependencies
        mock_integration = Mock()
        mock_integration.record_intelligent_outcome = Mock()

        mock_rl_coordinator = Mock()
        mock_rl_coordinator.record_outcome = Mock()

        mock_usage_analytics = Mock()
        mock_usage_analytics.flush = Mock()

        mock_sequence_tracker = Mock()
        mock_sequence_tracker.get_statistics = Mock(return_value={"unique_transitions": 5})

        mock_provider = Mock()
        mock_provider.name = "anthropic"

        mock_session = Mock()
        mock_session.provider = "anthropic"
        mock_session.model = "claude-3-opus"
        mock_session.error = None
        mock_session.cancelled = False
        mock_session.duration = 3.0
        mock_session.session_id = "workflow123"
        mock_session.metrics = None

        coordinator = EvaluationCoordinator(
            usage_analytics=mock_usage_analytics,
            sequence_tracker=mock_sequence_tracker,
            get_rl_coordinator_fn=lambda: mock_rl_coordinator,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: mock_provider,
            get_model_fn=lambda: "claude-3-opus",
            get_tool_calls_used_fn=lambda: 3,
            get_intelligent_integration_fn=lambda: mock_integration,
            enable_event_publishing=True,
        )

        # Execute workflow
        # 1. Record intelligent outcome
        await coordinator.record_intelligent_outcome(
            success=True,
            quality_score=0.95,
            user_satisfied=True,
            completed=True,
        )

        # 2. Send RL reward signal
        await coordinator.send_rl_reward_signal(mock_session)

        # 3. Flush analytics
        results = await coordinator.flush_analytics()

        # Assert all operations completed
        mock_integration.record_intelligent_outcome.assert_called_once()
        mock_rl_coordinator.record_outcome.assert_called_once()
        mock_usage_analytics.flush.assert_called_once()
        mock_sequence_tracker.get_statistics.assert_called_once()

        assert results["usage_analytics"] is True
        assert results["sequence_tracker"] is True

        # Assert events were published (3 events)
        assert mock_bus.emit.call_count == 3

        # Verify event topics
        topics = [call[1]["topic"] for call in mock_bus.emit.call_args_list]
        assert "evaluation.outcome_recorded" in topics
        assert "evaluation.rl_feedback" in topics
        assert "evaluation.analytics_flushed" in topics

    @pytest.mark.asyncio
    async def test_workflow_with_missing_dependencies(self):
        """Test workflow gracefully handles missing dependencies."""
        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        mock_session = Mock()

        # Should not raise despite all dependencies being None
        await coordinator.record_intelligent_outcome(success=True)
        await coordinator.send_rl_reward_signal(mock_session)
        results = await coordinator.flush_analytics()

        assert results["usage_analytics"] is False
        assert results["sequence_tracker"] is False

    @pytest.mark.asyncio
    async def test_workflow_with_event_publishing_disabled(self):
        """Test workflow with event publishing disabled."""
        mock_integration = Mock()
        mock_integration.record_intelligent_outcome = Mock()

        mock_rl_coordinator = Mock()
        mock_rl_coordinator.record_outcome = Mock()

        mock_usage_analytics = Mock()
        mock_usage_analytics.flush = Mock()

        coordinator = EvaluationCoordinator(
            usage_analytics=mock_usage_analytics,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: mock_rl_coordinator,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: mock_integration,
            enable_event_publishing=False,  # Disabled
        )

        mock_session = Mock()
        mock_session.provider = "test"
        mock_session.model = "test-model"
        mock_session.error = None
        mock_session.cancelled = False
        mock_session.duration = 2.0
        mock_session.session_id = "no_events123"
        mock_session.metrics = None

        # Execute workflow
        await coordinator.record_intelligent_outcome(success=True)
        await coordinator.send_rl_reward_signal(mock_session)
        await coordinator.flush_analytics()

        # Verify operations completed (event bus shouldn't be used)
        mock_integration.record_intelligent_outcome.assert_called_once()
        mock_rl_coordinator.record_outcome.assert_called_once()
        mock_usage_analytics.flush.assert_called_once()


class TestEvaluationCoordinatorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_record_outcome_with_zero_quality_score(self):
        """Test recording outcome with zero quality score."""
        mock_integration = Mock()

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: mock_integration,
            enable_event_publishing=False,
        )

        await coordinator.record_intelligent_outcome(
            success=False,
            quality_score=0.0,
            user_satisfied=False,
            completed=False,
        )

        mock_integration.record_intelligent_outcome.assert_called_once_with(
            success=False,
            quality_score=0.0,
            user_satisfied=False,
            completed=False,
            rl_coordinator=None,
            stream_context=None,
            vertical_context=None,
            provider_name="unknown",
            model="unknown",
            tool_calls_used=0,
            continuation_prompts=0,
            max_continuation_prompts_used=6,
            stuck_loop_detected=False,
        )

    @pytest.mark.asyncio
    async def test_record_outcome_with_max_quality_score(self):
        """Test recording outcome with maximum quality score."""
        mock_integration = Mock()

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: mock_integration,
            enable_event_publishing=False,
        )

        await coordinator.record_intelligent_outcome(
            success=True,
            quality_score=1.0,
            user_satisfied=True,
            completed=True,
        )

        call_kwargs = mock_integration.record_intelligent_outcome.call_args[1]
        assert call_kwargs["quality_score"] == 1.0

    @pytest.mark.asyncio
    async def test_send_rl_reward_signal_with_missing_session_attributes(self):
        """Test sending RL reward signal with minimal session attributes."""
        mock_rl_coordinator = Mock()
        mock_rl_coordinator.record_outcome = Mock()

        # Create session with minimal attributes
        mock_session = Mock(spec=["provider", "model"])
        mock_session.provider = "test_provider"
        mock_session.model = "test_model"

        coordinator = EvaluationCoordinator(
            usage_analytics=None,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: mock_rl_coordinator,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        # Should handle missing attributes gracefully
        await coordinator.send_rl_reward_signal(mock_session)

        # Verify outcome was created with defaults
        mock_rl_coordinator.record_outcome.assert_called_once()
        outcome = mock_rl_coordinator.record_outcome.call_args[0][1]
        assert outcome.provider == "test_provider"
        assert outcome.model == "test_model"

    @pytest.mark.asyncio
    async def test_flush_analytics_concurrent_calls(self):
        """Test that flush_analytics can be called concurrently."""
        mock_usage_analytics = Mock()
        mock_usage_analytics.flush = Mock()

        coordinator = EvaluationCoordinator(
            usage_analytics=mock_usage_analytics,
            sequence_tracker=None,
            get_rl_coordinator_fn=lambda: None,
            get_vertical_context_fn=lambda: None,
            get_stream_context_fn=lambda: None,
            get_provider_fn=lambda: None,
            get_model_fn=lambda: "unknown",
            get_tool_calls_used_fn=lambda: 0,
            get_intelligent_integration_fn=lambda: None,
            enable_event_publishing=False,
        )

        import asyncio

        # Call flush concurrently
        tasks = [coordinator.flush_analytics() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 5
        assert all(r["usage_analytics"] is True for r in results)
        assert mock_usage_analytics.flush.call_count == 5
