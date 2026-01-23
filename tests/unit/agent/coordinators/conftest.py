# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0

"""Shared fixtures for coordinator tests.

This module provides a single comprehensive mock orchestrator fixture
that all coordinator test classes can use, reducing duplication.
"""

import time
import pytest
from unittest.mock import AsyncMock, Mock
from types import SimpleNamespace

from victor.providers.base import CompletionResponse, StreamChunk
from victor.framework.task import TaskComplexity
from victor.agent.unified_task_tracker import TrackerTaskType


def create_base_mock_orchestrator(supports_tools: bool = True) -> Mock:
    """Create a comprehensive mock orchestrator with all required dependencies.

    Args:
        supports_tools: Whether the provider should support tools

    Returns:
        A fully configured Mock orchestrator
    """
    orch = Mock()

    # Conversation
    orch.conversation = Mock()
    orch.conversation.ensure_system_prompt = Mock()
    orch.conversation.message_count = Mock(return_value=5)

    # Provider
    orch.provider = Mock()
    orch.provider.supports_tools = Mock(return_value=supports_tools)
    orch.provider.chat = AsyncMock(
        return_value=CompletionResponse(
            content="Response content", role="assistant", tool_calls=None
        )
    )

    # Model settings
    orch.model = "test-model"
    orch.temperature = 0.7
    orch.max_tokens = 4096
    orch.tool_budget = 10
    orch.tool_calls_used = 0
    orch.thinking = False

    # Messages
    orch.messages = []
    orch.add_message = Mock()
    orch._system_added = False

    # Task classification
    orch.task_classifier = Mock()
    orch.task_classifier.classify = Mock(
        return_value=Mock(tool_budget=5, complexity=TaskComplexity.MEDIUM)
    )

    # Settings
    orch.settings = Mock()
    orch.settings.chat_max_iterations = 10
    orch.settings.stream_idle_timeout_seconds = 300

    # Tool selector
    orch.tool_selector = Mock()
    orch.tool_selector.select_tools = AsyncMock(return_value=[])
    orch.tool_selector.prioritize_by_stage = Mock(return_value=[])

    # Conversation state
    orch.conversation_state = Mock()
    orch.conversation_state.state = Mock()
    orch.conversation_state.state.stage = None
    orch._context_compactor = None

    # Tool handling
    orch._handle_tool_calls = AsyncMock(return_value=[])
    orch._prepare_intelligent_request = AsyncMock(return_value=None)

    # Response completer
    orch.response_completer = Mock()
    orch.response_completer.ensure_response = AsyncMock(
        return_value=Mock(content="Fallback response")
    )
    orch.response_completer.format_tool_failure_message = Mock(
        return_value="Tool failed message"
    )

    # Token usage
    orch._cumulative_token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    orch._current_stream_context = None

    # Streaming components
    orch._metrics_coordinator = Mock()
    orch._metrics_coordinator.start_streaming = Mock()
    orch._metrics_coordinator.stop_streaming = Mock()
    orch._metrics_collector = Mock()
    # Create a proper StreamMetrics object for init_stream_metrics
    from victor.providers.stream_adapter import StreamMetrics
    orch._metrics_collector.init_stream_metrics = Mock(return_value=StreamMetrics(start_time=0.0))
    orch._metrics_collector.record_first_token = Mock()

    # Session state
    orch._session_state = Mock()
    orch._session_state.reset_for_new_turn = Mock()

    # Unified tracker
    orch.unified_tracker = Mock()
    orch.unified_tracker.reset = Mock()
    orch.unified_tracker.config = Mock()
    orch.unified_tracker.config.__getitem__ = Mock(side_effect=lambda k: 50)
    orch.unified_tracker.config.get = Mock(return_value=50)
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
    orch._sequence_tracker = None

    # Context manager
    orch._context_manager = SimpleNamespace()
    orch._context_manager.start_background_compaction = AsyncMock()
    orch._context_manager.get_max_context_chars = Mock(return_value=100000)

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

    # Cancellation checking
    orch._check_cancellation = Mock(return_value=False)

    # Provider coordinator
    orch._provider_coordinator = Mock()
    orch._provider_coordinator.get_rate_limit_wait_time = Mock(return_value=1.0)

    # Sanitizer
    orch.sanitizer = Mock()
    orch.sanitizer.sanitize = Mock(return_value="sanitized content")
    orch.sanitizer.strip_markup = Mock(return_value="plain text")
    orch.sanitizer.is_garbage_content = Mock(return_value=False)

    # Observed files
    orch.observed_files = []

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
    orch._recovery_coordinator.get_recovery_fallback_message = Mock(
        return_value="Fallback message"
    )

    # Intelligent outcome
    orch._record_intelligent_outcome = Mock()

    # Task completion detector
    orch._task_completion_detector = Mock()
    orch._task_completion_detector.analyze_response = Mock()
    orch._task_completion_detector.get_completion_confidence = Mock(return_value=None)

    # Force finalize
    orch._force_finalize = False

    # Streaming controller with session
    orch._streaming_controller = Mock()
    orch._streaming_controller.current_session = None
    # Mock start_session to return a proper session-like object
    mock_session = SimpleNamespace(
        session_id="test-session",
        model="test-model",
        provider="test-provider",
        start_time=time.time(),
        end_time=None,
        metrics=None,
        cancelled=False,
        error=None,
    )
    orch._streaming_controller.start_session = Mock(return_value=mock_session)

    return orch


@pytest.fixture
def base_mock_orchestrator() -> Mock:
    """Base mock orchestrator fixture that all tests can use."""
    return create_base_mock_orchestrator()
