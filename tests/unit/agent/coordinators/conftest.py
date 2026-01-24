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
from victor.core.state import ConversationStage


# =============================================================================
# Mock Factory Functions (Phase 3.2)
# =============================================================================


def create_mock_mode_controller(initial_mode: str = "build") -> Mock:
    """Create a mock mode controller.

    Args:
        initial_mode: Initial mode name (build, plan, explore)

    Returns:
        Mock ModeControllerProtocol implementation
    """
    from types import SimpleNamespace

    controller = Mock()
    controller.current_mode = SimpleNamespace(
        name=initial_mode.upper(),
        tool_whitelist=None,
        tool_blacklist=set(),
        exploration_multiplier=1.0 if initial_mode == "build" else 2.5,
    )
    controller.config = Mock()
    controller.switch_mode = Mock(return_value=True)
    controller.is_tool_allowed = Mock(return_value=True)
    controller.get_tool_priority = Mock(return_value=1.0)
    controller.get_exploration_multiplier = Mock(
        return_value=1.0 if initial_mode == "build" else 2.5
    )
    controller.get_system_prompt_addition = Mock(return_value="")

    return controller


def create_mock_provider_lifecycle_manager() -> Mock:
    """Create a mock provider lifecycle manager.

    Returns:
        Mock ProviderLifecycleProtocol implementation
    """
    manager = Mock()
    manager.apply_exploration_settings = Mock()
    manager.get_prompt_contributors = Mock(return_value=[])
    manager.create_prompt_builder = Mock(return_value=Mock())
    manager.calculate_tool_budget = Mock(return_value=50)
    manager.should_respect_sticky_budget = Mock(return_value=False)

    return manager


def create_mock_handler_registry() -> Mock:
    """Create a mock handler registry.

    Returns:
        Mock HandlerRegistry
    """
    registry = Mock()
    registry.register = Mock()
    registry.unregister = Mock(return_value=True)
    registry.get = Mock(return_value=None)
    registry.get_all = Mock(return_value={})
    registry.get_by_vertical = Mock(return_value=[])
    registry.list_handlers = Mock(return_value=[])
    registry.reset = Mock()

    return registry


def create_mock_stage_transition_engine(
    initial_stage: ConversationStage = ConversationStage.INITIAL,
) -> Mock:
    """Create a mock stage transition engine.

    Args:
        initial_stage: Initial conversation stage

    Returns:
        Mock StageTransitionProtocol implementation
    """
    engine = Mock()
    engine.current_stage = initial_stage
    engine.cooldown_seconds = 2.0
    engine.transition_history = []

    engine.can_transition = Mock(return_value=True)
    engine.transition_to = Mock(return_value=True)
    engine.get_valid_transitions = Mock(return_value=[
        ConversationStage.PLANNING,
        ConversationStage.READING,
    ])
    engine.get_tool_priority_multiplier = Mock(return_value=1.0)
    engine.register_callback = Mock()
    engine.unregister_callback = Mock()
    engine.reset = Mock()

    return engine


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
    # Set up default async stream generator
    async def default_stream(*args, **kwargs):
        from victor.providers.base import StreamChunk
        yield StreamChunk(content="Response content", is_final=True)
    # Use a property to return the generator function directly
    orch.provider.stream = default_stream

    # Model settings
    orch.model = "test-model"
    orch.temperature = 0.7
    orch.max_tokens = 4096
    orch.tool_budget = 10
    orch.tool_calls_used = 0
    orch.thinking = False
    orch.provider_name = "test_provider"

    # Messages
    orch.messages = []
    orch.add_message = Mock()
    orch._system_added = False

    # Task classification
    orch.task_classifier = Mock()
    orch.task_classifier.classify = Mock(
        return_value=Mock(tool_budget=5, complexity=TaskComplexity.MEDIUM)
    )
    # Intent classifier needs to return a proper object with confidence attribute
    orch.intent_classifier = Mock()
    # Use SimpleNamespace with nested intent.name
    mock_intent = SimpleNamespace()
    mock_intent.name = "unknown"
    orch.intent_classifier.classify_intent_sync = Mock(
        return_value=SimpleNamespace(
            intent=mock_intent,
            confidence=0.5,
            top_matches=[]
        )
    )

    # Settings
    orch.settings = Mock()
    orch.settings.chat_max_iterations = 10
    orch.settings.stream_idle_timeout_seconds = 300
    orch.settings.continuation_medium_max_interventions = 5
    orch.settings.continuation_medium_max_iterations = 50

    # Tool selector
    orch.tool_selector = Mock()
    orch.tool_selector.select_tools = AsyncMock(return_value=[])
    orch.tool_selector.prioritize_by_stage = Mock(return_value=[])
    orch.tool_selector.initialize_tool_embeddings = AsyncMock(return_value=None)

    # Conversation state
    orch.conversation_state = Mock()
    orch.conversation_state.state = Mock()
    orch.conversation_state.state.stage = None
    orch._context_compactor = None

    # Tool handling - return success by default so tool execution continues
    orch._handle_tool_calls = AsyncMock(return_value=[{"success": True}])
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

    # Task tracker (for recovery context)
    orch._task_tracker = Mock()
    orch._task_tracker.current_task_type = TrackerTaskType.EDIT
    orch._task_tracker.is_analysis_task = False
    orch._task_tracker.is_action_task = False

    # Reminder manager
    orch.reminder_manager = Mock()
    orch.reminder_manager.reset = Mock()
    orch.reminder_manager.update_state = Mock()
    orch.reminder_manager.get_consolidated_reminder = Mock(return_value=None)  # No reminder by default

    # Message adder
    orch.add_message = Mock()

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

    # Recovery context creator
    orch._create_recovery_context = Mock(
        return_value=SimpleNamespace(
            elapsed_time=0.0,
            iteration=1,
            tool_calls_used=0,
            consecutive_empty=0,
        )
    )

    # Sanitizer - pass through by default (identity function)
    orch.sanitizer = Mock()
    orch.sanitizer.sanitize = Mock(side_effect=lambda x: x)  # Return original content
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
    # Don't generate extra chunks by default - return empty chunks
    orch._chunk_generator.generate_metrics_chunk = Mock(return_value=StreamChunk(content="", is_final=False))
    orch._chunk_generator.generate_final_marker_chunk = Mock(return_value=StreamChunk(content="", is_final=False))
    # Tool execution chunks
    orch._chunk_generator.generate_tool_start_chunk = Mock(return_value=StreamChunk(content="", is_final=False))
    orch._chunk_generator.generate_tool_result_chunks = Mock(return_value=[])
    orch._chunk_generator.generate_thinking_status_chunk = Mock(return_value=StreamChunk(content="", is_final=False))

    # Streaming handler
    orch._streaming_handler = Mock()

    # Recovery coordinator
    orch._recovery_coordinator = Mock()
    orch._recovery_coordinator.check_natural_completion = Mock(return_value=None)
    orch._recovery_coordinator.handle_empty_response = Mock(return_value=(None, False))
    orch._recovery_coordinator.check_force_action = Mock(return_value=(False, None))
    orch._recovery_coordinator.check_tool_budget = Mock(return_value=None)  # No budget warning
    # truncate_tool_calls: return (tool_calls, num_truncated) - pass through tool_calls unchanged
    orch._recovery_coordinator.truncate_tool_calls = Mock(side_effect=lambda ctx, calls, remaining: (calls, 0))
    # filter_blocked_tool_calls: return (filtered_calls, blocked_chunks, blocked_count) - pass through unchanged
    orch._recovery_coordinator.filter_blocked_tool_calls = Mock(side_effect=lambda ctx, calls: (calls, [], 0))
    orch._recovery_coordinator.check_blocked_threshold = Mock(return_value=None)  # No blocked threshold issue
    orch._recovery_coordinator.get_recovery_fallback_message = Mock(
        return_value="Fallback message"
    )

    # Budget exhausted handler - async generator that yields nothing
    async def handle_budget_exhausted(*args, **kwargs):
        # Empty generator - never yield, just return
        if False:
            yield  # This makes it an async generator but never executes
        return
    orch._handle_budget_exhausted = handle_budget_exhausted

    # Check progress handler
    orch._check_progress_with_handler = Mock()

    # Tool status message generator
    orch._get_tool_status_message = Mock(return_value="Tool status")

    # Observed files set
    orch.observed_files = []

    # Force completion handler
    orch._handle_force_completion_with_handler = Mock(
        return_value=StreamChunk(content="", is_final=False)
    )

    # Intelligent outcome
    orch._record_intelligent_outcome = Mock()

    # Force final response handler - async generator that yields nothing
    async def handle_force_final_response(*args, **kwargs):
        # Empty generator - never yield, just return
        if False:
            yield  # This makes it an async generator but never executes
        return
    orch._handle_force_final_response = handle_force_final_response

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

    # Continuation tracking
    orch._cumulative_prompt_interventions = 0
    orch._continuation_prompts = 0
    orch._asking_input_prompts = 0
    orch._consecutive_blocked_attempts = 0
    orch._current_intent = None

    # ==========================================================================
    # New components from Phase 1 & 2 refactoring
    # ==========================================================================

    # Mode controller (Phase 1.1)
    orch._mode_controller = create_mock_mode_controller()

    # Provider lifecycle manager (Phase 1.2)
    orch._provider_lifecycle_manager = create_mock_provider_lifecycle_manager()

    # Stage transition engine (Phase 2.2)
    orch._stage_transition_engine = create_mock_stage_transition_engine()

    return orch


@pytest.fixture
def base_mock_orchestrator() -> Mock:
    """Base mock orchestrator fixture that all tests can use."""
    return create_base_mock_orchestrator()


@pytest.fixture
def mock_mode_controller() -> Mock:
    """Mock mode controller fixture."""
    return create_mock_mode_controller()


@pytest.fixture
def mock_provider_lifecycle_manager() -> Mock:
    """Mock provider lifecycle manager fixture."""
    return create_mock_provider_lifecycle_manager()


@pytest.fixture
def mock_handler_registry() -> Mock:
    """Mock handler registry fixture."""
    return create_mock_handler_registry()


@pytest.fixture
def mock_stage_transition_engine() -> Mock:
    """Mock stage transition engine fixture."""
    return create_mock_stage_transition_engine()
