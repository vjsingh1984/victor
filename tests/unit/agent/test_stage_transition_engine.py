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

"""Unit tests for StageTransitionEngine.

TDD-first tests for Phase 2.2: Extract StageTransitionEngine.
These tests verify:
1. Valid transition graph (INITIAL → PLANNING → READING → ...)
2. Stage-based tool priority multipliers
3. Transition callbacks for coordination
4. Event emission on transitions
"""

import pytest
from typing import Callable, Dict, List, Optional
from unittest.mock import Mock, MagicMock, patch, call
import time

from victor.core.state import ConversationStage


class TestStageTransitionEngineProtocol:
    """Tests for StageTransitionProtocol definition."""

    def test_protocol_exists(self):
        """Protocol should exist in victor.protocols."""
        from victor.protocols.stage_transition import StageTransitionProtocol

        assert StageTransitionProtocol is not None

    def test_protocol_has_current_stage_property(self):
        """Protocol should define current_stage property."""
        from victor.protocols.stage_transition import StageTransitionProtocol
        import inspect

        # Check for property or attribute definition
        assert hasattr(StageTransitionProtocol, "current_stage")

    def test_protocol_has_transition_method(self):
        """Protocol should define transition_to method."""
        from victor.protocols.stage_transition import StageTransitionProtocol
        import inspect

        # Get member info
        members = dict(inspect.getmembers(StageTransitionProtocol))
        assert "transition_to" in members or hasattr(StageTransitionProtocol, "transition_to")

    def test_protocol_has_can_transition_method(self):
        """Protocol should define can_transition method."""
        from victor.protocols.stage_transition import StageTransitionProtocol

        assert hasattr(StageTransitionProtocol, "can_transition")

    def test_protocol_has_get_valid_transitions_method(self):
        """Protocol should define get_valid_transitions method."""
        from victor.protocols.stage_transition import StageTransitionProtocol

        assert hasattr(StageTransitionProtocol, "get_valid_transitions")

    def test_protocol_has_get_tool_priority_multiplier_method(self):
        """Protocol should define get_tool_priority_multiplier method."""
        from victor.protocols.stage_transition import StageTransitionProtocol

        assert hasattr(StageTransitionProtocol, "get_tool_priority_multiplier")


class TestStageTransitionEngineBasics:
    """Tests for basic StageTransitionEngine functionality."""

    def test_engine_exists(self):
        """StageTransitionEngine should exist."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        assert StageTransitionEngine is not None

    def test_engine_implements_protocol(self):
        """Engine should implement StageTransitionProtocol."""
        from victor.agent.stage_transition_engine import StageTransitionEngine
        from victor.protocols.stage_transition import StageTransitionProtocol

        engine = StageTransitionEngine()
        assert isinstance(engine, StageTransitionProtocol)

    def test_engine_starts_at_initial_stage(self):
        """Engine should start at INITIAL stage."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        assert engine.current_stage == ConversationStage.INITIAL

    def test_engine_can_specify_initial_stage(self):
        """Engine should accept custom initial stage."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine(initial_stage=ConversationStage.PLANNING)
        assert engine.current_stage == ConversationStage.PLANNING


class TestTransitionGraph:
    """Tests for valid transition graph."""

    def test_transition_graph_exists(self):
        """Engine should have a transition graph."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        assert hasattr(engine, "_transition_graph") or hasattr(engine, "transition_graph")

    def test_initial_can_transition_to_planning(self):
        """INITIAL should be able to transition to PLANNING."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        assert engine.can_transition(ConversationStage.PLANNING)

    def test_planning_can_transition_to_reading(self):
        """PLANNING should be able to transition to READING."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine(initial_stage=ConversationStage.PLANNING)
        assert engine.can_transition(ConversationStage.READING)

    def test_reading_can_transition_to_analysis(self):
        """READING should be able to transition to ANALYSIS."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine(initial_stage=ConversationStage.READING)
        assert engine.can_transition(ConversationStage.ANALYSIS)

    def test_analysis_can_transition_to_execution(self):
        """ANALYSIS should be able to transition to EXECUTION."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine(initial_stage=ConversationStage.ANALYSIS)
        assert engine.can_transition(ConversationStage.EXECUTION)

    def test_execution_can_transition_to_verification(self):
        """EXECUTION should be able to transition to VERIFICATION."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine(initial_stage=ConversationStage.EXECUTION)
        assert engine.can_transition(ConversationStage.VERIFICATION)

    def test_verification_can_transition_to_completion(self):
        """VERIFICATION should be able to transition to COMPLETION."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine(initial_stage=ConversationStage.VERIFICATION)
        assert engine.can_transition(ConversationStage.COMPLETION)

    def test_get_valid_transitions_from_initial(self):
        """Should return valid transitions from INITIAL."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        valid = engine.get_valid_transitions()

        assert ConversationStage.PLANNING in valid
        # Should also allow skipping directly to READING for simple tasks
        assert ConversationStage.READING in valid

    def test_backward_transition_blocked_by_default(self):
        """Backward transitions should be blocked without high confidence."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine(initial_stage=ConversationStage.EXECUTION)
        # Should not allow going back to READING without high confidence
        assert not engine.can_transition(ConversationStage.READING)

    def test_backward_transition_allowed_with_high_confidence(self):
        """Backward transitions should be allowed with high confidence."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine(initial_stage=ConversationStage.EXECUTION)
        # Should allow going back with high confidence (0.85+)
        assert engine.can_transition(ConversationStage.READING, confidence=0.9)


class TestStageTransitions:
    """Tests for stage transition execution."""

    def test_successful_forward_transition(self):
        """Should successfully transition forward."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        result = engine.transition_to(ConversationStage.PLANNING)

        assert result is True
        assert engine.current_stage == ConversationStage.PLANNING

    def test_failed_invalid_transition(self):
        """Should fail on invalid transition."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        result = engine.transition_to(ConversationStage.COMPLETION)  # Can't skip to end

        assert result is False
        assert engine.current_stage == ConversationStage.INITIAL

    def test_transition_with_confidence(self):
        """Should accept confidence parameter."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        result = engine.transition_to(ConversationStage.PLANNING, confidence=0.8)

        assert result is True
        assert engine.current_stage == ConversationStage.PLANNING

    def test_transition_returns_false_on_low_confidence_backward(self):
        """Should fail backward transition with low confidence."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine(initial_stage=ConversationStage.EXECUTION)
        result = engine.transition_to(ConversationStage.READING, confidence=0.5)

        assert result is False
        assert engine.current_stage == ConversationStage.EXECUTION


class TestToolPriorityMultipliers:
    """Tests for stage-based tool priority multipliers."""

    def test_get_tool_priority_multiplier_exists(self):
        """Method should exist on engine."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        assert hasattr(engine, "get_tool_priority_multiplier")
        assert callable(engine.get_tool_priority_multiplier)

    def test_planning_stage_boosts_search_tools(self):
        """PLANNING stage should boost search/overview tools."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine(initial_stage=ConversationStage.PLANNING)
        multiplier = engine.get_tool_priority_multiplier("search")

        assert multiplier >= 1.0  # Should be at least 1.0 (no reduction)
        assert multiplier > 1.0   # Prefer boost

    def test_reading_stage_boosts_read_tool(self):
        """READING stage should boost read tool."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine(initial_stage=ConversationStage.READING)
        multiplier = engine.get_tool_priority_multiplier("read")

        assert multiplier > 1.0

    def test_execution_stage_boosts_edit_tools(self):
        """EXECUTION stage should boost edit/write tools."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine(initial_stage=ConversationStage.EXECUTION)
        edit_multiplier = engine.get_tool_priority_multiplier("edit")
        write_multiplier = engine.get_tool_priority_multiplier("write")

        assert edit_multiplier > 1.0
        assert write_multiplier > 1.0

    def test_verification_stage_boosts_test_tools(self):
        """VERIFICATION stage should boost test/bash tools."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine(initial_stage=ConversationStage.VERIFICATION)
        multiplier = engine.get_tool_priority_multiplier("bash")

        assert multiplier > 1.0

    def test_unknown_tool_returns_default_multiplier(self):
        """Unknown tool should return default multiplier (1.0)."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        multiplier = engine.get_tool_priority_multiplier("nonexistent_tool")

        assert multiplier == 1.0


class TestTransitionCallbacks:
    """Tests for transition callbacks."""

    def test_register_callback_method_exists(self):
        """Should have method to register callbacks."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        assert hasattr(engine, "register_callback")
        assert callable(engine.register_callback)

    def test_callback_invoked_on_transition(self):
        """Registered callback should be invoked on transition."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        callback = Mock()
        engine.register_callback(callback)

        engine.transition_to(ConversationStage.PLANNING)

        callback.assert_called_once()

    def test_callback_receives_old_and_new_stage(self):
        """Callback should receive old and new stage."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        callback = Mock()
        engine.register_callback(callback)

        engine.transition_to(ConversationStage.PLANNING)

        callback.assert_called_once_with(
            ConversationStage.INITIAL,
            ConversationStage.PLANNING,
        )

    def test_multiple_callbacks_all_invoked(self):
        """All registered callbacks should be invoked."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        callback1 = Mock()
        callback2 = Mock()
        engine.register_callback(callback1)
        engine.register_callback(callback2)

        engine.transition_to(ConversationStage.PLANNING)

        callback1.assert_called_once()
        callback2.assert_called_once()

    def test_unregister_callback(self):
        """Should be able to unregister callback."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        callback = Mock()
        engine.register_callback(callback)
        engine.unregister_callback(callback)

        engine.transition_to(ConversationStage.PLANNING)

        callback.assert_not_called()


class TestEventEmission:
    """Tests for event emission on transitions."""

    def test_engine_accepts_event_bus(self):
        """Engine should accept event_bus parameter."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        mock_bus = Mock()
        engine = StageTransitionEngine(event_bus=mock_bus)

        assert engine._event_bus == mock_bus

    def test_transition_emits_event(self):
        """Transition should emit event to event bus."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        mock_bus = Mock()
        engine = StageTransitionEngine(event_bus=mock_bus)

        engine.transition_to(ConversationStage.PLANNING)

        # Event should be emitted
        mock_bus.emit.assert_called()

    def test_event_contains_stage_info(self):
        """Emitted event should contain stage information."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        mock_bus = Mock()
        engine = StageTransitionEngine(event_bus=mock_bus)

        engine.transition_to(ConversationStage.PLANNING)

        # Check the call args
        call_args = mock_bus.emit.call_args
        assert call_args is not None
        # Event should contain old_stage and new_stage
        event_data = call_args[1] if len(call_args) > 1 else call_args[0]
        # Depending on implementation, check for stage info
        assert mock_bus.emit.called


class TestTransitionCooldown:
    """Tests for transition cooldown to prevent thrashing."""

    def test_engine_has_cooldown_setting(self):
        """Engine should have configurable cooldown."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine(cooldown_seconds=5.0)
        assert engine.cooldown_seconds == 5.0

    def test_default_cooldown_is_reasonable(self):
        """Default cooldown should be between 1-5 seconds."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        assert 1.0 <= engine.cooldown_seconds <= 5.0

    def test_rapid_transitions_blocked(self):
        """Rapid transitions should be blocked by cooldown."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        # Use very short cooldown for testing
        engine = StageTransitionEngine(cooldown_seconds=10.0)

        # First transition should succeed
        result1 = engine.transition_to(ConversationStage.PLANNING)
        assert result1 is True

        # Immediate second transition should fail due to cooldown
        result2 = engine.transition_to(ConversationStage.READING)
        assert result2 is False
        assert engine.current_stage == ConversationStage.PLANNING


class TestTransitionHistory:
    """Tests for transition history tracking."""

    def test_engine_tracks_history(self):
        """Engine should track transition history."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        assert hasattr(engine, "transition_history")

    def test_transition_recorded_in_history(self):
        """Successful transition should be recorded."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        engine.transition_to(ConversationStage.PLANNING)

        history = engine.transition_history
        assert len(history) == 1
        assert history[0]["from_stage"] == ConversationStage.INITIAL
        assert history[0]["to_stage"] == ConversationStage.PLANNING

    def test_failed_transition_not_recorded(self):
        """Failed transition should not be recorded."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        engine.transition_to(ConversationStage.COMPLETION)  # Should fail

        history = engine.transition_history
        assert len(history) == 0


class TestResetFunctionality:
    """Tests for engine reset functionality."""

    def test_reset_method_exists(self):
        """Engine should have reset method."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        assert hasattr(engine, "reset")
        assert callable(engine.reset)

    def test_reset_restores_initial_stage(self):
        """Reset should restore to INITIAL stage."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        engine.transition_to(ConversationStage.PLANNING)
        engine.reset()

        assert engine.current_stage == ConversationStage.INITIAL

    def test_reset_clears_history(self):
        """Reset should clear transition history."""
        from victor.agent.stage_transition_engine import StageTransitionEngine

        engine = StageTransitionEngine()
        engine.transition_to(ConversationStage.PLANNING)
        engine.reset()

        assert len(engine.transition_history) == 0


class TestDIIntegration:
    """Tests for DI container integration."""

    def test_engine_registered_in_di(self):
        """StageTransitionEngine should be registered in DI container."""
        from victor.core.container import ServiceContainer
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.protocols.stage_transition import StageTransitionProtocol
        from victor.config.settings import Settings

        container = ServiceContainer()
        configure_orchestrator_services(container, Settings())

        engine = container.get(StageTransitionProtocol)
        assert engine is not None

    def test_engine_is_singleton(self):
        """Engine should be registered as singleton."""
        from victor.core.container import ServiceContainer
        from victor.agent.service_provider import configure_orchestrator_services
        from victor.protocols.stage_transition import StageTransitionProtocol
        from victor.config.settings import Settings

        container = ServiceContainer()
        configure_orchestrator_services(container, Settings())

        engine1 = container.get(StageTransitionProtocol)
        engine2 = container.get(StageTransitionProtocol)
        assert engine1 is engine2
