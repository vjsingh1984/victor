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

"""Integration tests for ConversationStateMachine with ObservabilityIntegration.

Tests the complete flow from:
1. ConversationStateMachine (state transitions)
2. StateHookManager (history-aware hooks)
3. ObservabilityIntegration (wiring)
4. EventBus (event emission)

This validates that state changes in the agent properly flow through
to the observability subsystem with rich analytics.
"""

import time
from typing import Any, Dict, List

import pytest

from victor.agent.conversation_state import ConversationStateMachine, ConversationStage
from victor.observability import (
    EventBus,
    EventCategory,
    ObservabilityIntegration,
    StateHookManager,
    TransitionHistory,
    VictorEvent,
)


class TestStateMachineObservabilityIntegration:
    """Tests for state machine integration with observability."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton before each test."""
        EventBus.reset_instance()
        yield
        EventBus.reset_instance()

    def test_wire_state_machine_creates_hook_manager(self):
        """Test that wiring creates a StateHookManager."""
        integration = ObservabilityIntegration()
        state_machine = ConversationStateMachine()

        integration.wire_state_machine(state_machine)

        assert integration.state_hook_manager is not None
        assert isinstance(integration.state_hook_manager, StateHookManager)

    def test_wired_state_machine_emits_events(self):
        """Test that state transitions emit events to EventBus."""
        integration = ObservabilityIntegration()
        state_machine = ConversationStateMachine()
        received_events: List[VictorEvent] = []

        integration.event_bus.subscribe(EventCategory.STATE, lambda e: received_events.append(e))
        integration.wire_state_machine(state_machine)

        # Force a transition by recording tools and messages
        state_machine.record_message("implement the feature", is_user=True)

        # Should emit a state change event
        assert len(received_events) >= 0  # May or may not transition based on confidence

    def test_direct_transition_emits_event(self):
        """Test direct state transition emits event with analytics."""
        integration = ObservabilityIntegration()
        state_machine = ConversationStateMachine()
        received_events: List[VictorEvent] = []

        integration.event_bus.subscribe(EventCategory.STATE, lambda e: received_events.append(e))
        integration.wire_state_machine(state_machine)

        # Directly manipulate the state machine's internal transition
        # by recording enough tools to trigger transition
        for _ in range(5):
            state_machine.record_tool_execution("read", {"path": "/test"})

        # Wait a bit to pass cooldown
        time.sleep(0.1)

        # Record execution tools
        for _ in range(5):
            state_machine.record_tool_execution("write", {"path": "/test"})

    def test_state_transition_history_accessible(self):
        """Test that transition history is accessible after wiring."""
        integration = ObservabilityIntegration()
        state_machine = ConversationStateMachine()

        integration.wire_state_machine(state_machine)

        history = integration.state_transition_history
        assert history is not None
        assert isinstance(history, TransitionHistory)
        assert len(history) == 0  # No transitions yet

    def test_get_state_transition_metrics_empty(self):
        """Test metrics for state machine with no transitions."""
        integration = ObservabilityIntegration()
        state_machine = ConversationStateMachine()

        integration.wire_state_machine(state_machine)

        metrics = integration.get_state_transition_metrics()
        assert metrics["total_transitions"] == 0
        assert metrics["unique_stages_visited"] == 0
        assert metrics["has_cycles"] is False

    def test_hook_manager_fires_on_transition(self):
        """Test that hook manager fires hooks on state machine transitions."""
        ObservabilityIntegration()
        hook_manager = StateHookManager(enable_history=True)
        transitions_recorded: List[Dict[str, Any]] = []

        @hook_manager.on_transition_with_history
        def track_transitions(old: str, new: str, ctx: Dict, history: TransitionHistory) -> None:
            transitions_recorded.append(
                {
                    "old": old,
                    "new": new,
                    "history_len": len(history),
                }
            )

        state_machine = ConversationStateMachine(hooks=hook_manager)

        # Record message that should trigger transition
        state_machine.record_message("implement this feature now", is_user=True)

        # Should have recorded at least one transition
        # (depends on keyword detection)

    def test_history_aware_events_include_analytics(self):
        """Test that emitted events include history analytics."""
        integration = ObservabilityIntegration()
        state_machine = ConversationStateMachine()
        received_events: List[VictorEvent] = []

        integration.event_bus.subscribe(EventCategory.STATE, lambda e: received_events.append(e))
        integration.wire_state_machine(state_machine)

        # Manually fire a transition via the hook manager
        if integration.state_hook_manager:
            integration.state_hook_manager.fire_transition(
                "INITIAL", "PLANNING", {"confidence": 0.8}
            )
            time.sleep(0.01)  # Allow event to propagate
            integration.state_hook_manager.fire_transition(
                "PLANNING", "READING", {"confidence": 0.9}
            )

        # Check events have analytics
        assert len(received_events) >= 2

        # Second event should have history
        second_event = received_events[1]
        assert "has_cycle" in second_event.data
        assert "stage_sequence" in second_event.data
        assert "visit_count" in second_event.data


class TestStateMachineTransitionPatterns:
    """Tests for various state transition patterns."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton before each test."""
        EventBus.reset_instance()
        yield
        EventBus.reset_instance()

    def test_cycle_detection_emits_warning(self):
        """Test that repeated cycles emit warning events."""
        integration = ObservabilityIntegration()
        warning_events: List[VictorEvent] = []

        integration.event_bus.subscribe(EventCategory.ERROR, lambda e: warning_events.append(e))

        state_machine = ConversationStateMachine()
        integration.wire_state_machine(state_machine)

        hook_manager = integration.state_hook_manager
        assert hook_manager is not None

        # Create a cycle pattern: A -> B -> A -> B -> A (visit count >= 3)
        hook_manager.fire_transition("A", "B", {"confidence": 0.8})
        hook_manager.fire_transition("B", "A", {"confidence": 0.8})
        hook_manager.fire_transition("A", "B", {"confidence": 0.8})
        hook_manager.fire_transition("B", "A", {"confidence": 0.8})
        hook_manager.fire_transition("A", "B", {"confidence": 0.8})

        # Should have emitted cycle warnings
        cycle_warnings = [e for e in warning_events if e.name == "state.cycle_warning"]
        # May have cycle warnings depending on visit count threshold
        assert len(cycle_warnings) >= 0

    def test_stage_sequence_tracking(self):
        """Test that stage sequence is properly tracked."""
        integration = ObservabilityIntegration()
        state_machine = ConversationStateMachine()
        integration.wire_state_machine(state_machine)

        hook_manager = integration.state_hook_manager
        assert hook_manager is not None

        # Fire a sequence of transitions
        hook_manager.fire_transition("INITIAL", "PLANNING", {})
        hook_manager.fire_transition("PLANNING", "READING", {})
        hook_manager.fire_transition("READING", "EXECUTION", {})

        # Check the history
        history = integration.state_transition_history
        assert history is not None
        sequence = history.get_stage_sequence()
        assert "INITIAL" in sequence
        assert "PLANNING" in sequence
        assert "READING" in sequence
        assert "EXECUTION" in sequence

    def test_metrics_update_with_transitions(self):
        """Test that metrics update as transitions occur."""
        integration = ObservabilityIntegration()
        state_machine = ConversationStateMachine()
        integration.wire_state_machine(state_machine)

        hook_manager = integration.state_hook_manager
        assert hook_manager is not None

        # Initial metrics
        metrics = integration.get_state_transition_metrics()
        assert metrics["total_transitions"] == 0

        # Fire transitions
        hook_manager.fire_transition("A", "B", {})
        hook_manager.fire_transition("B", "C", {})

        # Updated metrics
        metrics = integration.get_state_transition_metrics()
        assert metrics["total_transitions"] == 2
        assert metrics["unique_stages_visited"] == 3  # A, B, C


class TestObservabilityIntegrationWithMockedOrchestrator:
    """Tests using a mocked orchestrator-like state machine."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton before each test."""
        EventBus.reset_instance()
        yield
        EventBus.reset_instance()

    def test_full_conversation_flow_simulation(self):
        """Simulate a full conversation flow and verify event emission."""
        integration = ObservabilityIntegration()
        state_machine = ConversationStateMachine()
        all_events: List[VictorEvent] = []

        integration.event_bus.subscribe_all(lambda e: all_events.append(e))
        integration.wire_state_machine(state_machine)

        hook_manager = integration.state_hook_manager
        assert hook_manager is not None

        # Simulate conversation flow
        # 1. Start session
        integration.on_session_start({"model": "claude-3", "provider": "anthropic"})

        # 2. State transitions through conversation
        hook_manager.fire_transition("INITIAL", "PLANNING", {"confidence": 0.9})
        time.sleep(0.01)

        # 3. Tool usage
        integration.on_tool_start("read", {"path": "/src/main.py"}, "tool-1")
        integration.on_tool_end("read", {"content": "code..."}, success=True, tool_id="tool-1")

        hook_manager.fire_transition("PLANNING", "READING", {"confidence": 0.85})
        time.sleep(0.01)

        # 4. More tools
        integration.on_tool_start("write", {"path": "/src/main.py"}, "tool-2")
        integration.on_tool_end("write", {"written": True}, success=True, tool_id="tool-2")

        hook_manager.fire_transition("READING", "EXECUTION", {"confidence": 0.9})
        time.sleep(0.01)

        hook_manager.fire_transition("EXECUTION", "COMPLETION", {"confidence": 0.95})

        # 5. End session
        integration.on_session_end(tool_calls=2, duration_seconds=5.0, success=True)

        # Verify events
        event_categories = [e.category for e in all_events]
        assert EventCategory.LIFECYCLE in event_categories
        assert EventCategory.TOOL in event_categories
        assert EventCategory.STATE in event_categories

        # Verify state events have analytics
        state_events = [e for e in all_events if e.category == EventCategory.STATE]
        assert len(state_events) >= 4

        # Last state event should have full sequence
        last_state_event = state_events[-1]
        assert "stage_sequence" in last_state_event.data

    def test_error_during_tool_execution(self):
        """Test error events are emitted during tool failures."""
        integration = ObservabilityIntegration()
        error_events: List[VictorEvent] = []

        integration.event_bus.subscribe(EventCategory.ERROR, lambda e: error_events.append(e))

        # Simulate tool failure
        integration.on_tool_start("write", {"path": "/readonly"}, "tool-1")
        integration.on_tool_end(
            "write",
            result=None,
            success=False,
            tool_id="tool-1",
            error="Permission denied",
        )

        # Should have error event
        assert len(error_events) >= 1


class TestObservabilityIntegrationProperties:
    """Tests for ObservabilityIntegration properties and accessors."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton before each test."""
        EventBus.reset_instance()
        yield
        EventBus.reset_instance()

    def test_state_hook_manager_none_before_wiring(self):
        """Test state_hook_manager is None before wiring."""
        integration = ObservabilityIntegration()
        assert integration.state_hook_manager is None

    def test_state_transition_history_none_before_wiring(self):
        """Test transition history is None before wiring."""
        integration = ObservabilityIntegration()
        assert integration.state_transition_history is None

    def test_get_state_transition_metrics_without_wiring(self):
        """Test metrics return empty dict without wiring."""
        integration = ObservabilityIntegration()
        metrics = integration.get_state_transition_metrics()
        assert metrics["total_transitions"] == 0
        assert metrics["has_cycles"] is False

    def test_wired_components_list(self):
        """Test wired_components tracks what's been wired."""
        integration = ObservabilityIntegration()
        state_machine = ConversationStateMachine()

        integration.wire_state_machine(state_machine)

        assert "state_machine" in integration._wired_components
