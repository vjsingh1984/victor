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

"""Full integration tests for the Victor observability subsystem.

Tests the complete event flow from:
1. ObservabilityIntegration (entry point)
2. EventBus (Pub/Sub)
3. CQRS Bridge (bidirectional conversion)
4. StateHookManager (state transitions with history)
5. Exporters (JSONL, callbacks)
6. Metrics collection

This demonstrates the Observer, Mediator, and Adapter patterns working together.
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from victor.observability import (
    CallbackExporter,
    CQRSEventAdapter,
    EventBus,
    EventCategory,
    JsonLineExporter,
    LoggingHook,
    MetricsHook,
    ObservabilityIntegration,
    StateHookManager,
    TransitionHistory,
    VictorEvent,
    create_unified_bridge,
)
from victor.core.event_sourcing import (
    EventDispatcher,
    StateChangedEvent,
    ToolCalledEvent,
    ToolResultEvent,
)


class TestFullEventFlow:
    """Tests for the complete event flow through the observability subsystem."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton before each test."""
        EventBus.reset_instance()
        yield
        EventBus.reset_instance()

    def test_event_bus_to_exporter_flow(self):
        """Test events flow from EventBus to exporters."""
        bus = EventBus.get_instance()
        received: List[VictorEvent] = []

        # Set up callback exporter
        exporter = CallbackExporter(lambda e: received.append(e))
        bus.add_exporter(exporter)

        # Publish events
        bus.publish(
            VictorEvent(
                category=EventCategory.TOOL,
                name="read.start",
                data={"path": "/test/file.py"},
            )
        )
        bus.publish(
            VictorEvent(
                category=EventCategory.TOOL,
                name="read.end",
                data={"path": "/test/file.py", "success": True},
            )
        )

        # Verify events were exported
        assert len(received) == 2
        assert received[0].name == "read.start"
        assert received[1].name == "read.end"

    def test_event_bus_subscription_filtering(self):
        """Test category-based subscription filtering."""
        bus = EventBus.get_instance()
        tool_events: List[VictorEvent] = []
        state_events: List[VictorEvent] = []
        all_events: List[VictorEvent] = []

        # Subscribe to specific categories
        bus.subscribe(EventCategory.TOOL, lambda e: tool_events.append(e))
        bus.subscribe(EventCategory.STATE, lambda e: state_events.append(e))
        bus.subscribe_all(lambda e: all_events.append(e))

        # Publish different event types
        bus.publish(VictorEvent(category=EventCategory.TOOL, name="read", data={}))
        bus.publish(VictorEvent(category=EventCategory.STATE, name="transition", data={}))
        bus.publish(VictorEvent(category=EventCategory.MODEL, name="response", data={}))

        # Verify filtering
        assert len(tool_events) == 1
        assert len(state_events) == 1
        assert len(all_events) == 3

    def test_jsonl_export_persistence(self):
        """Test events are persisted to JSONL file."""
        bus = EventBus.get_instance()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            filepath = f.name

        try:
            exporter = JsonLineExporter(filepath)
            bus.add_exporter(exporter)

            # Publish events
            bus.publish(
                VictorEvent(
                    category=EventCategory.TOOL,
                    name="test_event",
                    data={"key": "value"},
                )
            )

            # Flush and verify
            exporter.flush()

            with open(filepath) as f:
                lines = f.readlines()
                assert len(lines) == 1
                event_data = json.loads(lines[0])
                assert event_data["name"] == "test_event"
                assert event_data["data"]["key"] == "value"
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestCQRSBridgeIntegration:
    """Tests for bidirectional CQRS event bridging."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton before each test."""
        EventBus.reset_instance()
        yield
        EventBus.reset_instance()

    def test_observability_to_cqrs_conversion(self):
        """Test VictorEvent -> CQRSEvent conversion."""
        bus = EventBus.get_instance()
        dispatcher = EventDispatcher()
        adapter = CQRSEventAdapter(bus, dispatcher)

        # Test conversion
        victor_event = VictorEvent(
            category=EventCategory.TOOL,
            name="read.start",
            data={"tool_name": "read", "path": "/test"},
        )

        cqrs_event = adapter._convert_to_cqrs_event(victor_event)

        assert cqrs_event is not None
        assert isinstance(cqrs_event, ToolCalledEvent)
        assert cqrs_event.tool_name == "read"

    def test_cqrs_to_observability_conversion(self):
        """Test CQRSEvent -> VictorEvent conversion."""
        bus = EventBus.get_instance()
        dispatcher = EventDispatcher()
        adapter = CQRSEventAdapter(bus, dispatcher)

        # Test conversion
        cqrs_event = StateChangedEvent(
            task_id="test-task",
            from_state="INITIAL",
            to_state="PLANNING",
            reason="User requested",
        )

        victor_event = adapter._convert_to_victor_event(cqrs_event)

        assert victor_event is not None
        assert victor_event.category == EventCategory.STATE
        assert victor_event.name == "StateChangedEvent"

    def test_unified_bridge_context_manager(self):
        """Test UnifiedEventBridge as context manager."""
        bridge = create_unified_bridge(auto_start=False)

        assert not bridge.is_started

        with bridge as b:
            assert b.is_started
            assert b is bridge

        assert not bridge.is_started


class TestStateHookManagerIntegration:
    """Tests for StateHookManager with full event flow."""

    def test_hooks_with_metrics_and_logging(self):
        """Test combined metrics and logging hooks."""
        manager = StateHookManager()

        metrics = MetricsHook()
        logging_hook = LoggingHook(include_context=True)

        manager.add_hook(metrics.create_hook())
        manager.add_hook(logging_hook.create_hook())

        # Fire transitions
        manager.fire_transition("INITIAL", "PLANNING", {"tool": "read"})
        manager.fire_transition("PLANNING", "EXECUTION", {"tool": "write"})
        manager.fire_transition("EXECUTION", "COMPLETION", {})

        # Verify metrics
        stats = metrics.get_stats()
        assert stats["total_transitions"] == 3
        assert "INITIAL->PLANNING" in stats["transitions"]
        assert stats["stage_entries"]["EXECUTION"] == 1

    def test_history_aware_hooks_detect_patterns(self):
        """Test history-aware hooks can detect patterns."""
        manager = StateHookManager()
        detected_patterns: List[Dict[str, Any]] = []

        @manager.on_transition_with_history
        def detect_pattern(old: str, new: str, ctx: Dict, history: TransitionHistory) -> None:
            detected_patterns.append(
                {
                    "transition": f"{old}->{new}",
                    "sequence": history.get_stage_sequence(),
                    "has_cycle": history.has_cycle(),
                    "visit_count": history.get_stage_visit_count(new),
                }
            )

        manager.fire_transition("INITIAL", "PLANNING", {})
        manager.fire_transition("PLANNING", "EXECUTION", {})
        manager.fire_transition("EXECUTION", "PLANNING", {})  # Cycle back

        assert len(detected_patterns) == 3
        assert detected_patterns[2]["has_cycle"] is True
        assert detected_patterns[2]["visit_count"] == 2

    def test_history_with_event_bus_integration(self):
        """Test StateHookManager events flow to EventBus."""
        EventBus.reset_instance()
        bus = EventBus.get_instance()
        received_events: List[VictorEvent] = []

        bus.subscribe(EventCategory.STATE, lambda e: received_events.append(e))

        manager = StateHookManager()

        @manager.on_transition
        def emit_to_bus(old: str, new: str, ctx: Dict) -> None:
            bus.publish(
                VictorEvent(
                    category=EventCategory.STATE,
                    name="transition",
                    data={"old_stage": old, "new_stage": new, **ctx},
                )
            )

        manager.fire_transition("INITIAL", "PLANNING", {"confidence": 0.9})
        manager.fire_transition("PLANNING", "EXECUTION", {"confidence": 0.85})

        assert len(received_events) == 2
        assert received_events[0].data["old_stage"] == "INITIAL"
        assert received_events[1].data["new_stage"] == "EXECUTION"


class TestObservabilityIntegrationFullStack:
    """Tests for the full ObservabilityIntegration stack."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton before each test."""
        EventBus.reset_instance()
        yield
        EventBus.reset_instance()

    def test_integration_with_cqrs_bridge(self):
        """Test ObservabilityIntegration with CQRS bridge enabled."""
        integration = ObservabilityIntegration(
            session_id="test-session",
            enable_cqrs_bridge=True,
        )

        assert integration.cqrs_bridge is not None
        assert integration.cqrs_bridge.is_started
        assert integration.event_bus is not None

        # Clean up
        integration.disable_cqrs_bridge()
        assert integration.cqrs_bridge is None

    def test_tool_event_lifecycle(self):
        """Test complete tool event lifecycle through ObservabilityIntegration."""
        integration = ObservabilityIntegration()
        bus = integration.event_bus
        received: List[VictorEvent] = []

        bus.subscribe(EventCategory.TOOL, lambda e: received.append(e))

        # Simulate tool lifecycle
        integration.on_tool_start("read", {"path": "/test"}, "tool-1")
        time.sleep(0.01)  # Small delay for duration measurement
        integration.on_tool_end("read", {"content": "..."}, success=True, tool_id="tool-1")

        # Verify events
        assert len(received) == 2
        assert received[0].name == "read.start"
        assert received[1].name == "read.end"

    def test_session_lifecycle_events(self):
        """Test session start/end events."""
        integration = ObservabilityIntegration()
        bus = integration.event_bus
        received: List[VictorEvent] = []

        bus.subscribe(EventCategory.LIFECYCLE, lambda e: received.append(e))

        integration.on_session_start({"model": "test-model"})
        integration.on_session_end(tool_calls=5, duration_seconds=10.5, success=True)

        assert len(received) == 2
        assert received[0].name == "session.start"
        assert received[1].name == "session.end"
        assert received[1].data["tool_calls"] == 5

    def test_error_event_tracking(self):
        """Test error events are tracked."""
        integration = ObservabilityIntegration()
        bus = integration.event_bus
        received: List[VictorEvent] = []

        bus.subscribe(EventCategory.ERROR, lambda e: received.append(e))

        error = ValueError("Test error")
        integration.on_error(error, context={"tool": "read"}, recoverable=True)

        assert len(received) == 1
        # Error message could be in 'error' or 'message' key depending on implementation
        error_data = received[0].data
        error_str = str(error_data.get("error", "")) + str(error_data.get("message", ""))
        assert "Test error" in error_str or "ValueError" in str(error_data)


class TestEndToEndScenarios:
    """End-to-end scenarios demonstrating complete integration."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton before each test."""
        EventBus.reset_instance()
        yield
        EventBus.reset_instance()

    def test_complete_conversation_flow(self):
        """Simulate a complete conversation with all observability features."""
        # Setup
        integration = ObservabilityIntegration(
            session_id="conv-123",
            enable_cqrs_bridge=True,
        )
        bus = integration.event_bus

        # Set up state machine with hooks
        state_manager = StateHookManager()
        metrics_hook = MetricsHook()
        state_manager.add_hook(metrics_hook.create_hook())

        # Collectors
        all_events: List[VictorEvent] = []
        bus.subscribe_all(lambda e: all_events.append(e))

        # Simulate conversation
        integration.on_session_start({"model": "claude-3", "provider": "anthropic"})

        # State transitions
        state_manager.fire_transition("INITIAL", "PLANNING", {"query": "add tests"})
        integration.on_model_request("anthropic", "claude-3", message_count=1, tool_count=10)
        integration.on_model_response(
            "anthropic", "claude-3", tokens_used=500, tool_calls=2, latency_ms=1200
        )

        # Tool execution
        state_manager.fire_transition("PLANNING", "READING", {})
        integration.on_tool_start("read", {"path": "test.py"}, "tool-1")
        integration.on_tool_end("read", {"content": "code..."}, success=True, tool_id="tool-1")

        state_manager.fire_transition("READING", "EXECUTION", {})
        integration.on_tool_start("write", {"path": "test.py", "content": "new code"}, "tool-2")
        integration.on_tool_end("write", {"written": True}, success=True, tool_id="tool-2")

        state_manager.fire_transition("EXECUTION", "COMPLETION", {})
        integration.on_session_end(tool_calls=2, duration_seconds=5.0, success=True)

        # Verify full event flow
        event_names = [e.name for e in all_events]
        assert "session.start" in event_names
        assert "read.start" in event_names
        assert "read.end" in event_names
        assert "write.start" in event_names
        assert "write.end" in event_names
        assert "session.end" in event_names

        # Verify state metrics
        stats = metrics_hook.get_stats()
        assert stats["total_transitions"] == 4
        assert stats["stage_entries"]["COMPLETION"] == 1

        # Verify history tracking
        assert len(state_manager.history) == 4
        assert state_manager.history.current_stage == "COMPLETION"
        assert state_manager.history.get_stage_sequence() == [
            "INITIAL",
            "PLANNING",
            "READING",
            "EXECUTION",
            "COMPLETION",
        ]

        # Clean up
        integration.disable_cqrs_bridge()

    def test_error_recovery_scenario(self):
        """Test error handling and recovery flow."""
        integration = ObservabilityIntegration()

        state_manager = StateHookManager()
        error_counts: Dict[str, int] = {}

        @state_manager.on_transition_with_history
        def track_errors(old: str, new: str, ctx: Dict, history: TransitionHistory) -> None:
            if ctx.get("error"):
                error_counts[new] = error_counts.get(new, 0) + 1

        # Simulate error and retry
        state_manager.fire_transition("EXECUTION", "ERROR", {"error": "Tool failed"})
        integration.on_error(RuntimeError("Tool failed"), {"tool": "write"}, recoverable=True)

        # Retry
        state_manager.fire_transition("ERROR", "EXECUTION", {"retry": True})
        state_manager.fire_transition("EXECUTION", "COMPLETION", {})

        assert error_counts["ERROR"] == 1
        assert state_manager.history.has_visited("ERROR")
        assert state_manager.history.current_stage == "COMPLETION"
