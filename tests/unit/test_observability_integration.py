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

"""Unit tests for observability integration module.

Tests the ObservabilityIntegration mediator class and ToolEventMiddleware.
"""

import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from victor.observability import (
    EventBus,
    EventCategory,
    ObservabilityIntegration,
    ToolEventMiddleware,
    VictorEvent,
    setup_observability,
)


class TestObservabilityIntegration:
    """Tests for ObservabilityIntegration class."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton between tests."""
        EventBus._instance = None
        yield
        EventBus._instance = None

    @pytest.fixture
    def integration(self) -> ObservabilityIntegration:
        """Create ObservabilityIntegration for tests."""
        return ObservabilityIntegration()

    def test_default_event_bus(self, integration: ObservabilityIntegration):
        """Should use singleton EventBus by default."""
        assert integration.event_bus is EventBus.get_instance()

    def test_custom_event_bus(self):
        """Should accept custom EventBus."""
        custom_bus = EventBus()
        integration = ObservabilityIntegration(event_bus=custom_bus)
        assert integration.event_bus is custom_bus

    def test_set_session_id(self, integration: ObservabilityIntegration):
        """set_session_id should update both integration and EventBus."""
        integration.set_session_id("test-session-123")
        assert integration._session_id == "test-session-123"

    def test_session_id_in_constructor(self):
        """Session ID can be set in constructor."""
        integration = ObservabilityIntegration(session_id="constructor-session")
        assert integration._session_id == "constructor-session"


class TestToolEvents:
    """Tests for tool event emission."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton between tests."""
        EventBus._instance = None
        yield
        EventBus._instance = None

    @pytest.fixture
    def integration(self) -> ObservabilityIntegration:
        """Create ObservabilityIntegration for tests."""
        return ObservabilityIntegration()

    def test_tool_start_event(self, integration: ObservabilityIntegration):
        """on_tool_start should emit tool start event."""
        events: List[VictorEvent] = []
        integration.event_bus.subscribe(EventCategory.TOOL, events.append)

        integration.on_tool_start("read", {"file_path": "/tmp/test.txt"}, "tool-123")

        assert len(events) == 1
        assert events[0].name == "read.start"
        assert events[0].data["tool_name"] == "read"
        assert events[0].data["arguments"]["file_path"] == "/tmp/test.txt"

    def test_tool_end_event_success(self, integration: ObservabilityIntegration):
        """on_tool_end should emit tool end event with success."""
        events: List[VictorEvent] = []
        integration.event_bus.subscribe(EventCategory.TOOL, events.append)

        integration.on_tool_start("read", {"file_path": "/tmp/test.txt"}, "tool-123")
        integration.on_tool_end("read", "file contents", success=True, tool_id="tool-123")

        assert len(events) == 2
        assert events[1].name == "read.end"
        assert events[1].data["success"] is True
        assert events[1].data["result"] == "file contents"

    def test_tool_end_event_failure(self, integration: ObservabilityIntegration):
        """on_tool_end should emit error event on failure."""
        tool_events: List[VictorEvent] = []
        error_events: List[VictorEvent] = []

        integration.event_bus.subscribe(EventCategory.TOOL, tool_events.append)
        integration.event_bus.subscribe(EventCategory.ERROR, error_events.append)

        integration.on_tool_start("read", {"file_path": "/nonexistent"}, "tool-456")
        integration.on_tool_end(
            "read",
            result=None,
            success=False,
            tool_id="tool-456",
            error="File not found",
        )

        assert len(tool_events) == 2
        assert tool_events[1].data["success"] is False
        assert len(error_events) == 1
        assert "File not found" in error_events[0].data["error"]

    def test_tool_duration_calculation(self, integration: ObservabilityIntegration):
        """Tool duration should be calculated from start to end."""
        events: List[VictorEvent] = []
        integration.event_bus.subscribe(EventCategory.TOOL, events.append)

        integration.on_tool_start("read", {}, "tool-789")
        time.sleep(0.01)  # Small delay
        integration.on_tool_end("read", "result", tool_id="tool-789")

        assert "duration_ms" in events[1].data
        assert events[1].data["duration_ms"] > 0


class TestModelEvents:
    """Tests for model event emission."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton between tests."""
        EventBus._instance = None
        yield
        EventBus._instance = None

    @pytest.fixture
    def integration(self) -> ObservabilityIntegration:
        """Create ObservabilityIntegration for tests."""
        return ObservabilityIntegration()

    def test_model_request_event(self, integration: ObservabilityIntegration):
        """on_model_request should emit model request event."""
        events: List[VictorEvent] = []
        integration.event_bus.subscribe(EventCategory.MODEL, events.append)

        integration.on_model_request(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            message_count=5,
            tool_count=10,
        )

        assert len(events) == 1
        assert events[0].name == "request"
        assert events[0].data["provider"] == "anthropic"
        assert events[0].data["model"] == "claude-sonnet-4-20250514"
        assert events[0].data["message_count"] == 5

    def test_model_response_event(self, integration: ObservabilityIntegration):
        """on_model_response should emit model response event."""
        events: List[VictorEvent] = []
        integration.event_bus.subscribe(EventCategory.MODEL, events.append)

        integration.on_model_response(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            tokens_used=1500,
            tool_calls=2,
            latency_ms=1234.5,
        )

        assert len(events) == 1
        assert events[0].name == "response"
        assert events[0].data["tokens_used"] == 1500
        assert events[0].data["tool_calls"] == 2
        assert events[0].data["latency_ms"] == 1234.5


class TestLifecycleEvents:
    """Tests for session lifecycle event emission."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton between tests."""
        EventBus._instance = None
        yield
        EventBus._instance = None

    @pytest.fixture
    def integration(self) -> ObservabilityIntegration:
        """Create ObservabilityIntegration for tests."""
        return ObservabilityIntegration()

    def test_session_start_event(self, integration: ObservabilityIntegration):
        """on_session_start should emit lifecycle event."""
        events: List[VictorEvent] = []
        integration.event_bus.subscribe(EventCategory.LIFECYCLE, events.append)

        integration.on_session_start({"user": "test", "mode": "chat"})

        assert len(events) == 1
        assert events[0].name == "session.start"
        assert events[0].data["user"] == "test"

    def test_session_end_event(self, integration: ObservabilityIntegration):
        """on_session_end should emit lifecycle event with stats."""
        events: List[VictorEvent] = []
        integration.event_bus.subscribe(EventCategory.LIFECYCLE, events.append)

        integration.on_session_end(tool_calls=15, duration_seconds=60.5, success=True)

        assert len(events) == 1
        assert events[0].name == "session.end"
        assert events[0].data["tool_calls"] == 15
        assert events[0].data["duration_seconds"] == 60.5
        assert events[0].data["success"] is True


class TestErrorEvents:
    """Tests for error event emission."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton between tests."""
        EventBus._instance = None
        yield
        EventBus._instance = None

    @pytest.fixture
    def integration(self) -> ObservabilityIntegration:
        """Create ObservabilityIntegration for tests."""
        return ObservabilityIntegration()

    def test_error_event(self, integration: ObservabilityIntegration):
        """on_error should emit error event."""
        events: List[VictorEvent] = []
        integration.event_bus.subscribe(EventCategory.ERROR, events.append)

        error = ValueError("Something went wrong")
        integration.on_error(error, context={"operation": "read"}, recoverable=True)

        assert len(events) == 1
        # Event name is the exception class name (from emit_error in event_bus.py)
        assert events[0].name == "ValueError"
        assert "Something went wrong" in events[0].data["message"]
        assert events[0].data["recoverable"] is True


class TestWireOrchestrator:
    """Tests for wiring observability into orchestrator."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton between tests."""
        EventBus._instance = None
        yield
        EventBus._instance = None

    def test_wire_orchestrator_sets_observability(self):
        """wire_orchestrator should set _observability on orchestrator."""
        integration = ObservabilityIntegration()
        mock_orchestrator = MagicMock()
        mock_orchestrator.conversation_state = None

        integration.wire_orchestrator(mock_orchestrator)

        assert mock_orchestrator._observability is integration
        assert "orchestrator" in integration._wired_components

    def test_wire_orchestrator_with_state_machine(self):
        """wire_orchestrator should wire state machine if present."""
        integration = ObservabilityIntegration()
        mock_state_machine = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.conversation_state = mock_state_machine

        integration.wire_orchestrator(mock_orchestrator)

        assert "state_machine" in integration._wired_components


class TestToolEventMiddleware:
    """Tests for ToolEventMiddleware."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton between tests."""
        EventBus._instance = None
        yield
        EventBus._instance = None

    @pytest.fixture
    def middleware(self) -> ToolEventMiddleware:
        """Create ToolEventMiddleware for tests."""
        integration = ObservabilityIntegration()
        return ToolEventMiddleware(integration)

    def test_before_execute(self, middleware: ToolEventMiddleware):
        """before_execute should emit tool start event."""
        events: List[VictorEvent] = []
        middleware._integration.event_bus.subscribe(EventCategory.TOOL, events.append)

        middleware.before_execute("write", {"path": "/tmp/file.txt"}, "tool-abc")

        assert len(events) == 1
        assert events[0].name == "write.start"

    def test_after_execute(self, middleware: ToolEventMiddleware):
        """after_execute should emit tool end event."""
        events: List[VictorEvent] = []
        middleware._integration.event_bus.subscribe(EventCategory.TOOL, events.append)

        middleware.before_execute("write", {"path": "/tmp/file.txt"}, "tool-abc")
        middleware.after_execute("write", "Success", success=True, tool_id="tool-abc")

        assert len(events) == 2
        assert events[1].name == "write.end"
        assert events[1].data["success"] is True


class TestSetupObservability:
    """Tests for setup_observability convenience function."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton between tests."""
        EventBus._instance = None
        yield
        EventBus._instance = None

    def test_setup_observability(self):
        """setup_observability should create and wire integration."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.conversation_state = None

        integration = setup_observability(mock_orchestrator)

        assert isinstance(integration, ObservabilityIntegration)
        assert mock_orchestrator._observability is integration

    def test_setup_observability_with_session_id(self):
        """setup_observability should accept session_id."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.conversation_state = None

        integration = setup_observability(mock_orchestrator, session_id="my-session")

        assert integration._session_id == "my-session"


class TestEventBusIntegration:
    """Integration tests for EventBus with ObservabilityIntegration."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton between tests."""
        EventBus._instance = None
        yield
        EventBus._instance = None

    def test_full_tool_lifecycle(self):
        """Test complete tool execution lifecycle with events."""
        integration = ObservabilityIntegration(session_id="test-lifecycle")
        all_events: List[VictorEvent] = []

        # Subscribe to all categories
        for category in EventCategory:
            integration.event_bus.subscribe(category, all_events.append)

        # Simulate session lifecycle
        integration.on_session_start({"mode": "chat"})
        integration.on_model_request("anthropic", "claude-sonnet-4-20250514", 1, 10)
        integration.on_tool_start("read", {"path": "/tmp/test"}, "tool-1")
        integration.on_tool_end("read", "content", success=True, tool_id="tool-1")
        integration.on_model_response("anthropic", "claude-sonnet-4-20250514", 500, 1, 100.0)
        integration.on_session_end(tool_calls=1, duration_seconds=1.5, success=True)

        # Verify event flow
        assert len(all_events) == 6
        assert all_events[0].category == EventCategory.LIFECYCLE
        assert all_events[1].category == EventCategory.MODEL
        assert all_events[2].category == EventCategory.TOOL
        assert all_events[3].category == EventCategory.TOOL
        assert all_events[4].category == EventCategory.MODEL
        assert all_events[5].category == EventCategory.LIFECYCLE


class TestCQRSBridgeIntegration:
    """Tests for CQRS bridge integration with ObservabilityIntegration."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton between tests."""
        EventBus._instance = None
        yield
        EventBus._instance = None

    def test_cqrs_bridge_disabled_by_default(self):
        """CQRS bridge should be disabled by default."""
        integration = ObservabilityIntegration()
        assert integration.cqrs_bridge is None

    def test_enable_cqrs_bridge_on_init(self):
        """CQRS bridge can be enabled on initialization."""
        integration = ObservabilityIntegration(enable_cqrs_bridge=True)

        assert integration.cqrs_bridge is not None
        assert integration.cqrs_bridge.is_started
        assert "cqrs_bridge" in integration._wired_components

        # Cleanup
        integration.disable_cqrs_bridge()

    def test_enable_cqrs_bridge_method(self):
        """CQRS bridge can be enabled via method."""
        integration = ObservabilityIntegration()

        bridge = integration.enable_cqrs_bridge()

        assert bridge is not None
        assert integration.cqrs_bridge is bridge
        assert bridge.is_started

        # Cleanup
        integration.disable_cqrs_bridge()

    def test_disable_cqrs_bridge(self):
        """CQRS bridge can be disabled."""
        integration = ObservabilityIntegration(enable_cqrs_bridge=True)
        assert integration.cqrs_bridge is not None

        integration.disable_cqrs_bridge()

        assert integration.cqrs_bridge is None
        assert "cqrs_bridge" not in integration._wired_components

    def test_enable_cqrs_bridge_idempotent(self):
        """Enabling bridge multiple times returns same instance."""
        integration = ObservabilityIntegration()

        bridge1 = integration.enable_cqrs_bridge()
        bridge2 = integration.enable_cqrs_bridge()

        assert bridge1 is bridge2

        # Cleanup
        integration.disable_cqrs_bridge()

    def test_setup_observability_with_cqrs_bridge(self):
        """setup_observability should support enable_cqrs_bridge parameter."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.conversation_state = None

        integration = setup_observability(
            mock_orchestrator,
            enable_cqrs_bridge=True,
        )

        assert integration.cqrs_bridge is not None
        assert integration.cqrs_bridge.is_started

        # Cleanup
        integration.disable_cqrs_bridge()

    def test_cqrs_bridge_event_conversion(self):
        """Test that events can be converted through the CQRS bridge."""
        integration = ObservabilityIntegration(enable_cqrs_bridge=True)

        # Get the adapter
        adapter = integration.cqrs_bridge.adapter

        # Test observability -> CQRS conversion
        victor_event = VictorEvent(
            category=EventCategory.TOOL,
            name="read.start",
            data={"path": "/test", "tool_name": "read"},
        )

        cqrs_event = adapter._convert_to_cqrs_event(victor_event)
        assert cqrs_event is not None

        # Cleanup
        integration.disable_cqrs_bridge()

    def test_cqrs_bridge_reverse_conversion(self):
        """Test that CQRS events can be converted to observability format."""
        from victor.core.event_sourcing import ToolCalledEvent

        integration = ObservabilityIntegration(enable_cqrs_bridge=True)

        # Get the adapter
        adapter = integration.cqrs_bridge.adapter

        # Test CQRS -> observability conversion
        cqrs_event = ToolCalledEvent(
            task_id="test-task",
            tool_name="read",
            arguments={"path": "/test"},
        )

        victor_event = adapter._convert_to_victor_event(cqrs_event)
        assert victor_event is not None
        assert victor_event.name == "ToolCalledEvent"
        assert victor_event.category == EventCategory.TOOL

        # Cleanup
        integration.disable_cqrs_bridge()
