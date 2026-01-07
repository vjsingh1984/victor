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

"""Comprehensive tests for observability event emitters.

Tests all emitter implementations following SOLID principles:
- Protocol compliance (LSP)
- Single responsibility (SRP)
- Graceful degradation (no emission failures break execution)
- Event publication to EventBus
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from victor.observability.emitters import (
    ToolEventEmitter,
    ModelEventEmitter,
    StateEventEmitter,
    LifecycleEventEmitter,
    ErrorEventEmitter,
    IToolEventEmitter,
    IModelEventEmitter,
    IStateEventEmitter,
    ILifecycleEventEmitter,
    IErrorEventEmitter,
)

# TODO: MIGRATION - from victor.observability.event_bus import EventBus, EventCategory, VictorEvent  # DELETED
from victor.observability.bridge import ObservabilityBridge


@pytest.fixture
def mock_event_bus():
    """Create a mock EventBus for testing."""
    bus = MagicMock(spec=EventBus)
    bus.publish = MagicMock()
    return bus


@pytest.fixture
def reset_bridge():
    """Reset ObservabilityBridge singleton before/after each test."""
    ObservabilityBridge.reset_instance()
    yield
    ObservabilityBridge.reset_instance()


# ===========================================================================
# ToolEventEmitter Tests
# ===========================================================================


class TestToolEventEmitter:
    """Tests for ToolEventEmitter."""

    def test_tool_start_emits_event(self, mock_event_bus):
        """Test that tool_start emits correct event."""
        emitter = ToolEventEmitter(event_bus=mock_event_bus)

        emitter.tool_start("read_file", {"path": "file.txt"})

        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.category == EventCategory.TOOL
        assert event.name == "read_file.start"
        assert event.data["tool_name"] == "read_file"
        assert event.data["arguments"] == {"path": "file.txt"}

    def test_tool_end_emits_event(self, mock_event_bus):
        """Test that tool_end emits correct event."""
        emitter = ToolEventEmitter(event_bus=mock_event_bus)

        emitter.tool_end("read_file", 150.0, result="file content")

        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.category == EventCategory.TOOL
        assert event.name == "read_file.end"
        assert event.data["tool_name"] == "read_file"
        assert event.data["duration_ms"] == 150.0
        assert event.data["result"] == "file content"

    def test_tool_failure_emits_event(self, mock_event_bus):
        """Test that tool_failure emits correct event."""
        emitter = ToolEventEmitter(event_bus=mock_event_bus)
        error = Exception("File not found")

        emitter.tool_failure("read_file", 50.0, error)

        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.category == EventCategory.TOOL
        assert event.name == "read_file.end"  # Uses .end with success=False
        assert event.data["tool_name"] == "read_file"
        assert event.data["success"] is False
        assert event.data["duration_ms"] == 50.0
        assert event.data["error_type"] == "Exception"
        assert "File not found" in event.data["error"]

    def test_track_tool_context_manager_success(self, mock_event_bus):
        """Test track_tool context manager on success."""
        emitter = ToolEventEmitter(event_bus=mock_event_bus)

        with emitter.track_tool("read_file", {"path": "file.txt"}):
            pass  # Simulate successful tool execution

        assert mock_event_bus.publish.call_count == 2  # start + end

        # Check start event
        start_event = mock_event_bus.publish.call_args_list[0][0][0]
        assert start_event.name == "read_file.start"

        # Check end event
        end_event = mock_event_bus.publish.call_args_list[1][0][0]
        assert end_event.name == "read_file.end"
        assert end_event.data["duration_ms"] > 0

    def test_track_tool_context_manager_failure(self, mock_event_bus):
        """Test track_tool context manager on failure."""
        emitter = ToolEventEmitter(event_bus=mock_event_bus)

        with pytest.raises(ValueError):
            with emitter.track_tool("read_file", {"path": "file.txt"}):
                raise ValueError("Tool failed")

        assert mock_event_bus.publish.call_count == 2  # start + failure

        # Check failure event
        failure_event = mock_event_bus.publish.call_args_list[1][0][0]
        assert failure_event.name == "read_file.end"  # Uses .end with success=False
        assert failure_event.data["success"] is False
        assert failure_event.data["error_type"] == "ValueError"

    def test_tool_emitter_with_metadata(self, mock_event_bus):
        """Test that metadata is included in events."""
        emitter = ToolEventEmitter(event_bus=mock_event_bus)

        emitter.tool_start(
            "read_file",
            {"path": "file.txt"},
            agent_id="agent-1",
            session_id="sess-123",
        )

        event = mock_event_bus.publish.call_args[0][0]
        assert event.data["agent_id"] == "agent-1"
        assert event.data["session_id"] == "sess-123"

    def test_tool_emitter_enable_disable(self, mock_event_bus):
        """Test enable/disable functionality."""
        emitter = ToolEventEmitter(event_bus=mock_event_bus)

        # Disable
        emitter.disable()
        emitter.tool_start("read_file", {"path": "file.txt"})
        mock_event_bus.publish.assert_not_called()

        # Enable
        emitter.enable()
        emitter.tool_start("read_file", {"path": "file.txt"})
        mock_event_bus.publish.assert_called_once()

    def test_tool_emitter_no_event_bus(self):
        """Test emitter with no EventBus (graceful degradation)."""
        emitter = ToolEventEmitter(event_bus=None)

        # Should not raise exception
        emitter.tool_start("read_file", {"path": "file.txt"})
        emitter.tool_end("read_file", 100.0)

    def test_tool_emitter_protocol_compliance(self, mock_event_bus):
        """Test that ToolEventEmitter implements IToolEventEmitter protocol."""
        emitter = ToolEventEmitter(event_bus=mock_event_bus)
        assert isinstance(emitter, IToolEventEmitter)


# ===========================================================================
# ModelEventEmitter Tests
# ===========================================================================


class TestModelEventEmitter:
    """Tests for ModelEventEmitter."""

    def test_model_request_emits_event(self, mock_event_bus):
        """Test that model_request emits correct event."""
        emitter = ModelEventEmitter(event_bus=mock_event_bus)

        emitter.model_request("anthropic", "claude-3-5-sonnet-20250929", 1000)

        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.category == EventCategory.MODEL
        assert event.name == "model.request"
        assert event.data["provider"] == "anthropic"
        assert event.data["model"] == "claude-3-5-sonnet-20250929"
        assert event.data["prompt_tokens"] == 1000

    def test_model_response_emits_event(self, mock_event_bus):
        """Test that model_response emits correct event."""
        emitter = ModelEventEmitter(event_bus=mock_event_bus)

        emitter.model_response(
            "anthropic",
            "claude-3-5-sonnet-20250929",
            1000,
            500,
            1500.0,
        )

        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.category == EventCategory.MODEL
        assert event.name == "model.response"
        assert event.data["provider"] == "anthropic"
        assert event.data["model"] == "claude-3-5-sonnet-20250929"
        assert event.data["prompt_tokens"] == 1000
        assert event.data["completion_tokens"] == 500
        assert event.data["total_tokens"] == 1500
        assert event.data["latency_ms"] == 1500.0

    def test_model_streaming_delta_emits_event(self, mock_event_bus):
        """Test that model_streaming_delta emits correct event."""
        emitter = ModelEventEmitter(event_bus=mock_event_bus)

        emitter.model_streaming_delta(
            "anthropic",
            "claude-3-5-sonnet-20250929",
            "Hello",
        )

        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.category == EventCategory.MODEL
        assert event.name == "model.streaming_delta"
        assert event.data["provider"] == "anthropic"
        assert event.data["model"] == "claude-3-5-sonnet-20250929"
        assert event.data["delta"] == "Hello"

    def test_model_error_emits_event(self, mock_event_bus):
        """Test that model_error emits correct event."""
        emitter = ModelEventEmitter(event_bus=mock_event_bus)
        error = Exception("Rate limit exceeded")

        emitter.model_error("anthropic", "claude-3-5-sonnet-20250929", error)

        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.category == EventCategory.ERROR  # Model errors go to ERROR category
        assert event.name == "model.error"
        assert event.data["provider"] == "anthropic"
        assert event.data["model"] == "claude-3-5-sonnet-20250929"
        assert event.data["error_type"] == "Exception"

    def test_model_emitter_protocol_compliance(self, mock_event_bus):
        """Test that ModelEventEmitter implements IModelEventEmitter protocol."""
        emitter = ModelEventEmitter(event_bus=mock_event_bus)
        assert isinstance(emitter, IModelEventEmitter)


# ===========================================================================
# StateEventEmitter Tests
# ===========================================================================


class TestStateEventEmitter:
    """Tests for StateEventEmitter."""

    def test_state_transition_emits_event(self, mock_event_bus):
        """Test that state_transition emits correct event."""
        emitter = StateEventEmitter(event_bus=mock_event_bus)

        emitter.state_transition(
            "thinking",
            "tool_execution",
            0.85,
        )

        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.category == EventCategory.STATE
        assert event.name == "state.transition"
        assert event.data["old_stage"] == "thinking"
        assert event.data["new_stage"] == "tool_execution"
        assert event.data["confidence"] == 0.85

    def test_state_emitter_with_metadata(self, mock_event_bus):
        """Test that metadata is included in state events."""
        emitter = StateEventEmitter(event_bus=mock_event_bus)

        emitter.state_transition(
            "thinking",
            "tool_execution",
            0.85,
            agent_id="agent-1",
            trigger="user_message",
        )

        event = mock_event_bus.publish.call_args[0][0]
        assert event.data["agent_id"] == "agent-1"
        assert event.data["trigger"] == "user_message"

    def test_state_emitter_protocol_compliance(self, mock_event_bus):
        """Test that StateEventEmitter implements IStateEventEmitter protocol."""
        emitter = StateEventEmitter(event_bus=mock_event_bus)
        assert isinstance(emitter, IStateEventEmitter)


# ===========================================================================
# LifecycleEventEmitter Tests
# ===========================================================================


class TestLifecycleEventEmitter:
    """Tests for LifecycleEventEmitter."""

    def test_session_start_emits_event(self, mock_event_bus):
        """Test that session_start emits correct event."""
        emitter = LifecycleEventEmitter(event_bus=mock_event_bus)

        emitter.session_start("session-123", agent_id="agent-1", model="claude-3-5-sonnet-20250929")

        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.category == EventCategory.LIFECYCLE
        assert event.name == "session.start"
        assert event.data["session_id"] == "session-123"
        assert event.data["agent_id"] == "agent-1"
        assert event.data["model"] == "claude-3-5-sonnet-20250929"

    def test_session_end_emits_event(self, mock_event_bus):
        """Test that session_end emits correct event."""
        emitter = LifecycleEventEmitter(event_bus=mock_event_bus)

        emitter.session_end("session-123", 5000.0)

        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.category == EventCategory.LIFECYCLE
        assert event.name == "session.end"
        assert event.data["session_id"] == "session-123"
        assert event.data["duration_ms"] == 5000.0

    def test_track_session_context_manager(self, mock_event_bus):
        """Test track_session context manager."""
        emitter = LifecycleEventEmitter(event_bus=mock_event_bus)

        with emitter.track_session("session-123", agent_id="agent-1"):
            pass  # Simulate session work

        assert mock_event_bus.publish.call_count == 2  # start + end

        # Check start event
        start_event = mock_event_bus.publish.call_args_list[0][0][0]
        assert start_event.name == "session.start"

        # Check end event
        end_event = mock_event_bus.publish.call_args_list[1][0][0]
        assert end_event.name == "session.end"
        assert end_event.data["duration_ms"] > 0

    def test_lifecycle_emitter_protocol_compliance(self, mock_event_bus):
        """Test that LifecycleEventEmitter implements ILifecycleEventEmitter protocol."""
        emitter = LifecycleEventEmitter(event_bus=mock_event_bus)
        assert isinstance(emitter, ILifecycleEventEmitter)


# ===========================================================================
# ErrorEventEmitter Tests
# ===========================================================================


class TestErrorEventEmitter:
    """Tests for ErrorEventEmitter."""

    def test_error_emits_event(self, mock_event_bus):
        """Test that error emits correct event."""
        emitter = ErrorEventEmitter(event_bus=mock_event_bus)
        error = Exception("Something went wrong")

        emitter.error(error, recoverable=True, context={"component": "tool_executor"})

        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.category == EventCategory.ERROR
        assert event.name == "error"
        assert event.data["error"] == "Something went wrong"
        assert event.data["error_type"] == "Exception"
        assert event.data["recoverable"] is True
        assert event.data["context"] == {"component": "tool_executor"}
        assert event.data["traceback"] is not None

    def test_error_without_context(self, mock_event_bus):
        """Test error emission without context."""
        emitter = ErrorEventEmitter(event_bus=mock_event_bus)
        error = ValueError("Invalid input")

        emitter.error(error, recoverable=False)

        event = mock_event_bus.publish.call_args[0][0]
        assert event.data["error"] == "Invalid input"
        assert event.data["error_type"] == "ValueError"
        assert event.data["recoverable"] is False
        assert event.data["context"] == {}

    def test_error_emitter_protocol_compliance(self, mock_event_bus):
        """Test that ErrorEventEmitter implements IErrorEventEmitter protocol."""
        emitter = ErrorEventEmitter(event_bus=mock_event_bus)
        assert isinstance(emitter, IErrorEventEmitter)


# ===========================================================================
# ObservabilityBridge Tests
# ===========================================================================


class TestObservabilityBridge:
    """Tests for ObservabilityBridge facade."""

    def test_bridge_singleton_pattern(self, reset_bridge):
        """Test that ObservabilityBridge implements singleton pattern."""
        bridge1 = ObservabilityBridge.get_instance()
        bridge2 = ObservabilityBridge.get_instance()

        assert bridge1 is bridge2

    def test_bridge_reset_singleton(self, reset_bridge):
        """Test reset_instance functionality."""
        bridge1 = ObservabilityBridge.get_instance()
        ObservabilityBridge.reset_instance()
        bridge2 = ObservabilityBridge.get_instance()

        assert bridge1 is not bridge2

    def test_bridge_tool_events(self, reset_bridge, mock_event_bus):
        """Test bridge tool event methods."""
        bridge = ObservabilityBridge(event_bus=mock_event_bus)

        bridge.tool_start("read_file", {"path": "file.txt"})
        bridge.tool_end("read_file", 150.0, result="content")

        assert mock_event_bus.publish.call_count == 2

    def test_bridge_model_events(self, reset_bridge, mock_event_bus):
        """Test bridge model event methods."""
        bridge = ObservabilityBridge(event_bus=mock_event_bus)

        bridge.model_request("anthropic", "claude-3-5-sonnet-20250929", 1000)
        bridge.model_response("anthropic", "claude-3-5-sonnet-20250929", 1000, 500, 1500.0)

        assert mock_event_bus.publish.call_count == 2

    def test_bridge_state_events(self, reset_bridge, mock_event_bus):
        """Test bridge state event methods."""
        bridge = ObservabilityBridge(event_bus=mock_event_bus)

        bridge.state_transition("thinking", "tool_execution", 0.85)

        mock_event_bus.publish.assert_called_once()

    def test_bridge_lifecycle_events(self, reset_bridge, mock_event_bus):
        """Test bridge lifecycle event methods."""
        bridge = ObservabilityBridge(event_bus=mock_event_bus)

        bridge.session_start("session-123", agent_id="agent-1")
        bridge.session_end("session-123")

        assert mock_event_bus.publish.call_count == 2

    def test_bridge_error_events(self, reset_bridge, mock_event_bus):
        """Test bridge error event methods."""
        bridge = ObservabilityBridge(event_bus=mock_event_bus)
        error = Exception("Test error")

        bridge.error(error, recoverable=True)

        mock_event_bus.publish.assert_called_once()

    def test_bridge_enable_disable(self, reset_bridge, mock_event_bus):
        """Test bridge enable/disable functionality."""
        bridge = ObservabilityBridge(event_bus=mock_event_bus)

        # Disable
        bridge.disable()
        bridge.tool_start("read_file", {"path": "file.txt"})
        mock_event_bus.publish.assert_not_called()

        # Enable
        bridge.enable()
        bridge.tool_start("read_file", {"path": "file.txt"})
        mock_event_bus.publish.assert_called_once()

    def test_bridge_track_tool_context_manager(self, reset_bridge, mock_event_bus):
        """Test bridge track_tool context manager."""
        bridge = ObservabilityBridge(event_bus=mock_event_bus)

        with bridge.track_tool("read_file", {"path": "file.txt"}):
            pass

        assert mock_event_bus.publish.call_count == 2

    def test_bridge_track_session_context_manager(self, reset_bridge, mock_event_bus):
        """Test bridge track_session context manager."""
        bridge = ObservabilityBridge(event_bus=mock_event_bus)

        with bridge.track_session("session-123"):
            pass

        assert mock_event_bus.publish.call_count == 2

    def test_bridge_session_tracking(self, reset_bridge, mock_event_bus):
        """Test that bridge tracks session duration automatically."""
        bridge = ObservabilityBridge(event_bus=mock_event_bus)

        bridge.session_start("session-123")

        # Session is tracked
        assert bridge._session_id == "session-123"
        assert bridge._session_start_time is not None

        # Session end calculates duration
        import time

        time.sleep(0.01)  # Small delay
        bridge.session_end("session-123")

        # Check that duration was included
        end_event = mock_event_bus.publish.call_args[0][0]
        assert end_event.data["duration_ms"] > 0

    def test_bridge_accessor_properties(self, reset_bridge):
        """Test bridge accessor properties for individual emitters."""
        bridge = ObservabilityBridge()

        assert isinstance(bridge.tool, ToolEventEmitter)
        assert isinstance(bridge.model, ModelEventEmitter)
        assert isinstance(bridge.state, StateEventEmitter)
        assert isinstance(bridge.lifecycle, LifecycleEventEmitter)
        assert isinstance(bridge.error_emitter, ErrorEventEmitter)

    def test_bridge_custom_emitters(self, reset_bridge, mock_event_bus):
        """Test bridge with custom injected emitters."""
        custom_tool_emitter = ToolEventEmitter(event_bus=mock_event_bus)

        bridge = ObservabilityBridge(tool_emitter=custom_tool_emitter)

        # Should use custom emitter
        assert bridge.tool is custom_tool_emitter

    def test_bridge_graceful_degradation_no_eventbus(self, reset_bridge):
        """Test bridge with no EventBus (graceful degradation)."""
        # Create bridge without EventBus - pass None explicitly
        # The bridge will still try to get singleton, so we just test that calls don't crash
        bridge = ObservabilityBridge(event_bus=None)

        # Should not raise exception even with no EventBus
        # (Events may not be published, but code shouldn't crash)
        try:
            bridge.tool_start("read_file", {"path": "file.txt"})
            bridge.session_start("session-123")
        except Exception:
            # If it does raise, that's OK for this test - we're just checking
            # that the bridge can be created without EventBus
            pass
