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

"""Integration tests for Victor dashboard and observability system.

Tests all 9 dashboard tabs with real event emission and subscription.
Verifies that events flow correctly from emitters through EventBus to dashboard views.

Dashboard Tabs:
1. Events (EventLogView) - Real-time event log
2. Table (EventTableView) - Categorized event table
3. Tools (ToolExecutionView) - Aggregated tool statistics
4. Verticals (VerticalTraceView) - Vertical integration traces
5. History (HistoryView) - Historical event replay
6. Execution (ExecutionTraceView) - Execution span trees
7. Tool Calls (ToolCallHistoryView) - Detailed tool call history
8. State (StateTransitionView) - State machine transitions
9. Metrics (PerformanceMetricsView) - Performance metrics
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

# TODO: MIGRATION - from victor.observability.event_bus import EventBus, EventCategory, VictorEvent  # DELETED
from victor.observability.bridge import ObservabilityBridge
from victor.observability.emitters import (
    ToolEventEmitter,
    ModelEventEmitter,
    StateEventEmitter,
    LifecycleEventEmitter,
    ErrorEventEmitter,
)


@pytest.fixture
def reset_singletons():
    """Reset all singleton instances before/after tests."""
    EventBus.reset_instance()
    ObservabilityBridge.reset_instance()
    yield
    EventBus.reset_instance()
    ObservabilityBridge.reset_instance()


@pytest.fixture
def event_bus(reset_singletons):
    """Create EventBus instance for testing."""
    return EventBus.get_instance()


@pytest.fixture
def observability_bridge(event_bus):
    """Create ObservabilityBridge instance for testing."""
    return ObservabilityBridge(event_bus=event_bus)


# ===========================================================================
# Test 1: Events Tab (EventLogView)
# ===========================================================================


class TestEventsTabIntegration:
    """Tests for Events tab (EventLogView) - real-time event log."""

    @pytest.mark.asyncio
    async def test_events_tab_receives_all_events(self, observability_bridge, event_bus):
        """Test that Events tab receives and displays all events."""
        events_received = []

        def event_handler(event: VictorEvent):
            events_received.append(event)

        # Subscribe to all events (like Events tab does)
        event_bus.subscribe_all(event_handler)

        # Emit various events
        observability_bridge.tool_start("read_file", {"path": "test.txt"})
        observability_bridge.model_request("anthropic", "claude-3-5-sonnet-20250929", 1000)
        observability_bridge.state_transition("thinking", "tool_execution", 0.85)
        observability_bridge.session_start("session-123")

        # Verify all events were received
        assert len(events_received) == 4
        assert events_received[0].category == EventCategory.TOOL
        assert events_received[1].category == EventCategory.MODEL
        assert events_received[2].category == EventCategory.STATE
        assert events_received[3].category == EventCategory.LIFECYCLE

    @pytest.mark.asyncio
    async def test_events_tab_displays_event_details(self, observability_bridge, event_bus):
        """Test that Events tab displays complete event details."""
        events_received = []

        def event_handler(event: VictorEvent):
            events_received.append(event)

        event_bus.subscribe_all(event_handler)

        # Emit tool event with metadata
        observability_bridge.tool_start(
            "read_file",
            {"path": "test.txt"},
            agent_id="agent-1",
            session_id="sess-123",
        )

        # Verify event details
        assert len(events_received) == 1
        event = events_received[0]
        assert event.category == EventCategory.TOOL
        assert event.name == "read_file.start"
        assert event.data["tool_name"] == "read_file"
        assert event.data["arguments"] == {"path": "test.txt"}
        assert event.data["agent_id"] == "agent-1"
        assert event.data["session_id"] == "sess-123"
        assert event.timestamp is not None


# ===========================================================================
# Test 2: Table Tab (EventTableView)
# ===========================================================================


class TestTableTabIntegration:
    """Tests for Table tab (EventTableView) - categorized event table."""

    @pytest.mark.asyncio
    async def test_table_tab_categorizes_events(self, observability_bridge, event_bus):
        """Test that Table tab correctly categorizes events by type."""
        events_by_category = {cat: [] for cat in EventCategory}

        def category_handler(event: VictorEvent):
            events_by_category[event.category].append(event)

        # Subscribe to each category separately (like Table tab does)
        for category in EventCategory:
            event_bus.subscribe(category, category_handler)

        # Emit various events
        observability_bridge.tool_start("read_file", {"path": "test.txt"})
        observability_bridge.model_request("anthropic", "claude-3-5-sonnet-20250929", 1000)
        observability_bridge.state_transition("thinking", "tool_execution", 0.85)
        observability_bridge.session_start("session-123")
        observability_bridge.error(Exception("test"), recoverable=True)

        # Verify categorization
        assert len(events_by_category[EventCategory.TOOL]) == 1
        assert len(events_by_category[EventCategory.MODEL]) == 1
        assert len(events_by_category[EventCategory.STATE]) == 1
        assert len(events_by_category[EventCategory.LIFECYCLE]) == 1
        assert len(events_by_category[EventCategory.ERROR]) == 1

    @pytest.mark.asyncio
    async def test_table_tab_filters_by_category(self, observability_bridge, event_bus):
        """Test that Table tab can filter events by category."""
        tool_events = []

        def tool_handler(event: VictorEvent):
            tool_events.append(event)

        event_bus.subscribe(EventCategory.TOOL, tool_handler)

        # Emit mixed events
        observability_bridge.tool_start("read_file", {"path": "test.txt"})
        observability_bridge.model_request("anthropic", "claude-3-5-sonnet-20250929", 1000)
        observability_bridge.tool_end("read_file", 150.0)

        # Verify only tool events received
        assert len(tool_events) == 2
        assert all(e.category == EventCategory.TOOL for e in tool_events)


# ===========================================================================
# Test 3: Tools Tab (ToolExecutionView)
# ===========================================================================


class TestToolsTabIntegration:
    """Tests for Tools tab (ToolExecutionView) - aggregated tool statistics."""

    @pytest.mark.asyncio
    async def test_tools_tab_aggregates_stats(self, observability_bridge, event_bus):
        """Test that Tools tab correctly aggregates tool statistics."""
        tool_events = []

        def tool_handler(event: VictorEvent):
            tool_events.append(event)

        event_bus.subscribe(EventCategory.TOOL, tool_handler)

        # Simulate multiple tool executions
        for i in range(5):
            observability_bridge.tool_start("read_file", {"path": f"file{i}.txt"})
            observability_bridge.tool_end("read_file", 100.0 + i * 10)

        # Simulate one failure
        observability_bridge.tool_start("read_file", {"path": "bad.txt"})
        observability_bridge.tool_failure("read_file", 50.0, Exception("File not found"))

        # Verify statistics aggregation
        end_events = [e for e in tool_events if e.name.endswith(".end")]
        assert len(end_events) == 6  # 5 success + 1 failure

        # Count successes and failures
        successes = sum(1 for e in end_events if e.data.get("success", True))
        failures = sum(1 for e in end_events if not e.data.get("success", True))

        assert successes == 5
        assert failures == 1

    @pytest.mark.asyncio
    async def test_tools_tab_calculates_avg_time(self, observability_bridge, event_bus):
        """Test that Tools tab calculates average execution time correctly."""
        tool_events = []

        def tool_handler(event: VictorEvent):
            tool_events.append(event)

        event_bus.subscribe(EventCategory.TOOL, tool_handler)

        # Simulate tool executions with different durations
        durations = [100.0, 200.0, 300.0]
        for duration in durations:
            observability_bridge.tool_start("read_file", {"path": "test.txt"})
            observability_bridge.tool_end("read_file", duration)

        # Calculate average from events
        end_events = [e for e in tool_events if e.name.endswith(".end")]
        total_duration = sum(e.data.get("duration_ms", 0) for e in end_events)
        avg_duration = total_duration / len(end_events)

        assert avg_duration == 200.0  # (100 + 200 + 300) / 3

    @pytest.mark.asyncio
    async def test_tools_tab_tracks_multiple_tools(self, observability_bridge, event_bus):
        """Test that Tools tab tracks statistics for multiple tools separately."""
        tool_events = []

        def tool_handler(event: VictorEvent):
            tool_events.append(event)

        event_bus.subscribe(EventCategory.TOOL, tool_handler)

        # Simulate executions of different tools
        observability_bridge.tool_start("read_file", {"path": "test.txt"})
        observability_bridge.tool_end("read_file", 100.0)

        observability_bridge.tool_start("write_file", {"path": "test.txt"})
        observability_bridge.tool_end("write_file", 200.0)

        observability_bridge.tool_start("search", {"query": "test"})
        observability_bridge.tool_end("search", 150.0)

        # Verify tracking by tool name
        end_events = [e for e in tool_events if e.name.endswith(".end")]
        tool_names = {e.data.get("tool_name") for e in end_events}

        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "search" in tool_names
        assert len(tool_names) == 3


# ===========================================================================
# Test 4: Verticals Tab (VerticalTraceView)
# ===========================================================================


class TestVerticalsTabIntegration:
    """Tests for Verticals tab (VerticalTraceView) - vertical integration traces."""

    @pytest.mark.asyncio
    async def test_verticals_tab_receives_vertical_events(self, observability_bridge, event_bus):
        """Test that Verticals tab receives vertical-specific events."""
        vertical_events = []

        def vertical_handler(event: VictorEvent):
            vertical_events.append(event)

        event_bus.subscribe(EventCategory.VERTICAL, vertical_handler)

        # Simulate vertical events (these would be emitted by vertical plugins)
        # For now, we'll just verify the subscription works
        vertical_event = VictorEvent(
            category=EventCategory.VERTICAL,
            name="test_vertical.init",
            data={"vertical": "coding", "status": "initialized"},
        )
        event_bus.publish(vertical_event)

        assert len(vertical_events) == 1
        assert vertical_events[0].category == EventCategory.VERTICAL
        assert vertical_events[0].data["vertical"] == "coding"

    @pytest.mark.asyncio
    async def test_verticals_tab_filters_by_vertical(self, observability_bridge, event_bus):
        """Test that Verticals tab can filter by specific vertical."""
        vertical_events = []

        def vertical_handler(event: VictorEvent):
            vertical_events.append(event)

        event_bus.subscribe(EventCategory.VERTICAL, vertical_handler)

        # Publish events for different verticals
        coding_event = VictorEvent(
            category=EventCategory.VERTICAL,
            name="coding.plugin_load",
            data={"vertical": "coding"},
        )
        research_event = VictorEvent(
            category=EventCategory.VERTICAL,
            name="research.plugin_load",
            data={"vertical": "research"},
        )

        event_bus.publish(coding_event)
        event_bus.publish(research_event)

        assert len(vertical_events) == 2
        verticals = {e.data.get("vertical") for e in vertical_events}
        assert "coding" in verticals
        assert "research" in verticals


# ===========================================================================
# Test 5: History Tab (HistoryView)
# ===========================================================================


class TestHistoryTabIntegration:
    """Tests for History tab (HistoryView) - historical event replay."""

    @pytest.mark.asyncio
    async def test_history_tab_maintains_chronology(self, observability_bridge, event_bus):
        """Test that History tab maintains event chronology."""
        events_received = []

        def history_handler(event: VictorEvent):
            events_received.append(event)

        event_bus.subscribe_all(history_handler)

        # Emit events in sequence
        import time

        timestamps = []
        for i in range(3):
            observability_bridge.tool_start(f"tool_{i}", {"index": i})
            timestamps.append(events_received[-1].timestamp if events_received else None)
            time.sleep(0.01)  # Small delay

        # Verify chronological order
        assert len(events_received) == 3
        assert events_received[0].timestamp <= events_received[1].timestamp
        assert events_received[1].timestamp <= events_received[2].timestamp

    @pytest.mark.asyncio
    async def test_history_tab_replays_session(self, observability_bridge, event_bus):
        """Test that History tab can replay a session's events."""
        session_events = []

        def session_handler(event: VictorEvent):
            # Filter by session_id
            if event.data.get("session_id") == "test-session":
                session_events.append(event)

        event_bus.subscribe_all(session_handler)

        # Simulate session events
        observability_bridge.session_start("test-session", agent_id="agent-1")
        observability_bridge.tool_start(
            "read_file", {"path": "test.txt"}, session_id="test-session"
        )
        observability_bridge.tool_end("read_file", 100.0, session_id="test-session")
        observability_bridge.session_end("test-session")

        # Verify session events captured
        assert len(session_events) == 4
        assert session_events[0].name == "session.start"
        assert session_events[1].name == "read_file.start"
        assert session_events[2].name == "read_file.end"
        assert session_events[3].name == "session.end"


# ===========================================================================
# Test 6: Execution Tab (ExecutionTraceView)
# ===========================================================================


class TestExecutionTabIntegration:
    """Tests for Execution tab (ExecutionTraceView) - execution span trees."""

    @pytest.mark.asyncio
    async def test_execution_tab_receives_lifecycle_events(self, observability_bridge, event_bus):
        """Test that Execution tab receives lifecycle events for span tracking."""
        lifecycle_events = []

        def lifecycle_handler(event: VictorEvent):
            lifecycle_events.append(event)

        event_bus.subscribe(EventCategory.LIFECYCLE, lifecycle_handler)

        # Emit lifecycle events
        observability_bridge.session_start("session-123", agent_id="agent-1")
        observability_bridge.session_end("session-123")

        assert len(lifecycle_events) == 2
        assert lifecycle_events[0].name == "session.start"
        assert lifecycle_events[1].name == "session.end"

    @pytest.mark.asyncio
    async def test_execution_tab_tracks_duration(self, observability_bridge, event_bus):
        """Test that Execution tab correctly tracks session duration."""
        lifecycle_events = []

        def lifecycle_handler(event: VictorEvent):
            lifecycle_events.append(event)

        event_bus.subscribe(EventCategory.LIFECYCLE, lifecycle_handler)

        # Start session
        observability_bridge.session_start("session-123")

        import time

        time.sleep(0.05)  # 50ms delay

        # End session
        observability_bridge.session_end("session-123")

        # Verify duration was tracked
        assert len(lifecycle_events) == 2
        end_event = lifecycle_events[1]
        assert "duration_ms" in end_event.data
        assert end_event.data["duration_ms"] >= 40  # At least 40ms


# ===========================================================================
# Test 7: Tool Calls Tab (ToolCallHistoryView)
# ===========================================================================


class TestToolCallsTabIntegration:
    """Tests for Tool Calls tab (ToolCallHistoryView) - detailed tool call history."""

    @pytest.mark.asyncio
    async def test_tool_calls_tab_shows_detailed_history(self, observability_bridge, event_bus):
        """Test that Tool Calls tab shows detailed call-by-call history."""
        tool_events = []

        def tool_handler(event: VictorEvent):
            tool_events.append(event)

        event_bus.subscribe(EventCategory.TOOL, tool_handler)

        # Simulate tool calls
        observability_bridge.tool_start(
            "read_file",
            {"path": "test.txt"},
            agent_id="agent-1",
            session_id="sess-123",
        )
        observability_bridge.tool_end(
            "read_file",
            150.0,
            result="file content",
            agent_id="agent-1",
        )

        # Verify detailed history
        assert len(tool_events) == 2

        # Check start event
        start_event = tool_events[0]
        assert start_event.name == "read_file.start"
        assert start_event.data["arguments"] == {"path": "test.txt"}
        assert start_event.data["agent_id"] == "agent-1"
        assert start_event.data["session_id"] == "sess-123"

        # Check end event
        end_event = tool_events[1]
        assert end_event.name == "read_file.end"
        assert end_event.data["duration_ms"] == 150.0
        assert end_event.data["result"] == "file content"

    @pytest.mark.asyncio
    async def test_tool_calls_tab_shows_failures(self, observability_bridge, event_bus):
        """Test that Tool Calls tab shows tool call failures with details."""
        tool_events = []

        def tool_handler(event: VictorEvent):
            tool_events.append(event)

        event_bus.subscribe(EventCategory.TOOL, tool_handler)

        # Simulate failed tool call
        observability_bridge.tool_start("read_file", {"path": "nonexistent.txt"})
        observability_bridge.tool_failure(
            "read_file",
            50.0,
            Exception("File not found"),
            agent_id="agent-1",
        )

        # Verify failure details
        assert len(tool_events) == 2
        failure_event = tool_events[1]
        assert failure_event.name == "read_file.end"
        assert failure_event.data["success"] is False
        assert failure_event.data["error_type"] == "Exception"
        assert "File not found" in failure_event.data["error"]

    @pytest.mark.asyncio
    async def test_tool_calls_tab_maintains_call_order(self, observability_bridge, event_bus):
        """Test that Tool Calls tab maintains chronological call order."""
        tool_events = []

        def tool_handler(event: VictorEvent):
            tool_events.append(event)

        event_bus.subscribe(EventCategory.TOOL, tool_handler)

        # Simulate multiple tool calls
        tools = ["read_file", "search", "write_file"]
        for tool in tools:
            observability_bridge.tool_start(tool, {"arg": "value"})
            observability_bridge.tool_end(tool, 100.0)

        # Verify order
        end_events = [e for e in tool_events if e.name.endswith(".end")]
        tool_names = [e.data.get("tool_name") for e in end_events]

        assert tool_names == tools


# ===========================================================================
# Test 8: State Tab (StateTransitionView)
# ===========================================================================


class TestStateTabIntegration:
    """Tests for State tab (StateTransitionView) - state machine transitions."""

    @pytest.mark.asyncio
    async def test_state_tab_tracks_transitions(self, observability_bridge, event_bus):
        """Test that State tab tracks state transitions correctly."""
        state_events = []

        def state_handler(event: VictorEvent):
            state_events.append(event)

        event_bus.subscribe(EventCategory.STATE, state_handler)

        # Simulate state transitions
        observability_bridge.state_transition("thinking", "tool_execution", 0.85)
        observability_bridge.state_transition("tool_execution", "result_synthesis", 0.92)
        observability_bridge.state_transition("result_synthesis", "thinking", 0.88)

        # Verify transitions
        assert len(state_events) == 3

        # Check first transition
        assert state_events[0].data["old_stage"] == "thinking"
        assert state_events[0].data["new_stage"] == "tool_execution"
        assert state_events[0].data["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_state_tab_shows_metadata(self, observability_bridge, event_bus):
        """Test that State tab includes transition metadata."""
        state_events = []

        def state_handler(event: VictorEvent):
            state_events.append(event)

        event_bus.subscribe(EventCategory.STATE, state_handler)

        # Emit state transition with metadata
        observability_bridge.state_transition(
            "thinking",
            "tool_execution",
            0.85,
            agent_id="agent-1",
            trigger="user_message",
        )

        # Verify metadata
        assert len(state_events) == 1
        assert state_events[0].data["agent_id"] == "agent-1"
        assert state_events[0].data["trigger"] == "user_message"

    @pytest.mark.asyncio
    async def test_state_tab_maintains_transition_sequence(self, observability_bridge, event_bus):
        """Test that State tab maintains transition sequence."""
        state_events = []

        def state_handler(event: VictorEvent):
            state_events.append(event)

        event_bus.subscribe(EventCategory.STATE, state_handler)

        # Simulate state machine flow
        transitions = [
            ("thinking", "tool_execution", 0.85),
            ("tool_execution", "result_synthesis", 0.92),
            ("result_synthesis", "response_generation", 0.95),
        ]

        for old_stage, new_stage, confidence in transitions:
            observability_bridge.state_transition(old_stage, new_stage, confidence)

        # Verify sequence
        assert len(state_events) == 3
        for i, (old, new, conf) in enumerate(transitions):
            assert state_events[i].data["old_stage"] == old
            assert state_events[i].data["new_stage"] == new
            assert state_events[i].data["confidence"] == conf


# ===========================================================================
# Test 9: Metrics Tab (PerformanceMetricsView)
# ===========================================================================


class TestMetricsTabIntegration:
    """Tests for Metrics tab (PerformanceMetricsView) - performance metrics."""

    @pytest.mark.asyncio
    async def test_metrics_tab_aggregates_tool_metrics(self, observability_bridge, event_bus):
        """Test that Metrics tab aggregates tool performance metrics."""
        tool_events = []

        def tool_handler(event: VictorEvent):
            tool_events.append(event)

        event_bus.subscribe(EventCategory.TOOL, tool_handler)

        # Simulate tool executions
        durations = [100.0, 150.0, 200.0, 120.0, 180.0]
        for duration in durations:
            observability_bridge.tool_start("read_file", {"path": "test.txt"})
            observability_bridge.tool_end("read_file", duration)

        # Calculate metrics from events
        end_events = [e for e in tool_events if e.name.endswith(".end")]
        duration_values = [e.data.get("duration_ms", 0) for e in end_events]

        avg_duration = sum(duration_values) / len(duration_values)
        min_duration = min(duration_values)
        max_duration = max(duration_values)

        assert avg_duration == 150.0
        assert min_duration == 100.0
        assert max_duration == 200.0

    @pytest.mark.asyncio
    async def test_metrics_tab_tracks_model_usage(self, observability_bridge, event_bus):
        """Test that Metrics tab tracks model token usage."""
        model_events = []

        def model_handler(event: VictorEvent):
            model_events.append(event)

        event_bus.subscribe(EventCategory.MODEL, model_handler)

        # Simulate model requests
        observability_bridge.model_response(
            "anthropic",
            "claude-3-5-sonnet-20250929",
            1000,  # prompt_tokens
            500,  # completion_tokens
            1500.0,  # latency_ms
        )

        # Verify token tracking
        assert len(model_events) == 1
        event = model_events[0]
        assert event.data["total_tokens"] == 1500
        assert event.data["prompt_tokens"] == 1000
        assert event.data["completion_tokens"] == 500
        assert event.data["latency_ms"] == 1500.0

    @pytest.mark.asyncio
    async def test_metrics_tab_calculates_success_rates(self, observability_bridge, event_bus):
        """Test that Metrics tab calculates success rates correctly."""
        tool_events = []

        def tool_handler(event: VictorEvent):
            tool_events.append(event)

        event_bus.subscribe(EventCategory.TOOL, tool_handler)

        # Simulate mixed success/failure
        for i in range(8):
            observability_bridge.tool_start("read_file", {"path": f"file{i}.txt"})
            if i < 2:
                # 2 failures
                observability_bridge.tool_failure("read_file", 50.0, Exception("Error"))
            else:
                # 6 successes
                observability_bridge.tool_end("read_file", 100.0)

        # Calculate success rate
        end_events = [e for e in tool_events if e.name.endswith(".end")]
        successes = sum(1 for e in end_events if e.data.get("success", True))
        total = len(end_events)
        success_rate = (successes / total * 100) if total > 0 else 0

        assert success_rate == 75.0  # 6/8 = 75%


# ===========================================================================
# Cross-Tab Integration Tests
# ===========================================================================


class TestCrossTabIntegration:
    """Tests for cross-tab integration and data consistency."""

    @pytest.mark.asyncio
    async def test_tabs_receive_same_events(self, observability_bridge, event_bus):
        """Test that different tabs receive the same event data."""
        tool_tab_events = []
        tool_calls_tab_events = []
        metrics_tab_events = []

        # Subscribe to TOOL category from multiple "tabs"
        event_bus.subscribe(EventCategory.TOOL, lambda e: tool_tab_events.append(e))
        event_bus.subscribe(EventCategory.TOOL, lambda e: tool_calls_tab_events.append(e))
        event_bus.subscribe(EventCategory.TOOL, lambda e: metrics_tab_events.append(e))

        # Emit tool event
        observability_bridge.tool_start("read_file", {"path": "test.txt"})

        # Verify all tabs received the same event
        assert len(tool_tab_events) == len(tool_calls_tab_events) == len(metrics_tab_events) == 1
        assert tool_tab_events[0] is tool_calls_tab_events[0] is metrics_tab_events[0]

    @pytest.mark.asyncio
    async def test_event_flow_from_orchestrator_to_tabs(self, observability_bridge, event_bus):
        """Test complete event flow from orchestrator through bridge to all tabs."""
        all_events = []

        def universal_handler(event: VictorEvent):
            all_events.append(event)

        event_bus.subscribe_all(universal_handler)

        # Simulate orchestrator workflow
        session_id = "test-session"

        # 1. Start session
        observability_bridge.session_start(session_id, agent_id="agent-1")

        # 2. Tool execution
        observability_bridge.tool_start("read_file", {"path": "test.txt"}, session_id=session_id)
        observability_bridge.tool_end("read_file", 150.0, session_id=session_id)

        # 3. State transition
        observability_bridge.state_transition("thinking", "tool_execution", 0.85)

        # 4. Model call
        observability_bridge.model_response(
            "anthropic", "claude-3-5-sonnet-20250929", 1000, 500, 1500.0
        )

        # 5. End session
        observability_bridge.session_end(session_id)

        # Verify all events captured
        assert len(all_events) == 6
        assert all_events[0].category == EventCategory.LIFECYCLE  # session start
        assert all_events[1].category == EventCategory.TOOL  # tool start
        assert all_events[2].category == EventCategory.TOOL  # tool end
        assert all_events[3].category == EventCategory.STATE  # state transition
        assert all_events[4].category == EventCategory.MODEL  # model response
        assert all_events[5].category == EventCategory.LIFECYCLE  # session end
