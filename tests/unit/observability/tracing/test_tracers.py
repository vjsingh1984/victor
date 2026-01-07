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

"""Unit tests for execution and tool call tracers."""

import pytest

# TODO: MIGRATION - from victor.observability.event_bus import EventBus, EventCategory, VictorEvent  # DELETED
from victor.observability.tracing.execution import ExecutionSpan, ExecutionTracer
from victor.observability.tracing.tool_calls import ToolCallRecord, ToolCallTracer


# =============================================================================
# Test ExecutionTracer
# =============================================================================


class TestExecutionSpan:
    """Test ExecutionSpan dataclass."""

    def test_create_span(self):
        """Test creating an execution span."""
        span = ExecutionSpan(
            id="span-1",
            parent_id=None,
            agent_id="agent-1",
            span_type="agent",
            start_time=1234567890.0,
        )

        assert span.id == "span-1"
        assert span.parent_id is None
        assert span.agent_id == "agent-1"
        assert span.span_type == "agent"
        assert span.start_time == 1234567890.0
        assert span.end_time is None
        assert span.status == "running"
        assert span.children == []

    def test_span_with_metadata(self):
        """Test creating span with metadata."""
        span = ExecutionSpan(
            id="span-2",
            parent_id="span-1",
            agent_id="agent-2",
            span_type="tool",
            start_time=1234567890.0,
            metadata={"tool_name": "read_file"},
        )

        assert span.metadata == {"tool_name": "read_file"}

    def test_span_duration_ms_running(self):
        """Test duration_ms returns None for running spans."""
        span = ExecutionSpan(
            id="span-1",
            parent_id=None,
            agent_id="agent-1",
            span_type="agent",
            start_time=1234567890.0,
        )

        assert span.duration_ms is None

    def test_span_duration_ms_completed(self):
        """Test duration_ms returns duration for completed spans."""
        span = ExecutionSpan(
            id="span-1",
            parent_id=None,
            agent_id="agent-1",
            span_type="agent",
            start_time=1234567890.0,
            end_time=1234567895.0,
        )

        assert span.duration_ms == 5000.0

    def test_span_to_dict(self):
        """Test converting span to dictionary."""
        span = ExecutionSpan(
            id="span-1",
            parent_id=None,
            agent_id="agent-1",
            span_type="agent",
            start_time=1234567890.0,
            end_time=1234567895.0,
        )

        span_dict = span.to_dict()

        assert span_dict["id"] == "span-1"
        assert span_dict["parent_id"] is None
        assert span_dict["agent_id"] == "agent-1"
        assert span_dict["span_type"] == "agent"
        assert span_dict["duration_ms"] == 5000.0


class TestExecutionTracer:
    """Test ExecutionTracer class."""

    def test_create_tracer(self):
        """Test creating an execution tracer."""
        event_bus = EventBus()
        tracer = ExecutionTracer(event_bus)

        assert tracer.get_span_count() == 0
        assert tracer._spans == {}
        assert tracer._root_spans == []

    @pytest.mark.asyncio
    async def test_start_span(self):
        """Test starting a span."""
        event_bus = EventBus()
        tracer = ExecutionTracer(event_bus)

        span_id = tracer.start_span("agent-1", "agent")

        assert tracer.get_span_count() == 1
        assert len(tracer._root_spans) == 1

        span = tracer.get_span(span_id)
        assert span is not None
        assert span.agent_id == "agent-1"
        assert span.span_type == "agent"
        assert span.status == "running"

    @pytest.mark.asyncio
    async def test_start_child_span(self):
        """Test starting a child span with parent."""
        event_bus = EventBus()
        tracer = ExecutionTracer(event_bus)

        root_id = tracer.start_span("agent-1", "agent")
        child_id = tracer.start_span("agent-1", "tool", parent_id=root_id)

        root_span = tracer.get_span(root_id)
        child_span = tracer.get_span(child_id)

        assert child_span.parent_id == root_id
        assert child_id in root_span.children
        assert len(tracer._root_spans) == 1  # Only root is in root_spans

    @pytest.mark.asyncio
    async def test_end_span(self):
        """Test ending a span."""
        event_bus = EventBus()
        tracer = ExecutionTracer(event_bus)

        span_id = tracer.start_span("agent-1", "agent")
        tracer.end_span(span_id, status="success", result="output")

        span = tracer.get_span(span_id)
        assert span.status == "success"
        assert span.end_time is not None
        assert span.duration_ms is not None
        assert "result" in span.metadata

    @pytest.mark.asyncio
    async def test_end_span_with_error(self):
        """Test ending a span with error."""
        event_bus = EventBus()
        tracer = ExecutionTracer(event_bus)

        span_id = tracer.start_span("agent-1", "agent")
        tracer.end_span(span_id, status="error", error="Something went wrong")

        span = tracer.get_span(span_id)
        assert span.status == "error"
        assert span.metadata["error"] == "Something went wrong"

    @pytest.mark.asyncio
    async def test_end_nonexistent_span(self):
        """Test ending a span that doesn't exist (should not raise)."""
        event_bus = EventBus()
        tracer = ExecutionTracer(event_bus)

        # Should not raise
        tracer.end_span("nonexistent", status="success")

    @pytest.mark.asyncio
    async def test_get_span_tree(self):
        """Test getting span tree."""
        event_bus = EventBus()
        tracer = ExecutionTracer(event_bus)

        # Create span hierarchy
        root_id = tracer.start_span("agent-1", "agent")
        child1_id = tracer.start_span("agent-1", "tool", parent_id=root_id)
        child2_id = tracer.start_span("agent-1", "tool", parent_id=root_id)

        tracer.end_span(child1_id, status="success")
        tracer.end_span(child2_id, status="success")
        tracer.end_span(root_id, status="success")

        tree = tracer.get_span_tree(root_id)

        assert tree["id"] == root_id
        assert len(tree["children"]) == 2
        assert tree["children"][0]["id"] == child1_id
        assert tree["children"][1]["id"] == child2_id

    @pytest.mark.asyncio
    async def test_get_span_tree_default_root(self):
        """Test getting span tree with default root."""
        event_bus = EventBus()
        tracer = ExecutionTracer(event_bus)

        root_id = tracer.start_span("agent-1", "agent")
        tracer.start_span("agent-1", "tool", parent_id=root_id)

        tree = tracer.get_span_tree()

        assert tree["id"] == root_id

    @pytest.mark.asyncio
    async def test_get_all_spans(self):
        """Test getting all spans."""
        event_bus = EventBus()
        tracer = ExecutionTracer(event_bus)

        tracer.start_span("agent-1", "agent")
        tracer.start_span("agent-2", "agent")

        spans = tracer.get_all_spans()

        assert len(spans) == 2

    @pytest.mark.asyncio
    async def test_get_spans_by_agent(self):
        """Test getting spans filtered by agent."""
        event_bus = EventBus()
        tracer = ExecutionTracer(event_bus)

        tracer.start_span("agent-1", "agent")
        tracer.start_span("agent-1", "tool")
        tracer.start_span("agent-2", "agent")

        agent1_spans = tracer.get_spans_by_agent("agent-1")

        assert len(agent1_spans) == 2
        assert all(s.agent_id == "agent-1" for s in agent1_spans)

    @pytest.mark.asyncio
    async def test_get_spans_by_type(self):
        """Test getting spans filtered by type."""
        event_bus = EventBus()
        tracer = ExecutionTracer(event_bus)

        tracer.start_span("agent-1", "agent")
        tracer.start_span("agent-1", "tool")
        tracer.start_span("agent-1", "tool")

        tool_spans = tracer.get_spans_by_type("tool")

        assert len(tool_spans) == 2
        assert all(s.span_type == "tool" for s in tool_spans)

    @pytest.mark.asyncio
    async def test_get_active_spans(self):
        """Test getting active (running) spans."""
        event_bus = EventBus()
        tracer = ExecutionTracer(event_bus)

        span1 = tracer.start_span("agent-1", "agent")
        span2 = tracer.start_span("agent-2", "agent")

        tracer.end_span(span1, status="success")

        active_spans = tracer.get_active_spans()

        assert len(active_spans) == 1
        assert active_spans[0].id == span2

    @pytest.mark.asyncio
    async def test_clear_spans(self):
        """Test clearing all spans."""
        event_bus = EventBus()
        tracer = ExecutionTracer(event_bus)

        tracer.start_span("agent-1", "agent")
        tracer.start_span("agent-2", "agent")

        assert tracer.get_span_count() == 2

        tracer.clear_spans()

        assert tracer.get_span_count() == 0
        assert tracer._root_spans == []

    @pytest.mark.asyncio
    async def test_get_statistics(self):
        """Test getting span statistics."""
        event_bus = EventBus()
        tracer = ExecutionTracer(event_bus)

        # Empty stats
        stats = tracer.get_statistics()
        assert stats["total"] == 0

        # Add some spans
        span1 = tracer.start_span("agent-1", "agent")
        span2 = tracer.start_span("agent-1", "tool")
        span3 = tracer.start_span("agent-2", "llm_call")

        tracer.end_span(span1, status="success")
        tracer.end_span(span2, status="error")
        # span3 still running

        stats = tracer.get_statistics()

        assert stats["total"] == 3
        assert stats["active"] == 1
        assert stats["completed"] == 2
        assert stats["by_type"]["agent"] == 1
        assert stats["by_type"]["tool"] == 1
        assert stats["by_type"]["llm_call"] == 1
        assert stats["by_status"]["running"] == 1
        assert stats["by_status"]["success"] == 1
        assert stats["by_status"]["error"] == 1

    @pytest.mark.asyncio
    async def test_span_with_metadata(self):
        """Test starting span with metadata."""
        event_bus = EventBus()
        tracer = ExecutionTracer(event_bus)

        span_id = tracer.start_span(
            "agent-1", "tool", metadata={"tool_name": "read_file", "file": "test.py"}
        )

        span = tracer.get_span(span_id)
        assert span.metadata["tool_name"] == "read_file"
        assert span.metadata["file"] == "test.py"


# =============================================================================
# Test ToolCallTracer
# =============================================================================


class TestToolCallRecord:
    """Test ToolCallRecord dataclass."""

    def test_create_record(self):
        """Test creating a tool call record."""
        record = ToolCallRecord(
            call_id="tc-1",
            parent_span_id="span-1",
            tool_name="read_file",
            arguments={"file": "test.py"},
        )

        assert record.call_id == "tc-1"
        assert record.parent_span_id == "span-1"
        assert record.tool_name == "read_file"
        assert record.arguments == {"file": "test.py"}
        assert record.result is None
        assert record.error is None
        assert record.end_time is None

    def test_record_complete(self):
        """Test completing a record."""
        record = ToolCallRecord(
            call_id="tc-1",
            parent_span_id="span-1",
            tool_name="read_file",
            arguments={"file": "test.py"},
        )

        assert record.is_complete is False

    def test_record_successful(self):
        """Test checking if record is successful."""
        record = ToolCallRecord(
            call_id="tc-1",
            parent_span_id="span-1",
            tool_name="read_file",
            arguments={"file": "test.py"},
        )

        assert record.is_successful is False

    def test_record_to_dict(self):
        """Test converting record to dictionary."""
        record = ToolCallRecord(
            call_id="tc-1",
            parent_span_id="span-1",
            tool_name="read_file",
            arguments={"file": "test.py"},
        )

        record_dict = record.to_dict()

        assert record_dict["call_id"] == "tc-1"
        assert record_dict["tool_name"] == "read_file"
        assert record_dict["is_complete"] is False
        assert record_dict["is_successful"] is False


class TestToolCallTracer:
    """Test ToolCallTracer class."""

    def test_create_tracer(self):
        """Test creating a tool call tracer."""
        event_bus = EventBus()
        tracer = ToolCallTracer(event_bus)

        assert tracer.get_call_count() == 0
        assert tracer._calls == {}

    @pytest.mark.asyncio
    async def test_record_call(self):
        """Test recording a tool call."""
        event_bus = EventBus()
        tracer = ToolCallTracer(event_bus)

        call_id = tracer.record_call(
            tool_name="read_file",
            arguments={"file": "test.py"},
            parent_span_id="span-123",
        )

        assert tracer.get_call_count() == 1

        call = tracer.get_call(call_id)
        assert call is not None
        assert call.tool_name == "read_file"
        assert call.arguments == {"file": "test.py"}
        assert call.parent_span_id == "span-123"

    @pytest.mark.asyncio
    async def test_complete_call(self):
        """Test completing a tool call."""
        event_bus = EventBus()
        tracer = ToolCallTracer(event_bus)

        call_id = tracer.record_call("read_file", {"file": "test.py"}, "span-123")
        tracer.complete_call(call_id, result="file content")

        call = tracer.get_call(call_id)
        assert call.is_complete is True
        assert call.is_successful is True
        assert call.result == "file content"
        assert call.duration_ms is not None

    @pytest.mark.asyncio
    async def test_fail_call(self):
        """Test failing a tool call."""
        event_bus = EventBus()
        tracer = ToolCallTracer(event_bus)

        call_id = tracer.record_call("read_file", {"file": "test.py"}, "span-123")
        tracer.fail_call(call_id, error="File not found")

        call = tracer.get_call(call_id)
        assert call.is_complete is True
        assert call.is_successful is False
        assert call.error == "File not found"
        assert call.duration_ms is not None

    @pytest.mark.asyncio
    async def test_get_calls_by_tool_name(self):
        """Test getting calls filtered by tool name."""
        event_bus = EventBus()
        tracer = ToolCallTracer(event_bus)

        tracer.record_call("read_file", {"file": "test1.py"}, "span-1")
        tracer.record_call("write_file", {"file": "test2.py"}, "span-2")
        tracer.record_call("read_file", {"file": "test3.py"}, "span-3")

        read_calls = tracer.get_calls(tool_name="read_file")

        assert len(read_calls) == 2
        assert all(c.tool_name == "read_file" for c in read_calls)

    @pytest.mark.asyncio
    async def test_get_calls_by_parent_span(self):
        """Test getting calls filtered by parent span."""
        event_bus = EventBus()
        tracer = ToolCallTracer(event_bus)

        tracer.record_call("read_file", {"file": "test1.py"}, "span-1")
        tracer.record_call("write_file", {"file": "test2.py"}, "span-1")
        tracer.record_call("read_file", {"file": "test3.py"}, "span-2")

        span1_calls = tracer.get_calls(parent_span_id="span-1")

        assert len(span1_calls) == 2

    @pytest.mark.asyncio
    async def test_get_calls_with_limit(self):
        """Test getting calls with limit."""
        event_bus = EventBus()
        tracer = ToolCallTracer(event_bus)

        for i in range(10):
            tracer.record_call(f"tool_{i}", {"arg": i}, f"span-{i}")

        calls = tracer.get_calls(limit=5)

        assert len(calls) == 5

    @pytest.mark.asyncio
    async def test_get_calls_by_span(self):
        """Test getting all calls for a specific span."""
        event_bus = EventBus()
        tracer = ToolCallTracer(event_bus)

        tracer.record_call("read_file", {"file": "test1.py"}, "span-1")
        tracer.record_call("write_file", {"file": "test2.py"}, "span-1")
        tracer.record_call("read_file", {"file": "test3.py"}, "span-2")

        span1_calls = tracer.get_calls_by_span("span-1")

        assert len(span1_calls) == 2
        assert all(c.parent_span_id == "span-1" for c in span1_calls)

    @pytest.mark.asyncio
    async def test_get_failed_calls(self):
        """Test getting failed tool calls."""
        event_bus = EventBus()
        tracer = ToolCallTracer(event_bus)

        call1 = tracer.record_call("read_file", {"file": "test1.py"}, "span-1")
        call2 = tracer.record_call("write_file", {"file": "test2.py"}, "span-2")
        call3 = tracer.record_call("delete_file", {"file": "test3.py"}, "span-3")

        tracer.complete_call(call1, result="content")
        tracer.fail_call(call2, error="Permission denied")
        tracer.complete_call(call3, result="deleted")  # Complete call3

        failed_calls = tracer.get_failed_calls()

        # Only call2 failed (has error)
        assert len(failed_calls) == 1
        assert failed_calls[0].call_id == call2
        assert failed_calls[0].error == "Permission denied"

    @pytest.mark.asyncio
    async def test_get_all_calls(self):
        """Test getting all tool calls."""
        event_bus = EventBus()
        tracer = ToolCallTracer(event_bus)

        tracer.record_call("tool1", {"arg": 1}, "span-1")
        tracer.record_call("tool2", {"arg": 2}, "span-2")

        calls = tracer.get_all_calls()

        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_clear_calls(self):
        """Test clearing all tool calls."""
        event_bus = EventBus()
        tracer = ToolCallTracer(event_bus)

        tracer.record_call("tool1", {"arg": 1}, "span-1")
        tracer.record_call("tool2", {"arg": 2}, "span-2")

        assert tracer.get_call_count() == 2

        tracer.clear_calls()

        assert tracer.get_call_count() == 0

    @pytest.mark.asyncio
    async def test_get_statistics(self):
        """Test getting tool call statistics."""
        event_bus = EventBus()
        tracer = ToolCallTracer(event_bus)

        # Empty stats
        stats = tracer.get_statistics()
        assert stats["total"] == 0

        # Add some calls
        call1 = tracer.record_call("read_file", {"file": "test1.py"}, "span-1")
        call2 = tracer.record_call("write_file", {"file": "test2.py"}, "span-2")
        call3 = tracer.record_call("read_file", {"file": "test3.py"}, "span-3")

        tracer.complete_call(call1, result="content")
        tracer.fail_call(call2, error="Error")
        # call3 still running

        stats = tracer.get_statistics()

        assert stats["total"] == 3
        assert stats["successful"] == 1
        assert stats["failed"] == 1
        assert stats["running"] == 1
        assert stats["by_tool"]["read_file"] == 2
        assert stats["by_tool"]["write_file"] == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestTracerIntegration:
    """Integration tests for tracers."""

    @pytest.mark.asyncio
    async def test_execution_tracer_event_bus_integration(self):
        """Test ExecutionTracer integrates with EventBus."""
        event_bus = EventBus()
        tracer = ExecutionTracer(event_bus)

        # Subscribe to LIFECYCLE events
        received_events = []

        def on_event(event):
            if event.name in ["span_started", "span_ended"]:
                received_events.append(event)

        event_bus.subscribe(EventCategory.LIFECYCLE, on_event)

        # Start and end span
        span_id = tracer.start_span("agent-1", "agent", metadata={"task": "test"})
        tracer.end_span(span_id, status="success", result="output")

        # Verify events were emitted
        assert len(received_events) >= 2

        started = [e for e in received_events if e.name == "span_started"]
        ended = [e for e in received_events if e.name == "span_ended"]

        assert len(started) >= 1
        assert len(ended) >= 1

    @pytest.mark.asyncio
    async def test_tool_call_tracer_event_bus_integration(self):
        """Test ToolCallTracer integrates with EventBus."""
        event_bus = EventBus()
        tracer = ToolCallTracer(event_bus)

        # Subscribe to TOOL events
        received_events = []

        def on_event(event):
            if event.name in ["tool_call_started", "tool_call_completed"]:
                received_events.append(event)

        event_bus.subscribe(EventCategory.TOOL, on_event)

        # Record and complete call
        call_id = tracer.record_call("read_file", {"file": "test.py"}, "span-123")
        tracer.complete_call(call_id, result="file content")

        # Verify events were emitted
        assert len(received_events) >= 2

        started = [e for e in received_events if e.name == "tool_call_started"]
        completed = [e for e in received_events if e.name == "tool_call_completed"]

        assert len(started) >= 1
        assert len(completed) >= 1

    @pytest.mark.asyncio
    async def test_linked_execution_and_tool_calls(self):
        """Test tool calls linked to execution spans."""
        event_bus = EventBus()
        exec_tracer = ExecutionTracer(event_bus)
        tool_tracer = ToolCallTracer(event_bus)

        # Create execution span
        span_id = exec_tracer.start_span("agent-1", "agent")

        # Record tool call linked to span
        call_id = tool_tracer.record_call("read_file", {"file": "test.py"}, span_id)

        # Complete both
        tool_tracer.complete_call(call_id, result="content")
        exec_tracer.end_span(span_id, status="success")

        # Verify linkage (tool call links to span via parent_span_id)
        call = tool_tracer.get_call(call_id)
        assert call.parent_span_id == span_id
        assert call.is_complete
        assert call.is_successful

        # Verify span exists and is complete
        span = exec_tracer.get_span(span_id)
        assert span.status == "success"
        # Note: span.children only contains other execution spans, not tool calls
        # Tool calls are linked via parent_span_id field


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
