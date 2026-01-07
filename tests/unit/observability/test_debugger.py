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

"""Unit tests for AgentDebugger.

Tests the unified debugging facade that combines ExecutionTracer,
ToolCallTracer, and StateTracer for comprehensive agent debugging.
"""

import pytest
from unittest.mock import Mock, MagicMock

# TODO: MIGRATION - from victor.observability.event_bus import EventBus, VictorEvent, EventCategory  # DELETED
from victor.observability.debugger import AgentDebugger
from victor.observability.tracing.execution import ExecutionTracer
from victor.observability.tracing.tool_calls import ToolCallTracer


class TestAgentDebuggerInitialization:
    """Test AgentDebugger initialization and setup."""

    @pytest.mark.asyncio
    async def test_init_with_event_bus_only(self):
        """Test initialization with just EventBus."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        assert debugger._execution is not None
        assert debugger._tool_calls is not None
        assert debugger._state is not None

    @pytest.mark.asyncio
    async def test_init_with_custom_tracers(self):
        """Test initialization with custom tracer instances."""
        event_bus = EventBus()
        exec_tracer = ExecutionTracer(event_bus)
        tool_tracer = ToolCallTracer(event_bus)
        state_tracer = Mock()

        debugger = AgentDebugger(
            event_bus,
            execution_tracer=exec_tracer,
            tool_call_tracer=tool_tracer,
            state_tracer=state_tracer,
        )

        assert debugger._execution is exec_tracer
        assert debugger._tool_calls is tool_tracer
        assert debugger._state is state_tracer


class TestExecutionTraceMethods:
    """Test execution trace retrieval methods."""

    @pytest.mark.asyncio
    async def test_get_execution_trace_default(self):
        """Test getting default execution trace."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Create some spans
        tracer = debugger._execution
        root_id = tracer.start_span("agent-1", "agent")
        child_id = tracer.start_span("agent-1", "tool", parent_id=root_id)
        tracer.end_span(child_id, status="success")
        tracer.end_span(root_id, status="success")

        # Get trace
        trace = debugger.get_execution_trace()

        assert trace["id"] == root_id
        assert len(trace["children"]) == 1
        assert trace["children"][0]["id"] == child_id

    @pytest.mark.asyncio
    async def test_get_execution_trace_by_agent_id(self):
        """Test getting execution trace filtered by agent ID."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Create spans for different agents
        tracer = debugger._execution
        agent1_root = tracer.start_span("agent-1", "agent")
        agent2_root = tracer.start_span("agent-2", "agent")

        # Get trace for agent-1
        trace = debugger.get_execution_trace(agent_id="agent-1")

        assert trace["agent_id"] == "agent-1"

    @pytest.mark.asyncio
    async def test_get_execution_trace_by_span_id(self):
        """Test getting execution trace for specific span ID."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Create spans
        tracer = debugger._execution
        root_id = tracer.start_span("agent-1", "agent")
        child_id = tracer.start_span("agent-1", "tool", parent_id=root_id)

        # Get trace for child span
        trace = debugger.get_execution_trace(span_id=child_id)

        assert trace["id"] == child_id
        assert len(trace["children"]) == 0

    @pytest.mark.asyncio
    async def test_get_execution_spans(self):
        """Test getting execution spans as list."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Create spans
        tracer = debugger._execution
        tracer.start_span("agent-1", "agent")
        tracer.start_span("agent-2", "tool")
        tracer.start_span("agent-1", "llm_call")

        # Get all spans
        spans = debugger.get_execution_spans()

        assert len(spans) == 3

    @pytest.mark.asyncio
    async def test_get_execution_spans_filtered(self):
        """Test getting execution spans with filters."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Create spans
        tracer = debugger._execution
        tracer.start_span("agent-1", "agent")
        tracer.start_span("agent-1", "tool")
        tracer.start_span("agent-2", "agent")

        # Filter by agent
        agent1_spans = debugger.get_execution_spans(agent_id="agent-1")
        assert len(agent1_spans) == 2

        # Filter by type
        agent_spans = debugger.get_execution_spans(span_type="agent")
        assert len(agent_spans) == 2

    @pytest.mark.asyncio
    async def test_get_active_spans(self):
        """Test getting active spans."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Create spans
        tracer = debugger._execution
        tracer.start_span("agent-1", "agent")
        running_id = tracer.start_span("agent-1", "tool")
        completed_id = tracer.start_span("agent-1", "llm_call")
        tracer.end_span(completed_id, status="success")

        # Get active spans
        active_spans = debugger.get_active_spans()

        # Should have 2 active spans (agent and tool)
        assert len(active_spans) == 2
        active_ids = [s["id"] for s in active_spans]
        assert running_id in active_ids
        assert completed_id not in active_ids


class TestToolCallMethods:
    """Test tool call retrieval methods."""

    @pytest.mark.asyncio
    async def test_get_tool_calls(self):
        """Test getting tool calls."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Create tool calls
        tracer = debugger._tool_calls
        call1 = tracer.record_call("read_file", {"file": "test1.py"}, "span-1")
        call2 = tracer.record_call("write_file", {"file": "test2.py"}, "span-2")
        tracer.complete_call(call1, result="file content")

        # Get all calls
        calls = debugger.get_tool_calls()

        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_get_tool_calls_filtered_by_name(self):
        """Test getting tool calls filtered by tool name."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Create tool calls
        tracer = debugger._tool_calls
        tracer.record_call("read_file", {"file": "test1.py"}, "span-1")
        tracer.record_call("read_file", {"file": "test2.py"}, "span-2")
        tracer.record_call("write_file", {"file": "test3.py"}, "span-3")

        # Filter by tool name
        read_calls = debugger.get_tool_calls(tool_name="read_file")

        assert len(read_calls) == 2
        assert all(c["tool_name"] == "read_file" for c in read_calls)

    @pytest.mark.asyncio
    async def test_get_tool_calls_filtered_by_parent_span(self):
        """Test getting tool calls filtered by parent span ID."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Create tool calls
        tracer = debugger._tool_calls
        tracer.record_call("read_file", {"file": "test1.py"}, "span-1")
        tracer.record_call("write_file", {"file": "test2.py"}, "span-1")
        tracer.record_call("read_file", {"file": "test3.py"}, "span-2")

        # Filter by parent span
        span1_calls = debugger.get_tool_calls(parent_span_id="span-1")

        assert len(span1_calls) == 2

    @pytest.mark.asyncio
    async def test_get_tool_calls_with_limit(self):
        """Test getting tool calls with limit."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Create tool calls
        tracer = debugger._tool_calls
        for i in range(10):
            tracer.record_call("read_file", {"file": f"test{i}.py"}, "span-1")

        # Get with limit
        calls = debugger.get_tool_calls(limit=5)

        assert len(calls) == 5

    @pytest.mark.asyncio
    async def test_get_failed_tool_calls(self):
        """Test getting failed tool calls."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Create tool calls
        tracer = debugger._tool_calls
        call1 = tracer.record_call("read_file", {"file": "test1.py"}, "span-1")
        call2 = tracer.record_call("read_file", {"file": "test2.py"}, "span-2")
        call3 = tracer.record_call("write_file", {"file": "test3.py"}, "span-3")

        tracer.complete_call(call1, result="content")
        tracer.fail_call(call2, error="File not found")
        tracer.fail_call(call3, error="Permission denied")

        # Get failed calls
        failed_calls = debugger.get_failed_tool_calls()

        assert len(failed_calls) == 2
        assert all(c["error"] is not None for c in failed_calls)

    @pytest.mark.asyncio
    async def test_get_tool_calls_by_span(self):
        """Test getting all tool calls for a specific span."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Create tool calls
        tracer = debugger._tool_calls
        tracer.record_call("read_file", {"file": "test1.py"}, "span-1")
        tracer.record_call("write_file", {"file": "test2.py"}, "span-1")
        tracer.record_call("read_file", {"file": "test3.py"}, "span-2")

        # Get calls for span-1
        span1_calls = debugger.get_tool_calls_by_span("span-1")

        assert len(span1_calls) == 2


class TestStateTransitionMethods:
    """Test state transition retrieval methods."""

    @pytest.mark.asyncio
    async def test_get_state_transitions(self):
        """Test getting state transitions."""
        event_bus = EventBus()
        state_tracer = Mock()
        state_tracer.get_history.return_value = []

        debugger = AgentDebugger(event_bus, state_tracer=state_tracer)

        # Get transitions
        transitions = debugger.get_state_transitions()

        assert isinstance(transitions, list)
        state_tracer.get_history.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_state_transitions_filtered(self):
        """Test getting state transitions with filters."""
        event_bus = EventBus()
        state_tracer = Mock()
        state_tracer.get_history.return_value = []

        debugger = AgentDebugger(event_bus, state_tracer=state_tracer)

        # Get with filters
        transitions = debugger.get_state_transitions(scope="workflow", key="task_id", limit=50)

        state_tracer.get_history.assert_called_once_with(scope="workflow", key="task_id", limit=50)


class TestPerformanceSummaryMethods:
    """Test performance summary methods."""

    @pytest.mark.asyncio
    async def test_get_performance_summary(self):
        """Test getting performance summary."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Create some data
        exec_tracer = debugger._execution
        exec_tracer.start_span("agent-1", "agent")

        tool_tracer = debugger._tool_calls
        call_id = tool_tracer.record_call("read_file", {"file": "test.py"}, "span-1")
        tool_tracer.complete_call(call_id, result="content")

        # Get summary
        summary = debugger.get_performance_summary()

        assert "execution" in summary
        assert "tool_calls" in summary
        assert "state" in summary
        assert "summary" in summary

    @pytest.mark.asyncio
    async def test_get_performance_summary_aggregates_metrics(self):
        """Test that performance summary aggregates metrics correctly."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Create execution spans
        exec_tracer = debugger._execution
        exec_tracer.start_span("agent-1", "agent")
        exec_tracer.start_span("agent-1", "tool")

        # Create tool calls
        tool_tracer = debugger._tool_calls
        call1 = tool_tracer.record_call("read_file", {"file": "test1.py"}, "span-1")
        call2 = tool_tracer.record_call("read_file", {"file": "test2.py"}, "span-2")
        tool_tracer.complete_call(call1, result="content1")
        tool_tracer.fail_call(call2, error="Error")

        # Get summary
        summary = debugger.get_performance_summary()

        assert summary["summary"]["total_spans"] == 2
        assert summary["summary"]["total_tool_calls"] == 2
        assert summary["summary"]["failed_tool_calls"] == 1

    @pytest.mark.asyncio
    async def test_get_slow_tool_calls(self):
        """Test getting slow tool calls."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Create tool calls with different durations
        import time

        tool_tracer = debugger._tool_calls

        # Fast call
        call1 = tool_tracer.record_call("read_file", {"file": "test1.py"}, "span-1")
        tool_tracer.complete_call(call1, result="content")

        # Slow call (manually set duration)
        call2 = tool_tracer.record_call("read_file", {"file": "test2.py"}, "span-2")
        tool_tracer.complete_call(call2, result="content")
        # Manually set a slow duration
        call2_record = tool_tracer.get_call(call2)
        if call2_record:
            call2_record.duration_ms = 2000.0

        # Get slow calls
        slow_calls = debugger.get_slow_tool_calls(threshold_ms=1000.0)

        # Should have at least the 2000ms call
        assert len(slow_calls) >= 1
        assert all(c["duration_ms"] > 1000.0 for c in slow_calls)

    @pytest.mark.asyncio
    async def test_get_slow_tool_calls_with_limit(self):
        """Test getting slow tool calls with limit."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Create multiple slow calls
        tool_tracer = debugger._tool_calls
        for i in range(10):
            call_id = tool_tracer.record_call("read_file", {"file": f"test{i}.py"}, "span-1")
            tool_tracer.complete_call(call_id, result="content")
            # Manually set slow duration
            call_record = tool_tracer.get_call(call_id)
            if call_record:
                call_record.duration_ms = 1500.0 + i * 100

        # Get with limit
        slow_calls = debugger.get_slow_tool_calls(threshold_ms=1000.0, limit=5)

        assert len(slow_calls) == 5
        # Should be sorted by duration (slowest first)
        assert slow_calls[0]["duration_ms"] >= slow_calls[-1]["duration_ms"]


class TestUtilityMethods:
    """Test utility methods."""

    @pytest.mark.asyncio
    async def test_clear_all(self):
        """Test clearing all debugging data."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Create some data
        exec_tracer = debugger._execution
        exec_tracer.start_span("agent-1", "agent")

        tool_tracer = debugger._tool_calls
        call_id = tool_tracer.record_call("read_file", {"file": "test.py"}, "span-1")
        tool_tracer.complete_call(call_id, result="content")

        # Clear all
        debugger.clear_all()

        # Verify cleared
        assert exec_tracer.get_span_count() == 0
        assert tool_tracer.get_call_count() == 0

    @pytest.mark.asyncio
    async def test_get_summary(self):
        """Test getting brief summary."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Create some data
        exec_tracer = debugger._execution
        exec_tracer.start_span("agent-1", "agent")

        tool_tracer = debugger._tool_calls
        call_id = tool_tracer.record_call("read_file", {"file": "test.py"}, "span-1")
        tool_tracer.complete_call(call_id, result="content")

        # Get summary
        summary = debugger.get_summary()

        assert "spans" in summary
        assert "active_spans" in summary
        assert "tool_calls" in summary
        assert "failed_tool_calls" in summary
        assert "state_transitions" in summary

        assert summary["spans"] == 1
        assert summary["tool_calls"] == 1


class TestDebuggerIntegration:
    """Integration tests for AgentDebugger with real data."""

    @pytest.mark.asyncio
    async def test_full_debugging_workflow(self):
        """Test complete debugging workflow with execution, tools, and state."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Simulate agent execution
        span_id = debugger._execution.start_span("agent-1", "agent")

        # Simulate tool calls
        call1_id = debugger._tool_calls.record_call("read_file", {"file": "test.py"}, span_id)
        debugger._tool_calls.complete_call(call1_id, result="file content")

        call2_id = debugger._tool_calls.record_call("write_file", {"file": "output.txt"}, span_id)
        debugger._tool_calls.complete_call(call2_id, result="success")

        # End execution
        debugger._execution.end_span(span_id, status="success")

        # Get execution trace
        trace = debugger.get_execution_trace()
        assert trace["agent_id"] == "agent-1"
        assert trace["status"] == "success"

        # Get tool calls for the span
        calls = debugger.get_tool_calls_by_span(span_id)
        assert len(calls) == 2

        # Get performance summary
        summary = debugger.get_performance_summary()
        assert summary["summary"]["total_spans"] == 1
        assert summary["summary"]["total_tool_calls"] == 2

    @pytest.mark.asyncio
    async def test_multi_agent_debugging(self):
        """Test debugging multiple agents."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Agent 1 execution
        agent1_span = debugger._execution.start_span("agent-1", "agent")
        call1 = debugger._tool_calls.record_call("search", {"query": "test"}, agent1_span)
        debugger._tool_calls.complete_call(call1, result="results")
        debugger._execution.end_span(agent1_span, status="success")

        # Agent 2 execution
        agent2_span = debugger._execution.start_span("agent-2", "agent")
        call2 = debugger._tool_calls.record_call("read", {"file": "data.txt"}, agent2_span)
        debugger._tool_calls.complete_call(call2, result="data")
        debugger._execution.end_span(agent2_span, status="success")

        # Get traces by agent
        agent1_trace = debugger.get_execution_trace(agent_id="agent-1")
        assert agent1_trace["agent_id"] == "agent-1"

        agent2_trace = debugger.get_execution_trace(agent_id="agent-2")
        assert agent2_trace["agent_id"] == "agent-2"

        # Get all spans
        all_spans = debugger.get_execution_spans()
        assert len(all_spans) == 2

    @pytest.mark.asyncio
    async def test_error_tracking_workflow(self):
        """Test tracking errors across execution and tool calls."""
        event_bus = EventBus()
        debugger = AgentDebugger(event_bus)

        # Agent execution with tool error
        span_id = debugger._execution.start_span("agent-1", "agent")

        # Successful tool call
        call1_id = debugger._tool_calls.record_call("read_file", {"file": "test.py"}, span_id)
        debugger._tool_calls.complete_call(call1_id, result="content")

        # Failed tool call
        call2_id = debugger._tool_calls.record_call("write_file", {"file": "readonly.txt"}, span_id)
        debugger._tool_calls.fail_call(call2_id, error="Permission denied")

        # End execution with error
        debugger._execution.end_span(span_id, status="error", error="Tool failed")

        # Get failed tool calls
        failed_calls = debugger.get_failed_tool_calls()
        assert len(failed_calls) == 1
        assert "Permission denied" in failed_calls[0]["error"]

        # Get execution trace with error
        trace = debugger.get_execution_trace()
        assert trace["status"] == "error"

        # Get performance summary
        summary = debugger.get_performance_summary()
        assert summary["summary"]["failed_tool_calls"] == 1
