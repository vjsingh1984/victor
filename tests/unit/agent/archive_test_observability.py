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

# ==============================================================================
# ARCHIVED TEST - This test is for an obsolete module
# ==============================================================================
#
# This test file has been archived because the module it tests (observability.py)
# has been moved to victor/agent/archive/obsolete/observability.py
#
# The observability module is superseded by:
#   - victor.core.events (get_observability_bus)
#   - The canonical ObservabilityBus with CQRS event sourcing
#
# This test file is preserved for reference but is not run as part of the
# regular test suite. Rename to test_observability.py to run if needed.
#
# ==============================================================================

"""Unit tests for observability module (ARCHIVED - Module is obsolete)."""

import asyncio
import pytest
import time

from victor.agent.archive.obsolete.observability import (  # noqa: ARCHIVED
    TracingProvider,
    Span,
    SpanKind,
    SpanStatus,
    get_observability,
    set_observability,
    traced,
)


class TestSpan:
    """Tests for Span class."""

    def test_span_creation(self) -> None:
        """Test basic span creation."""
        span = Span(
            name="test_span",
            trace_id="trace-123",
            span_id="span-456",
        )
        assert span.name == "test_span"
        assert span.trace_id == "trace-123"
        assert span.span_id == "span-456"
        assert span.status == SpanStatus.UNSET
        assert span.kind == SpanKind.INTERNAL

    def test_span_set_attribute(self) -> None:
        """Test setting span attributes."""
        span = Span(name="test", trace_id="t", span_id="s")
        span.set_attribute("key1", "value1")
        span.set_attribute("key2", 42)

        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == 42

    def test_span_add_event(self) -> None:
        """Test adding events to span."""
        span = Span(name="test", trace_id="t", span_id="s")
        span.add_event("checkpoint", {"progress": 50})

        assert len(span.events) == 1
        assert span.events[0]["name"] == "checkpoint"
        assert span.events[0]["attributes"]["progress"] == 50

    def test_span_end(self) -> None:
        """Test ending a span."""
        span = Span(name="test", trace_id="t", span_id="s")
        time.sleep(0.01)  # Small delay
        span.end(SpanStatus.OK)

        assert span.end_time is not None
        assert span.status == SpanStatus.OK
        assert span.duration_ms > 0

    def test_span_to_dict(self) -> None:
        """Test span serialization."""
        span = Span(
            name="test",
            trace_id="t",
            span_id="s",
            kind=SpanKind.CLIENT,
        )
        span.set_attribute("key", "value")
        span.end()

        data = span.to_dict()
        assert data["name"] == "test"
        assert data["trace_id"] == "t"
        assert data["kind"] == "client"
        assert data["attributes"]["key"] == "value"


class TestTracingProvider:
    """Tests for TracingProvider class."""

    @pytest.fixture
    def obs(self) -> TracingProvider:
        """Create fresh tracing provider."""
        return TracingProvider()

    def test_start_trace(self, obs: TracingProvider) -> None:
        """Test starting a trace."""
        trace_id = obs.start_trace()
        assert trace_id is not None
        assert obs.get_current_trace_id() == trace_id

    def test_span_context_manager(self, obs: TracingProvider) -> None:
        """Test span as context manager."""
        with obs.span("test_operation") as span:
            assert span.name == "test_operation"
            assert obs.get_current_span() is span
            span.set_attribute("test", True)

        # Span should be ended
        assert span.status == SpanStatus.OK
        assert span.end_time is not None

    def test_nested_spans(self, obs: TracingProvider) -> None:
        """Test nested span hierarchy."""
        with obs.span("parent") as parent_span:
            parent_id = parent_span.span_id

            with obs.span("child") as child_span:
                assert child_span.parent_span_id == parent_id
                assert obs.get_current_span() is child_span

            # Back to parent
            assert obs.get_current_span() is parent_span

        # No current span after all exit
        assert obs.get_current_span() is None

    def test_span_error_handling(self, obs: TracingProvider) -> None:
        """Test span records error on exception."""
        with pytest.raises(ValueError):
            with obs.span("failing_op") as span:
                raise ValueError("Test error")

        assert span.status == SpanStatus.ERROR
        assert span.attributes["error.type"] == "ValueError"
        assert "Test error" in span.attributes["error.message"]

    def test_span_start_hook(self, obs: TracingProvider) -> None:
        """Test span start hook is called."""
        started_spans = []
        obs.on_span_start(lambda s: started_spans.append(s.name))

        with obs.span("hooked_span"):
            pass

        assert "hooked_span" in started_spans

    def test_span_end_hook(self, obs: TracingProvider) -> None:
        """Test span end hook is called."""
        ended_spans = []
        obs.on_span_end(lambda s: ended_spans.append((s.name, s.duration_ms)))

        with obs.span("timed_span"):
            time.sleep(0.01)

        assert len(ended_spans) == 1
        assert ended_spans[0][0] == "timed_span"
        assert ended_spans[0][1] > 0

    def test_metric_hook(self, obs: TracingProvider) -> None:
        """Test metric recording hook."""
        recorded_metrics = []
        obs.on_metric(lambda name, value, labels: recorded_metrics.append((name, value, labels)))

        obs.record_metric("latency_ms", 42.5, {"endpoint": "/api/test"})

        assert len(recorded_metrics) == 1
        assert recorded_metrics[0][0] == "latency_ms"
        assert recorded_metrics[0][1] == 42.5
        assert recorded_metrics[0][2]["endpoint"] == "/api/test"

    def test_completed_spans(self, obs: TracingProvider) -> None:
        """Test completed spans are stored."""
        with obs.span("span1"):
            pass
        with obs.span("span2"):
            pass

        completed = obs.get_completed_spans()
        assert len(completed) == 2
        assert completed[0].name == "span1"
        assert completed[1].name == "span2"

    def test_clear_completed_spans(self, obs: TracingProvider) -> None:
        """Test clearing completed spans."""
        with obs.span("span1"):
            pass

        obs.clear_completed_spans()
        assert len(obs.get_completed_spans()) == 0

    def test_end_trace(self, obs: TracingProvider) -> None:
        """Test ending a trace."""
        obs.start_trace()
        assert obs.get_current_trace_id() is not None

        obs.end_trace()
        assert obs.get_current_trace_id() is None


class TestTracedDecorator:
    """Tests for @traced decorator."""

    def test_traced_sync_function(self) -> None:
        """Test tracing synchronous function."""
        obs = TracingProvider()
        set_observability(obs)

        @traced()
        def sync_func(x: int) -> int:
            return x * 2

        result = sync_func(5)
        assert result == 10

        completed = obs.get_completed_spans()
        assert len(completed) == 1
        assert completed[0].name == "sync_func"

    @pytest.mark.asyncio
    async def test_traced_async_function(self) -> None:
        """Test tracing async function."""
        obs = TracingProvider()
        set_observability(obs)

        @traced()
        async def async_func(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        result = await async_func(5)
        assert result == 10

        completed = obs.get_completed_spans()
        assert len(completed) == 1
        assert completed[0].name == "async_func"

    def test_traced_with_custom_name(self) -> None:
        """Test tracing with custom span name."""
        obs = TracingProvider()
        set_observability(obs)

        @traced("custom_operation", kind=SpanKind.CLIENT)
        def my_func() -> str:
            return "done"

        my_func()

        completed = obs.get_completed_spans()
        assert completed[0].name == "custom_operation"
        assert completed[0].kind == SpanKind.CLIENT


class TestGlobalObservability:
    """Tests for global observability instance."""

    def test_get_observability_singleton(self) -> None:
        """Test global instance is a singleton."""
        obs1 = get_observability()
        obs2 = get_observability()
        assert obs1 is obs2

    def test_set_observability(self) -> None:
        """Test setting custom global instance."""
        custom = TracingProvider()
        set_observability(custom)
        assert get_observability() is custom


class TestSpanEdgeCases:
    """Edge case tests for Span class."""

    def test_duration_ms_without_end_time(self) -> None:
        """Test duration_ms returns elapsed time when span not ended (covers line 68)."""
        span = Span(name="test", trace_id="t", span_id="s")
        # Don't end the span - should return elapsed time since start (not 0)
        # The implementation uses current time when end_time is None
        assert span.duration_ms >= 0.0  # Should be small but non-zero

    def test_add_event_without_attributes(self) -> None:
        """Test adding event without attributes."""
        span = Span(name="test", trace_id="t", span_id="s")
        span.add_event("simple_event")
        assert len(span.events) == 1
        assert span.events[0]["attributes"] == {}


class TestTracingProviderEdgeCases:
    """Edge case tests for TracingProvider."""

    def test_span_without_trace(self) -> None:
        """Test creating span without starting trace first."""
        obs = TracingProvider()
        # Don't start trace
        with obs.span("test") as span:
            assert span.trace_id is not None

    def test_get_current_span_empty(self) -> None:
        """Test get_current_span when no span active."""
        obs = TracingProvider()
        assert obs.get_current_span() is None

    def test_get_current_trace_id_empty(self) -> None:
        """Test get_current_trace_id when no trace active."""
        obs = TracingProvider()
        assert obs.get_current_trace_id() is None
