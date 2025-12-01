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

"""Observability hooks for tracing, metrics, and logging.

Provides a lightweight observability layer that can be extended with
OpenTelemetry, Prometheus, or other backends.
"""

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Generator

logger = logging.getLogger(__name__)


class SpanKind(Enum):
    """Type of span in a trace."""

    INTERNAL = "internal"
    CLIENT = "client"  # Outgoing request (e.g., LLM call)
    SERVER = "server"  # Incoming request
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Status of a completed span."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class Span:
    """A trace span representing a unit of work."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    kind: SpanKind = SpanKind.INTERNAL
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span."""
        self.events.append(
            {
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {},
            }
        )

    def end(self, status: SpanStatus = SpanStatus.OK) -> None:
        """End the span."""
        self.end_time = time.time()
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for export."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events,
        }


# Type aliases for hooks
SpanHook = Callable[[Span], None]
MetricHook = Callable[[str, float, Dict[str, str]], None]


class ObservabilityManager:
    """Central manager for observability hooks and tracing.

    Provides:
    - Span creation and management for distributed tracing
    - Hook registration for span lifecycle events
    - Metric recording hooks
    - Integration points for external observability systems

    Usage:
        obs = ObservabilityManager()

        # Register hooks
        obs.on_span_start(lambda span: print(f"Started: {span.name}"))
        obs.on_span_end(lambda span: print(f"Ended: {span.name} ({span.duration_ms:.2f}ms)"))

        # Create spans
        with obs.span("my_operation") as span:
            span.set_attribute("key", "value")
            # ... do work ...
    """

    def __init__(self) -> None:
        """Initialize the observability manager."""
        self._current_trace_id: Optional[str] = None
        self._current_span: Optional[Span] = None
        self._span_stack: List[Span] = []

        # Hooks
        self._on_span_start: List[SpanHook] = []
        self._on_span_end: List[SpanHook] = []
        self._on_metric: List[MetricHook] = []

        # Completed spans (for export/debugging)
        self._completed_spans: List[Span] = []
        self._max_completed_spans = 1000

    def on_span_start(self, hook: SpanHook) -> None:
        """Register a hook to be called when a span starts."""
        self._on_span_start.append(hook)

    def on_span_end(self, hook: SpanHook) -> None:
        """Register a hook to be called when a span ends."""
        self._on_span_end.append(hook)

    def on_metric(self, hook: MetricHook) -> None:
        """Register a hook for metric recording."""
        self._on_metric.append(hook)

    def start_trace(self) -> str:
        """Start a new trace and return the trace ID."""
        self._current_trace_id = str(uuid.uuid4())
        return self._current_trace_id

    def get_current_trace_id(self) -> Optional[str]:
        """Get the current trace ID."""
        return self._current_trace_id

    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return self._current_span

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """Create a span context manager.

        Args:
            name: Name of the span
            kind: Type of span
            attributes: Initial span attributes

        Yields:
            The active Span object
        """
        # Ensure we have a trace
        if not self._current_trace_id:
            self.start_trace()

        # Create span
        parent_span_id = self._current_span.span_id if self._current_span else None
        span = Span(
            name=name,
            trace_id=self._current_trace_id or str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_span_id,
            kind=kind,
            attributes=attributes or {},
        )

        # Push onto stack
        self._span_stack.append(span)
        self._current_span = span

        # Call start hooks
        for hook in self._on_span_start:
            try:
                hook(span)
            except Exception as e:
                logger.warning(f"Span start hook error: {e}")

        try:
            yield span
            span.end(SpanStatus.OK)
        except Exception as e:
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            span.end(SpanStatus.ERROR)
            raise
        finally:
            # Call end hooks
            for hook in self._on_span_end:
                try:
                    hook(span)
                except Exception as e:
                    logger.warning(f"Span end hook error: {e}")

            # Pop from stack
            self._span_stack.pop()
            self._current_span = self._span_stack[-1] if self._span_stack else None

            # Store completed span
            self._completed_spans.append(span)
            if len(self._completed_spans) > self._max_completed_spans:
                self._completed_spans = self._completed_spans[-self._max_completed_spans :]

    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels/tags
        """
        labels = labels or {}
        for hook in self._on_metric:
            try:
                hook(name, value, labels)
            except Exception as e:
                logger.warning(f"Metric hook error: {e}")

    def get_completed_spans(self) -> List[Span]:
        """Get recently completed spans."""
        return self._completed_spans.copy()

    def clear_completed_spans(self) -> None:
        """Clear the completed spans buffer."""
        self._completed_spans.clear()

    def end_trace(self) -> None:
        """End the current trace."""
        self._current_trace_id = None
        self._current_span = None
        self._span_stack.clear()


# Global instance for convenience
_global_observability: Optional[ObservabilityManager] = None


def get_observability() -> ObservabilityManager:
    """Get the global observability manager instance."""
    global _global_observability
    if _global_observability is None:
        _global_observability = ObservabilityManager()
    return _global_observability


def set_observability(manager: ObservabilityManager) -> None:
    """Set the global observability manager instance."""
    global _global_observability
    _global_observability = manager


# Convenience decorators
def traced(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Callable:
    """Decorator to trace a function.

    Args:
        name: Span name (defaults to function name)
        kind: Type of span

    Usage:
        @traced()
        async def my_function():
            ...

        @traced("custom_name", kind=SpanKind.CLIENT)
        def call_external_api():
            ...
    """

    def decorator(func: Callable) -> Callable:
        import functools
        import asyncio

        span_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            obs = get_observability()
            with obs.span(span_name, kind=kind) as span:
                span.set_attribute("function", func.__name__)
                span.set_attribute("module", func.__module__)
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            obs = get_observability()
            with obs.span(span_name, kind=kind) as span:
                span.set_attribute("function", func.__name__)
                span.set_attribute("module", func.__module__)
                return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
