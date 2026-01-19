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

"""Distributed tracing system for production observability.

This module provides DistributedTracer which enables end-to-end tracing
of requests across coordinators, tools, and external services.

Features:
- OpenTelemetry integration for distributed tracing
- Automatic trace context propagation
- Span lifecycle management
- Performance bottleneck identification
- Request flow visualization

Design Patterns:
- Facade Pattern: Unified tracing interface
- Context Manager Pattern: Automatic span management
- Decorator Pattern: Easy function instrumentation

Example:
    from victor.observability.distributed_tracing import (
        DistributedTracer,
        trace_function,
        trace_method,
    )

    # Create tracer
    tracer = DistributedTracer("victor.agent")

    # Context manager
    with tracer.start_span("process_request") as span:
        span.set_attribute("user_id", "123")
        # Nested spans
        with tracer.start_span("call_llm", parent=span):
            ...

    # Decorator
    @trace_function("my_function")
    def my_function(arg1, arg2):
        ...
"""

from __future__ import annotations

import contextvars
import functools
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    try:
        from opentelemetry import trace
        from opentelemetry.trace import Span, Status, StatusCode
    except ImportError:
        trace = None
        Span = None
        Status = None
        StatusCode = None

logger = logging.getLogger(__name__)

# =============================================================================
# Trace Context
# =============================================================================

_trace_context: contextvars.ContextVar[Optional["TraceContext"]] = contextvars.ContextVar(
    "_trace_context", default=None
)


class SpanKind(str, Enum):
    """Span kinds for categorizing spans."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


@dataclass
class TraceContext:
    """Trace context for propagation.

    Attributes:
        trace_id: Unique trace identifier
        span_id: Current span identifier
        parent_span_id: Parent span identifier
        baggage: Key-value metadata for propagation
    """

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)

    def child_span(self, span_id: Optional[str] = None) -> "TraceContext":
        """Create child trace context.

        Args:
            span_id: Optional child span ID

        Returns:
            New TraceContext for child span
        """
        return TraceContext(
            trace_id=self.trace_id,
            span_id=span_id or uuid.uuid4().hex[:16],
            parent_span_id=self.span_id,
            baggage=dict(self.baggage),
        )

    def inject(self, carrier: Dict[str, str]) -> None:
        """Inject trace context into carrier.

        Args:
            carrier: Dictionary to inject context into
        """
        carrier["trace_id"] = self.trace_id
        carrier["span_id"] = self.span_id
        carrier["parent_span_id"] = self.parent_span_id or ""

        # Inject baggage
        for key, value in self.baggage.items():
            carrier[f"baggage_{key}"] = value

    @classmethod
    def extract(cls, carrier: Dict[str, str]) -> Optional["TraceContext"]:
        """Extract trace context from carrier.

        Args:
            carrier: Dictionary containing trace context

        Returns:
            TraceContext or None if not found
        """
        trace_id = carrier.get("trace_id")
        if not trace_id:
            return None

        span_id = carrier.get("span_id", uuid.uuid4().hex[:16])
        parent_span_id = carrier.get("parent_span_id") or None

        # Extract baggage
        baggage = {}
        for key, value in carrier.items():
            if key.startswith("baggage_"):
                baggage[key[8:]] = value

        return cls(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=baggage,
        )


@dataclass
class SpanData:
    """Data for a completed span.

    Attributes:
        name: Span name
        trace_id: Trace identifier
        span_id: Span identifier
        parent_span_id: Parent span identifier
        start_time: Span start timestamp
        end_time: Span end timestamp
        duration_ms: Span duration in milliseconds
        attributes: Span attributes
        events: Span events
        status: Span status
        kind: Span kind
    """

    name: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    start_time: datetime
    end_time: datetime
    duration_ms: float
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    kind: SpanKind = SpanKind.INTERNAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
            "kind": self.kind.value,
        }


# =============================================================================
# Distributed Tracer
# =============================================================================


class DistributedTracer:
    """Distributed tracer for request tracing.

    Provides span creation, context management, and integration with
    OpenTelemetry when available.

    Attributes:
        service_name: Service name for traces
        enabled: Whether tracing is enabled
        use_opentelemetry: Whether to use OpenTelemetry

    Example:
        tracer = DistributedTracer("victor.agent")

        with tracer.start_span("process_request") as span:
            span.set_attribute("user_id", "123")

            with tracer.start_span("call_tool"):
                # Tool execution
                ...

            # span automatically ends
    """

    def __init__(
        self,
        service_name: str,
        enabled: bool = True,
        use_opentelemetry: bool = True,
    ) -> None:
        """Initialize distributed tracer.

        Args:
            service_name: Service name for traces
            enabled: Whether tracing is enabled
            use_opentelemetry: Whether to use OpenTelemetry if available
        """
        self.service_name = service_name
        self.enabled = enabled
        self.use_opentelemetry = use_opentelemetry

        # Check if OpenTelemetry is available
        self._otel_available = False
        self._otel_tracer: Optional[Any] = None

        if use_opentelemetry:
            try:
                from victor.observability.telemetry import get_tracer, is_telemetry_enabled

                if is_telemetry_enabled():
                    self._otel_tracer = get_tracer(service_name)
                    self._otel_available = True
                    logger.debug("OpenTelemetry tracing enabled")
            except Exception as e:
                logger.debug(f"OpenTelemetry not available: {e}")

        # Span storage for export
        self._spans: Dict[str, SpanData] = {}
        self._root_context: Optional[TraceContext] = None

    @property
    def current_context(self) -> Optional[TraceContext]:
        """Get current trace context.

        Returns:
            Current TraceContext or None
        """
        return _trace_context.get()

    def start_span(
        self,
        name: str,
        parent: Optional["Span"] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "Span":
        """Start a new span.

        Args:
            name: Span name
            parent: Parent span (if None, uses current context)
            kind: Span kind
            attributes: Initial span attributes

        Returns:
            Started Span object

        Example:
            with tracer.start_span("operation") as span:
                span.set_attribute("key", "value")
                # Do work
        """
        if not self.enabled:
            return NoOpSpan()

        # Get or create trace context
        current_ctx = self.current_context
        if parent:
            ctx = current_ctx.child_span() if current_ctx else TraceContext()
        else:
            ctx = TraceContext()

        # Create span
        span = Span(
            tracer=self,
            name=name,
            context=ctx,
            kind=kind,
            attributes=attributes or {},
        )

        # Set as current context
        _trace_context.set(ctx)

        # Start OpenTelemetry span if available
        if self._otel_available and self._otel_tracer:
            try:
                self._otel_tracer.start_as_current_span(name)
            except Exception as e:
                logger.warning(f"Failed to start OTel span: {e}")

        return span

    def get_trace_context(self) -> Optional[TraceContext]:
        """Get current trace context for propagation.

        Returns:
            Current TraceContext or None
        """
        return self.current_context

    def inject_context(self, carrier: Dict[str, str]) -> None:
        """Inject trace context into carrier for propagation.

        Args:
            carrier: Dictionary to inject context into

        Example:
            headers = {}
            tracer.inject_context(headers)
            # Send headers with request
        """
        ctx = self.current_context
        if ctx:
            ctx.inject(carrier)

    def extract_context(self, carrier: Dict[str, str]) -> Optional[TraceContext]:
        """Extract trace context from carrier.

        Args:
            carrier: Dictionary containing trace context

        Returns:
            Extracted TraceContext or None

        Example:
            headers = request.headers
            ctx = tracer.extract_context(headers)
            if ctx:
                _trace_context.set(ctx)
        """
        ctx = TraceContext.extract(carrier)
        if ctx:
            _trace_context.set(ctx)
        return ctx

    def add_span(self, span_data: SpanData) -> None:
        """Add completed span to storage.

        Args:
            span_data: Completed span data
        """
        self._spans[span_data.span_id] = span_data

    def get_trace(self, trace_id: str) -> list[SpanData]:
        """Get all spans for a trace.

        Args:
            trace_id: Trace identifier

        Returns:
            List of spans in the trace
        """
        return [span for span in self._spans.values() if span.trace_id == trace_id]

    def export_trace(self, trace_id: str) -> Dict[str, Any]:
        """Export trace as JSON.

        Args:
            trace_id: Trace identifier

        Returns:
            Dictionary with trace data
        """
        spans = self.get_trace(trace_id)

        # Sort by start time
        spans.sort(key=lambda s: s.start_time)

        return {
            "trace_id": trace_id,
            "service": self.service_name,
            "spans": [span.to_dict() for span in spans],
            "span_count": len(spans),
            "total_duration_ms": max((s.duration_ms for s in spans), default=0),
        }

    def clear_spans(self) -> None:
        """Clear all stored spans."""
        self._spans.clear()


# =============================================================================
# Span Implementation
# =============================================================================


class Span:
    """Span representing an operation.

    Spans should be used as context managers to ensure proper timing.

    Attributes:
        tracer: Parent tracer
        name: Span name
        context: Trace context
        kind: Span kind
        attributes: Span attributes
        events: Span events
        start_time: Start timestamp
        end_time: End timestamp

    Example:
        with tracer.start_span("operation") as span:
            span.set_attribute("key", "value")
            span.add_event(" milestone")
    """

    def __init__(
        self,
        tracer: DistributedTracer,
        name: str,
        context: TraceContext,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize span.

        Args:
            tracer: Parent tracer
            name: Span name
            context: Trace context
            kind: Span kind
            attributes: Initial attributes
        """
        self.tracer = tracer
        self.name = name
        self.context = context
        self.kind = kind
        self.attributes: Dict[str, Any] = attributes or {}
        self.events: List[Dict[str, Any]] = []
        self.start_time = datetime.now(timezone.utc)
        self.end_time: Optional[datetime] = None
        self.status: str = "ok"
        self._otel_span: Optional[Any] = None

        # Start OpenTelemetry span if available
        if tracer._otel_available and tracer._otel_tracer:
            try:
                self._otel_span = tracer._otel_tracer.start_as_current_span(name)
                if hasattr(self._otel_span, "__enter__"):
                    self._otel_span = self._otel_span.__enter__()
            except Exception as e:
                logger.warning(f"Failed to enter OTel span: {e}")

    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute.

        Args:
            key: Attribute key
            value: Attribute value (must be serializable)
        """
        self.attributes[key] = value

        # Set on OTel span if available
        if self._otel_span and hasattr(self._otel_span, "set_attribute"):
            try:
                self._otel_span.set_attribute(key, value)
            except Exception:
                pass

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Add event to span.

        Args:
            name: Event name
            attributes: Event attributes
            timestamp: Event timestamp (now if None)
        """
        event = {
            "name": name,
            "timestamp": (timestamp or datetime.now(timezone.utc)).isoformat(),
            "attributes": attributes or {},
        }
        self.events.append(event)

        # Add to OTel span if available
        if self._otel_span and hasattr(self._otel_span, "add_event"):
            try:
                self._otel_span.add_event(name, attributes or {})
            except Exception:
                pass

    def set_error(self, exception: Exception) -> None:
        """Mark span as errored.

        Args:
            exception: Exception that occurred
        """
        self.status = "error"
        self.set_attribute("error.type", type(exception).__name__)
        self.set_attribute("error.message", str(exception))

        # Record as event
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
            },
        )

        # Set on OTel span if available
        if self._otel_span and hasattr(self._otel_span, "set_status"):
            try:
                from opentelemetry.trace import Status, StatusCode

                self._otel_span.set_status(Status(StatusCode.ERROR, str(exception)))
            except Exception:
                pass

    def __enter__(self) -> "Span":
        """Enter span context.

        Returns:
            Self
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit span context.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        """
        self.end_time = datetime.now(timezone.utc)
        duration = self.end_time - self.start_time
        duration_ms = duration.total_seconds() * 1000

        # Handle exception
        if exc_val:
            self.set_error(exc_val)

        # Exit OTel span if available
        if self._otel_span and hasattr(self._otel_span, "__exit__"):
            try:
                self._otel_span.__exit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass

        # Store span data
        span_data = SpanData(
            name=self.name,
            trace_id=self.context.trace_id,
            span_id=self.context.span_id,
            parent_span_id=self.context.parent_span_id,
            start_time=self.start_time,
            end_time=self.end_time,
            duration_ms=duration_ms,
            attributes=self.attributes,
            events=self.events,
            status=self.status,
            kind=self.kind,
        )

        self.tracer.add_span(span_data)


class NoOpSpan:
    """No-op span for when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op."""
        pass

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """No-op."""
        pass

    def set_error(self, exception: Exception) -> None:
        """No-op."""
        pass

    def __enter__(self) -> "NoOpSpan":
        """No-op."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """No-op."""
        pass


# =============================================================================
# Decorators
# =============================================================================


def trace_function(
    span_name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
):
    """Decorator to trace function execution.

    Args:
        span_name: Span name (uses function name if None)
        kind: Span kind

    Returns:
        Decorated function

    Example:
        @trace_function("process_data")
        def process_data(data):
            # Automatically traced
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_current_tracer()
            if not tracer or not tracer.enabled:
                return func(*args, **kwargs)

            name = span_name or f"{func.__module__}.{func.__name__}"

            with tracer.start_span(name, kind=kind) as span:
                # Add function arguments as attributes (be careful with sensitive data)
                try:
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    span.set_attribute("function.args_count", len(args))

                    result = func(*args, **kwargs)

                    span.set_attribute("function.success", True)
                    return result

                except Exception as e:
                    span.set_error(e)
                    span.set_attribute("function.success", False)
                    raise

        return wrapper

    return decorator


def trace_async_function(
    span_name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
):
    """Decorator to trace async function execution.

    Args:
        span_name: Span name (uses function name if None)
        kind: Span kind

    Returns:
        Decorated async function

    Example:
        @trace_async_function("fetch_data")
        async def fetch_data(url):
            # Automatically traced
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_current_tracer()
            if not tracer or not tracer.enabled:
                return await func(*args, **kwargs)

            name = span_name or f"{func.__module__}.{func.__name__}"

            with tracer.start_span(name, kind=kind) as span:
                try:
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    span.set_attribute("function.args_count", len(args))

                    result = await func(*args, **kwargs)

                    span.set_attribute("function.success", True)
                    return result

                except Exception as e:
                    span.set_error(e)
                    span.set_attribute("function.success", False)
                    raise

        return wrapper

    return decorator


# =============================================================================
# Global Tracer Management
# =============================================================================

_global_tracer: Optional[DistributedTracer] = None


def set_global_tracer(tracer: DistributedTracer) -> None:
    """Set global tracer instance.

    Args:
        tracer: Tracer instance to set as global
    """
    global _global_tracer
    _global_tracer = tracer


def get_global_tracer() -> Optional[DistributedTracer]:
    """Get global tracer instance.

    Returns:
        Global tracer or None
    """
    return _global_tracer


def get_current_tracer() -> Optional[DistributedTracer]:
    """Get current tracer (global or from context).

    Returns:
        Current tracer or None
    """
    return _global_tracer


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "DistributedTracer",
    "Span",
    "TraceContext",
    "SpanData",
    "SpanKind",
    "trace_function",
    "trace_async_function",
    "set_global_tracer",
    "get_global_tracer",
    "get_current_tracer",
]
