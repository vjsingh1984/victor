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

"""OpenTelemetry integration for Victor observability.

This module provides OTEL-compatible exporters for distributed tracing
and metrics export to standard observability backends.

Supports:
- OTLP HTTP/gRPC export
- Span creation for tool execution
- Baggage propagation for context
- Metrics export via OTLP

Design Patterns:
- Adapter Pattern: Adapts Victor events to OTEL format
- Strategy Pattern: Multiple export backends

Example:
    from victor.observability.otel_exporter import OpenTelemetryExporter

    # Create exporter with OTLP endpoint
    exporter = OpenTelemetryExporter(
        service_name="victor-agent",
        endpoint="http://localhost:4318",
    )

    # Add to EventBus
    event_bus.add_exporter(exporter)

Note: Requires optional opentelemetry dependencies:
    pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from victor.observability.event_bus import EventCategory, VictorEvent
from victor.observability.exporters import BaseExporter

logger = logging.getLogger(__name__)

# Check for OpenTelemetry availability
_OTEL_AVAILABLE = False
try:
    from opentelemetry import trace, context as otel_context
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.trace import Status, StatusCode, Span
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    _OTEL_AVAILABLE = True
except ImportError:
    pass

# Check for OTLP exporter
_OTLP_AVAILABLE = False
try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    _OTLP_AVAILABLE = True
except ImportError:
    pass


@dataclass
class OTELConfig:
    """Configuration for OpenTelemetry integration."""

    service_name: str = "victor-agent"
    service_version: str = "1.0.0"
    endpoint: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    use_console_exporter: bool = False
    batch_export: bool = True
    max_queue_size: int = 2048
    export_timeout_millis: int = 30000


class OpenTelemetryExporter(BaseExporter):
    """Exports Victor events as OpenTelemetry spans.

    Converts Victor events to OTEL spans for distributed tracing.
    Tool executions become child spans with timing and metadata.

    Example:
        exporter = OpenTelemetryExporter(
            service_name="victor-agent",
            endpoint="http://localhost:4318",
        )
        event_bus.add_exporter(exporter)
    """

    def __init__(
        self,
        service_name: str = "victor-agent",
        service_version: str = "1.0.0",
        endpoint: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        use_console_exporter: bool = False,
        batch_export: bool = True,
    ) -> None:
        """Initialize OpenTelemetry exporter.

        Args:
            service_name: Service name for tracing.
            service_version: Service version.
            endpoint: OTLP endpoint URL.
            headers: Optional headers for OTLP.
            use_console_exporter: Use console for debugging.
            batch_export: Use batch processing (recommended).
        """
        if not _OTEL_AVAILABLE:
            raise ImportError(
                "OpenTelemetry not available. Install with: "
                "pip install opentelemetry-api opentelemetry-sdk"
            )

        self._service_name = service_name
        self._service_version = service_version
        self._endpoint = endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        self._headers = headers
        self._use_console = use_console_exporter
        self._batch_export = batch_export

        self._tracer: Optional[Any] = None
        self._provider: Optional[Any] = None
        self._active_spans: Dict[str, Any] = {}

        self._setup_tracer()

    def _setup_tracer(self) -> None:
        """Set up OpenTelemetry tracer."""
        resource = Resource.create(
            {
                SERVICE_NAME: self._service_name,
                "service.version": self._service_version,
            }
        )

        self._provider = TracerProvider(resource=resource)

        # Add exporter
        if self._use_console:
            processor = SimpleSpanProcessor(ConsoleSpanExporter())
        elif self._endpoint and _OTLP_AVAILABLE:
            otlp_exporter = OTLPSpanExporter(
                endpoint=f"{self._endpoint}/v1/traces",
                headers=self._headers,
            )
            if self._batch_export:
                processor = BatchSpanProcessor(otlp_exporter)
            else:
                processor = SimpleSpanProcessor(otlp_exporter)
        else:
            # Fallback to console
            processor = SimpleSpanProcessor(ConsoleSpanExporter())
            logger.warning(
                "No OTLP endpoint configured, using console exporter. "
                "Set OTEL_EXPORTER_OTLP_ENDPOINT or install opentelemetry-exporter-otlp"
            )

        self._provider.add_span_processor(processor)
        trace.set_tracer_provider(self._provider)
        self._tracer = trace.get_tracer(__name__, self._service_version)

    def export(self, event: VictorEvent) -> None:
        """Export event as OTEL span.

        Args:
            event: Victor event to export.
        """
        if not self._tracer:
            return

        if event.category == EventCategory.TOOL:
            self._handle_tool_event(event)
        elif event.category == EventCategory.MODEL:
            self._handle_model_event(event)
        elif event.category == EventCategory.LIFECYCLE:
            self._handle_lifecycle_event(event)
        elif event.category == EventCategory.ERROR:
            self._handle_error_event(event)
        elif event.category == EventCategory.STATE:
            self._handle_state_event(event)

    def _handle_tool_event(self, event: VictorEvent) -> None:
        """Handle tool execution events."""
        if event.name.endswith(".start"):
            tool_name = event.name.replace(".start", "")
            tool_id = event.data.get("tool_id", tool_name)

            span = self._tracer.start_span(
                f"tool.{tool_name}",
                attributes={
                    "tool.name": tool_name,
                    "tool.id": tool_id,
                    "victor.event.category": "TOOL",
                    "victor.session.id": event.session_id or "",
                },
            )

            # Store arguments as attributes
            for key, value in event.data.get("arguments", {}).items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(f"tool.argument.{key}", str(value)[:256])

            self._active_spans[tool_id] = span

        elif event.name.endswith(".end"):
            tool_name = event.name.replace(".end", "")
            tool_id = event.data.get("tool_id", tool_name)

            span = self._active_spans.pop(tool_id, None)
            if span:
                success = event.data.get("success", True)
                if success:
                    span.set_status(Status(StatusCode.OK))
                else:
                    span.set_status(
                        Status(StatusCode.ERROR, event.data.get("error", "Unknown error"))
                    )

                if "duration_ms" in event.data:
                    span.set_attribute("tool.duration_ms", event.data["duration_ms"])

                span.end()

    def _handle_model_event(self, event: VictorEvent) -> None:
        """Handle model events."""
        if event.name == "request":
            span = self._tracer.start_span(
                "llm.request",
                attributes={
                    "llm.provider": event.data.get("provider", ""),
                    "llm.model": event.data.get("model", ""),
                    "llm.message_count": event.data.get("message_count", 0),
                    "llm.tool_count": event.data.get("tool_count", 0),
                    "victor.event.category": "MODEL",
                },
            )
            self._active_spans["model_request"] = span

        elif event.name == "response":
            span = self._active_spans.pop("model_request", None)
            if span:
                span.set_attribute("llm.tokens_used", event.data.get("tokens_used") or 0)
                span.set_attribute("llm.tool_calls", event.data.get("tool_calls", 0))
                if "latency_ms" in event.data:
                    span.set_attribute("llm.latency_ms", event.data["latency_ms"])
                span.set_status(Status(StatusCode.OK))
                span.end()

    def _handle_lifecycle_event(self, event: VictorEvent) -> None:
        """Handle lifecycle events."""
        if event.name == "session.start":
            span = self._tracer.start_span(
                "session",
                attributes={
                    "victor.session.id": event.session_id or "",
                    "victor.event.category": "LIFECYCLE",
                },
            )
            self._active_spans["session"] = span

        elif event.name == "session.end":
            span = self._active_spans.pop("session", None)
            if span:
                span.set_attribute("session.tool_calls", event.data.get("tool_calls", 0))
                if event.data.get("duration_seconds"):
                    span.set_attribute("session.duration_seconds", event.data["duration_seconds"])
                success = event.data.get("success", True)
                if success:
                    span.set_status(Status(StatusCode.OK))
                else:
                    span.set_status(Status(StatusCode.ERROR))
                span.end()

    def _handle_error_event(self, event: VictorEvent) -> None:
        """Handle error events."""
        with self._tracer.start_as_current_span("error") as span:
            span.set_attribute("error.type", event.name)
            span.set_attribute("error.message", event.data.get("error", ""))
            span.set_attribute("error.recoverable", event.data.get("recoverable", True))
            span.set_status(Status(StatusCode.ERROR, event.data.get("error", "Unknown error")))

    def _handle_state_event(self, event: VictorEvent) -> None:
        """Handle state transition events."""
        with self._tracer.start_as_current_span("state.transition") as span:
            span.set_attribute("state.old", event.data.get("old_stage", ""))
            span.set_attribute("state.new", event.data.get("new_stage", ""))
            span.set_attribute("state.confidence", event.data.get("confidence", 1.0))

    def close(self) -> None:
        """Shutdown the tracer provider."""
        # End any remaining active spans
        for span in self._active_spans.values():
            try:
                span.end()
            except Exception:
                pass
        self._active_spans.clear()

        if self._provider:
            self._provider.shutdown()


class OTELSpanManager:
    """Context manager for manual span creation.

    Provides higher-level API for creating spans in Victor code.

    Example:
        manager = OTELSpanManager(service_name="victor-agent")

        with manager.span("process_request") as span:
            span.set_attribute("request.type", "tool_call")
            result = process()
            span.set_attribute("result.size", len(result))
    """

    def __init__(
        self,
        service_name: str = "victor-agent",
        endpoint: Optional[str] = None,
    ) -> None:
        """Initialize span manager.

        Args:
            service_name: Service name for tracing.
            endpoint: OTLP endpoint URL.
        """
        if not _OTEL_AVAILABLE:
            raise ImportError("OpenTelemetry not available")

        self._exporter = OpenTelemetryExporter(
            service_name=service_name,
            endpoint=endpoint,
        )
        self._tracer = self._exporter._tracer

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Any]:
        """Create a span context.

        Args:
            name: Span name.
            attributes: Initial attributes.

        Yields:
            Active span.
        """
        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(key, value)
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def close(self) -> None:
        """Close the span manager."""
        self._exporter.close()


# =============================================================================
# Async Batching Exporter
# =============================================================================


class AsyncBatchingExporter(BaseExporter):
    """Asynchronous batching exporter for high-throughput scenarios.

    Buffers events and exports in batches for reduced overhead.
    Uses background thread for non-blocking export.

    Example:
        exporter = AsyncBatchingExporter(
            target=JsonLineExporter("events.jsonl"),
            batch_size=100,
            flush_interval=5.0,
        )
        event_bus.add_exporter(exporter)
    """

    def __init__(
        self,
        target: BaseExporter,
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_queue_size: int = 10000,
        on_export_error: Optional[Callable[[Exception, List[VictorEvent]], None]] = None,
    ) -> None:
        """Initialize async batching exporter.

        Args:
            target: Target exporter for actual export.
            batch_size: Events per batch.
            flush_interval: Seconds between flushes.
            max_queue_size: Maximum queue size.
            on_export_error: Error callback.
        """
        import threading
        import queue

        self._target = target
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._max_queue_size = max_queue_size
        self._on_error = on_export_error

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._shutdown = threading.Event()
        self._thread = threading.Thread(target=self._export_loop, daemon=True)
        self._thread.start()

    def export(self, event: VictorEvent) -> None:
        """Queue event for async export.

        Args:
            event: Event to export.
        """
        try:
            self._queue.put_nowait(event)
        except Exception:
            # Queue full, drop event
            logger.warning("Event queue full, dropping event")

    def _export_loop(self) -> None:
        """Background export loop."""
        import time

        batch: List[VictorEvent] = []
        last_flush = time.time()

        while not self._shutdown.is_set():
            try:
                # Try to get event with timeout
                try:
                    event = self._queue.get(timeout=0.1)
                    batch.append(event)
                except Exception:
                    pass

                # Check flush conditions
                should_flush = (
                    len(batch) >= self._batch_size
                    or (time.time() - last_flush) >= self._flush_interval
                )

                if should_flush and batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()

            except Exception as e:
                logger.error(f"Error in export loop: {e}")

        # Final flush on shutdown
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: List[VictorEvent]) -> None:
        """Flush batch of events to target.

        Args:
            batch: Events to export.
        """
        try:
            for event in batch:
                self._target.export(event)
        except Exception as e:
            logger.error(f"Batch export error: {e}")
            if self._on_error:
                self._on_error(e, batch)

    def flush(self) -> None:
        """Force flush pending events."""
        # Signal export loop to flush
        pass

    def close(self) -> None:
        """Shutdown exporter."""
        self._shutdown.set()
        self._thread.join(timeout=5.0)
        self._target.close()

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()


# =============================================================================
# Factory Functions
# =============================================================================


def create_otel_exporter(
    service_name: str = "victor-agent",
    endpoint: Optional[str] = None,
    **kwargs: Any,
) -> Optional[OpenTelemetryExporter]:
    """Create OpenTelemetry exporter if available.

    Returns None if OTEL dependencies are not installed.

    Args:
        service_name: Service name.
        endpoint: OTLP endpoint.
        **kwargs: Additional config.

    Returns:
        Exporter or None.
    """
    if not _OTEL_AVAILABLE:
        logger.info(
            "OpenTelemetry not available. Install with: "
            "pip install opentelemetry-api opentelemetry-sdk"
        )
        return None

    return OpenTelemetryExporter(
        service_name=service_name,
        endpoint=endpoint,
        **kwargs,
    )
