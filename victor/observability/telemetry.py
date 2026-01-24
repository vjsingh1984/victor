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

"""Comprehensive OpenTelemetry integration for Victor AI.

This module provides automatic span creation for:
- LLM calls (prompts, responses, tokens)
- Tool execution
- Workflow execution
- Error tracking and reporting
- Custom instrumentation

Supports multiple exporters:
- OTLP (OpenTelemetry Protocol)
- Console (development)
- Jaeger (legacy)

Usage:
    from victor.observability.telemetry import (
        setup_opentelemetry,
        get_tracer,
        trace_llm_call,
        trace_tool_execution,
    )

    # Initialize at startup
    setup_opentelemetry(
        service_name="victor",
        service_version="0.5.1",
        otlp_endpoint="http://localhost:4317",
    )

    # Trace LLM calls
    with trace_llm_call("anthropic", "claude-sonnet-4-5") as span:
        response = await provider.chat(messages)
        span.set_response(response)

    # Trace tool execution
    @trace_tool_execution("read_file")
    async def read_file(path: str):
        with open(path) as f:
            return f.read()
"""

from __future__ import annotations

import functools
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Optional, Callable, Dict, List, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Track OpenTelemetry availability
_otel_available = False
_otel_configured = False
_tracer: Optional[Any] = None
_meter: Optional[Any] = None

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        PeriodicExportingMetricReader,
        ConsoleMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.context import Context

    _otel_available = True
except ImportError:
    logger.debug(
        "OpenTelemetry not available. Install with: "
        "pip install opentelemetry-sdk opentelemetry-exporter-otlp"
    )


def is_telemetry_enabled() -> bool:
    """Check if OpenTelemetry is available and configured.

    Returns:
        True if telemetry is enabled and configured.
    """
    return _otel_available and _otel_configured


def setup_opentelemetry(
    service_name: str = "victor",
    service_version: str = "0.5.1",
    otlp_endpoint: Optional[str] = None,
    enable_tracing: bool = True,
    enable_metrics: bool = True,
    exporter: str = "otlp",
    sampling_rate: float = 1.0,
    batch_timeout: int = 30,
    batch_max_size: int = 512,
) -> tuple[Optional[Any], Optional[Any]]:
    """Initialize OpenTelemetry tracing and metrics.

    Sets up comprehensive tracing with automatic instrumentation for:
    - LLM provider calls
    - Tool execution
    - Workflow orchestration
    - Custom spans

    Args:
        service_name: Name of the service.
        service_version: Version of the service.
        otlp_endpoint: OTLP collector endpoint (from env if not provided).
        enable_tracing: Enable distributed tracing.
        enable_metrics: Enable metrics collection.
        exporter: Exporter type (otlp, console, jaeger).
        sampling_rate: Trace sampling rate (0.0 to 1.0).
        batch_timeout: Batch export timeout in seconds.
        batch_max_size: Maximum batch size for exports.

    Returns:
        Tuple of (tracer, meter) or (None, None) if unavailable.

    Example:
        tracer, meter = setup_opentelemetry(
            service_name="victor",
            otlp_endpoint="http://jaeger:4317",
            sampling_rate=0.5,
        )
    """
    global _otel_configured, _tracer, _meter

    if not _otel_available:
        logger.warning("OpenTelemetry SDK not installed")
        return None, None

    if _otel_configured:
        logger.debug("OpenTelemetry already configured")
        return _tracer, _meter

    # Get endpoint from environment if not provided
    endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    try:
        # Create resource with service information
        resource = Resource.create(
            {
                SERVICE_NAME: service_name,
                SERVICE_VERSION: service_version,
                "service.namespace": os.getenv("VICTOR_ENV", "development"),
                "deployment.environment": os.getenv("VICTOR_ENV", "development"),
            }
        )

        # Setup tracing
        if enable_tracing:
            tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(tracer_provider)

            # Configure exporter
            if exporter == "console":
                span_exporter = ConsoleSpanExporter()
                tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
                logger.info("OpenTelemetry tracing enabled with console exporter")
            elif exporter == "otlp":
                span_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
                processor = BatchSpanProcessor(
                    span_exporter,
                    max_queue_size=batch_max_size * 2,
                    schedule_delay_millis=batch_timeout * 1000,
                    max_export_batch_size=batch_max_size,
                )
                tracer_provider.add_span_processor(processor)
                logger.info(f"OpenTelemetry tracing enabled with OTLP exporter: {endpoint}")
            else:
                logger.warning(f"Unknown exporter type: {exporter}, tracing disabled")

            _tracer = trace.get_tracer(service_name, service_version)

        # Setup metrics
        if enable_metrics:
            if exporter == "console":
                metric_reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
            elif exporter == "otlp":
                metric_exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
                metric_reader = PeriodicExportingMetricReader(
                    metric_exporter,
                    export_interval_millis=60000,
                )
            else:
                metric_reader = None

            if metric_reader:
                meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
                metrics.set_meter_provider(meter_provider)
                _meter = metrics.get_meter(service_name, service_version)
                logger.info("OpenTelemetry metrics enabled")
            else:
                logger.warning("Metrics disabled (no valid exporter)")

        _otel_configured = True
        return _tracer, _meter

    except Exception as e:
        logger.error(f"Failed to setup OpenTelemetry: {e}", exc_info=True)
        return None, None


def get_tracer(name: Optional[str] = None) -> Optional[Any]:
    """Get a tracer instance for instrumentation.

    Args:
        name: Optional tracer name.

    Returns:
        Tracer instance or None if not configured.
    """
    if not _otel_available:
        return None

    if _tracer:
        return _tracer

    if _otel_configured:
        from opentelemetry import trace

        return trace.get_tracer(name or "victor")

    return None


def get_meter(name: Optional[str] = None) -> Optional[Any]:
    """Get a meter instance for metrics.

    Args:
        name: Optional meter name.

    Returns:
        Meter instance or None if not configured.
    """
    if not _otel_available:
        return None

    if _meter:
        return _meter

    if _otel_configured:
        from opentelemetry import metrics

        return metrics.get_meter(name or "victor")

    return None


@contextmanager
def trace_llm_call(
    provider: str,
    model: str,
    tracer: Optional[Any] = None,
):
    """Context manager for tracing LLM calls.

    Automatically captures:
    - Provider and model name
    - Prompt length and token count
    - Response details and tokens
    - Latency and error status

    Args:
        provider: LLM provider name (e.g., "anthropic", "openai").
        model: Model name (e.g., "claude-sonnet-4-5").
        tracer: Optional tracer instance.

    Example:
        with trace_llm_call("anthropic", "claude-sonnet-4-5") as span:
            response = await provider.chat(messages)
            span.set_response(response)
    """
    if not is_telemetry_enabled():
        yield _DummySpan()
        return

    tracer = tracer or get_tracer("victor.llm")
    if tracer is None:
        yield _DummySpan()
        return

    with tracer.start_as_current_span(f"llm.{provider}.{model}") as span:
        span.set_attribute("llm.provider", provider)
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.call_type", "chat")

        yield _LLMSpan(span)


@contextmanager
def trace_tool_execution(
    tool_name: str,
    tracer: Optional[Any] = None,
):
    """Context manager for tracing tool execution.

    Automatically captures:
    - Tool name and arguments
    - Execution duration
    - Success/failure status
    - Error details if applicable

    Args:
        tool_name: Name of the tool.
        tracer: Optional tracer instance.

    Example:
        with trace_tool_execution("read_file") as span:
            result = await tool.execute(path="/tmp/file.txt")
            span.set_result(result)
    """
    if not is_telemetry_enabled():
        yield _DummySpan()
        return

    tracer = tracer or get_tracer("victor.tools")
    if tracer is None:
        yield _DummySpan()
        return

    with tracer.start_as_current_span(f"tool.{tool_name}") as span:
        span.set_attribute("tool.name", tool_name)

        yield _ToolSpan(span)


@contextmanager
def trace_workflow(
    workflow_name: str,
    tracer: Optional[Any] = None,
):
    """Context manager for tracing workflow execution.

    Automatically captures:
    - Workflow name
    - Node execution
    - Total duration
    - Success/failure status

    Args:
        workflow_name: Name of the workflow.
        tracer: Optional tracer instance.

    Example:
        with trace_workflow("code_review") as span:
            result = await workflow.execute(context)
            span.set_result(result)
    """
    if not is_telemetry_enabled():
        yield _DummySpan()
        return

    tracer = tracer or get_tracer("victor.workflows")
    if tracer is None:
        yield _DummySpan()
        return

    with tracer.start_as_current_span(f"workflow.{workflow_name}") as span:
        span.set_attribute("workflow.name", workflow_name)

        yield _WorkflowSpan(span)


def trace_function(
    name: Optional[str] = None,
    tracer: Optional[Any] = None,
):
    """Decorator for automatic function tracing.

    Args:
        name: Span name (uses function name if not provided).
        tracer: Optional tracer instance.

    Example:
        @trace_function()
        async def process_request(request):
            # Function execution is automatically traced
            return await handler(request)
    """

    def decorator(func: Callable) -> Callable:
        if not is_telemetry_enabled():
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t = tracer or get_tracer(func.__module__)
            if t is None:
                return func(*args, **kwargs)

            span_name = name or f"{func.__module__}.{func.__name__}"

            with t.start_as_current_span(span_name) as span:
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            t = tracer or get_tracer(func.__module__)
            if t is None:
                return await func(*args, **kwargs)

            span_name = name or f"{func.__module__}.{func.__name__}"

            with t.start_as_current_span(span_name) as span:
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# =============================================================================
# Span Helper Classes
# =============================================================================


class _LLMSpan:
    """Helper class for LLM span attributes."""

    def __init__(self, span: Any):
        self._span = span
        self._start_time = time.time()

    def set_prompt(self, messages: List[Dict[str, Any]], tokens: Optional[int] = None):
        """Set prompt attributes.

        Args:
            messages: List of prompt messages.
            tokens: Estimated prompt tokens.
        """
        self._span.set_attribute("llm.prompt.messages_count", len(messages))
        if tokens:
            self._span.set_attribute("llm.prompt.tokens", tokens)

    def set_response(
        self,
        content: str,
        tokens: Optional[int] = None,
        finish_reason: Optional[str] = None,
    ):
        """Set response attributes.

        Args:
            content: Response content.
            tokens: Completion tokens used.
            finish_reason: Reason for completion.
        """
        duration_ms = (time.time() - self._start_time) * 1000
        self._span.set_attribute("llm.response.duration_ms", round(duration_ms, 2))
        self._span.set_attribute("llm.response.content_length", len(content))

        if tokens:
            self._span.set_attribute("llm.response.tokens", tokens)
        if finish_reason:
            self._span.set_attribute("llm.response.finish_reason", finish_reason)

    def set_error(self, error: Exception):
        """Set error attributes.

        Args:
            error: Exception that occurred.
        """
        self._span.set_status(Status(StatusCode.ERROR, str(error)))
        self._span.record_exception(error)


class _ToolSpan:
    """Helper class for tool span attributes."""

    def __init__(self, span: Any):
        self._span = span
        self._start_time = time.time()

    def set_arguments(self, **kwargs):
        """Set tool arguments.

        Args:
            **kwargs: Tool arguments.
        """
        for key, value in kwargs.items():
            if isinstance(value, (str, int, float, bool)):
                self._span.set_attribute(f"tool.args.{key}", str(value))

    def set_result(self, result: Any, success: bool = True):
        """Set result attributes.

        Args:
            result: Tool execution result.
            success: Whether execution was successful.
        """
        duration_ms = (time.time() - self._start_time) * 1000
        self._span.set_attribute("tool.duration_ms", round(duration_ms, 2))
        self._span.set_attribute("tool.success", success)

        if success:
            self._span.set_status(Status(StatusCode.OK))
        else:
            self._span.set_status(Status(StatusCode.ERROR, "Tool execution failed"))


class _WorkflowSpan:
    """Helper class for workflow span attributes."""

    def __init__(self, span: Any):
        self._span = span
        self._start_time = time.time()

    def set_node(self, node_name: str, node_type: str):
        """Set current workflow node.

        Args:
            node_name: Node name.
            node_type: Node type (agent, compute, condition, etc.).
        """
        self._span.set_attribute("workflow.current_node", node_name)
        self._span.set_attribute("workflow.current_node_type", node_type)

    def set_result(self, result: Any, success: bool = True):
        """Set workflow result.

        Args:
            result: Workflow result.
            success: Whether workflow completed successfully.
        """
        duration_ms = (time.time() - self._start_time) * 1000
        self._span.set_attribute("workflow.duration_ms", round(duration_ms, 2))
        self._span.set_attribute("workflow.success", success)

        if success:
            self._span.set_status(Status(StatusCode.OK))
        else:
            self._span.set_status(Status(StatusCode.ERROR, "Workflow failed"))


class _DummySpan:
    """No-op span for when telemetry is disabled."""

    def set_prompt(self, *args, **kwargs):
        pass

    def set_response(self, *args, **kwargs):
        pass

    def set_error(self, *args, **kwargs):
        pass

    def set_arguments(self, *args, **kwargs):
        pass

    def set_result(self, *args, **kwargs):
        pass

    def set_node(self, *args, **kwargs):
        pass


# Import asyncio for async function detection
import asyncio
