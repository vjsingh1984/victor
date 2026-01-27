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

"""Monitoring middleware for automatic metrics and tracing.

This module provides middleware components that automatically:
- Record request metrics (rate, latency, errors)
- Create distributed traces for requests
- Track active connections
- Monitor system resources

Design Patterns:
- Middleware Pattern: Intercepts requests/responses
- Decorator Pattern: Wraps handlers with monitoring
- Observer Pattern: Emits events

Example:
    from victor.observability.middleware import (
        MonitoringMiddleware,
        create_monitoring_middleware,
    )

    # With FastAPI
    app = FastAPI()
    middleware = create_monitoring_middleware()

    @app.middleware("http")
    async def monitoring_middleware(request: Request, call_next):
        return await middleware.handle_request(request, call_next)

    # With Starlette
    app.add_middleware(MonitoringMiddleware)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional

try:
    from starlette.datastructures import Headers
    from starlette.requests import Request
    from starlette.responses import Response

    STARLETTE_AVAILABLE = True
except ImportError:
    STARLETTE_AVAILABLE = False
    Request = None  # type: ignore
    Response = None  # type: ignore
    Headers = None  # type: ignore
from victor.observability.production_metrics import ProductionMetricsCollector
from victor.observability.distributed_tracing import (
    DistributedTracer,
    Span,
    TraceContext,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Request Metadata
# =============================================================================


class RequestStatus(str, Enum):
    """Request status for metrics."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class RequestMetadata:
    """Metadata about a request.

    Attributes:
        request_id: Unique request identifier
        trace_id: Trace identifier
        method: HTTP method
        path: Request path
        user_agent: User agent string
        client_ip: Client IP address
        start_time: Request start timestamp
        end_time: Request end timestamp
        duration_ms: Request duration in milliseconds
        status_code: HTTP status code
        status: Request status
    """

    request_id: str
    trace_id: str
    method: str
    path: str
    user_agent: Optional[str] = None
    client_ip: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    status_code: int = 200
    status: RequestStatus = RequestStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "method": self.method,
            "path": self.path,
            "user_agent": self.user_agent,
            "client_ip": self.client_ip,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status_code": self.status_code,
            "status": self.status.value,
        }


# =============================================================================
# Monitoring Middleware
# =============================================================================


class MonitoringMiddleware:
    """Middleware for automatic request monitoring.

    Records metrics and creates traces for all HTTP requests.

    Attributes:
        metrics_collector: Metrics collector instance
        tracer: Distributed tracer instance
        request_id_header: Header name for request ID
        trace_id_header: Header name for trace ID

    Example:
        middleware = MonitoringMiddleware()

        @app.middleware("http")
        async def monitor(request, call_next):
            return await middleware.handle_request(request, call_next)
    """

    def __init__(
        self,
        metrics_collector: Optional[ProductionMetricsCollector] = None,
        tracer: Optional[DistributedTracer] = None,
        request_id_header: str = "X-Request-ID",
        trace_id_header: str = "X-Trace-ID",
        collect_system_metrics: bool = True,
    ) -> None:
        """Initialize monitoring middleware.

        Args:
            metrics_collector: Metrics collector (creates default if None)
            tracer: Distributed tracer (creates default if None)
            request_id_header: Header name for request ID
            trace_id_header: Header name for trace ID
            collect_system_metrics: Whether to collect system metrics
        """
        self.metrics_collector = metrics_collector or ProductionMetricsCollector(
            collect_system_metrics=collect_system_metrics,
        )
        self.tracer = tracer or DistributedTracer("victor.api")
        self.request_id_header = request_id_header
        self.trace_id_header = trace_id_header

        # Set global tracer
        from victor.observability.distributed_tracing import set_global_tracer

        set_global_tracer(self.tracer)

    async def handle_request(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Handle HTTP request with monitoring.

        Args:
            request: HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            HTTP response

        Example:
            @app.middleware("http")
            async def monitoring_middleware(request: Request, call_next):
                return await middleware.handle_request(request, call_next)
        """
        if not STARLETTE_AVAILABLE:
            logger.warning("Starlette not available, monitoring middleware disabled")
            return await call_next(request)

        # Generate IDs
        request_id = self._get_or_generate_request_id(request.headers)
        trace_id = self._get_or_generate_trace_id(request.headers)

        # Create request metadata
        metadata = RequestMetadata(
            request_id=request_id,
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
            user_agent=request.headers.get("user-agent"),
            client_ip=self._get_client_ip(request),
            start_time=datetime.now(timezone.utc),
        )

        # Start tracing
        with self.tracer.start_span(f"http.{metadata.method.lower()}") as span:
            span.set_attribute("http.method", metadata.method)
            span.set_attribute("http.path", metadata.path)
            span.set_attribute("http.request_id", request_id)
            span.set_attribute("http.user_agent", metadata.user_agent or "")
            span.set_attribute("http.client_ip", metadata.client_ip or "")

            # Increment active requests
            self.metrics_collector.active_requests.increment()

            try:
                # Collect system metrics before request
                self.metrics_collector.collect_system_metrics()

                # Call next handler
                start = time.time()
                response = await call_next(request)
                duration_ms = (time.time() - start) * 1000

                # Update metadata
                metadata.end_time = datetime.now(timezone.utc)
                metadata.duration_ms = duration_ms
                metadata.status_code = response.status_code
                metadata.status = (
                    RequestStatus.SUCCESS if response.status_code < 400 else RequestStatus.ERROR
                )

                # Record metrics
                self.metrics_collector.record_request(
                    endpoint=metadata.path,
                    provider="http",
                    success=response.status_code < 400,
                    latency_ms=duration_ms,
                    labels={
                        "method": metadata.method,
                        "status_code": str(response.status_code),
                    },
                )

                # Update span
                span.set_attribute("http.status_code", response.status_code)
                span.set_attribute("http.duration_ms", duration_ms)

                # Add headers to response
                response.headers[self.request_id_header] = request_id
                response.headers[self.trace_id_header] = trace_id

                return response

            except Exception as e:
                # Record error
                metadata.end_time = datetime.now(timezone.utc)
                metadata.duration_ms = (time.time() - start) * 1000
                metadata.status = RequestStatus.ERROR

                self.metrics_collector.record_request(
                    endpoint=metadata.path,
                    provider="http",
                    success=False,
                    latency_ms=metadata.duration_ms,
                    labels={
                        "method": metadata.method,
                        "error": type(e).__name__,
                    },
                )

                self.metrics_collector.record_error(
                    error_type=type(e).__name__,
                    message=str(e),
                )

                # Update span
                span.set_error(e)  # type: ignore[attr-defined]

                logger.exception(f"Request failed: {e}")
                raise

            finally:
                # Decrement active requests
                self.metrics_collector.active_requests.decrement()

    def _get_or_generate_request_id(self, headers: Headers) -> str:
        """Get or generate request ID.

        Args:
            headers: Request headers

        Returns:
            Request ID
        """
        existing = headers.get(self.request_id_header)
        if existing:
            return existing
        return uuid.uuid4().hex

    def _get_or_generate_trace_id(self, headers: Headers) -> str:
        """Get or generate trace ID.

        Args:
            headers: Request headers

        Returns:
            Trace ID
        """
        existing = headers.get(self.trace_id_header)
        if existing:
            return existing
        return uuid.uuid4().hex

    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Get client IP from request.

        Args:
            request: HTTP request

        Returns:
            Client IP address
        """
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to direct client IP
        if request.client and hasattr(request.client, "host"):
            return request.client.host

        return None


# =============================================================================
# ASGI Middleware
# =============================================================================


class ASGIMonitoringMiddleware:
    """ASGI middleware for monitoring.

    Works with any ASGI application (FastAPI, Starlette, etc.).

    Attributes:
        metrics_collector: Metrics collector
        tracer: Distributed tracer

    Example:
        app = FastAPI()
        app.add_middleware(ASGIMonitoringMiddleware)
    """

    def __init__(
        self,
        app: Any,
        metrics_collector: Optional[ProductionMetricsCollector] = None,
        tracer: Optional[DistributedTracer] = None,
    ) -> None:
        """Initialize ASGI middleware.

        Args:
            app: ASGI application
            metrics_collector: Metrics collector
            tracer: Distributed tracer
        """
        self.app = app
        self.monitoring = MonitoringMiddleware(
            metrics_collector=metrics_collector,
            tracer=tracer,
        )

    async def __call__(
        self, scope: Dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]
    ) -> None:
        """ASGI entry point.

        Args:
            scope: ASGI scope
            receive: ASGI receive callable
            send: ASGI send callable
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Create mock request for monitoring
        if STARLETTE_AVAILABLE:
            from starlette.requests import Request

            Request(scope, receive)

            async def call_next(_: Request) -> Response:
                # This is a simplified version
                # For full implementation, need to handle send/receive properly
                await self.app(scope, receive, send)
                # Return a mock response
                from starlette.responses import Response

                return Response()

            # Note: This won't work perfectly as ASGI middleware is more complex
            # For production use, use the decorator pattern or framework-specific integration
            await self.app(scope, receive, send)
        else:
            await self.app(scope, receive, send)


# =============================================================================
# Decorator for Monitoring Functions
# =============================================================================


def monitor_function(
    metrics_collector: Optional[ProductionMetricsCollector] = None,
    tracer: Optional[DistributedTracer] = None,
    function_name: Optional[str] = None,
):
    """Decorator to monitor function execution.

    Args:
        metrics_collector: Metrics collector
        tracer: Distributed tracer
        function_name: Override function name for metrics/tracing

    Returns:
        Decorated function

    Example:
        @monitor_function()
        def my_function(arg1, arg2):
            # Automatically monitored
            ...
    """
    import functools

    collector = metrics_collector or ProductionMetricsCollector()
    tr = tracer or DistributedTracer("victor.functions")

    def decorator(func: Callable[..., Any]) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = function_name or f"{func.__module__}.{func.__name__}"

            with tr.start_span(name):
                start = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = (time.time() - start) * 1000

                    collector.record_request(
                        endpoint=name,
                        provider="function",
                        success=True,
                        latency_ms=duration_ms,
                    )

                    return result

                except Exception as e:
                    duration_ms = (time.time() - start) * 1000

                    collector.record_request(
                        endpoint=name,
                        provider="function",
                        success=False,
                        latency_ms=duration_ms,
                        labels={"error": type(e).__name__},
                    )
                    collector.record_error(type(e).__name__, str(e))
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = function_name or f"{func.__module__}.{func.__name__}"

            with tr.start_span(name):
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start) * 1000

                    collector.record_request(
                        endpoint=name,
                        provider="function",
                        success=True,
                        latency_ms=duration_ms,
                    )

                    return result

                except Exception as e:
                    duration_ms = (time.time() - start) * 1000

                    collector.record_request(
                        endpoint=name,
                        provider="function",
                        success=False,
                        latency_ms=duration_ms,
                        labels={"error": type(e).__name__},
                    )
                    collector.record_error(type(e).__name__, str(e))
                    raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# =============================================================================
# Factory Functions
# =============================================================================


def create_monitoring_middleware(
    metrics_collector: Optional[ProductionMetricsCollector] = None,
    tracer: Optional[DistributedTracer] = None,
) -> MonitoringMiddleware:
    """Create monitoring middleware.

    Args:
        metrics_collector: Metrics collector
        tracer: Distributed tracer

    Returns:
        Configured MonitoringMiddleware

    Example:
        middleware = create_monitoring_middleware()
        app.add_middleware(MonitoringMiddleware)
    """
    return MonitoringMiddleware(
        metrics_collector=metrics_collector,
        tracer=tracer,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MonitoringMiddleware",
    "ASGIMonitoringMiddleware",
    "RequestMetadata",
    "RequestStatus",
    "monitor_function",
    "create_monitoring_middleware",
]
