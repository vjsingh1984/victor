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

"""Structured JSON logging for Victor AI.

This module provides structured logging with:
- JSON format for machine parsing
- Correlation IDs for request tracing
- Request/response logging at appropriate levels
- Performance logging (slow queries, etc.)
- Error context and stack traces
- Sampling for high-volume logs

Environment Variables:
    VICTOR_LOG_FORMAT: Log format (json, text) (default: text)
    VICTOR_LOG_LEVEL: Log level (DEBUG, INFO, WARNING, ERROR) (default: INFO)
    VICTOR_LOG_SAMPLING: Sampling rate for high-volume logs (default: 1.0)
    VICTOR_LOG_REQUEST_ENABLED: Enable request/response logging (default: true)
    VICTOR_LOG_PERFORMANCE_ENABLED: Enable performance logging (default: true)
    VICTOR_LOG_SLOW_THRESHOLD_MS: Slow operation threshold (default: 1000)

Example:
    # Enable structured JSON logging
    export VICTOR_LOG_FORMAT=json
    export VICTOR_LOG_LEVEL=info

    from victor.observability.structured_logging import setup_structured_logging

    setup_structured_logging()

    # Logs are now in JSON format with correlation IDs
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Processing request", extra={"request_id": "123"})
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional, Callable

# Context variable for correlation ID (async-safe)
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class LogFormat(str, Enum):
    """Log format types."""

    JSON = "json"
    TEXT = "text"


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for Python logging.

    Outputs logs in JSON format with structured fields:
    - timestamp: ISO 8601 timestamp
    - level: Log level (DEBUG, INFO, WARNING, ERROR)
    - logger: Logger name
    - message: Log message
    - correlation_id: Request correlation ID (if available)
    - context: Additional context from log record
    - exception: Exception details (if applicable)
    """

    def __init__(
        self,
        service_name: str = "victor",
        environment: str = "development",
        include_extra: bool = True,
    ):
        """Initialize structured formatter.

        Args:
            service_name: Service name for logs.
            environment: Deployment environment.
            include_extra: Include extra fields from log records.
        """
        super().__init__()
        self.service_name = service_name
        self.environment = environment
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format.

        Returns:
            JSON-formatted log string.
        """
        # Base log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "environment": self.environment,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "process": record.process,
        }

        # Add correlation ID if available
        corr_id = correlation_id_var.get()
        if corr_id:
            log_data["correlation_id"] = corr_id

        # Add extra fields from log record
        if self.include_extra and hasattr(record, "extra"):
            log_data.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Add stack trace if available
        if record.stack_info:
            log_data["stack_trace"] = self.formatStack(record.stack_info)

        return json.dumps(log_data, default=str)


class PerformanceLogger:
    """Logger for performance monitoring.

    Tracks operation durations and logs slow operations.
    """

    def __init__(
        self,
        logger: logging.Logger,
        slow_threshold_ms: float = 1000.0,
        enabled: bool = True,
    ):
        """Initialize performance logger.

        Args:
            logger: Logger instance.
            slow_threshold_ms: Threshold for logging slow operations.
            enabled: Whether performance logging is enabled.
        """
        self.logger = logger
        self.slow_threshold_ms = slow_threshold_ms
        self.enabled = enabled

    def track_operation(
        self,
        operation_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracking operation performance.

        Args:
            operation_name: Name of the operation.
            metadata: Additional metadata to log.

        Returns:
            Context manager for timing.

        Example:
            perf_logger = PerformanceLogger(logger)
            with perf_logger.track_operation("database_query", {"table": "users"}):
                result = db.query("SELECT * FROM users")
        """

        class OperationTimer:
            def __init__(
                self, perf_logger: PerformanceLogger, name: str, meta: Optional[Dict[str, Any]]
            ):
                self.perf_logger = perf_logger
                self.name = name
                self.meta = meta or {}
                self.start_time: Optional[float] = None

            def __enter__(self):
                if self.perf_logger.enabled:
                    self.start_time = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if not self.perf_logger.enabled or self.start_time is None:
                    return

                duration_ms = (time.perf_counter() - self.start_time) * 1000

                log_data = {
                    "operation": self.name,
                    "duration_ms": round(duration_ms, 2),
                    "slow": duration_ms >= self.perf_logger.slow_threshold_ms,
                    **self.meta,
                }

                if duration_ms >= self.perf_logger.slow_threshold_ms:
                    self.perf_logger.logger.warning(
                        f"Slow operation: {self.name}",
                        extra={"performance": log_data},
                    )
                else:
                    self.perf_logger.logger.debug(
                        f"Operation: {self.name}",
                        extra={"performance": log_data},
                    )

        return OperationTimer(self, operation_name, metadata)


class RequestLogger:
    """Logger for HTTP/request-response operations.

    Logs incoming requests and outgoing responses with correlation IDs.
    """

    def __init__(
        self,
        logger: logging.Logger,
        enabled: bool = True,
        log_body: bool = False,
        max_body_length: int = 1000,
    ):
        """Initialize request logger.

        Args:
            logger: Logger instance.
            enabled: Whether request logging is enabled.
            log_body: Whether to log request/response bodies.
            max_body_length: Maximum body length to log.
        """
        self.logger = logger
        self.enabled = enabled
        self.log_body = log_body
        self.max_body_length = max_body_length

    def log_request(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ):
        """Log incoming request.

        Args:
            method: HTTP method.
            path: Request path.
            headers: Request headers.
            body: Request body.
            correlation_id: Request correlation ID.
        """
        if not self.enabled:
            return

        if correlation_id:
            correlation_id_var.set(correlation_id)

        log_data = {
            "type": "request",
            "method": method,
            "path": path,
            "headers": self._sanitize_headers(headers or {}),
        }

        if self.log_body and body:
            log_data["body"] = body[: self.max_body_length]

        self.logger.info(f"{method} {path}", extra={"http": log_data})

    def log_response(
        self,
        status_code: int,
        duration_ms: float,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """Log outgoing response.

        Args:
            status_code: HTTP status code.
            duration_ms: Request duration in milliseconds.
            headers: Response headers.
            body: Response body.
            error: Error message if applicable.
        """
        if not self.enabled:
            return

        log_data = {
            "type": "response",
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
            "headers": headers or {},
        }

        if self.log_body and body:
            log_data["body"] = body[: self.max_body_length]

        if error:
            log_data["error"] = error

        level = logging.WARNING if status_code >= 400 else logging.INFO
        self.logger.log(
            level,
            f"Response {status_code}",
            extra={"http": log_data},
        )

    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize headers by removing sensitive data.

        Args:
            headers: Original headers.

        Returns:
            Sanitized headers.
        """
        sensitive_headers = {"authorization", "cookie", "set-cookie", "x-api-key"}
        sanitized = {}
        for key, value in headers.items():
            if key.lower() in sensitive_headers:
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = value
        return sanitized


class SamplingLogger:
    """Logger with sampling support for high-volume logs.

    Samples logs based on configured rate to reduce volume.
    """

    def __init__(
        self,
        logger: logging.Logger,
        sample_rate: float = 1.0,
        sample_key: Optional[str] = None,
    ):
        """Initialize sampling logger.

        Args:
            logger: Logger instance.
            sample_rate: Sampling rate (0.0 to 1.0).
            sample_key: Optional key for consistent sampling.
        """
        self.logger = logger
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.sample_key = sample_key
        self._counter = 0
        self._lock = threading.Lock()

    def _should_log(self) -> bool:
        """Determine if log should be sampled.

        Returns:
            True if log should be emitted.
        """
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False

        with self._lock:
            self._counter += 1
            return (self._counter % int(1.0 / self.sample_rate)) == 0

    def info(self, msg: str, *args, **kwargs):
        """Log info message with sampling."""
        if self._should_log():
            self.logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        """Log debug message with sampling."""
        if self._should_log():
            self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log warning message (never sampled)."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log error message (never sampled)."""
        self.logger.error(msg, *args, **kwargs)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID from context.

    Returns:
        Correlation ID or None.
    """
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID in context.

    Args:
        correlation_id: Correlation ID to set.
    """
    correlation_id_var.set(correlation_id)


def setup_structured_logging(
    log_format: str = "text",
    log_level: str = "INFO",
    service_name: str = "victor",
    environment: str = "development",
    log_file: Optional[str] = None,
    sampling_rate: float = 1.0,
) -> tuple[logging.Logger, Optional[PerformanceLogger], Optional[RequestLogger]]:
    """Setup structured logging for Victor AI.

    Configures Python logging with structured JSON or text format,
    correlation ID support, and performance/request logging.

    Args:
        log_format: Log format (json or text).
        log_level: Log level (DEBUG, INFO, WARNING, ERROR).
        service_name: Service name for logs.
        environment: Deployment environment.
        log_file: Optional log file path.
        sampling_rate: Sampling rate for high-volume logs.

    Returns:
        Tuple of (root_logger, performance_logger, request_logger).

    Example:
        logger, perf_logger, req_logger = setup_structured_logging(
            log_format="json",
            log_level="info",
        )

        # Use correlation IDs
        set_correlation_id("req-123")
        logger.info("Processing request")

        # Track performance
        with perf_logger.track_operation("database_query"):
            result = db.query(...)

        # Log HTTP requests
        req_logger.log_request("GET", "/api/users", headers={"user-agent": "..."})
        req_logger.log_response(200, 45.2)
    """
    # Get configuration from environment if not provided
    log_format = os.getenv("VICTOR_LOG_FORMAT", log_format).lower()
    log_level_str = os.getenv("VICTOR_LOG_LEVEL", log_level).upper()
    sampling_rate = float(os.getenv("VICTOR_LOG_SAMPLING", str(sampling_rate)))

    # Parse log level
    numeric_level = getattr(logging, log_level_str, logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter: logging.Formatter
    if log_format == "json":
        formatter = StructuredFormatter(
            service_name=service_name,
            environment=environment,
        )
    else:
        # Use standard text formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)

    # Create performance logger
    perf_enabled = os.getenv("VICTOR_LOG_PERFORMANCE_ENABLED", "true").lower() in ("true", "1")
    slow_threshold = float(os.getenv("VICTOR_LOG_SLOW_THRESHOLD_MS", "1000"))
    perf_logger = PerformanceLogger(
        logger=root_logger,
        slow_threshold_ms=slow_threshold,
        enabled=perf_enabled,
    )

    # Create request logger
    req_enabled = os.getenv("VICTOR_LOG_REQUEST_ENABLED", "true").lower() in ("true", "1")
    req_logger = RequestLogger(
        logger=root_logger,
        enabled=req_enabled,
        log_body=os.getenv("VICTOR_LOG_REQUEST_BODY", "false").lower() in ("true", "1"),
    )

    # Log configuration
    root_logger.info(
        f"Structured logging configured: format={log_format}, level={log_level_str}, "
        f"sampling={sampling_rate}, perf={perf_enabled}, req={req_enabled}"
    )

    return root_logger, perf_logger, req_logger
