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

"""Structured logging configuration for coordinators.

This module provides structured logging with:
- JSON format for log aggregation (ELK, Splunk, etc.)
- Request ID tracking for distributed tracing
- Coordinator-specific log levels
- Contextual logging with metadata
- Performance tracking
- Error logging with stack traces

Compatible with:
- Python logging framework
- Structured logging (JSON)
- Log aggregators (ELK, Splunk, CloudWatch, etc.)
- Distributed tracing (OpenTelemetry)

Example:
    from victor.observability.coordinator_logging import setup_coordinator_logging, get_coordinator_logger

    # Setup logging
    setup_coordinator_logging(
        level="INFO",
        format="json",  # or "text"
        output_file="coordinators.log",
    )

    # Get logger
    logger = get_coordinator_logger("ChatCoordinator")

    # Log with context
    logger.info("Processing request", extra={
        "request_id": "abc-123",
        "user_id": "user@example.com",
        "prompt_length": 150,
    })
"""

from __future__ import annotations

import json
import logging
import logging.config
import sys
import threading
import time
import traceback
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Context variable for request ID tracking
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


# =============================================================================
# Log Level Enum
# =============================================================================


class CoordinatorLogLevel(str, Enum):
    """Log levels for coordinator operations."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# =============================================================================
# JSON Formatter
# =============================================================================


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    Outputs logs in JSON format for aggregation in ELK, Splunk, etc.

    Format:
    {
        "timestamp": "2025-01-14T10:30:45.123Z",
        "level": "INFO",
        "logger": "ChatCoordinator",
        "message": "Processing request",
        "request_id": "abc-123",
        "coordinator": "ChatCoordinator",
        "duration_ms": 123.45,
        ... // custom fields
    }
    """

    def __init__(
        self,
        service_name: str = "victor",
        environment: str = "production",
        include_context: bool = True,
    ) -> None:
        """Initialize JSON formatter.

        Args:
            service_name: Service name for logs.
            environment: Environment (dev, staging, production).
            include_context: Whether to include context variables.
        """
        super().__init__()
        self._service_name = service_name
        self._environment = environment
        self._include_context = include_context
        self._reserved_fields = {
            "timestamp",
            "level",
            "logger",
            "message",
            "service",
            "environment",
            "request_id",
            "coordinator",
            "exception",
            "stack_trace",
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record.

        Returns:
            JSON string.
        """
        # Base log data
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self._service_name,
            "environment": self._environment,
        }

        # Add request ID from context if available (will be overridden by extra field if present)
        request_id = request_id_ctx.get()
        if request_id:
            log_data["request_id"] = request_id

        # Extract coordinator name from logger
        if "Coordinator" in record.name:
            log_data["coordinator"] = record.name

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "stack_trace": self.formatException(record.exc_info) if record.exc_info else None,
            }

        # Add custom fields from record
        # Extra fields override base fields and context values
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "asctime",
            }:
                # Always add custom fields, allowing them to override base/context fields
                log_data[key] = value

        return json.dumps(log_data, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Text formatter with color support for development.

    Format:
        [2025-01-14 10:30:45] [INFO] [ChatCoordinator] [req-abc123] Processing request
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def __init__(self, use_colors: bool = True) -> None:
        """Initialize text formatter.

        Args:
            use_colors: Whether to use ANSI colors.
        """
        super().__init__()
        self._use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as text.

        Args:
            record: Log record.

        Returns:
            Formatted string.
        """
        # Base format
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        level = record.levelname
        logger = record.name

        # Add colors
        if self._use_colors:
            level_color = self.COLORS.get(level, "")
            reset = self.COLORS["RESET"]
            level = f"{level_color}{level}{reset}"

        # Build message
        message = f"[{timestamp}] [{level}] [{logger}]"

        # Add request ID if available
        request_id = request_id_ctx.get()
        if request_id:
            message += f" [req-{request_id[:8]}]"

        # Add log message
        message += f" {record.getMessage()}"

        # Add exception info
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

        return message


# =============================================================================
# Coordinator Logger
# =============================================================================


class CoordinatorLogger:
    """Structured logger for coordinators.

    Provides:
    - Structured logging with context
    - Performance tracking
    - Error logging with metadata
    - Request ID propagation

    Example:
        logger = get_coordinator_logger("ChatCoordinator")

        # Info log
        logger.info("Starting request", extra={
            "request_id": "abc-123",
            "user_id": "user@example.com",
        })

        # Performance tracking
        with logger.track_duration("process_request"):
            # Do work
            pass

        # Error logging
        try:
            risky_operation()
        except Exception as e:
            logger.error("Operation failed", exc_info=e, extra={
                "operation": "risky_operation",
                "context": {...},
            })
    """

    def __init__(
        self,
        name: str,
        logger: logging.Logger,
    ) -> None:
        """Initialize coordinator logger.

        Args:
            name: Coordinator name.
            logger: Python logger instance.
        """
        self._name = name
        self._logger = logger

    def debug(
        self,
        message: str,
        exc_info: Optional[Exception] = None,
        **kwargs: Any,
    ) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, exc_info, kwargs)

    def info(
        self,
        message: str,
        exc_info: Optional[Exception] = None,
        **kwargs: Any,
    ) -> None:
        """Log info message."""
        self._log(logging.INFO, message, exc_info, kwargs)

    def warning(
        self,
        message: str,
        exc_info: Optional[Exception] = None,
        **kwargs: Any,
    ) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, exc_info, kwargs)

    def error(
        self,
        message: str,
        exc_info: Optional[Exception] = None,
        **kwargs: Any,
    ) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, exc_info, kwargs)

    def critical(
        self,
        message: str,
        exc_info: Optional[Exception] = None,
        **kwargs: Any,
    ) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, exc_info, kwargs)

    def _log(
        self,
        level: int,
        message: str,
        exc_info: Optional[Exception],
        extra: Dict[str, Any],
    ) -> None:
        """Internal log method."""
        # Add coordinator name
        extra["coordinator"] = self._name

        # Log
        self._logger.log(level, message, exc_info=exc_info, extra=extra)

    def track_duration(self, operation: str) -> "_DurationTracker":
        """Track operation duration.

        Args:
            operation: Operation name.

        Returns:
            Context manager for tracking.

        Example:
            with logger.track_duration("process_request"):
                process()
        """
        return _DurationTracker(self, operation)

    def set_request_id(self, request_id: str) -> None:
        """Set request ID for logging context.

        Args:
            request_id: Request ID.
        """
        request_id_ctx.set(request_id)

    def clear_request_id(self) -> None:
        """Clear request ID from context."""
        request_id_ctx.set(None)


class _DurationTracker:
    """Context manager for tracking operation duration."""

    def __init__(self, logger: CoordinatorLogger, operation: str) -> None:
        """Initialize tracker."""
        self._logger = logger
        self._operation = operation
        self._start_time: Optional[float] = None

    def __enter__(self) -> "_DurationTracker":
        """Start tracking."""
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop tracking and log."""
        if self._start_time is None:
            return

        duration_ms = (time.perf_counter() - self._start_time) * 1000

        self._logger.info(
            f"Operation completed: {self._operation}",
            extra={
                "operation": self._operation,
                "duration_ms": duration_ms,
                "success": exc_type is None,
            },
        )


# =============================================================================
# Logging Configuration
# =============================================================================


def setup_coordinator_logging(
    level: Union[str, int] = logging.INFO,
    format_type: str = "json",  # "json" or "text"
    output_file: Optional[str] = None,
    service_name: str = "victor",
    environment: str = "production",
    log_coordinator_level: str = "INFO",
    log_execution_level: str = "DEBUG",
    log_error_level: str = "ERROR",
) -> None:
    """Setup structured logging for coordinators.

    Args:
        level: Overall log level.
        format_type: Log format type ("json" or "text").
        output_file: Optional log file path (stdout if not provided).
        service_name: Service name for logs.
        environment: Environment name.
        log_coordinator_level: Log level for coordinator lifecycle events.
        log_execution_level: Log level for execution tracking.
        log_error_level: Log level for error logging.

    Example:
        # Production JSON logging to file
        setup_coordinator_logging(
            level="INFO",
            format_type="json",
            output_file="coordinators.log",
        )

        # Development text logging to console
        setup_coordinator_logging(
            level="DEBUG",
            format_type="text",
        )
    """
    # Convert level string to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Create handler
    if output_file:
        handler = logging.FileHandler(output_file)
    else:
        handler = logging.StreamHandler(sys.stderr)

    # Set formatter
    if format_type == "json":
        formatter = StructuredFormatter(
            service_name=service_name,
            environment=environment,
        )
    else:
        formatter = TextFormatter(use_colors=True)

    handler.setFormatter(formatter)
    handler.setLevel(level)

    # Configure root logger
    root_logger = logging.getLogger("victor")
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Coordinator-specific loggers
    coordinator_logger = logging.getLogger("victor.coordinators")
    coordinator_logger.setLevel(getattr(logging, log_coordinator_level.upper(), logging.INFO))

    execution_logger = logging.getLogger("victor.coordinators.execution")
    execution_logger.setLevel(getattr(logging, log_execution_level.upper(), logging.DEBUG))

    error_logger = logging.getLogger("victor.coordinators.errors")
    error_logger.setLevel(getattr(logging, log_error_level.upper(), logging.ERROR))

    # Prevent propagation to avoid duplicate logs
    coordinator_logger.propagate = False
    coordinator_logger.addHandler(handler)

    execution_logger.propagate = False
    execution_logger.addHandler(handler)

    error_logger.propagate = False
    error_logger.addHandler(handler)

    logging.info(f"Coordinator logging configured: level={level}, format={format_type}")


def get_coordinator_logger(name: str) -> CoordinatorLogger:
    """Get structured logger for a coordinator.

    Args:
        name: Coordinator name.

    Returns:
        CoordinatorLogger instance.

    Example:
        logger = get_coordinator_logger("ChatCoordinator")
        logger.info("Processing request", extra={"request_id": "abc-123"})
    """
    python_logger = logging.getLogger(f"victor.coordinators.{name}")
    return CoordinatorLogger(name, python_logger)


def set_request_id(request_id: str) -> None:
    """Set request ID for current context.

    Args:
        request_id: Request ID.

    Example:
        from victor.observability.coordinator_logging import set_request_id

        async def handle_request(request_id: str):
            set_request_id(request_id)
            # All logs in this context will include request_id
    """
    request_id_ctx.set(request_id)


def clear_request_id() -> None:
    """Clear request ID from current context."""
    request_id_ctx.set(None)


def get_request_id() -> Optional[str]:
    """Get current request ID from context.

    Returns:
        Request ID or None.
    """
    return request_id_ctx.get()


# =============================================================================
# Decorators for Auto-Logging
# =============================================================================


def log_execution(
    logger: Optional[CoordinatorLogger] = None,
    operation: Optional[str] = None,
    log_level: str = "INFO",
    log_errors: bool = True,
):
    """Decorator to automatically log method execution.

    Args:
        logger: Logger instance (uses coordinator name if not provided).
        operation: Operation name (uses method name if not provided).
        log_level: Log level for success messages.
        log_errors: Whether to log errors separately.

    Example:
        logger = get_coordinator_logger("ChatCoordinator")

        @log_execution(logger, "process_request")
        async def process(self, request):
            # Method is automatically logged
            return result
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get logger
            nonlocal logger
            if logger is None:
                # Try to extract from class name
                if args and hasattr(args[0], "__class__"):
                    class_name = args[0].__class__.__name__
                    logger = get_coordinator_logger(class_name)
                else:
                    logger = get_coordinator_logger(func.__qualname__)

            # Get operation name
            op_name = operation or func.__name__

            # Track execution
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000

                logger.info(
                    f"Operation completed: {op_name}",
                    extra={
                        "operation": op_name,
                        "duration_ms": duration_ms,
                        "success": True,
                    },
                )
                return result

            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000

                if log_errors:
                    logger.error(
                        f"Operation failed: {op_name}",
                        exc_info=e,
                        extra={
                            "operation": op_name,
                            "duration_ms": duration_ms,
                            "success": False,
                            "error_type": type(e).__name__,
                        },
                    )
                raise

        return wrapper

    return decorator
