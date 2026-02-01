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

"""Telemetry configuration for Victor AI observability.

This module provides configuration for OpenTelemetry tracing, metrics,
and structured logging. All features can be enabled/disabled via
environment variables for production deployment.

Environment Variables:
    VICTOR_TELEMETRY_ENABLED: Enable/disable telemetry (default: false)
    VICTOR_TELEMETRY_EXPORTER: Exporter type (otlp, console, jaeger) (default: otlp)
    VICTOR_TELEMETRY_ENDPOINT: OTLP endpoint (default: http://localhost:4317)
    VICTOR_TELEMETRY_SAMPLING: Sampling rate 0.0-1.0 (default: 1.0)
    VICTOR_TELEMETRY_TRACING_ENABLED: Enable tracing (default: true)
    VICTOR_TELEMETRY_METRICS_ENABLED: Enable metrics (default: true)
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (alias for VICTOR_TELEMETRY_ENDPOINT)
    OTEL_SERVICE_NAME: Service name for tracing (default: victor)
    OTEL_SERVICE_VERSION: Service version (default: from __version__)

Example:
    # Enable OTLP tracing with sampling
    export VICTOR_TELEMETRY_ENABLED=true
    export VICTOR_TELEMETRY_EXPORTER=otlp
    export VICTOR_TELEMETRY_ENDPOINT=http://jaeger:4317
    export VICTOR_TELEMETRY_SAMPLING=0.5

    # Console exporter for development
    export VICTOR_TELEMETRY_ENABLED=true
    export VICTOR_TELEMETRY_EXPORTER=console

    from victor.config.telemetry_config import TelemetryConfig, setup_telemetry

    config = TelemetryConfig.from_env()
    if config.enabled:
        setup_telemetry(config)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TelemetryExporter(str, Enum):
    """Telemetry exporter types."""

    OTLP = "otlp"
    CONSOLE = "console"
    JAEGER = "jaeger"
    NONE = "none"


@dataclass
class TelemetryConfig:
    """Configuration for OpenTelemetry tracing and metrics.

    Attributes:
        enabled: Whether telemetry is enabled
        exporter: Exporter type (otlp, console, jaeger)
        endpoint: OTLP endpoint URL
        sampling: Sampling rate (0.0 to 1.0)
        tracing_enabled: Enable distributed tracing
        metrics_enabled: Enable metrics collection
        service_name: Service name for tracing
        service_version: Service version
        environment: Deployment environment
        batch_timeout: Batch export timeout in seconds
        batch_max_size: Maximum batch size for exports
        exporters: List of exporter types to use
        resource_attributes: Additional resource attributes
    """

    enabled: bool = False
    exporter: TelemetryExporter = TelemetryExporter.OTLP
    endpoint: Optional[str] = None
    sampling: float = 1.0
    tracing_enabled: bool = True
    metrics_enabled: bool = True
    service_name: str = "victor"
    service_version: str = "0.5.1"
    environment: str = "development"
    batch_timeout: int = 30
    batch_max_size: int = 512
    exporters: list[TelemetryExporter] = field(default_factory=list)
    resource_attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "TelemetryConfig":
        """Create configuration from environment variables.

        Returns:
            TelemetryConfig instance.
        """
        # Check if telemetry is enabled
        enabled = os.getenv("VICTOR_TELEMETRY_ENABLED", "false").lower() in ("true", "1", "yes")

        # Get exporter type
        exporter_str = os.getenv("VICTOR_TELEMETRY_EXPORTER", "otlp").lower()
        try:
            exporter = TelemetryExporter(exporter_str)
        except ValueError:
            logger.warning(f"Invalid exporter type: {exporter_str}, using OTLP")
            exporter = TelemetryExporter.OTLP

        # Get endpoint (prefer VICTOR_*, fallback to OTEL_*)
        endpoint = os.getenv("VICTOR_TELEMETRY_ENDPOINT") or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )

        # Get sampling rate
        sampling_str = os.getenv("VICTOR_TELEMETRY_SAMPLING", "1.0")
        try:
            sampling = float(sampling_str)
            sampling = max(0.0, min(1.0, sampling))  # Clamp to [0, 1]
        except ValueError:
            logger.warning(f"Invalid sampling rate: {sampling_str}, using 1.0")
            sampling = 1.0

        # Get tracing/metrics flags
        tracing_enabled = os.getenv("VICTOR_TELEMETRY_TRACING_ENABLED", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        metrics_enabled = os.getenv("VICTOR_TELEMETRY_METRICS_ENABLED", "true").lower() in (
            "true",
            "1",
            "yes",
        )

        # Service info
        service_name = os.getenv("OTEL_SERVICE_NAME", "victor")
        service_version = os.getenv("OTEL_SERVICE_VERSION", "0.5.1")
        environment = os.getenv("VICTOR_ENV", os.getenv("OTEL_ENVIRONMENT", "development"))

        # Batch configuration
        batch_timeout = int(os.getenv("VICTOR_TELEMETRY_BATCH_TIMEOUT", "30"))
        batch_max_size = int(os.getenv("VICTOR_TELEMETRY_BATCH_MAX_SIZE", "512"))

        # Parse exporters list
        exporters_str = os.getenv("VICTOR_TELEMETRY_EXPORTERS", "")
        exporters = []
        if exporters_str:
            for exp in exporters_str.split(","):
                exp = exp.strip().lower()
                try:
                    exporters.append(TelemetryExporter(exp))
                except ValueError:
                    logger.warning(f"Invalid exporter in list: {exp}")

        # If no exporters specified but telemetry enabled, use configured exporter
        if enabled and not exporters:
            exporters = [exporter]

        # Additional resource attributes
        resource_attrs = {}
        for key, value in os.environ.items():
            if key.startswith("OTEL_RESOURCE_ATTRIBUTE_"):
                attr_name = key[len("OTEL_RESOURCE_ATTRIBUTE_") :].lower()
                resource_attrs[attr_name] = value

        return cls(
            enabled=enabled,
            exporter=exporter,
            endpoint=endpoint,
            sampling=sampling,
            tracing_enabled=tracing_enabled,
            metrics_enabled=metrics_enabled,
            service_name=service_name,
            service_version=service_version,
            environment=environment,
            batch_timeout=batch_timeout,
            batch_max_size=batch_max_size,
            exporters=exporters,
            resource_attributes=resource_attrs,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "enabled": self.enabled,
            "exporter": self.exporter.value,
            "endpoint": self.endpoint,
            "sampling": self.sampling,
            "tracing_enabled": self.tracing_enabled,
            "metrics_enabled": self.metrics_enabled,
            "service_name": self.service_name,
            "service_version": self.service_version,
            "environment": self.environment,
            "batch_timeout": self.batch_timeout,
            "batch_max_size": self.batch_max_size,
            "exporters": [e.value for e in self.exporters],
            "resource_attributes": self.resource_attributes,
        }

    def validate(self) -> list[str]:
        """Validate configuration.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        if self.enabled:
            if self.exporter == TelemetryExporter.OTLP and not self.endpoint:
                errors.append("OTLP exporter requires endpoint")

            if not (0.0 <= self.sampling <= 1.0):
                errors.append("Sampling rate must be between 0.0 and 1.0")

            if self.batch_timeout <= 0:
                errors.append("Batch timeout must be positive")

            if self.batch_max_size <= 0:
                errors.append("Batch max size must be positive")

        return errors


def get_telemetry_config() -> TelemetryConfig:
    """Get telemetry configuration from environment.

    This is a convenience function that loads configuration from
    environment variables and caches it for subsequent calls.

    Returns:
        TelemetryConfig instance.
    """
    if not hasattr(get_telemetry_config, "_config"):
        config = TelemetryConfig.from_env()
        errors = config.validate()
        if errors:
            logger.warning(f"Telemetry configuration errors: {errors}")
        object.__setattr__(get_telemetry_config, "_config", config)
    return object.__getattribute__(get_telemetry_config, "_config")


def setup_telemetry(config: Optional[TelemetryConfig] = None) -> Optional[tuple[Any, Any]]:
    """Setup OpenTelemetry based on configuration.

    This function initializes OpenTelemetry tracing and metrics
    based on the provided configuration. It should be called once
    at application startup.

    Args:
        config: Telemetry configuration (uses env if not provided).

    Returns:
        Tuple of (tracer, meter) or (None, None) if disabled.

    Example:
        from victor.config.telemetry_config import setup_telemetry, get_telemetry_config

        config = get_telemetry_config()
        if config.enabled:
            tracer, meter = setup_telemetry(config)
            # Use tracer and meter for instrumentation
    """
    if config is None:
        config = get_telemetry_config()

    if not config.enabled:
        logger.info("Telemetry is disabled")
        return None

    try:
        from victor.observability.telemetry import setup_opentelemetry

        logger.info(
            f"Setting up telemetry: exporter={config.exporter.value}, "
            f"sampling={config.sampling}, tracing={config.tracing_enabled}, "
            f"metrics={config.metrics_enabled}"
        )

        tracer, meter = setup_opentelemetry(
            service_name=config.service_name,
            service_version=config.service_version,
            otlp_endpoint=config.endpoint if config.exporter == TelemetryExporter.OTLP else None,
            enable_tracing=config.tracing_enabled,
            enable_metrics=config.metrics_enabled,
        )

        logger.info("Telemetry setup complete")
        return tracer, meter

    except ImportError as e:
        logger.error(f"OpenTelemetry dependencies not installed: {e}")
        logger.info("Install with: pip install victor-ai[observability]")
        return None
    except Exception as e:
        logger.error(f"Failed to setup telemetry: {e}")
        return None
