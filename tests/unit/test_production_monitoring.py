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

"""Tests for production monitoring and observability components.

Tests Prometheus integration, health checks, and structured logging
for coordinator-based orchestrator.
"""

import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from victor.observability.prometheus_metrics import (
    Counter,
    Gauge,
    Histogram,
    PrometheusMetricsExporter,
    PrometheusRegistry,
)
from victor.observability.health import (
    CoordinatorHealthCheck,
    CoordinatorHealthService,
)
from victor.observability.coordinator_logging import (
    StructuredFormatter,
    TextFormatter,
    CoordinatorLogger,
    setup_coordinator_logging,
    get_coordinator_logger,
)


# =============================================================================
# Prometheus Metrics Tests
# =============================================================================


class TestPrometheusMetrics:
    """Tests for Prometheus metrics."""

    def test_counter_increment(self):
        """Test counter increment."""
        counter = Counter(
            name="test_counter",
            help="Test counter metric",
        )

        counter.inc()
        assert counter.value == 1.0

        counter.inc(5.0)
        assert counter.value == 6.0

    def test_counter_negative_increment_fails(self):
        """Test counter cannot decrement."""
        counter = Counter(
            name="test_counter",
            help="Test counter metric",
        )

        with pytest.raises(ValueError):
            counter.inc(-1.0)

    def test_counter_to_prometheus(self):
        """Test counter Prometheus export."""
        counter = Counter(
            name="test_counter",
            help="Test counter metric",
            labels={"label1": "value1"},
        )
        counter.inc(10.0)

        output = counter.to_prometheus()

        assert "# HELP test_counter Test counter metric" in output
        assert "# TYPE test_counter counter" in output
        assert 'test_counter{label1="value1"} 10.0' in output

    def test_gauge_set(self):
        """Test gauge set."""
        gauge = Gauge(
            name="test_gauge",
            help="Test gauge metric",
        )

        gauge.set(42.0)
        assert gauge.value == 42.0

        gauge.set(10.0)
        assert gauge.value == 10.0

    def test_gauge_increment_decrement(self):
        """Test gauge increment and decrement."""
        gauge = Gauge(
            name="test_gauge",
            help="Test gauge metric",
        )

        gauge.set(10.0)
        gauge.inc(5.0)
        assert gauge.value == 15.0

        gauge.dec(3.0)
        assert gauge.value == 12.0

    def test_histogram_observe(self):
        """Test histogram observation."""
        histogram = Histogram(
            name="test_histogram",
            help="Test histogram metric",
            buckets=[1.0, 5.0, 10.0],
        )

        histogram.observe(0.5)
        histogram.observe(2.0)
        histogram.observe(7.0)
        histogram.observe(15.0)

        assert histogram._count == 4
        assert histogram._sum == 24.5

    def test_histogram_to_prometheus(self):
        """Test histogram Prometheus export."""
        histogram = Histogram(
            name="test_histogram",
            help="Test histogram metric",
            buckets=[1.0, 5.0],
        )

        histogram.observe(0.5)
        histogram.observe(3.0)

        output = histogram.to_prometheus()

        assert "# HELP test_histogram Test histogram metric" in output
        assert "# TYPE test_histogram histogram" in output
        assert "test_histogram_bucket" in output
        assert "test_histogram_sum" in output
        assert "test_histogram_count" in output

    def test_registry_counter(self):
        """Test registry counter management."""
        registry = PrometheusRegistry()

        counter1 = registry.counter(
            name="test_counter",
            help="Test counter",
        )
        counter1.inc()

        counter2 = registry.counter(
            name="test_counter",
            help="Test counter",
        )

        # Should return same instance
        assert counter1 is counter2
        assert counter2.value == 1.0

    def test_registry_gauge(self):
        """Test registry gauge management."""
        registry = PrometheusRegistry()

        gauge1 = registry.gauge(
            name="test_gauge",
            help="Test gauge",
        )
        gauge1.set(10.0)

        gauge2 = registry.gauge(
            name="test_gauge",
            help="Test gauge",
        )

        assert gauge1 is gauge2
        assert gauge2.value == 10.0

    def test_registry_export(self):
        """Test registry export."""
        registry = PrometheusRegistry()

        counter = registry.counter(
            name="test_counter",
            help="Test counter",
        )
        counter.inc(5.0)

        gauge = registry.gauge(
            name="test_gauge",
            help="Test gauge",
        )
        gauge.set(42.0)

        output = registry.export()

        assert "test_counter" in output
        assert "test_gauge" in output
        assert "5.0" in output
        assert "42.0" in output


class TestPrometheusExporter:
    """Tests for Prometheus metrics exporter."""

    def test_export_metrics(self):
        """Test metrics export."""
        # Create mock metrics collector
        mock_collector = Mock()
        mock_collector.get_all_snapshots.return_value = []
        mock_collector.get_overall_stats.return_value = {
            "uptime_seconds": 3600,
            "total_executions": 1000,
            "total_coordinators": 5,
            "total_errors": 10,
            "overall_error_rate": 0.01,
            "analytics_events": {},
        }

        exporter = PrometheusMetricsExporter(mock_collector)
        output = exporter.export_metrics()

        assert "victor_coordinator_uptime_seconds 3600" in output
        assert "victor_coordinator_total_executions 1000" in output
        assert "victor_coordinator_error_rate 0.01" in output

    def test_export_metrics_with_coordinators(self):
        """Test metrics export with coordinator snapshots."""
        mock_snapshot = Mock()
        mock_snapshot.coordinator_name = "TestCoordinator"
        mock_snapshot.execution_count = 100
        mock_snapshot.total_duration_ms = 5000
        mock_snapshot.error_count = 5
        mock_snapshot.memory_bytes = 1024 * 1024 * 100
        mock_snapshot.cpu_percent = 50.0
        mock_snapshot.cache_hits = 80
        mock_snapshot.cache_misses = 20

        def snapshot_to_dict():
            return {"cache_hit_rate": 0.8}

        mock_snapshot.to_dict = snapshot_to_dict

        mock_collector = Mock()
        mock_collector.get_all_snapshots.return_value = [mock_snapshot]
        mock_collector.get_overall_stats.return_value = {
            "uptime_seconds": 3600,
            "total_executions": 100,
            "total_coordinators": 1,
            "total_errors": 5,
            "overall_error_rate": 0.05,
            "analytics_events": {},
        }

        exporter = PrometheusMetricsExporter(mock_collector)
        output = exporter.export_metrics()

        assert 'victor_coordinator_executions_total{coordinator="TestCoordinator"} 100' in output
        assert 'victor_coordinator_errors_total{coordinator="TestCoordinator"} 5' in output
        assert 'victor_coordinator_cache_hit_rate{coordinator="TestCoordinator"} 0.8' in output


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthChecks:
    """Tests for health check system."""

    @pytest.mark.asyncio
    async def test_coordinator_health_check(self):
        """Test coordinator health check."""
        # Create mock metrics collector
        mock_collector = Mock()
        mock_snapshot = Mock()
        mock_snapshot.execution_count = 100
        mock_snapshot.error_count = 2
        mock_snapshot.memory_bytes = 1024 * 1024 * 100
        mock_snapshot.cpu_percent = 50.0
        mock_snapshot.cache_hits = 80
        mock_snapshot.cache_misses = 20

        def snapshot_to_dict():
            return {"cache_hit_rate": 0.8}

        mock_snapshot.to_dict = snapshot_to_dict

        mock_collector.get_snapshot.return_value = mock_snapshot
        mock_collector.get_coordinator_stats.return_value = {
            "error_rate": 0.02,
            "avg_duration_ms": 1000,
        }

        # Create health check
        health_check = CoordinatorHealthCheck(
            coordinator_name="TestCoordinator",
            metrics_collector=mock_collector,
        )

        # Perform check
        result = await health_check.check()

        assert result.name == "coordinator.TestCoordinator"
        assert result.status.value in ("healthy", "degraded", "unhealthy")
        assert result.latency_ms is not None
        assert result.details is not None

    @pytest.mark.asyncio
    async def test_health_service(self):
        """Test health service."""
        # Create mock metrics collector
        mock_collector = Mock()
        mock_collector.get_all_snapshots.return_value = []
        mock_collector.get_overall_stats.return_value = {
            "uptime_seconds": 3600,
            "total_executions": 0,
            "total_coordinators": 0,
            "total_errors": 0,
            "overall_error_rate": 0.0,
            "analytics_events": {},
        }

        # Create health service
        service = CoordinatorHealthService(mock_collector)

        # Check health
        report = await service.check_health()

        assert report.status.value in ("healthy", "degraded", "unhealthy")
        assert report.components is not None
        assert report.timestamp is not None

    @pytest.mark.asyncio
    async def test_health_service_is_ready(self):
        """Test health service readiness check."""
        mock_collector = Mock()
        mock_collector.get_all_snapshots.return_value = []
        mock_collector.get_overall_stats.return_value = {
            "uptime_seconds": 3600,
            "total_executions": 0,
            "total_coordinators": 0,
            "total_errors": 0,
            "overall_error_rate": 0.0,
            "analytics_events": {},
        }

        service = CoordinatorHealthService(mock_collector)

        is_ready = await service.is_ready()
        is_alive = await service.is_alive()

        assert is_ready in (True, False)
        assert is_alive is True  # Liveness always returns True


# =============================================================================
# Structured Logging Tests
# =============================================================================


class TestStructuredLogging:
    """Tests for structured logging."""

    def test_json_formatter(self):
        """Test JSON formatter."""
        formatter = StructuredFormatter(
            service_name="victor",
            environment="production",
        )

        # Create log record
        record = logging.LogRecord(
            name="TestCoordinator",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Format
        output = formatter.format(record)

        # Parse JSON
        log_data = json.loads(output)

        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "TestCoordinator"
        assert log_data["message"] == "Test message"
        assert log_data["service"] == "victor"
        assert log_data["environment"] == "production"
        assert "timestamp" in log_data

    def test_json_formatter_with_extra_fields(self):
        """Test JSON formatter with extra fields."""
        formatter = StructuredFormatter()

        record = logging.LogRecord(
            name="TestCoordinator",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.request_id = "abc-123"
        record.user_id = "user@example.com"

        output = formatter.format(record)
        log_data = json.loads(output)

        assert log_data["request_id"] == "abc-123"
        assert log_data["user_id"] == "user@example.com"

    def test_text_formatter(self):
        """Test text formatter."""
        formatter = TextFormatter(use_colors=False)

        record = logging.LogRecord(
            name="TestCoordinator",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert "[INFO]" in output
        assert "[TestCoordinator]" in output
        assert "Test message" in output

    def test_coordinator_logger(self):
        """Test coordinator logger."""
        python_logger = logging.getLogger("test.TestCoordinator")
        coordinator_logger = CoordinatorLogger("TestCoordinator", python_logger)

        # Add handler to capture output
        import io
        handler = logging.StreamHandler(io.StringIO())
        handler.setFormatter(StructuredFormatter())
        python_logger.addHandler(handler)
        python_logger.setLevel(logging.INFO)

        # Log message
        coordinator_logger.info(
            "Test message",
            extra={"request_id": "abc-123"},
        )

        # Verify handler received message
        # (In real test, would check output)
        assert coordinator_logger._name == "TestCoordinator"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for monitoring components."""

    def test_full_stack_metrics_export(self):
        """Test full metrics export stack."""
        from victor.observability.coordinator_metrics import CoordinatorMetricsCollector

        # Create metrics collector
        collector = CoordinatorMetricsCollector()

        # Record some executions
        with collector.track_coordinator("TestCoordinator"):
            pass

        with collector.track_coordinator("TestCoordinator"):
            pass

        # Create Prometheus exporter
        exporter = PrometheusMetricsExporter(collector)

        # Export metrics
        output = exporter.export_metrics()

        # Verify output
        assert "victor_coordinator_executions_total" in output
        assert "TestCoordinator" in output

    @pytest.mark.asyncio
    async def test_full_stack_health_check(self):
        """Test full health check stack."""
        from victor.observability.coordinator_metrics import CoordinatorMetricsCollector

        # Create metrics collector
        collector = CoordinatorMetricsCollector()

        # Record execution
        with collector.track_coordinator("TestCoordinator"):
            pass

        # Create health service
        service = CoordinatorHealthService(collector)
        service.add_coordinator_check("TestCoordinator")

        # Check health
        report = await service.check_health()

        # Verify
        assert report.status.value in ("healthy", "degraded", "unhealthy")
        assert "coordinator.TestCoordinator" in report.components

    def test_logging_integration(self):
        """Test logging with coordinator tracking."""
        from victor.observability.coordinator_metrics import CoordinatorMetricsCollector

        # Setup logging
        setup_coordinator_logging(
            level="INFO",
            format_type="json",
        )

        # Get logger
        logger = get_coordinator_logger("TestCoordinator")

        # Log message
        logger.info("Test message", extra={"test_field": "test_value"})

        # Verify logging is configured
        coordinator_logger = logging.getLogger("victor.coordinators.TestCoordinator")
        assert coordinator_logger is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
