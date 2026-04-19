"""Tests for performance monitoring modules."""

import time
from unittest.mock import patch, MagicMock

import pytest

from victor.monitoring.registry_metrics import (
    OperationMetrics,
    RegistryMetricsCollector,
    get_metrics_collector,
    monitored_operation,
)
from victor.monitoring.alerting import (
    AlertSeverity,
    PerformanceAlert,
    LoggingAlertHandler,
    PerformanceAlertManager,
)


class TestOperationMetrics:
    """Tests for OperationMetrics dataclass."""

    def test_record_operation(self):
        """Test recording a single operation."""
        metrics = OperationMetrics(operation="test_op")

        metrics.record(10.0, error=None)

        assert metrics.count == 1
        assert metrics.total_duration_ms == 10.0
        assert metrics.min_duration_ms == 10.0
        assert metrics.max_duration_ms == 10.0
        assert metrics.last_duration_ms == 10.0
        assert metrics.error_count == 0

    def test_record_multiple_operations(self):
        """Test recording multiple operations."""
        metrics = OperationMetrics(operation="test_op")

        metrics.record(10.0)
        metrics.record(20.0)
        metrics.record(30.0)

        assert metrics.count == 3
        assert metrics.total_duration_ms == 60.0
        assert metrics.min_duration_ms == 10.0
        assert metrics.max_duration_ms == 30.0
        assert metrics.avg_duration_ms == 20.0

    def test_record_errors(self):
        """Test recording operations with errors."""
        metrics = OperationMetrics(operation="test_op")

        metrics.record(10.0, error="Test error")
        metrics.record(20.0)  # Success
        metrics.record(30.0, error="Another error")

        assert metrics.count == 3
        assert metrics.error_count == 2
        assert abs(metrics.error_rate - 66.67) < 0.01  # Allow floating point tolerance
        assert metrics.last_error == "Another error"

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = OperationMetrics(operation="test_op")
        metrics.record(10.0)
        metrics.record(20.0, error="Test error")

        data = metrics.to_dict()

        assert data["operation"] == "test_op"
        assert data["count"] == 2
        assert data["avg_duration_ms"] == 15.0
        assert data["error_rate"] == 50.0
        assert "last_update" in data


class TestRegistryMetricsCollector:
    """Tests for RegistryMetricsCollector."""

    def test_collector_initialization(self):
        """Test collector initialization."""
        collector = RegistryMetricsCollector(enabled=True)

        assert collector._enabled is True
        assert len(collector._metrics) == 0
        assert collector._cache_metrics["feature_flag_hits"] == 0

    def test_disabled_collector(self):
        """Test that disabled collector doesn't record."""
        collector = RegistryMetricsCollector(enabled=False)

        with collector.record_operation("test_op"):
            time.sleep(0.01)

        # Should not record anything
        assert len(collector._metrics) == 0

    def test_record_operation_context_manager(self):
        """Test recording operation with context manager."""
        collector = RegistryMetricsCollector(enabled=True)

        def test_operation():
            time.sleep(0.01)
            return "result"

        with collector.record_operation("test_op"):
            result = test_operation()

        assert result == "result"

        metrics = collector.get_metrics("test_op")
        assert metrics["count"] == 1
        assert metrics["avg_duration_ms"] > 0

    def test_record_operation_with_error(self):
        """Test recording operation that raises error."""
        collector = RegistryMetricsCollector(enabled=True)

        with pytest.raises(ValueError):
            with collector.record_operation("test_op"):
                raise ValueError("Test error")

        metrics = collector.get_metrics("test_op")
        assert metrics["error_count"] == 1
        assert metrics["last_error"] == "Test error"

    def test_get_metrics_single_operation(self):
        """Test getting metrics for specific operation."""
        collector = RegistryMetricsCollector(enabled=True)

        with collector.record_operation("op1"):
            pass
        with collector.record_operation("op2"):
            pass

        op1_metrics = collector.get_metrics("op1")
        assert op1_metrics["operation"] == "op1"
        assert op1_metrics["count"] == 1

    def test_get_metrics_all_operations(self):
        """Test getting metrics for all operations."""
        collector = RegistryMetricsCollector(enabled=True)

        for i in range(3):
            with collector.record_operation(f"op{i}"):
                pass

        all_metrics = collector.get_metrics()
        assert len(all_metrics) == 3
        assert "op0" in all_metrics
        assert "op1" in all_metrics
        assert "op2" in all_metrics

    def test_record_cache_stats(self):
        """Test recording cache statistics."""
        collector = RegistryMetricsCollector(enabled=True)

        stats = {
            "hits": 100,
            "misses": 20,
            "evictions": 5,
            "cache_size": 50,
            "hit_rate": 0.83
        }

        collector.record_cache_stats("query", stats)

        assert collector._cache_metrics["query_hits"] == 100
        assert collector._cache_metrics["query_misses"] == 20
        assert collector._cache_metrics["query_evictions"] == 5
        assert collector._cache_metrics["query_size"] == 50

    def test_get_summary(self):
        """Test getting metrics summary."""
        collector = RegistryMetricsCollector(enabled=True)

        # Record some operations
        with collector.record_operation("fast_op"):
            time.sleep(0.001)
        with collector.record_operation("slow_op"):
            time.sleep(0.01)

        summary = collector.get_summary()

        assert summary["total_operations"] == 2
        assert summary["operation_count"] == 2
        assert "slowest_operations" in summary
        assert summary["slowest_operations"][0]["operation"] == "slow_op"

    def test_reset_metrics(self):
        """Test resetting all metrics."""
        collector = RegistryMetricsCollector(enabled=True)

        with collector.record_operation("test_op"):
            pass

        assert len(collector._metrics) > 0

        collector.reset()

        assert len(collector._metrics) == 0
        assert collector._cache_metrics["feature_flag_hits"] == 0

    def test_monitored_operation_decorator(self):
        """Test @monitored_operation decorator."""
        collector = get_metrics_collector()

        @monitored_operation("decorated_op")
        def my_function(x: int) -> int:
            return x * 2

        result = my_function(5)

        assert result == 10

        metrics = collector.get_metrics("decorated_op")
        assert metrics["count"] == 1


class TestPerformanceAlert:
    """Tests for PerformanceAlert dataclass."""

    def test_alert_creation(self):
        """Test creating an alert."""
        alert = PerformanceAlert(
            alert_type="performance",
            severity=AlertSeverity.WARNING,
            operation="test_op",
            message="Test alert",
            current_value=100.0,
            threshold_value=50.0
        )

        assert alert.alert_type == "performance"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.current_value == 100.0
        assert alert.threshold_value == 50.0

    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        alert = PerformanceAlert(
            alert_type="performance",
            severity=AlertSeverity.ERROR,
            operation="test_op",
            message="Test alert",
            current_value=100.0,
            threshold_value=50.0,
            metadata={"baseline_ms": 25.0}
        )

        data = alert.to_dict()

        assert data["alert_type"] == "performance"
        assert data["severity"] == "error"
        assert data["operation"] == "test_op"
        assert data["metadata"]["baseline_ms"] == 25.0


class TestLoggingAlertHandler:
    """Tests for LoggingAlertHandler."""

    def test_handle_alert(self, caplog):
        """Test handling alert with logging."""
        import logging
        caplog.set_level(logging.INFO)

        handler = LoggingAlertHandler()
        alert = PerformanceAlert(
            alert_type="test",
            severity=AlertSeverity.INFO,
            operation="test_op",
            message="Test alert",
            current_value=100.0,
            threshold_value=50.0
        )

        handler.handle_alert(alert)

        assert "test_op" in caplog.text
        assert "Test alert" in caplog.text

    def test_severity_levels(self, caplog):
        """Test different severity levels."""
        import logging
        caplog.set_level(logging.DEBUG)

        handler = LoggingAlertHandler()

        # Test WARNING
        alert_warning = PerformanceAlert(
            alert_type="test",
            severity=AlertSeverity.WARNING,
            operation="test_op",
            message="Warning",
            current_value=100.0,
            threshold_value=50.0
        )
        handler.handle_alert(alert_warning)
        assert "WARNING" in caplog.text

        # Test ERROR
        alert_error = PerformanceAlert(
            alert_type="test",
            severity=AlertSeverity.ERROR,
            operation="test_op",
            message="Error",
            current_value=100.0,
            threshold_value=50.0
        )
        handler.handle_alert(alert_error)
        assert "ERROR" in caplog.text


class TestPerformanceAlertManager:
    """Tests for PerformanceAlertManager."""

    def test_set_baseline(self):
        """Test setting performance baseline."""
        manager = PerformanceAlertManager()

        manager.set_baseline("test_op", 10.0)

        assert "test_op" in manager._baselines
        assert manager._baselines["test_op"] == 10.0

    def test_check_performance_no_baseline(self):
        """Test performance check without baseline."""
        manager = PerformanceAlertManager()

        alert = manager.check_performance("test_op", 5.0)

        assert alert is None  # No baseline set

    def test_check_performance_within_threshold(self):
        """Test performance check within threshold."""
        manager = PerformanceAlertManager(
            performance_threshold_pct=200.0  # 2×
        )
        manager.set_baseline("test_op", 10.0)

        # 15ms is within 2× baseline (20ms)
        alert = manager.check_performance("test_op", 15.0)

        assert alert is None

    def test_check_performance_exceeds_threshold(self):
        """Test performance check exceeds threshold."""
        manager = PerformanceAlertManager(
            performance_threshold_pct=200.0  # 2×
        )

        # Mock handler to capture alert
        mock_handler = MagicMock()
        manager.alert_handlers = [mock_handler]

        manager.set_baseline("test_op", 10.0)

        # 30ms exceeds 2× baseline (20ms)
        alert = manager.check_performance("test_op", 30.0)

        assert alert is not None
        assert alert.alert_type == "performance_regression"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.current_value == 30.0
        assert alert.threshold_value == 20.0

        # Verify handler was called
        mock_handler.handle_alert.assert_called_once()

    def test_check_error_rate_within_threshold(self):
        """Test error rate check within threshold."""
        manager = PerformanceAlertManager(
            error_rate_threshold_pct=5.0
        )

        metrics = {
            "error_rate": 2.0,  # Below 5%
            "error_count": 1,
            "count": 50
        }

        alert = manager.check_error_rate("test_op", metrics)

        assert alert is None

    def test_check_error_rate_exceeds_threshold(self):
        """Test error rate check exceeds threshold."""
        manager = PerformanceAlertManager(
            error_rate_threshold_pct=5.0
        )

        # Mock handler
        mock_handler = MagicMock()
        manager.alert_handlers = [mock_handler]

        metrics = {
            "error_rate": 10.0,  # Exceeds 5%
            "error_count": 5,
            "count": 50
        }

        alert = manager.check_error_rate("test_op", metrics)

        assert alert is not None
        assert alert.alert_type == "error_rate_spike"
        assert alert.severity == AlertSeverity.ERROR

    def test_check_cache_hit_rate_within_threshold(self):
        """Test cache hit rate check within threshold."""
        manager = PerformanceAlertManager(
            cache_hit_rate_threshold_pct=50.0
        )

        stats = {
            "hit_rate": 0.8,  # 80% - above 50%
            "hits": 80,
            "misses": 20,
            "cache_size": 100
        }

        alert = manager.check_cache_hit_rate("query", stats)

        assert alert is None

    def test_check_cache_hit_rate_below_threshold(self):
        """Test cache hit rate below threshold."""
        manager = PerformanceAlertManager(
            cache_hit_rate_threshold_pct=50.0
        )

        # Mock handler
        mock_handler = MagicMock()
        manager.alert_handlers = [mock_handler]

        stats = {
            "hit_rate": 0.3,  # 30% - below 50%
            "hits": 30,
            "misses": 70,
            "cache_size": 100
        }

        alert = manager.check_cache_hit_rate("query", stats)

        assert alert is not None
        assert alert.alert_type == "cache_degradation"
        assert alert.severity == AlertSeverity.WARNING

    def test_alert_suppression(self):
        """Test alert suppression."""
        manager = PerformanceAlertManager()
        manager.set_baseline("test_op", 10.0)

        # Suppress alerts for 1 minute
        manager.suppress_alerts("test_op", duration_minutes=1)

        # Should trigger alert but it will be suppressed
        alert = manager.check_performance("test_op", 30.0)

        # Alert is created but handler not called
        assert alert is not None
        assert "test_op" in manager._suppressed_until

    def test_get_alert_history(self):
        """Test getting alert history."""
        manager = PerformanceAlertManager()
        manager.set_baseline("test_op", 10.0)

        # Trigger some alerts
        manager.check_performance("test_op", 30.0)
        manager.check_performance("test_op", 40.0)

        history = manager.get_alert_history()

        assert len(history) == 2
        assert history[0].current_value == 40.0  # Most recent first

    def test_get_alert_summary(self):
        """Test getting alert summary."""
        manager = PerformanceAlertManager()
        manager.set_baseline("test_op", 10.0)

        # Trigger alerts
        manager.check_performance("test_op", 30.0)

        summary = manager.get_alert_summary()

        assert summary["total_alerts"] == 1
        assert summary["by_severity"]["warning"] == 1
        assert summary["by_type"]["performance_regression"] == 1
