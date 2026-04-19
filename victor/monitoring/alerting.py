"""Performance alerting for tool registry operations.

This module provides alerting logic for detecting and responding to
performance degradation in tool registry operations.

Alerts can be triggered by:
- Performance regression (operation exceeds threshold)
- Error rate spike (error rate exceeds threshold)
- Cache degradation (hit rate drops below threshold)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Callable as CallableType
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Severity levels for alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """A performance alert.

    Attributes:
        alert_type: Type of alert (performance, error_rate, cache)
        severity: Severity level of the alert
        operation: Operation that triggered the alert
        message: Human-readable alert message
        current_value: Current metric value
        threshold_value: Threshold that was exceeded
        timestamp: When the alert was triggered
        metadata: Additional context about the alert
    """

    alert_type: str
    severity: AlertSeverity
    operation: str
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "operation": self.operation,
            "message": self.message,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class AlertHandler:
    """Base class for alert handlers.

    Subclasses can implement different alert delivery mechanisms
    (e.g., logging, email, Slack, PagerDuty).
    """

    def handle_alert(self, alert: PerformanceAlert) -> None:
        """Handle an alert.

        Args:
            alert: The alert to handle
        """
        raise NotImplementedError


class LoggingAlertHandler(AlertHandler):
    """Log alerts to the logging system."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize logging alert handler.

        Args:
            logger: Logger to use (defaults to module logger)
        """
        self.logger = logger or logging.getLogger(__name__)

    def handle_alert(self, alert: PerformanceAlert) -> None:
        """Log alert at appropriate severity level.

        Args:
            alert: The alert to log
        """
        log_message = (
            f"[{alert.alert_type.upper()}] {alert.operation}: {alert.message} "
            f"(current: {alert.current_value:.3f}, threshold: {alert.threshold_value:.3f})"
        )

        if alert.severity == AlertSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif alert.severity == AlertSeverity.ERROR:
            self.logger.error(log_message)
        elif alert.severity == AlertSeverity.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)


class PerformanceAlertManager:
    """Manages performance alerting for registry operations.

    Monitors metrics and triggers alerts when thresholds are exceeded.

    Thresholds:
    - Performance: operation duration exceeds baseline by %
    - Error rate: error rate exceeds %
    - Cache: hit rate drops below %
    """

    def __init__(
        self,
        performance_threshold_pct: float = 200.0,  # 2× baseline
        error_rate_threshold_pct: float = 5.0,  # 5% errors
        cache_hit_rate_threshold_pct: float = 50.0,  # 50% hit rate
        alert_handlers: Optional[List[AlertHandler]] = None
    ):
        """Initialize alert manager.

        Args:
            performance_threshold_pct: Performance regression threshold (%)
            error_rate_threshold_pct: Error rate threshold (%)
            cache_hit_rate_threshold_pct: Minimum cache hit rate (%)
            alert_handlers: List of alert handlers (defaults to logging)
        """
        self.performance_threshold_pct = performance_threshold_pct
        self.error_rate_threshold_pct = error_rate_threshold_pct
        self.cache_hit_rate_threshold_pct = cache_hit_rate_threshold_pct

        self.alert_handlers = alert_handlers or [LoggingAlertHandler()]
        self._baselines: Dict[str, float] = {}
        self._alert_history: List[PerformanceAlert] = []
        self._suppressed_until: Dict[str, datetime] = {}

    def set_baseline(self, operation: str, duration_ms: float) -> None:
        """Set baseline performance for an operation.

        Args:
            operation: Operation name
            duration_ms: Baseline duration in milliseconds
        """
        self._baselines[operation] = duration_ms
        logger.info(f"Set baseline for {operation}: {duration_ms:.3f}ms")

    def check_performance(self, operation: str, duration_ms: float) -> Optional[PerformanceAlert]:
        """Check if operation performance exceeds threshold.

        Args:
            operation: Operation name
            duration_ms: Current duration in milliseconds

        Returns:
            Alert if threshold exceeded, None otherwise
        """
        if operation not in self._baselines:
            return None

        baseline = self._baselines[operation]
        threshold = baseline * (self.performance_threshold_pct / 100.0)

        if duration_ms > threshold:
            alert = PerformanceAlert(
                alert_type="performance_regression",
                severity=AlertSeverity.WARNING,
                operation=operation,
                message=f"Operation exceeded {self.performance_threshold_pct:.0f}% of baseline",
                current_value=duration_ms,
                threshold_value=threshold,
                metadata={
                    "baseline_ms": baseline,
                    "regression_pct": ((duration_ms - baseline) / baseline) * 100
                }
            )
            self._trigger_alert(alert)
            return alert

        return None

    def check_error_rate(self, operation: str, metrics: Dict[str, Any]) -> Optional[PerformanceAlert]:
        """Check if error rate exceeds threshold.

        Args:
            operation: Operation name
            metrics: Operation metrics dictionary

        Returns:
            Alert if threshold exceeded, None otherwise
        """
        error_rate = metrics.get("error_rate", 0.0)

        if error_rate > self.error_rate_threshold_pct:
            alert = PerformanceAlert(
                alert_type="error_rate_spike",
                severity=AlertSeverity.ERROR,
                operation=operation,
                message=f"Error rate exceeded {self.error_rate_threshold_pct:.0f}%",
                current_value=error_rate,
                threshold_value=self.error_rate_threshold_pct,
                metadata={
                    "error_count": metrics.get("error_count", 0),
                    "total_count": metrics.get("count", 0),
                }
            )
            self._trigger_alert(alert)
            return alert

        return None

    def check_cache_hit_rate(self, cache_type: str, stats: Dict[str, Any]) -> Optional[PerformanceAlert]:
        """Check if cache hit rate drops below threshold.

        Args:
            cache_type: Type of cache (e.g., "feature_flag", "query")
            stats: Cache statistics dictionary

        Returns:
            Alert if threshold exceeded, None otherwise
        """
        hit_rate = stats.get("hit_rate", 0.0) * 100  # Convert to percentage

        if hit_rate < self.cache_hit_rate_threshold_pct:
            alert = PerformanceAlert(
                alert_type="cache_degradation",
                severity=AlertSeverity.WARNING,
                operation=f"{cache_type}_cache",
                message=f"Cache hit rate below {self.cache_hit_rate_threshold_pct:.0f}%",
                current_value=hit_rate,
                threshold_value=self.cache_hit_rate_threshold_pct,
                metadata={
                    "cache_type": cache_type,
                    "hits": stats.get("hits", 0),
                    "misses": stats.get("misses", 0),
                    "cache_size": stats.get("cache_size", 0),
                }
            )
            self._trigger_alert(alert)
            return alert

        return None

    def _trigger_alert(self, alert: PerformanceAlert) -> None:
        """Trigger alert to all handlers.

        Args:
            alert: Alert to trigger
        """
        # Check if alert is suppressed
        if alert.operation in self._suppressed_until:
            if datetime.now() < self._suppressed_until[alert.operation]:
                return
            else:
                # Suppression expired
                del self._suppressed_until[alert.operation]

        # Add to history
        self._alert_history.append(alert)

        # Keep only last 1000 alerts
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-1000:]

        # Send to all handlers
        for handler in self.alert_handlers:
            try:
                handler.handle_alert(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def suppress_alerts(self, operation: str, duration_minutes: int = 60) -> None:
        """Suppress alerts for an operation.

        Args:
            operation: Operation to suppress alerts for
            duration_minutes: Duration to suppress alerts (default 60 minutes)
        """
        until = datetime.now() + timedelta(minutes=duration_minutes)
        self._suppressed_until[operation] = until
        logger.info(f"Suppressed alerts for {operation} until {until.isoformat()}")

    def get_alert_history(
        self,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100
    ) -> List[PerformanceAlert]:
        """Get alert history.

        Args:
            severity: Filter by severity (None for all)
            limit: Maximum number of alerts to return

        Returns:
            List of alerts (most recent first)
        """
        alerts = self._alert_history

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts[-limit:][::-1]  # Most recent first

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alerts.

        Returns:
            Dictionary with alert statistics
        """
        total_alerts = len(self._alert_history)

        # Count by severity
        severity_counts = {}
        for alert in self._alert_history:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Count by type
        type_counts = {}
        for alert in self._alert_history:
            alert_type = alert.alert_type
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1

        return {
            "total_alerts": total_alerts,
            "by_severity": severity_counts,
            "by_type": type_counts,
            "suppressed_operations": list(self._suppressed_until.keys()),
        }


__all__ = [
    "AlertSeverity",
    "PerformanceAlert",
    "AlertHandler",
    "LoggingAlertHandler",
    "PerformanceAlertManager",
]
