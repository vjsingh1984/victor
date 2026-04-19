"""Performance monitoring for Victor AI framework.

This package provides metrics collection, alerting, and performance
monitoring for tool registry operations.

Components:
- RegistryMetricsCollector: Collect operation metrics and cache stats
- PerformanceAlertManager: Detect and alert on performance degradation
- Integrations with Prometheus and Grafana

Usage:
    >>> from victor.monitoring import get_metrics_collector, get_alert_manager
    >>>
    >>> # Collect metrics
    >>> metrics = get_metrics_collector()
    >>> with metrics.record_operation("register_tool"):
    ...     registry.register(tool)
    >>>
    >>> # Check for alerts
    >>> alert_manager = get_alert_manager()
    >>> alert_manager.check_performance("register_tool", duration_ms=5.0)
"""

from typing import Optional

from victor.monitoring.registry_metrics import (
    RegistryMetricsCollector,
    OperationMetrics,
    get_metrics_collector,
    monitored_operation,
)

from victor.monitoring.alerting import (
    AlertSeverity,
    PerformanceAlert,
    AlertHandler,
    LoggingAlertHandler,
    PerformanceAlertManager,
)

__all__ = [
    "RegistryMetricsCollector",
    "OperationMetrics",
    "get_metrics_collector",
    "monitored_operation",
    "AlertSeverity",
    "PerformanceAlert",
    "AlertHandler",
    "LoggingAlertHandler",
    "PerformanceAlertManager",
]

# Global instances
_global_alert_manager: Optional[PerformanceAlertManager] = None


def get_alert_manager() -> PerformanceAlertManager:
    """Get the global alert manager instance.

    Returns:
        Global PerformanceAlertManager instance
    """
    global _global_alert_manager

    if _global_alert_manager is None:
        from victor.monitoring.alerting import LoggingAlertHandler
        _global_alert_manager = PerformanceAlertManager(
            alert_handlers=[LoggingAlertHandler()]
        )

    return _global_alert_manager
