"""Backward compatibility shim for victor.monitoring.

.. deprecated::
    victor.monitoring is deprecated and will be removed in a future version.
    This module should not be used in new code.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.monitoring is deprecated. This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

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
