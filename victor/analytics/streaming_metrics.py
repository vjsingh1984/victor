"""Backward compatibility shim for victor.analytics.streaming_metrics.

.. deprecated::
    Import from victor.observability.analytics.streaming_metrics instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.analytics.streaming_metrics is deprecated. "
    "Import from victor.observability.analytics.streaming_metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.observability.analytics.streaming_metrics import (  # noqa: F401, F403
    StreamingMetricsCollector,
    StreamMetrics,
)

__all__ = ["StreamingMetricsCollector", "StreamMetrics"]
