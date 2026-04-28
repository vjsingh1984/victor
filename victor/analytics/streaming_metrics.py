"""DEPRECATED: Streaming metrics has moved to victor.observability.analytics.streaming_metrics.

.. deprecated::
    Import from victor.observability.analytics.streaming_metrics instead.
    This module remains as a backward compatibility redirect.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.analytics.streaming_metrics is deprecated. Import from victor.observability.analytics.streaming_metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.observability.analytics.streaming_metrics import *  # noqa: F403
