"""Backward compatibility shim for victor.analytics.logger.

.. deprecated::
    Import from victor.observability.analytics.logger instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.analytics.logger is deprecated. "
    "Import from victor.observability.analytics.logger instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.observability.analytics.logger import *  # noqa: F401, F403

__all__ = ["UsageLogger"]
