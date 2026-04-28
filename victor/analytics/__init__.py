"""DEPRECATED: Analytics module has moved to victor.observability.analytics.

.. deprecated::
    Import from victor.observability.analytics instead.
    This module remains as a backward compatibility redirect.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.analytics is deprecated. Import from victor.observability.analytics instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.observability.analytics import *  # noqa: F403

__all__ = []  # Re-export everything from observability.analytics
