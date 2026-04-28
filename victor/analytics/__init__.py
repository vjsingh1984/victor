"""Backward compatibility shim for victor.analytics.

.. deprecated::
    victor.analytics is deprecated. Import from victor.observability.analytics instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.analytics is deprecated. Import from victor.observability.analytics instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.observability.analytics import *  # noqa: F401, F403
