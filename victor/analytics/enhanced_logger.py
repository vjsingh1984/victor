"""DEPRECATED: Enhanced logger has moved to victor.observability.analytics.enhanced_logger.

.. deprecated::
    Import from victor.observability.analytics.enhanced_logger instead.
    This module remains as a backward compatibility redirect.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.analytics.enhanced_logger is deprecated. Import from victor.observability.analytics.enhanced_logger instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.observability.analytics.enhanced_logger import *  # noqa: F403
