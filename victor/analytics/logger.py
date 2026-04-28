"""DEPRECATED: Logger has moved to victor.observability.analytics.logger.

.. deprecated::
    Import from victor.observability.analytics.logger instead.
    This module remains as a backward compatibility redirect.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.analytics.logger is deprecated. Import from victor.observability.analytics.logger instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.observability.analytics.logger import *  # noqa: F403
