"""DEPRECATED: Debug adapter has moved to victor.observability.debug.adapter.

.. deprecated::
    Import from victor.observability.debug.adapter instead.
    This module remains as a backward compatibility redirect.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.debug.adapter is deprecated. Import from victor.observability.debug.adapter instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.observability.debug.adapter import *  # noqa: F403
