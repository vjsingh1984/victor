"""DEPRECATED: Debug protocol has moved to victor.observability.debug.protocol.

.. deprecated::
    Import from victor.observability.debug.protocol instead.
    This module remains as a backward compatibility redirect.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.debug.protocol is deprecated. Import from victor.observability.debug.protocol instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.observability.debug.protocol import *  # noqa: F403
