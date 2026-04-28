"""DEPRECATED: Profiler protocol has moved to victor.observability.profiler.protocol.

.. deprecated::
    Import from victor.observability.profiler.protocol instead.
    This module remains as a backward compatibility redirect.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.profiler.protocol is deprecated. Import from victor.observability.profiler.protocol instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.observability.profiler.protocol import *  # noqa: F403
