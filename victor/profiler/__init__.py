"""DEPRECATED: Profiler module has moved to victor.observability.profiler.

.. deprecated::
    Import from victor.observability.profiler instead.
    This module remains as a backward compatibility redirect.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.profiler is deprecated. Import from victor.observability.profiler instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.observability.profiler import *  # noqa: F403
