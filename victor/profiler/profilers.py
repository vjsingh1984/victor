"""DEPRECATED: Profilers has moved to victor.observability.profiler.profilers.

.. deprecated::
    Import from victor.observability.profiler.profilers instead.
    This module remains as a backward compatibility redirect.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.profiler.profilers is deprecated. Import from victor.observability.profiler.profilers instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.observability.profiler.profilers import *  # noqa: F403
