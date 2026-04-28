"""DEPRECATED: Profiler manager has moved to victor.observability.profiler.manager.

.. deprecated::
    Import from victor.observability.profiler.manager instead.
    This module remains as a backward compatibility redirect.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.profiler.manager is deprecated. Import from victor.observability.profiler.manager instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.observability.profiler.manager import *  # noqa: F403
