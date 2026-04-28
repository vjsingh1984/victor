"""DEPRECATED: Debug registry has moved to victor.observability.debug.registry.

.. deprecated::
    Import from victor.observability.debug.registry instead.
    This module remains as a backward compatibility redirect.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.debug.registry is deprecated. Import from victor.observability.debug.registry instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.observability.debug.registry import *  # noqa: F403
