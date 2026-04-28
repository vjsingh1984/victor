"""DEPRECATED: Debug module has moved to victor.observability.debug.

.. deprecated::
    Import from victor.observability.debug instead.
    This module remains as a backward compatibility redirect.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.debug is deprecated. Import from victor.observability.debug instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.observability.debug import *  # noqa: F403

__all__ = []  # Re-export everything from observability.debug
