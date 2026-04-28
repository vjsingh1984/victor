"""DEPRECATED: Debug manager has moved to victor.observability.debug.manager.

.. deprecated::
    Import from victor.observability.debug.manager instead.
    This module remains as a backward compatibility redirect.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.debug.manager is deprecated. Import from victor.observability.debug.manager instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.observability.debug.manager import *  # noqa: F403
