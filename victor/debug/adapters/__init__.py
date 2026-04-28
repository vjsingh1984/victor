"""DEPRECATED: Debug adapters has moved to victor.observability.debug.adapters.

.. deprecated::
    Import from victor.observability.debug.adapters instead.
    This module remains as a backward compatibility redirect.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "victor.debug.adapters is deprecated. Import from victor.observability.debug.adapters instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.observability.debug.adapters import *  # noqa: F403
