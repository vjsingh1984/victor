"""Re-export facade for capability method mappings (DEPRECATED).

.. deprecated::
    Import from victor.core.capability_registry instead.
    This re-export facade will be removed in version 0.10.0.

CAPABILITY_METHOD_MAPPINGS and get_method_for_capability have been
consolidated into victor.core.capability_registry. This module is kept
for backward compatibility with existing importers.
"""

import warnings as _warnings

# Direct imports from canonical location
from victor.core.capability_registry import (
    CAPABILITY_METHOD_MAPPINGS,
    get_method_for_capability,
)

__all__ = [
    "CAPABILITY_METHOD_MAPPINGS",
    "get_method_for_capability",
]


# Emit deprecation warning on module import
_warnings.warn(
    "Import from victor.framework.capability_registry is deprecated. "
    "Use 'from victor.core.capability_registry import CAPABILITY_METHOD_MAPPINGS, get_method_for_capability' instead. "
    "This re-export facade will be removed in version 0.10.0.",
    DeprecationWarning,
    stacklevel=2,
)
