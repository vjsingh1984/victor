# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Editor module shim.

.. deprecated:: 0.3.0
    This module has moved to ``victor_coding.editing.editor``.
    Please update your imports. This shim will be removed in version 0.5.0.
"""

import warnings

warnings.warn(
    "Importing from 'victor.editing.editor' is deprecated. "
    "Please use 'victor_coding.editing.editor' instead. "
    "This compatibility shim will be removed in version 0.5.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from victor_coding for backward compatibility
from victor_coding.editing.editor import *  # noqa: F401, F403
