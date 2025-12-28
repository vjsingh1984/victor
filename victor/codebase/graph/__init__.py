# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Codebase graph storage module.

.. deprecated:: 0.3.0
    This module has moved to ``victor_coding.codebase.graph``.
    Please update your imports. This shim will be removed in version 0.5.0.
"""

import warnings

warnings.warn(
    "Importing from 'victor.codebase.graph' is deprecated. "
    "Please use 'victor_coding.codebase.graph' instead. "
    "This compatibility shim will be removed in version 0.5.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from victor_coding for backward compatibility
from victor_coding.codebase.graph import *  # noqa: F401, F403
from victor_coding.codebase.graph import __all__  # noqa: F401
