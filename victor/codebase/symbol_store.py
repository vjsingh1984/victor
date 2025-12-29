# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Symbol store module shim."""
import warnings

warnings.warn(
    "Importing from 'victor.codebase.symbol_store' is deprecated. Please use 'victor_coding.codebase.symbol_store' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from victor_coding.codebase.symbol_store import *  # noqa: F401, F403
