# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Languages base module shim."""
import warnings

warnings.warn(
    "Importing from 'victor.languages.base' is deprecated. Please use 'victor_coding.languages.base' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from victor_coding.languages.base import *  # noqa: F401, F403
