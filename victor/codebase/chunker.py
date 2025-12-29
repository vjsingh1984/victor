# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Chunker module shim."""
import warnings

warnings.warn(
    "Importing from 'victor.codebase.chunker' is deprecated. Please use 'victor_coding.codebase.chunker' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from victor_coding.codebase.chunker import *  # noqa: F401, F403
