# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Embeddings base module shim."""
import warnings

warnings.warn(
    "Importing from 'victor.codebase.embeddings.base' is deprecated. Please use 'victor_coding.codebase.embeddings.base' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from victor_coding.codebase.embeddings.base import *  # noqa: F401, F403
