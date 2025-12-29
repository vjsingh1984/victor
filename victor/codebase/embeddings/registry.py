# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Embedding registry module shim."""
import warnings

warnings.warn(
    "Importing from 'victor.codebase.embeddings.registry' is deprecated. Please use 'victor_coding.codebase.embeddings.registry' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from victor_coding.codebase.embeddings.registry import *  # noqa: F401, F403
