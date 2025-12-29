# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""LanceDB provider module shim."""
import warnings

warnings.warn(
    "Importing from 'victor.codebase.embeddings.lancedb_provider' is deprecated. Please use 'victor_coding.codebase.embeddings.lancedb_provider' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from victor_coding.codebase.embeddings.lancedb_provider import *  # noqa: F401, F403
