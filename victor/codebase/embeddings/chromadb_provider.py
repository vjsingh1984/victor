# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""ChromaDB provider module shim."""
import warnings

warnings.warn(
    "Importing from 'victor.codebase.embeddings.chromadb_provider' is deprecated. Please use 'victor_coding.codebase.embeddings.chromadb_provider' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from victor_coding.codebase.embeddings.chromadb_provider import *  # noqa: F401, F403
