# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""SQLite graph store module shim."""
import warnings

warnings.warn(
    "Importing from 'victor.codebase.graph.sqlite_store' is deprecated. Please use 'victor_coding.codebase.graph.sqlite_store' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from victor_coding.codebase.graph.sqlite_store import *  # noqa: F401, F403
