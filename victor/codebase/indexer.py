# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Indexer module shim."""
import warnings
warnings.warn("Importing from 'victor.codebase.indexer' is deprecated. Please use 'victor_coding.codebase.indexer' instead.", DeprecationWarning, stacklevel=2)
from victor_coding.codebase.indexer import *  # noqa: F401, F403
