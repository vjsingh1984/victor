# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Ignore patterns module shim."""
import warnings
warnings.warn("Importing from 'victor.codebase.ignore_patterns' is deprecated. Please use 'victor_coding.codebase.ignore_patterns' instead.", DeprecationWarning, stacklevel=2)
from victor_coding.codebase.ignore_patterns import *  # noqa: F401, F403
