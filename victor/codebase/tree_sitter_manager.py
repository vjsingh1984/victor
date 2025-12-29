# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Tree-sitter manager module shim."""
import warnings

warnings.warn(
    "Importing from 'victor.codebase.tree_sitter_manager' is deprecated. Please use 'victor_coding.codebase.tree_sitter_manager' instead.",
    DeprecationWarning,
    stacklevel=2,
)
from victor_coding.codebase.tree_sitter_manager import *  # noqa: F401, F403

# Also export private variables for backward compatibility with tests
from victor_coding.codebase.tree_sitter_manager import _language_cache, _parser_cache  # noqa: F401
