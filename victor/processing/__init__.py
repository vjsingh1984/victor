# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Victor Processing Module - Unified processing-related components.

This module consolidates processing-related functionality:

- **serialization**: Token-optimized serialization for LLM communication
- **native**: Rust-powered native extensions for performance
- **editing**: File editing and transaction support
- **merge**: Merge conflict resolution
- **completion**: Code completion providers for IDE integration
- **file_types**: File type detection and categorization

Example usage:
    # Import from submodules
    from victor.processing.serialization import AdaptiveSerializer
    from victor.processing.native import batch_cosine_similarity
    from victor.processing.editing import FileEditor
    from victor.processing.merge import MergeManager
    from victor.processing.completion import CompletionManager
    from victor.processing.file_types import detect_file_type

    # Or import submodules directly
    from victor.processing import serialization, native, editing
"""

# Expose submodules for easy access
from victor.processing import (
    serialization,
    native,
    editing,
    merge,
    completion,
    file_types,
)

__all__ = [
    "serialization",
    "native",
    "editing",
    "merge",
    "completion",
    "file_types",
]
