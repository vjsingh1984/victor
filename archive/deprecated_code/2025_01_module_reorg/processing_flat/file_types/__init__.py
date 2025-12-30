# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""This module has moved to victor.processing.file_types.

This module is maintained for backward compatibility only.
Please update your imports to use the new location:

    # OLD:
    from victor.file_types import detect_file_type, FileCategory

    # NEW (preferred):
    from victor.processing.file_types import detect_file_type, FileCategory
"""

# Re-export everything from the new location for backward compatibility
from victor.processing.file_types import (
    # Types
    FileCategory,
    FileType,
    FileTypeDetector,
    # Registry
    FileTypeRegistry,
    # Functions
    detect_file_type,
    get_file_category,
    is_code_file,
    is_config_file,
    is_data_file,
)

__all__ = [
    # Types
    "FileCategory",
    "FileType",
    "FileTypeDetector",
    # Registry
    "FileTypeRegistry",
    # Functions
    "detect_file_type",
    "get_file_category",
    "is_code_file",
    "is_config_file",
    "is_data_file",
]
