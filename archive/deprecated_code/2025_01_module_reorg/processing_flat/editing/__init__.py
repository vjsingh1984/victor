# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""This module has moved to victor.processing.editing.

This module is maintained for backward compatibility only.
Please update your imports to use the new location:

    # OLD:
    from victor.editing import FileEditor

    # NEW (preferred):
    from victor.processing.editing import FileEditor
"""

# Re-export everything from the new location for backward compatibility
from victor.processing.editing import (
    OperationType,
    EditOperation,
    EditTransaction,
    FileEditor,
)

__all__ = [
    "OperationType",
    "EditOperation",
    "EditTransaction",
    "FileEditor",
]
