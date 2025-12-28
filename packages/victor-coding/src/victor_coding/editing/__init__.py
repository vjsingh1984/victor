# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""File editing and transaction support.

This module re-exports from the core victor.editing module for backward
compatibility. The file editing infrastructure is now part of victor-core
and available to all verticals.

For new code, prefer importing directly from victor.editing:
    from victor.editing import FileEditor, EditTransaction
"""

# Re-export from victor-core
from victor.editing import (
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
