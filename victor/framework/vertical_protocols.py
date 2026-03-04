# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Vertical integration protocols for framework-vertical dependency-free integration.

This module provides protocol definitions that enable framework tools to work
with vertical-specific implementations (victor-coding, victor-rag, etc.) without
creating direct dependencies on external packages.

SOLID Principles:
- DIP (Dependency Inversion): Framework depends on protocols, not concrete implementations
- ISP (Interface Segregation): Focused protocols for each capability
- OCP (Open/Closed): New verticals can implement protocols without framework changes
- LSP (Liskov Substitution): All protocol implementations are interchangeable
- SRP (Single Responsibility): Each protocol handles one capability

Usage:
    from victor.framework.vertical_protocols import EditorProtocol

    # Use any editor implementation
    editor: EditorProtocol = get_editor()
    result = await editor.edit_file(file_path, edits)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# =============================================================================
# File Editing Protocols
# =============================================================================


@runtime_checkable
class EditorProtocol(Protocol):
    """
    Protocol for file editing operations.

    This protocol defines the interface for file editors, allowing the
    framework to work with multiple implementations without direct
    dependencies on external vertical packages.

    Implementations can be provided by:
    - victor.contrib.editing (default diff-based editor)
    - victor-coding (advanced coding-specific editor)
    - Custom implementations

    Example:
        class MyEditor:
            async def edit_file(
                self,
                file_path: str,
                edits: List[EditOperation],
                preview: bool = False,
                **kwargs: Any,
            ) -> EditResult:
                # Apply edits to file
                ...
    """

    async def edit_file(
        self,
        file_path: str,
        edits: List["EditOperation"],
        preview: bool = False,
        **kwargs: Any,
    ) -> "EditResult":
        """Apply edits to a file."""
        ...

    async def validate_edit(
        self,
        file_path: str,
        old_str: str,
        new_str: str,
        **kwargs: Any,
    ) -> "EditValidationResult":
        """Validate an edit operation before applying."""
        ...

    def get_editor_info(self) -> Dict[str, Any]:
        """Get editor metadata."""
        ...


@dataclass
class EditOperation:
    """Single edit operation.

    Attributes:
        old_str: String to find in file (must match exactly)
        new_str: Replacement string
        start_line: Optional start line for context (0-indexed)
        end_line: Optional end line for context (0-indexed)
        allow_multiple: If False, error if old_str appears multiple times
    """

    old_str: str
    new_str: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    allow_multiple: bool = False


@dataclass
class EditResult:
    """Result of file edit operation.

    Attributes:
        success: Whether the edit was successful
        file_path: Path to the edited file
        edits_applied: Number of edits successfully applied
        edits_failed: Number of edits that failed
        preview: Optional preview of changes
        error: Optional error message if edit failed
        metadata: Optional implementation-specific metadata
    """

    success: bool
    file_path: str
    edits_applied: int
    edits_failed: int = 0
    preview: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_edits(self) -> int:
        """Total number of edits processed."""
        return self.edits_applied + self.edits_failed


@dataclass
class EditValidationResult:
    """Result of edit validation.

    Attributes:
        valid: Whether the edit is valid
        file_path: Path to the file validated against
        old_str_found: Whether old_str was found in file
        match_count: Number of times old_str appears in file
        error: Optional error message if validation failed
        warnings: Optional list of warnings about the edit
    """

    valid: bool
    file_path: str
    old_str_found: bool
    match_count: int = 0
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Editing protocols
    "EditorProtocol",
    "EditOperation",
    "EditResult",
    "EditValidationResult",
]
