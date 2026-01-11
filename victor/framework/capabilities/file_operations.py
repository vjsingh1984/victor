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

"""Generic file operations capability provider (Phase 3)."""

from typing import List, Set
from dataclasses import dataclass
from enum import Enum


class FileOperationType(Enum):
    """Types of file operations."""

    READ = "read"
    WRITE = "write"
    EDIT = "edit"
    SEARCH = "search"


@dataclass
class FileOperation:
    """A file operation configuration."""

    operation: FileOperationType
    tool_name: str
    required: bool = True


class FileOperationsCapability:
    """Generic file operations capability provider.

    Provides common file operation tools that are used across multiple
    verticals (Coding, DevOps, RAG, Research, DataAnalysis).

    This promotes the DRY principle and reduces code duplication by
    centralizing file operation tool definitions.

    Phase 3: Generic Capabilities - Framework Layer
    """

    DEFAULT_OPERATIONS = [
        FileOperation(FileOperationType.READ, "read", required=True),
        FileOperation(FileOperationType.WRITE, "write", required=True),
        FileOperation(FileOperationType.EDIT, "edit", required=True),
        FileOperation(FileOperationType.SEARCH, "grep", required=True),
    ]

    def __init__(
        self,
        operations: List[FileOperation] = None,
    ):
        """Initialize file operations capability.

        Args:
            operations: List of file operations (uses DEFAULT if None)
        """
        self.operations = operations or self.DEFAULT_OPERATIONS.copy()

    def get_tools(self) -> Set[str]:
        """Get tool names for this capability.

        Returns:
            Set of tool names
        """
        return {op.tool_name for op in self.operations if op.required}

    def get_tool_list(self) -> List[str]:
        """Get tool names as a list.

        Returns:
            List of tool names
        """
        return [op.tool_name for op in self.operations if op.required]
