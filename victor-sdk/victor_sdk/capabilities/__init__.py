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

"""Zero-dependency capability contracts for external vertical packages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Set


class FileOperationType(Enum):
    """Types of generic file operations exposed by a vertical."""

    READ = "read"
    WRITE = "write"
    EDIT = "edit"
    SEARCH = "search"


@dataclass(frozen=True)
class FileOperation:
    """Serializable file operation requirement."""

    operation: FileOperationType
    tool_name: str
    required: bool = True


class FileOperationsCapability:
    """Declarative file-operation capability contract."""

    DEFAULT_OPERATIONS: ClassVar[List[FileOperation]] = [
        FileOperation(FileOperationType.READ, "read", required=True),
        FileOperation(FileOperationType.WRITE, "write", required=True),
        FileOperation(FileOperationType.EDIT, "edit", required=True),
        FileOperation(FileOperationType.SEARCH, "grep", required=True),
    ]

    def __init__(self, operations: Optional[Iterable[FileOperation]] = None) -> None:
        self.operations = (
            list(operations) if operations is not None else list(self.DEFAULT_OPERATIONS)
        )

    def get_tools(self) -> Set[str]:
        """Return the required tool names for this capability."""

        return {op.tool_name for op in self.operations if op.required}

    def get_tool_list(self) -> List[str]:
        """Return the required tool names in declaration order."""

        return [op.tool_name for op in self.operations if op.required]


@dataclass(frozen=True)
class PromptContribution:
    """Serializable prompt contribution contract."""

    name: str
    task_type: str
    hint: str
    tool_budget: int = 15
    priority: int = 50
    system_section: str = ""


class PromptContributionCapability:
    """Declarative prompt contribution capability contract."""

    COMMON_HINTS: ClassVar[List[PromptContribution]] = [
        PromptContribution(
            name="read_first",
            task_type="edit",
            hint="Always read the file before making edits",
            tool_budget=5,
        ),
        PromptContribution(
            name="verify_changes",
            task_type="edit",
            hint="Verify changes compile and pass tests",
            tool_budget=10,
        ),
        PromptContribution(
            name="search_code",
            task_type="search",
            hint="Use grep to search code before reading",
            tool_budget=10,
        ),
    ]

    def __init__(
        self,
        contributions: Optional[Iterable[PromptContribution]] = None,
    ) -> None:
        self.contributions = (
            list(contributions) if contributions is not None else list(self.COMMON_HINTS)
        )

    def get_task_hints(self) -> Dict[str, Dict[str, Any]]:
        """Return prompt hints in a pure-serializable form."""

        hints: Dict[str, Dict[str, Any]] = {}
        for contribution in self.contributions:
            hints[contribution.task_type] = {
                "hint": contribution.hint,
                "tool_budget": contribution.tool_budget,
            }
        return hints


__all__ = [
    "FileOperation",
    "FileOperationsCapability",
    "FileOperationType",
    "PromptContribution",
    "PromptContributionCapability",
]
