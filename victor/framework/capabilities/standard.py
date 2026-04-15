"""Standard capability implementations moved from SDK.

FileOperationsCapability and PromptContributionCapability have runtime
logic (get_tools(), get_task_hints()) that violates the zero-dependency
SDK principle. They now live here in the framework.

The SDK keeps the pure data types (FileOperation, FileOperationType,
PromptContribution) and provides a backward-compat re-export shim.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Iterable, List, Optional, Set

from victor_sdk.capabilities import (
    FileOperation,
    FileOperationType,
    PromptContribution,
)


class FileOperationsCapability:
    """Declarative file-operation capability with tool resolution."""

    DEFAULT_OPERATIONS: ClassVar[List[FileOperation]] = [
        FileOperation(FileOperationType.READ, "read", required=True),
        FileOperation(FileOperationType.WRITE, "write", required=True),
        FileOperation(FileOperationType.EDIT, "edit", required=True),
        FileOperation(FileOperationType.SEARCH, "grep", required=True),
    ]

    def __init__(self, operations: Optional[Iterable[FileOperation]] = None) -> None:
        self.operations = (
            list(operations)
            if operations is not None
            else list(self.DEFAULT_OPERATIONS)
        )

    def get_tools(self) -> Set[str]:
        """Return the required tool names for this capability."""
        return {op.tool_name for op in self.operations if op.required}

    def get_tool_list(self) -> List[str]:
        """Return the required tool names in declaration order."""
        return [op.tool_name for op in self.operations if op.required]


class PromptContributionCapability:
    """Declarative prompt contribution with hint serialization."""

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
            list(contributions)
            if contributions is not None
            else list(self.COMMON_HINTS)
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
