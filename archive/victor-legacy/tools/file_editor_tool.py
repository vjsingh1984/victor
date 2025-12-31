"""File editor tool for agent to perform multi-file edits safely.

This tool wraps the FileEditor class to provide transaction-based file editing
with diff preview and rollback capability to the agent.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path

from victor.editing import FileEditor
from victor.tools.base import BaseTool, ToolParameter, ToolResult


class FileEditorTool(BaseTool):
    """Tool for safe multi-file editing with transactions."""

    def __init__(self):
        """Initialize file editor tool."""
        super().__init__()
        self.editor: Optional[FileEditor] = None
        self.current_transaction_id: Optional[str] = None

    @property
    def name(self) -> str:
        """Get tool name."""
        return "file_editor"

    @property
    def description(self) -> str:
        """Get tool description."""
        return """Multi-file editor with atomic operations and rollback.

Use this tool when you need to safely modify multiple files with:
- Transaction-based editing (all-or-nothing)
- Diff preview before applying changes
- Automatic backups
- Rollback capability

Operations:
1. start_transaction - Begin a new edit transaction
2. add_create - Queue file creation
3. add_modify - Queue file modification
4. add_delete - Queue file deletion
5. add_rename - Queue file rename
6. preview - Preview all queued changes with diffs
7. commit - Apply all changes (or dry_run)
8. rollback - Undo all changes
9. abort - Cancel transaction without applying

Example workflow:
1. start_transaction(description="Update auth module")
2. add_modify(path="auth.py", new_content="...")
3. add_create(path="auth_test.py", content="...")
4. preview() - Review changes
5. commit() - Apply changes
"""

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameters."""
        return self.convert_parameters_to_schema(
            [
                ToolParameter(
                    name="operation",
                    type="string",
                    description="Operation to perform: start_transaction, add_create, add_modify, add_delete, add_rename, preview, commit, rollback, abort, status",
                    required=True,
                ),
                ToolParameter(
                    name="description",
                    type="string",
                    description="Transaction description (for start_transaction)",
                    required=False,
                ),
                ToolParameter(
                    name="path",
                    type="string",
                    description="File path (for add_create, add_modify, add_delete, add_rename)",
                    required=False,
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="File content (for add_create)",
                    required=False,
                ),
                ToolParameter(
                    name="new_content",
                    type="string",
                    description="New file content (for add_modify)",
                    required=False,
                ),
                ToolParameter(
                    name="new_path",
                    type="string",
                    description="New file path (for add_rename)",
                    required=False,
                ),
                ToolParameter(
                    name="dry_run",
                    type="boolean",
                    description="If true, preview changes without applying (for commit)",
                    required=False,
                ),
                ToolParameter(
                    name="context_lines",
                    type="integer",
                    description="Number of context lines in diff preview (default: 3)",
                    required=False,
                ),
            ]
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute file editor operation.

        Args:
            operation: Operation to perform
            **kwargs: Operation-specific parameters

        Returns:
            Tool result with success/failure info
        """
        operation = kwargs.get("operation")

        if not operation:
            return ToolResult(
                success=False, output="", error="Missing required parameter: operation"
            )

        try:
            if operation == "start_transaction":
                return await self._start_transaction(kwargs)
            elif operation == "add_create":
                return await self._add_create(kwargs)
            elif operation == "add_modify":
                return await self._add_modify(kwargs)
            elif operation == "add_delete":
                return await self._add_delete(kwargs)
            elif operation == "add_rename":
                return await self._add_rename(kwargs)
            elif operation == "preview":
                return await self._preview(kwargs)
            elif operation == "commit":
                return await self._commit(kwargs)
            elif operation == "rollback":
                return await self._rollback(kwargs)
            elif operation == "abort":
                return await self._abort(kwargs)
            elif operation == "status":
                return await self._status(kwargs)
            else:
                return ToolResult(success=False, output="", error=f"Unknown operation: {operation}")

        except Exception as e:
            return ToolResult(success=False, output="", error=f"File editor error: {str(e)}")

    async def _start_transaction(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Start a new transaction."""
        if self.editor is not None:
            return ToolResult(
                success=False,
                output="",
                error="Transaction already in progress. Commit, rollback, or abort first.",
            )

        description = kwargs.get("description", "")
        backup_dir = Path.home() / ".victor" / "backups"

        self.editor = FileEditor(backup_dir=str(backup_dir))
        self.current_transaction_id = self.editor.start_transaction(description)

        return ToolResult(
            success=True,
            output=f"Started transaction: {self.current_transaction_id}\n{description}",
            error="",
        )

    async def _add_create(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Add file creation operation."""
        if self.editor is None:
            return ToolResult(
                success=False,
                output="",
                error="No active transaction. Call start_transaction first.",
            )

        path = kwargs.get("path")
        content = kwargs.get("content", "")

        if not path:
            return ToolResult(success=False, output="", error="Missing required parameter: path")

        self.editor.add_create(path, content)

        return ToolResult(success=True, output=f"Queued file creation: {path}", error="")

    async def _add_modify(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Add file modification operation."""
        if self.editor is None:
            return ToolResult(
                success=False,
                output="",
                error="No active transaction. Call start_transaction first.",
            )

        path = kwargs.get("path")
        new_content = kwargs.get("new_content")

        if not path:
            return ToolResult(success=False, output="", error="Missing required parameter: path")

        if new_content is None:
            return ToolResult(
                success=False, output="", error="Missing required parameter: new_content"
            )

        self.editor.add_modify(path, new_content)

        return ToolResult(success=True, output=f"Queued file modification: {path}", error="")

    async def _add_delete(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Add file deletion operation."""
        if self.editor is None:
            return ToolResult(
                success=False,
                output="",
                error="No active transaction. Call start_transaction first.",
            )

        path = kwargs.get("path")

        if not path:
            return ToolResult(success=False, output="", error="Missing required parameter: path")

        self.editor.add_delete(path)

        return ToolResult(success=True, output=f"Queued file deletion: {path}", error="")

    async def _add_rename(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Add file rename operation."""
        if self.editor is None:
            return ToolResult(
                success=False,
                output="",
                error="No active transaction. Call start_transaction first.",
            )

        path = kwargs.get("path")
        new_path = kwargs.get("new_path")

        if not path:
            return ToolResult(success=False, output="", error="Missing required parameter: path")

        if not new_path:
            return ToolResult(
                success=False, output="", error="Missing required parameter: new_path"
            )

        self.editor.add_rename(path, new_path)

        return ToolResult(success=True, output=f"Queued file rename: {path} → {new_path}", error="")

    async def _preview(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Preview all queued changes."""
        if self.editor is None:
            return ToolResult(
                success=False,
                output="",
                error="No active transaction. Call start_transaction first.",
            )

        context_lines = kwargs.get("context_lines", 3)

        # Capture preview output
        self.editor.preview_diff(context_lines=context_lines)

        summary = self.editor.get_transaction_summary()

        return ToolResult(
            success=True,
            output=f"Preview shown. Transaction has {summary['operations']} operations.",
            error="",
        )

    async def _commit(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Commit the transaction."""
        if self.editor is None:
            return ToolResult(
                success=False,
                output="",
                error="No active transaction. Call start_transaction first.",
            )

        dry_run = kwargs.get("dry_run", False)

        success = self.editor.commit(dry_run=dry_run)

        if success:
            output = (
                "Dry run complete - no changes applied"
                if dry_run
                else "Transaction committed successfully"
            )
            self.editor = None
            self.current_transaction_id = None

            return ToolResult(success=True, output=output, error="")
        else:
            return ToolResult(
                success=False,
                output="",
                error="Transaction commit failed. Changes were rolled back.",
            )

    async def _rollback(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Rollback the transaction."""
        if self.editor is None:
            return ToolResult(
                success=False,
                output="",
                error="No active transaction. Call start_transaction first.",
            )

        success = self.editor.rollback()
        self.editor = None
        self.current_transaction_id = None

        if success:
            return ToolResult(success=True, output="Transaction rolled back successfully", error="")
        else:
            return ToolResult(success=False, output="", error="Rollback failed")

    async def _abort(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Abort the transaction without applying changes."""
        if self.editor is None:
            return ToolResult(success=False, output="", error="No active transaction")

        self.editor.abort()
        self.editor = None
        self.current_transaction_id = None

        return ToolResult(success=True, output="Transaction aborted", error="")

    async def _status(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Get transaction status."""
        if self.editor is None:
            return ToolResult(success=True, output="No active transaction", error="")

        summary = self.editor.get_transaction_summary()

        output = f"""Transaction Status:
ID: {summary['id']}
Description: {summary['description']}
Total operations: {summary['operations']}
By type:
  - Create: {summary['by_type']['create']}
  - Modify: {summary['by_type']['modify']}
  - Delete: {summary['by_type']['delete']}
  - Rename: {summary['by_type']['rename']}
"""

        return ToolResult(success=True, output=output, error="")
