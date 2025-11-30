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

"""File editor tool for agent to perform multi-file edits safely.

This tool provides transaction-based file editing with diff preview and
rollback capability to the agent.
"""

from typing import Any, Dict, List
from pathlib import Path

from victor.editing import FileEditor
from victor.tools.decorators import tool


@tool
async def edit_files(
    operations: List[Dict[str, Any]],
    preview: bool = False,
    auto_commit: bool = True,
    description: str = "",
    context_lines: int = 3,
) -> Dict[str, Any]:
    """
    Unified file editing with transaction support.

    Perform multiple file operations (create, modify, delete, rename) in a single
    transaction with built-in preview and rollback capability. Consolidates all
    file editing functionality into one unified interface.

    Args:
        operations: List of file operations. Each operation is a dict with:
            - type: "create", "modify", "delete", or "rename" (required)
            - path: File path (required)
            - content: File content (for create/modify, optional for create)
            - new_content: New file content (alias for content in modify)
            - new_path: New file path (required for rename)
        preview: If True, show diff preview without applying changes (default: False).
        auto_commit: If True, automatically commit changes after queuing (default: True).
        description: Optional description of this edit operation.
        context_lines: Number of context lines to show in diffs (default: 3).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - operations_queued: Number of operations queued
        - operations_applied: Number of operations applied (if auto_commit=True)
        - by_type: Breakdown of operations by type
        - message: Status message
        - preview_output: Diff preview text (if preview=True)
        - error: Error message if failed

    Examples:
        # Create and modify files
        edit_files(operations=[
            {"type": "create", "path": "foo.py", "content": "print('hello')"},
            {"type": "modify", "path": "bar.py", "content": "updated content"}
        ])

        # Delete and rename files
        edit_files(operations=[
            {"type": "delete", "path": "old.py"},
            {"type": "rename", "path": "temp.py", "new_path": "final.py"}
        ])

        # Preview changes without applying
        edit_files(operations=[...], preview=True, auto_commit=False)

        # Queue operations without committing
        edit_files(operations=[...], auto_commit=False)
    """
    # Allow callers (models) to pass operations as a JSON string; normalize to list[dict]
    if isinstance(operations, str):
        import json

        try:
            operations = json.loads(operations)
        except json.JSONDecodeError as exc:
            return {"success": False, "error": f"Invalid JSON for operations: {exc}"}

    if not operations:
        return {"success": False, "error": "No operations provided"}

    # Validate operations
    for i, op in enumerate(operations):
        if not isinstance(op, dict):
            return {
                "success": False,
                "error": f"Operation {i} must be a dictionary, got {type(op).__name__}",
            }

        op_type = op.get("type")
        if not op_type:
            return {"success": False, "error": f"Operation {i} missing required field: type"}

        if op_type not in ["create", "modify", "delete", "rename"]:
            return {
                "success": False,
                "error": f"Operation {i} has invalid type: {op_type}. Must be create, modify, delete, or rename",
            }

        if "path" not in op:
            return {"success": False, "error": f"Operation {i} missing required field: path"}

        # Validate type-specific requirements
        if op_type == "rename" and "new_path" not in op:
            return {
                "success": False,
                "error": f"Rename operation {i} missing required field: new_path",
            }

    # Initialize editor
    backup_dir = Path.home() / ".victor" / "backups"
    editor = FileEditor(backup_dir=str(backup_dir))
    transaction_id = editor.start_transaction(description)

    # Count operations by type
    by_type = {"create": 0, "modify": 0, "delete": 0, "rename": 0}

    # Queue all operations
    try:
        for op in operations:
            op_type = op["type"]
            path = op["path"]

            if op_type == "create":
                content = op.get("content", "")
                editor.add_create(path, content)
                by_type["create"] += 1

            elif op_type == "modify":
                # Support both "content" and "new_content" keys
                content = op.get("new_content") or op.get("content")
                if content is None:
                    return {
                        "success": False,
                        "error": f"Modify operation for {path} missing content or new_content",
                    }
                editor.add_modify(path, content)
                by_type["modify"] += 1

            elif op_type == "delete":
                editor.add_delete(path)
                by_type["delete"] += 1

            elif op_type == "rename":
                new_path = op["new_path"]
                editor.add_rename(path, new_path)
                by_type["rename"] += 1

    except Exception as e:
        editor.abort()
        return {"success": False, "error": f"Failed to queue operations: {str(e)}"}

    operations_queued = len(operations)

    # Handle preview mode
    if preview:
        import io
        import sys

        # Capture preview output
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        try:
            editor.preview_diff(context_lines=context_lines)
            preview_text = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        if not auto_commit:
            editor.abort()
            return {
                "success": True,
                "operations_queued": operations_queued,
                "operations_applied": 0,
                "by_type": by_type,
                "preview_output": preview_text,
                "message": f"Preview generated for {operations_queued} operations (not applied)",
            }
        else:
            # Preview but still commit
            success = editor.commit(dry_run=False)
            if success:
                return {
                    "success": True,
                    "operations_queued": operations_queued,
                    "operations_applied": operations_queued,
                    "by_type": by_type,
                    "preview_output": preview_text,
                    "message": f"Applied {operations_queued} operations successfully",
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to commit changes. Transaction rolled back.",
                }

    # Handle auto_commit
    if auto_commit:
        success = editor.commit(dry_run=False)
        if success:
            return {
                "success": True,
                "operations_queued": operations_queued,
                "operations_applied": operations_queued,
                "by_type": by_type,
                "message": f"Successfully applied {operations_queued} operations",
                "transaction_id": transaction_id,
            }
        else:
            return {"success": False, "error": "Failed to commit changes. Transaction rolled back."}
    else:
        # Queue only, don't commit
        editor.abort()  # Abort to clean up, since we're not committing
        return {
            "success": True,
            "operations_queued": operations_queued,
            "operations_applied": 0,
            "by_type": by_type,
            "message": f"Queued {operations_queued} operations (not applied, auto_commit=False)",
        }


# Keep the class for backward compatibility during transition
# This can be removed once all imports are updated
class FileEditorTool:
    """Deprecated: Use edit_files function instead."""

    def __init__(self):
        """Initialize - deprecated."""
        import warnings

        warnings.warn(
            "FileEditorTool class is deprecated. Use edit_files function instead.",
            DeprecationWarning,
            stacklevel=2,
        )
