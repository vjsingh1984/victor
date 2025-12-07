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


@tool(
    category="filesystem",
    keywords=[
        "edit",
        "modify",
        "change",
        "update",
        "replace",
        "fix",
        "refactor",
        "create",
        "delete",
        "rename",
        "file",
        "code",
    ],
    use_cases=[
        "editing existing files",
        "modifying source code",
        "fixing bugs in code",
        "creating new files",
        "deleting files",
        "renaming files",
        "refactoring code",
        "applying code changes",
    ],
    examples=[
        "edit the file to fix the bug",
        "change the function to return 42",
        "replace 'old_value' with 'new_value' in config.py",
        "create a new file called utils.py",
        "delete the old_module.py file",
        "rename temp.py to final.py",
    ],
    priority_hints=[
        "PREFERRED tool for making code changes",
        "Use 'replace' operation for surgical edits (most reliable)",
        "Use 'create' for new files, 'modify' only for complete rewrites",
        "Supports undo via /undo command",
        "Can perform multiple operations atomically",
    ],
)
async def edit_files(
    operations: List[Dict[str, Any]],
    preview: bool = False,
    auto_commit: bool = True,
    description: str = "",
    context_lines: int = 3,
) -> Dict[str, Any]:
    """PRIMARY tool for code changes with transaction support and undo.

    Operations: replace (old_str→new_str), create, modify, delete, rename.
    Use preview=True to see changes before applying. Supports /undo command.

    Example: [{"type": "replace", "path": "main.py", "old_str": "x=1", "new_str": "x=42"}]
    """
    # Allow callers (models) to pass operations as a JSON string; normalize to list[dict]
    if isinstance(operations, str):
        import json
        import re

        # Try to parse the JSON, with recovery for common issues
        try:
            operations = json.loads(operations)
        except json.JSONDecodeError as exc:
            # Try to provide helpful error message and recovery hints
            error_context = ""

            # Detect control character issues (common with embedded newlines)
            if "control character" in str(exc).lower():
                error_context = (
                    "\n\nHINT: JSON strings cannot contain raw newlines or tabs. "
                    "Use \\n for newlines and \\t for tabs within string values."
                )
                # Try to fix by escaping control characters in strings
                try:
                    fixed = re.sub(
                        r'(?<!\\)\n(?=(?:[^"]*"[^"]*")*[^"]*"[^"]*$)', r"\\n", operations
                    )
                    fixed = re.sub(r'(?<!\\)\t(?=(?:[^"]*"[^"]*")*[^"]*"[^"]*$)', r"\\t", fixed)
                    operations = json.loads(fixed)
                    # If fixed, log and continue
                    import logging

                    logging.getLogger(__name__).info("Auto-fixed JSON control characters")
                except json.JSONDecodeError:
                    pass  # Recovery failed, use original error

            if isinstance(operations, str):  # Still a string = parsing failed
                # Detect delimiter issues
                if "delimiter" in str(exc).lower():
                    error_context = "\n\nHINT: Check for missing commas between array elements or object properties."
                # Detect structure issues
                elif "Expecting" in str(exc):
                    error_context = (
                        "\n\nHINT: Check JSON structure - ensure arrays use [], objects use {}, "
                        "and strings are quoted."
                    )

                example = (
                    "\n\nCorrect format example:\n"
                    '[{"type": "replace", "path": "file.py", "old_str": "x=1", "new_str": "x=2"}]'
                )
                return {
                    "success": False,
                    "error": f"Invalid JSON for operations: {exc}{error_context}{example}",
                }

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

        if op_type not in ["create", "modify", "delete", "rename", "replace"]:
            return {
                "success": False,
                "error": f"Operation {i} has invalid type: {op_type}. Must be create, modify, delete, rename, or replace",
            }

        if "path" not in op:
            return {"success": False, "error": f"Operation {i} missing required field: path"}

        # Validate type-specific requirements
        if op_type == "rename" and "new_path" not in op:
            return {
                "success": False,
                "error": f"Rename operation {i} missing required field: new_path",
            }

    from victor.agent.change_tracker import ChangeType, get_change_tracker
    from victor.config.settings import get_project_paths

    # Initialize editor
    backup_dir = get_project_paths().backups_dir
    editor = FileEditor(backup_dir=str(backup_dir))
    transaction_id = editor.start_transaction(description)

    # Initialize change tracker for undo/redo
    tracker = get_change_tracker()
    tracker.begin_change_group("edit_files", description or f"Edit {len(operations)} files")

    # Count operations by type
    by_type = {"create": 0, "modify": 0, "delete": 0, "rename": 0}

    # Queue all operations
    try:
        for op in operations:
            op_type = op["type"]
            path = op["path"]
            file_path = Path(path).expanduser().resolve()

            if op_type == "create":
                content = op.get("content", "")
                editor.add_create(path, content)
                by_type["create"] += 1
                # Track for undo
                tracker.record_change(
                    file_path=str(file_path),
                    change_type=ChangeType.CREATE,
                    original_content=None,
                    new_content=content,
                    tool_name="edit_files",
                    tool_args={"type": "create", "path": path},
                )

            elif op_type == "modify":
                # Support both "content" and "new_content" keys
                content = op.get("new_content") or op.get("content")
                if content is None:
                    return {
                        "success": False,
                        "error": f"Modify operation for {path} missing content or new_content",
                    }
                # Read original content for undo
                original_content = None
                if file_path.exists():
                    original_content = file_path.read_text(encoding="utf-8")
                editor.add_modify(path, content)
                by_type["modify"] += 1
                # Track for undo
                tracker.record_change(
                    file_path=str(file_path),
                    change_type=ChangeType.MODIFY,
                    original_content=original_content,
                    new_content=content,
                    tool_name="edit_files",
                    tool_args={"type": "modify", "path": path},
                )

            elif op_type == "delete":
                # Read content before delete for undo
                original_content = None
                if file_path.exists():
                    original_content = file_path.read_text(encoding="utf-8")
                editor.add_delete(path)
                by_type["delete"] += 1
                # Track for undo
                tracker.record_change(
                    file_path=str(file_path),
                    change_type=ChangeType.DELETE,
                    original_content=original_content,
                    new_content=None,
                    tool_name="edit_files",
                    tool_args={"type": "delete", "path": path},
                )

            elif op_type == "rename":
                new_path = op["new_path"]
                editor.add_rename(path, new_path)
                by_type["rename"] += 1
                # Track for undo
                new_file_path = Path(new_path).expanduser().resolve()
                tracker.record_change(
                    file_path=str(new_file_path),
                    change_type=ChangeType.RENAME,
                    original_path=str(file_path),
                    tool_name="edit_files",
                    tool_args={"type": "rename", "path": path, "new_path": new_path},
                )

            elif op_type == "replace":
                # Surgical string replacement (Claude Code style)
                old_str = op.get("old_str")
                new_str = op.get("new_str")

                if old_str is None:
                    return {
                        "success": False,
                        "error": f"Replace operation for {path} missing required field: old_str",
                    }
                if new_str is None:
                    return {
                        "success": False,
                        "error": f"Replace operation for {path} missing required field: new_str",
                    }

                # File must exist for replace
                if not file_path.exists():
                    return {
                        "success": False,
                        "error": f"Replace operation failed: file {path} does not exist",
                    }

                # Read current content
                original_content = file_path.read_text(encoding="utf-8")

                # Check if old_str exists in file
                occurrences = original_content.count(old_str)
                if occurrences == 0:
                    # Build helpful error message
                    old_str_preview = old_str[:80] + "..." if len(old_str) > 80 else old_str
                    old_str_first_line = old_str.split("\n")[0][:60]

                    # Try to find similar content to help debug
                    hint = ""
                    if old_str_first_line in original_content:
                        hint = (
                            f" The first line '{old_str_first_line}' exists in file but "
                            f"subsequent lines don't match. Check line endings and indentation."
                        )
                    elif old_str.rstrip() in original_content:
                        hint = " Found match without trailing whitespace. Remove trailing newlines from old_str."
                    elif old_str.lstrip() in original_content:
                        hint = " Found match without leading whitespace. Check indentation at start of old_str."

                    return {
                        "success": False,
                        "error": (
                            f"Replace operation failed: old_str not found in {path}.{hint} "
                            f"Make sure the string matches exactly including whitespace. "
                            f"Searched for: {repr(old_str_preview)}"
                        ),
                    }
                if occurrences > 1:
                    return {
                        "success": False,
                        "error": f"Replace operation failed: old_str found {occurrences} times in {path}. "
                        f"Ambiguous match - provide more context to make the match unique.",
                    }

                # Perform replacement
                new_content = original_content.replace(old_str, new_str, 1)

                # Queue as a modify operation
                editor.add_modify(path, new_content)
                if "replace" not in by_type:
                    by_type["replace"] = 0
                by_type["replace"] += 1

                # Track for undo
                tracker.record_change(
                    file_path=str(file_path),
                    change_type=ChangeType.MODIFY,
                    original_content=original_content,
                    new_content=new_content,
                    tool_name="edit_files",
                    tool_args={"type": "replace", "path": path, "old_str": old_str[:50]},
                )

    except Exception as e:
        editor.abort()
        tracker.commit_change_group()  # Empty commit to reset state
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
            # Commit the change group for undo/redo
            tracker.commit_change_group()
            return {
                "success": True,
                "operations_queued": operations_queued,
                "operations_applied": operations_queued,
                "by_type": by_type,
                "message": f"Successfully applied {operations_queued} operations. Use /undo to revert.",
                "transaction_id": transaction_id,
            }
        else:
            # Clear change group on failure
            tracker._current_group = None
            return {"success": False, "error": "Failed to commit changes. Transaction rolled back."}
    else:
        # Queue only, don't commit
        editor.abort()  # Abort to clean up, since we're not committing
        tracker._current_group = None  # Clear uncommitted changes
        return {
            "success": True,
            "operations_queued": operations_queued,
            "operations_applied": 0,
            "by_type": by_type,
            "message": f"Queued {operations_queued} operations (not applied, auto_commit=False)",
        }
