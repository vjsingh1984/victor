"""File editor tool for agent to perform multi-file edits safely.

This tool provides transaction-based file editing with diff preview and
rollback capability to the agent.
"""

from typing import Any, Dict, Optional
from pathlib import Path

from victor.editing import FileEditor
from victor.tools.decorators import tool

# Global editor state (one active transaction at a time)
_editor: Optional[FileEditor] = None
_current_transaction_id: Optional[str] = None


def _get_editor() -> Optional[FileEditor]:
    """Get the current editor instance."""
    return _editor


def _clear_editor() -> None:
    """Clear the current editor instance."""
    global _editor, _current_transaction_id
    _editor = None
    _current_transaction_id = None


@tool
async def file_editor_start_transaction(description: str = "") -> Dict[str, Any]:
    """
    Start a new file editing transaction.

    Begins a new transaction-based editing session where you can queue
    multiple file operations (create, modify, delete, rename) and apply
    them atomically with preview and rollback capability.

    Args:
        description: Description of this transaction (optional).

    Returns:
        Dictionary containing:
        - success: Whether transaction started
        - transaction_id: ID of the started transaction
        - message: Status message
        - error: Error message if failed
    """
    global _editor, _current_transaction_id

    if _editor is not None:
        return {
            "success": False,
            "error": "Transaction already in progress. Commit, rollback, or abort first."
        }

    backup_dir = Path.home() / ".victor" / "backups"
    _editor = FileEditor(backup_dir=str(backup_dir))
    _current_transaction_id = _editor.start_transaction(description)

    return {
        "success": True,
        "transaction_id": _current_transaction_id,
        "message": f"Started transaction: {_current_transaction_id}",
        "description": description
    }


@tool
async def file_editor_add_create(path: str, content: str = "") -> Dict[str, Any]:
    """
    Queue a file creation operation.

    Adds a file creation to the current transaction. The file will be
    created when you commit the transaction.

    Args:
        path: Path where the file will be created.
        content: Content for the new file (default: empty string).

    Returns:
        Dictionary containing:
        - success: Whether operation was queued
        - path: Path of file to be created
        - message: Status message
        - error: Error message if failed
    """
    if _editor is None:
        return {
            "success": False,
            "error": "No active transaction. Call file_editor_start_transaction first."
        }

    if not path:
        return {
            "success": False,
            "error": "Missing required parameter: path"
        }

    _editor.add_create(path, content)

    return {
        "success": True,
        "path": path,
        "message": f"Queued file creation: {path}"
    }


@tool
async def file_editor_add_modify(path: str, new_content: str) -> Dict[str, Any]:
    """
    Queue a file modification operation.

    Adds a file modification to the current transaction. The file will be
    modified when you commit the transaction.

    Args:
        path: Path of the file to modify.
        new_content: New content for the file.

    Returns:
        Dictionary containing:
        - success: Whether operation was queued
        - path: Path of file to be modified
        - message: Status message
        - error: Error message if failed
    """
    if _editor is None:
        return {
            "success": False,
            "error": "No active transaction. Call file_editor_start_transaction first."
        }

    if not path:
        return {
            "success": False,
            "error": "Missing required parameter: path"
        }

    if new_content is None:
        return {
            "success": False,
            "error": "Missing required parameter: new_content"
        }

    _editor.add_modify(path, new_content)

    return {
        "success": True,
        "path": path,
        "message": f"Queued file modification: {path}"
    }


@tool
async def file_editor_add_delete(path: str) -> Dict[str, Any]:
    """
    Queue a file deletion operation.

    Adds a file deletion to the current transaction. The file will be
    deleted when you commit the transaction.

    Args:
        path: Path of the file to delete.

    Returns:
        Dictionary containing:
        - success: Whether operation was queued
        - path: Path of file to be deleted
        - message: Status message
        - error: Error message if failed
    """
    if _editor is None:
        return {
            "success": False,
            "error": "No active transaction. Call file_editor_start_transaction first."
        }

    if not path:
        return {
            "success": False,
            "error": "Missing required parameter: path"
        }

    _editor.add_delete(path)

    return {
        "success": True,
        "path": path,
        "message": f"Queued file deletion: {path}"
    }


@tool
async def file_editor_add_rename(path: str, new_path: str) -> Dict[str, Any]:
    """
    Queue a file rename operation.

    Adds a file rename to the current transaction. The file will be
    renamed when you commit the transaction.

    Args:
        path: Current path of the file.
        new_path: New path for the file.

    Returns:
        Dictionary containing:
        - success: Whether operation was queued
        - old_path: Original file path
        - new_path: New file path
        - message: Status message
        - error: Error message if failed
    """
    if _editor is None:
        return {
            "success": False,
            "error": "No active transaction. Call file_editor_start_transaction first."
        }

    if not path:
        return {
            "success": False,
            "error": "Missing required parameter: path"
        }

    if not new_path:
        return {
            "success": False,
            "error": "Missing required parameter: new_path"
        }

    _editor.add_rename(path, new_path)

    return {
        "success": True,
        "old_path": path,
        "new_path": new_path,
        "message": f"Queued file rename: {path} → {new_path}"
    }


@tool
async def file_editor_preview(context_lines: int = 3) -> Dict[str, Any]:
    """
    Preview all queued changes with diffs.

    Shows a preview of all file operations queued in the current transaction
    with unified diff format for modifications.

    Args:
        context_lines: Number of context lines to show in diffs (default: 3).

    Returns:
        Dictionary containing:
        - success: Whether preview was generated
        - operations_count: Number of queued operations
        - message: Status message
        - error: Error message if failed
    """
    if _editor is None:
        return {
            "success": False,
            "error": "No active transaction. Call file_editor_start_transaction first."
        }

    # Capture preview output
    _editor.preview_diff(context_lines=context_lines)

    summary = _editor.get_transaction_summary()

    return {
        "success": True,
        "operations_count": summary['operations'],
        "message": f"Preview shown. Transaction has {summary['operations']} operations."
    }


@tool
async def file_editor_commit(dry_run: bool = False) -> Dict[str, Any]:
    """
    Commit the transaction and apply all changes.

    Applies all queued file operations. If dry_run is True, previews
    changes without applying them.

    Args:
        dry_run: If True, preview changes without applying (default: False).

    Returns:
        Dictionary containing:
        - success: Whether commit succeeded
        - message: Status message
        - error: Error message if failed
    """
    global _editor, _current_transaction_id

    if _editor is None:
        return {
            "success": False,
            "error": "No active transaction. Call file_editor_start_transaction first."
        }

    success = _editor.commit(dry_run=dry_run)

    if success:
        message = "Dry run complete - no changes applied" if dry_run else "Transaction committed successfully"
        _editor = None
        _current_transaction_id = None

        return {
            "success": True,
            "message": message
        }
    else:
        return {
            "success": False,
            "error": "Transaction commit failed. Changes were rolled back."
        }


@tool
async def file_editor_rollback() -> Dict[str, Any]:
    """
    Rollback the transaction and undo all changes.

    Reverts all file operations that were applied during commit.
    Only works if the transaction was already committed.

    Returns:
        Dictionary containing:
        - success: Whether rollback succeeded
        - message: Status message
        - error: Error message if failed
    """
    global _editor, _current_transaction_id

    if _editor is None:
        return {
            "success": False,
            "error": "No active transaction. Call file_editor_start_transaction first."
        }

    success = _editor.rollback()
    _editor = None
    _current_transaction_id = None

    if success:
        return {
            "success": True,
            "message": "Transaction rolled back successfully"
        }
    else:
        return {
            "success": False,
            "error": "Rollback failed"
        }


@tool
async def file_editor_abort() -> Dict[str, Any]:
    """
    Abort the transaction without applying changes.

    Cancels the current transaction and discards all queued operations
    without applying them.

    Returns:
        Dictionary containing:
        - success: Whether abort succeeded
        - message: Status message
        - error: Error message if failed
    """
    global _editor, _current_transaction_id

    if _editor is None:
        return {
            "success": False,
            "error": "No active transaction"
        }

    _editor.abort()
    _editor = None
    _current_transaction_id = None

    return {
        "success": True,
        "message": "Transaction aborted"
    }


@tool
async def file_editor_status() -> Dict[str, Any]:
    """
    Get the current transaction status.

    Returns information about the active transaction including ID,
    description, and counts of queued operations by type.

    Returns:
        Dictionary containing:
        - success: Always True
        - transaction_id: ID of active transaction (if any)
        - description: Transaction description
        - operations: Total number of queued operations
        - by_type: Breakdown of operations by type (create, modify, delete, rename)
        - message: Status message
    """
    if _editor is None:
        return {
            "success": True,
            "message": "No active transaction"
        }

    summary = _editor.get_transaction_summary()

    return {
        "success": True,
        "transaction_id": summary['id'],
        "description": summary['description'],
        "operations": summary['operations'],
        "by_type": summary['by_type'],
        "message": f"Active transaction with {summary['operations']} operations"
    }


# Keep the class for backward compatibility during transition
# This can be removed once all imports are updated
class FileEditorTool:
    """Deprecated: Use individual file_editor_* functions instead."""

    def __init__(self):
        """Initialize - deprecated."""
        import warnings
        warnings.warn(
            "FileEditorTool class is deprecated. Use file_editor_* functions instead.",
            DeprecationWarning,
            stacklevel=2
        )
