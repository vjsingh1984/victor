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

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path


# =============================================================================
# EDIT OPERATION NORMALIZATION
# =============================================================================
# LLMs (especially DeepSeek, Ollama models) often use alternate parameter names.
# This normalization layer bridges provider-specific formats to our canonical format.
# =============================================================================

# Maps alternate parameter names to canonical names for edit operations
EDIT_PARAMETER_ALIASES: Dict[str, str] = {
    # Path aliases
    "file_path": "path",
    "file": "path",
    "filepath": "path",
    "filename": "path",
    # Content aliases for create/modify
    "new_content": "content",
    "text": "content",
    "data": "content",
    # Replace operation aliases
    "old": "old_str",
    "old_string": "old_str",
    "find": "old_str",
    "search": "old_str",
    "new": "new_str",
    "new_string": "new_str",
    "replace": "new_str",
    "replacement": "new_str",
    # Rename operation aliases
    "new_name": "new_path",
    "to": "new_path",
    "destination": "new_path",
    "dest": "new_path",
}

# Maps operation type aliases to canonical types
EDIT_TYPE_ALIASES: Dict[str, str] = {
    "write": "create",
    "add": "create",
    "new": "create",
    "update": "modify",
    "change": "modify",
    "edit": "modify",
    "remove": "delete",
    "rm": "delete",
    "move": "rename",
    "mv": "rename",
    "find_replace": "replace",
    "substitute": "replace",
    "sub": "replace",
}

# Keys that indicate an implicit operation type (when 'type' is missing)
TYPE_INFERENCE_KEYS: Dict[str, str] = {
    "old_str": "replace",
    "old": "replace",
    "find": "replace",
    "search": "replace",
    "new_path": "rename",
    "new_name": "rename",
    "destination": "rename",
}


def normalize_edit_operation(op: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single edit operation for provider-agnostic handling.

    Handles:
    1. Parameter name aliases (e.g., 'file_path' -> 'path')
    2. Operation type aliases (e.g., 'write' -> 'create')
    3. Type inference from keys (e.g., presence of 'old_str' implies 'replace')

    Args:
        op: Raw operation dictionary from LLM

    Returns:
        Normalized operation dictionary with canonical parameter names
    """
    logger = logging.getLogger(__name__)
    normalized: Dict[str, Any] = {}
    applied_aliases: List[str] = []

    # Step 1: Normalize all parameter names
    for key, value in op.items():
        if key in EDIT_PARAMETER_ALIASES:
            canonical = EDIT_PARAMETER_ALIASES[key]
            # Only apply if canonical doesn't already have a value
            if canonical not in normalized or normalized[canonical] is None:
                normalized[canonical] = value
                applied_aliases.append(f"{key}->{canonical}")
        else:
            # Keep original key
            normalized[key] = value

    # Step 2: Normalize operation type
    if "type" in normalized:
        op_type = normalized["type"]
        if isinstance(op_type, str) and op_type.lower() in EDIT_TYPE_ALIASES:
            old_type = normalized["type"]
            normalized["type"] = EDIT_TYPE_ALIASES[op_type.lower()]
            applied_aliases.append(f"type:{old_type}->{normalized['type']}")

    # Step 3: Infer type from keys if not specified
    if "type" not in normalized or not normalized.get("type"):
        for key, inferred_type in TYPE_INFERENCE_KEYS.items():
            if key in op:  # Check original op keys too
                normalized["type"] = inferred_type
                applied_aliases.append(f"inferred_type:{inferred_type}")
                break
        # Default to create if has content but no type
        if "type" not in normalized and "content" in normalized:
            normalized["type"] = "create"
            applied_aliases.append("inferred_type:create")

    if applied_aliases:
        logger.debug(f"Normalized edit operation: {applied_aliases}")

    return normalized


def normalize_edit_operations(ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a list of edit operations.

    This function should be called after JSON parsing but before validation
    to handle provider-specific parameter variations gracefully.

    Args:
        ops: List of raw operation dictionaries

    Returns:
        List of normalized operation dictionaries (non-dicts passed through for validation)
    """
    # Only normalize actual dicts; pass through non-dicts for validation to catch
    return [normalize_edit_operation(op) if isinstance(op, dict) else op for op in ops]


from victor.coding.editing import FileEditor
from victor.tools.base import AccessMode, DangerLevel, Priority
from victor.tools.decorators import tool
from victor.tools.filesystem import enforce_sandbox_path


@tool(
    category="filesystem",
    priority=Priority.CRITICAL,  # Always available for selection
    access_mode=AccessMode.WRITE,  # Creates/modifies/deletes files
    danger_level=DangerLevel.LOW,  # Changes are undoable via transaction system
    # Registry-driven metadata for tool selection and loop detection
    progress_params=["ops"],  # Different operations indicate progress, not loops
    stages=["execution"],  # Conversation stages where relevant
    task_types=["edit", "action"],  # Task types for classification-aware selection
    execution_category="write",  # Cannot run in parallel with conflicting ops
    keywords=["edit", "modify", "replace", "create", "delete", "rename", "file", "text"],
    # Examples help LLMs understand correct parameter format
    examples=[
        'edit(ops=[{"type": "replace", "path": "config.py", "old_str": "DEBUG = True", "new_str": "DEBUG = False"}])',
        'edit(ops=[{"type": "create", "path": "README.md", "content": "# Project\\nDescription here"}])',
        'edit(ops=[{"type": "delete", "path": "temp_file.txt"}])',
        'edit(ops=[{"type": "rename", "path": "old.py", "new_path": "new.py"}])',
    ],
    priority_hints=[
        "The 'ops' parameter is REQUIRED - it must be a list of operation dictionaries",
        "Each operation needs 'type' (replace/create/delete/rename) and 'path'",
        "For replace: also include 'old_str' and 'new_str'",
    ],
)
async def edit(
    ops: Optional[List[Dict[str, Any]]] = None,
    preview: bool = False,
    commit: bool = True,
    desc: str = "",
    ctx: int = 3,
    validate: bool = True,
    strict_validation: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Edit files atomically with undo. REQUIRED: 'ops' parameter with operation list.

    IMPORTANT: You MUST provide the 'ops' parameter. Example:
        edit(ops=[{"type": "replace", "path": "file.py", "old_str": "old", "new_str": "new"}])

    Performs literal string replacement. Does NOT understand code structure.
    WARNING: May cause false positives in code (e.g., 'foo' matches 'foobar').
    For Python symbol renaming, use rename() from refactor_tool instead.

    Operation types:
        - replace: Find and replace text. Requires: path, old_str, new_str
        - create: Create new file. Requires: path, content
        - modify: Overwrite file content. Requires: path, content
        - delete: Remove file. Requires: path
        - rename: Rename/move file. Requires: path, new_path

    Args:
        ops: REQUIRED! List of operations. Each op must have 'type' and 'path'.
             Example: [{"type": "replace", "path": "f.py", "old_str": "x", "new_str": "y"}]
        preview: Show diff without applying (default: False)
        commit: Auto-apply changes (default: True)
        desc: Change description for tracking
        ctx: Diff context lines (default: 3)
        validate: Run syntax validation on code files (default: True)
        strict_validation: Block writes if validation fails (default: False)
        **kwargs: Captures malformed calls where models pass edit parameters directly
                  (e.g., file="x.py", old_str="a", new_str="b") instead of using ops.

    When to use:
        - Config files (JSON, YAML, TOML, etc.)
        - Documentation (Markdown, text files)
        - Any text-based file changes

    Note:
        In EXPLORE/PLAN modes, edits are restricted to .victor/sandbox/.
        Use /mode build to enable unrestricted file edits.
    """
    # =============================================================================
    # MALFORMED CALL RECOVERY
    # =============================================================================
    # Some models (especially Ollama/local models) call edit() with parameters
    # directly instead of wrapping them in ops=[{...}]. For example:
    #   edit(file="x.py", old_str="a", new_str="b")  # WRONG
    # Instead of:
    #   edit(ops=[{"path": "x.py", "old_str": "a", "new_str": "b"}])  # CORRECT
    #
    # We detect this pattern and auto-convert to the correct format.
    # =============================================================================
    if ops is None and kwargs:
        import logging

        logger = logging.getLogger(__name__)

        # Check if kwargs contains edit operation parameters
        path_keys = {"file", "file_path", "filepath", "filename", "path"}
        content_keys = {"content", "new_content", "text", "data"}
        replace_keys = {"old_str", "new_str", "old", "new", "old_string", "new_string"}

        has_path = any(k in kwargs for k in path_keys)
        has_content = any(k in kwargs for k in content_keys)
        has_replace = any(k in kwargs for k in replace_keys)

        if has_path or has_content or has_replace:
            logger.info(
                f"Auto-converting malformed edit() call to ops format. "
                f"Received kwargs: {list(kwargs.keys())}"
            )

            # Build operation from kwargs
            op: Dict[str, Any] = {}

            # Extract path
            for key in path_keys:
                if key in kwargs:
                    op["path"] = kwargs.pop(key)
                    break

            # Determine operation type and extract relevant fields
            if any(k in kwargs for k in {"old_str", "old", "old_string", "find", "search"}):
                op["type"] = "replace"
                # Extract old_str
                for key in ["old_str", "old", "old_string", "find", "search"]:
                    if key in kwargs:
                        op["old_str"] = kwargs.pop(key)
                        break
                # Extract new_str
                for key in ["new_str", "new", "new_string", "replace", "replacement"]:
                    if key in kwargs:
                        op["new_str"] = kwargs.pop(key)
                        break
            elif any(k in kwargs for k in content_keys):
                # Check if file exists to determine create vs modify
                path = op.get("path", "")
                if path and Path(path).expanduser().resolve().exists():
                    op["type"] = "modify"
                else:
                    op["type"] = "create"
                # Extract content
                for key in content_keys:
                    if key in kwargs:
                        op["content"] = kwargs.pop(key)
                        break
            elif "new_path" in kwargs or "new_name" in kwargs or "destination" in kwargs:
                op["type"] = "rename"
                for key in ["new_path", "new_name", "destination", "dest", "to"]:
                    if key in kwargs:
                        op["new_path"] = kwargs.pop(key)
                        break
            elif "path" in op and not has_content and not has_replace:
                # Just a path with no content/replace = delete
                op["type"] = kwargs.pop("type", "delete")
            else:
                # Default to modify if we have content
                op["type"] = kwargs.pop("type", "modify")

            # Set ops to our constructed operation
            if "path" in op:
                ops = [op]
                logger.info(f"Converted to ops: {ops}")

    # Validate required 'ops' parameter with helpful error message
    if ops is None:
        return {
            "error": "Missing required parameter: 'ops'",
            "hint": "The 'ops' parameter must be a list of edit operations. Example:\n"
            '  ops=[{"type": "replace", "path": "file.py", "old_str": "foo", "new_str": "bar"}]\n'
            '  ops=[{"type": "create", "path": "new_file.txt", "content": "Hello"}]\n'
            '  ops=[{"type": "delete", "path": "old_file.txt"}]',
            "success": False,
        }

    # Allow callers (models) to pass ops as a JSON string; normalize to list[dict[str, Any]]
    if isinstance(ops, str):
        import json
        import logging

        logger = logging.getLogger(__name__)

        def _fix_json_control_chars(json_str: str) -> str:
            """Fix raw control characters inside JSON strings.

            Models sometimes embed literal newlines/tabs inside JSON strings instead
            of using \\n and \\t escape sequences. This function identifies string
            regions and escapes control characters within them.
            """
            result = []
            in_string = False
            escape_next = False
            i = 0

            while i < len(json_str):
                char = json_str[i]

                if escape_next:
                    # Previous char was \, so this char is escaped
                    result.append(char)
                    escape_next = False
                elif char == "\\":
                    result.append(char)
                    escape_next = True
                elif char == '"' and not escape_next:
                    # Toggle string state
                    in_string = not in_string
                    result.append(char)
                elif in_string:
                    # Inside a string - escape control characters
                    if char == "\n":
                        result.append("\\n")
                    elif char == "\t":
                        result.append("\\t")
                    elif char == "\r":
                        result.append("\\r")
                    elif ord(char) < 32:  # Other control characters
                        result.append(f"\\u{ord(char):04x}")
                    else:
                        result.append(char)
                else:
                    # Outside string - newlines/whitespace are OK in JSON
                    result.append(char)

                i += 1

            return "".join(result)

        # Try to parse the JSON, with recovery for common issues
        try:
            ops = json.loads(ops)
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
                    fixed = _fix_json_control_chars(ops)
                    ops = json.loads(fixed)
                    # If fixed, log and continue
                    logger.info("Auto-fixed JSON control characters in edit ops")
                except json.JSONDecodeError as fix_exc:
                    logger.debug(f"JSON fix attempt failed: {fix_exc}")
                    pass  # Recovery failed, use original error

            if isinstance(ops, str):  # Still a string = parsing failed
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

    if not ops:
        return {"success": False, "error": "No operations provided"}

    # Normalize operations to handle provider-specific parameter variations
    # This bridges LLM-specific formats (e.g., 'file_path' vs 'path') to canonical format
    ops = normalize_edit_operations(ops)

    # Validate operations
    for i, op in enumerate(ops):
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
    transaction_id = editor.start_transaction(desc)

    # Initialize change tracker for undo/redo
    tracker = get_change_tracker()
    tracker.begin_change_group("edit_files", desc or f"Edit {len(ops)} files")

    # Count operations by type
    by_type = {"create": 0, "modify": 0, "delete": 0, "rename": 0}

    # Queue all operations
    try:
        for op in ops:
            op_type = op["type"]
            path = op["path"]
            file_path = Path(path).expanduser().resolve()

            # Enforce sandbox restrictions in EXPLORE/PLAN modes
            # (skip for read-only operations if we ever add them)
            if op_type in ("create", "modify", "delete", "rename", "replace"):
                enforce_sandbox_path(file_path)

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

    operations_queued = len(ops)

    # =============================================================================
    # CODE VALIDATION
    # =============================================================================
    # Validate code before committing to prevent writing syntactically invalid code.
    # This catches errors early before they're written to disk.
    # =============================================================================
    validation_errors = []
    validation_warnings = []

    if validate:
        try:
            from victor.core.language_capabilities.hooks import CodeGroundingHook

            hook = CodeGroundingHook.instance()

            for op in ops:
                op_type = op["type"]
                path = op["path"]
                file_path = Path(path).expanduser().resolve()

                # Only validate operations that write code
                if op_type in ("create", "modify"):
                    content = op.get("new_content") or op.get("content") or ""
                elif op_type == "replace":
                    # For replace, we need to compute the final content
                    old_str = op.get("old_str", "")
                    new_str = op.get("new_str", "")
                    if file_path.exists():
                        original = file_path.read_text(encoding="utf-8")
                        content = original.replace(old_str, new_str, 1)
                    else:
                        continue  # Skip if file doesn't exist (will fail later anyway)
                else:
                    continue  # Skip non-write operations

                # Check if we can validate this file type
                if not hook.can_validate(file_path):
                    continue

                # Validate the content
                should_proceed, result = hook.validate_before_write_sync(
                    content, file_path, strict=strict_validation
                )

                if not result.is_valid:
                    for issue in result.errors:
                        validation_errors.append(
                            {
                                "path": path,
                                "line": issue.line,
                                "column": issue.column,
                                "message": issue.message,
                                "severity": "error",
                            }
                        )

                for issue in result.warnings:
                    validation_warnings.append(
                        {
                            "path": path,
                            "line": issue.line,
                            "message": issue.message,
                            "severity": "warning",
                        }
                    )

        except ImportError:
            # Language capabilities not available
            pass
        except Exception as e:
            # Don't fail the edit due to validation errors
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(f"Code validation skipped due to error: {e}")

    # Block operation if strict validation is enabled and there are errors
    if strict_validation and validation_errors:
        editor.abort()
        tracker._current_group = None
        return {
            "success": False,
            "error": f"Validation failed: {len(validation_errors)} syntax errors detected",
            "validation_errors": validation_errors,
            "hint": "Fix the syntax errors or use strict_validation=False to proceed anyway",
        }

    # If there are validation errors, report them but still allow the operation
    # (validation is advisory, not blocking, unless strict mode is used)
    if validation_errors:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Code validation found {len(validation_errors)} errors in pending edits")

    # Handle preview mode
    if preview:
        import io
        import sys

        # Capture preview output
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        try:
            editor.preview_diff(context_lines=ctx)
            preview_text = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        if not commit:
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

    # Handle commit
    if commit:
        success = editor.commit(dry_run=False)
        if success:
            # Commit the change group for undo/redo
            tracker.commit_change_group()
            result = {
                "success": True,
                "operations_queued": operations_queued,
                "operations_applied": operations_queued,
                "by_type": by_type,
                "message": f"Successfully applied {operations_queued} operations. Use /undo to revert.",
                "transaction_id": transaction_id,
            }
            # Include validation info if there were warnings/errors
            if validation_errors:
                result["validation_errors"] = validation_errors
                result["message"] += f" Note: {len(validation_errors)} syntax issues detected."
            if validation_warnings:
                result["validation_warnings"] = validation_warnings
            return result
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
            "message": f"Queued {operations_queued} operations (not applied, commit=False)",
        }
