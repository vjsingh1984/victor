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
from typing import Any, Dict, List, Optional, Set
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


def _normalize_edit_line(line: str) -> str:
    """Collapse a line's internal whitespace and strip ends (CRLF-tolerant)."""
    return " ".join(line.split())


def _find_unique_fuzzy_span(content: str, old_str: str) -> Optional[str]:
    """Locate ``old_str`` in ``content`` tolerating per-line whitespace differences.

    Matches whole-line spans only (the common "indentation / line-ending drifted"
    case) and returns the *exact* substring from ``content`` so the replacement
    stays byte-precise. Returns ``None`` unless exactly one span matches — never
    guesses when the match is absent or ambiguous.
    """
    old_lines = old_str.split("\n")
    n = len(old_lines)
    if n == 0:
        return None
    target = [_normalize_edit_line(line) for line in old_lines]
    content_lines = content.split("\n")
    spans: List[str] = []
    for i in range(0, len(content_lines) - n + 1):
        window = content_lines[i : i + n]
        if [_normalize_edit_line(line) for line in window] == target:
            # Re-joining a contiguous slice on the same separator reproduces the
            # original substring exactly, preserving CRLF / trailing whitespace.
            spans.append("\n".join(window))
        if len(spans) > 1:
            return None  # ambiguous — bail early
    return spans[0] if len(spans) == 1 else None


def _replace_not_found_detail(current_content: str, old_str: str, path: str) -> str:
    """Build a rich, actionable 'old_str not found' message.

    Surfaces the actual surrounding file content when the first line matches but
    the block drifted — the exact feedback that stops a model re-guessing
    ``old_str`` from memory.
    """
    old_str_preview = old_str[:80] + "..." if len(old_str) > 80 else old_str
    old_str_first_line = old_str.split("\n")[0][:60]

    hint = ""
    context_str = ""
    suggestion = ""

    if old_str_first_line in current_content:
        hint = (
            f" The first line '{old_str_first_line}' exists in file but "
            f"subsequent lines don't match. Check line endings and indentation."
        )
        file_lines = current_content.splitlines()
        for i, line in enumerate(file_lines):
            if old_str_first_line in line:
                start = max(0, i - 3)
                end = min(len(file_lines), i + 8)
                numbered = [f"{start + j + 1}: {file_lines[start + j]}" for j in range(end - start)]
                context_str = "\n\nActual file content around match:\n" + "\n".join(numbered)
                break
    elif old_str.rstrip() in current_content:
        hint = " Found match without trailing whitespace. Remove trailing newlines from old_str."
    elif old_str.lstrip() in current_content:
        hint = " Found match without leading whitespace. Check indentation at start of old_str."
    else:
        file_lines = current_content.splitlines()
        old_str_words = old_str_first_line.split()[:3]
        for line in file_lines:
            line_words = line.split()
            if len(set(old_str_words) & set(line_words)) >= 2:
                suggestion = f"\n\nDid you mean: '{line.strip()}'?"
                break

    return (
        f"Replace operation failed: old_str not found in {path}.{hint} "
        f"Searched for: {repr(old_str_preview)}{suggestion}{context_str}"
        + (
            "\n\nTo fix: Copy the EXACT text from the file content above "
            "as your old_str. Do NOT type it from memory."
            if context_str
            else ""
        )
    )


def _resolve_replace(current_content: str, old_str: str, new_str: str, path: str) -> Dict[str, Any]:
    """Resolve a ``replace`` op to concrete new content.

    Tries an exact unique match first, then a single unambiguous whitespace-
    tolerant span. Returns a dict with ``ok`` plus either ``new_content`` (and
    whether a ``fuzzy`` match was used) or a human-readable ``reason``.
    """
    occurrences = current_content.count(old_str)
    if occurrences == 1:
        return {
            "ok": True,
            "new_content": current_content.replace(old_str, new_str, 1),
            "fuzzy": False,
        }
    if occurrences > 1:
        return {
            "ok": False,
            "reason": (
                f"Replace operation failed: old_str found {occurrences} times in "
                f"{path}. Ambiguous match - provide more context to make the match unique."
            ),
        }
    # Exact match missing — attempt a single whitespace-tolerant span.
    span = _find_unique_fuzzy_span(current_content, old_str)
    if span is not None and span != new_str:
        return {
            "ok": True,
            "new_content": current_content.replace(span, new_str, 1),
            "fuzzy": True,
        }
    return {"ok": False, "reason": _replace_not_found_detail(current_content, old_str, path)}


from victor.tools.base import AccessMode, DangerLevel, Priority

# =============================================================================
# EDITOR PROTOCOL + GRACEFUL FALLBACK
# =============================================================================
# Framework depends on EditorProtocol (abstraction), not concrete implementations
# This follows the Dependency Inversion Principle (DIP)
#
# The file editor tool requires the victor-coding package for transaction-based
# file editing with undo/redo support. This is an advanced feature that goes
# beyond simple diff-based editing.
#
# Fallback:
# - victor-coding (external vertical) - transaction-based editor with backups
# - Error message (no editor available)
# =============================================================================


def _create_file_editor(backup_dir: str):
    """Create a FileEditor instance via capability registry.

    The registry may return:
    - A class (FileEditor) → instantiate with backup_dir
    - A _LazyCapabilityProxy → resolves to class or instance
    - An already-instantiated FileEditor → use directly
    """
    from victor.core.capability_registry import CapabilityRegistry
    from victor.framework.vertical_protocols import EditorProtocol

    registry = CapabilityRegistry.get_instance()
    provider = registry.get(EditorProtocol)
    if provider is not None and registry.is_enhanced(EditorProtocol):
        # If it's a class, instantiate it
        if isinstance(provider, type):
            return provider(backup_dir=backup_dir)
        # Already an instance — reset any stale transaction state
        # FileEditor uses current_transaction (Optional[EditTransaction]), not _in_transaction
        if hasattr(provider, "current_transaction"):
            provider.current_transaction = None
        return provider
    return None


def _is_file_editor_available() -> bool:
    """Check if file editor is available (via registry or direct import)."""
    from victor.core.capability_registry import CapabilityRegistry
    from victor.framework.vertical_protocols import EditorProtocol

    return CapabilityRegistry.get_instance().is_enhanced(EditorProtocol)


from victor.tools.decorators import tool
from victor.tools.filesystem import enforce_sandbox_path


@tool(
    category="filesystem",
    priority=Priority.CRITICAL,  # Always available for selection
    access_mode=AccessMode.WRITE,  # Creates/modifies/deletes files
    danger_level=DangerLevel.LOW,  # Changes are undoable via transaction system
    # Registry-driven metadata for tool selection and loop detection
    signature_params=["ops"],  # Different operations indicate progress, not loops
    stages=["execution"],  # Conversation stages where relevant
    task_types=["edit", "action"],  # Task types for classification-aware selection
    execution_category="write",  # Cannot run in parallel with conflicting ops
    keywords=[
        "edit",
        "modify",
        "replace",
        "create",
        "delete",
        "rename",
        "file",
        "text",
    ],
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
             Prefer passing a structured list/object. JSON strings are accepted
             as a compatibility fallback for weaker tool-calling models.
        preview: Show diff without applying (default: False)
        commit: Auto-apply changes (default: True)
        desc: Change description for tracking
        ctx: Diff context lines (default: 3)

    When to use:
        - Config files (JSON, YAML, TOML, etc.)
        - Documentation (Markdown, text files)
        - Any text-based file changes

    Note:
        In EXPLORE/PLAN modes, edits are restricted to .victor/sandbox/.
        Use /mode build to enable unrestricted file edits.
    """
    # Check if FileEditor is available from capability registry
    from victor.core.capability_registry import CapabilityRegistry
    from victor.framework.vertical_protocols import EditorProtocol

    if not CapabilityRegistry.get_instance().is_enhanced(EditorProtocol):
        return {
            "success": False,
            "error": "File editing requires the victor-coding package to be installed. "
            "Install it with: pip install victor-coding",
            "ops": ops,
        }

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

    # Allow callers (models) to pass ops as a JSON string; normalize to list[dict]
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
            # Detect control character issues (common with embedded newlines)
            if "control character" in str(exc).lower():
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
                # Build a targeted correction prompt showing the error location
                error_pos = getattr(exc, "pos", 0) or 0
                snippet_start = max(0, error_pos - 40)
                snippet_end = min(len(ops), error_pos + 40)
                error_snippet = ops[snippet_start:snippet_end]
                pointer = " " * min(40, error_pos - snippet_start) + "^"

                correction_prompt = (
                    f"Your edit JSON has a syntax error at position {error_pos}:\n"
                    f"  ...{error_snippet}...\n"
                    f"  {pointer} {exc.msg if hasattr(exc, 'msg') else str(exc)}\n\n"
                    f"Please call edit() again with corrected JSON. Common fixes:\n"
                    f"- Escape newlines in strings: use \\n not actual newlines\n"
                    f'- Escape quotes in strings: use \\" not bare quotes\n'
                    f"- Ensure commas between array elements and object properties\n"
                    f"- Ensure old_str matches the file content EXACTLY\n\n"
                    f"Correct format:\n"
                    f'edit(ops=[{{"type": "replace", "path": "file.py", '
                    f'"old_str": "exact match", "new_str": "replacement"}}])'
                )

                return {
                    "success": False,
                    "error": correction_prompt,
                    "retryable": True,
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
            return {
                "success": False,
                "error": f"Operation {i} missing required field: type",
            }

        if op_type not in ["create", "modify", "delete", "rename", "replace"]:
            return {
                "success": False,
                "error": f"Operation {i} has invalid type: {op_type}. Must be create, modify, delete, rename, or replace",
            }

        if "path" not in op:
            return {
                "success": False,
                "error": f"Operation {i} missing required field: path",
            }

        # Validate type-specific requirements
        if op_type == "rename" and "new_path" not in op:
            return {
                "success": False,
                "error": f"Rename operation {i} missing required field: new_path",
            }

    # ------------------------------------------------------------------
    # Per-file validation pre-pass (isolation + structured reporting).
    #
    # Group ops by target file and validate each group independently so a
    # failure in one file (e.g. a stale old_str) never discards a valid edit to
    # a *different* file. Only fully-valid file groups are queued/committed; the
    # rest are reported back so the model can retry just those. Same-file ops
    # stay all-or-nothing (a later op may depend on an earlier one).
    # ------------------------------------------------------------------
    from collections import OrderedDict as _OrderedDict

    def _op_file_key(_op: Dict[str, Any]) -> str:
        try:
            return str(Path(_op["path"]).expanduser().resolve())
        except Exception:
            return str(_op.get("path"))

    _groups: Dict[str, List[tuple]] = _OrderedDict()
    for _idx, _op in enumerate(ops):
        _groups.setdefault(_op_file_key(_op), []).append((_idx, _op))

    failed_files: List[Dict[str, Any]] = []
    _applicable_keys: Set[str] = set()
    for _key, _group in _groups.items():
        _working: Optional[str] = None  # lazy per-file working copy
        _group_ok = True
        for _idx, _op in _group:
            _otype = _op.get("type")
            _path = _op.get("path")
            _fp = Path(_path).expanduser().resolve()
            if _working is None:
                _working = _fp.read_text(encoding="utf-8") if _fp.exists() else ""
            if _otype == "replace":
                _old = _op.get("old_str")
                _new = _op.get("new_str")
                if _old is None:
                    _reason = f"Replace operation for {_path} missing required field: old_str"
                elif _new is None:
                    _reason = f"Replace operation for {_path} missing required field: new_str"
                elif _old == _new:
                    _reason = "no-op edit rejected: old_str and new_str are identical"
                elif not _fp.exists() and not _working:
                    _reason = f"file {_path} does not exist"
                else:
                    _res = _resolve_replace(_working, _old, _new, _path)
                    if _res["ok"]:
                        _working = _res["new_content"]
                        # Hand resolved content to the queue loop so fuzzy matches
                        # apply deterministically (its exact match would miss).
                        _op["__resolved_new_content__"] = _res["new_content"]
                        _reason = None
                    else:
                        _reason = _res["reason"]
                if _reason is not None:
                    failed_files.append({"path": _path, "op_index": _idx, "error": _reason})
                    _group_ok = False
                    break
            elif _otype == "modify":
                _content = _op.get("new_content")
                if _content is None:
                    _content = _op.get("content")
                if _content is None:
                    failed_files.append(
                        {
                            "path": _path,
                            "op_index": _idx,
                            "error": "modify op missing content / new_content",
                        }
                    )
                    _group_ok = False
                    break
                _working = _content
            elif _otype == "create":
                _working = _op.get("content", "")
            # delete / rename: no text content to validate at this stage
        if _group_ok:
            _applicable_keys.add(_key)

    if failed_files:
        # Drop ops belonging to any failed file group; keep the rest.
        ops = [op for op in ops if _op_file_key(op) in _applicable_keys]

    if not ops:
        # Nothing applicable — surface the first failure as the primary error
        # while still returning the full structured list for retry.
        _first = failed_files[0] if failed_files else {}
        return {
            "success": False,
            "error": _first.get("error", "No applicable operations"),
            "failed": failed_files,
            "operations_queued": 0,
            "operations_applied": 0,
        }

    from victor.agent.change_tracker import ChangeType, get_change_tracker
    from victor.config.settings import get_project_paths

    # Initialize editor
    backup_dir = get_project_paths().backups_dir
    editor = _create_file_editor(backup_dir=str(backup_dir))
    transaction_id = editor.start_transaction(desc)

    # Initialize change tracker for undo/redo
    tracker = get_change_tracker()
    tracker.begin_change_group("edit", desc or f"Edit {len(ops)} files")

    # Maintain pending contents for files being modified multiple times in one transaction
    pending_contents: Dict[str, str] = {}
    # Track paths for which we've already captured the original disk content for the tracker
    captured_original: Set[str] = set()

    def _get_current_content(path: str, file_path: Path) -> str:
        if path in pending_contents:
            return pending_contents[path]
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            pending_contents[path] = content
            return content
        return ""

    def _get_tracker_original(path: str, file_path: Path) -> Optional[str]:
        if path in captured_original:
            return None
        captured_original.add(path)
        if file_path.exists():
            return file_path.read_text(encoding="utf-8")
        return None

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

                # If creating over existing file, capture original
                tracker_original = _get_tracker_original(path, file_path)

                pending_contents[path] = content
                editor.add_create(path, content)
                by_type["create"] += 1
                # Track for undo
                tracker.record_change(
                    file_path=str(file_path),
                    change_type=ChangeType.CREATE,
                    original_content=tracker_original,
                    new_content=content,
                    tool_name="edit",
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
                tracker_original = _get_tracker_original(path, file_path)

                pending_contents[path] = content
                editor.add_modify(path, content)
                by_type["modify"] += 1
                # Track for undo
                tracker.record_change(
                    file_path=str(file_path),
                    change_type=ChangeType.MODIFY,
                    original_content=tracker_original,
                    new_content=content,
                    tool_name="edit",
                    tool_args={"type": "modify", "path": path},
                )

            elif op_type == "delete":
                # Read content before delete for undo
                tracker_original = _get_tracker_original(path, file_path)

                pending_contents.pop(path, None)
                editor.add_delete(path)
                by_type["delete"] += 1
                # Track for undo
                tracker.record_change(
                    file_path=str(file_path),
                    change_type=ChangeType.DELETE,
                    original_content=tracker_original,
                    new_content=None,
                    tool_name="edit",
                    tool_args={"type": "delete", "path": path},
                )

            elif op_type == "rename":
                new_path = op["new_path"]

                # If renamed file is already in transaction state
                if path in pending_contents:
                    pending_contents[new_path] = pending_contents.pop(path)
                if path in captured_original:
                    captured_original.remove(path)
                    captured_original.add(new_path)

                editor.add_rename(path, new_path)
                by_type["rename"] += 1
                # Track for undo
                new_file_path = Path(new_path).expanduser().resolve()
                tracker.record_change(
                    file_path=str(new_file_path),
                    change_type=ChangeType.RENAME,
                    original_path=str(file_path),
                    tool_name="edit",
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

                # Read current content (from pending state if already modified in this call)
                current_content = _get_current_content(path, file_path)

                # The validation pre-pass already resolved this op (including any
                # whitespace-tolerant fuzzy match) into concrete new content.
                resolved = op.pop("__resolved_new_content__", None)
                if resolved is not None:
                    new_content = resolved
                else:
                    # Defensive path (pre-pass should have caught any failure).
                    if not file_path.exists() and path not in pending_contents:
                        return {
                            "success": False,
                            "error": f"Replace operation failed: file {path} does not exist",
                        }
                    res = _resolve_replace(current_content, old_str, new_str, path)
                    if not res["ok"]:
                        return {"success": False, "error": res["reason"]}
                    new_content = res["new_content"]

                # Read original content for undo (true disk state)
                tracker_original = _get_tracker_original(path, file_path)

                pending_contents[path] = new_content

                # Queue as a modify operation (overwrites previous add_modify for this path)
                editor.add_modify(path, new_content)
                if "replace" not in by_type:
                    by_type["replace"] = 0
                by_type["replace"] += 1

                # Track for undo
                tracker.record_change(
                    file_path=str(file_path),
                    change_type=ChangeType.MODIFY,
                    original_content=tracker_original,
                    new_content=new_content,
                    tool_name="edit",
                    tool_args={
                        "type": "replace",
                        "path": path,
                        "old_str": old_str[:50],
                    },
                )

    except Exception as e:
        editor.abort()
        tracker.commit_change_group()  # Empty commit to reset state
        return {"success": False, "error": f"Failed to queue operations: {str(e)}"}

    operations_queued = len(ops)

    import io
    import sys

    def _capture_stdout(fn):
        """Call fn() while capturing any stdout it writes; return captured text."""
        old_stdout = sys.stdout
        sys.stdout = buf = io.StringIO()
        try:
            fn()
        finally:
            sys.stdout = old_stdout
        return buf.getvalue()

    def _format_diff_for_console(preview_text: str) -> str:
        """Format diff output for Rich console with syntax highlighting.

        Converts unified diff format to Rich-compatible markdown with
        proper syntax highlighting for additions/deletions.

        Args:
            preview_text: Raw unified diff output from FileEditor

        Returns:
            Formatted diff string with Rich markup
        """
        if not preview_text:
            return ""

        lines = preview_text.splitlines()
        formatted_lines = []

        for line in lines:
            # Unified diff markers
            if line.startswith("---") or line.startswith("+++"):
                # File paths - show in cyan
                formatted_lines.append(f"[cyan]{line}[/]")
            elif line.startswith("@@"):
                # Line numbers - show in dim
                formatted_lines.append(f"[dim]{line}[/]")
            elif line.startswith("+"):
                # Additions - show in green
                formatted_lines.append(f"[green]{line}[/]")
            elif line.startswith("-"):
                # Deletions - show in red
                formatted_lines.append(f"[red]{line}[/]")
            elif line.startswith(" "):
                # Context lines - show in dim
                formatted_lines.append(f"[dim]{line}[/]")
            else:
                # Other lines (headers, etc.)
                formatted_lines.append(line)

        return "\n".join(formatted_lines)

    # Always capture diff for preview (after successful operations)
    # This ensures the preview shows actual changes, not just requested changes
    diff_output = None
    formatted_diff = None

    # Handle preview mode
    if preview:
        preview_text = _capture_stdout(lambda: editor.preview_diff(context_lines=ctx))
        formatted_diff = _format_diff_for_console(preview_text)

        if not commit:
            editor.abort()
            return {
                "success": True,
                "operations_queued": operations_queued,
                "operations_applied": 0,
                "by_type": by_type,
                "preview_output": preview_text,
                "diff": preview_text,  # Raw diff for programmatic use
                "diff_formatted": formatted_diff,  # Rich-formatted diff for console
                "message": f"Preview generated for {operations_queued} operations (not applied)",
            }
        else:
            # Preview but still commit (suppress FileEditor's commit stdout)
            _capture_stdout(lambda: None)  # discard; real commit captured below
            success_ref = [False]

            def _do_commit():
                success_ref[0] = editor.commit(dry_run=False)

            _capture_stdout(_do_commit)
            success = success_ref[0]
            if success:
                # FileEditor clears its transaction on commit, so preserve the
                # already-computed preview diff as the authoritative change view.
                diff_output = preview_text
                formatted_diff = _format_diff_for_console(diff_output)
                return {
                    "success": True,
                    "operations_queued": operations_queued,
                    "operations_applied": operations_queued,
                    "by_type": by_type,
                    "preview_output": preview_text,
                    "diff": diff_output,  # Raw diff for programmatic use
                    "diff_formatted": formatted_diff,  # Rich-formatted diff for console
                    "message": f"Applied {operations_queued} operations successfully",
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to commit changes. Transaction rolled back.",
                }

    # Handle commit (suppress FileEditor's transaction stdout — Victor formats its own output)
    if commit:
        # FileEditor clears current_transaction during commit(), so capture the
        # diff while the transaction is still active.
        diff_output = _capture_stdout(lambda: editor.preview_diff(context_lines=ctx))
        formatted_diff = _format_diff_for_console(diff_output) if diff_output else None
        success_ref = [False]

        def _do_commit():
            success_ref[0] = editor.commit(dry_run=False)

        _capture_stdout(_do_commit)
        success = success_ref[0]
        if success:
            # Commit the change group for undo/redo
            tracker.commit_change_group()

            # Invalidate file content cache for all modified paths
            try:
                from victor.tools.filesystem import (
                    get_file_content_cache,
                    is_file_cache_enabled,
                )

                if is_file_cache_enabled():
                    cache = get_file_content_cache()
                    for op in ops:
                        if "path" in op:
                            cache.invalidate(op["path"])
                        if op.get("type") == "rename" and "new_path" in op:
                            cache.invalidate(op["new_path"])
            except (ImportError, Exception) as e:
                logger.debug(f"Failed to invalidate file cache: {e}")

            _msg = f"Successfully applied {operations_queued} operations. Use /undo to revert."
            if failed_files:
                _bad = ", ".join(sorted({str(f["path"]) for f in failed_files}))
                _msg = (
                    f"Applied {operations_queued} operation(s); skipped "
                    f"{len(failed_files)} that failed validation ({_bad}). "
                    "Other files were committed — retry only the failed ones."
                )
            result = {
                "success": True,
                "operations_queued": operations_queued,
                "operations_applied": operations_queued,
                "by_type": by_type,
                "message": _msg,
                "transaction_id": transaction_id,
            }
            if failed_files:
                result["partial"] = True
                result["failed"] = failed_files

            # Include diff if available (for preview renderer)
            if diff_output:
                result["diff"] = diff_output
            if formatted_diff:
                result["diff_formatted"] = formatted_diff

            return result
        else:
            # Clear change group on failure
            tracker._current_group = None
            return {
                "success": False,
                "error": "Failed to commit changes. Transaction rolled back.",
            }
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
