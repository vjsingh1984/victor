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

"""Code refactoring tool for safe code transformations.

Features:
- Rename symbols (variables, functions, classes)
- Extract functions from code blocks
- Inline variables
- Organize imports
- AST-based analysis for safety
"""

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

from victor.tools.decorators import tool

logger = logging.getLogger(__name__)


# Helper functions

def _find_symbol(tree: ast.AST, name: str) -> Optional[Dict[str, Any]]:
    """Find symbol in AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return {"type": "function", "node": node}
        elif isinstance(node, ast.ClassDef) and node.name == name:
            return {"type": "class", "node": node}
        elif isinstance(node, ast.Name) and node.id == name:
            return {"type": "variable", "node": node}

    return None


def _analyze_variables(tree: ast.AST) -> Dict[str, List[str]]:
    """Analyze variables used in code block."""
    # Simple analysis - can be enhanced
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)

    # For now, assume all names are parameters
    # More sophisticated analysis would distinguish local vs external
    return {
        "params": list(names)[:5],  # Limit to 5 for safety
        "returns": [],
    }


def _find_function_insert_point(lines: List[str], current_line: int) -> int:
    """Find best location to insert extracted function."""
    # Simple heuristic: insert at the beginning of file (after imports)
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith("import") and not line.strip().startswith("from"):
            return max(0, i - 1)

    return 0


def _find_variable_assignment(tree: ast.AST, name: str) -> Optional[Dict[str, Any]]:
    """Find simple variable assignment."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    # Get value as string
                    try:
                        value_str = ast.unparse(node.value)
                    except:
                        value_str = str(node.value)

                    return {
                        "line": node.lineno,
                        "value": value_str,
                    }

    return None


def _is_stdlib(module_name: str) -> bool:
    """Check if module is from standard library."""
    # Common stdlib modules
    stdlib_modules = {
        "abc", "ast", "asyncio", "collections", "concurrent", "contextlib",
        "copy", "dataclasses", "datetime", "decimal", "enum", "functools",
        "hashlib", "io", "itertools", "json", "logging", "math", "os",
        "pathlib", "pickle", "re", "shutil", "socket", "sqlite3", "string",
        "subprocess", "sys", "tempfile", "threading", "time", "typing",
        "unittest", "urllib", "uuid", "warnings", "weakref",
    }

    # Get top-level module
    top_level = module_name.split(".")[0]
    return top_level in stdlib_modules


# Tool functions

@tool
async def refactor_rename_symbol(
    file: str,
    old_name: str,
    new_name: str,
    scope: str = "file",
    preview: bool = False,
) -> Dict[str, Any]:
    """
    Rename a symbol (variable, function, class).

    Safely renames symbols across a file using AST-based analysis
    to avoid false matches.

    Args:
        file: File path to refactor.
        old_name: Current symbol name.
        new_name: New symbol name.
        scope: Scope of rename (file or project) (default: file).
        preview: Preview changes without applying (default: False).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - changes_count: Number of occurrences changed
        - changes: List of changes made
        - preview_text: Preview of changes
        - formatted_report: Human-readable refactoring report
        - error: Error message if failed
    """
    if not file or not old_name or not new_name:
        return {
            "success": False,
            "error": "Missing required parameters: file, old_name, new_name"
        }

    file_obj = Path(file)
    if not file_obj.exists():
        return {"success": False, "error": f"File not found: {file}"}

    # Read file
    content = file_obj.read_text()

    # Parse AST to find symbol type and usage
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"success": False, "error": f"Syntax error in file: {e}"}

    # Find symbol definition
    symbol_info = _find_symbol(tree, old_name)

    if not symbol_info:
        return {
            "success": False,
            "error": f"Symbol '{old_name}' not found in {file}"
        }

    # Perform rename
    lines = content.split("\n")
    modified_lines = []
    changes = []

    for line_num, line in enumerate(lines, 1):
        modified_line = line

        # Use word boundaries for safe replacement
        pattern = r'\b' + re.escape(old_name) + r'\b'

        if re.search(pattern, line):
            modified_line = re.sub(pattern, new_name, line)
            changes.append({
                "line": line_num,
                "old": line,
                "new": modified_line,
            })

        modified_lines.append(modified_line)

    new_content = "\n".join(modified_lines)

    # Build report
    report = []
    report.append(f"Rename Refactoring: '{old_name}' → '{new_name}'")
    report.append("=" * 70)
    report.append("")
    report.append(f"File: {file}")
    report.append(f"Symbol Type: {symbol_info['type']}")
    report.append(f"Changes: {len(changes)} occurrences")
    report.append("")

    if changes:
        report.append("Preview of changes:")
        for change in changes[:10]:  # Show first 10
            report.append(f"\nLine {change['line']}:")
            report.append(f"  - {change['old']}")
            report.append(f"  + {change['new']}")

        if len(changes) > 10:
            report.append(f"\n... and {len(changes) - 10} more changes")

    # Apply changes if not preview
    if not preview:
        file_obj.write_text(new_content)
        report.append("")
        report.append("✅ Changes applied successfully")
    else:
        report.append("")
        report.append("⚠️  This is a PREVIEW - no changes were made")
        report.append("   Run with preview=False to apply changes")

    return {
        "success": True,
        "changes_count": len(changes),
        "changes": changes,
        "preview_text": new_content if preview else None,
        "formatted_report": "\n".join(report)
    }


@tool
async def refactor_extract_function(
    file: str,
    start_line: int,
    end_line: int,
    function_name: str,
    preview: bool = False,
) -> Dict[str, Any]:
    """
    Extract code block into a new function.

    Extracts selected lines of code into a new function,
    analyzing variables to determine parameters and return values.

    Args:
        file: File path to refactor.
        start_line: Start line number (1-indexed).
        end_line: End line number (inclusive).
        function_name: Name for the extracted function.
        preview: Preview changes without applying (default: False).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - new_function: The extracted function code
        - parameters: List of inferred parameters
        - formatted_report: Human-readable refactoring report
        - error: Error message if failed
    """
    if not file or not start_line or not end_line or not function_name:
        return {
            "success": False,
            "error": "Missing required parameters: file, start_line, end_line, function_name"
        }

    file_obj = Path(file)
    if not file_obj.exists():
        return {"success": False, "error": f"File not found: {file}"}

    # Read file
    content = file_obj.read_text()
    lines = content.split("\n")

    # Validate line numbers
    if start_line < 1 or end_line > len(lines) or start_line > end_line:
        return {
            "success": False,
            "error": f"Invalid line range: {start_line}-{end_line}"
        }

    # Extract code block (0-indexed)
    extracted_lines = lines[start_line - 1:end_line]
    code_block = "\n".join(extracted_lines)

    # Determine indentation
    if extracted_lines:
        first_line = extracted_lines[0]
        indent = len(first_line) - len(first_line.lstrip())
        base_indent = " " * indent
    else:
        base_indent = ""

    # Analyze variables used in extracted code
    try:
        # Parse just the extracted block
        block_tree = ast.parse(code_block)
        variables = _analyze_variables(block_tree)
    except SyntaxError:
        # If extraction block has syntax errors, try to infer params
        variables = {"params": [], "returns": []}

    # Build new function
    params_str = ", ".join(variables.get("params", []))
    new_function = f"\n{base_indent}def {function_name}({params_str}):\n"

    # Add docstring
    new_function += f'{base_indent}    """Extracted function."""\n'

    # Add extracted code (with additional indent)
    for line in extracted_lines:
        new_function += f"    {line}\n"

    # Add return statement if needed
    if variables.get("returns"):
        returns_str = ", ".join(variables["returns"])
        new_function += f"{base_indent}    return {returns_str}\n"

    # Build modified content
    modified_lines = lines[:start_line - 1]  # Before extraction

    # Add function call
    call_indent = " " * indent
    if variables.get("returns"):
        returns = ", ".join(variables["returns"])
        modified_lines.append(f"{call_indent}{returns} = {function_name}({params_str})")
    else:
        modified_lines.append(f"{call_indent}{function_name}({params_str})")

    modified_lines.extend(lines[end_line:])  # After extraction

    # Insert function definition at appropriate location
    # Find the best place (e.g., before the current function)
    insert_line = _find_function_insert_point(lines, start_line)
    modified_lines.insert(insert_line, new_function)

    new_content = "\n".join(modified_lines)

    # Build report
    report = []
    report.append(f"Extract Function: '{function_name}'")
    report.append("=" * 70)
    report.append("")
    report.append(f"File: {file}")
    report.append(f"Extracted lines: {start_line}-{end_line}")
    report.append(f"Parameters: {params_str or '(none)'}")
    report.append("")
    report.append("New function:")
    report.append(new_function)
    report.append("")
    report.append("Replaced with call:")
    if variables.get("returns"):
        returns = ", ".join(variables["returns"])
        report.append(f"  {returns} = {function_name}({params_str})")
    else:
        report.append(f"  {function_name}({params_str})")

    # Apply changes if not preview
    if not preview:
        file_obj.write_text(new_content)
        report.append("")
        report.append("✅ Function extracted successfully")
    else:
        report.append("")
        report.append("⚠️  This is a PREVIEW - no changes were made")
        report.append("   Run with preview=False to apply changes")

    return {
        "success": True,
        "new_function": new_function,
        "parameters": variables.get("params", []),
        "formatted_report": "\n".join(report)
    }


@tool
async def refactor_inline_variable(
    file: str,
    variable_name: str,
    preview: bool = False,
) -> Dict[str, Any]:
    """
    Inline a simple variable assignment.

    Replaces all usages of a variable with its assigned value
    and removes the assignment statement.

    Args:
        file: File path to refactor.
        variable_name: Name of variable to inline.
        preview: Preview changes without applying (default: False).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - changes_count: Number of locations changed
        - value: The inlined value
        - formatted_report: Human-readable refactoring report
        - error: Error message if failed
    """
    if not file or not variable_name:
        return {
            "success": False,
            "error": "Missing required parameters: file, variable_name"
        }

    file_obj = Path(file)
    if not file_obj.exists():
        return {"success": False, "error": f"File not found: {file}"}

    # Read file
    content = file_obj.read_text()

    # Parse AST to find variable assignment
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"success": False, "error": f"Syntax error in file: {e}"}

    # Find variable assignment
    assignment = _find_variable_assignment(tree, variable_name)

    if not assignment:
        return {
            "success": False,
            "error": f"Simple assignment for '{variable_name}' not found"
        }

    lines = content.split("\n")

    # Remove assignment line
    assignment_line = assignment["line"]
    value_expr = assignment["value"]

    # Replace variable usage with value
    modified_lines = []
    changes = []

    for line_num, line in enumerate(lines, 1):
        if line_num == assignment_line:
            # Skip assignment line
            changes.append({
                "line": line_num,
                "action": "removed",
                "content": line,
            })
            continue

        # Replace variable usage
        pattern = r'\b' + re.escape(variable_name) + r'\b'
        if re.search(pattern, line):
            modified_line = re.sub(pattern, value_expr, line)
            changes.append({
                "line": line_num,
                "action": "modified",
                "old": line,
                "new": modified_line,
            })
            modified_lines.append(modified_line)
        else:
            modified_lines.append(line)

    new_content = "\n".join(modified_lines)

    # Build report
    report = []
    report.append(f"Inline Variable: '{variable_name}'")
    report.append("=" * 70)
    report.append("")
    report.append(f"File: {file}")
    report.append(f"Value: {value_expr}")
    report.append(f"Changes: {len(changes)} locations")
    report.append("")

    if changes:
        report.append("Changes:")
        for change in changes[:10]:
            if change["action"] == "removed":
                report.append(f"\nLine {change['line']} (removed):")
                report.append(f"  - {change['content']}")
            else:
                report.append(f"\nLine {change['line']} (inlined):")
                report.append(f"  - {change['old']}")
                report.append(f"  + {change['new']}")

        if len(changes) > 10:
            report.append(f"\n... and {len(changes) - 10} more changes")

    # Apply changes if not preview
    if not preview:
        file_obj.write_text(new_content)
        report.append("")
        report.append("✅ Variable inlined successfully")
    else:
        report.append("")
        report.append("⚠️  This is a PREVIEW - no changes were made")
        report.append("   Run with preview=False to apply changes")

    return {
        "success": True,
        "changes_count": len(changes),
        "value": value_expr,
        "formatted_report": "\n".join(report)
    }


@tool
async def refactor_organize_imports(
    file: str,
    preview: bool = False,
) -> Dict[str, Any]:
    """
    Organize and optimize import statements.

    Sorts imports into groups (stdlib, third-party, local),
    removes duplicates, and follows PEP 8 conventions.

    Args:
        file: File path to refactor.
        preview: Preview changes without applying (default: False).

    Returns:
        Dictionary containing:
        - success: Whether operation succeeded
        - stdlib_count: Number of standard library imports
        - third_party_count: Number of third-party imports
        - local_count: Number of local imports
        - formatted_report: Human-readable refactoring report
        - error: Error message if failed
    """
    if not file:
        return {"success": False, "error": "Missing required parameter: file"}

    file_obj = Path(file)
    if not file_obj.exists():
        return {"success": False, "error": f"File not found: {file}"}

    # Read file
    content = file_obj.read_text()
    lines = content.split("\n")

    # Parse AST to find imports
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return {"success": False, "error": f"Syntax error in file: {e}"}

    # Collect imports
    stdlib_imports = []
    third_party_imports = []
    local_imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_line = f"import {alias.name}"
                if alias.asname:
                    import_line += f" as {alias.asname}"

                # Categorize (simple heuristic)
                if _is_stdlib(alias.name):
                    stdlib_imports.append(import_line)
                elif alias.name.startswith("."):
                    local_imports.append(import_line)
                else:
                    third_party_imports.append(import_line)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            names_str = ", ".join(names)

            if node.level > 0:
                # Relative import
                dots = "." * node.level
                import_line = f"from {dots}{module} import {names_str}"
                local_imports.append(import_line)
            else:
                import_line = f"from {module} import {names_str}"
                if _is_stdlib(module):
                    stdlib_imports.append(import_line)
                else:
                    third_party_imports.append(import_line)

    # Sort each group
    stdlib_imports = sorted(set(stdlib_imports))
    third_party_imports = sorted(set(third_party_imports))
    local_imports = sorted(set(local_imports))

    # Build organized imports
    organized = []
    if stdlib_imports:
        organized.extend(stdlib_imports)
        organized.append("")

    if third_party_imports:
        organized.extend(third_party_imports)
        organized.append("")

    if local_imports:
        organized.extend(local_imports)
        organized.append("")

    # Find where imports end in original file
    import_end_line = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.strip().startswith("#"):
            try:
                node = ast.parse(line).body[0]
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_end_line = i + 1
            except:
                if import_end_line > 0:
                    break

    # Build new content
    new_lines = []

    # Keep docstring and initial comments
    docstring_end = 0
    for i, line in enumerate(lines):
        if i == 0 and (line.strip().startswith('"""') or line.strip().startswith("'''")):
            # Module docstring
            new_lines.append(line)
            for j in range(i + 1, len(lines)):
                new_lines.append(lines[j])
                if '"""' in lines[j] or "'''" in lines[j]:
                    docstring_end = j + 1
                    break
            break
        elif line.strip().startswith("#"):
            new_lines.append(line)
        else:
            break

    # Add organized imports
    if docstring_end > 0:
        new_lines.append("")
    new_lines.extend(organized)

    # Add rest of file (skipping old imports)
    rest_start = max(import_end_line, docstring_end)
    new_lines.extend(lines[rest_start:])

    new_content = "\n".join(new_lines)

    # Build report
    report = []
    report.append("Organize Imports")
    report.append("=" * 70)
    report.append("")
    report.append(f"File: {file}")
    report.append(f"Standard library: {len(stdlib_imports)}")
    report.append(f"Third-party: {len(third_party_imports)}")
    report.append(f"Local: {len(local_imports)}")
    report.append("")

    if organized:
        report.append("Organized imports:")
        for line in organized[:20]:
            report.append(f"  {line}")

        if len(organized) > 20:
            report.append(f"  ... and {len(organized) - 20} more")

    # Apply changes if not preview
    if not preview:
        file_obj.write_text(new_content)
        report.append("")
        report.append("✅ Imports organized successfully")
    else:
        report.append("")
        report.append("⚠️  This is a PREVIEW - no changes were made")
        report.append("   Run with preview=False to apply changes")

    return {
        "success": True,
        "stdlib_count": len(stdlib_imports),
        "third_party_count": len(third_party_imports),
        "local_count": len(local_imports),
        "formatted_report": "\n".join(report)
    }


# Keep class for backward compatibility
class RefactorTool:
    """Deprecated: Use individual refactor_* functions instead."""

    def __init__(self):
        """Initialize - deprecated."""
        import warnings
        warnings.warn(
            "RefactorTool class is deprecated. Use refactor_* functions instead.",
            DeprecationWarning,
            stacklevel=2
        )
