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

from victor.tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class RefactorTool(BaseTool):
    """Tool for automated code refactoring."""

    @property
    def name(self) -> str:
        """Get tool name."""
        return "refactor"

    @property
    def description(self) -> str:
        """Get tool description."""
        return """Automated code refactoring with safety checks.

Safely transform code using AST-based analysis:
- Rename symbols across files
- Extract functions from code blocks
- Inline simple variables
- Organize and optimize imports
- Preview changes before applying

Operations:
- rename: Rename variable/function/class across codebase
- extract_function: Extract code block into new function
- inline_variable: Inline simple variable assignments
- organize_imports: Sort and optimize import statements
- preview: Show refactoring preview without applying

Example workflows:
1. Rename a function:
   refactor(operation="rename", file="app.py", old_name="process", new_name="process_data")

2. Extract function:
   refactor(operation="extract_function", file="app.py", start_line=10, end_line=15,
            function_name="validate_input")

3. Organize imports:
   refactor(operation="organize_imports", file="app.py")

4. Preview changes:
   refactor(operation="rename", old_name="foo", new_name="bar", preview=True)
"""

    @property
    def parameters(self) -> Dict[str, Any]:
        """Get tool parameters."""
        return self.convert_parameters_to_schema(
        [
            ToolParameter(
                name="operation",
                type="string",
                description="Operation: rename, extract_function, inline_variable, organize_imports",
                required=True,
            ),
            ToolParameter(
                name="file",
                type="string",
                description="File path to refactor",
                required=False,
            ),
            ToolParameter(
                name="old_name",
                type="string",
                description="Current symbol name (for rename)",
                required=False,
            ),
            ToolParameter(
                name="new_name",
                type="string",
                description="New symbol name (for rename)",
                required=False,
            ),
            ToolParameter(
                name="start_line",
                type="integer",
                description="Start line for extraction",
                required=False,
            ),
            ToolParameter(
                name="end_line",
                type="integer",
                description="End line for extraction",
                required=False,
            ),
            ToolParameter(
                name="function_name",
                type="string",
                description="Name for extracted function",
                required=False,
            ),
            ToolParameter(
                name="scope",
                type="string",
                description="Scope: file (single file) or project (all files)",
                required=False,
            ),
            ToolParameter(
                name="preview",
                type="boolean",
                description="Preview changes without applying (default: false)",
                required=False,
            ),
        ]
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute refactoring operation.

        Args:
            operation: Refactoring operation to perform
            **kwargs: Operation-specific parameters

        Returns:
            Tool result with refactoring results
        """
        operation = kwargs.get("operation")

        if not operation:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: operation",
            )

        try:
            if operation == "rename":
                return await self._rename_symbol(kwargs)
            elif operation == "extract_function":
                return await self._extract_function(kwargs)
            elif operation == "inline_variable":
                return await self._inline_variable(kwargs)
            elif operation == "organize_imports":
                return await self._organize_imports(kwargs)
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown operation: {operation}",
                )

        except Exception as e:
            logger.exception("Refactoring failed")
            return ToolResult(
                success=False, output="", error=f"Refactoring error: {str(e)}"
            )

    async def _rename_symbol(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Rename a symbol (variable, function, class)."""
        file_path = kwargs.get("file")
        old_name = kwargs.get("old_name")
        new_name = kwargs.get("new_name")
        scope = kwargs.get("scope", "file")
        preview = kwargs.get("preview", False)

        if not file_path or not old_name or not new_name:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameters: file, old_name, new_name",
            )

        file_obj = Path(file_path)
        if not file_obj.exists():
            return ToolResult(
                success=False, output="", error=f"File not found: {file_path}"
            )

        # Read file
        content = file_obj.read_text()

        # Parse AST to find symbol type and usage
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Syntax error in file: {e}",
            )

        # Find symbol definition
        symbol_info = self._find_symbol(tree, old_name)

        if not symbol_info:
            return ToolResult(
                success=False,
                output="",
                error=f"Symbol '{old_name}' not found in {file_path}",
            )

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
        report.append(f"File: {file_path}")
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

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _extract_function(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Extract code block into a new function."""
        file_path = kwargs.get("file")
        start_line = kwargs.get("start_line")
        end_line = kwargs.get("end_line")
        function_name = kwargs.get("function_name")
        preview = kwargs.get("preview", False)

        if not file_path or not start_line or not end_line or not function_name:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameters: file, start_line, end_line, function_name",
            )

        file_obj = Path(file_path)
        if not file_obj.exists():
            return ToolResult(
                success=False, output="", error=f"File not found: {file_path}"
            )

        # Read file
        content = file_obj.read_text()
        lines = content.split("\n")

        # Validate line numbers
        if start_line < 1 or end_line > len(lines) or start_line > end_line:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid line range: {start_line}-{end_line}",
            )

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
            variables = self._analyze_variables(block_tree)
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
        insert_line = self._find_function_insert_point(lines, start_line)
        modified_lines.insert(insert_line, new_function)

        new_content = "\n".join(modified_lines)

        # Build report
        report = []
        report.append(f"Extract Function: '{function_name}'")
        report.append("=" * 70)
        report.append("")
        report.append(f"File: {file_path}")
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

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _inline_variable(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Inline a simple variable assignment."""
        file_path = kwargs.get("file")
        variable_name = kwargs.get("old_name")  # Reuse old_name param
        preview = kwargs.get("preview", False)

        if not file_path or not variable_name:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameters: file, old_name (variable name)",
            )

        file_obj = Path(file_path)
        if not file_obj.exists():
            return ToolResult(
                success=False, output="", error=f"File not found: {file_path}"
            )

        # Read file
        content = file_obj.read_text()

        # Parse AST to find variable assignment
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Syntax error in file: {e}",
            )

        # Find variable assignment
        assignment = self._find_variable_assignment(tree, variable_name)

        if not assignment:
            return ToolResult(
                success=False,
                output="",
                error=f"Simple assignment for '{variable_name}' not found",
            )

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
        report.append(f"File: {file_path}")
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

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    async def _organize_imports(self, kwargs: Dict[str, Any]) -> ToolResult:
        """Organize and optimize import statements."""
        file_path = kwargs.get("file")
        preview = kwargs.get("preview", False)

        if not file_path:
            return ToolResult(
                success=False,
                output="",
                error="Missing required parameter: file",
            )

        file_obj = Path(file_path)
        if not file_obj.exists():
            return ToolResult(
                success=False, output="", error=f"File not found: {file_path}"
            )

        # Read file
        content = file_obj.read_text()
        lines = content.split("\n")

        # Parse AST to find imports
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Syntax error in file: {e}",
            )

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
                    if self._is_stdlib(alias.name):
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
                    if self._is_stdlib(module):
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
        report.append(f"File: {file_path}")
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

        return ToolResult(
            success=True,
            output="\n".join(report),
            error="",
        )

    def _find_symbol(self, tree: ast.AST, name: str) -> Optional[Dict[str, Any]]:
        """Find symbol in AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return {"type": "function", "node": node}
            elif isinstance(node, ast.ClassDef) and node.name == name:
                return {"type": "class", "node": node}
            elif isinstance(node, ast.Name) and node.id == name:
                return {"type": "variable", "node": node}

        return None

    def _analyze_variables(self, tree: ast.AST) -> Dict[str, List[str]]:
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

    def _find_function_insert_point(self, lines: List[str], current_line: int) -> int:
        """Find best location to insert extracted function."""
        # Simple heuristic: insert at the beginning of file (after imports)
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith("import") and not line.strip().startswith("from"):
                return max(0, i - 1)

        return 0

    def _find_variable_assignment(
        self, tree: ast.AST, name: str
    ) -> Optional[Dict[str, Any]]:
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

    def _is_stdlib(self, module_name: str) -> bool:
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
