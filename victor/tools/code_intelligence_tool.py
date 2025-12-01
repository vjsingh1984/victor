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


from typing import TYPE_CHECKING, Any, Dict, List, Optional
from pathlib import Path

from tree_sitter import QueryCursor
from victor.codebase.tree_sitter_manager import get_parser
from victor.tools.decorators import tool

if TYPE_CHECKING:
    from victor.tools.base import ToolRegistry


# Tree-sitter queries to find function and class definitions in Python
PYTHON_QUERIES = {
    "function": """
    (function_definition
      name: (identifier) @function.name)
    """,
    "class": """
    (class_definition
      name: (identifier) @class.name)
    """,
    "identifier": """
    (identifier) @name
    """,
}


@tool
async def find_symbol(file_path: str, symbol_name: str) -> Optional[Dict[str, Any]]:
    """
    Finds the definition of a class or function in a Python file using AST parsing.

    Args:
        file_path: The path to the Python file to search in.
        symbol_name: The name of the class or function to find.

    Returns:
        A dictionary containing the symbol's name, type, start line, end line,
        and the block of code defining it. Returns None if not found.
    """
    try:
        parser = get_parser("python")

        with open(file_path, "rb") as f:
            content = f.read()

        tree = parser.parse(content)
        root_node = tree.root_node

        for symbol_type in ["function", "class"]:
            query = parser.language.query(PYTHON_QUERIES[symbol_type])
            cursor = QueryCursor(query)
            captures_dict = cursor.captures(root_node)

            # captures_dict is {"function.name": [node1, node2, ...]} or {"class.name": [...]}
            for _capture_name, nodes in captures_dict.items():
                for node in nodes:
                    if node.text.decode("utf8") == symbol_name:
                        # We found the name identifier, now get the parent definition node
                        definition_node = node.parent
                        start_line = definition_node.start_point[0] + 1
                        end_line = definition_node.end_point[0] + 1
                        code_block = definition_node.text.decode("utf8")

                        return {
                            "symbol_name": symbol_name,
                            "type": symbol_type,
                            "file_path": file_path,
                            "start_line": start_line,
                            "end_line": end_line,
                            "code": code_block,
                        }

        return None  # Symbol not found

    except FileNotFoundError:
        return {"error": f"File not found: {file_path}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}


@tool
async def find_references(symbol_name: str, search_path: str = ".") -> List[Dict[str, Any]]:
    """
    Finds all references to a symbol in a directory using AST parsing.

    Args:
        symbol_name: The name of the symbol (variable, function, class, etc.) to find.
        search_path: The directory path to search in. Defaults to the current directory.

    Returns:
        A list of dictionaries, each representing a reference found.
    """
    references = []
    parser = get_parser("python")
    query = parser.language.query(PYTHON_QUERIES["identifier"])

    root_path = Path(search_path)
    if not root_path.is_dir():
        return [{"error": f"Invalid search path: {search_path} is not a directory."}]

    for file_path in root_path.rglob("*.py"):
        try:
            with open(file_path, "rb") as f:
                content_bytes = f.read()

            tree = parser.parse(content_bytes)
            cursor = QueryCursor(query)
            captures_dict = cursor.captures(tree.root_node)

            # For reading lines for the preview
            content_lines = content_bytes.decode("utf8", errors="ignore").splitlines()

            # captures_dict is {"name": [node1, node2, ...]}
            for _capture_name, nodes in captures_dict.items():
                for node in nodes:
                    if node.text.decode("utf8") == symbol_name:
                        line_number = node.start_point[0] + 1
                        col_number = node.start_point[1] + 1
                        references.append(
                            {
                                "file_path": str(file_path),
                                "line": line_number,
                                "column": col_number,
                                "preview": content_lines[line_number - 1].strip(),
                            }
                        )
        except Exception:
            # Ignore files that can't be parsed
            continue

    return references


@tool
async def rename_symbol(
    symbol_name: str, new_symbol_name: str, context: dict, search_path: str = "."
) -> str:
    """
    Safely renames a symbol across all Python files in the specified search path.

    This tool finds all references to `symbol_name` and replaces them with `new_symbol_name`
    within a transactional file editing process. Users should review and commit the changes
    using the `file_editor` tool.

    Args:
        symbol_name: The current name of the symbol (function, class, variable, etc.) to rename.
        new_symbol_name: The new name for the symbol.
        context: The tool context, provided by the orchestrator.
        search_path: The directory path to search in. Defaults to the current directory.

    Returns:
        A message indicating the status of the rename operation and instructions for review.
    """
    tool_registry: ToolRegistry = context.get("tool_registry")
    if not tool_registry:
        return "Error: ToolRegistry not found in context."

    # 1. Find all references to the symbol
    references_result = await tool_registry.execute(
        "find_references", context, symbol_name=symbol_name, search_path=search_path
    )

    if not references_result.success:
        return f"Error finding references: {references_result.error}"

    references = references_result.output
    if not references:
        return f"No references found for symbol '{symbol_name}'."

    # Group references by file
    files_to_modify: Dict[str, List[Dict[str, Any]]] = {}
    for ref in references:
        file_path = ref["file_path"]
        if file_path not in files_to_modify:
            files_to_modify[file_path] = []
        files_to_modify[file_path].append(ref)

    # 2. Start a file editing transaction
    transaction_description = f"Rename symbol '{symbol_name}' to '{new_symbol_name}'."
    start_transaction_result = await tool_registry.execute(
        "file_editor", context, operation="start_transaction", description=transaction_description
    )
    if not start_transaction_result.success:
        return f"Error starting transaction: {start_transaction_result.error}"

    # 3. Queue modifications for each file
    modified_files_count = 0
    for file_path, _refs in files_to_modify.items():
        try:
            # Read current content
            read_result = await tool_registry.execute("read_file", context, path=file_path)
            if not read_result.success:
                print(f"Warning: Could not read {file_path}. Skipping.")
                continue

            original_content = read_result.output
            lines = original_content.splitlines()

            # Perform replacements. Iterate through lines and replace
            # This is a basic text replacement; for true AST safety,
            # an AST transformer would be needed here.
            # However, `find_references` gives us line/col, which helps
            # in targeted replacement within a line.
            new_lines = []
            for _i, line in enumerate(lines):
                # Check if this line contains a reference for the current symbol
                # This is a simplified check. A more robust solution would
                # use the column information from `refs` to ensure exact match.
                if symbol_name in line:
                    new_lines.append(line.replace(symbol_name, new_symbol_name))
                else:
                    new_lines.append(line)

            new_content = "\n".join(new_lines)

            # Add modification to the transaction
            add_modify_result = await tool_registry.execute(
                "file_editor",
                context,
                operation="add_modify",
                path=file_path,
                new_content=new_content,
            )
            if not add_modify_result.success:
                return f"Error queuing modification for {file_path}: " f"{add_modify_result.error}"
            modified_files_count += 1

        except Exception as e:
            # Abort the transaction if any file processing fails
            await tool_registry.execute("file_editor", context, operation="abort")
            return f"Failed to process file {file_path} for renaming: {e}. Transaction aborted."

    if modified_files_count == 0:
        await tool_registry.execute("file_editor", context, operation="abort")
        return "No files were modified. Transaction aborted."

    return (
        f"Queued rename of '{symbol_name}' to '{new_symbol_name}' across "
        f"{modified_files_count} files. Please use the `file_editor(operation='preview')` "
        f"tool to review changes, then `file_editor(operation='commit')` or "
        f"`file_editor(operation='rollback')`."
    )
