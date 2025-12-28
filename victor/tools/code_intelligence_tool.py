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


from typing import Any, Dict, List, Optional
from pathlib import Path

from tree_sitter import Query, QueryCursor
from victor_coding.codebase.tree_sitter_manager import get_parser
from victor.tools.base import AccessMode, DangerLevel, Priority, ExecutionCategory
from victor.tools.decorators import tool


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


@tool(
    category="code_intelligence",
    priority=Priority.HIGH,  # Important for code navigation
    access_mode=AccessMode.READONLY,  # Only reads files for analysis
    danger_level=DangerLevel.SAFE,  # No side effects
    keywords=[
        "symbol",
        "find",
        "definition",
        "function",
        "class",
        "locate",
        "ast",
        # Natural language patterns for symbol lookup (semantic matching)
        "method",
        "where is",
        "show me",
        "look up",
        "go to definition",
        "find the",
        "locate the",
        "get definition",
        "code for",
        "implementation",
    ],
    stages=["initial", "planning", "reading", "analysis"],  # Available early when symbol name known
    mandatory_keywords=[
        "find",
        "find symbol",
        "find definition",
    ],  # From MANDATORY_TOOL_KEYWORDS "find" -> ["symbol", "refs"]
)
async def symbol(file_path: str, symbol_name: str) -> Optional[Dict[str, Any]]:
    """[AST-AWARE] Get FULL CODE of a function/class definition in a specific file.

    Returns the complete code block (not just a reference). Use when you KNOW:
    1. The exact file path
    2. The exact symbol name

    DIFFERS FROM:
    - search(): Finds files/snippets when you DON'T know where code is
    - refs(): Finds all USAGES of a symbol across project (not the definition)
    - graph(): Shows RELATIONSHIPS between symbols (not code content)

    Args:
        file_path: Path to Python file (.py). MUST be known.
        symbol_name: Exact name of function or class. MUST be known.

    Returns:
        Dict with symbol_name, type, file_path, start_line, end_line, code.
        None if not found. {"error": ...} on file/parse errors.

    Example:
        symbol(file_path="victor/agent/orchestrator.py", symbol_name="chat")
        # Returns full code of the chat() method with line numbers
    """
    try:
        parser = get_parser("python")

        with open(file_path, "rb") as f:
            content = f.read()

        tree = parser.parse(content)
        root_node = tree.root_node

        for symbol_type in ["function", "class"]:
            query = Query(parser.language, PYTHON_QUERIES[symbol_type])
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


@tool(
    category="code_intelligence",
    priority=Priority.HIGH,  # Important for code navigation
    access_mode=AccessMode.READONLY,  # Only reads files for analysis
    danger_level=DangerLevel.SAFE,  # No side effects
    keywords=[
        "refs",
        "references",
        "usage",
        "occurrences",
        "ast",
        # Natural language patterns for usage lookup (semantic matching)
        "used",
        "called",
        "invoked",
        "callers",
        "who calls",
        "all usages",
        "find usages",
        "list references",
        "where used",
    ],
    stages=["initial", "planning", "reading", "analysis"],  # Available early for impact analysis
    execution_category=ExecutionCategory.READ_ONLY,
    mandatory_keywords=[
        "find",
        "find references",
        "find usages",
    ],  # From MANDATORY_TOOL_KEYWORDS "find" -> ["symbol", "refs"]
)
async def refs(symbol_name: str, search_path: str = ".") -> List[Dict[str, Any]]:
    """[AST-AWARE] Find all USAGES of a symbol across the project.

    Project-wide scan using AST parsing. More accurate than grep (exact identifier
    matches only, no substrings). Use when you KNOW the symbol name but want to
    find WHERE it's used.

    DIFFERS FROM:
    - symbol(): Gets the DEFINITION code (not usages). Use when you know file + name.
    - search(): Finds files by CONCEPT/TEXT. Use when you DON'T know the exact name.
    - graph(): Shows RELATIONSHIPS/dependencies. Use for impact analysis, not locations.

    Args:
        symbol_name: Exact symbol name (function, class, variable) to find usages of.
        search_path: Directory to search recursively. Default: current directory.

    Returns:
        List of dicts: [{file_path, line, column, preview}, ...]
        Empty list if no references found.

    Example:
        refs(symbol_name="BaseProvider")
        # Returns all files/lines where BaseProvider is referenced

    When NOT to use:
        - Non-Python files (use grep instead)
        - Finding text patterns (use search() with mode="literal")
        - Getting the definition code (use symbol() instead)
    """
    references = []
    parser = get_parser("python")
    query = Query(parser.language, PYTHON_QUERIES["identifier"])

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


# Note: For project-wide symbol renaming, use the consolidated `rename` tool
# from refactor_tool.py with scope="project". Example:
#   rename(old_name="foo", new_name="bar", scope="project")
