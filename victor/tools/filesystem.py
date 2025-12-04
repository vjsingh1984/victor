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

"""Filesystem tools for reading, writing, and listing contents."""

from pathlib import Path
from typing import List, Dict, Any

import aiofiles

from victor.tools.decorators import tool


@tool
async def read_file(
    path: str,
    offset: int = 0,
    limit: int = 0,
    search: str = "",
    context: int = 2,
    regex: bool = False,
) -> str:
    """
    Read the contents of a file from the filesystem.

    TOKEN-EFFICIENT MODES:
    - Default: Returns full file content
    - With search: Returns ONLY matching lines with context (huge token savings!)
    - With offset/limit: Returns specific line range

    Args:
        path: The path to the file to read.
        offset: Line number to start reading from (0-based). Default 0 (start of file).
        limit: Maximum number of lines to read. Default 0 (read all lines).
        search: Pattern to search for. If provided, returns only matching lines
                with context instead of full file. Use this for targeted lookups!
        context: Number of lines before/after each match when using search. Default 2.
        regex: If True, treat search as regex pattern. Default False (literal string).

    Returns:
        If search is empty: Full file content (or offset/limit range)
        If search is provided: Only matching lines with context, formatted as:
            [N matches in file.py (total lines: M)]
            line_num: matching_line
            ...

    Raises:
        FileNotFoundError: If the file doesn't exist.
        IsADirectoryError: If the path is a directory.
        PermissionError: If access is denied.

    Examples:
        Read a Python source file:
            await read_file("src/main.py")

        Read first 100 lines:
            await read_file("src/main.py", limit=100)

        TOKEN-EFFICIENT: Search for specific function (returns ~50 tokens instead of 5000):
            await read_file("src/main.py", search="def calculate")

        Search with more context:
            await read_file("src/main.py", search="class User", context=5)

        Regex search for all function definitions:
            await read_file("src/main.py", search="def \\w+\\(", regex=True)
    """
    file_path = Path(path).expanduser().resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not file_path.is_file():
        raise IsADirectoryError(f"Path is not a file: {path}")

    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        content = await f.read()

    # Normalize parameters (handle non-int input from model)
    def _to_int(val, default: int) -> int:
        if isinstance(val, int):
            return val
        if isinstance(val, str) and val.isdigit():
            return int(val)
        return default

    offset = _to_int(offset, 0)
    limit = _to_int(limit, 0)
    context = _to_int(context, 2)

    # TOKEN-EFFICIENT MODE: Search/grep
    if search:
        from victor.tools.output_utils import grep_lines

        result = grep_lines(
            content=content,
            pattern=search,
            context_before=context,
            context_after=context,
            case_sensitive=True,
            is_regex=regex,
            max_matches=50,
            file_path=path,
        )
        return result.to_string(show_line_numbers=True, max_matches=50)

    # If no offset/limit, return full content
    if offset == 0 and limit == 0:
        return content

    # Handle offset and limit
    lines = content.split("\n")
    total_lines = len(lines)

    # Clamp offset to valid range
    offset = max(0, min(offset, total_lines))

    # Select lines
    if limit > 0:
        selected = lines[offset : offset + limit]
        end_line = min(offset + limit, total_lines)
    else:
        selected = lines[offset:]
        end_line = total_lines

    # Add header showing line range
    header = f"[Lines {offset + 1}-{end_line} of {total_lines}]\n"
    return header + "\n".join(selected)


@tool
async def write_file(path: str, content: str) -> str:
    """
    Write content to a file, creating it if it doesn't exist.

    Args:
        path: The path to the file to write to.
        content: The content to write into the file.

    Returns:
        A confirmation message upon success.

    Raises:
        IsADirectoryError: If the path is a directory.
        PermissionError: If access is denied.

    Examples:
        Create a new Python module:
            await write_file("src/utils.py", "def helper():\\n    pass")

        Save configuration:
            await write_file("config.yaml", "debug: true\\nport: 8000")

        Create a README:
            await write_file("README.md", "# Project Title\\n\\nDescription here")
    """
    from victor.agent.change_tracker import ChangeType, get_change_tracker

    file_path = Path(path).expanduser().resolve()

    if file_path.exists() and file_path.is_dir():
        raise IsADirectoryError(f"Cannot write to directory: {path}")

    # Track the change for undo/redo
    tracker = get_change_tracker()
    original_content = None
    change_type = ChangeType.CREATE

    if file_path.exists():
        # File exists - this is a modification
        change_type = ChangeType.MODIFY
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            original_content = await f.read()

    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Record the change
    tracker.begin_change_group("write_file", f"Write to {path}")
    tracker.record_change(
        file_path=str(file_path),
        change_type=change_type,
        original_content=original_content,
        new_content=content,
        tool_name="write_file",
        tool_args={"path": path},
    )

    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(content)

    tracker.commit_change_group()

    action = "created" if change_type == ChangeType.CREATE else "modified"
    return f"Successfully {action} {path} ({len(content)} characters). Use /undo to revert."


@tool
async def list_directory(
    path: str,
    recursive: bool = False,
    pattern: str = "",
    extensions: str = "",
    dirs_only: bool = False,
    files_only: bool = False,
    max_items: int = 500,
) -> List[Dict[str, Any]]:
    """
    List the contents of a directory with optional filtering.

    TOKEN-EFFICIENT MODES:
    - Default: Returns all items (can be large)
    - With pattern: Returns only items matching glob pattern
    - With extensions: Returns only files with specific extensions

    Args:
        path: The path to the directory to list.
        recursive: Whether to list subdirectories recursively. Defaults to False.
        pattern: Glob pattern to filter results (e.g., "*.py", "test_*.ts").
        extensions: Comma-separated list of extensions to include (e.g., "py,ts,js").
        dirs_only: If True, return only directories.
        files_only: If True, return only files.
        max_items: Maximum number of items to return. Default 500.

    Returns:
        A list of dictionaries, where each dictionary represents a file or directory.

    Examples:
        List files in current directory:
            await list_directory(".")

        TOKEN-EFFICIENT: List only Python files:
            await list_directory("src", pattern="*.py")

        List test files recursively:
            await list_directory(".", recursive=True, pattern="test_*.py")

        List specific extensions:
            await list_directory(".", extensions="py,ts,js")

        List only directories:
            await list_directory(".", dirs_only=True)
    """
    import fnmatch

    try:
        dir_path = Path(path).expanduser().resolve()

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")

        # Parse extensions if provided
        ext_set = None
        if extensions:
            ext_set = {ext if ext.startswith(".") else f".{ext}" for ext in extensions.split(",")}

        # Normalize max_items (handle non-int input from model)
        if not isinstance(max_items, int):
            max_items = (
                int(max_items) if isinstance(max_items, str) and max_items.isdigit() else 500
            )

        items = []
        count = 0

        if recursive:
            iterator = dir_path.rglob("*")
        else:
            iterator = sorted(dir_path.iterdir())

        for p in iterator:
            if count >= max_items:
                break

            is_dir = p.is_dir()
            name = str(p.relative_to(dir_path)) if recursive else p.name

            # Apply filters
            if dirs_only and not is_dir:
                continue
            if files_only and is_dir:
                continue

            # Pattern filter (glob)
            if pattern and not fnmatch.fnmatch(name, pattern):
                continue

            # Extension filter
            if ext_set and not is_dir and p.suffix not in ext_set:
                continue

            items.append(
                {
                    "path" if recursive else "name": name,
                    "type": "directory" if is_dir else "file",
                }
            )
            count += 1

        # Add metadata if filtered
        if pattern or extensions or dirs_only or files_only:
            # Return with metadata header
            return {
                "items": items,
                "count": len(items),
                "truncated": count >= max_items,
                "filters": {
                    "pattern": pattern or None,
                    "extensions": extensions or None,
                    "dirs_only": dirs_only,
                    "files_only": files_only,
                },
            }

        return items

    except Exception as e:
        # Let the decorator handle the exception and format it
        raise e
