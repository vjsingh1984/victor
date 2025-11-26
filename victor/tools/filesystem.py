"""Filesystem tools for reading, writing, and listing contents."""

from pathlib import Path
from typing import List, Dict, Any

import aiofiles

from victor.tools.decorators import tool


@tool
async def read_file(path: str) -> str:
    """
    Read the contents of a file from the filesystem.

    Args:
        path: The path to the file to read.

    Returns:
        The content of the file as a string.
    """
    try:
        file_path = Path(path).expanduser().resolve()

        if not file_path.exists():
            return f"Error: File not found at {path}"
        if not file_path.is_file():
            return f"Error: Path {path} is not a file."

        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            return await f.read()

    except Exception as e:
        return f"An unexpected error occurred: {e}"


@tool
async def write_file(path: str, content: str) -> str:
    """
    Write content to a file, creating it if it doesn't exist.

    Args:
        path: The path to the file to write to.
        content: The content to write into the file.

    Returns:
        A confirmation message upon success.
    """
    try:
        file_path = Path(path).expanduser().resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(content)

        return f"Successfully wrote {len(content)} characters to {path}."

    except Exception as e:
        return f"An unexpected error occurred: {e}"


@tool
async def list_directory(path: str, recursive: bool = False) -> List[Dict[str, Any]]:
    """
    List the contents of a directory.

    Args:
        path: The path to the directory to list.
        recursive: Whether to list subdirectories recursively. Defaults to False.

    Returns:
        A list of dictionaries, where each dictionary represents a file or directory.
    """
    try:
        dir_path = Path(path).expanduser().resolve()

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")

        if recursive:
            items = [
                {
                    "path": str(p.relative_to(dir_path)),
                    "type": "directory" if p.is_dir() else "file",
                }
                for p in dir_path.rglob("*")
            ]
        else:
            items = [
                {
                    "name": p.name,
                    "type": "directory" if p.is_dir() else "file",
                }
                for p in sorted(dir_path.iterdir())
            ]
        return items

    except Exception as e:
        # Let the decorator handle the exception and format it
        raise e

