"""Filesystem tools for reading, writing, and editing files."""

import os
from pathlib import Path
from typing import Any, Dict

import aiofiles

from codingagent.tools.base import BaseTool, ToolResult


class ReadFileTool(BaseTool):
    """Tool for reading file contents."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file from the filesystem"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Read file contents.

        Args:
            path: File path to read

        Returns:
            ToolResult with file contents
        """
        path = kwargs.get("path")
        if not path:
            return ToolResult(
                success=False,
                output=None,
                error="Missing required parameter: path",
            )

        try:
            file_path = Path(path).expanduser().resolve()

            if not file_path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"File not found: {path}",
                )

            if not file_path.is_file():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Path is not a file: {path}",
                )

            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                contents = await f.read()

            return ToolResult(
                success=True,
                output=contents,
                metadata={
                    "path": str(file_path),
                    "size": len(contents),
                    "lines": contents.count("\n") + 1,
                },
            )

        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                output=None,
                error=f"File is not text or uses unsupported encoding: {path}",
            )
        except PermissionError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Permission denied reading file: {path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to read file: {str(e)}",
            )


class WriteFileTool(BaseTool):
    """Tool for writing content to files."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file, creating it if it doesn't exist"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Write content to file.

        Args:
            path: File path to write
            content: Content to write

        Returns:
            ToolResult with success status
        """
        path = kwargs.get("path")
        content = kwargs.get("content")

        if not path:
            return ToolResult(
                success=False,
                output=None,
                error="Missing required parameter: path",
            )

        if content is None:
            return ToolResult(
                success=False,
                output=None,
                error="Missing required parameter: content",
            )

        try:
            file_path = Path(path).expanduser().resolve()

            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(content)

            return ToolResult(
                success=True,
                output=f"Successfully wrote {len(content)} characters to {path}",
                metadata={
                    "path": str(file_path),
                    "size": len(content),
                    "lines": content.count("\n") + 1,
                },
            )

        except PermissionError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Permission denied writing to file: {path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to write file: {str(e)}",
            )


class ListDirectoryTool(BaseTool):
    """Tool for listing directory contents."""

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return "List the contents of a directory"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to list subdirectories recursively",
                },
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        """List directory contents.

        Args:
            path: Directory path
            recursive: Whether to list recursively

        Returns:
            ToolResult with directory listing
        """
        path = kwargs.get("path")
        recursive = kwargs.get("recursive", False)

        if not path:
            return ToolResult(
                success=False,
                output=None,
                error="Missing required parameter: path",
            )

        try:
            dir_path = Path(path).expanduser().resolve()

            if not dir_path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Directory not found: {path}",
                )

            if not dir_path.is_dir():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Path is not a directory: {path}",
                )

            if recursive:
                items = [
                    {
                        "path": str(p.relative_to(dir_path)),
                        "type": "directory" if p.is_dir() else "file",
                        "size": p.stat().st_size if p.is_file() else None,
                    }
                    for p in dir_path.rglob("*")
                ]
            else:
                items = [
                    {
                        "name": p.name,
                        "type": "directory" if p.is_dir() else "file",
                        "size": p.stat().st_size if p.is_file() else None,
                    }
                    for p in sorted(dir_path.iterdir())
                ]

            return ToolResult(
                success=True,
                output=items,
                metadata={
                    "path": str(dir_path),
                    "count": len(items),
                    "recursive": recursive,
                },
            )

        except PermissionError:
            return ToolResult(
                success=False,
                output=None,
                error=f"Permission denied accessing directory: {path}",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to list directory: {str(e)}",
            )
