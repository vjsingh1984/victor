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

"""
High-performance file system operations using Rust extensions.

This module provides parallel file system operations with 2-3x speedup
compared to Python's os.walk for directory traversal.

Performance characteristics:
- walk_directory_parallel: 2-3x faster than os.walk
- collect_metadata: 3-5x faster than individual stat calls
- filter_by_extension: Near-instant (set-based lookup)
- filter_by_size: Parallel filtering with Rayon
- get_directory_stats: Batch statistics collection

Example usage:
    >>> from victor.native.rust import file_ops
    >>>
    >>> # Walk directory with glob patterns
    >>> files = file_ops.walk_directory_parallel(
    ...     "src",
    ...     patterns=["*.py", "**/*.rs"],
    ...     max_depth=10,
    ...     ignore_patterns=["*.pyc", "__pycache__"]
    ... )
    >>>
    >>> # Filter by extension
    >>> py_files = file_ops.filter_by_extension(files, ["py"])
    >>>
    >>> # Get metadata
    >>> metadata = file_ops.collect_metadata([f.path for f in py_files])
    >>>
    >>> # Get directory statistics
    >>> stats = file_ops.get_directory_stats("src")
    >>> print(f"Total size: {stats['total_size']} bytes")
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Union

# Import Rust extension (will be available after compilation)
try:
    from victor_native import (  # type: ignore[import-not-found]
        FileInfo,
        FileMetadata,
        walk_directory_parallel,
        collect_metadata,
        filter_by_extension,
        filter_by_size,
        get_directory_stats,
        group_by_directory,
        filter_by_modified_time,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

    # Stub classes for type hints when Rust is not available
    class FileInfo:  # type: ignore
        """Stub class when Rust extension is not available."""

        pass

    class FileMetadata:  # type: ignore
        """Stub class when Rust extension is not available."""

        pass


class FileOpsError(Exception):
    """Exception raised for file operations errors."""

    pass


def walk_directory(
    root: Union[str, Path],
    patterns: Optional[List[str]] = None,
    max_depth: int = 100,
    follow_symlinks: bool = False,
    ignore_patterns: Optional[List[str]] = None,
) -> List[FileInfo]:
    """
    Walk directory tree in parallel with pattern matching.

    High-performance directory traversal (2-3x faster than os.walk).

    Args:
        root: Root directory path to traverse
        patterns: List of glob patterns (e.g., ["*.py", "**/*.rs"])
        max_depth: Maximum traversal depth (0 = root only, default: 100)
        follow_symlinks: Whether to follow symbolic links
        ignore_patterns: Patterns to ignore (e.g., ["*.pyc", "__pycache__"])

    Returns:
        List of FileInfo objects with matched files/directories

    Raises:
        FileOpsError: If root directory doesn't exist or Rust unavailable

    Examples:
        >>> # Find all Python files in src directory
        >>> files = walk_directory("src", patterns=["*.py"])
        >>>
        >>> # Find all code files recursively
        >>> files = walk_directory(
        ...     "src",
        ...     patterns=["*.py", "*.rs", "*.java"],
        ...     max_depth=20
        ... )
        >>>
        >>> # Exclude cache directories
        >>> files = walk_directory(
        ...     "src",
        ...     patterns=["*"],
        ...     ignore_patterns=["__pycache__", "*.pyc", ".git"]
        ... )
    """
    if not RUST_AVAILABLE:
        raise FileOpsError(
            "Rust extension not available. Install with: pip install victor-ai[native]"
        )

    if not isinstance(root, (str, Path)):
        raise TypeError(f"root must be str or Path, got {type(root)}")

    root_str = str(root)

    if not os.path.exists(root_str):
        raise FileOpsError(f"Root directory does not exist: {root_str}")

    if not os.path.isdir(root_str):
        raise FileOpsError(f"Root path is not a directory: {root_str}")

    patterns = patterns or []
    ignore_patterns = ignore_patterns or []

    try:
        return walk_directory_parallel(
            root_str,
            patterns=patterns,
            max_depth=max_depth,
            follow_symlinks=follow_symlinks,
            ignore_patterns=ignore_patterns,
        )
    except Exception as e:
        raise FileOpsError(f"Error walking directory: {e}") from e


def get_file_metadata(paths: List[Union[str, Path]]) -> List[FileMetadata]:
    """
    Collect metadata for multiple files in parallel.

    Batch metadata collection (3-5x faster than individual stat calls).

    Args:
        paths: List of file paths to get metadata for

    Returns:
        List of FileMetadata objects. Skips paths that don't exist.

    Raises:
        FileOpsError: If Rust extension is unavailable

    Examples:
        >>> metadata = get_file_metadata(["src/main.py", "README.md"])
        >>> for m in metadata:
        ...     print(f"{m.path}: {m.size} bytes")
    """
    if not RUST_AVAILABLE:
        raise FileOpsError(
            "Rust extension not available. Install with: pip install victor-ai[native]"
        )

    if not isinstance(paths, list):
        raise TypeError(f"paths must be a list, got {type(paths)}")

    paths_str = [str(p) for p in paths]

    try:
        return collect_metadata(paths_str)
    except Exception as e:
        raise FileOpsError(f"Error collecting metadata: {e}") from e


def filter_files_by_extension(files: List[FileInfo], extensions: List[str]) -> List[FileInfo]:
    """
    Filter files by extension using efficient set-based lookup.

    Near-instant filtering with O(1) extension lookup.

    Args:
        files: List of FileInfo objects to filter
        extensions: List of extensions (e.g., ["py", "rs", "java"])

    Returns:
        Filtered list of FileInfo objects matching extensions

    Examples:
        >>> files = walk_directory("src", patterns=["*"])
        >>> code_files = filter_files_by_extension(files, ["py", "rs"])
    """
    if not RUST_AVAILABLE:
        raise FileOpsError(
            "Rust extension not available. Install with: pip install victor-ai[native]"
        )

    if not isinstance(files, list):
        raise TypeError(f"files must be a list, got {type(files)}")

    if not isinstance(extensions, list):
        raise TypeError(f"extensions must be a list, got {type(extensions)}")

    try:
        return filter_by_extension(files, extensions)
    except Exception as e:
        raise FileOpsError(f"Error filtering by extension: {e}") from e


def filter_files_by_size(
    files: List[FileInfo], min_size: int = 0, max_size: int = 0
) -> List[FileInfo]:
    """
    Filter files by size range.

    Args:
        files: List of FileInfo objects to filter
        min_size: Minimum file size in bytes (0 = no minimum)
        max_size: Maximum file size in bytes (0 = no maximum)

    Returns:
        Filtered list of FileInfo objects within size range

    Examples:
        >>> # Find medium-sized files (1KB to 1MB)
        >>> medium_files = filter_files_by_size(
        ...     files,
        ...     min_size=1024,
        ...     max_size=1024*1024
        ... )
    """
    if not RUST_AVAILABLE:
        raise FileOpsError(
            "Rust extension not available. Install with: pip install victor-ai[native]"
        )

    try:
        return filter_by_size(files, min_size=min_size, max_size=max_size)
    except Exception as e:
        raise FileOpsError(f"Error filtering by size: {e}") from e


def get_directory_statistics(
    root: Union[str, Path], max_depth: int = 100
) -> Dict[str, Union[int, List[tuple]]]:
    """
    Get directory statistics including total size and largest files.

    Args:
        root: Root directory path to analyze
        max_depth: Maximum traversal depth

    Returns:
        Dictionary with:
        - total_size: Total size in bytes
        - file_count: Number of files
        - dir_count: Number of directories
        - largest_files: List of (path, size) tuples for top 10 files

    Examples:
        >>> stats = get_directory_statistics("src")
        >>> print(f"Total size: {stats['total_size']} bytes")
        >>> print(f"Files: {stats['file_count']}, Dirs: {stats['dir_count']}")
    """
    if not RUST_AVAILABLE:
        raise FileOpsError(
            "Rust extension not available. Install with: pip install victor-ai[native]"
        )

    root_str = str(root)

    if not os.path.exists(root_str):
        raise FileOpsError(f"Root directory does not exist: {root_str}")

    try:
        return get_directory_stats(root_str, max_depth=max_depth)
    except Exception as e:
        raise FileOpsError(f"Error getting directory stats: {e}") from e


def group_files_by_directory(files: List[FileInfo]) -> Dict[str, List[FileInfo]]:
    """
    Group files by their parent directory.

    Args:
        files: List of FileInfo objects to group

    Returns:
        Dictionary mapping directory paths to lists of FileInfo objects

    Examples:
        >>> files = walk_directory("src", patterns=["*.py"])
        >>> grouped = group_files_by_directory(files)
        >>> for dir_path, dir_files in grouped.items():
        ...     print(f"{dir_path}: {len(dir_files)} files")
    """
    if not RUST_AVAILABLE:
        raise FileOpsError(
            "Rust extension not available. Install with: pip install victor-ai[native]"
        )

    try:
        return group_by_directory(files)
    except Exception as e:
        raise FileOpsError(f"Error grouping by directory: {e}") from e


def filter_files_by_modified_time(
    files: List[FileInfo], since: int, until: int = 0
) -> List[FileInfo]:
    """
    Filter files by modification time.

    Args:
        files: List of FileInfo objects to filter
        since: Unix timestamp for earliest modification time
        until: Unix timestamp for latest modification time (0 = now)

    Returns:
        Filtered list of FileInfo objects modified in the time range

    Examples:
        >>> import time
        >>> # Find files modified in the last 24 hours
        >>> one_day_ago = int(time.time()) - 86400
        >>> recent = filter_files_by_modified_time(files, since=one_day_ago)
    """
    if not RUST_AVAILABLE:
        raise FileOpsError(
            "Rust extension not available. Install with: pip install victor-ai[native]"
        )

    try:
        return filter_by_modified_time(files, since=since, until=until)
    except Exception as e:
        raise FileOpsError(f"Error filtering by modified time: {e}") from e


def find_code_files(
    root: Union[str, Path],
    extensions: Optional[List[str]] = None,
    ignore_dirs: Optional[List[str]] = None,
    max_depth: int = 100,
) -> List[FileInfo]:
    """
    Convenience function to find code files in a directory.

    Common code file extensions are included by default.

    Args:
        root: Root directory to search
        extensions: List of file extensions (default: common code extensions)
        ignore_dirs: Directories to ignore (default: common ignore patterns)
        max_depth: Maximum traversal depth

    Returns:
        List of FileInfo objects for code files

    Examples:
        >>> # Find all code files with default extensions
        >>> code_files = find_code_files("src")
        >>>
        >>> # Find only Python and Rust files
        >>> code_files = find_code_files("src", extensions=["py", "rs"])
    """
    if extensions is None:
        extensions = [
            "py",
            "rs",
            "js",
            "ts",
            "tsx",
            "java",
            "go",
            "c",
            "cpp",
            "h",
            "hpp",
            "cs",
            "rb",
            "php",
            "swift",
            "kt",
            "scala",
            "sh",
            "yaml",
            "yml",
            "toml",
            "json",
        ]

    if ignore_dirs is None:
        ignore_dirs = [
            "__pycache__",
            ".git",
            ".svn",
            "node_modules",
            ".venv",
            "venv",
            ".tox",
            "dist",
            "build",
            "*.egg-info",
            ".pytest_cache",
            ".mypy_cache",
            "target",
            "bin",
            "obj",
        ]

    # Build patterns from extensions
    patterns = [f"*.{ext}" for ext in extensions]

    # Walk directory
    files = walk_directory(
        root,
        patterns=patterns,
        max_depth=max_depth,
        ignore_patterns=ignore_dirs,
    )

    # Filter to only files (not directories)
    return [f for f in files if f.file_type == "file"]


__all__ = [
    "FileInfo",
    "FileMetadata",
    "walk_directory",
    "get_file_metadata",
    "filter_files_by_extension",
    "filter_files_by_size",
    "get_directory_statistics",
    "group_files_by_directory",
    "filter_files_by_modified_time",
    "find_code_files",
    "FileOpsError",
    "RUST_AVAILABLE",
]
