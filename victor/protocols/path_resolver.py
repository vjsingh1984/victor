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

"""Centralized Path Resolution.

This module provides unified path normalization and resolution, replacing
per-tool path handling scattered throughout the codebase.

Design:
- Normalization pipeline with composable normalizers
- Fuzzy matching for similar path suggestions
- Cached resolution for performance
- Clear separation of file vs directory resolution

Common Issues Fixed:
- LLMs using paths like "project/utils" when cwd is already "project"
- Relative paths with incorrect depth
- Mixed path separators
- Trailing slashes on files

Usage:
    resolver = PathResolver()
    result = resolver.resolve_file("investor_homelab/models/news.py")

    if result.was_normalized:
        logger.info(f"Path normalized: {result.original_path} -> {result.resolved_path}")

    # Fuzzy matching for suggestions
    suggestions = resolver.suggest_similar("modls/news.py")  # Typo
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable

logger = logging.getLogger(__name__)


# =============================================================================
# Path Resolution Result
# =============================================================================


@dataclass
class PathResolution:
    """Result of path resolution.

    Attributes:
        original_path: The path as provided by caller
        resolved_path: The normalized, resolved Path object
        was_normalized: True if any normalization was applied
        normalization_applied: Description of normalization (if any)
        exists: Whether the resolved path exists
        is_file: True if resolved path is a file
        is_directory: True if resolved path is a directory
    """
    original_path: str
    resolved_path: Path
    was_normalized: bool = False
    normalization_applied: Optional[str] = None
    exists: bool = False
    is_file: bool = False
    is_directory: bool = False

    @property
    def path_str(self) -> str:
        """Get resolved path as string."""
        return str(self.resolved_path)

    def __str__(self) -> str:
        """Human-readable representation."""
        if self.was_normalized:
            return f"'{self.original_path}' -> '{self.resolved_path}' ({self.normalization_applied})"
        return str(self.resolved_path)


# =============================================================================
# Path Resolver Protocol
# =============================================================================


@runtime_checkable
class IPathResolver(Protocol):
    """Protocol for path resolution.

    Provides centralized path normalization and resolution
    for all filesystem operations.
    """

    def resolve(self, path: str, must_exist: bool = True) -> PathResolution:
        """Resolve a path with normalization.

        Args:
            path: Path to resolve (relative or absolute)
            must_exist: If True, raises error if path doesn't exist

        Returns:
            PathResolution with resolved path and metadata
        """
        ...

    def resolve_file(self, path: str) -> PathResolution:
        """Resolve a file path.

        Args:
            path: File path to resolve

        Returns:
            PathResolution, raises if not a file
        """
        ...

    def resolve_directory(self, path: str) -> PathResolution:
        """Resolve a directory path.

        Args:
            path: Directory path to resolve

        Returns:
            PathResolution, raises if not a directory
        """
        ...

    def suggest_similar(self, path: str, limit: int = 5) -> List[str]:
        """Suggest similar paths that exist.

        Useful for providing helpful error messages when a path doesn't exist.

        Args:
            path: Non-existent path to find matches for
            limit: Maximum suggestions to return

        Returns:
            List of similar existing paths
        """
        ...


# =============================================================================
# Normalizer Functions
# =============================================================================


def strip_cwd_prefix(path: str, cwd: Path) -> Tuple[Optional[str], str]:
    """Strip redundant cwd prefix from path.

    When working in a subdirectory, LLMs sometimes include the directory name
    in paths (e.g., "project/utils" when cwd is already "project/").

    Args:
        path: Path to normalize
        cwd: Current working directory

    Returns:
        Tuple of (normalized_path or None, description)
    """
    if not path or path.startswith("/") or path.startswith("~"):
        return None, ""

    cwd_name = cwd.name

    # Check if path starts with cwd name
    if path.startswith(f"{cwd_name}/"):
        stripped = path[len(cwd_name) + 1:]
        if stripped:
            return stripped, f"stripped_cwd_prefix:{cwd_name}"

    return None, ""


def strip_first_component(path: str, cwd: Path) -> Tuple[Optional[str], str]:
    """Try stripping first path component if it matches a cwd component.

    Args:
        path: Path to normalize
        cwd: Current working directory

    Returns:
        Tuple of (normalized_path or None, description)
    """
    if not path or "/" not in path:
        return None, ""

    parts = path.split("/", 1)
    if len(parts) != 2:
        return None, ""

    first_component = parts[0]
    remainder = parts[1]

    # Check if first component is any part of cwd
    cwd_parts = set(cwd.parts)
    if first_component in cwd_parts:
        # Verify the remainder exists
        check_path = cwd / remainder
        if check_path.exists():
            return remainder, f"stripped_component:{first_component}"

    return None, ""


def strip_common_prefix(path: str, cwd: Path) -> Tuple[Optional[str], str]:
    """Strip common prefix if path has nested duplicate.

    Handles cases like "src/components/src/components/Button.tsx"
    where the path was accidentally duplicated.

    Args:
        path: Path to normalize
        cwd: Current working directory

    Returns:
        Tuple of (normalized_path or None, description)
    """
    if "/" not in path:
        return None, ""

    parts = path.split("/")
    n = len(parts)

    # Look for repeated sequences
    for seq_len in range(1, n // 2 + 1):
        prefix = parts[:seq_len]
        if parts[seq_len:seq_len + seq_len] == prefix:
            # Found duplicate prefix
            stripped = "/".join(parts[seq_len:])
            check_path = cwd / stripped
            if check_path.exists():
                return stripped, f"stripped_duplicate:{'/'.join(prefix)}"

    return None, ""


def normalize_separators(path: str, cwd: Path) -> Tuple[Optional[str], str]:
    """Normalize path separators and trailing slashes.

    Args:
        path: Path to normalize
        cwd: Current working directory

    Returns:
        Tuple of (normalized_path or None, description)
    """
    # Replace backslashes with forward slashes (Windows paths)
    if "\\" in path:
        normalized = path.replace("\\", "/")
        return normalized, "normalized_separators"

    # Remove trailing slashes (except for root)
    if path.endswith("/") and len(path) > 1:
        normalized = path.rstrip("/")
        return normalized, "stripped_trailing_slash"

    return None, ""


# =============================================================================
# Path Resolver Implementation
# =============================================================================


@dataclass
class PathResolver(IPathResolver):
    """Centralized path resolution with normalization pipeline.

    Applies a series of normalizers to resolve paths that may have
    common issues (redundant prefixes, wrong separators, etc.).

    Attributes:
        cwd: Base directory for resolution (defaults to os.getcwd())
        normalizers: List of normalizer functions to apply
        _cache: Cache for resolved paths
    """

    cwd: Optional[Path] = None
    normalizers: List[Callable[[str, Path], Tuple[Optional[str], str]]] = field(
        default_factory=list
    )
    _cache: Dict[str, PathResolution] = field(default_factory=dict)
    _known_paths: Optional[Set[str]] = None  # For fuzzy matching

    def __post_init__(self) -> None:
        """Initialize with defaults if not provided."""
        if self.cwd is None:
            self.cwd = Path.cwd()
        else:
            self.cwd = Path(self.cwd).resolve()

        if not self.normalizers:
            self.normalizers = [
                normalize_separators,
                strip_cwd_prefix,
                strip_first_component,
                strip_common_prefix,
            ]

    def _apply_normalizers(self, path: str) -> Tuple[str, Optional[str]]:
        """Apply normalizers in sequence until one succeeds.

        Args:
            path: Path to normalize

        Returns:
            Tuple of (normalized_path, description or None)
        """
        current_path = path
        applied_normalizations: List[str] = []

        for normalizer in self.normalizers:
            result, description = normalizer(current_path, self.cwd)
            if result is not None:
                current_path = result
                applied_normalizations.append(description)

        if applied_normalizations:
            return current_path, ", ".join(applied_normalizations)
        return path, None

    def resolve(self, path: str, must_exist: bool = True) -> PathResolution:
        """Resolve a path with normalization.

        Tries the path directly first, then applies normalizers
        to find a valid path.

        Args:
            path: Path to resolve
            must_exist: If True, raises FileNotFoundError if not found

        Returns:
            PathResolution with resolved path

        Raises:
            FileNotFoundError: If must_exist and path not found
        """
        # Check cache
        cache_key = f"{path}:{must_exist}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Handle empty path
        if not path:
            result = PathResolution(
                original_path=path,
                resolved_path=self.cwd,
                exists=True,
                is_directory=True,
            )
            self._cache[cache_key] = result
            return result

        # Try original path first
        try:
            resolved = Path(path).expanduser()
            if not resolved.is_absolute():
                resolved = (self.cwd / resolved).resolve()
            else:
                resolved = resolved.resolve()

            if resolved.exists():
                result = PathResolution(
                    original_path=path,
                    resolved_path=resolved,
                    exists=True,
                    is_file=resolved.is_file(),
                    is_directory=resolved.is_dir(),
                )
                self._cache[cache_key] = result
                return result
        except (OSError, ValueError):
            pass  # Continue to normalization

        # Apply normalizers
        normalized, description = self._apply_normalizers(path)

        if normalized != path:
            try:
                resolved = (self.cwd / normalized).resolve()
                if resolved.exists():
                    result = PathResolution(
                        original_path=path,
                        resolved_path=resolved,
                        was_normalized=True,
                        normalization_applied=description,
                        exists=True,
                        is_file=resolved.is_file(),
                        is_directory=resolved.is_dir(),
                    )
                    logger.debug(f"Path resolved: {result}")
                    self._cache[cache_key] = result
                    return result
            except (OSError, ValueError):
                pass

        # Path not found
        if must_exist:
            # Provide helpful error with suggestions
            suggestions = self.suggest_similar(path, limit=3)
            suggestion_text = ""
            if suggestions:
                suggestion_text = f"\nDid you mean: {', '.join(suggestions)}"
            raise FileNotFoundError(f"Path not found: {path}{suggestion_text}")

        # Return non-existent path resolution
        try:
            resolved = Path(path).expanduser()
            if not resolved.is_absolute():
                resolved = (self.cwd / resolved).resolve()
        except (OSError, ValueError):
            resolved = self.cwd / path

        result = PathResolution(
            original_path=path,
            resolved_path=resolved,
            exists=False,
        )
        self._cache[cache_key] = result
        return result

    def resolve_file(self, path: str) -> PathResolution:
        """Resolve a file path.

        Args:
            path: File path to resolve

        Returns:
            PathResolution for an existing file

        Raises:
            FileNotFoundError: If file doesn't exist
            IsADirectoryError: If path is a directory
        """
        result = self.resolve(path, must_exist=True)

        if result.is_directory:
            raise IsADirectoryError(
                f"Cannot read directory as file: {path}\n"
                f"Suggestion: Use list_directory(path='{path}') to explore its contents."
            )

        return result

    def resolve_directory(self, path: str) -> PathResolution:
        """Resolve a directory path.

        Args:
            path: Directory path to resolve

        Returns:
            PathResolution for an existing directory

        Raises:
            FileNotFoundError: If directory doesn't exist
            NotADirectoryError: If path is a file
        """
        result = self.resolve(path, must_exist=True)

        if result.is_file:
            raise NotADirectoryError(
                f"Path is not a directory: {path}\n"
                f"Suggestion: Use read_file(path='{path}') to read file contents."
            )

        return result

    def suggest_similar(self, path: str, limit: int = 5) -> List[str]:
        """Suggest similar paths that exist.

        Uses difflib for fuzzy matching against known paths in cwd.

        Args:
            path: Non-existent path to find matches for
            limit: Maximum suggestions to return

        Returns:
            List of similar existing paths
        """
        # Build known paths if not cached
        if self._known_paths is None:
            self._known_paths = self._scan_directory_names()

        # Get filename component for matching
        path_obj = Path(path)
        filename = path_obj.name

        # Find close matches
        matches = get_close_matches(
            filename,
            list(self._known_paths),
            n=limit,
            cutoff=0.5,
        )

        # If the path has multiple components, try matching against full paths
        if "/" in path or "\\" in path:
            # Also try matching the full path
            full_paths = self._get_relative_file_paths()
            full_matches = get_close_matches(
                path,
                full_paths,
                n=limit,
                cutoff=0.4,
            )
            matches = list(dict.fromkeys(full_matches + matches))[:limit]

        return matches

    def _scan_directory_names(self, max_depth: int = 3) -> Set[str]:
        """Scan directory for file/directory names.

        Args:
            max_depth: Maximum directory depth to scan

        Returns:
            Set of file and directory names
        """
        names: Set[str] = set()

        def _scan(path: Path, depth: int) -> None:
            if depth > max_depth:
                return
            try:
                for entry in path.iterdir():
                    if entry.name.startswith("."):
                        continue
                    names.add(entry.name)
                    if entry.is_dir():
                        _scan(entry, depth + 1)
            except PermissionError:
                pass

        _scan(self.cwd, 0)
        return names

    def _get_relative_file_paths(self, max_depth: int = 4) -> List[str]:
        """Get relative file paths from cwd.

        Args:
            max_depth: Maximum depth to scan

        Returns:
            List of relative file paths
        """
        paths: List[str] = []

        def _scan(path: Path, depth: int, prefix: str) -> None:
            if depth > max_depth:
                return
            try:
                for entry in path.iterdir():
                    if entry.name.startswith("."):
                        continue
                    rel_path = f"{prefix}/{entry.name}" if prefix else entry.name
                    if entry.is_file():
                        paths.append(rel_path)
                    elif entry.is_dir():
                        paths.append(rel_path + "/")
                        _scan(entry, depth + 1, rel_path)
            except PermissionError:
                pass

        _scan(self.cwd, 0, "")
        return paths

    def set_cwd(self, cwd: Path) -> None:
        """Update the base directory.

        Clears all caches when cwd changes.

        Args:
            cwd: New working directory
        """
        self.cwd = Path(cwd).resolve()
        self._cache.clear()
        self._known_paths = None

    def clear_cache(self) -> None:
        """Clear all cached resolutions."""
        self._cache.clear()
        self._known_paths = None


# =============================================================================
# Factory Function
# =============================================================================


def create_path_resolver(cwd: Optional[Path] = None) -> PathResolver:
    """Create a configured PathResolver instance.

    Factory function for DI registration.

    Args:
        cwd: Base directory (defaults to os.getcwd())

    Returns:
        Configured PathResolver instance
    """
    return PathResolver(cwd=cwd)
