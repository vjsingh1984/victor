"""Basic ignore patterns for codebase analysis.

Provides default directory/file filtering when victor-coding is not installed.
Enhanced by victor-coding when installed (more comprehensive patterns).
"""

from __future__ import annotations

from pathlib import Path
from typing import FrozenSet, List, Optional


DEFAULT_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "*.pyc",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        "dist",
        "build",
        "*.egg-info",
    }
)


class BasicIgnorePatterns:
    """Basic ignore patterns for codebase analysis."""

    def get_default_skip_dirs(self) -> FrozenSet[str]:
        """Get the default set of directories to skip."""
        return DEFAULT_SKIP_DIRS

    def is_hidden_path(self, path: str) -> bool:
        """Check if a path is hidden (starts with dot)."""
        return Path(path).name.startswith(".")

    def should_ignore_path(
        self,
        path: str,
        skip_dirs: Optional[FrozenSet[str]] = None,
        extra_patterns: Optional[List[str]] = None,
    ) -> bool:
        """Check if a path should be ignored."""
        p = Path(path)
        if p.name.startswith("."):
            return True
        dirs = skip_dirs if skip_dirs is not None else DEFAULT_SKIP_DIRS
        if extra_patterns:
            extra = frozenset(extra_patterns)
            if any(part in extra for part in p.parts):
                return True
        return any(skip_dir in p.parts for skip_dir in dirs)
