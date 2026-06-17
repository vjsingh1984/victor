"""Helpers for locating extracted vertical repositories near the core repo."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

DEFAULT_RELATIVE_EXTRACTED_REPO_PATHS = (
    "../victor-coding",
    "../victor-research",
    "../victor-devops",
)


def normalize_extracted_repo_paths(
    paths: Iterable[str | Path],
    *,
    cwd: Path,
) -> list[Path]:
    """Return de-duplicated absolute paths preserving input order."""
    normalized: list[Path] = []
    seen: set[Path] = set()

    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (cwd / path).resolve()
        else:
            path = path.resolve()

        if path in seen:
            continue
        seen.add(path)
        normalized.append(path)

    return normalized


def discover_default_extracted_repo_paths(*, repo_root: Path) -> list[Path]:
    """Discover existing extracted vertical repos next to the core repo."""
    return [
        path
        for path in normalize_extracted_repo_paths(
            DEFAULT_RELATIVE_EXTRACTED_REPO_PATHS,
            cwd=repo_root,
        )
        if path.exists() and path.is_dir()
    ]
