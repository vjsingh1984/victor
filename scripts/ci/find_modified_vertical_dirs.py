#!/usr/bin/env python3
"""Find changed directories that belong to a Victor vertical package.

This script reads newline-delimited repo-relative paths from stdin and emits a
space-separated list of unique directories that contain a `victor-vertical.toml`
file. It lets GitHub Actions validate an entire vertical whenever any file in
that vertical changes.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path.cwd().resolve()
MARKER = "victor-vertical.toml"


def find_vertical_dirs(lines: list[str]) -> list[str]:
    """Return sorted vertical directories for the provided changed paths."""
    directories: set[str] = set()

    for raw_path in lines:
        rel_path = raw_path.strip()
        if not rel_path:
            continue

        candidate = (ROOT / rel_path).resolve()
        current = candidate if candidate.is_dir() else candidate.parent

        while current == ROOT or ROOT in current.parents:
            if (current / MARKER).is_file():
                directories.add(current.relative_to(ROOT).as_posix())
                break
            if current == ROOT:
                break
            current = current.parent

    return sorted(directories)


def main() -> int:
    vertical_dirs = find_vertical_dirs(sys.stdin.read().splitlines())
    sys.stdout.write(" ".join(vertical_dirs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
