"""Shared project-instruction discovery for Victor runtimes.

This module keeps the workspace instruction import behavior explicit and
consistent across the legacy agent runtime and the framework prompt builder.

Discovery rules:
- Walk from the current workspace directory upward toward the filesystem root.
- Prefer nearer directories over farther ancestors.
- Within a directory, use a deterministic filename precedence.
- Deduplicate files with identical content so the same instructions are not
  rebroadcast under multiple compatibility filenames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

DEFAULT_INSTRUCTION_FILENAMES: tuple[str, ...] = (
    ".victor/init.md",
    ".victor/instructions.md",
    "AGENTS.local.md",
    "AGENTS.md",
    ".claude/CLAUDE.local.md",
    ".claude/CLAUDE.md",
    "CLAUDE.local.md",
    "CLAUDE.md",
    ".github/copilot-instructions.md",
    ".cursor/rules",
    ".victor.md",
)


@dataclass(frozen=True)
class InstructionFile:
    """A discovered instruction file and its scope metadata."""

    path: Path
    content: str
    scope: str
    source_type: str
    mtime: float
    size: int


def discover_instruction_files(
    cwd: Path,
    *,
    filenames: Sequence[str] = DEFAULT_INSTRUCTION_FILENAMES,
) -> List[InstructionFile]:
    """Discover instruction files visible from *cwd*.

    Args:
        cwd: Workspace directory to start searching from.
        filenames: Ordered candidate filenames to check in each directory.

    Returns:
        Ordered list of discovered instruction files.
    """
    resolved_cwd = cwd.resolve()
    current = resolved_cwd
    root = Path(current.anchor)
    git_root = _find_git_root(resolved_cwd)
    seen_contents: set[str] = set()
    results: List[InstructionFile] = []

    while True:
        for name in filenames:
            candidate = current / name
            if not candidate.is_file():
                continue

            try:
                content = candidate.read_text(encoding="utf-8")
                stat = candidate.stat()
            except OSError:
                continue

            if content in seen_contents:
                continue
            seen_contents.add(content)

            results.append(
                InstructionFile(
                    path=candidate,
                    content=content,
                    scope=_classify_scope(candidate.parent, resolved_cwd, git_root),
                    source_type=_classify_source_type(candidate),
                    mtime=stat.st_mtime,
                    size=stat.st_size,
                )
            )

        if current == root:
            break
        current = current.parent

    return results


def build_instruction_signature(files: Iterable[InstructionFile]) -> tuple[tuple[str, float, int], ...]:
    """Return a stable signature for prompt invalidation and cache keys."""
    return tuple((str(item.path), item.mtime, item.size) for item in files)


def _find_git_root(start: Path) -> Path | None:
    """Return the nearest git root containing *start*, if any."""
    current = start
    root = Path(current.anchor)
    while True:
        if (current / ".git").exists():
            return current
        if current == root:
            return None
        current = current.parent


def _classify_scope(file_dir: Path, workspace_dir: Path, git_root: Path | None) -> str:
    """Classify file scope for prompt rendering."""
    if file_dir == workspace_dir:
        return "workspace"
    if git_root is not None:
        try:
            file_dir.relative_to(git_root)
            return "project"
        except ValueError:
            pass
    return "user"


def _classify_source_type(path: Path) -> str:
    """Return the compatibility surface represented by *path*."""
    path_str = str(path).lower()
    name = path.name.lower()
    if name == "agents.md" or name == "agents.local.md":
        return "agents"
    if "claude" in name:
        return "claude"
    if path_str.endswith(".victor/init.md"):
        return "victor_init"
    if path_str.endswith(".victor/instructions.md"):
        return "victor_instructions"
    if name == ".victor.md":
        return "victor_legacy"
    if "copilot" in name:
        return "copilot"
    if ".cursor/" in path_str:
        return "cursor"
    return "generic"
