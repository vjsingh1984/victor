from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import FrozenSet, Optional

_VICTOR_DIR = ".victor"


class WritePathTier(IntEnum):
    OS_TEMP = 0  # /tmp/** (macOS: /private/tmp/**)
    LOCAL_ANALYSIS = 1  # .victor/analysis/**, ./tmp/**
    DOCS_SAFE = 2  # ./docs/** (extension-filtered)
    PREFIXED_ROOT = 3  # ./ANALYSIS-*.md, ./CHECKPOINT-*.md at repo root
    SOURCE_CODE = 4  # everything else — blocked in analysis mode


@dataclass(frozen=True)
class WritePathPolicy:
    """Immutable policy controlling which paths are writable under a given task mode.

    Use factory presets rather than constructing directly:
      WritePathPolicy.read_only()       — blocks all writes
      WritePathPolicy.analysis_safe()   — allows .victor/analysis/, ./tmp/, ./docs/*.md
      WritePathPolicy.full_access()     — allows all paths (equivalent to write_allowed=True)
    """

    max_tier: WritePathTier = WritePathTier.SOURCE_CODE
    docs_extensions: FrozenSet[str] = frozenset({".md", ".txt", ".adoc", ".rst"})
    _read_only: bool = False  # Internal flag for read-only mode

    def allows(self, file_path: Path) -> bool:
        """Return True if file_path is writable under this policy."""
        # Check read-only flag first
        if self._read_only:
            return False

        tier = self._classify(file_path)
        if tier > self.max_tier:
            return False
        if tier == WritePathTier.DOCS_SAFE:
            return file_path.suffix.lower() in self.docs_extensions
        return True

    def _classify(self, file_path: Path) -> WritePathTier:
        resolved = file_path.resolve()

        # Resolve /tmp — on macOS it's a symlink to /private/tmp
        try:
            resolved.relative_to(Path("/tmp").resolve())
            return WritePathTier.OS_TEMP
        except ValueError:
            pass

        cwd = Path.cwd().resolve()

        # .victor/analysis/ and ./tmp/ are LOCAL_ANALYSIS (safe analysis dirs)
        for rel_root in (f"{_VICTOR_DIR}/analysis", "tmp"):
            try:
                resolved.relative_to(cwd / rel_root)
                return WritePathTier.LOCAL_ANALYSIS
            except ValueError:
                pass

        try:
            resolved.relative_to(cwd / "docs")
            return WritePathTier.DOCS_SAFE
        except ValueError:
            pass

        if resolved.parent == cwd and re.match(r"^(ANALYSIS|CHECKPOINT)-.*\.md$", resolved.name):
            return WritePathTier.PREFIXED_ROOT

        return WritePathTier.SOURCE_CODE

    @classmethod
    def read_only(cls) -> "WritePathPolicy":
        """Block all writes. Backward-compat equivalent of write_allowed=False."""
        return cls(max_tier=WritePathTier.SOURCE_CODE, _read_only=True)

    @classmethod
    def analysis_safe(cls) -> "WritePathPolicy":
        """Allow OS temp, .victor/analysis/, ./tmp/, and docs/*.md — but not source code."""
        return cls(max_tier=WritePathTier.DOCS_SAFE)

    @classmethod
    def full_access(cls) -> "WritePathPolicy":
        """Allow all paths. Backward-compat equivalent of write_allowed=True."""
        return cls(max_tier=WritePathTier.SOURCE_CODE)


_active_policy: Optional[WritePathPolicy] = None


def get_active_write_policy() -> Optional[WritePathPolicy]:
    """Return the session-active write policy, or None if unset (legacy sandbox applies)."""
    return _active_policy


def set_active_write_policy(policy: Optional[WritePathPolicy]) -> None:
    """Activate a write policy for the current session. Called by IsolationMapper."""
    global _active_policy
    _active_policy = policy
