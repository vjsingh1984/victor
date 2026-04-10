"""Settings access — isolates victor.config.settings imports.

External verticals should use these helpers instead of importing
get_project_paths/load_settings directly from victor.config.settings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Context file name (default: init.md)
VICTOR_CONTEXT_FILE: str = "init.md"


def get_project_paths(project_root: Optional[str] = None) -> Any:
    """Get project paths via victor's settings, or return a minimal fallback.

    Isolates the victor.config.settings import to this single module.
    """
    try:
        from victor.config.settings import get_project_paths as _get_paths

        return _get_paths(project_root)
    except ImportError:
        logger.debug("victor-ai not installed — using minimal project paths")
        root = Path(project_root) if project_root else Path.cwd()
        return _MinimalPaths(root)


def load_settings() -> Any:
    """Load settings via victor, or return a minimal dict fallback."""
    try:
        from victor.config.settings import load_settings as _load

        return _load()
    except ImportError:
        logger.debug("victor-ai not installed — using default settings")
        return {}


# Try to import the real context file name
try:
    from victor.config.settings import VICTOR_CONTEXT_FILE as _VCF

    VICTOR_CONTEXT_FILE = _VCF
except ImportError:
    pass


class _MinimalPaths:
    """Minimal ProjectPaths fallback when victor-ai is not installed."""

    def __init__(self, root: Path):
        self._root = root

    @property
    def project_root(self) -> Path:
        return self._root

    @property
    def victor_dir(self) -> Path:
        return self._root / ".victor"

    @property
    def embeddings_dir(self) -> Path:
        return self.victor_dir / "embeddings"

    @property
    def graph_dir(self) -> Path:
        return self.victor_dir / "graph"

    @property
    def logs_dir(self) -> Path:
        return self.victor_dir / "logs"

    @property
    def sessions_dir(self) -> Path:
        return self.victor_dir / "sessions"

    @property
    def backups_dir(self) -> Path:
        return self.victor_dir / "backups"

    @property
    def conversation_db(self) -> Path:
        return self.victor_dir / "conversation.db"

    @property
    def global_embeddings_dir(self) -> Path:
        return Path.home() / ".victor" / "embeddings"
