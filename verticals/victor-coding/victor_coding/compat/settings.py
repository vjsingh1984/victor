"""Settings access - contract-first with framework fallback.

External verticals should use these helpers instead of importing
directly from victor.config.settings. Resolution order:

1. victor_contracts.verticals.protocols.config (contract protocol - no framework dependency)
2. victor.config.settings (framework — deferred import)
3. _MinimalPaths fallback (standalone — no victor-ai at all)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Context file name (default: init.md)
VICTOR_CONTEXT_FILE: str = "init.md"


def get_project_paths(project_root: Optional[str] = None) -> Any:
    """Get project paths via contract protocol, framework, or minimal fallback."""
    root = project_root or str(Path.cwd())

    # 1. Contract protocol (preferred - no framework coupling)
    try:
        from victor_contracts.verticals.protocols.config import ProjectPathsData

        return ProjectPathsData(project_root=root)
    except ImportError:
        pass

    # 2. Framework (deferred import)
    try:
        from victor.config.settings import get_project_paths as _get_paths

        return _get_paths(project_root)
    except ImportError:
        pass

    # 3. Minimal fallback
    logger.debug("victor-ai/victor-contracts not installed — using minimal project paths")
    return _MinimalPaths(Path(root))


def load_settings() -> Any:
    """Load settings via framework, or return empty dict fallback."""
    try:
        from victor.config.settings import load_settings as _load

        return _load()
    except ImportError:
        logger.debug("victor-ai not installed — using default settings")
        return {}


# Try to import the real context file name from SDK then framework
try:
    from victor_contracts.verticals.protocols.config import ProjectPathsData as _PPD

    VICTOR_CONTEXT_FILE = _PPD.context_file_name  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    try:
        from victor.config.settings import VICTOR_CONTEXT_FILE as _VCF

        VICTOR_CONTEXT_FILE = _VCF
    except ImportError:
        pass


class _MinimalPaths:
    """Minimal ProjectPaths fallback when neither SDK nor framework is installed."""

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
    def project_db(self) -> Path:
        return self.victor_dir / "project.db"

    @property
    def global_embeddings_dir(self) -> Path:
        return Path.home() / ".victor" / "embeddings"
