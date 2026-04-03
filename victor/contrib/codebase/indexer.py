"""Null codebase index factory stub.

Provides a factory that raises informative errors when codebase indexing
is not available. Enhanced by victor-coding when installed.
"""

from __future__ import annotations

from typing import Any


class NullCodebaseIndexFactory:
    """Stub codebase index factory that raises on create()."""

    def create(self, root_path: str, **kwargs: Any) -> Any:
        """Raise ImportError — indexing not available without victor-coding."""
        raise ImportError(
            "Codebase indexing requires victor-coding package. "
            "Install with: pip install victor-coding"
        )
