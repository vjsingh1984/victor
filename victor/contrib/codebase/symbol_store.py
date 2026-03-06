"""Null symbol store factory stub.

Provides a factory that raises informative errors when symbol analysis
is not available. Enhanced by victor-coding when installed.
"""

from __future__ import annotations

from typing import Any, List, Optional


class NullSymbolStore:
    """Stub symbol store factory that raises on create()."""

    def create(
        self,
        root: str,
        include_dirs: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None,
    ) -> Any:
        """Raise ImportError — symbol analysis not available without victor-coding."""
        raise ImportError(
            "Symbol store requires victor-coding package. "
            "Install with: pip install victor-coding"
        )
