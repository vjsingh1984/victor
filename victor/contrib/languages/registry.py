"""Null language registry stub.

Returns empty results when language plugins are not available.
Enhanced by victor-coding when installed.
"""

from __future__ import annotations

from typing import Any, List


class NullLanguageRegistry:
    """Stub language registry with no plugins."""

    def discover_plugins(self) -> int:
        """Return 0 — no plugins available."""
        return 0

    def get(self, language: str) -> Any:
        """Raise KeyError — no language plugins available."""
        raise KeyError(
            f"Language '{language}' not available (victor-coding not installed)"
        )

    def get_supported_languages(self) -> List[str]:
        """Return empty list."""
        return []
