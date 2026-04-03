"""Null tree-sitter parser stub.

Returns None/empty results when tree-sitter is not available.
Enhanced by victor-coding when installed.
"""

from __future__ import annotations

from typing import Any, List


class NullTreeSitterParser:
    """Stub tree-sitter parser that returns None for all languages."""

    def get_parser(self, language: str) -> Any:
        """Return None — no parser available without tree-sitter."""
        return None

    def get_supported_languages(self) -> List[str]:
        """Return empty list — no languages supported without tree-sitter."""
        return []
