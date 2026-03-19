"""Null tree-sitter extractor stub.

Returns empty results when tree-sitter is not available.
Enhanced by victor-coding when installed.
"""

from __future__ import annotations

from typing import Any, Dict, List


class NullTreeSitterExtractor:
    """Stub tree-sitter extractor that returns empty symbol lists."""

    def extract_symbols(self, content: str, language: str) -> List[Dict[str, Any]]:
        """Return empty list — no extraction without tree-sitter."""
        return []
