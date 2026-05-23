"""Null tree-sitter analysis provider.

Provides the analysis-level capability shape used by root framework code while
enhanced tree-sitter analysis is unavailable. The victor-coding package can
replace this stub through the capability registry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class NullTreeSitterAnalysis:
    """Stub analysis provider that reports no supported languages."""

    def supports_language(self, language: str) -> bool:
        """Return False because no enhanced analysis is installed."""
        return False

    def parse(
        self,
        content: bytes,
        language: str,
        *,
        file_path: Optional[str] = None,
    ) -> Any:
        """Return None because parsing is unavailable."""
        return None

    def extract_symbols(
        self,
        content: bytes,
        language: str,
        *,
        file_path: str,
    ) -> List[Dict[str, Any]]:
        """Return no symbols."""
        return []

    def extract_edges(
        self,
        content: bytes,
        language: str,
        *,
        file_path: str,
    ) -> List[Dict[str, Any]]:
        """Return no relationship edges."""
        return []

    def extract_imports(
        self,
        content: bytes,
        language: str,
        *,
        file_path: Optional[str] = None,
    ) -> List[str]:
        """Return no imports."""
        return []

    def build_chunk_context(
        self,
        content: str,
        language: str,
        *,
        file_path: Optional[str] = None,
    ) -> Any:
        """Return None because structural chunk context is unavailable."""
        return None
