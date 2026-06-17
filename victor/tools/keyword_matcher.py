"""Keyword matcher extracted from MetadataRegistry.

Indexes tool keywords and matches user messages to tools.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


class KeywordMatcher:
    """Match user messages to tools via keyword indexing.

    Extracted from MetadataRegistry (tool_registry.py) for
    independent testing and reuse.
    """

    def __init__(self):
        self._tool_keywords: dict[str, list[str]] = {}
        self._mandatory_keywords: dict[str, list[str]] = {}
        self._keyword_index: dict[str, set[str]] = defaultdict(set)

    def index_tool(
        self,
        name: str,
        keywords: list[str],
        mandatory_keywords: Optional[list[str]] = None,
    ) -> None:
        """Index a tool with its keywords.

        Args:
            name: Tool name.
            keywords: Keywords that trigger this tool.
            mandatory_keywords: Keywords that MUST trigger this tool.
        """
        self._tool_keywords[name] = [k.lower() for k in keywords]
        if mandatory_keywords:
            self._mandatory_keywords[name] = [k.lower() for k in mandatory_keywords]

        for kw in keywords:
            kw_lower = kw.lower()
            self._keyword_index[kw_lower].add(name)

    def match(self, message: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Match a user message to tools by keyword overlap.

        Args:
            message: User message text.
            top_k: Maximum number of results.

        Returns:
            List of (tool_name, score) tuples, sorted by score descending.
        """
        message_lower = message.lower()
        message_words = set(re.findall(r"\w+", message_lower))

        scores: dict[str, float] = defaultdict(float)

        for tool_name, keywords in self._tool_keywords.items():
            for kw in keywords:
                kw_words = set(kw.split())
                if kw_words.issubset(message_words) or kw in message_lower:
                    scores[tool_name] += 1.0

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_mandatory_matches(self, message: str) -> list[str]:
        """Get tools whose mandatory keywords appear in the message.

        Args:
            message: User message text.

        Returns:
            List of tool names that MUST be suggested.
        """
        message_lower = message.lower()
        matches = []
        for tool_name, keywords in self._mandatory_keywords.items():
            for kw in keywords:
                if kw in message_lower:
                    matches.append(tool_name)
                    break
        return matches

    def remove_tool(self, name: str) -> None:
        """Remove a tool from all keyword indexes."""
        keywords = self._tool_keywords.pop(name, [])
        self._mandatory_keywords.pop(name, None)
        for kw in keywords:
            kw_lower = kw.lower()
            tool_names = self._keyword_index.get(kw_lower)
            if tool_names is None:
                continue
            tool_names.discard(name)
            if not tool_names:
                self._keyword_index.pop(kw_lower, None)
