"""Tool category resolver extracted from MetadataRegistry."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


class ToolCategoryResolver:
    """Resolve tool categories and find tools by category.

    Extracted from MetadataRegistry for focused responsibility.
    """

    def __init__(self):
        self._tool_categories: dict[str, str] = {}
        self._category_tools: dict[str, list[str]] = defaultdict(list)

    def register(self, tool_name: str, category: str) -> None:
        """Register a tool's category."""
        self._tool_categories[tool_name] = category
        if tool_name not in self._category_tools[category]:
            self._category_tools[category].append(tool_name)

    def get_category(self, name: str) -> Optional[str]:
        """Get a tool's category."""
        return self._tool_categories.get(name)

    def get_tools_in_category(self, category: str) -> list[str]:
        """Get all tools in a category."""
        return list(self._category_tools.get(category, []))

    def get_all_categories(self) -> list[str]:
        """Get all known categories."""
        return sorted(self._category_tools.keys())

    def remove_tool(self, tool_name: str) -> None:
        """Remove a tool from category tracking."""
        category = self._tool_categories.pop(tool_name, None)
        if category and tool_name in self._category_tools.get(category, []):
            self._category_tools[category].remove(tool_name)
