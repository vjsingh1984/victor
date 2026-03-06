"""Tool schema cache extracted from MetadataRegistry."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ToolSchemaCache:
    """Cache for tool JSON schemas.

    Extracted from MetadataRegistry for focused responsibility.
    """

    def __init__(self):
        self._schemas: dict[str, dict] = {}

    def register(self, name: str, schema: dict) -> None:
        """Register a tool schema."""
        self._schemas[name] = schema

    def get_schema(self, name: str) -> Optional[dict]:
        """Get a tool's schema."""
        return self._schemas.get(name)

    def invalidate(self, name: str) -> None:
        """Remove a tool's cached schema."""
        self._schemas.pop(name, None)

    def has_schema(self, name: str) -> bool:
        """Check if a schema is cached."""
        return name in self._schemas

    @property
    def registered_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._schemas.keys())

    def clear(self) -> None:
        """Clear all cached schemas."""
        self._schemas.clear()
