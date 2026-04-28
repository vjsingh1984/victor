"""Tool schema cache using CacheNamespace.

DEPRECATED: This module now wraps victor.tools.cache_manager.CacheNamespace.
For new code, use ToolCacheManager.get_namespace("tool_schemas") directly.
"""

from __future__ import annotations

import logging
from typing import Optional

from victor.tools.cache_manager import CacheNamespace, ToolCacheManager

logger = logging.getLogger(__name__)


class ToolSchemaCache:
    """Cache for tool JSON schemas using CacheNamespace.

    This is a thin wrapper around CacheNamespace for backward compatibility.
    """

    def __init__(self, cache_manager: Optional[ToolCacheManager] = None):
        """Initialize the schema cache.

        Args:
            cache_manager: Optional ToolCacheManager instance. Creates default if None.
        """
        self._cache_manager = cache_manager or ToolCacheManager()
        # Create a dedicated namespace for tool schemas
        self._namespace = CacheNamespace(
            name="tool_schemas", max_entries=200, default_ttl=None
        )

    def register(self, name: str, schema: dict) -> None:
        """Register a tool schema."""
        self._namespace[name] = schema

    def get_schema(self, name: str) -> Optional[dict]:
        """Get a tool's schema."""
        return self._namespace.get(name)

    def invalidate(self, name: str) -> None:
        """Remove a tool's cached schema."""
        self._namespace.delete(name)

    def has_schema(self, name: str) -> bool:
        """Check if a schema is cached."""
        return name in self._namespace

    @property
    def registered_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._namespace.keys())

    def clear(self) -> None:
        """Clear all cached schemas."""
        self._namespace.clear()
