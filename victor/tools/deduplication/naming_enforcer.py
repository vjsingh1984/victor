"""Naming convention enforcement for tool deduplication.

This module enforces unified naming conventions across tool sources:
- Native tools: no prefix (read, write, edit)
- LangChain tools: lgc_ prefix (lgc_wikipedia, lgc_wolfram_alpha)
- MCP tools: mcp_ prefix (mcp_github_search, mcp_filesystem_read)
- Plugin tools: plg_ prefix (plg_custom_tool)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from victor.tools.deduplication.tool_deduplicator import ToolSource

logger = logging.getLogger(__name__)


class NamingConvention(Enum):
    """Naming convention for each tool source."""

    NATIVE_PREFIX = ""  # No prefix
    LANGCHAIN_PREFIX = "lgc_"
    MCP_PREFIX = "mcp_"
    PLUGIN_PREFIX = "plg_"

    def __str__(self) -> str:
        """Return the prefix string."""
        return self.value


class NamingEnforcer:
    """Enforces naming conventions for tools across sources.

    Usage:
        enforcer = NamingEnforcer()
        new_name = enforcer.enforce_name(tool, source=ToolSource.LANGCHAIN)
    """

    def __init__(self, enforce: bool = True) -> None:
        """Initialize naming enforcer.

        Args:
            enforce: Whether to enforce naming conventions (default: True)
        """
        self._enforce = enforce

    def enforce_name(self, tool: Any, source: ToolSource) -> str:
        """Enforce naming convention by adding source prefix if needed.

        Args:
            tool: Tool to check/rename
            source: Tool source (NATIVE, LANGCHAIN, MCP, PLUGIN)

        Returns:
            Original name if convention already followed, else new name with prefix
        """
        if not self._enforce:
            return self._get_tool_name(tool)

        tool_name = self._get_tool_name(tool)

        # Get expected prefix for source
        expected_prefix = self._get_prefix_for_source(source)

        # Skip if no prefix expected (native tools)
        if not expected_prefix:
            return tool_name

        # Check if prefix already present
        if tool_name.lower().startswith(expected_prefix.lower()):
            return tool_name

        # Add prefix
        new_name = f"{expected_prefix}{tool_name}"
        logger.debug(
            f"Naming enforcement: {tool_name} → {new_name} (source={source.value})"
        )
        return new_name

    def strip_prefix(self, tool_name: str) -> str:
        """Strip source prefix from tool name.

        Args:
            tool_name: Tool name with potential prefix

        Returns:
            Tool name without prefix
        """
        for source in ToolSource:
            prefix = self._get_prefix_for_source(source)
            if prefix and tool_name.lower().startswith(prefix.lower()):
                return tool_name[len(prefix) :]
        return tool_name

    def detect_source_from_name(self, tool_name: str) -> ToolSource:
        """Detect tool source from naming convention.

        Args:
            tool_name: Tool name to analyze

        Returns:
            Detected ToolSource (NATIVE if no prefix detected)
        """
        tool_name_lower = tool_name.lower()

        for source in ToolSource:
            prefix = self._get_prefix_for_source(source)
            if prefix and tool_name_lower.startswith(prefix.lower()):
                return source

        return ToolSource.NATIVE

    def _get_prefix_for_source(self, source: ToolSource) -> str:
        """Get expected prefix for a tool source."""
        prefixes = {
            ToolSource.NATIVE: NamingConvention.NATIVE_PREFIX.value,
            ToolSource.LANGCHAIN: NamingConvention.LANGCHAIN_PREFIX.value,
            ToolSource.MCP: NamingConvention.MCP_PREFIX.value,
            ToolSource.PLUGIN: NamingConvention.PLUGIN_PREFIX.value,
        }
        return prefixes.get(source, "")

    def _get_tool_name(self, tool: Any) -> str:
        """Extract tool name."""
        if hasattr(tool, "name"):
            return tool.name
        elif hasattr(tool, "__name__"):
            return tool.__name__
        return str(tool)
