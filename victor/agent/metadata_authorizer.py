# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metadata-Based Action Authorization for Phase 5.

This module provides the MetadataActionAuthorizer which replaces hard-coded
tool lists in action_authorizer.py with metadata-based authorization.

Phase 5: Tool Metadata System
============================
- MetadataActionAuthorizer uses ToolAuthMetadataRegistry
- Authorization based on tool capabilities and safety levels
- New verticals can define tools without core changes

Benefits:
- No hard-coded tool lists
- Self-documenting authorization rules
- Flexible, metadata-driven policies

Usage:
    authorizer = MetadataActionAuthorizer()
    is_authorized = authorizer.authorize_tool(
        tool_name="write_file",
        intent=ActionIntent.DISPLAY_ONLY,
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from victor.agent.action_authorizer import ActionIntent

from victor.tools.auth_metadata import (
    ToolAuthMetadataRegistry,
    ToolSafety,
    get_tool_auth_metadata_registry,
)

logger = logging.getLogger(__name__)


class MetadataActionAuthorizer:
    """Authorize tool actions based on metadata.

    This authorizer uses the ToolAuthMetadataRegistry to determine
    whether a tool can be used based on:
    - Tool capabilities (file_write, network, execute, etc.)
    - Tool safety level (SAFE, REQUIRES_CONFIRMATION, DESTRUCTIVE, BLOCKED)
    - User intent (DISPLAY_ONLY, READ_ONLY, WRITE_ALLOWED, AMBIGUOUS)

    This replaces hard-coded tool lists with metadata-based authorization.

    Example:
        authorizer = MetadataActionAuthorizer()

        # Check if tool is authorized for display-only intent
        if authorizer.authorize_tool("write_file", ActionIntent.DISPLAY_ONLY):
            # Tool is authorized
            pass
        else:
            # Tool is blocked
            logger.warning("Tool not authorized for DISPLAY_ONLY intent")

        # Get blocked tools for an intent
        blocked_tools = authorizer.get_blocked_tools(ActionIntent.DISPLAY_ONLY)
    """

    def __init__(self, metadata_registry: Optional[ToolAuthMetadataRegistry] = None):
        """Initialize the authorizer.

        Args:
            metadata_registry: Optional metadata registry (uses global singleton if None)
        """
        self._registry = (
            metadata_registry if metadata_registry else get_tool_auth_metadata_registry()
        )
        self._cache: dict[tuple[str, str], bool] = {}

    def authorize_tool(self, tool_name: str, intent: "ActionIntent") -> bool:
        """Check if a tool is authorized for a given intent.

        Authorization is based on tool metadata:
        - DISPLAY_ONLY: Blocks tools with file_write capability or destructive safety
        - READ_ONLY: Blocks file_write, generate_code, and destructive tools
        - WRITE_ALLOWED: Allows all tools except BLOCKED
        - AMBIGUOUS: Same as DISPLAY_ONLY (safe default)

        Args:
            tool_name: Name of the tool to authorize
            intent: User intent (ActionIntent enum)

        Returns:
            True if tool is authorized, False otherwise
        """
        # Check cache first
        cache_key = (tool_name, intent.value)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Get tool metadata
        metadata = self._registry.get(tool_name)

        # If no metadata, default to allowed (backward compatibility)
        if metadata is None:
            logger.debug(f"No metadata for tool '{tool_name}', defaulting to allowed")
            return True

        # Check based on safety level
        if metadata.safety == ToolSafety.BLOCKED:
            logger.debug(f"Tool '{tool_name}' is BLOCKED")
            self._cache[cache_key] = False
            return False

        # Check intent-specific rules
        intent_str = intent.value if hasattr(intent, "value") else str(intent)

        if intent_str in ("display_only", "read_only"):
            # Read-only intents cannot use destructive tools
            if metadata.is_destructive():
                logger.debug(
                    f"Tool '{tool_name}' is destructive and not allowed for {intent_str} intent"
                )
                self._cache[cache_key] = False
                return False

            # Read-only intents cannot use file_write capability
            if metadata.has_capability("file_write"):
                logger.debug(
                    f"Tool '{tool_name}' has file_write capability and is not allowed for {intent_str} intent"
                )
                self._cache[cache_key] = False
                return False

            # READ_ONLY intent also blocks code generation
            if intent_str == "read_only" and metadata.has_capability("code_generation"):
                logger.debug(
                    f"Tool '{tool_name}' has code_generation capability and is not allowed for READ_ONLY intent"
                )
                self._cache[cache_key] = False
                return False

        # Tool is authorized
        logger.debug(f"Tool '{tool_name}' is authorized for {intent_str} intent")
        self._cache[cache_key] = True
        return True

    def get_blocked_tools(self, intent: "ActionIntent") -> set[str]:
        """Get all tools that are blocked for a given intent.

        This method queries the metadata registry to find all tools
        that should be blocked for the specified intent.

        Args:
            intent: User intent (ActionIntent enum)

        Returns:
            Set of tool names that are blocked
        """
        blocked_tools = set()
        all_tools = self._registry.list_all_tools()

        for tool_name in all_tools:
            if not self.authorize_tool(tool_name, intent):
                blocked_tools.add(tool_name)

        return blocked_tools

    def get_authorized_tools(
        self, intent: "ActionIntent", available_tools: Optional[list[str]] = None
    ) -> set[str]:
        """Get all tools that are authorized for a given intent.

        Args:
            intent: User intent (ActionIntent enum)
            available_tools: Optional list of tools to filter (uses all registered if None)

        Returns:
            Set of tool names that are authorized
        """
        if available_tools is None:
            available_tools = self._registry.list_all_tools()

        authorized_tools = set()
        for tool_name in available_tools:
            if self.authorize_tool(tool_name, intent):
                authorized_tools.add(tool_name)

        return authorized_tools

    def filter_tools(self, tools: list[str], intent: "ActionIntent") -> list[str]:
        """Filter a list of tools to only authorized ones.

        Args:
            tools: List of tool names to filter
            intent: User intent (ActionIntent enum)

        Returns:
            List of authorized tool names (in original order)
        """
        return [tool for tool in tools if self.authorize_tool(tool, intent)]

    def get_tool_safety(self, tool_name: str) -> Optional[ToolSafety]:
        """Get the safety level for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            ToolSafety level or None if tool not found
        """
        metadata = self._registry.get(tool_name)
        return metadata.safety if metadata else None

    def requires_confirmation(self, tool_name: str) -> bool:
        """Check if a tool requires user confirmation.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool requires confirmation
        """
        metadata = self._registry.get(tool_name)
        return metadata.requires_user_confirmation() if metadata else False

    def get_tools_by_capability(self, capability: str) -> list[str]:
        """Get all tools with a specific capability.

        Args:
            capability: Capability to filter by

        Returns:
            List of tool names with the capability
        """
        return self._registry.get_tools_by_capability(capability)

    def get_tools_by_domain(self, domain: str) -> list[str]:
        """Get all tools in a specific domain.

        Args:
            domain: Domain to filter by (coding, devops, research, etc.)

        Returns:
            List of tool names in the domain
        """
        return self._registry.get_tools_by_domain(domain)

    def clear_cache(self) -> None:
        """Clear the authorization cache.

        This should be called if tool metadata is updated at runtime.
        """
        self._cache.clear()
        logger.debug("Cleared authorization cache")

    def get_authorization_summary(self) -> dict[str, Any]:
        """Get a summary of authorization rules.

        Returns:
            Dict with authorization statistics
        """
        from victor.agent.action_authorizer import ActionIntent

        total_tools = len(self._registry.list_all_tools())

        summary: dict[str, Any] = {
            "total_tools": total_tools,
            "tools_by_safety": {},
            "blocked_tools_by_intent": {},
        }

        # Count tools by safety level
        for tool_name in self._registry.list_all_tools():
            safety = self.get_tool_safety(tool_name)
            if safety:
                safety_str = safety.value
                summary["tools_by_safety"][safety_str] = (
                    summary["tools_by_safety"].get(safety_str, 0) + 1
                )

        # Count blocked tools by intent
        for intent in ActionIntent:
            blocked = self.get_blocked_tools(intent)
            summary["blocked_tools_by_intent"][intent.value] = len(blocked)

        return summary


# =============================================================================
# Convenience Functions
# =============================================================================


def get_metadata_authorizer() -> MetadataActionAuthorizer:
    """Get the global metadata authorizer instance.

    Returns:
        MetadataActionAuthorizer using the global registry
    """
    return MetadataActionAuthorizer()


def authorize_with_metadata(tool_name: str, intent: "ActionIntent") -> bool:
    """Authorize a tool using metadata.

    Convenience function for quick authorization checks.

    Args:
        tool_name: Name of the tool to authorize
        intent: User intent (ActionIntent enum)

    Returns:
        True if tool is authorized, False otherwise
    """
    authorizer = get_metadata_authorizer()
    return authorizer.authorize_tool(tool_name, intent)


__all__ = [
    "MetadataActionAuthorizer",
    "get_metadata_authorizer",
    "authorize_with_metadata",
]
