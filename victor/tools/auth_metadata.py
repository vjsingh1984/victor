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

"""Tool Authorization Metadata System for Phase 5.

This module provides the authorization metadata system that replaces hard-coded
tool lists in action_authorizer.py with metadata-based authorization.

Phase 5: Tool Metadata System
============================
Eliminate hard-coded tool names in action_authorizer.py by:
- Defining ToolAuthMetadata with categories, capabilities, safety, domain
- ToolAuthMetadataRegistry for centralized metadata management
- MetadataActionAuthorizer for metadata-based authorization

Benefits:
- New verticals can define tools without core changes
- Self-documenting tool capabilities
- Flexible authorization policies based on metadata

Usage:
    # Define tool authorization metadata
    metadata = ToolAuthMetadata(
        name="write_file",
        categories=["coding", "file_ops"],
        capabilities=["file_write"],
        safety=ToolSafety.REQUIRES_CONFIRMATION,
        domain="coding",
        cost_tier=2,
    )

    # Register with global registry
    registry = ToolAuthMetadataRegistry.get_instance()
    registry.register(metadata)

    # Query by category
    write_tools = registry.get_tools_by_capability("file_write")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Safety Enum
# =============================================================================


class ToolSafety(Enum):
    """Safety level for tools.

    Determines authorization requirements:
    - SAFE: Always allowed, no confirmation needed
    - REQUIRES_CONFIRMATION: Requires user approval before execution
    - DESTRUCTIVE: Only allowed with explicit write intent
    - BLOCKED: Never allowed (e.g., security risks)
    """

    SAFE = "safe"  # Read-only operations, no side effects
    REQUIRES_CONFIRMATION = "requires_confirmation"  # Destructive but reversible
    DESTRUCTIVE = "destructive"  # Irreversible changes
    BLOCKED = "blocked"  # Security risks, never allowed


# =============================================================================
# Tool Authorization Metadata Dataclass
# =============================================================================


@dataclass(frozen=True)
class ToolAuthMetadata:
    """Authorization metadata for tools.

    This dataclass captures all metadata needed for authorization decisions,
    replacing hard-coded tool lists in action_authorizer.py.

    This is separate from the semantic ToolMetadata used for tool selection.
    ToolAuthMetadata is specifically for authorization/security decisions.

    Attributes:
        name: Tool name (must match BaseTool.name)
        categories: High-level categories (e.g., ['coding', 'refactor', 'file_ops'])
        capabilities: Specific capabilities (e.g., ['file_write', 'network', 'execute'])
        safety: Safety level for authorization
        domain: Vertical domain ('coding', 'devops', 'research', 'generic')
        cost_tier: Execution cost tier (1=low, 5=high)
        description: Human-readable description
    """

    name: str
    categories: list[str]
    capabilities: list[str]
    safety: ToolSafety
    domain: str
    cost_tier: int = 1
    description: Optional[str] = None

    def __post_init__(self):
        """Validate metadata after initialization."""
        # Validate cost tier is in range
        if self.cost_tier < 1 or self.cost_tier > 5:
            raise ValueError(f"cost_tier must be 1-5, got {self.cost_tier}")

        # Validate domain is known
        valid_domains = {"coding", "devops", "rag", "dataanalysis", "research", "generic"}
        if self.domain not in valid_domains:
            logger.warning(f"Unknown domain '{self.domain}'. Expected one of {valid_domains}")

    def has_capability(self, capability: str) -> bool:
        """Check if tool has a specific capability.

        Args:
            capability: Capability to check

        Returns:
            True if tool has this capability
        """
        return capability in self.capabilities

    def has_category(self, category: str) -> bool:
        """Check if tool belongs to a category.

        Args:
            category: Category to check

        Returns:
            True if tool is in this category
        """
        return category in self.categories

    def is_destructive(self) -> bool:
        """Check if tool is destructive.

        Returns:
            True if safety level is DESTRUCTIVE or BLOCKED
        """
        return self.safety in (ToolSafety.DESTRUCTIVE, ToolSafety.BLOCKED)

    def requires_user_confirmation(self) -> bool:
        """Check if tool requires user confirmation.

        Returns:
            True if safety level requires confirmation
        """
        return self.safety in (
            ToolSafety.REQUIRES_CONFIRMATION,
            ToolSafety.DESTRUCTIVE,
            ToolSafety.BLOCKED,
        )

    def is_safe_for_intent(self, intent: str) -> bool:
        """Check if tool is safe for a given intent.

        Args:
            intent: User intent (e.g., 'DISPLAY_ONLY', 'READ_ONLY', 'WRITE_ALLOWED')

        Returns:
            True if tool can be used with this intent
        """
        # Read-only intents cannot use destructive tools
        if intent in ("DISPLAY_ONLY", "READ_ONLY"):
            if "file_write" in self.capabilities:
                return False
            if self.is_destructive():
                return False

        # Write intents must confirm destructive tools
        if intent == "WRITE_ALLOWED":
            return True

        # Default to safe
        return True


# =============================================================================
# Tool Authorization Metadata Registry
# =============================================================================


class ToolAuthMetadataRegistry:
    """Registry for tool authorization metadata.

    This singleton registry maintains authorization metadata for all tools,
    enabling metadata-based authorization instead of hard-coded lists.

    Thread-safe with RLock for concurrent access.

    Usage:
        registry = ToolAuthMetadataRegistry.get_instance()
        registry.register(metadata)
        metadata = registry.get("write_file")
        write_tools = registry.get_tools_by_capability("file_write")
    """

    _instance: Optional["ToolAuthMetadataRegistry"] = None
    _lock: RLock = RLock()

    def __init__(self):
        """Private constructor (use get_instance())."""
        if ToolAuthMetadataRegistry._instance is not None:
            raise RuntimeError("Use get_instance() to get the singleton instance")

        self._metadata: dict[str, ToolAuthMetadata] = {}
        self._categories: dict[str, set[str]] = {}
        self._capabilities: dict[str, set[str]] = {}
        self._domains: dict[str, set[str]] = {}

    @classmethod
    def get_instance(cls) -> "ToolAuthMetadataRegistry":
        """Get the singleton registry instance.

        Returns:
            ToolAuthMetadataRegistry singleton
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_testing(cls) -> None:
        """Reset the singleton (testing only).

        Warning: This should only be used in tests to isolate test cases.
        """
        with cls._lock:
            cls._instance = None

    def register(self, metadata: ToolAuthMetadata) -> None:
        """Register tool authorization metadata.

        Args:
            metadata: Tool metadata to register

        Raises:
            ValueError: If metadata already exists with different name
        """
        with self._lock:
            name = metadata.name

            # Check for existing metadata with same name
            if name in self._metadata:
                existing = self._metadata[name]
                if existing != metadata:
                    logger.warning(
                        f"Tool auth metadata already exists for '{name}', "
                        f"overwriting: {existing} -> {metadata}"
                    )

            # Store metadata
            self._metadata[name] = metadata

            # Index by category
            for category in metadata.categories:
                if category not in self._categories:
                    self._categories[category] = set()
                self._categories[category].add(name)

            # Index by capability
            for capability in metadata.capabilities:
                if capability not in self._capabilities:
                    self._capabilities[capability] = set()
                self._capabilities[capability].add(name)

            # Index by domain
            domain = metadata.domain
            if domain not in self._domains:
                self._domains[domain] = set()
            self._domains[domain].add(name)

            logger.debug(f"Registered tool auth metadata: {name}")

    def get(self, tool_name: str) -> Optional[ToolAuthMetadata]:
        """Get metadata for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            ToolAuthMetadata or None if not found
        """
        with self._lock:
            return self._metadata.get(tool_name)

    def has_capability(self, tool_name: str, capability: str) -> bool:
        """Check if tool has a capability.

        Args:
            tool_name: Name of the tool
            capability: Capability to check

        Returns:
            True if tool has this capability
        """
        metadata = self.get(tool_name)
        return metadata.has_capability(capability) if metadata else False

    def get_tools_by_category(self, category: str) -> list[str]:
        """Get all tools in a category.

        Args:
            category: Category to filter by

        Returns:
            List of tool names in the category
        """
        with self._lock:
            tools = self._categories.get(category, set())
            return sorted(tools)

    def get_tools_by_capability(self, capability: str) -> list[str]:
        """Get all tools with a capability.

        Args:
            capability: Capability to filter by

        Returns:
            List of tool names with the capability
        """
        with self._lock:
            tools = self._capabilities.get(capability, set())
            return sorted(tools)

    def get_tools_by_domain(self, domain: str) -> list[str]:
        """Get all tools in a domain.

        Args:
            domain: Domain to filter by

        Returns:
            List of tool names in the domain
        """
        with self._lock:
            tools = self._domains.get(domain, set())
            return sorted(tools)

    def list_all_tools(self) -> list[str]:
        """List all registered tools.

        Returns:
            Sorted list of tool names
        """
        with self._lock:
            return sorted(self._metadata.keys())

    def list_categories(self) -> list[str]:
        """List all categories.

        Returns:
            Sorted list of categories
        """
        with self._lock:
            return sorted(self._categories.keys())

    def list_capabilities(self) -> list[str]:
        """List all capabilities.

        Returns:
            Sorted list of capabilities
        """
        with self._lock:
            return sorted(self._capabilities.keys())

    def list_domains(self) -> list[str]:
        """List all domains.

        Returns:
            Sorted list of domains
        """
        with self._lock:
            return sorted(self._domains.keys())

    def get_metadata_summary(self) -> dict[str, Any]:
        """Get summary of registered metadata.

        Returns:
            Dict with counts and statistics
        """
        with self._lock:
            return {
                "total_tools": len(self._metadata),
                "total_categories": len(self._categories),
                "total_capabilities": len(self._capabilities),
                "total_domains": len(self._domains),
                "tools_by_domain": {domain: len(tools) for domain, tools in self._domains.items()},
            }

    def clear(self) -> None:
        """Clear all metadata (testing only).

        Warning: This should only be used in tests.
        """
        with self._lock:
            self._metadata.clear()
            self._categories.clear()
            self._capabilities.clear()
            self._domains.clear()
            logger.debug("Cleared all tool auth metadata")


# =============================================================================
# Convenience Functions
# =============================================================================


def get_tool_auth_metadata_registry() -> ToolAuthMetadataRegistry:
    """Get the global tool authorization metadata registry.

    Returns:
        ToolAuthMetadataRegistry singleton instance
    """
    return ToolAuthMetadataRegistry.get_instance()


def register_tool_auth_metadata(
    name: str,
    categories: list[str],
    capabilities: list[str],
    safety: ToolSafety,
    domain: str,
    cost_tier: int = 1,
    description: Optional[str] = None,
) -> ToolAuthMetadata:
    """Create and register tool authorization metadata.

    Convenience function for creating and registering metadata.

    Args:
        name: Tool name
        categories: Tool categories
        capabilities: Tool capabilities
        safety: Safety level
        domain: Tool domain
        cost_tier: Cost tier (1-5)
        description: Optional description

    Returns:
        Created ToolAuthMetadata instance
    """
    metadata = ToolAuthMetadata(
        name=name,
        categories=categories,
        capabilities=capabilities,
        safety=safety,
        domain=domain,
        cost_tier=cost_tier,
        description=description,
    )

    registry = get_tool_auth_metadata_registry()
    registry.register(metadata)

    return metadata


__all__ = [
    # Enum
    "ToolSafety",
    # Dataclass
    "ToolAuthMetadata",
    # Registry
    "ToolAuthMetadataRegistry",
    # Convenience
    "get_tool_auth_metadata_registry",
    "register_tool_auth_metadata",
]
