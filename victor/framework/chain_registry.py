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

"""Registry for composed chains across verticals.

Enables discovery and access to Runnable chains from any vertical.

Design Philosophy:
- Singleton pattern for global chain registry
- Thread-safe operations for concurrent access
- Namespace support for vertical-specific chains
- Metadata support for discoverability

Usage:
    from victor.framework.chain_registry import (
        ChainRegistry,
        get_chain_registry,
        register_chain,
        get_chain,
    )

    # Get the global registry
    registry = get_chain_registry()

    # Register a chain
    registry.register(
        "code_review",
        code_review_chain,
        vertical="coding",
        description="Chain for reviewing code changes",
    )

    # Get a chain
    chain = registry.get("code_review", vertical="coding")

    # List available chains
    chains = registry.list_chains()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Singleton instance
_registry_instance: Optional["ChainRegistry"] = None
_registry_lock = threading.Lock()


@dataclass
class ChainMetadata:
    """Metadata for a registered chain.

    Attributes:
        name: Short name of the chain
        vertical: Optional vertical namespace (e.g., "coding", "devops")
        description: Human-readable description
        input_type: Optional type hint for chain input
        output_type: Optional type hint for chain output
        tags: Tags for filtering/discovery
    """

    name: str
    vertical: Optional[str] = None
    description: str = ""
    input_type: Optional[str] = None
    output_type: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        """Get the full qualified name with namespace."""
        if self.vertical:
            return f"{self.vertical}:{self.name}"
        return self.name


class ChainRegistry:
    """Registry for composed chains across verticals.

    Chains are namespaced by vertical: "{vertical}:{name}"

    This class provides a centralized location for registering and
    querying Runnable chains from any vertical.

    Thread-safe for concurrent access.

    Example:
        registry = ChainRegistry()

        # Register a chain
        registry.register(
            "code_review",
            code_review_chain,
            vertical="coding",
            description="Chain for reviewing code",
        )

        # Query chains
        chain = registry.get("code_review", vertical="coding")
        coding_chains = registry.find_by_vertical("coding")
    """

    def __init__(self):
        """Initialize the registry."""
        self._chains: Dict[str, Any] = {}
        self._metadata: Dict[str, ChainMetadata] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        chain: Any,
        vertical: Optional[str] = None,
        description: str = "",
        input_type: Optional[str] = None,
        output_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        replace: bool = False,
    ) -> None:
        """Register a chain with optional vertical namespace.

        Args:
            name: Short name for the chain
            chain: The chain object (typically a Runnable)
            vertical: Optional vertical namespace
            description: Human-readable description
            input_type: Optional type hint for chain input
            output_type: Optional type hint for chain output
            tags: Tags for filtering/discovery
            replace: If True, replace existing registration

        Raises:
            ValueError: If name already registered and replace=False
        """
        key = f"{vertical}:{name}" if vertical else name

        with self._lock:
            if key in self._chains and not replace:
                logger.warning(f"Chain '{key}' already registered, skipping")
                return

            self._chains[key] = chain
            self._metadata[key] = ChainMetadata(
                name=name,
                vertical=vertical,
                description=description,
                input_type=input_type,
                output_type=output_type,
                tags=tags or [],
            )
            logger.debug(f"Registered chain: {key}")

    def unregister(self, name: str, vertical: Optional[str] = None) -> bool:
        """Unregister a chain.

        Args:
            name: Chain name to unregister
            vertical: Optional vertical namespace

        Returns:
            True if unregistered, False if not found
        """
        key = f"{vertical}:{name}" if vertical else name

        with self._lock:
            if key in self._chains:
                del self._chains[key]
                del self._metadata[key]
                logger.debug(f"Unregistered chain: {key}")
                return True
            return False

    def get(self, name: str, vertical: Optional[str] = None) -> Optional[Any]:
        """Get a chain by name, optionally scoped to vertical.

        Args:
            name: Chain name to retrieve
            vertical: Optional vertical to scope the search

        Returns:
            Chain object or None if not found
        """
        with self._lock:
            if vertical:
                key = f"{vertical}:{name}"
                if key in self._chains:
                    return self._chains[key]
            return self._chains.get(name)

    def get_metadata(self, name: str, vertical: Optional[str] = None) -> Optional[ChainMetadata]:
        """Get metadata for a chain.

        Args:
            name: Chain name
            vertical: Optional vertical namespace

        Returns:
            ChainMetadata or None if not found
        """
        with self._lock:
            if vertical:
                key = f"{vertical}:{name}"
                if key in self._metadata:
                    return self._metadata[key]
            return self._metadata.get(name)

    def list_chains(self, vertical: Optional[str] = None) -> List[str]:
        """List all registered chain names.

        Args:
            vertical: If provided, only list chains from this vertical

        Returns:
            List of chain names (full keys)
        """
        with self._lock:
            if vertical:
                prefix = f"{vertical}:"
                return [k for k in self._chains.keys() if k.startswith(prefix)]
            return list(self._chains.keys())

    def list_metadata(self, vertical: Optional[str] = None) -> List[ChainMetadata]:
        """List all chain metadata.

        Args:
            vertical: If provided, only list metadata from this vertical

        Returns:
            List of ChainMetadata objects
        """
        with self._lock:
            if vertical:
                prefix = f"{vertical}:"
                return [m for k, m in self._metadata.items() if k.startswith(prefix)]
            return list(self._metadata.values())

    def find_by_vertical(self, vertical: str) -> Dict[str, Any]:
        """Get all chains for a specific vertical.

        Args:
            vertical: Vertical name to filter by

        Returns:
            Dict mapping full names to chain objects
        """
        prefix = f"{vertical}:"
        with self._lock:
            return {k: v for k, v in self._chains.items() if k.startswith(prefix)}

    def find_by_tag(self, tag: str) -> Dict[str, Any]:
        """Find chains with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            Dict mapping names to chain objects
        """
        with self._lock:
            return {k: self._chains[k] for k, m in self._metadata.items() if tag in m.tags}

    def find_by_tags(self, tags: List[str], match_all: bool = False) -> Dict[str, Any]:
        """Find chains matching multiple tags.

        Args:
            tags: List of tags to match
            match_all: If True, match all tags; if False, match any

        Returns:
            Dict mapping names to chain objects
        """
        tag_set = set(tags)
        with self._lock:
            results = {}
            for key, metadata in self._metadata.items():
                entry_tags = set(metadata.tags)
                if match_all:
                    if tag_set.issubset(entry_tags):
                        results[key] = self._chains[key]
                else:
                    if tag_set & entry_tags:
                        results[key] = self._chains[key]
            return results

    def clear(self) -> None:
        """Clear all registered chains (for testing)."""
        with self._lock:
            self._chains.clear()
            self._metadata.clear()
            logger.debug("Cleared chain registry")

    def register_from_vertical(
        self,
        vertical_name: str,
        chains: Dict[str, Any],
        replace: bool = True,
    ) -> int:
        """Register multiple chains from a vertical.

        Convenience method for bulk registration with namespace prefixing.

        Args:
            vertical_name: Vertical name for namespace
            chains: Dict mapping chain names to chain objects
            replace: If True, replace existing registrations

        Returns:
            Number of chains registered
        """
        count = 0
        for name, chain in chains.items():
            try:
                self.register(
                    name,
                    chain,
                    vertical=vertical_name,
                    replace=replace,
                )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to register {vertical_name}:{name}: {e}")

        logger.info(f"Registered {count} chains from vertical '{vertical_name}'")
        return count


def get_chain_registry() -> ChainRegistry:
    """Get the global chain registry.

    Thread-safe singleton access.

    Returns:
        Global ChainRegistry instance
    """
    global _registry_instance

    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = ChainRegistry()

    return _registry_instance


def register_chain(
    name: str,
    chain: Any,
    *,
    vertical: Optional[str] = None,
    description: str = "",
    tags: Optional[List[str]] = None,
    replace: bool = False,
) -> None:
    """Register a chain in the global registry.

    Convenience function for quick registration.

    Args:
        name: Chain name
        chain: Chain object
        vertical: Optional vertical namespace
        description: Human-readable description
        tags: Tags for discovery
        replace: Replace existing if present
    """
    get_chain_registry().register(
        name,
        chain,
        vertical=vertical,
        description=description,
        tags=tags,
        replace=replace,
    )


def get_chain(name: str, vertical: Optional[str] = None) -> Optional[Any]:
    """Get a chain from the global registry.

    Args:
        name: Chain name
        vertical: Optional vertical namespace

    Returns:
        Chain object or None
    """
    return get_chain_registry().get(name, vertical=vertical)


__all__ = [
    "ChainRegistry",
    "ChainMetadata",
    "get_chain_registry",
    "register_chain",
    "get_chain",
]
