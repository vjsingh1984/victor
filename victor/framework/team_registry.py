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

"""Team Specification Registry for reusable team blueprints.

This module provides a centralized registry for multi-agent team specifications,
enabling any vertical to register and query reusable team blueprints.

Design Philosophy:
- Singleton pattern for global team registry
- Thread-safe operations for concurrent access
- Namespace support for vertical-specific teams
- Inheritance and composition for team specs

Usage:
    from victor.framework.team_registry import (
        TeamSpecRegistry,
        get_team_registry,
        register_team_spec,
    )

    # Get the global registry
    registry = get_team_registry()

    # Register a team spec
    registry.register("coding:code_review", code_review_team_spec)

    # Get a team spec
    spec = registry.get("coding:code_review")

    # List available teams
    teams = registry.list_teams()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Singleton instance
_registry_instance: Optional["TeamSpecRegistry"] = None
_registry_lock = threading.Lock()


@dataclass
class TeamSpecEntry:
    """Entry in the team spec registry.

    Attributes:
        name: Full qualified name (e.g., "coding:code_review")
        spec: The team specification object
        vertical: Optional vertical namespace
        tags: Tags for filtering/discovery
        description: Human-readable description
    """

    name: str
    spec: Any
    vertical: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    description: str = ""

    @property
    def namespace(self) -> Optional[str]:
        """Get the namespace portion of the name."""
        if ":" in self.name:
            return self.name.split(":")[0]
        return None

    @property
    def short_name(self) -> str:
        """Get the short name without namespace."""
        if ":" in self.name:
            return self.name.split(":", 1)[1]
        return self.name


class TeamSpecRegistry:
    """Registry for multi-agent team specifications.

    This class provides a centralized location for registering and
    querying team specifications from any vertical.

    Thread-safe for concurrent access.

    Example:
        registry = TeamSpecRegistry()

        # Register a team
        registry.register(
            "coding:code_review",
            CodeReviewTeam,
            tags={"review", "quality"},
            description="Team for reviewing code changes"
        )

        # Query teams
        review_teams = registry.find_by_tag("review")
        coding_teams = registry.find_by_vertical("coding")
    """

    def __init__(self):
        """Initialize the registry."""
        self._teams: Dict[str, TeamSpecEntry] = {}
        self._lock = threading.RLock()

    def register(
        self,
        name: str,
        spec: Any,
        *,
        vertical: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        description: str = "",
        replace: bool = False,
    ) -> None:
        """Register a team specification.

        Args:
            name: Full qualified name (e.g., "coding:code_review")
            spec: The team specification object
            vertical: Optional vertical namespace (auto-detected from name if not provided)
            tags: Tags for filtering/discovery
            description: Human-readable description
            replace: If True, replace existing registration

        Raises:
            ValueError: If name already registered and replace=False
        """
        with self._lock:
            if name in self._teams and not replace:
                raise ValueError(
                    f"Team spec '{name}' already registered. "
                    f"Use replace=True to overwrite."
                )

            # Auto-detect vertical from namespaced name
            if vertical is None and ":" in name:
                vertical = name.split(":")[0]

            entry = TeamSpecEntry(
                name=name,
                spec=spec,
                vertical=vertical,
                tags=tags or set(),
                description=description,
            )

            self._teams[name] = entry
            logger.debug(f"Registered team spec: {name}")

    def unregister(self, name: str) -> bool:
        """Unregister a team specification.

        Args:
            name: Team spec name to unregister

        Returns:
            True if unregistered, False if not found
        """
        with self._lock:
            if name in self._teams:
                del self._teams[name]
                logger.debug(f"Unregistered team spec: {name}")
                return True
            return False

    def get(self, name: str) -> Optional[Any]:
        """Get a team specification by name.

        Args:
            name: Full qualified name

        Returns:
            Team spec or None if not found
        """
        with self._lock:
            entry = self._teams.get(name)
            return entry.spec if entry else None

    def get_entry(self, name: str) -> Optional[TeamSpecEntry]:
        """Get the full entry for a team specification.

        Args:
            name: Full qualified name

        Returns:
            TeamSpecEntry or None if not found
        """
        with self._lock:
            return self._teams.get(name)

    def list_teams(self) -> List[str]:
        """List all registered team names.

        Returns:
            List of team spec names
        """
        with self._lock:
            return list(self._teams.keys())

    def list_entries(self) -> List[TeamSpecEntry]:
        """List all team spec entries.

        Returns:
            List of TeamSpecEntry objects
        """
        with self._lock:
            return list(self._teams.values())

    def find_by_vertical(self, vertical: str) -> Dict[str, Any]:
        """Find team specs by vertical.

        Args:
            vertical: Vertical name to filter by

        Returns:
            Dict mapping names to team specs
        """
        with self._lock:
            return {
                name: entry.spec
                for name, entry in self._teams.items()
                if entry.vertical == vertical
            }

    def find_by_tag(self, tag: str) -> Dict[str, Any]:
        """Find team specs by tag.

        Args:
            tag: Tag to filter by

        Returns:
            Dict mapping names to team specs
        """
        with self._lock:
            return {
                name: entry.spec
                for name, entry in self._teams.items()
                if tag in entry.tags
            }

    def find_by_tags(self, tags: Set[str], match_all: bool = False) -> Dict[str, Any]:
        """Find team specs matching multiple tags.

        Args:
            tags: Set of tags to match
            match_all: If True, match all tags; if False, match any

        Returns:
            Dict mapping names to team specs
        """
        with self._lock:
            results = {}
            for name, entry in self._teams.items():
                if match_all:
                    if tags.issubset(entry.tags):
                        results[name] = entry.spec
                else:
                    if tags & entry.tags:
                        results[name] = entry.spec
            return results

    def clear(self) -> None:
        """Clear all registered team specs."""
        with self._lock:
            self._teams.clear()
            logger.debug("Cleared team spec registry")

    def register_from_vertical(
        self,
        vertical_name: str,
        team_specs: Dict[str, Any],
        replace: bool = True,
    ) -> int:
        """Register multiple team specs from a vertical.

        Convenience method for bulk registration with namespace prefixing.

        Args:
            vertical_name: Vertical name for namespace
            team_specs: Dict mapping team names to specs
            replace: If True, replace existing registrations

        Returns:
            Number of specs registered
        """
        count = 0
        for name, spec in team_specs.items():
            full_name = f"{vertical_name}:{name}"
            try:
                self.register(
                    full_name,
                    spec,
                    vertical=vertical_name,
                    replace=replace,
                )
                count += 1
            except Exception as e:
                logger.warning(f"Failed to register {full_name}: {e}")

        logger.info(f"Registered {count} team specs from vertical '{vertical_name}'")
        return count


def get_team_registry() -> TeamSpecRegistry:
    """Get the global team spec registry.

    Thread-safe singleton access.

    Returns:
        Global TeamSpecRegistry instance
    """
    global _registry_instance

    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = TeamSpecRegistry()

    return _registry_instance


def register_team_spec(
    name: str,
    spec: Any,
    *,
    vertical: Optional[str] = None,
    tags: Optional[Set[str]] = None,
    description: str = "",
    replace: bool = False,
) -> None:
    """Register a team spec in the global registry.

    Convenience function for quick registration.

    Args:
        name: Full qualified name
        spec: Team specification object
        vertical: Optional vertical namespace
        tags: Tags for discovery
        description: Human-readable description
        replace: Replace existing if present
    """
    get_team_registry().register(
        name,
        spec,
        vertical=vertical,
        tags=tags,
        description=description,
        replace=replace,
    )


def get_team_spec(name: str) -> Optional[Any]:
    """Get a team spec from the global registry.

    Args:
        name: Full qualified name

    Returns:
        Team spec or None
    """
    return get_team_registry().get(name)


__all__ = [
    "TeamSpecRegistry",
    "TeamSpecEntry",
    "get_team_registry",
    "register_team_spec",
    "get_team_spec",
]
