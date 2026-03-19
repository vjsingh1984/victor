"""Team registry protocol definitions.

These protocols define how verticals interact with the team registry.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Dict, Any, List, Optional


@runtime_checkable
class TeamRegistry(Protocol):
    """Protocol for team registration and discovery.

    Team registries manage multi-agent team configurations and
    enable verticals to provide team-based solutions.
    """

    def register_team(self, team_spec: Dict[str, Any]) -> str:
        """Register a team specification.

        Args:
            team_spec: Team configuration dictionary

        Returns:
            Team identifier
        """
        ...

    def get_team(self, team_id: str) -> Optional[Dict[str, Any]]:
        """Get a team specification by ID.

        Args:
            team_id: Team identifier

        Returns:
            Team specification or None
        """
        ...

    def list_teams(self) -> List[str]:
        """List all registered team IDs.

        Returns:
            List of team identifiers
        """
        ...

    def find_teams_for_vertical(self, vertical_name: str) -> List[Dict[str, Any]]:
        """Find teams configured for a specific vertical.

        Args:
            vertical_name: Vertical identifier

        Returns:
            List of team specifications
        """
        ...
