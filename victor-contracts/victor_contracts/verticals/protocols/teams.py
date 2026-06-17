"""Team-related protocol definitions.

These protocols define how verticals provide team configurations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Dict, Any, List, Optional


@runtime_checkable
class TeamProvider(Protocol):
    """Protocol for providing team configurations.

    Team providers define multi-agent team structures that can
    collaborate to accomplish complex tasks.
    """

    def get_team_spec(self) -> Dict[str, Any]:
        """Return team specification.

        Returns:
            Dictionary representing the team structure
        """
        ...

    def get_team_members(self) -> List[Dict[str, Any]]:
        """Return list of team member specifications.

        Returns:
            List of team member configurations
        """
        ...

    def get_formation_type(self) -> str:
        """Return the team formation type.

        Examples: "sequential", "parallel", "hierarchical", "pipeline"

        Returns:
            Formation type identifier
        """
        ...

    def get_coordination_strategy(self) -> Optional[str]:
        """Return the coordination strategy for this team.

        Returns:
            Strategy name or None
        """
        ...
