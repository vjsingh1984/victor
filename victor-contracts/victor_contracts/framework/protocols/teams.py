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


@runtime_checkable
class TeamAgent(Protocol):
    """Minimal protocol for an executable member of a team."""

    @property
    def id(self) -> str:
        """Stable team-local member identifier."""
        ...

    @property
    def role(self) -> Any:
        """Runtime or declarative role metadata."""
        ...

    async def execute_task(self, task: str, context: Dict[str, Any]) -> Any:
        """Execute one delegated task."""
        ...

    async def receive_message(self, message: Any) -> Optional[Any]:
        """Receive an inter-agent message and optionally respond."""
        ...


@runtime_checkable
class SupervisorAgent(TeamAgent, Protocol):
    """Contract for the coordinating member in a supervised team formation.

    A supervisor is allowed to decompose work, delegate to team members, and
    synthesize their results. Hierarchical formations should designate exactly
    one supervisor, either explicitly through
    ``agent_category='supervisor'`` or through the legacy ``is_manager=True``
    compatibility flag.
    """

    @property
    def is_supervisor(self) -> bool:
        """Whether this member is currently acting as team supervisor."""
        ...

    @property
    def can_delegate(self) -> bool:
        """Whether this member may delegate work to other members."""
        ...

    @property
    def delegation_targets(self) -> Optional[List[str]]:
        """Optional allow-list of member IDs the supervisor may delegate to."""
        ...
