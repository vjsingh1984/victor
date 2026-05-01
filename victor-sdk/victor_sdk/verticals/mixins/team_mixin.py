"""Team coordination mixin for VerticalBase."""

from __future__ import annotations

from typing import Any, Dict, Optional

from victor_sdk.core.types import (
    TeamDefinitionLike,
    TeamMetadata,
    normalize_team_definitions,
    normalize_team_metadata,
)


class TeamMixin:
    """Opt-in mixin providing multi-agent team hooks.

    Methods:
        get_team_spec_provider: Return the team-spec provider.
        get_team_specs: Return team specifications dict.
        get_team_declarations: Return declarative team definitions.
        get_default_team: Return the default team name.
        get_team_metadata: Return serializable team metadata.
    """

    @classmethod
    def get_team_spec_provider(cls) -> Optional[Any]:
        """Return the team-spec provider for this vertical, if any."""
        return None

    @classmethod
    def get_team_specs(cls) -> Dict[str, Any]:
        """Return team specifications for this vertical."""
        return {}

    @classmethod
    def get_team_declarations(cls) -> Dict[str, TeamDefinitionLike]:
        """Return declarative team definitions for this vertical."""
        return {}

    @classmethod
    def get_default_team(cls) -> Optional[str]:
        """Return the default declarative team for this vertical."""
        return None

    @classmethod
    def get_team_metadata(cls) -> TeamMetadata:
        """Return serializable team metadata for this vertical."""
        return normalize_team_metadata(
            {
                "teams": normalize_team_definitions(cls.get_team_declarations()),
                "default_team": cls.get_default_team(),
            }
        )
