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

"""Base team provider for YAML-based team specifications.

This module provides a centralized system for loading team formations from
YAML configuration files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


class TeamFormationType(str):
    """Team formation patterns.

    Valid types:
        PIPELINE: Sequential execution
        PARALLEL: Concurrent execution
        SEQUENTIAL: Step-by-step with handoff
        HIERARCHICAL: Manager-worker pattern
        CONSENSUS: Vote-based decision making
    """

    PIPELINE = "pipeline"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    CONSENSUS = "consensus"


@dataclass
class AgentRole:
    """Canonical agent role definition.

    Attributes:
        name: Unique role identifier
        display_name: Human-readable name
        description: Role purpose and responsibilities
        persona: System prompt/persona for the agent
        system_prompt_template: Path to prompt template file
        tool_categories: Categories of tools this role can use
        capabilities: Specific capabilities this role has
        config: Additional role configuration
    """

    name: str
    display_name: str
    description: str
    persona: str = ""
    system_prompt_template: str = ""
    tool_categories: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"AgentRole(name={self.name}, display_name={self.display_name})"


@dataclass
class TeamSpecification:
    """Canonical team specification.

    Attributes:
        name: Unique team identifier
        display_name: Human-readable name
        description: Team purpose and use cases
        formation: Team formation pattern
        roles: List of agent roles in the team
        communication_style: How agents communicate (structured, ad_hoc, broadcast)
        voting_strategy: Voting mechanism for consensus (optional)
        max_iterations: Maximum iterations before stopping
    """

    name: str
    display_name: str
    description: str
    formation: str
    roles: list[AgentRole]
    communication_style: str = "structured"
    voting_strategy: Optional[str] = None
    max_iterations: int = 10

    def get_role(self, role_name: str) -> AgentRole | None:
        """Get role by name.

        Args:
            role_name: Name of the role

        Returns:
            AgentRole instance or None
        """
        for role in self.roles:
            if role.name == role_name:
                return role
        return None

    def list_roles(self) -> list[str]:
        """List all role names.

        Returns:
            List of role names
        """
        return [role.name for role in self.roles]

    def __repr__(self) -> str:
        return f"TeamSpecification(name={self.name}, roles={len(self.roles)}, formation={self.formation})"


class BaseYAMLTeamProvider:
    """Base provider for YAML-based team specifications.

    This provider loads team formations from YAML configuration files,
    enabling data-driven team composition without code changes.

    Example:
        provider = BaseYAMLTeamProvider("coding")
        teams = provider.list_teams()
        review_team = provider.get_team("code_review_team")
    """

    _instances: dict[str, "BaseYAMLTeamProvider"] = {}

    def __init__(self, vertical_name: str, config_dir: Path | None = None):
        """Initialize team provider.

        Args:
            vertical_name: Name of the vertical
            config_dir: Directory containing team YAML files
        """
        self._vertical_name = vertical_name
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "config" / "teams"
        self._config_dir = Path(config_dir)
        self._teams: dict[str, TeamSpecification] = {}

    @classmethod
    def get_provider(cls, vertical_name: str) -> "BaseYAMLTeamProvider":
        """Get or create team provider for vertical.

        Args:
            vertical_name: Name of the vertical

        Returns:
            BaseYAMLTeamProvider instance
        """
        if vertical_name not in cls._instances:
            cls._instances[vertical_name] = cls(vertical_name)
        return cls._instances[vertical_name]

    def load_teams(self) -> dict[str, TeamSpecification]:
        """Load all teams from YAML files.

        Returns:
            Dictionary mapping team names to TeamSpecification instances
        """
        if self._teams:
            return self._teams

        config_file = self._config_dir / f"{self._vertical_name}_teams.yaml"
        if config_file.exists():
            self._teams = self._load_from_yaml(config_file)
        else:
            logger.warning(f"No team config found for {self._vertical_name}")
            self._teams = {}

        return self._teams

    def _load_from_yaml(self, path: Path) -> dict[str, TeamSpecification]:
        """Load teams from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Dictionary of TeamSpecification instances
        """
        try:
            with open(path) as f:
                data = yaml.safe_load(f)

            teams = {}
            for team_data in data.get("teams", []):
                roles = []
                for role_data in team_data.get("roles", []):
                    role = AgentRole(
                        name=role_data.get("name", ""),
                        display_name=role_data.get("display_name", ""),
                        description=role_data.get("description", ""),
                        persona=role_data.get("persona", ""),
                        system_prompt_template=role_data.get("system_prompt_template", ""),
                        tool_categories=role_data.get("tool_categories", []),
                        capabilities=role_data.get("capabilities", []),
                        config=role_data.get("config", {}),
                    )
                    roles.append(role)

                team = TeamSpecification(
                    name=team_data.get("name", ""),
                    display_name=team_data.get("display_name", ""),
                    description=team_data.get("description", ""),
                    formation=team_data.get("formation", "pipeline"),
                    roles=roles,
                    communication_style=team_data.get("communication_style", "structured"),
                    voting_strategy=team_data.get("voting_strategy"),
                    max_iterations=team_data.get("max_iterations", 10),
                )
                teams[team.name] = team

            logger.debug(f"Loaded {len(teams)} teams from {path}")
            return teams
        except Exception as e:
            logger.error(f"Failed to load teams from {path}: {e}")
            return {}

    def get_team(self, team_name: str) -> TeamSpecification | None:
        """Get specific team specification.

        Args:
            team_name: Name of the team

        Returns:
            TeamSpecification instance or None
        """
        teams = self.load_teams()
        return teams.get(team_name)

    def list_teams(self) -> list[str]:
        """List available teams.

        Returns:
            List of team names
        """
        teams = self.load_teams()
        return list(teams.keys())

    def get_all_teams(self) -> dict[str, TeamSpecification]:
        """Get all team specifications.

        Returns:
            Dictionary mapping team names to specifications
        """
        return self.load_teams()

    def invalidate_cache(self) -> None:
        """Invalidate cached team data.

        This forces a reload from YAML on next access.
        """
        self._teams.clear()


__all__ = [
    "TeamFormationType",
    "AgentRole",
    "TeamSpecification",
    "BaseYAMLTeamProvider",
]
