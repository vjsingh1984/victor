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

"""Unified team specification schema.

This module provides a single TeamSpec schema that all verticals use,
eliminating the duplication of CodingTeamSpec, ResearchTeamSpec, etc.

Design Goals:
- Single source of truth for team specifications
- Verticals provide data only, not schema definitions
- Support for rich persona attributes (backstory, personality, expertise)
- Tool name canonicalization at schema level

Usage:
    from victor.framework.team_schema import TeamSpec, RoleConfig

    # Define a team spec
    code_review_team = TeamSpec(
        name="Code Review Team",
        description="Reviews code changes for quality",
        vertical="coding",
        formation=TeamFormation.PIPELINE,
        members=[...],
    )

    # Register with the registry
    from victor.framework.team_registry import get_team_registry
    registry = get_team_registry()
    registry.register("coding:code_review", code_review_team)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from victor.framework.teams import TeamFormation, TeamMemberSpec
from victor.framework.tool_naming import canonicalize_tool_list


@dataclass
class RoleConfig:
    """Configuration for a role within a vertical.

    Provides a standardized way to define roles that can be
    used across different team configurations.

    Attributes:
        base_role: Base agent role (e.g., "developer", "analyst", "reviewer")
        tools: Tools available to this role (will be canonicalized)
        tool_budget: Default tool budget for this role
        description: Role description
    """

    base_role: str
    tools: List[str]
    tool_budget: int
    description: str = ""

    def __post_init__(self) -> None:
        """Canonicalize tool names on creation."""
        if self.tools:
            self.tools = canonicalize_tool_list(self.tools)


@dataclass
class TeamSpec:
    """Universal team specification schema.

    This is the single, unified schema for team specifications
    across all verticals. Verticals provide data using this schema
    rather than defining their own dataclasses.

    Attributes:
        name: Team name (e.g., "Code Review Team")
        description: Team description
        vertical: Source vertical (e.g., "coding", "research", "devops")
        formation: How agents are organized (PIPELINE, SEQUENTIAL, etc.)
        members: Team member specifications
        total_tool_budget: Total tool budget for the team
        max_iterations: Maximum iterations for team execution
        tags: Tags for discovery and filtering
        task_types: Task types this team can handle
        metadata: Additional metadata
    """

    name: str
    description: str
    vertical: str
    formation: TeamFormation
    members: List[TeamMemberSpec]
    total_tool_budget: int = 100
    max_iterations: int = 50
    tags: List[str] = field(default_factory=list)
    task_types: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Canonicalize tool names in members on creation."""
        for member in self.members:
            if hasattr(member, "tools") and member.tools:
                # TeamMemberSpec may not have tools, check first
                pass  # Canonicalization happens in registry

    @property
    def member_count(self) -> int:
        """Get the number of team members."""
        return len(self.members)

    @property
    def roles(self) -> Set[str]:
        """Get the set of roles in this team."""
        return {m.role for m in self.members}

    def get_member_by_role(self, role: str) -> Optional[TeamMemberSpec]:
        """Get a member by role.

        Args:
            role: Role to find

        Returns:
            TeamMemberSpec or None
        """
        for member in self.members:
            if member.role == role:
                return member
        return None

    def get_members_by_role(self, role: str) -> List[TeamMemberSpec]:
        """Get all members with a specific role.

        Args:
            role: Role to find

        Returns:
            List of matching members
        """
        return [m for m in self.members if m.role == role]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "description": self.description,
            "vertical": self.vertical,
            "formation": (
                self.formation.value if hasattr(self.formation, "value") else str(self.formation)
            ),
            "members": [
                {
                    "role": m.role,
                    "goal": m.goal,
                    "name": getattr(m, "name", None),
                    "tool_budget": m.tool_budget,
                    "backstory": getattr(m, "backstory", None),
                    "expertise": getattr(m, "expertise", None),
                    "personality": getattr(m, "personality", None),
                    "memory": getattr(m, "memory", False),
                    "is_manager": getattr(m, "is_manager", False),
                }
                for m in self.members
            ],
            "total_tool_budget": self.total_tool_budget,
            "max_iterations": self.max_iterations,
            "tags": self.tags,
            "task_types": self.task_types,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeamSpec":
        """Create from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            TeamSpec instance
        """
        members = []
        for m_data in data.get("members", []):
            member = TeamMemberSpec(
                role=m_data["role"],
                goal=m_data["goal"],
                name=m_data.get("name"),
                tool_budget=m_data.get("tool_budget", 10),
                backstory=m_data.get("backstory"),
                expertise=m_data.get("expertise"),
                personality=m_data.get("personality"),
                memory=m_data.get("memory", False),
                is_manager=m_data.get("is_manager", False),
            )
            members.append(member)

        formation_str = data.get("formation", "SEQUENTIAL")
        if isinstance(formation_str, str):
            formation = TeamFormation[formation_str.upper()]
        else:
            formation = formation_str

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            vertical=data["vertical"],
            formation=formation,
            members=members,
            total_tool_budget=data.get("total_tool_budget", 100),
            max_iterations=data.get("max_iterations", 50),
            tags=data.get("tags", []),
            task_types=data.get("task_types", []),
            metadata=data.get("metadata", {}),
        )


def create_team_spec(
    name: str,
    description: str,
    vertical: str,
    formation: TeamFormation,
    members: List[TeamMemberSpec],
    **kwargs: Any,
) -> TeamSpec:
    """Factory function for creating team specs.

    Provides a cleaner API for creating team specifications.

    Args:
        name: Team name
        description: Team description
        vertical: Source vertical
        formation: Team formation
        members: Team members
        **kwargs: Additional TeamSpec attributes

    Returns:
        TeamSpec instance
    """
    return TeamSpec(
        name=name,
        description=description,
        vertical=vertical,
        formation=formation,
        members=members,
        **kwargs,
    )


__all__ = [
    "TeamSpec",
    "RoleConfig",
    "create_team_spec",
]
