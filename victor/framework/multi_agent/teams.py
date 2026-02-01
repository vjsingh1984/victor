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

"""Generic team structures for multi-agent collaboration."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from victor.framework.multi_agent.personas import PersonaTraits


class TeamTopology(Enum):
    """Team organization topologies.

    Defines how agents are organized and communicate within a team.
    """

    HIERARCHY = "hierarchy"
    """Tree structure with manager delegating to subordinates."""

    MESH = "mesh"
    """Fully connected network where any agent can communicate with any other."""

    PIPELINE = "pipeline"
    """Linear sequence where output of one agent feeds into the next."""

    HUB_SPOKE = "hub_spoke"
    """Central coordinator with specialized workers around it."""


class TaskAssignmentStrategy(Enum):
    """Strategies for assigning tasks to team members.

    Defines the algorithm used to match tasks with appropriate agents.
    """

    ROUND_ROBIN = "round_robin"
    """Distribute tasks evenly across members in rotation."""

    SKILL_MATCH = "skill_match"
    """Assign tasks based on member skills and expertise."""

    LOAD_BALANCED = "load_balanced"
    """Assign tasks to least busy available member."""


@dataclass
class TeamMember:
    """A member of a multi-agent team.

    TeamMember combines a PersonaTraits with team-specific configuration
    like role within the team, leadership status, and resource constraints.

    Attributes:
        persona: The persona traits for this member.
        role_in_team: The specific role this member plays in the team.
        is_leader: Whether this member is the team leader.
        max_concurrent_tasks: Maximum tasks this member can handle at once.
        tool_access: List of tools this member is allowed to use.

    Example:
        researcher_persona = PersonaTraits(
            name="Research Lead",
            role="researcher",
            description="Leads research efforts",
        )
        member = TeamMember(
            persona=researcher_persona,
            role_in_team="lead_researcher",
            is_leader=True,
            tool_access=["web_search", "read_file"],
        )
    """

    persona: PersonaTraits
    role_in_team: str
    is_leader: bool = False
    max_concurrent_tasks: int = 1
    tool_access: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Get the member's display name from persona."""
        return self.persona.name

    @property
    def expertise_level(self) -> Any:
        """Get the member's expertise level from persona."""
        return self.persona.expertise_level


@dataclass
class TeamTemplate:
    """Generic template for team configuration.

    TeamTemplate defines the structure and policies for a team without
    specifying concrete members. It can be used to create consistent
    team configurations across different projects.

    Attributes:
        name: Display name for this team template.
        description: Description of the team's purpose and capabilities.
        topology: How agents are organized in the team.
        assignment_strategy: How tasks are assigned to members.
        member_slots: Required roles and their counts (e.g., {"researcher": 2}).
        shared_context_keys: Context keys shared across all members.
        escalation_threshold: Confidence threshold for escalating to leader (0.0-1.0).
        max_iterations: Maximum iterations for team execution.
        config: Additional team-specific configuration.

    Example:
        template = TeamTemplate(
            name="Code Review Team",
            description="Reviews code for quality and correctness",
            topology=TeamTopology.PIPELINE,
            assignment_strategy=TaskAssignmentStrategy.SKILL_MATCH,
            member_slots={"researcher": 1, "reviewer": 2, "approver": 1},
            escalation_threshold=0.7,
        )
    """

    name: str
    description: str
    topology: TeamTopology = TeamTopology.HIERARCHY
    assignment_strategy: TaskAssignmentStrategy = TaskAssignmentStrategy.SKILL_MATCH
    member_slots: dict[str, int] = field(default_factory=dict)
    shared_context_keys: list[str] = field(default_factory=list)
    escalation_threshold: float = 0.8
    max_iterations: int = 10
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate template values after initialization."""
        if not 0.0 <= self.escalation_threshold <= 1.0:
            raise ValueError(
                f"escalation_threshold must be between 0.0 and 1.0, "
                f"got {self.escalation_threshold}"
            )
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")

    def to_dict(self) -> dict[str, Any]:
        """Convert template to dictionary for serialization.

        Returns:
            Dictionary representation with enum values converted to strings.
        """
        return {
            "name": self.name,
            "description": self.description,
            "topology": self.topology.value,
            "assignment_strategy": self.assignment_strategy.value,
            "member_slots": self.member_slots,
            "shared_context_keys": self.shared_context_keys,
            "escalation_threshold": self.escalation_threshold,
            "max_iterations": self.max_iterations,
            "config": self.config,
        }


@dataclass
class TeamSpec:
    """Concrete team specification with assigned members.

    TeamSpec combines a TeamTemplate with actual TeamMember instances
    to create a fully configured team ready for execution.

    Attributes:
        template: The template defining team structure and policies.
        members: List of team members.

    Example:
        template = TeamTemplate(
            name="Review Team",
            description="Code review team",
            topology=TeamTopology.PIPELINE,
        )
        spec = TeamSpec(
            template=template,
            members=[
                TeamMember(persona=researcher, role_in_team="researcher"),
                TeamMember(persona=reviewer, role_in_team="reviewer", is_leader=True),
            ],
        )
        leader = spec.leader  # Returns the reviewer
    """

    template: TeamTemplate
    members: list[TeamMember] = field(default_factory=list)

    @property
    def leader(self) -> Optional[TeamMember]:
        """Get the team leader if one exists.

        Returns:
            The TeamMember marked as leader, or None if no leader is set.
        """
        for member in self.members:
            if member.is_leader:
                return member
        return None

    @property
    def name(self) -> str:
        """Get the team name from template."""
        return self.template.name

    @property
    def topology(self) -> TeamTopology:
        """Get the team topology from template."""
        return self.template.topology

    def get_members_by_role(self, role: str) -> list[TeamMember]:
        """Get all members with a specific role.

        Args:
            role: The role_in_team to filter by.

        Returns:
            List of TeamMembers with the specified role.
        """
        return [m for m in self.members if m.role_in_team == role]

    def validate_slots(self) -> list[str]:
        """Validate that member assignments match template slots.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []
        role_counts: dict[str, int] = {}
        for member in self.members:
            role_counts[member.role_in_team] = role_counts.get(member.role_in_team, 0) + 1

        for role, required_count in self.template.member_slots.items():
            actual_count = role_counts.get(role, 0)
            if actual_count < required_count:
                errors.append(
                    f"Role '{role}' requires {required_count} members, got {actual_count}"
                )

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert spec to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "template": self.template.to_dict(),
            "members": [
                {
                    "persona": m.persona.to_dict(),
                    "role_in_team": m.role_in_team,
                    "is_leader": m.is_leader,
                    "max_concurrent_tasks": m.max_concurrent_tasks,
                    "tool_access": m.tool_access,
                }
                for m in self.members
            ],
        }


__all__ = [
    "TaskAssignmentStrategy",
    "TeamMember",
    "TeamSpec",
    "TeamTemplate",
    "TeamTopology",
]
