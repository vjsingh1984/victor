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

"""Core team data structures for multi-agent coordination.

This module defines the fundamental building blocks for agent teams:
- TeamFormation: How agents are organized and coordinated
- TeamMember: Individual agent configuration within a team
- TeamConfig: Complete team configuration
- TeamResult: Execution outcome and metrics

Design Principles:
- Immutable configuration (dataclasses)
- Flexible formation patterns (sequential, parallel, hierarchical, pipeline)
- Built on existing SubAgentRole infrastructure
- Serializable for persistence and API transport
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from victor.agent.subagents.base import SubAgentRole


class TeamFormation(Enum):
    """How agents are organized within a team.

    Each formation defines a different coordination pattern:

    - SEQUENTIAL: Agents execute one after another, each receiving context
      from previous agents. Good for dependent tasks.

    - PARALLEL: All agents execute simultaneously on independent aspects
      of the same problem. Good for research and exploration.

    - HIERARCHICAL: A manager agent delegates to worker agents and
      synthesizes their results. Good for complex tasks.

    - PIPELINE: Output of each agent feeds directly into the next.
      Good for multi-stage processing (research → plan → execute → review).
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    PIPELINE = "pipeline"


class MemberStatus(Enum):
    """Status of a team member during execution."""

    IDLE = "idle"
    WORKING = "working"
    DELEGATING = "delegating"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TeamMember:
    """Represents an agent in a team.

    Each member has a specific role and goal within the team context.
    Members can optionally delegate to other team members (in hierarchical
    formations) or be designated as the team manager.

    Attributes:
        id: Unique identifier for this member within the team
        role: SubAgentRole specialization (researcher, planner, executor, etc.)
        name: Human-readable name for display and logging
        goal: Specific objective for this member to achieve
        tool_budget: Maximum tool calls for this member (default: role-based)
        allowed_tools: Override for allowed tools (default: role-based)
        can_delegate: Whether this member can delegate to others
        reports_to: ID of the manager this member reports to (hierarchical)
        is_manager: Whether this member is the team manager
        priority: Execution priority (lower = earlier, for sequential/pipeline)

    Example:
        researcher = TeamMember(
            id="auth_researcher",
            role=SubAgentRole.RESEARCHER,
            name="Authentication Researcher",
            goal="Find all authentication code and patterns",
            tool_budget=20,
        )
    """

    id: str
    role: SubAgentRole
    name: str
    goal: str
    tool_budget: int = 15
    allowed_tools: Optional[List[str]] = None
    can_delegate: bool = False
    reports_to: Optional[str] = None
    is_manager: bool = False
    priority: int = 0

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = f"{self.role.value}_{uuid.uuid4().hex[:8]}"


@dataclass
class TeamConfig:
    """Configuration for an agent team.

    Defines the complete structure of a team including members, formation,
    and resource constraints. Teams execute as a coordinated unit toward
    a shared goal.

    Attributes:
        name: Human-readable team name
        goal: Overall team objective
        members: List of team members
        formation: How agents are organized (default: SEQUENTIAL)
        max_iterations: Maximum total iterations across all members
        total_tool_budget: Total tool calls across all members
        shared_context: Initial context shared with all members
        on_member_complete: Optional callback when a member completes
        allow_dynamic_membership: Can agents spawn new team members?
        timeout_seconds: Maximum total execution time

    Example:
        config = TeamConfig(
            name="Code Review Team",
            goal="Comprehensive code review with fixes",
            members=[researcher, reviewer, executor],
            formation=TeamFormation.PIPELINE,
            total_tool_budget=100,
        )
    """

    name: str
    goal: str
    members: List[TeamMember]
    formation: TeamFormation = TeamFormation.SEQUENTIAL
    max_iterations: int = 50
    total_tool_budget: int = 100
    shared_context: Dict[str, Any] = field(default_factory=dict)
    on_member_complete: Optional[Callable[[str, Any], None]] = None
    allow_dynamic_membership: bool = False
    timeout_seconds: int = 600

    def __post_init__(self):
        """Validate team configuration."""
        if not self.members:
            raise ValueError("Team must have at least one member")

        # Validate member IDs are unique
        member_ids = [m.id for m in self.members]
        if len(member_ids) != len(set(member_ids)):
            raise ValueError("Team member IDs must be unique")

        # For hierarchical, ensure exactly one manager
        if self.formation == TeamFormation.HIERARCHICAL:
            managers = [m for m in self.members if m.is_manager]
            if len(managers) != 1:
                raise ValueError("Hierarchical teams must have exactly one manager")

    def get_member(self, member_id: str) -> Optional[TeamMember]:
        """Get a member by ID."""
        for member in self.members:
            if member.id == member_id:
                return member
        return None

    def get_manager(self) -> Optional[TeamMember]:
        """Get the team manager (if any)."""
        for member in self.members:
            if member.is_manager:
                return member
        return None

    def get_workers(self) -> List[TeamMember]:
        """Get all non-manager members."""
        return [m for m in self.members if not m.is_manager]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "goal": self.goal,
            "members": [
                {
                    "id": m.id,
                    "role": m.role.value,
                    "name": m.name,
                    "goal": m.goal,
                    "tool_budget": m.tool_budget,
                    "can_delegate": m.can_delegate,
                    "reports_to": m.reports_to,
                    "is_manager": m.is_manager,
                    "priority": m.priority,
                }
                for m in self.members
            ],
            "formation": self.formation.value,
            "max_iterations": self.max_iterations,
            "total_tool_budget": self.total_tool_budget,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class MemberResult:
    """Result from a single team member's execution.

    Attributes:
        member_id: ID of the member that produced this result
        success: Whether the member completed successfully
        output: Primary output from the member
        tool_calls_used: Number of tool calls made
        duration_seconds: Execution time
        discoveries: Key findings to share with team
        error: Error message if failed
    """

    member_id: str
    success: bool
    output: str
    tool_calls_used: int
    duration_seconds: float
    discoveries: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class TeamResult:
    """Result from team execution.

    Contains the final outcome, individual member results, and aggregate
    statistics about the team execution.

    Attributes:
        success: Whether the team achieved its goal
        final_output: Synthesized final output from team
        member_results: Results from each team member
        total_tool_calls: Sum of tool calls across all members
        total_duration: Total execution time
        communication_log: Inter-agent messages exchanged
        shared_context: Final state of shared context
        formation_used: Formation pattern that was executed

    Example:
        if result.success:
            print(result.final_output)
            for member_id, member_result in result.member_results.items():
                print(f"{member_id}: {member_result.tool_calls_used} tool calls")
    """

    success: bool
    final_output: str
    member_results: Dict[str, MemberResult]
    total_tool_calls: int
    total_duration: float
    communication_log: List[Dict[str, Any]] = field(default_factory=list)
    shared_context: Dict[str, Any] = field(default_factory=dict)
    formation_used: TeamFormation = TeamFormation.SEQUENTIAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "final_output": self.final_output,
            "member_results": {
                mid: {
                    "member_id": mr.member_id,
                    "success": mr.success,
                    "output": mr.output,
                    "tool_calls_used": mr.tool_calls_used,
                    "duration_seconds": mr.duration_seconds,
                    "discoveries": mr.discoveries,
                    "error": mr.error,
                }
                for mid, mr in self.member_results.items()
            },
            "total_tool_calls": self.total_tool_calls,
            "total_duration": self.total_duration,
            "formation_used": self.formation_used.value,
        }


__all__ = [
    "TeamFormation",
    "MemberStatus",
    "TeamMember",
    "TeamConfig",
    "MemberResult",
    "TeamResult",
]
