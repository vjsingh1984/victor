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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

from victor.agent.subagents.base import SubAgentRole

if TYPE_CHECKING:
    from victor.agent.protocols import UnifiedMemoryCoordinatorProtocol


@dataclass
class MemoryConfig:
    """Configuration for per-agent memory behavior.

    Controls how an agent stores and retrieves memories across task executions.
    This allows fine-grained control over memory persistence, search behavior,
    and memory type preferences.

    Attributes:
        enabled: Whether memory is enabled for this agent
        persist_across_sessions: Store memories beyond current session
        search_own_memories_only: Only search this agent's memories (vs team)
        memory_types: Types of memories to store (entity, episodic, semantic)
        max_memories_per_query: Maximum memories returned per search
        relevance_threshold: Minimum relevance score for memory retrieval (0.0-1.0)
        auto_summarize: Automatically summarize long memories
        ttl_seconds: Time-to-live for memories (None = permanent)

    Example:
        memory_config = MemoryConfig(
            enabled=True,
            persist_across_sessions=True,
            memory_types={"entity", "episodic"},
            max_memories_per_query=20,
            relevance_threshold=0.7,
        )
    """

    enabled: bool = True
    persist_across_sessions: bool = False
    search_own_memories_only: bool = False
    memory_types: Set[str] = field(default_factory=lambda: {"entity", "episodic", "semantic"})
    max_memories_per_query: int = 10
    relevance_threshold: float = 0.5
    auto_summarize: bool = True
    ttl_seconds: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "persist_across_sessions": self.persist_across_sessions,
            "search_own_memories_only": self.search_own_memories_only,
            "memory_types": list(self.memory_types),
            "max_memories_per_query": self.max_memories_per_query,
            "relevance_threshold": self.relevance_threshold,
            "auto_summarize": self.auto_summarize,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryConfig":
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            persist_across_sessions=data.get("persist_across_sessions", False),
            search_own_memories_only=data.get("search_own_memories_only", False),
            memory_types=set(data.get("memory_types", ["entity", "episodic", "semantic"])),
            max_memories_per_query=data.get("max_memories_per_query", 10),
            relevance_threshold=data.get("relevance_threshold", 0.5),
            auto_summarize=data.get("auto_summarize", True),
            ttl_seconds=data.get("ttl_seconds"),
        )


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
    """Represents an agent in a team with rich persona support.

    Each member has a specific role, goal, and optional persona attributes
    within the team context. Members can delegate to other team members
    (in hierarchical formations) or be designated as the team manager.

    The persona system is inspired by CrewAI and allows rich characterization
    of agents through backstory, expertise domains, and personality traits.

    Attributes:
        id: Unique identifier for this member within the team
        role: SubAgentRole specialization (researcher, planner, executor, etc.)
        name: Human-readable name for display and logging
        goal: Specific objective for this member to achieve
        tool_budget: Maximum tool calls for this member (default: role-based)
        allowed_tools: Override for allowed tools (default: role-based)
        can_delegate: Whether this member can delegate to others
        delegation_targets: Specific member IDs this member can delegate to
        reports_to: ID of the manager this member reports to (hierarchical)
        is_manager: Whether this member is the team manager
        priority: Execution priority (lower = earlier, for sequential/pipeline)
        backstory: Rich persona description defining agent's history and character
        expertise: List of domain expertise areas for context-aware behavior
        personality: Communication style and behavioral traits
        max_delegation_depth: Maximum levels of delegation allowed (0 = no delegation)
        memory: Whether to persist discoveries across tasks (simple flag)
        memory_config: Detailed memory configuration (overrides simple memory flag)
        cache: Whether to cache tool results
        verbose: Whether to show detailed execution logs
        max_iterations: Per-member iteration limit (None = use team default)

    Example:
        # Basic usage with minimal configuration
        researcher = TeamMember(
            id="auth_researcher",
            role=SubAgentRole.RESEARCHER,
            name="Authentication Researcher",
            goal="Find all authentication code and patterns",
        )

        # Rich persona with full CrewAI-style attributes
        security_analyst = TeamMember(
            id="security_analyst",
            role=SubAgentRole.RESEARCHER,
            name="Security Analyst",
            goal="Find authentication vulnerabilities",
            backstory="10 years of security experience at major tech companies. "
                      "Previously led red team exercises at a Fortune 500 company. "
                      "Known for finding subtle authentication bypass bugs.",
            expertise=["security", "authentication", "oauth", "jwt", "session-management"],
            personality="methodical and thorough; prefers depth over breadth; "
                        "communicates findings with severity ratings",
            max_delegation_depth=2,
            memory=True,
            memory_config=MemoryConfig(
                enabled=True,
                persist_across_sessions=True,
                memory_types={"entity", "semantic"},
                relevance_threshold=0.7,
            ),
            cache=True,
            tool_budget=25,
        )

        # Team manager with delegation capabilities
        tech_lead = TeamMember(
            id="tech_lead",
            role=SubAgentRole.PLANNER,
            name="Technical Lead",
            goal="Coordinate team to deliver secure authentication system",
            backstory="15 years as a software architect, now leading teams.",
            expertise=["architecture", "team-leadership", "security", "scalability"],
            personality="collaborative; focuses on big picture; empowers team members",
            is_manager=True,
            can_delegate=True,
            max_delegation_depth=3,
            delegation_targets=["security_analyst", "code_implementer", "test_writer"],
            tool_budget=15,
        )
    """

    id: str
    role: SubAgentRole
    name: str
    goal: str
    tool_budget: int = 15
    allowed_tools: Optional[List[str]] = None
    can_delegate: bool = False
    delegation_targets: Optional[List[str]] = None
    reports_to: Optional[str] = None
    is_manager: bool = False
    priority: int = 0
    # Rich persona attributes (CrewAI-compatible)
    backstory: str = ""
    expertise: List[str] = field(default_factory=list)
    personality: str = ""
    max_delegation_depth: int = 0
    memory: bool = False
    memory_config: Optional[MemoryConfig] = None
    cache: bool = True
    verbose: bool = False
    max_iterations: Optional[int] = None
    # Memory coordinator for persistent memory across tasks
    memory_coordinator: Optional["UnifiedMemoryCoordinatorProtocol"] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = f"{self.role.value}_{uuid.uuid4().hex[:8]}"

    def to_system_prompt(self) -> str:
        """Generate system prompt from member persona.

        Creates a rich system prompt incorporating all persona attributes
        including backstory, expertise, personality, and delegation capabilities.

        Returns:
            System prompt incorporating role, goal, backstory, expertise,
            personality, and delegation information

        Example:
            prompt = member.to_system_prompt()
            # Returns formatted prompt like:
            # # Role: Security Analyst
            # You are a researcher agent.
            #
            # ## Goal
            # Find authentication vulnerabilities
            #
            # ## Background
            # 10 years of security experience...
            #
            # ## Expertise
            # Your areas of expertise: security, authentication, oauth
            #
            # ## Communication Style
            # methodical and thorough; communicates findings with severity ratings
        """
        lines = [
            f"# Role: {self.name}",
            f"You are a {self.role.value} agent.",
            "",
            "## Goal",
            self.goal,
            "",
        ]

        if self.backstory:
            lines.extend(
                [
                    "## Background",
                    self.backstory,
                    "",
                ]
            )

        if self.expertise:
            lines.extend(
                [
                    "## Expertise",
                    f"Your areas of expertise: {', '.join(self.expertise)}",
                    "",
                ]
            )

        if self.personality:
            lines.extend(
                [
                    "## Communication Style",
                    self.personality,
                    "",
                ]
            )

        if self.can_delegate:
            targets = self.delegation_targets or []
            depth_info = ""
            if self.max_delegation_depth > 0:
                depth_info = f" (max {self.max_delegation_depth} levels deep)"
            if targets:
                lines.extend(
                    [
                        "## Delegation",
                        f"You can delegate tasks to: {', '.join(targets)}{depth_info}",
                        "",
                    ]
                )
            else:
                lines.extend(
                    [
                        "## Delegation",
                        f"You can delegate tasks to other team members when appropriate{depth_info}.",
                        "",
                    ]
                )

        return "\n".join(lines)

    @property
    def memory_enabled(self) -> bool:
        """Check if memory is enabled for this member.

        Memory is enabled if either the simple `memory` flag is True,
        or if `memory_config` is provided with `enabled=True`.

        Returns:
            True if memory is enabled, False otherwise
        """
        if self.memory_config is not None:
            return self.memory_config.enabled
        return self.memory

    def get_memory_config(self) -> MemoryConfig:
        """Get the effective memory configuration.

        Returns the explicit memory_config if set, otherwise creates
        a default MemoryConfig based on the simple memory flag.

        Returns:
            MemoryConfig for this member
        """
        if self.memory_config is not None:
            return self.memory_config
        return MemoryConfig(enabled=self.memory)

    async def remember(
        self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store a discovery in memory for future tasks.

        Only works if memory is enabled and a memory_coordinator is attached.
        Uses memory_config settings if available for fine-grained control.

        Args:
            key: Unique key for the memory
            value: Value to store (discovery, finding, etc.)
            metadata: Optional metadata about the discovery

        Returns:
            True if stored successfully, False otherwise

        Example:
            await member.remember(
                "auth_patterns",
                {"patterns": ["JWT", "OAuth2"], "files": ["auth.py"]},
                metadata={"task": "authentication_review"}
            )
        """
        if not self.memory_enabled or not self.memory_coordinator:
            return False

        try:
            from victor.memory.unified import MemoryType

            full_metadata = {"member_id": self.id, "member_name": self.name}
            if self.expertise:
                full_metadata["expertise"] = self.expertise
            if metadata:
                full_metadata.update(metadata)

            # Apply TTL from memory config if set
            config = self.get_memory_config()
            if config.ttl_seconds:
                full_metadata["ttl_seconds"] = config.ttl_seconds

            return await self.memory_coordinator.store(
                MemoryType.ENTITY,
                f"{self.id}:{key}",
                value,
                metadata=full_metadata,
            )
        except Exception:
            return False

    async def recall(self, query: str, limit: Optional[int] = None) -> list:
        """Recall memories relevant to a query.

        Searches across memory types for relevant discoveries from this
        member's previous tasks. Uses memory_config settings if available.

        Args:
            query: Search query
            limit: Maximum results to return (default from memory_config)

        Returns:
            List of memory results

        Example:
            memories = await member.recall("authentication patterns")
            for mem in memories:
                print(f"Found: {mem.content}")
        """
        if not self.memory_enabled or not self.memory_coordinator:
            return []

        try:
            config = self.get_memory_config()
            effective_limit = limit or config.max_memories_per_query

            # Build filters based on memory config
            filters: Dict[str, Any] = {}
            if config.search_own_memories_only:
                filters["member_id"] = self.id

            results = await self.memory_coordinator.search_all(
                query=query,
                limit=effective_limit,
                filters=filters if filters else {"member_id": self.id},
            )

            # Apply relevance threshold from config
            if config.relevance_threshold > 0:
                results = [
                    r for r in results if getattr(r, "score", 1.0) >= config.relevance_threshold
                ]

            return results
        except Exception:
            return []

    def attach_memory_coordinator(
        self,
        coordinator: "UnifiedMemoryCoordinatorProtocol",
    ) -> None:
        """Attach a memory coordinator to this member.

        Call this to enable persistent memory when memory=True.

        Args:
            coordinator: UnifiedMemoryCoordinator instance

        Example:
            from victor.memory import get_memory_coordinator
            member.attach_memory_coordinator(get_memory_coordinator())
        """
        self.memory_coordinator = coordinator


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
                    "delegation_targets": m.delegation_targets,
                    "reports_to": m.reports_to,
                    "is_manager": m.is_manager,
                    "priority": m.priority,
                    # Rich persona attributes (CrewAI-compatible)
                    "backstory": m.backstory,
                    "expertise": m.expertise,
                    "personality": m.personality,
                    "max_delegation_depth": m.max_delegation_depth,
                    "memory": m.memory,
                    "memory_config": m.memory_config.to_dict() if m.memory_config else None,
                    "cache": m.cache,
                    "verbose": m.verbose,
                    "max_iterations": m.max_iterations,
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
    "MemoryConfig",
    "TeamFormation",
    "MemberStatus",
    "TeamMember",
    "TeamConfig",
    "MemberResult",
    "TeamResult",
]
