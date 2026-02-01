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

"""Canonical types for multi-agent team coordination.

This module is the SINGLE SOURCE OF TRUTH for all team-related types.
Import from here, not from victor.framework.agent_protocols or victor.agent.teams.

Example:
    from victor.teams.types import TeamFormation, MessageType, AgentMessage

Type Consolidation:
    - TeamFormation: Unified from framework (5 values) and agent/teams (4 values)
    - MessageType: Unified from framework (5 values) and agent/teams (7 values)
    - AgentMessage: Superset of both implementations
    - MemberResult: Superset of both implementations
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional
from collections.abc import Awaitable, Callable

if TYPE_CHECKING:
    # Team member dependencies
    from victor.agent.subagents.base import SubAgentRole
    from victor.agent.protocols import UnifiedMemoryCoordinatorProtocol
    from victor.agent.presentation import PresentationProtocol


class TeamFormation(str, Enum):
    """Team organization patterns for multi-agent coordination.

    Unified superset combining:
    - Framework formations: SEQUENTIAL, PARALLEL, HIERARCHICAL, PIPELINE, CONSENSUS
    - Agent/teams formations: SEQUENTIAL, PARALLEL, HIERARCHICAL, PIPELINE

    Basic Formations:
        SEQUENTIAL: Execute members one after another, context chaining
        PARALLEL: Execute all members simultaneously, independent work
        HIERARCHICAL: Manager delegates to workers, synthesizes results
        PIPELINE: Output of one member feeds into the next
        CONSENSUS: All members must agree (multiple rounds if needed)
    """

    # Basic formations
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    PIPELINE = "pipeline"
    CONSENSUS = "consensus"


class MessageType(str, Enum):
    """Types of messages for inter-agent communication.

    Unified superset combining:
    - Framework types: TASK, RESULT, QUERY, FEEDBACK, DELEGATION
    - Agent/teams types: DISCOVERY, REQUEST, RESPONSE, STATUS, ALERT, HANDOFF, RESULT

    Attributes:
        TASK: Task assignment message
        RESULT: Result from task execution
        QUERY: Information request
        FEEDBACK: Feedback on work
        DELEGATION: Task delegation request
        DISCOVERY: Discovery/finding announcement
        REQUEST: Generic request
        RESPONSE: Response to a request
        STATUS: Status update
        ALERT: Alert/warning message
        HANDOFF: Task handoff between agents
    """

    # From framework/agent_protocols.py
    TASK = "task"
    RESULT = "result"
    QUERY = "query"
    FEEDBACK = "feedback"
    DELEGATION = "delegation"

    # From agent/teams/communication.py
    DISCOVERY = "discovery"
    REQUEST = "request"
    RESPONSE = "response"
    STATUS = "status"
    ALERT = "alert"
    HANDOFF = "handoff"


class MessagePriority(int, Enum):
    """Priority levels for agent messages."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class AgentMessage:
    """Unified agent message for inter-agent communication.

    Superset combining:
    - Framework AgentMessage: sender_id, content, message_type, recipient_id, metadata
    - Agent/teams AgentMessage: type, from_agent, to_agent, content, data, timestamp,
                                reply_to, priority, id

    This canonical version uses consistent naming and supports all features.

    Attributes:
        sender_id: ID of the sending agent
        content: Message content string
        message_type: Type of message (MessageType enum)
        recipient_id: ID of the recipient (None for broadcast)
        data: Structured data payload
        timestamp: Unix timestamp when created
        reply_to: ID of message this is replying to
        priority: Message priority level
        id: Unique message identifier
    """

    sender_id: str
    content: str
    message_type: MessageType
    recipient_id: Optional[str] = None  # None = broadcast to all
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None
    priority: MessagePriority = MessagePriority.NORMAL
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # Compatibility aliases for framework AgentMessage
    @property
    def metadata(self) -> dict[str, Any]:
        """Alias for data (framework compatibility)."""
        return self.data

    # Compatibility aliases for agent/teams AgentMessage
    @property
    def from_agent(self) -> str:
        """Alias for sender_id (agent/teams compatibility)."""
        return self.sender_id

    @property
    def to_agent(self) -> Optional[str]:
        """Alias for recipient_id (agent/teams compatibility)."""
        return self.recipient_id

    @property
    def type(self) -> MessageType:
        """Alias for message_type (agent/teams compatibility)."""
        return self.message_type

    def is_broadcast(self) -> bool:
        """Check if this is a broadcast message."""
        return self.recipient_id is None

    def is_reply(self) -> bool:
        """Check if this is a reply to another message."""
        return self.reply_to is not None

    def to_context_string(self, presentation: Optional["PresentationProtocol"] = None) -> str:
        """Format message for inclusion in agent context.

        Args:
            presentation: Optional presentation adapter for icons.
                If None, creates default adapter.

        Returns:
            Formatted string for agent context.
        """
        if presentation is None:
            from victor.agent.presentation import create_presentation_adapter

            presentation = create_presentation_adapter()

        arrow = presentation.icon("arrow_right", with_color=False)
        header = f"[{self.message_type.value.upper()}] {self.sender_id}"
        if self.recipient_id:
            header += f" {arrow} {self.recipient_id}"
        return f"{header}: {self.content}"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "content": self.content,
            "data": self.data,
            "timestamp": self.timestamp,
            "reply_to": self.reply_to,
            "priority": self.priority.value,
        }


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
    memory_types: set[str] = field(default_factory=lambda: {"entity", "episodic", "semantic"})
    max_memories_per_query: int = 10
    relevance_threshold: float = 0.5
    auto_summarize: bool = True
    ttl_seconds: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> "MemoryConfig":
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


class MemberStatus(str, Enum):
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
    role: "SubAgentRole"
    name: str
    goal: str
    tool_budget: int = 15
    allowed_tools: Optional[list[str]] = None
    can_delegate: bool = False
    delegation_targets: Optional[list[str]] = None
    reports_to: Optional[str] = None
    is_manager: bool = False
    priority: int = 0
    # Rich persona attributes (CrewAI-compatible)
    backstory: str = ""
    expertise: list[str] = field(default_factory=list)
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

    def __post_init__(self) -> None:
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
        self, key: str, value: Any, metadata: Optional[dict[str, Any]] = None
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
            from victor.storage.memory.unified import MemoryType

            full_metadata: dict[str, Any] = {"member_id": self.id, "member_name": self.name}
            if self.expertise:
                full_metadata["expertise"] = ",".join(self.expertise)
            if metadata:
                full_metadata.update(metadata)

            # Apply TTL from memory config if set
            config = self.get_memory_config()
            if config.ttl_seconds is not None:
                full_metadata["ttl_seconds"] = str(config.ttl_seconds)

            return await self.memory_coordinator.store(
                MemoryType.ENTITY,
                f"{self.id}:{key}",
                value,
                metadata=full_metadata,
            )
        except Exception:
            return False

    async def recall(self, query: str, limit: Optional[int] = None) -> list[Any]:
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
            filters: dict[str, Any] = {}
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
            from victor.storage.memory import get_memory_coordinator
            member.attach_memory_coordinator(get_memory_coordinator())
        """
        self.memory_coordinator = coordinator

    def to_dict(self) -> dict[str, Any]:
        """Convert TeamMember to dictionary for serialization.

        Returns:
            Dictionary representation of the member

        Example:
            member_dict = member.to_dict()
        """
        return {
            "id": self.id,
            "role": self.role.value,
            "name": self.name,
            "goal": self.goal,
            "tool_budget": self.tool_budget,
            "allowed_tools": self.allowed_tools,
            "can_delegate": self.can_delegate,
            "delegation_targets": self.delegation_targets,
            "reports_to": self.reports_to,
            "is_manager": self.is_manager,
            "priority": self.priority,
            # Rich persona attributes (CrewAI-compatible)
            "backstory": self.backstory,
            "expertise": self.expertise,
            "personality": self.personality,
            "max_delegation_depth": self.max_delegation_depth,
            "memory": self.memory,
            "memory_config": self.memory_config.to_dict() if self.memory_config else None,
            "cache": self.cache,
            "verbose": self.verbose,
            "max_iterations": self.max_iterations,
        }


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
    members: list[TeamMember]
    formation: TeamFormation = TeamFormation.SEQUENTIAL
    max_iterations: int = 50
    total_tool_budget: int = 100
    shared_context: dict[str, Any] = field(default_factory=dict)
    on_member_complete: Optional[Callable[[str, Any], None]] = None
    allow_dynamic_membership: bool = False
    timeout_seconds: int = 600

    def __post_init__(self) -> None:
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

    def get_workers(self) -> list[TeamMember]:
        """Get all non-manager members."""
        return [m for m in self.members if not m.is_manager]

    def to_dict(self) -> dict[str, Any]:
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
    """Result from a team member's task execution.

    Superset combining:
    - Framework MemberResult: member_id, success, output, error, metadata
    - Agent/teams MemberResult: member_id, success, output, tool_calls_used,
                                duration_seconds, discoveries, error

    Attributes:
        member_id: ID of the team member
        success: Whether execution succeeded
        output: Output/result string
        error: Error message if failed
        metadata: Additional metadata
        tool_calls_used: Number of tool calls made
        duration_seconds: Execution duration
        discoveries: Key findings/discoveries made
    """

    member_id: str
    success: bool
    output: str
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_calls_used: int = 0
    duration_seconds: float = 0.0
    discoveries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "member_id": self.member_id,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
            "tool_calls_used": self.tool_calls_used,
            "duration_seconds": self.duration_seconds,
            "discoveries": self.discoveries,
        }


@dataclass
class TeamResult:
    """Result from team task execution.

    Attributes:
        success: Whether the team succeeded overall
        final_output: Synthesized final output
        member_results: Results from each member
        formation: Formation pattern used
        total_tool_calls: Total tool calls across all members
        total_duration: Total execution duration
        communication_log: Log of inter-agent messages
        shared_context: Final shared context state
        consensus_achieved: Whether consensus was reached (for CONSENSUS formation)
        consensus_rounds: Number of rounds needed for consensus
        error: Error message if failed
    """

    success: bool
    final_output: str
    member_results: dict[str, MemberResult]
    formation: TeamFormation
    total_tool_calls: int = 0
    total_duration: float = 0.0
    communication_log: list[AgentMessage] = field(default_factory=list)
    shared_context: dict[str, Any] = field(default_factory=dict)
    consensus_achieved: Optional[bool] = None
    consensus_rounds: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "final_output": self.final_output,
            "member_results": {k: v.to_dict() for k, v in self.member_results.items()},
            "formation": self.formation.value,
            "total_tool_calls": self.total_tool_calls,
            "total_duration": self.total_duration,
            "consensus_achieved": self.consensus_achieved,
            "consensus_rounds": self.consensus_rounds,
            "error": self.error,
        }


@dataclass
class TeamMemberAdapter:
    """Adapter that makes a TeamMember configuration implement the IAgent protocol.

    This adapter allows TeamMember configuration objects to be used wherever
    IAgent implementations are expected, by providing an executor function
    that handles actual task execution.

    Attributes:
        member: The TeamMember configuration
        executor: Callable that executes tasks for this member
        message_handler: Optional callable to handle incoming messages
    """

    member: "TeamMember"
    executor: Callable[[str, dict[str, Any]], Any]
    message_handler: Optional[Callable[[AgentMessage], Optional[AgentMessage]]] = None

    @property
    def id(self) -> str:
        """Unique identifier for this agent."""
        return self.member.id

    @property
    def role(self) -> str:
        """Role of this agent."""
        return self.member.role

    @property
    def persona(self) -> Optional[str]:
        """Persona of this member (None for TeamMemberAdapter)."""
        return None

    async def execute_task(self, task: str, context: dict[str, Any]) -> Any:
        """Execute a task using the provided executor."""
        return await self.executor(task, context)

    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive and optionally respond to a message."""
        if self.message_handler:
            result = self.message_handler(message)
            # Check if result is awaitable
            if isinstance(result, Awaitable):
                handler_result = await result  # type: ignore[unreachable]
            else:
                handler_result = result
            # The handler is typed to return Optional[AgentMessage], so this should be safe
            return (
                handler_result
                if isinstance(handler_result, AgentMessage) or handler_result is None
                else None
            )
        return None


# Type aliases for backward compatibility
TeamFormationType = TeamFormation
MessageTypeEnum = MessageType
