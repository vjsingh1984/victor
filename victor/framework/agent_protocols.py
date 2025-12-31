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

"""Agent protocols for multi-agent team coordination.

This module defines the core protocols and types for CrewAI-style multi-agent
orchestration in Victor. It provides:

- AgentCapability: Enum defining what agents can do (READ, WRITE, EXECUTE, etc.)
- MessageType: Enum for inter-agent message types
- AgentMessage: Dataclass for agent-to-agent communication
- IAgentRole: Protocol defining agent role contracts
- IAgentPersona: Protocol defining agent personality and style
- ITeamMember: Protocol for individual team members
- ITeamCoordinator: Protocol for team coordination
- TeamFormation: Enum for team organization patterns

Example:
    from victor.framework.agent_protocols import (
        AgentCapability,
        IAgentRole,
        TeamFormation,
    )

    # Check if a role has delegation capability
    if AgentCapability.DELEGATE in manager_role.capabilities:
        # Manager can delegate tasks to other agents
        pass

    # Configure team formation
    coordinator.set_formation(TeamFormation.HIERARCHICAL)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    runtime_checkable,
)


# =============================================================================
# Agent Capability Enum
# =============================================================================


class AgentCapability(str, Enum):
    """Capabilities that an agent can have.

    These capabilities determine what actions an agent is allowed to perform
    and inform the team coordinator about agent abilities.

    Attributes:
        READ: Agent can read files, search codebase, observe state
        WRITE: Agent can write files, make changes, create content
        EXECUTE: Agent can execute commands, run scripts, deploy
        SEARCH: Agent can search codebases, web, documentation
        COMMUNICATE: Agent can communicate with other agents
        DELEGATE: Agent can delegate tasks to other team members
        APPROVE: Agent can approve/reject changes made by others
    """

    READ = "read"
    """Agent can read files, search codebase, observe state."""

    WRITE = "write"
    """Agent can write files, make changes, create content."""

    EXECUTE = "execute"
    """Agent can execute commands, run scripts, deploy."""

    SEARCH = "search"
    """Agent can search codebases, web, documentation."""

    COMMUNICATE = "communicate"
    """Agent can communicate with other agents."""

    DELEGATE = "delegate"
    """Agent can delegate tasks to other team members."""

    APPROVE = "approve"
    """Agent can approve/reject changes made by others."""


# =============================================================================
# Message Types Enum
# =============================================================================


class MessageType(str, Enum):
    """Types of messages exchanged between agents.

    These message types enable structured inter-agent communication
    with clear semantics for each message category.

    Attributes:
        TASK: A task assignment from one agent to another
        RESULT: The result of a completed task
        QUERY: A question or information request
        FEEDBACK: Feedback on work performed
        DELEGATION: A delegated task from manager to worker
    """

    TASK = "task"
    """A task assignment from one agent to another."""

    RESULT = "result"
    """The result of a completed task."""

    QUERY = "query"
    """A question or information request."""

    FEEDBACK = "feedback"
    """Feedback on work performed."""

    DELEGATION = "delegation"
    """A delegated task from manager to worker."""


# =============================================================================
# Agent Message Dataclass
# =============================================================================


@dataclass
class AgentMessage:
    """A message exchanged between agents.

    AgentMessage provides structured communication between team members,
    enabling coordination, delegation, and feedback loops.

    Attributes:
        sender_id: ID of the agent sending the message
        recipient_id: ID of the intended recipient agent
        content: The message content
        message_type: Type of message (TASK, RESULT, QUERY, etc.)
        metadata: Additional context for the message

    Example:
        msg = AgentMessage(
            sender_id="manager_001",
            recipient_id="researcher_001",
            content="Analyze the authentication module",
            message_type=MessageType.TASK,
            metadata={"priority": "high", "deadline": "2h"},
        )
    """

    sender_id: str
    recipient_id: str
    content: str
    message_type: MessageType
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Team Formation Enum
# =============================================================================


class TeamFormation(str, Enum):
    """Team organization patterns for multi-agent coordination.

    These formations determine how agents coordinate to complete tasks,
    inspired by CrewAI's team patterns.

    Attributes:
        SEQUENTIAL: Agents work in order, each passing output to the next
        PARALLEL: Agents work simultaneously on different aspects
        HIERARCHICAL: Manager delegates and coordinates worker agents
        PIPELINE: Agents form a processing pipeline with defined stages
        CONSENSUS: All agents must agree before proceeding
    """

    SEQUENTIAL = "sequential"
    """Agents work in order, each passing output to the next."""

    PARALLEL = "parallel"
    """Agents work simultaneously on different aspects."""

    HIERARCHICAL = "hierarchical"
    """Manager delegates and coordinates worker agents."""

    PIPELINE = "pipeline"
    """Agents form a processing pipeline with defined stages."""

    CONSENSUS = "consensus"
    """All agents must agree before proceeding."""


# =============================================================================
# Agent Role Protocol
# =============================================================================


@runtime_checkable
class IAgentRole(Protocol):
    """Protocol defining an agent's role in a team.

    An agent role specifies what the agent is responsible for, what it can do,
    and what tools it has access to. Roles inform the system prompt and
    constrain the agent's behavior.

    Attributes:
        name: Unique identifier for this role
        capabilities: Set of AgentCapability values this role has
        allowed_tools: Set of tool names this role can use
        tool_budget: Maximum number of tool calls for this role

    Example:
        @dataclass
        class ResearcherRole:
            name: str = "researcher"
            capabilities: Set[AgentCapability] = field(
                default_factory=lambda: {AgentCapability.READ, AgentCapability.SEARCH}
            )
            allowed_tools: Set[str] = field(
                default_factory=lambda: {"read_file", "grep", "semantic_search"}
            )
            tool_budget: int = 20

            def get_system_prompt_section(self) -> str:
                return "You are a research specialist..."
    """

    @property
    def name(self) -> str:
        """Get the role name."""
        ...

    @property
    def capabilities(self) -> Set[AgentCapability]:
        """Get the capabilities this role has."""
        ...

    @property
    def allowed_tools(self) -> Set[str]:
        """Get the tools this role can use."""
        ...

    @property
    def tool_budget(self) -> int:
        """Get the tool budget for this role."""
        ...

    def get_system_prompt_section(self) -> str:
        """Get the system prompt section for this role.

        Returns:
            A string to be included in the agent's system prompt
            that describes the role and its responsibilities.
        """
        ...


# =============================================================================
# Agent Persona Protocol
# =============================================================================


@runtime_checkable
class IAgentPersona(Protocol):
    """Protocol defining an agent's personality and communication style.

    A persona adds character to an agent, making interactions more natural
    and consistent. It includes background, expertise, and communication style.

    Attributes:
        name: Display name for this persona
        background: Background story or expertise description
        communication_style: How this agent communicates

    Example:
        @dataclass
        class SecurityExpertPersona:
            name: str = "Security Analyst"
            background: str = "10 years of experience in cybersecurity"
            communication_style: str = "Professional, thorough, risk-focused"

            def format_message(self, content: str) -> str:
                return f"[Security Analysis] {content}"
    """

    @property
    def name(self) -> str:
        """Get the persona's display name."""
        ...

    @property
    def background(self) -> str:
        """Get the persona's background description."""
        ...

    @property
    def communication_style(self) -> str:
        """Get the persona's communication style."""
        ...

    def format_message(self, content: str) -> str:
        """Format a message according to this persona's style.

        Args:
            content: The raw message content

        Returns:
            Formatted message reflecting this persona's style
        """
        ...


# =============================================================================
# Team Member Protocol
# =============================================================================


@runtime_checkable
class ITeamMember(Protocol):
    """Protocol for individual team members.

    A team member combines a role, persona, and execution capability.
    Team members can execute tasks and communicate with other members.

    Attributes:
        id: Unique identifier for this team member
        role: The role this member plays (IAgentRole)
        persona: The personality of this member (IAgentPersona or None)

    Example:
        class ResearchAgent:
            def __init__(self):
                self.id = "researcher_001"
                self.role = ResearcherRole()
                self.persona = SecurityExpertPersona()

            async def execute_task(self, task, context):
                # Execute the research task
                return result

            async def receive_message(self, message):
                # Process incoming message
                return response_message
    """

    @property
    def id(self) -> str:
        """Get the member's unique identifier."""
        ...

    @property
    def role(self) -> IAgentRole:
        """Get the member's role."""
        ...

    @property
    def persona(self) -> Optional[IAgentPersona]:
        """Get the member's persona (may be None)."""
        ...

    async def execute_task(self, task: str, context: Dict[str, Any]) -> str:
        """Execute a task with the given context.

        Args:
            task: Description of the task to execute
            context: Additional context for task execution

        Returns:
            The result of task execution as a string
        """
        ...

    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive and process a message from another agent.

        Args:
            message: The incoming message

        Returns:
            Optional response message, or None if no response needed
        """
        ...


# =============================================================================
# Team Coordinator Protocol
# =============================================================================


@runtime_checkable
class ITeamCoordinator(Protocol):
    """Protocol for team coordination.

    The team coordinator manages team members, sets formations, and
    orchestrates task execution across the team.

    Example:
        coordinator = TeamCoordinator()
        coordinator.add_member(researcher)
        coordinator.add_member(executor)
        coordinator.set_formation(TeamFormation.SEQUENTIAL)

        result = await coordinator.execute_task(
            "Implement authentication feature",
            {"target_dir": "src/auth/"}
        )
    """

    def add_member(self, member: ITeamMember) -> "ITeamCoordinator":
        """Add a member to the team.

        Args:
            member: The team member to add

        Returns:
            Self for fluent chaining
        """
        ...

    def set_formation(self, formation: TeamFormation) -> "ITeamCoordinator":
        """Set the team's formation pattern.

        Args:
            formation: The formation to use

        Returns:
            Self for fluent chaining
        """
        ...

    async def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using the configured team.

        Args:
            task: The task description
            context: Context for task execution

        Returns:
            Dict containing execution results from team members
        """
        ...

    async def broadcast(self, message: AgentMessage) -> List[Optional[AgentMessage]]:
        """Broadcast a message to all team members.

        Args:
            message: The message to broadcast

        Returns:
            List of responses from team members
        """
        ...


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Enums
    "AgentCapability",
    "MessageType",
    "TeamFormation",
    # Dataclasses
    "AgentMessage",
    # Protocols
    "IAgentRole",
    "IAgentPersona",
    "ITeamMember",
    "ITeamCoordinator",
]
