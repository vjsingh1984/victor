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

"""Team coordination protocols (canonical location).

This module defines the protocols for team coordination in a neutral location
that both victor.teams and victor.framework can import from, breaking
circular dependencies.

Protocols:
    IAgent: Base agent protocol
    ITeamMember: Team member protocol
    ITeamCoordinator: Base coordinator protocol
    IObservableCoordinator: Observability capabilities
    IRLCoordinator: RL integration capabilities
    IMessageBusProvider: Message bus provider
    ISharedMemoryProvider: Shared memory provider
    IEnhancedTeamCoordinator: Combined capabilities

Design Principles:
    - ISP: Small, focused protocols that can be implemented independently
    - DIP: Depend on protocols, not concrete implementations
    - OCP: New capabilities added via new protocols, not modification
    - No circular dependencies: Both teams and framework import from here
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Protocol,
    runtime_checkable,
)
from collections.abc import Callable

# Import types from victor.teams.types (canonical location for types)
# Use TYPE_CHECKING to avoid circular import during module initialization
if TYPE_CHECKING:
    from victor.teams.types import AgentMessage, TeamFormation
else:
    # Import at runtime but use lazy loading to avoid circular dependency
    def __getattr__(name: str):
        """Lazy import types from victor.teams.types to avoid circular dependency."""
        import importlib

        if name in ("AgentMessage", "TeamFormation", "MemberResult", "TeamResult"):
            module = importlib.import_module("victor.teams.types")
            globals()[name] = getattr(module, name)
            return globals()[name]

        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@runtime_checkable
class IAgent(Protocol):
    """Unified protocol for all agent implementations.

    This is the minimal interface that all agents in the Victor system
    must implement. It provides a consistent way to execute tasks
    regardless of the underlying implementation (SubAgent, team member, etc.).

    Attributes:
        id: Unique identifier for this agent
        role: Role of this agent (IAgentRole, SubAgentRole, or any type)
    """

    @property
    def id(self) -> str:
        """Unique identifier for this agent."""
        ...

    @property
    def role(self) -> Any:
        """Role of this agent (IAgentRole, SubAgentRole, or any type)."""
        ...

    async def execute_task(self, task: str, context: dict[str, Any]) -> Any:
        """Execute a task and return the result.

        Args:
            task: Task description
            context: Execution context with shared state

        Returns:
            Result of task execution (string, dict, or any type)
        """
        ...


@runtime_checkable
class ITeamMember(IAgent, Protocol):
    """Protocol for team members.

    Defines the minimal interface for an agent that can participate
    in team coordination.

    Note: execute_task returns Any (not str) to maintain LSP compliance
    with IAgent. Implementations may return str, but the protocol allows
    any return type for substitutability.
    """

    @property
    def persona(self) -> Optional[Any]:
        """Persona of this member (IAgentPersona or None)."""
        ...

    async def execute_task(self, task: str, context: dict[str, Any]) -> Any:
        """Execute a task and return the result.

        Args:
            task: Task description
            context: Execution context with shared state

        Returns:
            Result of task execution (typically str, but Any for LSP compliance)
        """
        ...

    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive and optionally respond to a message.

        Args:
            message: Incoming message

        Returns:
            Optional response message
        """
        ...


@runtime_checkable
class ITeamCoordinator(Protocol):
    """Base protocol for team coordinators.

    This is the canonical protocol that all team coordinators must implement.
    Ensures LSP compliance across implementations.
    """

    def add_member(self, member: ITeamMember) -> "ITeamCoordinator":
        """Add a member to the team.

        Args:
            member: Team member to add

        Returns:
            Self for fluent chaining
        """
        ...

    def set_formation(self, formation: TeamFormation) -> "ITeamCoordinator":
        """Set the team formation pattern.

        Args:
            formation: Formation to use

        Returns:
            Self for fluent chaining
        """
        ...

    async def execute_task(self, task: str, context: dict[str, Any]) -> dict[str, Any]:
        """Execute a task with the team.

        Args:
            task: Task description
            context: Execution context

        Returns:
            Result dictionary with success, member_results, final_output
        """
        ...

    async def broadcast(self, message: AgentMessage) -> list[Optional[AgentMessage]]:
        """Broadcast a message to all team members.

        Args:
            message: Message to broadcast

        Returns:
            List of responses from members
        """
        ...


@runtime_checkable
class IObservableCoordinator(Protocol):
    """Protocol for coordinators with observability capabilities.

    Adds EventBus integration, progress tracking, and execution context.
    This is an optional capability that can be composed with ITeamCoordinator.
    """

    def set_execution_context(
        self,
        task_type: str,
        complexity: str,
        vertical: str,
        trigger: str,
    ) -> None:
        """Set execution context for observability.

        Args:
            task_type: Type of task (e.g., "feature", "bugfix")
            complexity: Task complexity level
            vertical: Domain vertical (e.g., "coding", "devops")
            trigger: How the task was triggered (e.g., "auto", "manual")
        """
        ...

    def set_progress_callback(
        self,
        callback: Callable[[str, str, float], None],
    ) -> None:
        """Set callback for progress updates.

        Args:
            callback: Function(member_id, status, progress_percent)
        """
        ...


@runtime_checkable
class IRLCoordinator(Protocol):
    """Protocol for coordinators with RL (Reinforcement Learning) integration.

    Adds support for recording outcomes and learning optimal team compositions.
    This is an optional capability that can be composed with ITeamCoordinator.
    """

    def set_rl_coordinator(self, rl_coordinator: Any) -> None:
        """Set the RL coordinator for outcome recording.

        Args:
            rl_coordinator: RL coordinator instance
        """
        ...


@runtime_checkable
class IMessageBusProvider(Protocol):
    """Protocol for coordinators that provide a message bus.

    Enables inter-agent communication beyond simple broadcast.
    """

    @property
    def message_bus(self) -> Any:
        """Get the team message bus.

        Returns:
            TeamMessageBus instance or None if not available
        """
        ...


@runtime_checkable
class ISharedMemoryProvider(Protocol):
    """Protocol for coordinators that provide shared memory.

    Enables shared context and state across team members.
    """

    @property
    def shared_memory(self) -> Any:
        """Get the team shared memory.

        Returns:
            TeamSharedMemory instance or None if not available
        """
        ...


@runtime_checkable
class IEnhancedTeamCoordinator(
    ITeamCoordinator,
    IObservableCoordinator,
    IRLCoordinator,
    IMessageBusProvider,
    ISharedMemoryProvider,
    Protocol,
):
    """Enhanced protocol combining all team coordinator capabilities.

    This is the full-featured protocol that production coordinators
    should implement for maximum capability.

    Combines:
        - ITeamCoordinator: Base team coordination
        - IObservableCoordinator: EventBus and progress tracking
        - IRLCoordinator: RL integration for learning
        - IMessageBusProvider: Inter-agent messaging
        - ISharedMemoryProvider: Shared team context
    """

    pass
