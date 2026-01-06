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

"""Framework-level TeamCoordinator for multi-agent orchestration.

This module provides a protocol-compliant team coordinator that implements
CrewAI-style team coordination patterns. It supports multiple formation
patterns and mediates all agent interactions.

Formation Patterns:
- SEQUENTIAL: Agents execute in order, each after the previous completes
- PARALLEL: Agents execute simultaneously
- HIERARCHICAL: Manager delegates to workers
- PIPELINE: Agents form a processing pipeline with output passing
- CONSENSUS: All agents must agree before proceeding

Example:
    from victor.framework.team_coordinator import FrameworkTeamCoordinator
    from victor.framework.agent_protocols import TeamFormation

    coordinator = FrameworkTeamCoordinator()
    coordinator.add_member(researcher)
    coordinator.add_member(executor)
    coordinator.add_member(reviewer)
    coordinator.set_formation(TeamFormation.PIPELINE)

    result = await coordinator.execute_task(
        "Implement authentication feature",
        {"target_dir": "src/auth/"}
    )
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor.framework.agent_protocols import (
    AgentCapability,
    ITeamCoordinator,
    ITeamMember,
)
from victor.teams import (
    AgentMessage,
    MemberResult,
    MessageType,
    TeamFormation,
    TeamResult,
)


# =============================================================================
# MemberResult and TeamResult - imported from victor.teams
# =============================================================================
# NOTE: MemberResult and TeamResult are now imported from victor.teams.
# The canonical definitions live in victor.teams.types.


# =============================================================================
# Framework Team Coordinator
# =============================================================================


class FrameworkTeamCoordinator:
    """Protocol-compliant team coordinator for multi-agent orchestration.

    FrameworkTeamCoordinator implements the ITeamCoordinator protocol and
    provides CrewAI-style team coordination patterns. It mediates all
    agent interactions and supports multiple formation patterns.

    Attributes:
        members: List of team members
        formation: Current team formation pattern
        manager: Optional manager for hierarchical formation

    Example:
        coordinator = FrameworkTeamCoordinator()

        # Build team with fluent API
        coordinator
            .add_member(researcher)
            .add_member(executor)
            .add_member(reviewer)
            .set_formation(TeamFormation.PIPELINE)
            .set_manager(manager)

        # Execute task
        result = await coordinator.execute_task(
            "Implement authentication",
            {"target_dir": "src/"}
        )
    """

    def __init__(self) -> None:
        """Initialize the team coordinator."""
        self._members: List[ITeamMember] = []
        self._formation: TeamFormation = TeamFormation.SEQUENTIAL
        self._manager: Optional[ITeamMember] = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def members(self) -> List[ITeamMember]:
        """Get list of team members."""
        return self._members.copy()

    @property
    def formation(self) -> TeamFormation:
        """Get current team formation."""
        return self._formation

    @property
    def manager(self) -> Optional[ITeamMember]:
        """Get team manager (for hierarchical formation)."""
        return self._manager

    # =========================================================================
    # Configuration Methods (Fluent Interface)
    # =========================================================================

    def add_member(self, member: ITeamMember) -> "FrameworkTeamCoordinator":
        """Add a member to the team.

        Args:
            member: Team member to add

        Returns:
            Self for fluent chaining

        Example:
            coordinator.add_member(researcher).add_member(executor)
        """
        self._members.append(member)
        return self

    def set_formation(self, formation: TeamFormation) -> "FrameworkTeamCoordinator":
        """Set the team's formation pattern.

        Args:
            formation: Formation pattern to use

        Returns:
            Self for fluent chaining

        Example:
            coordinator.set_formation(TeamFormation.HIERARCHICAL)
        """
        self._formation = formation
        return self

    def set_manager(self, manager: ITeamMember) -> "FrameworkTeamCoordinator":
        """Set the team manager for hierarchical formation.

        Args:
            manager: Team member to act as manager

        Returns:
            Self for fluent chaining

        Example:
            coordinator.set_manager(manager_agent)
        """
        self._manager = manager
        return self

    # =========================================================================
    # Task Execution
    # =========================================================================

    async def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using the configured team.

        Dispatches to the appropriate execution method based on the
        current formation pattern.

        Args:
            task: Task description
            context: Context for task execution

        Returns:
            Dict containing execution results

        Example:
            result = await coordinator.execute_task(
                "Analyze codebase",
                {"target_dir": "src/"}
            )
        """
        if self._formation == TeamFormation.SEQUENTIAL:
            return await self._execute_sequential(task, context)
        elif self._formation == TeamFormation.PARALLEL:
            return await self._execute_parallel(task, context)
        elif self._formation == TeamFormation.HIERARCHICAL:
            return await self._execute_hierarchical(task, context)
        elif self._formation == TeamFormation.PIPELINE:
            return await self._execute_pipeline(task, context)
        elif self._formation == TeamFormation.CONSENSUS:
            return await self._execute_consensus(task, context)
        else:
            # Default to sequential
            return await self._execute_sequential(task, context)

    async def _execute_sequential(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with sequential formation.

        Members execute one after another in order.

        Args:
            task: Task description
            context: Execution context

        Returns:
            Dict with member_results and success status
        """
        member_results: Dict[str, MemberResult] = {}

        for member in self._members:
            try:
                output = await member.execute_task(task, context)
                member_results[member.id] = MemberResult(
                    member_id=member.id,
                    success=True,
                    output=output,
                )
            except Exception as e:
                member_results[member.id] = MemberResult(
                    member_id=member.id,
                    success=False,
                    output="",
                    error=str(e),
                )

        success = all(r.success for r in member_results.values())
        return {
            "success": success,
            "member_results": member_results,
            "formation": self._formation.value,
        }

    async def _execute_parallel(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with parallel formation.

        Members execute simultaneously.

        Args:
            task: Task description
            context: Execution context

        Returns:
            Dict with member_results and success status
        """
        member_results: Dict[str, MemberResult] = {}

        # Create tasks for all members
        async def execute_member(member: ITeamMember) -> tuple:
            try:
                output = await member.execute_task(task, context)
                return member.id, MemberResult(
                    member_id=member.id,
                    success=True,
                    output=output,
                )
            except Exception as e:
                return member.id, MemberResult(
                    member_id=member.id,
                    success=False,
                    output="",
                    error=str(e),
                )

        # Execute all members in parallel
        results = await asyncio.gather(*[execute_member(m) for m in self._members])

        for member_id, result in results:
            member_results[member_id] = result

        success = all(r.success for r in member_results.values())
        return {
            "success": success,
            "member_results": member_results,
            "formation": self._formation.value,
        }

    async def _execute_hierarchical(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with hierarchical formation.

        Manager executes first to coordinate, then workers execute.

        Args:
            task: Task description
            context: Execution context

        Returns:
            Dict with member_results and success status
        """
        member_results: Dict[str, MemberResult] = {}

        # Determine manager
        manager = self._manager
        if not manager and self._members:
            # Auto-select manager: first member with DELEGATE capability, or first member
            for member in self._members:
                if AgentCapability.DELEGATE in member.role.capabilities:
                    manager = member
                    break
            if not manager:
                manager = self._members[0]

        workers = [m for m in self._members if m != manager]

        # Execute manager first
        if manager:
            try:
                output = await manager.execute_task(task, context)
                member_results[manager.id] = MemberResult(
                    member_id=manager.id,
                    success=True,
                    output=output,
                )
            except Exception as e:
                member_results[manager.id] = MemberResult(
                    member_id=manager.id,
                    success=False,
                    output="",
                    error=str(e),
                )

        # Execute workers
        for worker in workers:
            try:
                output = await worker.execute_task(task, context)
                member_results[worker.id] = MemberResult(
                    member_id=worker.id,
                    success=True,
                    output=output,
                )
            except Exception as e:
                member_results[worker.id] = MemberResult(
                    member_id=worker.id,
                    success=False,
                    output="",
                    error=str(e),
                )

        success = all(r.success for r in member_results.values())
        return {
            "success": success,
            "member_results": member_results,
            "formation": self._formation.value,
        }

    async def _execute_pipeline(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with pipeline formation.

        Members execute sequentially, passing output to the next stage.

        Args:
            task: Task description
            context: Execution context

        Returns:
            Dict with member_results and success status
        """
        member_results: Dict[str, MemberResult] = {}
        current_context = context.copy()
        previous_output = ""

        for member in self._members:
            # Include previous output in context
            current_context["previous_output"] = previous_output

            try:
                output = await member.execute_task(task, current_context)
                member_results[member.id] = MemberResult(
                    member_id=member.id,
                    success=True,
                    output=output,
                )
                previous_output = output
            except Exception as e:
                member_results[member.id] = MemberResult(
                    member_id=member.id,
                    success=False,
                    output="",
                    error=str(e),
                )
                # Stop pipeline on error
                break

        success = all(r.success for r in member_results.values())
        return {
            "success": success,
            "member_results": member_results,
            "formation": self._formation.value,
            "final_output": previous_output,
        }

    async def _execute_consensus(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with consensus formation.

        All members execute in parallel and results are compared.

        Args:
            task: Task description
            context: Execution context

        Returns:
            Dict with member_results and consensus status
        """
        # First, execute in parallel
        result = await self._execute_parallel(task, context)

        # For now, consensus is achieved if all members succeed
        # A more sophisticated implementation would compare outputs
        result["consensus_achieved"] = result["success"]

        return result

    # =========================================================================
    # Communication Methods
    # =========================================================================

    async def broadcast(self, message: AgentMessage) -> List[Optional[AgentMessage]]:
        """Broadcast a message to all team members.

        Args:
            message: Message to broadcast

        Returns:
            List of responses from each member

        Example:
            responses = await coordinator.broadcast(
                AgentMessage(
                    sender_id="coordinator",
                    recipient_id="all",
                    content="Status update required",
                    message_type=MessageType.QUERY,
                )
            )
        """
        responses: List[Optional[AgentMessage]] = []

        for member in self._members:
            # Create member-specific message
            member_message = AgentMessage(
                sender_id=message.sender_id,
                recipient_id=member.id,
                content=message.content,
                message_type=message.message_type,
                data=message.data.copy(),
            )

            response = await member.receive_message(member_message)
            responses.append(response)

        return responses

    async def send_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Send a message to a specific team member.

        Args:
            message: Message with recipient_id set

        Returns:
            Response from the recipient, or None

        Example:
            response = await coordinator.send_message(
                AgentMessage(
                    sender_id="coordinator",
                    recipient_id="agent1",
                    content="Please review this code",
                    message_type=MessageType.TASK,
                )
            )
        """
        # Find recipient
        for member in self._members:
            if member.id == message.recipient_id:
                return await member.receive_message(message)

        return None


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Coordinator
    "FrameworkTeamCoordinator",
    # Result types
    "MemberResult",
    "TeamResult",
]
