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

**DEPRECATED**: This module is deprecated. Use victor.teams.UnifiedTeamCoordinator
instead with lightweight_mode=True for testing.

This module provides a protocol-compliant team coordinator that implements
CrewAI-style team coordination patterns. It supports multiple formation
patterns and mediates all agent interactions.

Formation Patterns:
- SEQUENTIAL: Agents execute in order, each after the previous completes
- PARALLEL: Agents execute simultaneously
- HIERARCHICAL: Manager delegates to workers
- PIPELINE: Agents form a processing pipeline with output passing
- CONSENSUS: All agents must agree before proceeding

Example (deprecated):
    from victor.framework.team_coordinator import FrameworkTeamCoordinator

    coordinator = FrameworkTeamCoordinator()

Example (new approach):
    from victor.teams import create_coordinator

    coordinator = create_coordinator(lightweight=True)
"""

from __future__ import annotations

import asyncio
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from victor.framework.agent_protocols import AgentCapability
from victor.protocols.team import ITeamCoordinator, ITeamMember
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


class FrameworkTeamCoordinator(ITeamCoordinator):
    """**DEPRECATED**: Protocol-compliant team coordinator.

    This class is deprecated. Use victor.teams.UnifiedTeamCoordinator instead
    with lightweight_mode=True for testing, or use victor.teams.create_coordinator().

    This coordinator now delegates to UnifiedTeamCoordinator internally to
    maintain backward compatibility while using the unified implementation.

    **Migration Guide**:
        Old: coordinator = FrameworkTeamCoordinator()
        New: coordinator = create_coordinator(lightweight=True)

    Attributes:
        _unified: Internal UnifiedTeamCoordinator instance

    Example (deprecated):
        coordinator = FrameworkTeamCoordinator()
        coordinator.add_member(researcher).add_member(executor)

    Example (new approach):
        from victor.teams import create_coordinator
        coordinator = create_coordinator(lightweight=True)
        coordinator.add_member(researcher).add_member(executor)
    """

    def __init__(self) -> None:
        """Initialize the deprecated coordinator.

        Issues a deprecation warning and delegates to UnifiedTeamCoordinator.
        """
        warnings.warn(
            "FrameworkTeamCoordinator is deprecated. "
            "Use victor.teams.create_coordinator(lightweight=True) instead. "
            "This class will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Delegate to UnifiedTeamCoordinator in lightweight mode
        from victor.teams.unified_coordinator import UnifiedTeamCoordinator

        self._unified = UnifiedTeamCoordinator(
            orchestrator=None,
            enable_observability=False,
            enable_rl=False,
            lightweight_mode=True,
        )

    # =========================================================================
    # Properties (delegated to UnifiedTeamCoordinator)
    # =========================================================================

    @property
    def members(self) -> List[ITeamMember]:
        """Get list of team members."""
        return self._unified.members

    @property
    def formation(self) -> TeamFormation:
        """Get current team formation."""
        return self._unified.formation

    @property
    def manager(self) -> Optional[ITeamMember]:
        """Get team manager (for hierarchical formation)."""
        return self._unified.manager

    # =========================================================================
    # Configuration Methods (delegated to UnifiedTeamCoordinator)
    # =========================================================================

    def add_member(self, member: ITeamMember) -> "FrameworkTeamCoordinator":
        """Add a member to the team.

        Args:
            member: Team member to add

        Returns:
            Self for fluent chaining
        """
        self._unified.add_member(member)
        return self

    def set_formation(self, formation: TeamFormation) -> "FrameworkTeamCoordinator":
        """Set the team's formation pattern.

        Args:
            formation: Formation pattern to use

        Returns:
            Self for fluent chaining
        """
        self._unified.set_formation(formation)
        return self

    def set_manager(self, manager: ITeamMember) -> "FrameworkTeamCoordinator":
        """Set the team manager for hierarchical formation.

        Args:
            manager: Team member to act as manager

        Returns:
            Self for fluent chaining
        """
        self._unified.set_manager(manager)
        return self

    # =========================================================================
    # Task Execution (delegated to UnifiedTeamCoordinator)
    # =========================================================================

    async def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using the configured team.

        Args:
            task: Task description
            context: Context for task execution

        Returns:
            Dict containing execution results
        """
        return await self._unified.execute_task(task, context)

    # =========================================================================
    # Communication Methods (delegated to UnifiedTeamCoordinator)
    # =========================================================================

    async def broadcast(self, message: AgentMessage) -> List[Optional[AgentMessage]]:
        """Broadcast a message to all team members.

        Args:
            message: Message to broadcast

        Returns:
            List of responses from each member
        """
        return await self._unified.broadcast(message)

    async def send_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Send a message to a specific team member.

        Args:
            message: Message with recipient_id set

        Returns:
            Response from the recipient, or None
        """
        return await self._unified.send_message(message)


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
