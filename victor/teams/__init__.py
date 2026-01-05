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

"""Victor Teams - Unified multi-agent team coordination.

This is the recommended import location for all team functionality.
Import from here instead of victor.framework or victor.agent.teams.

Example:
    from victor.teams import (
        TeamFormation,
        MessageType,
        AgentMessage,
        UnifiedTeamCoordinator,
        create_coordinator,
    )

    # Create a coordinator
    coordinator = create_coordinator(orchestrator)
    coordinator.add_member(researcher).add_member(executor)
    coordinator.set_formation(TeamFormation.PIPELINE)
    result = await coordinator.execute_task("Implement feature", {})

Factory Functions:
    create_coordinator: Create appropriate coordinator based on requirements

Types (canonical):
    TeamFormation: Team organization patterns
    MessageType: Message types for inter-agent communication
    AgentMessage: Agent message structure
    MemberResult: Result from member execution
    TeamResult: Result from team execution

Protocols:
    ITeamCoordinator: Base coordinator protocol
    ITeamMember: Team member protocol
    IObservableCoordinator: Observability protocol
    IRLCoordinator: RL integration protocol
    IEnhancedTeamCoordinator: Full-featured protocol

Coordinators:
    UnifiedTeamCoordinator: Production coordinator with all features

Mixins:
    ObservabilityMixin: Add observability to custom coordinators
    RLMixin: Add RL integration to custom coordinators
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

# Canonical types - single source of truth
from victor.teams.types import (
    AgentMessage,
    MemberResult,
    MessagePriority,
    MessageType,
    TeamFormation,
    TeamResult,
)

# Protocols
from victor.teams.protocols import (
    IEnhancedTeamCoordinator,
    IMessageBusProvider,
    IObservableCoordinator,
    IRLCoordinator,
    ISharedMemoryProvider,
    ITeamCoordinator,
    ITeamMember,
)

# Coordinator
from victor.teams.unified_coordinator import UnifiedTeamCoordinator

# Framework coordinator (for testing/lightweight usage)
from victor.framework.team_coordinator import FrameworkTeamCoordinator

# Communication infrastructure (from agent.teams)
from victor.agent.teams.communication import TeamMessageBus, TeamSharedMemory

# Team configuration types (from agent.teams)
from victor.agent.teams.team import TeamConfig, TeamMember

# Mixins
from victor.teams.mixins import ObservabilityMixin, RLMixin

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator


def create_coordinator(
    orchestrator: Optional["AgentOrchestrator"] = None,
    *,
    lightweight: bool = False,
    with_observability: bool = True,
    with_rl: bool = True,
) -> ITeamCoordinator:
    """Factory function for creating team coordinators.

    This is the recommended way to create coordinators as it handles
    the appropriate configuration based on requirements.

    Args:
        orchestrator: Agent orchestrator (optional, for SubAgent spawning)
        lightweight: Use lightweight framework coordinator (for testing)
        with_observability: Enable EventBus integration (default: True)
        with_rl: Enable RL integration (default: True)

    Returns:
        ITeamCoordinator implementation

    Example:
        # Production coordinator with all features
        coordinator = create_coordinator(orchestrator)

        # Lightweight for testing
        coordinator = create_coordinator(lightweight=True)

        # Without RL
        coordinator = create_coordinator(orchestrator, with_rl=False)
    """
    if lightweight:
        # Use lightweight framework coordinator for testing
        from victor.framework.team_coordinator import FrameworkTeamCoordinator

        return FrameworkTeamCoordinator()

    # Use unified production coordinator
    return UnifiedTeamCoordinator(
        orchestrator,
        enable_observability=with_observability,
        enable_rl=with_rl,
    )


__all__ = [
    # Types
    "TeamFormation",
    "MessageType",
    "MessagePriority",
    "AgentMessage",
    "MemberResult",
    "TeamResult",
    # Protocols
    "ITeamCoordinator",
    "ITeamMember",
    "IObservableCoordinator",
    "IRLCoordinator",
    "IMessageBusProvider",
    "ISharedMemoryProvider",
    "IEnhancedTeamCoordinator",
    # Coordinators
    "UnifiedTeamCoordinator",
    "FrameworkTeamCoordinator",
    # Communication
    "TeamMessageBus",
    "TeamSharedMemory",
    # Team configuration
    "TeamConfig",
    "TeamMember",
    # Mixins
    "ObservabilityMixin",
    "RLMixin",
    # Factory
    "create_coordinator",
]
