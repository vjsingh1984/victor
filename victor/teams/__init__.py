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

**Basic Usage**:
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

**Formation Patterns**:
    # Sequential: Chain agents one after another
    coordinator.set_formation(TeamFormation.SEQUENTIAL)

    # Parallel: Execute all agents simultaneously
    coordinator.set_formation(TeamFormation.PARALLEL)

    # Hierarchical: Manager delegates to workers
    coordinator.set_formation(TeamFormation.HIERARCHICAL)

    # Pipeline: Processing pipeline with output passing
    coordinator.set_formation(TeamFormation.PIPELINE)

    # Consensus: Multi-round agreement building
    coordinator.set_formation(TeamFormation.CONSENSUS)

**Testing**:
    # Lightweight coordinator for testing (no dependencies)
    coordinator = create_coordinator(lightweight=True)

**Production**:
    # Full-featured coordinator with observability and RL
    coordinator = create_coordinator(
        orchestrator=my_orchestrator,
        enable_observability=True,
        enable_rl=True,
    )

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

Documentation:
    - MIGRATION_GUIDE.md: Step-by-step migration instructions
    - RELEASE_NOTES.md: Release notes and changes
    - CONSOLIDATION.md: Architecture consolidation details
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

# Protocols - use lazy loading to avoid circular import with victor.protocols.team
# The canonical location is victor.protocols.team
def __getattr__(name: str):
    """Lazy import protocols to avoid circular dependency."""
    import importlib

    protocol_map = {
        "IEnhancedTeamCoordinator": ("victor.protocols.team", "IEnhancedTeamCoordinator"),
        "IMessageBusProvider": ("victor.protocols.team", "IMessageBusProvider"),
        "IObservableCoordinator": ("victor.protocols.team", "IObservableCoordinator"),
        "IRLCoordinator": ("victor.protocols.team", "IRLCoordinator"),
        "ISharedMemoryProvider": ("victor.protocols.team", "ISharedMemoryProvider"),
        "ITeamCoordinator": ("victor.protocols.team", "ITeamCoordinator"),
        "ITeamMember": ("victor.protocols.team", "IAgent"),  # IAgent is the canonical name
    }

    if name in protocol_map:
        module_path, attr_name = protocol_map[name]
        module = importlib.import_module(module_path)
        globals()[name] = getattr(module, attr_name)
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Coordinator
from victor.teams.unified_coordinator import UnifiedTeamCoordinator

# Framework coordinator (for testing/lightweight usage)
# Note: Imported in create_coordinator() to avoid circular dependency

# Communication infrastructure (from agent.teams)
from victor.agent.teams.communication import TeamMessageBus, TeamSharedMemory

# Team configuration types (canonical)
from victor.teams.types import TeamConfig, TeamMember

# Mixins
from victor.teams.mixins import ObservabilityMixin, RLMixin

# Advanced formations
from victor.teams.advanced_formations import (
    NegotiationFormation,
    SwitchingCriteria,
    SwitchingFormation,
    VotingFormation,
)

# ML and optimization
from victor.teams.team_analytics import TeamAnalytics
from victor.teams.team_learning import TeamLearningSystem
from victor.teams.team_optimizer import TeamOptimizer
from victor.teams.team_predictor import TeamPredictor

# ML models
from victor.teams.ml import FormationPredictor, PerformancePredictor, TeamMemberSelector

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
        lightweight: Use lightweight mode (disables mixins, for testing)
        with_observability: Enable EventBus integration (default: True)
        with_rl: Enable RL integration (default: True)

    Returns:
        ITeamCoordinator implementation (UnifiedTeamCoordinator)

    Examples:
        # Lightweight for testing (replaces FrameworkTeamCoordinator)
        coordinator = create_coordinator(lightweight=True)

        # Production coordinator with all features
        coordinator = create_coordinator(orchestrator)

        # Without RL
        coordinator = create_coordinator(orchestrator, with_rl=False)

        # Without observability (minimal dependencies)
        coordinator = create_coordinator(
            orchestrator,
            with_observability=False,
            with_rl=False,
        )

    See Also:
        MIGRATION_GUIDE.md: Complete migration instructions
        UnifiedTeamCoordinator: Direct coordinator class
    """
    # Always use UnifiedTeamCoordinator with appropriate mode
    # This effectively merges FrameworkTeamCoordinator into UnifiedTeamCoordinator
    return UnifiedTeamCoordinator(
        orchestrator,
        enable_observability=with_observability if not lightweight else False,
        enable_rl=with_rl if not lightweight else False,
        lightweight_mode=lightweight,
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
    # Advanced formations
    "SwitchingFormation",
    "SwitchingCriteria",
    "NegotiationFormation",
    "VotingFormation",
    # ML and optimization
    "TeamPredictor",
    "TeamOptimizer",
    "TeamAnalytics",
    "TeamLearningSystem",
    # ML models
    "TeamMemberSelector",
    "FormationPredictor",
    "PerformancePredictor",
]
