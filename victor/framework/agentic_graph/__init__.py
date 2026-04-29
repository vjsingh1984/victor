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

"""Agentic Graph - StateGraph-based agentic loop implementation.

This module provides a StateGraph-based implementation of the agentic loop,
replacing the custom while loop with a declarative graph-based approach.

Architecture:
    StateGraph (PERCEIVE -> PLAN -> ACT -> EVALUATE -> DECIDE)
        ├── perceive_node: Intent understanding via PerceptionIntegration
        ├── plan_node: Execution planning via PlanningCoordinator
        ├── act_node: Tool execution via TurnExecutor
        ├── evaluate_node: Progress evaluation via FulfillmentDetector
        └── decide: Conditional edge routing back to PERCEIVE or END

Components:
    - state: AgenticLoopStateModel and helpers
    - nodes: Core agentic loop nodes (perceive, plan, act, evaluate)
    - builder: Graph construction utilities
    - executor: AgenticLoopGraphExecutor for running the graph
    - coordinator_nodes: State-passed coordinator adapters
    - service_nodes: Service provider nodes (chat, tool, context, provider)
    - team_selector: Formation selection logic

Example:
    from victor.framework.agentic_graph import (
        create_agentic_loop_graph,
        AgenticLoopGraphExecutor,
    )

    # Create graph
    graph = create_agentic_loop_graph(max_iterations=10)
    executor = AgenticLoopGraphExecutor(execution_context, graph)

    # Execute
    result = await executor.run("Fix the authentication bug")

Team Architecture Note:
    Teams are formations, not separate graphs. Use UnifiedTeamCoordinator
    directly as a StateGraph node:

        from victor.teams import UnifiedTeamCoordinator, TeamFormation
        coordinator = UnifiedTeamCoordinator(orchestrator)
        coordinator.set_formation(TeamFormation.PARALLEL)
        graph.add_node("team", coordinator)  # Direct usage!
"""

# Core state and graph components
from victor.framework.agentic_graph.state import (
    AgenticLoopState,
    AgenticLoopStateModel,
    create_initial_state,
    should_continue_loop,
)
from victor.framework.agentic_graph.nodes import (
    perceive_node,
    plan_node,
    act_node,
    evaluate_node,
    decide_edge,
)
from victor.framework.agentic_graph.builder import create_agentic_loop_graph
from victor.framework.agentic_graph.executor import (
    AgenticLoopGraphExecutor,
    LoopResult,
)

# Coordinator adapters
from victor.framework.agentic_graph.coordinator_nodes import (
    CoordinatorAdapter,
    exploration_node,
    safety_node,
    system_prompt_node,
)

# Service provider nodes
from victor.framework.agentic_graph.service_nodes import (
    chat_service_node,
    tool_service_node,
    context_service_node,
    provider_service_node,
    inject_execution_context,
)

# Team selector (formation selection logic)
from victor.framework.agentic_graph.team_selector import (
    select_formation,
    FormationCriteria,
    DEFAULT_FORMATION,
)

__all__ = [
    # State
    "AgenticLoopState",
    "AgenticLoopStateModel",
    "create_initial_state",
    "should_continue_loop",
    # Core nodes
    "perceive_node",
    "plan_node",
    "act_node",
    "evaluate_node",
    "decide_edge",
    # Builder and executor
    "create_agentic_loop_graph",
    "AgenticLoopGraphExecutor",
    "LoopResult",
    # Coordinator adapters
    "CoordinatorAdapter",
    "exploration_node",
    "safety_node",
    "system_prompt_node",
    # Service nodes
    "chat_service_node",
    "tool_service_node",
    "context_service_node",
    "provider_service_node",
    "inject_execution_context",
    # Team selector
    "select_formation",
    "FormationCriteria",
    "DEFAULT_FORMATION",
]
