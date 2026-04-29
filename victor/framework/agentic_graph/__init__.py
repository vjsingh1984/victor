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
"""

from victor.framework.agentic_graph.state import (
    AgenticLoopState,
    AgenticLoopStateModel,
    create_initial_state,
    should_continue_loop,
)

__all__ = [
    "AgenticLoopState",
    "AgenticLoopStateModel",
    "create_initial_state",
    "should_continue_loop",
]
