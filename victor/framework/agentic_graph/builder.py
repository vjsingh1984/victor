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

"""Agentic loop graph builder - creates StateGraph for agentic loop execution.

This module provides the create_agentic_loop_graph function that builds
a StateGraph implementing the PERCEIVE-PLAN-ACT-EVALUATE-DECIDE loop.

Graph Structure:
    START -> perceive -> plan -> act -> evaluate -> decide
    decide -> perceive (loop) or __end__ (terminate)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from victor.framework.graph import StateGraph, END
from victor.framework.agentic_graph.state import AgenticLoopStateModel
from victor.framework.agentic_graph.nodes import (
    perceive_node,
    plan_node,
    act_node,
    evaluate_node,
    decide_edge,
)

logger = logging.getLogger(__name__)


def create_agentic_loop_graph(
    max_iterations: int = 10,
    enable_fulfillment: bool = True,
    enable_adaptive_iterations: bool = True,
    *,
    include_prompt_node: bool = False,
    prompt_node: Optional[Callable[[Any], Any]] = None,
    runtime_intelligence: Optional[Any] = None,
    runtime_intelligence_resolver: Optional[Callable[[], Any]] = None,
    planning_coordinator: Optional[Any] = None,
    planning_coordinator_resolver: Optional[Callable[[], Any]] = None,
    use_llm_planning: bool = False,
    use_llm_planning_resolver: Optional[Callable[[], bool]] = None,
    turn_executor: Optional[Any] = None,
    turn_executor_resolver: Optional[Callable[[], Any]] = None,
    evaluator: Optional[Any] = None,
    evaluator_resolver: Optional[Callable[[], Any]] = None,
    fulfillment_detector: Optional[Any] = None,
    fulfillment_detector_resolver: Optional[Callable[[], Any]] = None,
) -> StateGraph:
    """Create a StateGraph that implements the agentic loop.

    The graph implements the PERCEIVE-PLAN-ACT-EVALUATE-DECIDE loop
    using StateGraph nodes and conditional edges.

    Args:
        max_iterations: Maximum number of loop iterations (default: 10)
        enable_fulfillment: Whether to enable fulfillment checks (default: True)
        enable_adaptive_iterations: Whether to enable adaptive termination (default: True)

    Returns:
        Compiled StateGraph ready for execution

    Example:
        graph = create_agentic_loop_graph(max_iterations=5)
        compiled = graph.compile()
        result = await compiled.invoke({"query": "Write code"})
    """
    graph = StateGraph(AgenticLoopStateModel)

    # Store configuration in graph metadata
    graph.metadata = {
        "max_iterations": max_iterations,
        "enable_fulfillment": enable_fulfillment,
        "enable_adaptive_iterations": enable_adaptive_iterations,
        "include_prompt_node": include_prompt_node,
    }

    resolved_prompt_node = prompt_node
    if include_prompt_node and resolved_prompt_node is None:
        from victor.framework.agentic_graph.service_nodes import prompt_service_node

        resolved_prompt_node = prompt_service_node

    # Add nodes
    # Note: We use lambda functions to inject configuration into nodes
    if include_prompt_node and resolved_prompt_node is not None:
        graph.add_node("prompt", resolved_prompt_node)

    graph.add_node(
        "perceive",
        lambda state: perceive_node(
            state,
            runtime_intelligence=(
                runtime_intelligence_resolver()
                if runtime_intelligence_resolver is not None
                else runtime_intelligence
            ),
        ),
    )

    graph.add_node(
        "plan",
        lambda state: plan_node(
            state,
            planning_coordinator=(
                planning_coordinator_resolver()
                if planning_coordinator_resolver is not None
                else planning_coordinator
            ),
            use_llm_planning=(
                use_llm_planning_resolver()
                if use_llm_planning_resolver is not None
                else use_llm_planning
            ),
        ),
    )

    graph.add_node(
        "act",
        lambda state: act_node(
            state,
            turn_executor=(
                turn_executor_resolver() if turn_executor_resolver is not None else turn_executor
            ),
        ),
    )

    graph.add_node(
        "evaluate",
        lambda state: evaluate_node(
            state,
            evaluator=(evaluator_resolver() if evaluator_resolver is not None else evaluator),
            fulfillment_detector=(
                fulfillment_detector_resolver()
                if fulfillment_detector_resolver is not None
                else fulfillment_detector
            ),
            enable_fulfillment_check=enable_fulfillment,
        ),
    )

    # Set entry point
    graph.set_entry_point("prompt" if include_prompt_node else "perceive")

    # Add sequential edges (PROMPT -> PERCEIVE -> PLAN -> ACT -> EVALUATE)
    if include_prompt_node and resolved_prompt_node is not None:
        graph.add_edge("prompt", "perceive")
    graph.add_edge("perceive", "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "evaluate")

    # Add conditional edge from EVALUATE to DECIDE
    # The decide_edge function returns the next node name
    graph.add_conditional_edge(
        "evaluate",
        decide_edge,
        {
            "perceive": "perceive",  # Continue loop
            "act": "act",  # Retry
            "__end__": END,  # Complete or fail
        },
    )

    return graph
