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
from typing import Any, Optional

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
    }

    # Add nodes
    # Note: We use lambda functions to inject configuration into nodes
    graph.add_node(
        "perceive",
        lambda state: perceive_node(
            state,
            runtime_intelligence=None,  # Will be injected by executor
        ),
    )

    graph.add_node(
        "plan",
        lambda state: plan_node(
            state,
            planning_coordinator=None,  # Will be injected by executor
            use_llm_planning=False,  # Default to fast path
        ),
    )

    graph.add_node(
        "act",
        lambda state: act_node(
            state,
            turn_executor=None,  # Will be injected by executor
        ),
    )

    graph.add_node(
        "evaluate",
        lambda state: evaluate_node(
            state,
            evaluator=None,  # Will be injected by executor
            fulfillment_detector=None,  # Will be injected by executor
            enable_fulfillment_check=enable_fulfillment,
        ),
    )

    # Set entry point
    graph.set_entry_point("perceive")

    # Add sequential edges (PERCEIVE -> PLAN -> ACT -> EVALUATE)
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
