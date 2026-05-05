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


def _resolve_dependency(
    value: Optional[Any],
    resolver: Optional[Callable[[], Any]],
) -> Optional[Any]:
    """Resolve a node dependency from a live resolver or a static fallback."""
    return resolver() if resolver is not None else value


def _bind_configured_node(
    node_fn: Callable[..., Any],
    /,
    **dependencies: tuple[Optional[Any], Optional[Callable[[], Any]]],
) -> Callable[[Any], Any]:
    """Create a named node wrapper that resolves dependencies at execution time."""

    def _configured_node(state: Any) -> Any:
        resolved_dependencies = {
            name: _resolve_dependency(value, resolver)
            for name, (value, resolver) in dependencies.items()
        }
        return node_fn(state, **resolved_dependencies)

    _configured_node.__name__ = getattr(node_fn, "__name__", "configured_node")
    return _configured_node


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
        _bind_configured_node(
            perceive_node,
            runtime_intelligence=(runtime_intelligence, runtime_intelligence_resolver),
        ),
    )

    graph.add_node(
        "plan",
        _bind_configured_node(
            plan_node,
            planning_coordinator=(planning_coordinator, planning_coordinator_resolver),
            use_llm_planning=(use_llm_planning, use_llm_planning_resolver),
        ),
    )

    graph.add_node(
        "act",
        _bind_configured_node(
            act_node,
            turn_executor=(turn_executor, turn_executor_resolver),
        ),
    )

    graph.add_node(
        "evaluate",
        _bind_configured_node(
            evaluate_node,
            evaluator=(evaluator, evaluator_resolver),
            fulfillment_detector=(fulfillment_detector, fulfillment_detector_resolver),
            enable_fulfillment_check=(enable_fulfillment, None),
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
