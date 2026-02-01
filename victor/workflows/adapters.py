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

"""Adapters for workflow engine interoperability.

This module provides adapters that bridge the gap between different
workflow engines in Victor:

1. WorkflowBuilder (victor/workflows) - DAG-based, dict context
2. StateGraph (victor/framework/graph.py) - Cyclic, typed state

The adapters enable unified execution while maintaining backward
compatibility with existing workflows.

Design Goals:
- Preserve existing WorkflowBuilder API
- Enable StateGraph features (cycles, typed state) for new workflows
- Single execution path for both workflow types
- Gradual migration from WorkflowBuilder to StateGraph

Usage:
    from victor.workflows.adapters import WorkflowToGraphAdapter

    # Adapt a WorkflowBuilder workflow to StateGraph
    adapter = WorkflowToGraphAdapter()
    state_graph = adapter.adapt(workflow_definition)

    # Execute with StateGraph runtime
    app = state_graph.compile()
    result = await app.invoke(initial_state)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    TypedDict,
    cast,
)
from collections.abc import Awaitable, Callable

from victor.workflows.definition import (
    WorkflowDefinition,
    WorkflowNode,
    WorkflowNodeType,
)

if TYPE_CHECKING:
    from victor.framework.graph import StateGraph
    from victor.workflows.executor import WorkflowExecutor

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict, total=False):
    """Standard state for adapted workflows.

    This state structure is used when adapting WorkflowBuilder
    workflows to StateGraph execution.

    Attributes:
        context: Original workflow context (from WorkflowBuilder)
        messages: Conversation messages
        current_node: Current node being executed
        visited_nodes: Set of nodes already visited
        results: Results from each node
        error: Error message if any
        is_complete: Whether workflow has completed
    """

    context: dict[str, Any]
    messages: list[dict[str, Any]]
    current_node: str
    visited_nodes: list[str]
    results: dict[str, Any]
    error: Optional[str]
    is_complete: bool


@dataclass
class AdaptedNode:
    """A workflow node adapted for StateGraph execution.

    Attributes:
        name: Node name
        node_type: Original node type
        handler: Async function to execute
        next_nodes: Static next nodes
        conditional_edges: Conditional routing
        tool_budget: Tool budget for this node
        allowed_tools: Tools allowed in this node
    """

    name: str
    node_type: WorkflowNodeType
    handler: Callable[[dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]
    next_nodes: list[str] = field(default_factory=list)
    conditional_edges: dict[str, str] = field(default_factory=dict)
    tool_budget: int = 10
    allowed_tools: list[str] = field(default_factory=list)


class WorkflowToGraphAdapter:
    """Adapts WorkflowDefinition to StateGraph for unified execution.

    This adapter enables existing WorkflowBuilder workflows to run
    on the StateGraph runtime, providing access to features like:
    - Proper cycle detection and max iteration limits
    - Typed state management
    - Checkpointing and recovery
    - Streaming execution

    Example:
        adapter = WorkflowToGraphAdapter()
        state_graph = adapter.adapt(feature_workflow)

        # Execute with StateGraph runtime
        app = state_graph.compile()
        result = await app.invoke({
            "context": {"task": "implement feature"},
            "messages": [],
        })
    """

    def __init__(
        self,
        *,
        max_iterations: int = 50,
        enable_cycles: bool = False,
    ):
        """Initialize the adapter.

        Args:
            max_iterations: Maximum iterations for cycle detection
            enable_cycles: Whether to allow cycles in adapted workflows
        """
        self.max_iterations = max_iterations
        self.enable_cycles = enable_cycles

    def adapt(self, workflow: WorkflowDefinition) -> "StateGraph[Any]":
        """Adapt a WorkflowDefinition to StateGraph.

        Creates a StateGraph that mimics the behavior of the
        original WorkflowBuilder workflow.

        Args:
            workflow: WorkflowDefinition to adapt

        Returns:
            StateGraph ready for compilation and execution
        """
        # Import here to avoid circular imports
        from victor.framework.graph import StateGraph, END

        # Create StateGraph with workflow state
        # Use Dict[str, Any] instead of WorkflowState TypedDict for compatibility
        graph: StateGraph[dict[str, Any]] = StateGraph(dict[str, Any])

        # Convert each node - workflow.nodes is a Dict[str, WorkflowNode]
        nodes_list = list(workflow.nodes.values())
        for node in nodes_list:
            adapted = self._adapt_node(node, workflow)
            graph.add_node(adapted.name, adapted.handler)

        # Add edges based on node connections
        for node in nodes_list:
            if node.next_nodes:
                for next_node in node.next_nodes:
                    if next_node:
                        graph.add_edge(node.name, next_node)
            else:
                # Terminal node - connect to END
                graph.add_edge(node.name, END)

        # Set entry point
        if nodes_list:
            graph.set_entry_point(nodes_list[0].name)

        logger.debug(
            f"Adapted workflow '{workflow.name}' to StateGraph " f"({len(nodes_list)} nodes)"
        )

        return graph

    def _adapt_node(
        self,
        node: WorkflowNode,
        workflow: WorkflowDefinition,
    ) -> AdaptedNode:
        """Adapt a single WorkflowNode to an AdaptedNode.

        Args:
            node: WorkflowNode to adapt
            workflow: Parent workflow for context

        Returns:
            AdaptedNode ready for StateGraph
        """

        def create_handler(
            n: WorkflowNode,
        ) -> Callable[[dict[str, Any]], dict[str, Any]]:
            """Create a state-updating handler for the node."""

            def handler(state: dict[str, Any]) -> dict[str, Any]:
                # Update state with node execution
                new_state = dict(state)
                new_state["current_node"] = n.name

                # Track visited nodes
                visited = list(state.get("visited_nodes", []))
                visited.append(n.name)
                new_state["visited_nodes"] = visited

                # Add placeholder result
                # In production, this would execute the actual agent
                results = dict(state.get("results", {}))
                results[n.name] = {
                    "status": "pending",
                    "node_type": (
                        n.node_type.value if hasattr(n.node_type, "value") else str(n.node_type)
                    ),
                }
                new_state["results"] = results

                return new_state

            return handler

        # Extract tool_budget and allowed_tools if node is an AgentNode
        tool_budget = 10  # default
        allowed_tools = []  # default

        if hasattr(node, "tool_budget") and node.tool_budget is not None:
            tool_budget = node.tool_budget

        if hasattr(node, "allowed_tools") and node.allowed_tools is not None:
            allowed_tools = node.allowed_tools

        return AdaptedNode(
            name=node.name,
            node_type=node.node_type,
            handler=create_handler(node),
            next_nodes=node.next_nodes,
            tool_budget=tool_budget,
            allowed_tools=allowed_tools,
        )

    def adapt_with_execution(
        self,
        workflow: WorkflowDefinition,
        executor: "WorkflowExecutor",
    ) -> "StateGraph[dict[str, Any]]":
        """Adapt with real execution handlers.

        This version connects the adapted graph to the actual
        WorkflowExecutor for real agent execution.

        Args:
            workflow: WorkflowDefinition to adapt
            executor: WorkflowExecutor for running agents

        Returns:
            StateGraph with real execution handlers
        """
        # Import here to avoid circular imports
        from victor.framework.graph import StateGraph, END

        # Use Dict[str, Any] for compatibility with StateGraph type constraints
        graph: StateGraph[dict[str, Any]] = StateGraph(dict[str, Any])

        # Convert each node with real execution - workflow.nodes is a Dict[str, WorkflowNode]
        nodes_list = list(workflow.nodes.values())
        for node in nodes_list:
            handler = self._create_execution_handler(node, executor)
            graph.add_node(node.name, handler)

        # Add edges
        for node in nodes_list:
            if node.next_nodes:
                for next_node in node.next_nodes:
                    if next_node:
                        graph.add_edge(node.name, next_node)
            else:
                graph.add_edge(node.name, END)

        if nodes_list:
            graph.set_entry_point(nodes_list[0].name)

        return graph

    def _create_execution_handler(
        self,
        node: WorkflowNode,
        executor: "WorkflowExecutor",
    ) -> Callable[[dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]:
        """Create an execution handler that uses the workflow executor.

        Args:
            node: WorkflowNode to execute
            executor: WorkflowExecutor instance

        Returns:
            Handler function for StateGraph
        """
        # Import here to avoid circular imports

        async def async_handler(state: dict[str, Any]) -> dict[str, Any]:
            """Execute the node using the workflow executor."""
            new_state = dict(state)
            new_state["current_node"] = node.name

            # Get context from state
            context = state.get("context", {})

            try:
                # Execute the node using the executor
                # This is a simplified version - production would handle
                # all the agent orchestration details
                result = await executor.execute_by_name(
                    node.name,
                    initial_context=context,
                )

                # Update state with result
                results = dict(state.get("results", {}))
                results[node.name] = result
                new_state["results"] = results

            except Exception as e:
                new_state["error"] = str(e)
                logger.error(f"Node {node.name} failed: {e}")

            # Track visited
            visited = list(state.get("visited_nodes", []))
            visited.append(node.name)
            new_state["visited_nodes"] = visited

            return new_state

        # Return sync wrapper that runs async handler
        def sync_handler(state: WorkflowState) -> WorkflowState:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Convert WorkflowState to Dict[str, Any] for async_handler
            state_dict: dict[str, Any] = dict(state)
            result_dict = loop.run_until_complete(async_handler(state_dict))
            # Convert result back to WorkflowState
            return cast(WorkflowState, result_dict)

        return sync_handler  # type: ignore[return-value]


class GraphToWorkflowAdapter:
    """Adapts StateGraph to WorkflowDefinition for compatibility.

    This adapter enables new StateGraph workflows to work with
    existing WorkflowBuilder-based infrastructure.

    Use case: A new workflow written with StateGraph needs to be
    registered in the existing workflow registry.
    """

    def adapt(self, graph: "StateGraph[Any]", name: str) -> WorkflowDefinition:
        """Adapt a StateGraph to WorkflowDefinition.

        Args:
            graph: StateGraph to adapt
            name: Name for the workflow

        Returns:
            WorkflowDefinition compatible with existing infrastructure
        """
        # Import concrete node types
        from victor.workflows.definition import AgentNode

        # Get nodes from graph
        # Note: This is a simplified adaptation - StateGraph
        # internal structure may vary
        nodes = getattr(graph, "_nodes", {})
        edges = getattr(graph, "_edges", {})
        entry_point = getattr(graph, "_entry_point", None)

        # Convert nodes, starting with entry point if defined
        ordered_nodes = list(nodes.keys())
        if entry_point and entry_point in ordered_nodes:
            ordered_nodes.remove(entry_point)
            ordered_nodes.insert(0, entry_point)

        # Create workflow nodes
        workflow_nodes = {}
        node_to_next = {}  # Map node name to list of next node IDs

        # Build mapping of edges
        for from_node, to_nodes in edges.items():
            node_to_next[from_node] = to_nodes

        for node_name in ordered_nodes:
            # Get next nodes from edge mapping
            next_nodes = node_to_next.get(node_name, [])

            # Create an AgentNode as a placeholder
            workflow_nodes[node_name] = AgentNode(
                id=node_name,
                name=node_name,
                role="adapted_node",
                goal=f"Execute {node_name}",
                next_nodes=next_nodes,
            )

        return WorkflowDefinition(
            name=name,
            description="Adapted from StateGraph",
            nodes=cast(dict[str, WorkflowNode], workflow_nodes),
        )


__all__ = [
    "WorkflowState",
    "AdaptedNode",
    "WorkflowToGraphAdapter",
    "GraphToWorkflowAdapter",
]
