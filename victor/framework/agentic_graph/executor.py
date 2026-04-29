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

"""Agentic loop graph executor - executes StateGraph-based agentic loop.

This module provides AgenticLoopGraphExecutor which runs the agentic
loop graph with proper service injection and result handling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING

from victor.framework.agentic_graph.state import (
    AgenticLoopStateModel,
    create_initial_state,
    should_continue_loop,
)
from victor.framework.agentic_graph.builder import create_agentic_loop_graph
from victor.framework.agentic_graph.service_nodes import inject_execution_context

if TYPE_CHECKING:
    from victor.framework.graph import CompiledGraph
    from victor.runtime.context import RuntimeExecutionContext

logger = logging.getLogger(__name__)


@dataclass
class LoopResult:
    """Result from agentic loop execution.

    Attributes:
        success: Whether execution completed successfully
        response: Final response content
        iterations: Number of iterations executed
        termination_reason: Why execution terminated
        metadata: Additional execution metadata
        error: Error message if execution failed
    """

    success: bool
    response: Optional[str]
    iterations: int
    termination_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @classmethod
    def from_graph_result(
        cls,
        graph_result: Any,
        final_state: AgenticLoopStateModel,
    ) -> "LoopResult":
        """Create LoopResult from graph execution result.

        Args:
            graph_result: Raw graph execution result
            final_state: Final state after execution

        Returns:
            LoopResult instance
        """
        # Extract response from final state
        response = None
        if final_state.action_result:
            response = final_state.action_result.get("response")

        # Determine termination reason
        termination_reason = "unknown"
        if final_state.evaluation:
            decision = final_state.evaluation.get("decision", "")
            if decision == "complete":
                termination_reason = "complete"
            elif decision == "fail":
                termination_reason = "failed"
            elif final_state.iteration >= final_state.max_iterations:
                termination_reason = "max_iterations"

        return cls(
            success=termination_reason != "failed",
            response=response,
            iterations=final_state.iteration,
            termination_reason=termination_reason,
            metadata={"final_state": final_state.to_dict()},
        )


class AgenticLoopGraphExecutor:
    """Executes the agentic loop using StateGraph.

    This executor manages:
    - Graph creation and compilation
    - Service injection into nodes
    - Execution lifecycle (run, stream)
    - Result aggregation and formatting

    Example:
        executor = AgenticLoopGraphExecutor(
            execution_context=context,
            max_iterations=10,
        )
        result = await executor.run("Write tests")
    """

    def __init__(
        self,
        execution_context: Any,
        max_iterations: int = 10,
        enable_fulfillment: bool = True,
        enable_adaptive_iterations: bool = True,
    ):
        """Initialize the executor.

        Args:
            execution_context: RuntimeExecutionContext for service injection
            max_iterations: Maximum number of loop iterations
            enable_fulfillment: Whether to enable fulfillment checks
            enable_adaptive_iterations: Whether to enable adaptive termination
        """
        self.execution_context = execution_context
        self.max_iterations = max_iterations
        self.enable_fulfillment = enable_fulfillment
        self.enable_adaptive_iterations = enable_adaptive_iterations

        # Create and compile graph
        self.graph = create_agentic_loop_graph(
            max_iterations=max_iterations,
            enable_fulfillment=enable_fulfillment,
            enable_adaptive_iterations=enable_adaptive_iterations,
        )
        self.compiled: CompiledGraph = self.graph.compile()

        # Services (injected lazily or set externally)
        self.runtime_intelligence: Optional[Any] = None
        self.planning_coordinator: Optional[Any] = None
        self.turn_executor: Optional[Any] = None
        self.evaluator: Optional[Any] = None
        self.fulfillment_detector: Optional[Any] = None

    async def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> LoopResult:
        """Run the agentic loop to completion.

        Args:
            query: User's query or task
            context: Optional additional context

        Returns:
            LoopResult with execution outcome
        """
        # Create initial state
        initial_state = create_initial_state(
            query=query,
            context=context or {},
            max_iterations=self.max_iterations,
        )

        # Inject ExecutionContext for service access
        initial_state = inject_execution_context(initial_state, self.execution_context)

        try:
            # Execute graph
            graph_result = await self.compiled.invoke(initial_state)

            # Extract final state
            final_state = self._extract_final_state(graph_result)

            # Create result
            result = LoopResult.from_graph_result(graph_result, final_state)

            logger.info(
                f"Agentic loop completed: {result.iterations} iterations, "
                f"termination={result.termination_reason}"
            )

            return result

        except Exception as e:
            logger.error(f"Agentic loop execution failed: {e}")
            return LoopResult(
                success=False,
                response=None,
                iterations=0,
                termination_reason="error",
                error=str(e),
            )

    async def stream(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream agentic loop execution events.

        Args:
            query: User's query or task
            context: Optional additional context

        Yields:
            Event dictionaries with node_name, state, and event_type
        """
        # Create initial state
        initial_state = create_initial_state(
            query=query,
            context=context or {},
            max_iterations=self.max_iterations,
        )

        # Inject ExecutionContext for service access
        initial_state = inject_execution_context(initial_state, self.execution_context)

        try:
            # Stream graph execution
            async for node_name, state in self.compiled.stream(initial_state):
                yield {
                    "node_name": node_name,
                    "state": state.to_dict() if hasattr(state, "to_dict") else state,
                    "event_type": "node_complete",
                }

        except Exception as e:
            yield {
                "node_name": "error",
                "state": {},
                "event_type": "error",
                "error": str(e),
            }

    def _extract_final_state(self, graph_result: Any) -> AgenticLoopStateModel:
        """Extract final state from graph result.

        Args:
            graph_result: Raw graph execution result

        Returns:
            Final AgenticLoopStateModel
        """
        # Handle different graph result formats
        if hasattr(graph_result, "state"):
            return graph_result.state
        elif isinstance(graph_result, dict):
            return AgenticLoopStateModel(**graph_result)
        else:
            # Assume it's already a state model
            return graph_result

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics.

        Returns:
            Dictionary with execution metrics
        """
        # Get node list from graph (_nodes is the internal attribute)
        graph_nodes = list(self.graph._nodes.keys()) if hasattr(self.graph, "_nodes") else []

        return {
            "max_iterations": self.max_iterations,
            "enable_fulfillment": self.enable_fulfillment,
            "enable_adaptive_iterations": self.enable_adaptive_iterations,
            "graph_nodes": graph_nodes,
        }
