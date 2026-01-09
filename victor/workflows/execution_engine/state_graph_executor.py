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

"""Pure workflow executor - NO compilation logic.

Executes compiled workflow graphs.
This is a stub that delegates to legacy implementation during migration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional

if TYPE_CHECKING:
    from victor.workflows.compiler_protocols import CompiledGraphProtocol, ExecutionEventProtocol, ExecutionResultProtocol

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """Pure executor for compiled workflows.

    Responsibility (SRP):
    - Execute compiled graphs
    - Manage execution context
    - Stream execution events
    - Handle checkpoints

    Non-responsibility:
    - Compilation (handled by WorkflowCompiler)
    - Caching (handled by decorator/wrapper)
    - Node execution logic (handled by node executors)
    - StateGraph execution (handled by LangGraph)

    Design:
    - SRP compliance: ONLY executes, doesn't compile
    - DIP compliance: Depends on protocols, not concrete classes
    - LSP compliance: Interchangeable with other executor implementations

    Attributes:
        _orchestrator_pool: Pool of orchestrators for multi-provider workflows

    Example:
        executor = WorkflowExecutor(orchestrator_pool=pool)
        result = await executor.execute(compiled_graph, {"input": "data"})

        # Or stream events
        async for event in executor.stream(compiled_graph, {"input": "data"}):
            print(f"{event.node_id}: {event.event_type}")
    """

    def __init__(self, orchestrator_pool: Any):
        """Initialize the executor.

        Args:
            orchestrator_pool: OrchestratorPool instance
        """
        self._orchestrator_pool = orchestrator_pool

    async def execute(
        self,
        compiled_graph: "CompiledGraphProtocol",
        initial_state: Dict[str, Any],
        *,
        thread_id: Optional[str] = None,
        checkpoint: Optional[str] = None,
    ) -> "ExecutionResultProtocol":
        """Execute a compiled workflow graph.

        Args:
            compiled_graph: Compiled workflow graph
            initial_state: Initial workflow state
            thread_id: Optional thread ID for checkpointing
            checkpoint: Optional checkpoint name to resume from

        Returns:
            ExecutionResultProtocol with execution outcome

        Example:
            executor = WorkflowExecutor(orchestrator_pool=pool)
            result = await executor.execute(compiled, {"query": "search"})
            print(result.final_state)
        """
        logger.info(f"Executing workflow graph...")

        # TODO: Implement proper execution
        # For now, delegate to legacy implementation
        return await self._execute_legacy(
            compiled_graph,
            initial_state,
            thread_id=thread_id,
            checkpoint=checkpoint,
        )

    async def _execute_legacy(
        self,
        compiled_graph: "CompiledGraphProtocol",
        initial_state: Dict[str, Any],
        *,
        thread_id: Optional[str] = None,
        checkpoint: Optional[str] = None,
    ) -> "ExecutionResultProtocol":
        """Delegate to legacy implementation (temporary stub)."""
        # The compiled_graph should have an invoke method from legacy implementation
        if hasattr(compiled_graph, "invoke"):
            return await compiled_graph.invoke(
                initial_state,
                thread_id=thread_id,
                checkpoint=checkpoint,
            )
        else:
            # Create a stub result
            return ExecutionResult(
                final_state=initial_state,
                metrics={
                    "duration_seconds": 0.0,
                    "nodes_executed": 0,
                },
            )

    async def stream(
        self,
        compiled_graph: "CompiledGraphProtocol",
        initial_state: Dict[str, Any],
        *,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator["ExecutionEventProtocol"]:
        """Stream execution events from a compiled workflow.

        Args:
            compiled_graph: Compiled workflow graph
            initial_state: Initial workflow state
            thread_id: Optional thread ID for checkpointing

        Yields:
            ExecutionEventProtocol: Execution events as they occur

        Example:
            async for event in executor.stream(compiled, {"query": "term"}):
                print(f"{event.node_id}: {event.event_type}")
        """
        # TODO: Implement proper streaming
        # For now, delegate to legacy implementation
        if hasattr(compiled_graph, "stream"):
            async for event in compiled_graph.stream(initial_state, thread_id=thread_id):
                yield event


class ExecutionResult:
    """Execution result from workflow executor.

    Minimal implementation for migration phase.
    """

    def __init__(self, final_state: Dict[str, Any], metrics: Dict[str, Any]):
        self._final_state = final_state
        self._metrics = metrics

    @property
    def final_state(self) -> Dict[str, Any]:
        return self._final_state

    @property
    def metrics(self) -> Dict[str, Any]:
        return self._metrics


__all__ = [
    "WorkflowExecutor",
    "ExecutionResult",
]
