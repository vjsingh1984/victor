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

"""Executor wrapper for compiled workflow graphs.

Provides a thin execution layer around compiled StateGraph workflows.
The actual execution logic lives in the compiled graph's invoke/stream methods.

Note: This is distinct from victor.workflows.executor.WorkflowExecutor which
executes workflow definitions directly with an orchestrator.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Optional

if TYPE_CHECKING:
    from victor.workflows.compiler_protocols import (
        CompiledGraphProtocol,
        ExecutionEventProtocol,
        ExecutionResultProtocol,
    )

logger = logging.getLogger(__name__)


class CompiledWorkflowExecutor:
    """Executor for compiled workflow graphs.

    Wraps compiled StateGraph execution with consistent interface.
    Delegates to the compiled graph's invoke/stream methods.

    Attributes:
        _orchestrator_pool: Pool of orchestrators for multi-provider workflows
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
        """
        logger.info("Executing compiled workflow graph...")

        if hasattr(compiled_graph, "invoke"):
            return await compiled_graph.invoke(
                initial_state,
                thread_id=thread_id,
                checkpoint=checkpoint,
            )
        else:
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
        """
        if hasattr(compiled_graph, "stream"):
            async for event in compiled_graph.stream(initial_state, thread_id=thread_id):
                yield event


class ExecutionResult:
    """Execution result from compiled workflow executor."""

    def __init__(self, final_state: Dict[str, Any], metrics: Dict[str, Any]):
        self._final_state = final_state
        self._metrics = metrics

    @property
    def final_state(self) -> Dict[str, Any]:
        return self._final_state

    @property
    def metrics(self) -> Dict[str, Any]:
        return self._metrics


# Backward compatibility aliases
WorkflowExecutor = CompiledWorkflowExecutor

__all__ = [
    "CompiledWorkflowExecutor",
    "WorkflowExecutor",
    "ExecutionResult",
]
