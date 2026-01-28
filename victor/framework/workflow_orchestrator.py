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

"""WorkflowOrchestrator - Domain-agnostic workflow-based chat orchestration.

This module provides the WorkflowOrchestrator, a domain-agnostic orchestrator
that executes chat workflows through the framework workflow engine. This
orchestrator has ZERO domain knowledge - no coding-specific logic, no
hard-coded tool names, no file path extraction.

Design Principles:
    - Domain Agnostic: Works with any vertical (coding, devops, research, etc.)
    - Protocol-Based: Depends on abstractions, not concrete implementations
    - Workflow-First: Uses StateGraph for execution, not imperative code
    - Observable: Full visibility into workflow execution

Architecture:
    WorkflowOrchestrator (Facade)
    ├── WorkflowCoordinator - Workflow registration and discovery
    ├── GraphExecutionCoordinator - StateGraph execution
    └── Protocol-based dependencies - Loose coupling

Key Features:
    - Execute chat workflows by name
    - Stream workflow execution for real-time updates
    - Support for multiple concurrent sessions
    - Checkpointing for recovery
    - Zero domain knowledge

Example:
    orchestrator = WorkflowOrchestrator(
        workflow_coordinator=workflow_coord,
        graph_coordinator=graph_coord,
    )

    # Execute coding chat workflow
    result = await orchestrator.chat(
        message="Fix the bug in main.py",
        workflow="coding_chat",
    )

    # Stream execution
    async for chunk in orchestrator.stream_chat(
        message="Deploy to production",
        workflow="devops_chat",
    ):
        print(chunk.content, end="")
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
)

from victor.framework.protocols import (
    ChatResultProtocol,
    ChatResult,
    MutableChatState,
)
from victor.providers.base import CompletionResponse, StreamChunk

if TYPE_CHECKING:
    from victor.agent.coordinators.workflow_coordinator import WorkflowCoordinator
    from victor.framework.coordinators.graph_coordinator import GraphExecutionCoordinator

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """Domain-agnostic orchestrator using framework workflow engine.

    This orchestrator has ZERO domain knowledge:
    - No coding-specific logic
    - No hard-coded tool names
    - No file path extraction
    - Pure workflow execution

    It delegates all domain-specific logic to the workflow definitions
    provided by verticals.

    Attributes:
        _workflow_coordinator: Handles workflow registration and discovery
        _graph_coordinator: Handles StateGraph execution
        _sessions: Active chat sessions keyed by session_id
    """

    def __init__(
        self,
        workflow_coordinator: "WorkflowCoordinator",
        graph_coordinator: "GraphExecutionCoordinator",
    ) -> None:
        """Initialize the WorkflowOrchestrator.

        Args:
            workflow_coordinator: Coordinator for workflow registration/discovery
            graph_coordinator: Coordinator for StateGraph execution
        """
        self._workflow_coordinator = workflow_coordinator
        self._graph_coordinator = graph_coordinator
        self._sessions: Dict[str, MutableChatState] = {}

    async def chat(
        self,
        message: str,
        workflow: str = "default_chat",
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Execute chat workflow.

        This method loads the specified workflow, executes it with the
        provided message and initial state, and returns a CompletionResponse.

        Args:
            message: User message
            workflow: Workflow name (e.g., 'coding_chat', 'devops_chat')
            session_id: Optional session identifier for conversation tracking
            **kwargs: Additional workflow-specific parameters

        Returns:
            CompletionResponse with result

        Raises:
            ValueError: If workflow not found
            RuntimeError: If workflow execution fails

        Example:
            result = await orchestrator.chat(
                message="Fix the bug in main.py",
                workflow="coding_chat",
            )
            print(result.content)
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Get or create session state
        session_state = self._get_or_create_session(session_id)

        # Prepare initial state for workflow
        initial_state = self._prepare_initial_state(
            message=message,
            session_state=session_state,
            **kwargs,
        )

        # Get and execute workflow
        workflow_def = self._get_workflow(workflow)

        logger.info(
            f"Executing workflow '{workflow}' for session {session_id}: "
            f"{len(initial_state.get('messages', []))} messages in history"
        )

        start_time = time.time()

        try:
            # Execute workflow via graph coordinator
            execution_result = await self._graph_coordinator.execute(
                graph=workflow_def.compiled_graph,
                initial_state=initial_state,
            )

            duration = time.time() - start_time

            # Extract results
            final_state = execution_result.final_state
            content = final_state.get("final_response", final_state.get("response", ""))
            iterations = final_state.get("iteration_count", 0)

            # Update session state
            session_state.add_message("user", message)
            session_state.add_message("assistant", content)
            session_state.increment_iteration()

            # Create CompletionResponse
            response = CompletionResponse(
                content=content,
                role="assistant",
                tool_calls=final_state.get("tool_calls"),
                model=None,
                raw_response=None,
                stop_reason="stop",
                usage=None,
                metadata={
                    "workflow": workflow,
                    "session_id": session_id,
                    "iterations": iterations,
                    "duration_seconds": duration,
                    "nodes_executed": execution_result.nodes_executed,
                },
            )

            logger.info(
                f"Workflow '{workflow}' completed in {duration:.2f}s: "
                f"{iterations} iterations, {len(execution_result.nodes_executed)} nodes"
            )

            return response

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            raise RuntimeError(f"Workflow execution failed: {e}") from e

    async def stream_chat(
        self,
        message: str,
        workflow: str = "default_chat",
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat workflow execution.

        This method executes the workflow and yields StreamChunk objects
        as the workflow progresses, enabling real-time updates.

        Args:
            message: User message
            workflow: Workflow name (e.g., 'coding_chat', 'devops_chat')
            session_id: Optional session identifier for conversation tracking
            **kwargs: Additional workflow-specific parameters

        Yields:
            StreamChunk objects with incremental response

        Raises:
            ValueError: If workflow not found
            RuntimeError: If workflow execution fails

        Example:
            async for chunk in await orchestrator.stream_chat(
                message="Deploy to production",
                workflow="devops_chat",
            ):
                print(chunk.content, end="")
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Get or create session state
        session_state = self._get_or_create_session(session_id)

        # Prepare initial state for workflow
        initial_state = self._prepare_initial_state(
            message=message,
            session_state=session_state,
            **kwargs,
        )

        # Get and execute workflow
        workflow_def = self._get_workflow(workflow)

        logger.info(
            f"Streaming workflow '{workflow}' for session {session_id}: "
            f"{len(initial_state.get('messages', []))} messages in history"
        )

        try:
            # Stream workflow execution
            async for event in self._graph_coordinator.stream(
                graph=workflow_def.compiled_graph,
                initial_state=initial_state,
            ):
                # Convert workflow event to StreamChunk
                if event.event_type == "node_complete":
                    node_output = event.state_snapshot
                    content = node_output.get("content", "")

                    if content:
                        yield StreamChunk(
                            content=content,
                            role="assistant",
                            finish_reason=None,
                            index=0,
                        )

                elif event.event_type == "error":
                    error_msg = event.data.get("error", "Unknown error") if event.data else "Unknown error"
                    logger.error(f"Workflow streaming error: {error_msg}")
                    yield StreamChunk(
                        content=f"\n[Error: {error_msg}]",
                        role="assistant",
                        finish_reason="error",
                        index=0,
                    )

            # Final update
            logger.info(f"Workflow '{workflow}' streaming completed")

        except Exception as e:
            logger.error(f"Workflow streaming failed: {e}", exc_info=True)
            raise RuntimeError(f"Workflow streaming failed: {e}") from e

    def get_available_workflows(self) -> List[str]:
        """Get list of available chat workflows.

        Returns:
            List of workflow names that can be used for chat
        """
        return self._workflow_coordinator.list_workflows()

    def has_workflow(self, workflow_name: str) -> bool:
        """Check if a chat workflow is available.

        Args:
            workflow_name: Name of the workflow to check

        Returns:
            True if workflow exists and can be used for chat
        """
        return self._workflow_coordinator.has_workflow(workflow_name)

    def end_session(self, session_id: str) -> None:
        """End a chat session and cleanup resources.

        Args:
            session_id: Session identifier to end
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Ended session {session_id}")

    def _get_or_create_session(self, session_id: str) -> MutableChatState:
        """Get existing session or create new one.

        Args:
            session_id: Session identifier

        Returns:
            MutableChatState for the session
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = MutableChatState()
            logger.debug(f"Created new session {session_id}")
        return self._sessions[session_id]

    def _prepare_initial_state(
        self,
        message: str,
        session_state: MutableChatState,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare initial state for workflow execution.

        Args:
            message: User message
            session_state: Current session state
            **kwargs: Additional parameters

        Returns:
            Initial state dictionary for workflow
        """
        return {
            "user_message": message,
            "messages": session_state.messages.copy(),
            "iteration_count": session_state.iteration_count,
            "metadata": session_state.metadata.copy(),
            **kwargs,
        }

    def _get_workflow(self, workflow_name: str) -> Any:
        """Get workflow definition by name.

        Args:
            workflow_name: Name of the workflow

        Returns:
            Workflow definition with compiled_graph attribute

        Raises:
            ValueError: If workflow not found
        """
        if not self._workflow_coordinator.has_workflow(workflow_name):
            available = self.get_available_workflows()
            raise ValueError(
                f"Workflow '{workflow_name}' not found. "
                f"Available workflows: {available}"
            )

        # Get workflow from registry
        from victor.workflows.base import get_global_registry

        workflow_registry = get_global_registry()
        workflow = workflow_registry.get(workflow_name)

        # Compile the workflow if it doesn't have a compiled_graph
        if not hasattr(workflow, "compiled_graph") or workflow.compiled_graph is None:
            from victor.framework.workflow_engine import WorkflowEngine

            # Create a simple engine for compilation
            engine = WorkflowEngine()
            workflow.compiled_graph = engine.compile_workflow(workflow)

        return workflow


__all__ = [
    "WorkflowOrchestrator",
]
