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

"""WorkflowChatCoordinator - Coordinates workflow-based chat execution.

This module provides the WorkflowChatCoordinator which handles the coordination
of workflow-based chat execution. It acts as a bridge between the agent layer
and the framework workflow engine.

Responsibilities:
    - Load chat workflows from verticals
    - Manage workflow execution lifecycle
    - Handle streaming via GraphExecutionCoordinator
    - Provide observability events

Design Principles:
    - Single Responsibility: Focus on workflow chat coordination
    - Open/Closed: Extensible via protocol-based dependencies
    - Dependency Inversion: Depend on abstractions, not concretions

Architecture:
    WorkflowChatCoordinator
    ├── WorkflowRegistry - Workflow discovery and registration
    ├── GraphExecutionCoordinator - StateGraph execution
    └── EventBus - Observability events

Key Features:
    - Automatic workflow discovery from verticals
    - Session management for multi-turn conversations
    - Streaming execution with real-time updates
    - Event emission for observability
    - Error handling with recovery

Example:
    coordinator = WorkflowChatCoordinator(
        workflow_registry=registry,
        graph_coordinator=graph_coord,
    )

    # Execute chat workflow
    result = await coordinator.execute_chat(
        workflow_name="coding_chat",
        user_message="Fix the bug",
    )

    # Stream execution
    async for event in coordinator.stream_chat(
        workflow_name="coding_chat",
        user_message="Add tests",
    ):
        print(f"Event: {event.event_type}")
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
)

if TYPE_CHECKING:
    from victor.framework.coordinators.graph_coordinator import GraphExecutionCoordinator
    from victor.framework.graph import CompiledGraph
    from victor.workflows.registry import WorkflowRegistry
    from victor.framework.workflow_engine import WorkflowEvent

logger = logging.getLogger(__name__)


class ChatEventType(str, Enum):
    """Types of chat execution events."""

    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_ERROR = "workflow_error"
    NODE_START = "node_start"
    NODE_COMPLETE = "node_complete"
    STREAM_CHUNK = "stream_chunk"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


@dataclass
class ChatExecutionEvent:
    """Event emitted during chat workflow execution.

    Attributes:
        event_type: Type of event
        workflow_name: Name of the workflow being executed
        session_id: Session identifier
        timestamp: Event timestamp
        node_id: Current node ID (for node events)
        data: Event-specific data
    """

    event_type: ChatEventType
    workflow_name: str
    session_id: str
    timestamp: float
    node_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "workflow_name": self.workflow_name,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "data": self.data,
        }


@dataclass
class ChatExecutionConfig:
    """Configuration for chat workflow execution.

    Attributes:
        max_iterations: Maximum agentic loop iterations
        enable_streaming: Enable event streaming
        enable_checkpoints: Enable state checkpointing
        checkpoint_interval: Checkpoint save interval (iterations)
        timeout_seconds: Execution timeout
        enable_events: Enable event bus emission
    """

    max_iterations: int = 50
    enable_streaming: bool = True
    enable_checkpoints: bool = True
    checkpoint_interval: int = 5
    timeout_seconds: int = 300
    enable_events: bool = True


class WorkflowChatCoordinator:
    """Coordinates workflow-based chat execution.

    This coordinator manages the complete lifecycle of chat workflow execution,
    from loading workflows to executing them and managing sessions.

    Responsibilities:
    - Load chat workflows from verticals
    - Manage workflow execution lifecycle
    - Handle streaming via GraphExecutionCoordinator
    - Provide observability events

    The coordinator maintains session state for multi-turn conversations
    and can execute workflows both synchronously and via streaming.

    Attributes:
        _workflow_registry: Registry for workflow discovery
        _graph_coordinator: Coordinator for StateGraph execution
        _sessions: Active chat sessions
        _config: Execution configuration
    """

    def __init__(
        self,
        workflow_registry: "WorkflowRegistry",
        graph_coordinator: "GraphExecutionCoordinator",
        config: Optional[ChatExecutionConfig] = None,
    ) -> None:
        """Initialize the WorkflowChatCoordinator.

        Args:
            workflow_registry: Registry for workflow discovery
            graph_coordinator: Coordinator for StateGraph execution
            config: Optional execution configuration
        """
        self._workflow_registry = workflow_registry
        self._graph_coordinator = graph_coordinator
        self._config = config or ChatExecutionConfig()
        self._sessions: Dict[str, Dict[str, Any]] = {}

        # Event callbacks
        self._event_callbacks: List[Callable[[ChatExecutionEvent], None]] = []

    async def execute_chat(
        self,
        workflow_name: str,
        user_message: str,
        session_id: Optional[str] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute chat workflow and return result.

        Args:
            workflow_name: Name of the workflow to execute
            user_message: User's input message
            session_id: Optional session identifier
            initial_state: Optional initial workflow state

        Returns:
            Dictionary with execution results

        Raises:
            ValueError: If workflow not found
            RuntimeError: If execution fails
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Get or create session
        session = self._get_or_create_session(session_id)

        # Load workflow + compiled graph
        compiled_graph = self._get_compiled_graph(workflow_name)

        # Prepare initial state
        state = self._prepare_state(
            user_message=user_message,
            session=session,
            initial_state=initial_state,
        )

        # Emit start event
        self._emit_event(
            ChatExecutionEvent(
                event_type=ChatEventType.WORKFLOW_START,
                workflow_name=workflow_name,
                session_id=session_id,
                timestamp=time.time(),
                data={"message_length": len(user_message)},
            )
        )

        start_time = time.time()

        try:
            # Execute workflow
            result = await self._graph_coordinator.execute(
                graph=compiled_graph,
                initial_state=state,
            )

            duration = time.time() - start_time

            # Update session
            self._update_session(session_id, result.final_state)

            # Emit complete event
            self._emit_event(
                ChatExecutionEvent(
                    event_type=ChatEventType.WORKFLOW_COMPLETE,
                    workflow_name=workflow_name,
                    session_id=session_id,
                    timestamp=time.time(),
                    data={
                        "duration_seconds": duration,
                        "iterations": result.final_state.get("iteration_count", 0),
                        "nodes_executed": len(result.nodes_executed),
                    },
                )
            )

            return {
                "success": result.success,
                "content": result.final_state.get("final_response", ""),
                "final_state": result.final_state,
                "duration_seconds": duration,
                "nodes_executed": result.nodes_executed,
            }

        except Exception as e:
            logger.error(f"Chat workflow execution failed: {e}", exc_info=True)

            # Emit error event
            self._emit_event(
                ChatExecutionEvent(
                    event_type=ChatEventType.WORKFLOW_ERROR,
                    workflow_name=workflow_name,
                    session_id=session_id,
                    timestamp=time.time(),
                    data={"error": str(e)},
                )
            )

            raise RuntimeError(f"Chat workflow execution failed: {e}") from e

    async def stream_chat(
        self,
        workflow_name: str,
        user_message: str,
        session_id: Optional[str] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[ChatExecutionEvent]:
        """Stream chat workflow execution events.

        Args:
            workflow_name: Name of the workflow to execute
            user_message: User's input message
            session_id: Optional session identifier
            initial_state: Optional initial workflow state

        Yields:
            ChatExecutionEvent for each execution step

        Raises:
            ValueError: If workflow not found
            RuntimeError: If execution fails
        """
        # Generate session ID if not provided
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Get or create session
        session = self._get_or_create_session(session_id)

        # Load workflow + compiled graph
        compiled_graph = self._get_compiled_graph(workflow_name)

        # Prepare initial state
        state = self._prepare_state(
            user_message=user_message,
            session=session,
            initial_state=initial_state,
        )

        # Emit start event
        yield ChatExecutionEvent(
            event_type=ChatEventType.WORKFLOW_START,
            workflow_name=workflow_name,
            session_id=session_id,
            timestamp=time.time(),
            data={"message_length": len(user_message)},
        )

        try:
            # Stream workflow execution
            async for event in self._graph_coordinator.stream(
                graph=compiled_graph,
                initial_state=state,
            ):
                # Convert workflow event to chat event
                if event.event_type == "node_complete":
                    yield ChatExecutionEvent(
                        event_type=ChatEventType.NODE_COMPLETE,
                        workflow_name=workflow_name,
                        session_id=session_id,
                        timestamp=event.timestamp,
                        node_id=event.node_id,
                        data={"state_snapshot": event.state_snapshot},
                    )

                elif event.event_type == "error":
                    yield ChatExecutionEvent(
                        event_type=ChatEventType.WORKFLOW_ERROR,
                        workflow_name=workflow_name,
                        session_id=session_id,
                        timestamp=event.timestamp,
                        data={"error": str(event.data.get("error", "Unknown error"))},
                    )

            # Emit complete event
            yield ChatExecutionEvent(
                event_type=ChatEventType.WORKFLOW_COMPLETE,
                workflow_name=workflow_name,
                session_id=session_id,
                timestamp=time.time(),
            )

        except Exception as e:
            logger.error(f"Chat workflow streaming failed: {e}", exc_info=True)

            yield ChatExecutionEvent(
                event_type=ChatEventType.WORKFLOW_ERROR,
                workflow_name=workflow_name,
                session_id=session_id,
                timestamp=time.time(),
                data={"error": str(e)},
            )

    def register_event_callback(
        self,
        callback: Callable[[ChatExecutionEvent], None],
    ) -> None:
        """Register a callback for chat execution events.

        Args:
            callback: Function to call with events
        """
        self._event_callbacks.append(callback)

    def unregister_event_callback(
        self,
        callback: Callable[[ChatExecutionEvent], None],
    ) -> None:
        """Unregister an event callback.

        Args:
            callback: Previously registered callback
        """
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)

    def end_session(self, session_id: str) -> None:
        """End a chat session.

        Args:
            session_id: Session identifier
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Ended session {session_id}")

    def _get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Get existing session or create new one.

        Args:
            session_id: Session identifier

        Returns:
            Session dictionary
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "messages": [],
                "iteration_count": 0,
                "metadata": {},
            }
            logger.debug(f"Created new session {session_id}")
        return self._sessions[session_id]

    def _prepare_state(
        self,
        user_message: str,
        session: Dict[str, Any],
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Prepare initial state for workflow execution.

        Args:
            user_message: User's message
            session: Session state
            initial_state: Additional initial state

        Returns:
            Initial state dictionary
        """
        state = {
            "user_message": user_message,
            "messages": session.get("messages", []).copy(),
            "iteration_count": session.get("iteration_count", 0),
            "metadata": session.get("metadata", {}).copy(),
            "max_iterations": self._config.max_iterations,
        }

        if initial_state:
            state.update(initial_state)

        return state

    def _update_session(self, session_id: str, final_state: Dict[str, Any]) -> None:
        """Update session with final state.

        Args:
            session_id: Session identifier
            final_state: Final workflow state
        """
        if session_id in self._sessions:
            self._sessions[session_id]["messages"] = final_state.get("messages", [])
            self._sessions[session_id]["iteration_count"] = final_state.get("iteration_count", 0)
            self._sessions[session_id]["metadata"] = final_state.get("metadata", {})

    def _get_workflow(self, workflow_name: str) -> Any:
        """Get workflow by name.

        Args:
            workflow_name: Name of the workflow

        Returns:
            Workflow definition or compiled workflow wrapper

        Raises:
            ValueError: If workflow not found
        """
        workflow = self._workflow_registry.get(workflow_name)
        if workflow is None:
            available = self._workflow_registry.list_workflows()
            raise ValueError(
                f"Workflow '{workflow_name}' not found. " f"Available workflows: {available}"
            )
        return workflow

    def _get_compiled_graph(self, workflow_name: str) -> "CompiledGraph":
        """Get compiled graph for a workflow.

        Compiles WorkflowDefinitions on-demand and caches compiled graphs
        on the workflow instance when possible.
        """
        workflow = self._get_workflow(workflow_name)

        # Legacy dict-based workflows
        if isinstance(workflow, dict):
            compiled = workflow.get("compiled_graph")
            if compiled is None:
                raise ValueError(f"Workflow '{workflow_name}' is not compiled")
            return compiled

        compiled_graph = getattr(workflow, "compiled_graph", None)
        if compiled_graph is None:
            from victor.workflows.graph_compiler import compile_workflow_definition

            compiled_graph = compile_workflow_definition(
                workflow,
                runner_registry=self._graph_coordinator.runner_registry,
            )
            try:
                workflow.compiled_graph = compiled_graph
            except Exception:
                pass

        return compiled_graph

    def _emit_event(self, event: ChatExecutionEvent) -> None:
        """Emit event to all registered callbacks.

        Args:
            event: Event to emit
        """
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback failed: {e}", exc_info=True)

        # Also emit to observability bus if enabled
        if self._config.enable_events:
            try:
                from victor.core.events import get_observability_bus

                event_bus = get_observability_bus()
                asyncio.create_task(
                    event_bus.emit(
                        topic=f"chat.{event.event_type.value}",
                        data=event.to_dict(),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to emit to observability bus: {e}")


__all__ = [
    "WorkflowChatCoordinator",
    "ChatExecutionEvent",
    "ChatEventType",
    "ChatExecutionConfig",
]
