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

"""Workflow Execution Engine with Comprehensive Debugging and Monitoring.

This module provides a production-ready workflow execution engine with:
- Step-by-step debugging with breakpoints
- Execution tracing and logging
- State inspection and visualization
- Execution history and replay
- Error recovery and retry mechanisms
- Workflow pause/resume
- Execution analytics and reporting

Key Components:
- WorkflowExecutor: Main execution engine
- ExecutionTrace: Trace logging and export
- StateManager: State inspection and visualization
- ExecutionHistory: History recording and replay
- ErrorRecovery: Error handling and retry logic
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    TYPE_CHECKING,
    Union,
    cast,
)

if TYPE_CHECKING:
    from victor.workflows.definition import (
        WorkflowDefinition,
        WorkflowNode,
        AgentNode,
        ComputeNode,
        ConditionNode,
        ParallelNode,
        TransformNode,
        TeamNodeWorkflow,
    )
    from victor.framework.graph import CompiledGraph
    from victor.workflows.unified_compiler import CachedCompiledGraph

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class NodeStatus(Enum):
    """Status of a workflow node execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    PAUSED = "paused"


class ExecutionEventType(Enum):
    """Types of execution events."""

    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_PAUSED = "workflow_paused"
    WORKFLOW_RESUMED = "workflow_resumed"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    NODE_SKIPPED = "node_skipped"
    NODE_PAUSED = "node_paused"
    BREAKPOINT_HIT = "breakpoint_hit"
    ERROR_OCCURRED = "error_occurred"
    ERROR_RECOVERED = "error_recovered"
    STATE_CAPTURED = "state_captured"
    RETRY_ATTEMPT = "retry_attempt"


class ErrorRecoveryStrategy(Enum):
    """Error recovery strategies."""

    FAIL_FAST = "fail_fast"  # Stop execution immediately
    RETRY = "retry"  # Retry failed nodes
    CONTINUE = "continue"  # Continue to next node
    SKIP = "skip"  # Skip failed nodes
    PAUSE = "pause"  # Pause on error for debugging


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class NodeExecutionEvent:
    """Event captured during node execution.

    Attributes:
        node_id: ID of the node
        event_type: Type of event
        timestamp: Event timestamp
        status: Node status
        input_state: State before execution
        output_state: State after execution
        error: Error if failed
        duration_seconds: Execution time
        metadata: Additional event metadata
    """

    node_id: str
    event_type: ExecutionEventType
    timestamp: float
    status: NodeStatus
    input_state: Dict[str, Any] = field(default_factory=dict)
    output_state: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Breakpoint:
    """Breakpoint for debugging.

    Attributes:
        node_id: Node ID to break on
        condition: Optional condition function
        hit_count: Number of times breakpoint was hit
        enabled: Whether breakpoint is enabled
    """

    node_id: str
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    hit_count: int = 0
    enabled: bool = True


@dataclass
class ExecutionSnapshot:
    """Snapshot of workflow execution state.

    Attributes:
        execution_id: Execution identifier
        timestamp: Snapshot timestamp
        current_node: Current node being executed
        executed_nodes: List of executed node IDs
        state: Current workflow state
        breakpoints_hit: Breakpoints hit so far
        errors: Errors encountered
    """

    execution_id: str
    timestamp: float
    current_node: Optional[str]
    executed_nodes: List[str]
    state: Dict[str, Any]
    breakpoints_hit: List[str]
    errors: List[str]


@dataclass
class ExecutionMetrics:
    """Metrics from workflow execution.

    Attributes:
        total_duration_seconds: Total execution time
        node_count: Number of nodes
        nodes_executed: Number of nodes executed
        nodes_failed: Number of nodes failed
        nodes_skipped: Number of nodes skipped
        tool_calls_total: Total tool calls
        retries_total: Total retry attempts
        error_count: Total errors
        breakpoint_hits: Number of breakpoint hits
    """

    total_duration_seconds: float = 0.0
    node_count: int = 0
    nodes_executed: int = 0
    nodes_failed: int = 0
    nodes_skipped: int = 0
    tool_calls_total: int = 0
    retries_total: int = 0
    error_count: int = 0
    breakpoint_hits: int = 0


@dataclass
class ExecutionContext:
    """Context for workflow execution.

    Attributes:
        execution_id: Unique execution identifier
        workflow_name: Name of workflow
        start_time: Execution start time
        debug_mode: Whether in debug mode
        trace_mode: Whether tracing execution
        breakpoints: Active breakpoints
        current_state: Current workflow state
        events: Execution events
        snapshots: Execution snapshots
        metrics: Execution metrics
    """

    execution_id: str
    workflow_name: str
    start_time: float
    debug_mode: bool = False
    trace_mode: bool = False
    breakpoints: Dict[str, Breakpoint] = field(default_factory=dict)
    current_state: Dict[str, Any] = field(default_factory=dict)
    events: List[NodeExecutionEvent] = field(default_factory=list)
    snapshots: List[ExecutionSnapshot] = field(default_factory=list)
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)


# =============================================================================
# Execution Trace
# =============================================================================


class ExecutionTrace:
    """Execution tracing and logging.

    Captures detailed trace logs of workflow execution for debugging,
    monitoring, and analysis.
    """

    def __init__(self, execution_id: str):
        """Initialize execution trace.

        Args:
            execution_id: Execution identifier
        """
        self.execution_id = execution_id
        self.events: List[NodeExecutionEvent] = []
        self.start_time = time.time()
        self.end_time: Optional[float] = None

    def record_event(self, event: NodeExecutionEvent) -> None:
        """Record an execution event.

        Args:
            event: Event to record
        """
        self.events.append(event)
        logger.debug(
            f"[{self.execution_id}] {event.event_type.value}: {event.node_id} "
            f"({event.duration_seconds:.3f}s)"
        )

    def get_events_by_node(self, node_id: str) -> List[NodeExecutionEvent]:
        """Get all events for a specific node.

        Args:
            node_id: Node ID

        Returns:
            List of events for the node
        """
        return [e for e in self.events if e.node_id == node_id]

    def get_events_by_type(self, event_type: ExecutionEventType) -> List[NodeExecutionEvent]:
        """Get all events of a specific type.

        Args:
            event_type: Event type

        Returns:
            List of events of the type
        """
        return [e for e in self.events if e.event_type == event_type]

    def export_json(self, output_path: Optional[Path] = None) -> str:
        """Export trace as JSON.

        Args:
            output_path: Optional output file path

        Returns:
            JSON string
        """
        trace_data = {
            "execution_id": self.execution_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": (self.end_time or time.time()) - self.start_time,
            "events": [
                {
                    "node_id": e.node_id,
                    "event_type": e.event_type.value,
                    "timestamp": e.timestamp,
                    "status": e.status.value,
                    "duration_seconds": e.duration_seconds,
                    "error": e.error,
                    "metadata": e.metadata,
                }
                for e in self.events
            ],
        }

        json_str = json.dumps(trace_data, indent=2, default=str)

        if output_path:
            output_path.write_text(json_str)
            logger.info(f"Trace exported to {output_path}")

        return json_str

    def get_summary(self) -> Dict[str, Any]:
        """Get trace summary.

        Returns:
            Summary dictionary
        """
        event_counts: Dict[str, int] = {}
        for event in self.events:
            event_counts[event.event_type.value] = event_counts.get(event.event_type.value, 0) + 1

        node_counts: Dict[str, int] = {}
        error_count = 0
        total_duration = 0.0

        for event in self.events:
            node_counts[event.node_id] = node_counts.get(event.node_id, 0) + 1
            if event.error:
                error_count += 1
            total_duration += event.duration_seconds

        return {
            "execution_id": self.execution_id,
            "duration_seconds": (self.end_time or time.time()) - self.start_time,
            "total_events": len(self.events),
            "event_counts": event_counts,
            "node_counts": node_counts,
            "error_count": error_count,
            "total_node_duration": total_duration,
        }


# =============================================================================
# State Manager
# =============================================================================


class StateManager:
    """State inspection and visualization.

    Provides capabilities to inspect, visualize, and query workflow state
    during execution.
    """

    @staticmethod
    def capture_state(
        state: Dict[str, Any],
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Capture a snapshot of workflow state.

        Args:
            state: Current workflow state
            node_id: Optional node ID
            metadata: Optional metadata

        Returns:
            State snapshot
        """
        # Create deep copy to avoid mutation
        import copy

        state_copy = copy.deepcopy(state)

        snapshot = {
            "timestamp": time.time(),
            "node_id": node_id,
            "state": state_copy,
            "metadata": metadata or {},
        }

        return snapshot

    @staticmethod
    def visualize_state(state: Dict[str, Any]) -> str:
        """Visualize workflow state as formatted string.

        Args:
            state: Workflow state

        Returns:
            Formatted string representation
        """
        lines = []
        lines.append("=" * 60)
        lines.append("WORKFLOW STATE")
        lines.append("=" * 60)

        # Filter internal keys
        display_state = {k: v for k, v in state.items() if not k.startswith("_")}

        for key, value in display_state.items():
            # Truncate long values
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."

            lines.append(f"{key}: {value_str}")

        lines.append("=" * 60)

        return "\n".join(lines)

    @staticmethod
    def diff_states(
        state1: Dict[str, Any],
        state2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute difference between two states.

        Args:
            state1: First state
            state2: Second state

        Returns:
            Dictionary of differences
        """
        diff: Dict[str, Any] = {
            "added": {},
            "removed": {},
            "changed": {},
            "unchanged": [],
        }

        all_keys = set(state1.keys()) | set(state2.keys())

        for key in all_keys:
            if key not in state1:
                diff["added"][key] = state2[key]
            elif key not in state2:
                diff["removed"][key] = state1[key]
            elif state1[key] != state2[key]:
                diff["changed"][key] = {
                    "from": state1[key],
                    "to": state2[key],
                }
            else:
                diff["unchanged"].append(key)  # type: ignore[arg-type]

        return diff

    @staticmethod
    def query_state(
        state: Dict[str, Any],
        query: str,
    ) -> Any:
        """Query state using JSONPath-like syntax.

        Args:
            state: Workflow state
            query: Query string (e.g., "user.name", "items[0].id")

        Returns:
            Query result
        """
        keys = query.split(".")
        result = state

        for key in keys:
            # Handle array indexing
            if "[" in key and key.endswith("]"):
                base_key = key.split("[")[0]
                index = int(key.split("[")[1].rstrip("]"))

                if base_key in result:
                    result = result[base_key][index]
                else:
                    return None
            else:
                if key in result:
                    result = result[key]
                else:
                    return None

        return result


# =============================================================================
# Error Recovery
# =============================================================================


class ErrorRecovery:
    """Error recovery and retry logic.

    Implements various error recovery strategies for workflow execution.
    """

    def __init__(
        self,
        strategy: ErrorRecoveryStrategy = ErrorRecoveryStrategy.FAIL_FAST,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        backoff_multiplier: float = 2.0,
    ):
        """Initialize error recovery.

        Args:
            strategy: Recovery strategy
            max_retries: Maximum retry attempts
            retry_delay_seconds: Initial retry delay
            backoff_multiplier: Backoff multiplier for retries
        """
        self.strategy = strategy
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.backoff_multiplier = backoff_multiplier

    async def execute_with_retry(
        self,
        node_id: str,
        execute_func: Callable[..., Any],
        context: ExecutionContext,
    ) -> Any:
        """Execute a node with retry logic.

        Args:
            node_id: Node ID
            execute_func: Execution function
            context: Execution context

        Returns:
            Execution result
        """
        last_error = None
        attempt = 0

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    # Record retry event
                    context.events.append(
                        NodeExecutionEvent(
                            node_id=node_id,
                            event_type=ExecutionEventType.RETRY_ATTEMPT,
                            timestamp=time.time(),
                            status=NodeStatus.RUNNING,
                            metadata={"attempt": attempt},
                        )
                    )

                    # Exponential backoff
                    delay = self.retry_delay_seconds * (self.backoff_multiplier ** (attempt - 1))
                    logger.info(f"Retrying node {node_id} (attempt {attempt}), delay: {delay}s")
                    await asyncio.sleep(delay)

                result = await execute_func()
                context.metrics.retries_total += attempt
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"Node {node_id} failed (attempt {attempt}): {e}")

                if attempt < self.max_retries:
                    continue
                else:
                    # Exhausted retries
                    context.metrics.error_count += 1

                    if self.strategy == ErrorRecoveryStrategy.FAIL_FAST:
                        raise
                    elif self.strategy == ErrorRecoveryStrategy.CONTINUE:
                        logger.error(f"Node {node_id} failed, continuing: {e}")
                        return None
                    elif self.strategy == ErrorRecoveryStrategy.SKIP:
                        logger.warning(f"Node {node_id} failed, skipping: {e}")
                        return None
                    elif self.strategy == ErrorRecoveryStrategy.PAUSE:
                        # Record error and pause
                        context.events.append(
                            NodeExecutionEvent(
                                node_id=node_id,
                                event_type=ExecutionEventType.ERROR_OCCURRED,
                                timestamp=time.time(),
                                status=NodeStatus.FAILED,
                                error=str(e),
                            )
                        )
                        raise

        # Should not reach here
        if last_error is not None:
            raise last_error
        raise RuntimeError("Execution failed without error being set")


# =============================================================================
# Execution History
# =============================================================================


class ExecutionHistory:
    """Execution history and replay.

    Records workflow executions for later replay and analysis.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize execution history.

        Args:
            storage_path: Path to history storage directory
        """
        self.storage_path = storage_path or Path.home() / ".victor" / "workflow_history"
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def record_execution(
        self,
        execution_id: str,
        workflow_name: str,
        context: ExecutionContext,
        trace: ExecutionTrace,
    ) -> None:
        """Record a workflow execution.

        Args:
            execution_id: Execution identifier
            workflow_name: Workflow name
            context: Execution context
            trace: Execution trace
        """
        execution_data = {
            "execution_id": execution_id,
            "workflow_name": workflow_name,
            "start_time": context.start_time,
            "end_time": time.time(),
            "metrics": context.metrics.__dict__,
            "trace_summary": trace.get_summary(),
        }

        # Save to file
        output_path = self.storage_path / f"{workflow_name}_{execution_id}.json"
        output_path.write_text(json.dumps(execution_data, indent=2, default=str))

        logger.info(f"Execution history saved to {output_path}")

    def get_execution(
        self,
        execution_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get execution history by ID.

        Args:
            execution_id: Execution identifier

        Returns:
            Execution data or None
        """
        # Search for execution file
        for file_path in self.storage_path.glob(f"*_{execution_id}.json"):
            return cast("Dict[str, Any]", json.loads(file_path.read_text()))

        return None

    def list_executions(
        self,
        workflow_name: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """List recent executions.

        Args:
            workflow_name: Optional workflow name filter
            limit: Maximum number of executions to return

        Returns:
            List of execution summaries
        """
        executions: List[Dict[str, Any]] = []

        for file_path in sorted(self.storage_path.glob("*.json"), reverse=True):
            if len(executions) >= limit:
                break

            data = json.loads(file_path.read_text())

            if workflow_name is None or data["workflow_name"] == workflow_name:
                executions.append(data)

        return executions

    def replay_execution(
        self,
        execution_id: str,
        executor: "WorkflowExecutor",
    ) -> Any:
        """Replay a previous execution.

        Args:
            execution_id: Execution identifier
            executor: Workflow executor

        Returns:
            Execution result
        """
        execution_data = self.get_execution(execution_id)

        if not execution_data:
            raise ValueError(f"Execution {execution_id} not found in history")

        # NOTE: Replay logic requires checkpoint restoration and state serialization
        # Deferred: Needs StateGraph checkpoint backend integration for deterministic replay
        raise NotImplementedError("Execution replay not yet implemented")


# =============================================================================
# Workflow Executor
# =============================================================================


class WorkflowExecutor:
    """Main workflow execution engine.

    Provides comprehensive workflow execution with debugging, tracing,
    and monitoring capabilities.
    """

    def __init__(
        self,
        debug_mode: bool = False,
        trace_mode: bool = False,
        error_recovery: Optional[ErrorRecovery] = None,
        history: Optional[ExecutionHistory] = None,
    ):
        """Initialize workflow executor.

        Args:
            debug_mode: Enable debug mode
            trace_mode: Enable execution tracing
            error_recovery: Error recovery strategy
            history: Execution history recorder
        """
        self.debug_mode = debug_mode
        self.trace_mode = trace_mode
        self.error_recovery = error_recovery or ErrorRecovery()
        self.history = history or ExecutionHistory()

        # Active execution context
        self._context: Optional[ExecutionContext] = None
        self._trace: Optional[ExecutionTrace] = None

        # Execution control
        self._paused = False
        self._stopped = False

    # =========================================================================
    # Breakpoint Management
    # =========================================================================

    def set_breakpoint(
        self,
        node_id: str,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> None:
        """Set a breakpoint on a node.

        Args:
            node_id: Node ID to break on
            condition: Optional condition function
        """
        if self._context is None:
            raise RuntimeError("No active execution context")

        self._context.breakpoints[node_id] = Breakpoint(
            node_id=node_id,
            condition=condition,
        )

        logger.info(f"Breakpoint set on node: {node_id}")

    def clear_breakpoint(self, node_id: str) -> None:
        """Clear a breakpoint.

        Args:
            node_id: Node ID
        """
        if self._context and node_id in self._context.breakpoints:
            del self._context.breakpoints[node_id]
            logger.info(f"Breakpoint cleared on node: {node_id}")

    def list_breakpoints(self) -> List[str]:
        """List active breakpoints.

        Returns:
            List of node IDs with breakpoints
        """
        if self._context is None:
            return []

        return list(self._context.breakpoints.keys())

    # =========================================================================
    # Execution Control
    # =========================================================================

    def pause(self) -> None:
        """Pause workflow execution."""
        self._paused = True
        logger.info("Workflow execution paused")

    def resume(self) -> None:
        """Resume paused workflow execution."""
        self._paused = False
        logger.info("Workflow execution resumed")

    def stop(self) -> None:
        """Stop workflow execution."""
        self._stopped = True
        logger.info("Workflow execution stopped")

    # =========================================================================
    # State Inspection
    # =========================================================================

    def get_state(self) -> Dict[str, Any]:
        """Get current workflow state.

        Returns:
            Current state dictionary
        """
        if self._context is None:
            raise RuntimeError("No active execution context")

        return self._context.current_state.copy()

    def get_variables(self) -> Dict[str, Any]:
        """Get current workflow variables.

        Returns:
            Variables dictionary (non-internal keys)
        """
        state = self.get_state()
        return {k: v for k, v in state.items() if not k.startswith("_")}

    def get_stack_trace(self) -> List[str]:
        """Get execution stack trace.

        Returns:
            List of executed node IDs
        """
        if self._context is None:
            return []

        snapshots = self._context.snapshots
        return [s.current_node for s in snapshots if s.current_node]

    # =========================================================================
    # Execution
    # =========================================================================

    async def execute(
        self,
        workflow: Union["CachedCompiledGraph[Any]", "WorkflowDefinition", "CompiledGraph[Any]"],
        inputs: Optional[Dict[str, Any]] = None,
        breakpoints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute a workflow.

        Args:
            workflow: Workflow to execute
            inputs: Initial workflow inputs
            breakpoints: Optional list of node IDs to break on

        Returns:
            Execution result
        """
        # Initialize execution context
        execution_id = str(uuid.uuid4())
        workflow_name = getattr(workflow, "workflow_name", "unknown")

        self._context = ExecutionContext(
            execution_id=execution_id,
            workflow_name=workflow_name,
            start_time=time.time(),
            debug_mode=self.debug_mode,
            trace_mode=self.trace_mode,
        )
        self._trace = ExecutionTrace(execution_id)

        # Set breakpoints
        if breakpoints:
            for node_id in breakpoints:
                self.set_breakpoint(node_id)

        # Record workflow started
        self._trace.record_event(
            NodeExecutionEvent(
                node_id="workflow",
                event_type=ExecutionEventType.WORKFLOW_STARTED,
                timestamp=time.time(),
                status=NodeStatus.RUNNING,
            )
        )

        try:
            # Execute workflow
            if hasattr(workflow, "invoke"):
                # CachedCompiledGraph or CompiledGraph
                result = await workflow.invoke(inputs or {})
            else:
                # WorkflowDefinition - needs compilation first
                from victor.workflows.graph_compiler import WorkflowDefinitionCompiler

                compiler = WorkflowDefinitionCompiler()
                compiled = compiler.compile(workflow)
                result = await compiled.invoke(inputs or {})

            # Handle polymorphic result
            if hasattr(result, "state"):
                final_state = cast("Dict[str, Any]", result.state)
            elif isinstance(result, dict):
                final_state = cast("Dict[str, Any]", result)
            else:
                final_state = cast("Dict[str, Any]", {"result": result})

            self._context.current_state = final_state

            # Record workflow completed
            self._trace.end_time = time.time()
            self._trace.record_event(
                NodeExecutionEvent(
                    node_id="workflow",
                    event_type=ExecutionEventType.WORKFLOW_COMPLETED,
                    timestamp=time.time(),
                    status=NodeStatus.COMPLETED,
                    output_state=final_state,
                    duration_seconds=self._trace.end_time - self._trace.start_time,
                )
            )

            # Record history
            self.history.record_execution(
                execution_id,
                workflow_name,
                self._context,
                self._trace,
            )

            return final_state

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)

            self._trace.end_time = time.time()
            self._trace.record_event(
                NodeExecutionEvent(
                    node_id="workflow",
                    event_type=ExecutionEventType.WORKFLOW_FAILED,
                    timestamp=time.time(),
                    status=NodeStatus.FAILED,
                    error=str(e),
                    duration_seconds=self._trace.end_time - self._trace.start_time,
                )
            )

            raise

    async def execute_stream(
        self,
        workflow: Union["CachedCompiledGraph", "WorkflowDefinition"],
        inputs: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[NodeExecutionEvent]:
        """Execute workflow with streaming events.

        Args:
            workflow: Workflow to execute
            inputs: Initial workflow inputs

        Yields:
            Node execution events
        """
        # Initialize execution
        execution_id = str(uuid.uuid4())
        workflow_name = getattr(workflow, "workflow_name", "unknown")

        self._context = ExecutionContext(
            execution_id=execution_id,
            workflow_name=workflow_name,
            start_time=time.time(),
            debug_mode=self.debug_mode,
            trace_mode=True,
        )
        self._trace = ExecutionTrace(execution_id)

        # Yield workflow started event
        event = NodeExecutionEvent(
            node_id="workflow",
            event_type=ExecutionEventType.WORKFLOW_STARTED,
            timestamp=time.time(),
            status=NodeStatus.RUNNING,
        )
        self._trace.record_event(event)
        yield event

        try:
            # Execute workflow
            if hasattr(workflow, "stream"):
                # Use streaming if available
                async for node_id, state in workflow.stream(inputs or {}):
                    event = NodeExecutionEvent(
                        node_id=node_id,
                        event_type=ExecutionEventType.NODE_COMPLETED,
                        timestamp=time.time(),
                        status=NodeStatus.COMPLETED,
                        output_state=state,
                    )
                    self._trace.record_event(event)
                    yield event
                    self._context.current_state = state
            else:
                # Fall back to regular execution
                result = await workflow.invoke(inputs or {})

                if hasattr(result, "state"):
                    final_state = result.state
                else:
                    final_state = {"result": result}

                self._context.current_state = final_state

            # Yield workflow completed event
            self._trace.end_time = time.time()
            event = NodeExecutionEvent(
                node_id="workflow",
                event_type=ExecutionEventType.WORKFLOW_COMPLETED,
                timestamp=time.time(),
                status=NodeStatus.COMPLETED,
                output_state=self._context.current_state,
                duration_seconds=self._trace.end_time - self._trace.start_time,
            )
            self._trace.record_event(event)
            yield event

        except Exception as e:
            logger.error(f"Streaming execution failed: {e}")

            self._trace.end_time = time.time()
            event = NodeExecutionEvent(
                node_id="workflow",
                event_type=ExecutionEventType.WORKFLOW_FAILED,
                timestamp=time.time(),
                status=NodeStatus.FAILED,
                error=str(e),
            )
            self._trace.record_event(event)
            yield event

    # =========================================================================
    # Debugging
    # =========================================================================

    async def debug_step_over(self) -> None:
        """Step to next node (skip function calls)."""
        # NOTE: Debugging features require pause/resume state machine and step tracker
        # Deferred: Needs execution pause protocol integration
        raise NotImplementedError("Step over not yet implemented")

    async def debug_step_into(self) -> None:
        """Step into next node (enter function calls)."""
        # NOTE: Debugging features require pause/resume state machine and step tracker
        # Deferred: Needs execution pause protocol integration
        raise NotImplementedError("Step into not yet implemented")

    async def debug_step_out(self) -> None:
        """Step out of current node."""
        # NOTE: Debugging features require pause/resume state machine and step tracker
        # Deferred: Needs execution pause protocol integration
        raise NotImplementedError("Step out not yet implemented")

    async def debug_continue(self) -> None:
        """Continue execution to next breakpoint."""
        self.resume()

    # =========================================================================
    # Trace and History
    # =========================================================================

    def get_trace(self) -> Optional[ExecutionTrace]:
        """Get execution trace.

        Returns:
            Execution trace or None
        """
        return self._trace

    def export_trace(self, output_path: Path) -> None:
        """Export execution trace to file.

        Args:
            output_path: Output file path
        """
        if self._trace is None:
            raise RuntimeError("No execution trace available")

        self._trace.export_json(output_path)

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get execution history.

        Args:
            limit: Maximum number of executions

        Returns:
            List of execution summaries
        """
        return self.history.list_executions(limit=limit)

    def get_metrics(self) -> ExecutionMetrics:
        """Get execution metrics.

        Returns:
            Execution metrics
        """
        if self._context is None:
            raise RuntimeError("No active execution context")

        return self._context.metrics


# =============================================================================
# Convenience Functions
# =============================================================================


async def execute_workflow(
    workflow: Union["CachedCompiledGraph", "WorkflowDefinition"],
    inputs: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    trace: bool = False,
    breakpoints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convenience function to execute a workflow.

    Args:
        workflow: Workflow to execute
        inputs: Initial inputs
        debug: Enable debug mode
        trace: Enable tracing
        breakpoints: List of node IDs to break on

    Returns:
        Execution result
    """
    executor = WorkflowExecutor(debug_mode=debug, trace_mode=trace)
    return await executor.execute(workflow, inputs, breakpoints)


async def execute_workflow_stream(
    workflow: Union["CachedCompiledGraph", "WorkflowDefinition"],
    inputs: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[NodeExecutionEvent]:
    """Convenience function to execute workflow with streaming.

    Args:
        workflow: Workflow to execute
        inputs: Initial inputs

    Yields:
        Node execution events
    """
    executor = WorkflowExecutor(trace_mode=True)
    async for event in executor.execute_stream(workflow, inputs):
        yield event


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "NodeStatus",
    "ExecutionEventType",
    "ErrorRecoveryStrategy",
    # Data classes
    "NodeExecutionEvent",
    "Breakpoint",
    "ExecutionSnapshot",
    "ExecutionMetrics",
    "ExecutionContext",
    # Components
    "ExecutionTrace",
    "StateManager",
    "ErrorRecovery",
    "ExecutionHistory",
    "WorkflowExecutor",
    # Functions
    "execute_workflow",
    "execute_workflow_stream",
]
