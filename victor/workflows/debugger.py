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

"""Workflow Debugger - Step-by-Step Debugging with Breakpoints.

This module provides comprehensive debugging capabilities for workflows:

Features:
- Set breakpoints on nodes (conditional and unconditional)
- Step through workflow execution (step over, into, out)
- Inspect state at any point during execution
- View variable values and state changes
- Continue to next breakpoint
- Skip nodes during debugging
- View stack trace
- Interactive debugging session

Usage:
    debugger = WorkflowDebugger(workflow)
    debugger.set_breakpoint("node_1")
    debugger.set_breakpoint("node_3", condition=lambda s: s.get("value") > 10)
    debugger.start(inputs={"task": "fix bug"})
    debugger.step_over()
    state = debugger.get_state()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
    Union,
)

if TYPE_CHECKING:
    from victor.workflows.definition import (
        WorkflowDefinition,
        WorkflowNode,
    )
    from victor.workflows.execution_engine import (
        ExecutionContext,
        ExecutionTrace,
        NodeStatus,
    )
    from victor.workflows.unified_compiler import CachedCompiledGraph

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class DebugAction(Enum):
    """Debug actions."""

    CONTINUE = "continue"
    STEP_OVER = "step_over"
    STEP_INTO = "step_into"
    STEP_OUT = "step_out"
    PAUSE = "pause"
    STOP = "stop"
    SKIP = "skip"


class DebugState(Enum):
    """Debugger state."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    AT_BREAKPOINT = "at_breakpoint"
    ERROR = "error"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BreakpointInfo:
    """Information about a breakpoint.

    Attributes:
        node_id: Node ID
        condition: Optional condition function
        hit_count: Number of times hit
        enabled: Whether enabled
        temp: Whether temporary (delete after first hit)
    """

    node_id: str
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    hit_count: int = 0
    enabled: bool = True
    temp: bool = False


@dataclass
class StackFrame:
    """Stack frame information.

    Attributes:
        node_id: Node ID
        node_type: Node type
        state: State at this frame
        timestamp: Timestamp
    """

    node_id: str
    node_type: str
    state: Dict[str, Any]
    timestamp: float


@dataclass
class DebugSession:
    """Debug session information.

    Attributes:
        session_id: Unique session ID
        workflow_name: Workflow name
        start_time: Session start time
        state: Current debug state
        breakpoints: Active breakpoints
        call_stack: Execution call stack
        current_state: Current workflow state
        events: Debug events
    """

    session_id: str
    workflow_name: str
    start_time: float
    state: DebugState = DebugState.IDLE
    breakpoints: Dict[str, BreakpointInfo] = field(default_factory=dict)
    call_stack: List[StackFrame] = field(default_factory=list)
    current_state: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# Workflow Debugger
# =============================================================================


class WorkflowDebugger:
    """Workflow debugger with breakpoints and step execution.

    Provides interactive debugging capabilities for workflows including
    breakpoints, stepping, state inspection, and stack traces.
    """

    def __init__(
        self,
        workflow: Union["WorkflowDefinition", "CachedCompiledGraph"],
    ):
        """Initialize workflow debugger.

        Args:
            workflow: Workflow to debug
        """
        self.workflow = workflow
        self.workflow_name = getattr(workflow, "name", "unknown")

        # Create debug session
        import uuid

        self.session = DebugSession(
            session_id=str(uuid.uuid4()),
            workflow_name=self.workflow_name,
            start_time=time.time(),
        )

        # Execution context and trace
        self._context: Optional[ExecutionContext] = None
        self._trace: Optional[ExecutionTrace] = None

        # Control flags
        self._step_mode = False
        self._step_action: Optional[DebugAction] = None
        self._paused_node: Optional[str] = None

        logger.info(f"WorkflowDebugger initialized for '{self.workflow_name}'")

    # =========================================================================
    # Breakpoint Management
    # =========================================================================

    def set_breakpoint(
        self,
        node_id: str,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        temporary: bool = False,
    ) -> None:
        """Set a breakpoint on a node.

        Args:
            node_id: Node ID to break on
            condition: Optional condition function (break if True)
            temporary: Whether to delete after first hit
        """
        self.session.breakpoints[node_id] = BreakpointInfo(
            node_id=node_id,
            condition=condition,
            temp=temporary,
        )

        logger.info(
            f"Breakpoint set on node '{node_id}'"
            f"{' (temporary)' if temporary else ''}"
            f"{' with condition' if condition else ''}"
        )

        self._log_event("breakpoint_set", {"node_id": node_id, "temporary": temporary})

    def clear_breakpoint(self, node_id: str) -> None:
        """Clear a breakpoint.

        Args:
            node_id: Node ID
        """
        if node_id in self.session.breakpoints:
            del self.session.breakpoints[node_id]
            logger.info(f"Breakpoint cleared on node '{node_id}'")
            self._log_event("breakpoint_cleared", {"node_id": node_id})

    def clear_all_breakpoints(self) -> None:
        """Clear all breakpoints."""
        count = len(self.session.breakpoints)
        self.session.breakpoints.clear()
        logger.info(f"Cleared {count} breakpoint(s)")
        self._log_event("all_breakpoints_cleared", {"count": count})

    def list_breakpoints(self) -> List[BreakpointInfo]:
        """List all breakpoints.

        Returns:
            List of breakpoint information
        """
        return list(self.session.breakpoints.values())

    def enable_breakpoint(self, node_id: str) -> None:
        """Enable a breakpoint.

        Args:
            node_id: Node ID
        """
        if node_id in self.session.breakpoints:
            self.session.breakpoints[node_id].enabled = True
            logger.info(f"Breakpoint enabled on node '{node_id}'")

    def disable_breakpoint(self, node_id: str) -> None:
        """Disable a breakpoint.

        Args:
            node_id: Node ID
        """
        if node_id in self.session.breakpoints:
            self.session.breakpoints[node_id].enabled = False
            logger.info(f"Breakpoint disabled on node '{node_id}'")

    # =========================================================================
    # Session Control
    # =========================================================================

    async def start(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        stop_on_entry: bool = False,
    ) -> Dict[str, Any]:
        """Start debugging session.

        Args:
            inputs: Initial workflow inputs
            stop_on_entry: Whether to stop before first node

        Returns:
            Initial state
        """
        logger.info(f"Starting debug session '{self.session.session_id}'")
        self.session.state = DebugState.RUNNING

        # Initialize execution context
        from victor.workflows.execution_engine import ExecutionContext

        self._context = ExecutionContext(
            execution_id=self.session.session_id,
            workflow_name=self.workflow_name,
            start_time=time.time(),
            debug_mode=True,
            trace_mode=True,
        )
        self._context.current_state = inputs or {}

        self.session.current_state = inputs or {}

        self._log_event("session_started", {"stop_on_entry": stop_on_entry})

        # Stop on entry if requested
        if stop_on_entry:
            self.session.state = DebugState.PAUSED
            logger.info("Stopped on entry")
            self._log_event("stopped_on_entry", {})

        return self.session.current_state.copy()

    async def stop(self) -> None:
        """Stop debugging session."""
        logger.info(f"Stopping debug session '{self.session.session_id}'")
        self.session.state = DebugState.STOPPED
        self._log_event("session_stopped", {})

    def pause(self) -> None:
        """Pause execution."""
        if self.session.state == DebugState.RUNNING:
            self.session.state = DebugState.PAUSED
            logger.info("Execution paused")
            self._log_event("paused", {})

    def resume(self) -> None:
        """Resume execution."""
        if self.session.state == DebugState.PAUSED:
            self.session.state = DebugState.RUNNING
            logger.info("Execution resumed")
            self._log_event("resumed", {})

    # =========================================================================
    # Step Execution
    # =========================================================================

    async def step_over(self) -> Dict[str, Any]:
        """Step to next node (skip function calls).

        Returns:
            State after step
        """
        if self.session.state not in {DebugState.PAUSED, DebugState.AT_BREAKPOINT}:
            raise RuntimeError(
                f"Cannot step over in state {self.session.state.value}. "
                "Session must be paused or at breakpoint."
            )

        logger.info("Stepping over")
        self._step_action = DebugAction.STEP_OVER
        self._step_mode = True
        self.session.state = DebugState.RUNNING

        self._log_event("step_over", {})

        # Wait for step to complete
        # In real implementation, this would wait for execution to advance
        # For now, return current state
        return self.session.current_state.copy()

    async def step_into(self) -> Dict[str, Any]:
        """Step into next node (enter function calls).

        Returns:
            State after step
        """
        if self.session.state not in {DebugState.PAUSED, DebugState.AT_BREAKPOINT}:
            raise RuntimeError(f"Cannot step into in state {self.session.state.value}")

        logger.info("Stepping into")
        self._step_action = DebugAction.STEP_INTO
        self._step_mode = True
        self.session.state = DebugState.RUNNING

        self._log_event("step_into", {})

        return self.session.current_state.copy()

    async def step_out(self) -> Dict[str, Any]:
        """Step out of current node.

        Returns:
            State after step
        """
        if self.session.state not in {DebugState.PAUSED, DebugState.AT_BREAKPOINT}:
            raise RuntimeError(f"Cannot step out in state {self.session.state.value}")

        logger.info("Stepping out")
        self._step_action = DebugAction.STEP_OUT
        self._step_mode = True
        self.session.state = DebugState.RUNNING

        self._log_event("step_out", {})

        return self.session.current_state.copy()

    async def continue_execution(self) -> Dict[str, Any]:
        """Continue execution to next breakpoint.

        Returns:
            State when breakpoint hit or execution complete
        """
        if self.session.state not in {DebugState.PAUSED, DebugState.AT_BREAKPOINT}:
            raise RuntimeError(f"Cannot continue in state {self.session.state.value}")

        logger.info("Continuing execution")
        self._step_action = DebugAction.CONTINUE
        self._step_mode = False
        self.session.state = DebugState.RUNNING

        self._log_event("continued", {})

        return self.session.current_state.copy()

    async def skip_node(self, node_id: str) -> Dict[str, Any]:
        """Skip execution of a node.

        Args:
            node_id: Node ID to skip

        Returns:
            Current state
        """
        logger.info(f"Skipping node '{node_id}'")
        self._log_event("node_skipped", {"node_id": node_id})

        # In real implementation, this would mark node to skip during execution
        return self.session.current_state.copy()

    # =========================================================================
    # State Inspection
    # =========================================================================

    def get_state(self) -> Dict[str, Any]:
        """Get current workflow state.

        Returns:
            Current state dictionary
        """
        return self.session.current_state.copy()

    def get_variables(self) -> Dict[str, Any]:
        """Get current workflow variables (non-internal).

        Returns:
            Variables dictionary
        """
        return {k: v for k, v in self.session.current_state.items() if not k.startswith("_")}

    def get_variable(self, name: str) -> Any:
        """Get a specific variable value.

        Args:
            name: Variable name

        Returns:
            Variable value or None
        """
        return self.session.current_state.get(name)

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable value (for debugging/testing).

        Args:
            name: Variable name
            value: New value
        """
        self.session.current_state[name] = value
        logger.debug(f"Variable '{name}' set to {value}")
        self._log_event("variable_set", {"name": name, "value": str(value)[:100]})

    # =========================================================================
    # Stack Trace
    # =========================================================================

    def get_stack_trace(self) -> List[StackFrame]:
        """Get execution stack trace.

        Returns:
            List of stack frames
        """
        return self.session.call_stack.copy()

    def get_call_stack(self) -> List[str]:
        """Get call stack as list of node IDs.

        Returns:
            List of node IDs in call order
        """
        return [frame.node_id for frame in self.session.call_stack]

    def get_current_frame(self) -> Optional[StackFrame]:
        """Get current stack frame.

        Returns:
            Current frame or None
        """
        if self.session.call_stack:
            return self.session.call_stack[-1]
        return None

    # =========================================================================
    # Session Info
    # =========================================================================

    def get_session_info(self) -> Dict[str, Any]:
        """Get debug session information.

        Returns:
            Session information dictionary
        """
        return {
            "session_id": self.session.session_id,
            "workflow_name": self.session.workflow_name,
            "state": self.session.state.value,
            "start_time": self.session.start_time,
            "duration_seconds": time.time() - self.session.start_time,
            "breakpoints": [
                {
                    "node_id": bp.node_id,
                    "enabled": bp.enabled,
                    "hit_count": bp.hit_count,
                    "temporary": bp.temp,
                }
                for bp in self.session.breakpoints.values()
            ],
            "stack_depth": len(self.session.call_stack),
        }

    def get_events(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get debug session events.

        Args:
            limit: Optional limit on number of events

        Returns:
            List of events
        """
        events = self.session.events
        if limit:
            return events[-limit:]
        return events.copy()

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _check_breakpoint(
        self,
        node_id: str,
        state: Dict[str, Any],
    ) -> bool:
        """Check if execution should break at node.

        Args:
            node_id: Node ID
            state: Current state

        Returns:
            True if should break
        """
        bp = self.session.breakpoints.get(node_id)
        if not bp or not bp.enabled:
            return False

        # Check condition
        if bp.condition and not bp.condition(state):
            return False

        # Hit breakpoint
        bp.hit_count += 1
        logger.info(f"Breakpoint hit at '{node_id}' (hit #{bp.hit_count})")
        self._log_event("breakpoint_hit", {"node_id": node_id, "hit_count": bp.hit_count})

        # Remove temporary breakpoints
        if bp.temp:
            del self.session.breakpoints[node_id]
            logger.info(f"Temporary breakpoint at '{node_id}' removed")

        return True

    def _push_stack_frame(
        self,
        node_id: str,
        node_type: str,
        state: Dict[str, Any],
    ) -> None:
        """Push frame onto call stack.

        Args:
            node_id: Node ID
            node_type: Node type
            state: Current state
        """
        frame = StackFrame(
            node_id=node_id,
            node_type=node_type,
            state=state.copy(),
            timestamp=time.time(),
        )
        self.session.call_stack.append(frame)
        logger.debug(f"Stack push: {node_id} (depth: {len(self.session.call_stack)})")

    def _pop_stack_frame(self) -> Optional[StackFrame]:
        """Pop frame from call stack.

        Returns:
            Popped frame or None
        """
        if self.session.call_stack:
            frame = self.session.call_stack.pop()
            logger.debug(f"Stack pop: {frame.node_id} (depth: {len(self.session.call_stack)})")
            return frame
        return None

    def _update_state(self, state: Dict[str, Any]) -> None:
        """Update current state.

        Args:
            state: New state
        """
        self.session.current_state = state.copy()

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a debug event.

        Args:
            event_type: Event type
            data: Event data
        """
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data,
        }
        self.session.events.append(event)
        logger.debug(f"Debug event: {event_type}")


# =============================================================================
# Convenience Functions
# =============================================================================


def create_debugger(
    workflow: Union["WorkflowDefinition", "CachedCompiledGraph"],
) -> WorkflowDebugger:
    """Create a workflow debugger.

    Args:
        workflow: Workflow to debug

    Returns:
        WorkflowDebugger instance
    """
    return WorkflowDebugger(workflow)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "DebugAction",
    "DebugState",
    # Data classes
    "BreakpointInfo",
    "StackFrame",
    "DebugSession",
    # Main class
    "WorkflowDebugger",
    # Functions
    "create_debugger",
]
