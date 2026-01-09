# Phase 3.2: Breakpoint and Debugging System Design for StateGraph Workflows

**Status**: Design Phase
**Author**: Claude Code
**Date**: 2025-01-09
**Related**: Phase 3.1 (Team Node Implementation)

---

## Executive Summary

This document designs a **breakpoint and debugging system** for Victor's StateGraph workflows that enables interactive workflow debugging with conditional breakpoints, step execution, and state inspection. The system extends the existing Debug Adapter Protocol (DAP) infrastructure to support workflow-level debugging while maintaining backward compatibility with non-debugging workflows.

**Key Design Goals:**
1. **Non-Invasive**: Zero impact on workflows without debugging enabled
2. **DAP-Aligned**: Follow Debug Adapter Protocol patterns for consistency
3. **Async-First**: Full support for async workflow execution
4. **Integration**: Seamless integration with existing StateGraph, EventBus, and AgentDebugger
5. **Performance**: Minimal overhead when debugging is disabled

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Breakpoint Data Structures](#breakpoint-data-structures)
3. [Execution Control](#execution-control)
4. [State Inspection](#state-inspection)
5. [Debug Protocol](#debug-protocol)
6. [Integration with StateGraph](#integration-with-stategraph)
7. [Implementation Plan](#implementation-plan)
8. [MVP Features](#mvp-features)
9. [Unit Test Strategy](#unit-test-strategy)

---

## 1. System Architecture

### 1.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Debug Client Layer                           │
│  REST API / WebSocket / CLI Commands                           │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              WorkflowDebugManager (Facade)                      │
│  - Session management                                           │
│  - Breakpoint management                                        │
│  - Execution control orchestration                              │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Debug Protocol Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Breakpoint  │  │   Execution  │  │   State      │          │
│  │  Manager     │  │   Controller │  │   Inspector  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              StateGraph Integration Points                      │
│  ┌──────────────────────────────────────────────────┐          │
│  │  CompiledGraph.invoke() with DebugHook          │          │
│  │  - Pre-node breakpoint check                     │          │
│  │  - Post-node breakpoint check                    │          │
│  │  - Conditional breakpoint evaluation             │          │
│  │  - Exception interception                        │          │
│  └──────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                   Existing Infrastructure                       │
│  - EventBus (event emission)                                   │
│  - AgentDebugger (inspection)                                  │
│  - ObservabilityEmitter (events)                               │
│  - WorkflowCheckpoint (persistence)                            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Module Structure

```
victor/framework/debugging/
├── __init__.py                 # Public API exports
├── breakpoints.py              # Breakpoint data structures and manager
├── execution.py                # Pause/resume execution controller
├── inspector.py                # State inspection and diffing
├── protocol.py                 # Debug protocol message types
├── hooks.py                    # DebugHook integration with StateGraph
└── session.py                  # DebugSession lifecycle management

victor/framework/
├── graph.py                    # Modified: Add DebugHook injection point
└── config.py                   # Extended: Add DebugConfig

tests/unit/framework/debugging/
├── test_breakpoints.py
├── test_execution.py
├── test_inspector.py
├── test_hooks.py
└── test_session.py
```

---

## 2. Breakpoint Data Structures

### 2.1 Breakpoint Types

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union
import asyncio

class BreakpointType(Enum):
    """Types of breakpoints for workflow debugging."""
    NODE = "node"                    # Pause before/after specific node
    CONDITIONAL = "conditional"      # Pause when condition is true
    EXCEPTION = "exception"          # Pause on any error
    STATE = "state"                  # Pause when state key matches value


class BreakpointPosition(Enum):
    """Position relative to node execution."""
    BEFORE = "before"                # Pause before node executes
    AFTER = "after"                  # Pause after node executes
    ON_ERROR = "on_error"            # Pause if node raises exception


@dataclass
class WorkflowBreakpoint:
    """A breakpoint in workflow execution.

    Attributes:
        id: Unique breakpoint identifier (UUID)
        type: Type of breakpoint (NODE, CONDITIONAL, EXCEPTION, STATE)
        position: When to pause (BEFORE, AFTER, ON_ERROR)
        node_id: Target node ID (for NODE type)
        condition: Optional condition function (for CONDITIONAL type)
        state_key: State key to watch (for STATE type)
        state_value: Expected state value (for STATE type)
        enabled: Whether breakpoint is active
        hit_count: Number of times breakpoint was hit
        ignore_count: Skip first N hits (default: 0)
        log_message: Optional log message instead of pausing
        metadata: Additional breakpoint metadata

    Example:
        # Node breakpoint
        bp = WorkflowBreakpoint(
            type=BreakpointType.NODE,
            position=BreakpointPosition.BEFORE,
            node_id="analyze_code"
        )

        # Conditional breakpoint
        bp = WorkflowBreakpoint(
            type=BreakpointType.CONDITIONAL,
            position=BreakpointPosition.AFTER,
            node_id="process_data",
            condition=lambda state: state.get("error_count", 0) > 5
        )

        # Exception breakpoint
        bp = WorkflowBreakpoint(
            type=BreakpointType.EXCEPTION,
            position=BreakpointPosition.ON_ERROR
        )
    """
    id: str
    type: BreakpointType
    position: BreakpointPosition
    node_id: Optional[str] = None
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    state_key: Optional[str] = None
    state_value: Any = None
    enabled: bool = True
    hit_count: int = 0
    ignore_count: int = 0
    log_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_hit(self, state: Dict[str, Any], node_id: str, error: Optional[Exception] = None) -> bool:
        """Check if breakpoint should be hit.

        Args:
            state: Current workflow state
            node_id: Current node ID
            error: Exception if one occurred

        Returns:
            True if breakpoint should trigger pause
        """
        if not self.enabled:
            return False

        # Check ignore count
        if self.hit_count < self.ignore_count:
            self.hit_count += 1
            return False

        # Node breakpoint
        if self.type == BreakpointType.NODE:
            return self.node_id == node_id

        # Conditional breakpoint
        if self.type == BreakpointType.CONDITIONAL:
            if self.node_id and self.node_id != node_id:
                return False
            if self.condition:
                return self.condition(state)
            return True

        # Exception breakpoint
        if self.type == BreakpointType.EXCEPTION:
            return error is not None

        # State breakpoint
        if self.type == BreakpointType.STATE:
            if self.state_key:
                current_value = state.get(self.state_key)
                return current_value == self.state_value

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "position": self.position.value,
            "node_id": self.node_id,
            "state_key": self.state_key,
            "state_value": str(self.state_value) if self.state_value else None,
            "enabled": self.enabled,
            "hit_count": self.hit_count,
            "ignore_count": self.ignore_count,
            "log_message": self.log_message,
            "metadata": self.metadata,
        }
```

### 2.2 Breakpoint Storage

```python
@dataclass
class BreakpointStorage:
    """Storage for active breakpoints.

    Supports both in-memory and persistent storage backends.
    """
    _breakpoints: Dict[str, WorkflowBreakpoint] = field(default_factory=dict)
    _node_index: Dict[str, List[str]] = field(default_factory=dict)  # node_id -> [bp_ids]
    _persist_enabled: bool = False
    _persist_path: Optional[str] = None

    def add(self, breakpoint: WorkflowBreakpoint) -> None:
        """Add a breakpoint."""
        self._breakpoints[breakpoint.id] = breakpoint
        if breakpoint.node_id:
            if breakpoint.node_id not in self._node_index:
                self._node_index[breakpoint.node_id] = []
            self._node_index[breakpoint.node_id].append(breakpoint.id)

    def remove(self, breakpoint_id: str) -> Optional[WorkflowBreakpoint]:
        """Remove a breakpoint by ID."""
        bp = self._breakpoints.pop(breakpoint_id, None)
        if bp and bp.node_id:
            self._node_index.setdefault(bp.node_id, []).remove(breakpoint_id)
        return bp

    def get(self, breakpoint_id: str) -> Optional[WorkflowBreakpoint]:
        """Get breakpoint by ID."""
        return self._breakpoints.get(breakpoint_id)

    def list_all(self) -> List[WorkflowBreakpoint]:
        """List all breakpoints."""
        return list(self._breakpoints.values())

    def get_for_node(self, node_id: str) -> List[WorkflowBreakpoint]:
        """Get all breakpoints for a specific node."""
        bp_ids = self._node_index.get(node_id, [])
        return [self._breakpoints[bp_id] for bp_id in bp_ids if bp_id in self._breakpoints]

    def clear(self) -> None:
        """Clear all breakpoints."""
        self._breakpoints.clear()
        self._node_index.clear()

    async def persist(self) -> None:
        """Persist breakpoints to disk if enabled."""
        if not self._persist_enabled or not self._persist_path:
            return

        import json
        from pathlib import Path

        data = {
            "breakpoints": [
                {**bp.to_dict(), "condition": None}  # Can't serialize functions
                for bp in self._breakpoints.values()
            ]
        }

        Path(self._persist_path).write_text(json.dumps(data, indent=2))

    async def load(self) -> None:
        """Load persisted breakpoints from disk."""
        if not self._persist_enabled or not self._persist_path:
            return

        import json
        from pathlib import Path

        path = Path(self._persist_path)
        if not path.exists():
            return

        data = json.loads(path.read_text())
        # Reconstruct breakpoints (conditions are lost on persistence)
        for bp_data in data.get("breakpoints", []):
            bp = WorkflowBreakpoint(
                id=bp_data["id"],
                type=BreakpointType(bp_data["type"]),
                position=BreakpointPosition(bp_data["position"]),
                node_id=bp_data.get("node_id"),
                enabled=bp_data.get("enabled", True),
                ignore_count=bp_data.get("ignore_count", 0),
                log_message=bp_data.get("log_message"),
                metadata=bp_data.get("metadata", {}),
            )
            self.add(bp)
```

### 2.3 Breakpoint Manager API

```python
class BreakpointManager:
    """Manages workflow breakpoints (SRP: Single Responsibility).

    Provides CRUD operations for breakpoints and evaluation logic
    for checking if breakpoints should be hit during execution.

    Attributes:
        storage: BreakpointStorage instance
        event_bus: EventBus for emitting breakpoint events

    Example:
        manager = BreakpointManager(event_bus)

        # Add node breakpoint
        bp = manager.set_breakpoint(
            node_id="analyze",
            position=BreakpointPosition.BEFORE
        )

        # Add conditional breakpoint
        bp = manager.set_breakpoint(
            node_id="process",
            position=BreakpointPosition.AFTER,
            condition=lambda state: state.get("errors", 0) > 0
        )

        # Clear breakpoint
        manager.clear_breakpoint(bp.id)
    """

    def __init__(self, event_bus: Any):
        """Initialize breakpoint manager.

        Args:
            event_bus: EventBus instance for emitting events
        """
        from victor.framework.debugging.breakpoints import BreakpointStorage

        self.storage = BreakpointStorage()
        self._event_bus = event_bus

    def set_breakpoint(
        self,
        node_id: Optional[str] = None,
        position: BreakpointPosition = BreakpointPosition.BEFORE,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        state_key: Optional[str] = None,
        state_value: Any = None,
        bp_type: BreakpointType = BreakpointType.NODE,
        ignore_count: int = 0,
        log_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowBreakpoint:
        """Set a new breakpoint.

        Args:
            node_id: Target node ID (required for NODE type)
            position: When to pause (BEFORE, AFTER, ON_ERROR)
            condition: Optional condition function
            state_key: State key to watch (for STATE type)
            state_value: Expected state value
            bp_type: Type of breakpoint
            ignore_count: Skip first N hits
            log_message: Optional log message instead of pausing
            metadata: Additional metadata

        Returns:
            Created WorkflowBreakpoint

        Raises:
            ValueError: If parameters are invalid
        """
        import uuid

        if bp_type == BreakpointType.NODE and not node_id:
            raise ValueError("node_id required for NODE breakpoints")

        bp = WorkflowBreakpoint(
            id=uuid.uuid4().hex,
            type=bp_type,
            position=position,
            node_id=node_id,
            condition=condition,
            state_key=state_key,
            state_value=state_value,
            ignore_count=ignore_count,
            log_message=log_message,
            metadata=metadata or {},
        )

        self.storage.add(bp)

        # Emit event
        self._emit_breakpoint_set(bp)

        return bp

    def clear_breakpoint(self, breakpoint_id: str) -> bool:
        """Clear a breakpoint by ID.

        Args:
            breakpoint_id: Breakpoint ID to clear

        Returns:
            True if breakpoint was found and cleared
        """
        bp = self.storage.remove(breakpoint_id)
        if bp:
            self._emit_breakpoint_cleared(bp)
            return True
        return False

    def list_breakpoints(
        self, node_id: Optional[str] = None, enabled_only: bool = False
    ) -> List[WorkflowBreakpoint]:
        """List breakpoints.

        Args:
            node_id: Optional filter by node ID
            enabled_only: Only return enabled breakpoints

        Returns:
            List of breakpoints matching filters
        """
        if node_id:
            bps = self.storage.get_for_node(node_id)
        else:
            bps = self.storage.list_all()

        if enabled_only:
            bps = [bp for bp in bps if bp.enabled]

        return bps

    def enable_breakpoint(self, breakpoint_id: str) -> bool:
        """Enable a breakpoint.

        Args:
            breakpoint_id: Breakpoint ID to enable

        Returns:
            True if breakpoint was found and enabled
        """
        bp = self.storage.get(breakpoint_id)
        if bp:
            bp.enabled = True
            return True
        return False

    def disable_breakpoint(self, breakpoint_id: str) -> bool:
        """Disable a breakpoint.

        Args:
            breakpoint_id: Breakpoint ID to disable

        Returns:
            True if breakpoint was found and disabled
        """
        bp = self.storage.get(breakpoint_id)
        if bp:
            bp.enabled = False
            return True
        return False

    def evaluate_breakpoints(
        self,
        state: Dict[str, Any],
        node_id: str,
        position: BreakpointPosition,
        error: Optional[Exception] = None,
    ) -> List[WorkflowBreakpoint]:
        """Evaluate which breakpoints should be hit.

        Called by DebugHook during workflow execution to check if
        any breakpoints should trigger a pause.

        Args:
            state: Current workflow state
            node_id: Current node ID
            position: Current position (BEFORE/AFTER/ON_ERROR)
            error: Exception if one occurred

        Returns:
            List of breakpoints that should be hit
        """
        # Get node-specific breakpoints
        node_bps = self.storage.get_for_node(node_id)

        # Get exception breakpoints
        exception_bps = [
            bp for bp in self.storage.list_all()
            if bp.type == BreakpointType.EXCEPTION
        ]

        # Combine and filter
        all_bps = node_bps + exception_bps

        hit_bps = []
        for bp in all_bps:
            # Check position match
            if bp.position != position:
                continue

            # Check if breakpoint should hit
            if bp.should_hit(state, node_id, error):
                bp.hit_count += 1
                hit_bps.append(bp)

                # Emit event
                self._emit_breakpoint_hit(bp, state, node_id)

                # Log message if set
                if bp.log_message:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Breakpoint log: {bp.log_message}")

        return hit_bps

    def _emit_breakpoint_set(self, bp: WorkflowBreakpoint) -> None:
        """Emit breakpoint_set event."""
        try:
            self._event_bus.emit_lifecycle_event(
                "breakpoint_set",
                {"breakpoint_id": bp.id, "type": bp.type.value, "node_id": bp.node_id}
            )
        except Exception:
            pass  # Event emission failures shouldn't break debugging

    def _emit_breakpoint_cleared(self, bp: WorkflowBreakpoint) -> None:
        """Emit breakpoint_cleared event."""
        try:
            self._event_bus.emit_lifecycle_event(
                "breakpoint_cleared",
                {"breakpoint_id": bp.id, "type": bp.type.value}
            )
        except Exception:
            pass

    def _emit_breakpoint_hit(
        self, bp: WorkflowBreakpoint, state: Dict[str, Any], node_id: str
    ) -> None:
        """Emit breakpoint_hit event."""
        try:
            self._event_bus.emit_lifecycle_event(
                "breakpoint_hit",
                {
                    "breakpoint_id": bp.id,
                    "node_id": node_id,
                    "hit_count": bp.hit_count,
                    "state_keys": list(state.keys()),
                }
            )
        except Exception:
            pass
```

---

## 3. Execution Control

### 3.1 Pause/Resume Mechanism

```python
import asyncio
from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

class ExecutionState(Enum):
    """State of debug execution."""
    RUNNING = "running"          # Normal execution
    PAUSED = "paused"            # Paused at breakpoint
    STEPPING = "stepping"        # Stepping through execution
    TERMINATED = "terminated"    # Execution finished


class StepMode(Enum):
    """Stepping modes."""
    STEP_OVER = "step_over"      # Execute next node, don't enter sub-workflows
    STEP_INTO = "step_into"      # Next node, enter sub-workflows
    STEP_OUT = "step_out"        # Complete current sub-workflow


@dataclass
class PauseContext:
    """Context when execution is paused.

    Captures the state at breakpoint for inspection and resumption.
    """
    session_id: str
    node_id: str
    position: BreakpointPosition
    state: Dict[str, Any]
    breakpoint_ids: List[str]
    timestamp: float
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "node_id": self.node_id,
            "position": self.position.value,
            "state_keys": list(self.state.keys()),
            "breakpoint_ids": self.breakpoint_ids,
            "timestamp": self.timestamp,
            "error": str(self.error) if self.error else None,
            "metadata": self.metadata,
        }


class ExecutionController:
    """Controls execution pause/resume for debugging (SRP).

    Manages the async coordination between workflow execution
    and debug commands (continue, step, pause).

    Thread Safety:
        This class uses asyncio primitives and is NOT thread-safe.
        Each debug session must have its own ExecutionController.

    Attributes:
        session_id: Unique debug session identifier
        state: Current execution state
        pause_context: Context when paused
        step_mode: Current stepping mode
        resume_event: Event for signaling resume

    Example:
        controller = ExecutionController(session_id="debug-123")

        # In workflow execution (DebugHook)
        if controller.should_pause(node_id, state):
            context = await controller.pause(node_id, state)
            # Execution waits here until resume

        # From debug client
        await controller.continue_execution()
    """

    def __init__(self, session_id: str):
        """Initialize execution controller.

        Args:
            session_id: Unique debug session ID
        """
        self.session_id = session_id
        self._state = ExecutionState.RUNNING
        self._pause_context: Optional[PauseContext] = None
        self._step_mode: Optional[StepMode] = None
        self._resume_event = asyncio.Event()
        self._pause_lock = asyncio.Lock()

    @property
    def state(self) -> ExecutionState:
        """Get current execution state."""
        return self._state

    @property
    def pause_context(self) -> Optional[PauseContext]:
        """Get current pause context."""
        return self._pause_context

    def should_pause(
        self,
        node_id: str,
        state: Dict[str, Any],
        breakpoints: List[WorkflowBreakpoint],
    ) -> bool:
        """Check if execution should pause.

        Called by DebugHook before/after node execution.

        Args:
            node_id: Current node ID
            state: Current workflow state
            breakpoints: Breakpoints that were hit

        Returns:
            True if execution should pause
        """
        # Pause if breakpoints hit
        if breakpoints:
            return True

        # Pause if in stepping mode
        if self._step_mode:
            return True

        # Pause if manually requested
        if self._state == ExecutionState.PAUSED:
            return True

        return False

    async def pause(
        self,
        node_id: str,
        state: Dict[str, Any],
        position: BreakpointPosition,
        breakpoints: List[WorkflowBreakpoint],
        error: Optional[Exception] = None,
    ) -> PauseContext:
        """Pause execution at breakpoint.

        Blocks until resume is called.

        Args:
            node_id: Current node ID
            state: Current workflow state
            position: Position relative to node
            breakpoints: Breakpoints that were hit
            error: Exception if one occurred

        Returns:
            PauseContext with captured state

        Thread Safety:
            Acquires pause lock to ensure only one pause at a time.
        """
        async with self._pause_lock:
            # Create pause context
            import time
            context = PauseContext(
                session_id=self.session_id,
                node_id=node_id,
                position=position,
                state=state,
                breakpoint_ids=[bp.id for bp in breakpoints],
                timestamp=time.time(),
                error=error,
            )

            self._pause_context = context
            self._state = ExecutionState.PAUSED

            # Clear resume event (wait for resume signal)
            self._resume_event.clear()

            # Wait for resume (blocks execution)
            await self._resume_event.wait()

            # Clear pause context after resume
            self._pause_context = None

            return context

    async def continue_execution(self) -> None:
        """Continue execution after pause.

        Signals the paused workflow to resume.
        """
        if self._state != ExecutionState.PAUSED:
            return

        self._step_mode = None
        self._state = ExecutionState.RUNNING
        self._resume_event.set()

    async def step_over(self) -> None:
        """Step to next node (don't enter sub-workflows)."""
        if self._state != ExecutionState.PAUSED:
            return

        self._step_mode = StepMode.STEP_OVER
        self._state = ExecutionState.STEPPING
        self._resume_event.set()

    async def step_into(self) -> None:
        """Step into next node (enter sub-workflows)."""
        if self._state != ExecutionState.PAUSED:
            return

        self._step_mode = StepMode.STEP_INTO
        self._state = ExecutionState.STEPPING
        self._resume_event.set()

    async def step_out(self) -> None:
        """Step out of current sub-workflow."""
        if self._state != ExecutionState.PAUSED:
            return

        self._step_mode = StepMode.STEP_OUT
        self._state = ExecutionState.STEPPING
        self._resume_event.set()

    async def pause_immediately(self) -> None:
        """Request immediate pause (user-requested pause).

        Sets state to PAUSED, which will be checked on next node.
        """
        self._state = ExecutionState.PAUSED

    def terminate(self) -> None:
        """Terminate execution (called on workflow completion)."""
        self._state = ExecutionState.TERMINATED
        self._resume_event.set()  # Release any waiting pause
```

### 3.2 Debug Session Lifecycle

```python
@dataclass
class DebugSessionConfig:
    """Configuration for debug session."""
    session_id: str
    workflow_id: str
    enable_breakpoints: bool = True
    enable_exceptions: bool = True
    session_timeout: float = 3600.0  # 1 hour default
    auto_resume: bool = False  # Auto-resume on timeout


class DebugSession:
    """Manages a debug session lifecycle.

    Coordinates breakpoint manager, execution controller, and
    state inspector for a single debugging session.

    Attributes:
        config: Session configuration
        breakpoint_mgr: BreakpointManager instance
        execution_ctrl: ExecutionController instance
        inspector: StateInspector instance
        created_at: Session creation timestamp

    Example:
        session = DebugSession(
            config=DebugSessionConfig(
                session_id="debug-123",
                workflow_id="code_review"
            )
        )

        # Set breakpoints
        session.set_breakpoint(node_id="analyze", position=BreakpointPosition.BEFORE)

        # Attach to workflow execution
        result = await workflow.invoke(
            input_state,
            debug_hook=session.create_hook()
        )

        # Control execution
        await session.continue()

        # Inspect state
        state = session.get_current_state()
    """

    def __init__(self, config: DebugSessionConfig, event_bus: Any):
        """Initialize debug session.

        Args:
            config: Session configuration
            event_bus: EventBus for events
        """
        self.config = config
        self._event_bus = event_bus
        self.created_at = asyncio.get_event_loop().time()

        # Create components
        from victor.framework.debugging.breakpoints import BreakpointManager
        from victor.framework.debugging.execution import ExecutionController
        from victor.framework.debugging.inspector import StateInspector

        self.breakpoint_mgr = BreakpointManager(event_bus)
        self.execution_ctrl = ExecutionController(config.session_id)
        self.inspector = StateInspector()

        # Session state
        self._active = True
        self._hook: Optional["DebugHook"] = None

    def create_hook(self) -> "DebugHook":
        """Create DebugHook for workflow execution.

        Returns:
            DebugHook configured for this session
        """
        from victor.framework.debugging.hooks import DebugHook

        self._hook = DebugHook(
            session_id=self.config.session_id,
            breakpoint_mgr=self.breakpoint_mgr,
            execution_ctrl=self.execution_ctrl,
            inspector=self.inspector,
            event_bus=self._event_bus,
        )
        return self._hook

    async def continue_execution(self) -> None:
        """Continue execution."""
        await self.execution_ctrl.continue_execution()

    async def step_over(self) -> None:
        """Step over next node."""
        await self.execution_ctrl.step_over()

    async def step_into(self) -> None:
        """Step into next node."""
        await self.execution_ctrl.step_into()

    async def step_out(self) -> None:
        """Step out of current sub-workflow."""
        await self.execution_ctrl.step_out()

    async def pause(self) -> None:
        """Request immediate pause."""
        await self.execution_ctrl.pause_immediately()

    def get_current_state(self) -> Optional[Dict[str, Any]]:
        """Get current workflow state at pause."""
        context = self.execution_ctrl.pause_context
        return context.state if context else None

    def get_pause_context(self) -> Optional[PauseContext]:
        """Get current pause context."""
        return self.execution_ctrl.pause_context

    def set_breakpoint(self, **kwargs) -> WorkflowBreakpoint:
        """Set a breakpoint (delegates to manager)."""
        return self.breakpoint_mgr.set_breakpoint(**kwargs)

    def clear_breakpoint(self, breakpoint_id: str) -> bool:
        """Clear a breakpoint (delegates to manager)."""
        return self.breakpoint_mgr.clear_breakpoint(breakpoint_id)

    def list_breakpoints(self, **kwargs) -> List[WorkflowBreakpoint]:
        """List breakpoints (delegates to manager)."""
        return self.breakpoint_mgr.list_breakpoints(**kwargs)

    async def stop(self) -> None:
        """Stop debug session."""
        self._active = False
        self.execution_ctrl.terminate()

        # Persist breakpoints if enabled
        await self.breakpoint_mgr.storage.persist()

    @property
    def is_active(self) -> bool:
        """Check if session is active."""
        return self._active

    @property
    def is_paused(self) -> bool:
        """Check if execution is paused."""
        return self.execution_ctrl.state == ExecutionState.PAUSED
```

---

## 4. State Inspection

### 4.1 State Snapshot Format

```python
@dataclass
class StateSnapshot:
    """Snapshot of workflow state at a point in time.

    Captures state for inspection and diffing.

    Attributes:
        timestamp: When snapshot was taken
        node_id: Current node ID
        state: Full state dictionary
        state_summary: Summary with keys and types
        size_bytes: Approximate size in memory
        metadata: Additional metadata
    """
    timestamp: float
    node_id: str
    state: Dict[str, Any]
    state_summary: Dict[str, str]  # key -> type
    size_bytes: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(cls, state: Dict[str, Any], node_id: str) -> "StateSnapshot":
        """Capture a state snapshot.

        Args:
            state: Current workflow state
            node_id: Current node ID

        Returns:
            StateSnapshot instance
        """
        import sys
        import time

        # Calculate approximate size
        size_bytes = sum(
            len(k) + len(str(v)) if not isinstance(v, (dict, list)) else 0
            for k, v in state.items()
        )

        # Create summary
        summary = {k: type(v).__name__ for k, v in state.items()}

        return cls(
            timestamp=time.time(),
            node_id=node_id,
            state=state.copy(),
            state_summary=summary,
            size_bytes=size_bytes,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "state_summary": self.state_summary,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
        }
```

### 4.2 State Diffing

```python
@dataclass
class StateDiff:
    """Difference between two state snapshots.

    Attributes:
        before_key: Keys present in before state
        after_key: Keys present in after state
        added_keys: Keys added in after
        removed_keys: Keys removed in after
        changed_keys: Keys with changed values
        unchanged_keys: Unchanged keys
    """
    before_keys: set
    after_keys: set
    added_keys: set
    removed_keys: set
    changed_keys: Dict[str, tuple[Any, Any]]  # key -> (old, new)
    unchanged_keys: set

    @classmethod
    def compare(cls, before: Dict[str, Any], after: Dict[str, Any]) -> "StateDiff":
        """Compare two state dictionaries.

        Args:
            before: State before
            after: State after

        Returns:
            StateDiff with differences
        """
        before_keys = set(before.keys())
        after_keys = set(after.keys())

        added_keys = after_keys - before_keys
        removed_keys = before_keys - after_keys
        common_keys = before_keys & after_keys

        changed_keys = {}
        unchanged_keys = set()

        for key in common_keys:
            if before[key] != after[key]:
                changed_keys[key] = (before[key], after[key])
            else:
                unchanged_keys.add(key)

        return cls(
            before_keys=before_keys,
            after_keys=after_keys,
            added_keys=added_keys,
            removed_keys=removed_keys,
            changed_keys=changed_keys,
            unchanged_keys=unchanged_keys,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "added_keys": list(self.added_keys),
            "removed_keys": list(self.removed_keys),
            "changed_keys": [
                {"key": k, "old": v[0], "new": v[1]}
                for k, v in self.changed_keys.items()
            ],
            "unchanged_count": len(self.unchanged_keys),
        }

    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return bool(self.added_keys or self.removed_keys or self.changed_keys)
```

### 4.3 State Inspector API

```python
class StateInspector:
    """Inspects workflow state for debugging (SRP).

    Provides state snapshot, diffing, and query capabilities.

    Attributes:
        _snapshots: History of state snapshots
        _max_snapshots: Maximum snapshots to keep

    Example:
        inspector = StateInspector()

        # Capture snapshot
        snapshot = inspector.capture_snapshot(state, "analyze")

        # Compare states
        diff = inspector.compare_states(before_state, after_state)

        # Query state
        value = inspector.get_value(state, "user.name")
    """

    def __init__(self, max_snapshots: int = 100):
        """Initialize state inspector.

        Args:
            max_snapshots: Maximum snapshots to keep in memory
        """
        self._snapshots: List[StateSnapshot] = []
        self._max_snapshots = max_snapshots

    def capture_snapshot(self, state: Dict[str, Any], node_id: str) -> StateSnapshot:
        """Capture a state snapshot.

        Args:
            state: Current workflow state
            node_id: Current node ID

        Returns:
            StateSnapshot instance
        """
        snapshot = StateSnapshot.capture(state, node_id)

        self._snapshots.append(snapshot)

        # Limit history
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots.pop(0)

        return snapshot

    def compare_states(
        self, before: Dict[str, Any], after: Dict[str, Any]
    ) -> StateDiff:
        """Compare two state dictionaries.

        Args:
            before: State before
            after: State after

        Returns:
            StateDiff with differences
        """
        return StateDiff.compare(before, after)

    def get_value(self, state: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """Get a value from state by key path.

        Supports nested key paths with dot notation:
            "user.profile.name" -> state["user"]["profile"]["name"]

        Args:
            state: State dictionary
            key_path: Dot-separated key path
            default: Default value if key not found

        Returns:
            Value at key path or default
        """
        keys = key_path.split(".")
        value = state

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, str]:
        """Get summary of state keys and types.

        Args:
            state: State dictionary

        Returns:
            Dictionary mapping keys to type names
        """
        return {k: type(v).__name__ for k, v in state.items()}

    def get_snapshots(self, limit: Optional[int] = None) -> List[StateSnapshot]:
        """Get state snapshots.

        Args:
            limit: Optional limit on number of snapshots

        Returns:
            List of snapshots
        """
        if limit:
            return self._snapshots[-limit:]
        return self._snapshots.copy()

    def get_snapshot_history(
        self, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get snapshot history as dictionaries.

        Args:
            session_id: Optional session ID filter

        Returns:
            List of snapshot dictionaries
        """
        return [s.to_dict() for s in self._snapshots]

    def clear_snapshots(self) -> None:
        """Clear all snapshots."""
        self._snapshots.clear()

    def get_large_state_keys(
        self, state: Dict[str, Any], threshold_bytes: int = 1024
    ) -> List[str]:
        """Find state keys with large values.

        Args:
            state: State dictionary
            threshold_bytes: Size threshold in bytes

        Returns:
            List of keys exceeding threshold
        """
        large_keys = []

        for key, value in state.items():
            size = len(str(value)) if not isinstance(value, (dict, list)) else 0
            if size > threshold_bytes:
                large_keys.append(key)

        return large_keys
```

---

## 5. Debug Protocol

### 5.1 Protocol Message Types

```python
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

class DebugMessageType(Enum):
    """Types of debug protocol messages."""
    # Client -> Server (Commands)
    SET_BREAKPOINTS = "set_breakpoints"
    CLEAR_BREAKPOINTS = "clear_breakpoints"
    CONTINUE = "continue"
    STEP_OVER = "step_over"
    STEP_INTO = "step_into"
    STEP_OUT = "step_out"
    PAUSE = "pause"
    INSPECT_STATE = "inspect_state"
    GET_DIFF = "get_diff"

    # Server -> Client (Events)
    BREAKPOINT_HIT = "breakpoint_hit"
    STATE_UPDATE = "state_update"
    EXCEPTION = "exception"
    PAUSED = "paused"
    RESUMED = "resumed"
    COMPLETED = "completed"


@dataclass
class DebugMessage:
    """Debug protocol message.

    Messages are JSON-serializable and can be sent over
    WebSocket, REST, or other transport.

    Attributes:
        type: Message type
        session_id: Debug session ID
        data: Message payload (varies by type)
        timestamp: Message timestamp
        request_id: Optional correlation ID for request-response

    Example:
        # Client command
        msg = DebugMessage(
            type=DebugMessageType.SET_BREAKPOINTS,
            session_id="debug-123",
            data={"node_id": "analyze", "position": "before"}
        )

        # Server event
        msg = DebugMessage(
            type=DebugMessageType.BREAKPOINT_HIT,
            session_id="debug-123",
            data={"node_id": "analyze", "state": {...}}
        )
    """
    type: DebugMessageType
    session_id: str
    data: Dict[str, Any]
    timestamp: float
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "session_id": self.session_id,
            "data": self.data,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DebugMessage":
        """Create from dictionary."""
        return cls(
            type=DebugMessageType(data["type"]),
            session_id=data["session_id"],
            data=data["data"],
            timestamp=data["timestamp"],
            request_id=data.get("request_id"),
        )
```

### 5.2 WebSocket vs REST Transport

#### WebSocket (Recommended)

**Pros:**
- Bidirectional messaging (real-time events)
- Lower latency for frequent messages
- Natural fit for step debugging

**Cons:**
- More complex setup
- Connection state management

**Example WebSocket Flow:**
```python
# Client connects
ws = await websocket_connect("ws://localhost:8000/debug/{session_id}")

# Client: Set breakpoint
await ws.send_json({
    "type": "set_breakpoints",
    "data": {"node_id": "analyze", "position": "before"}
})

# Server: Breakpoint hit event
await ws.send_json({
    "type": "breakpoint_hit",
    "data": {"node_id": "analyze", "state": {...}}
})

# Client: Continue
await ws.send_json({
    "type": "continue",
    "data": {}
})
```

#### REST (Fallback)

**Pros:**
- Simple to implement
- Works with existing HTTP infrastructure

**Cons:**
- Unidirectional (no push events)
- Polling required for events
- Higher latency

**Example REST API:**
```python
# POST /debug/{session_id}/breakpoints - Set breakpoint
# DELETE /debug/{session_id}/breakpoints/{id} - Clear breakpoint
# GET /debug/{session_id}/breakpoints - List breakpoints
# POST /debug/{session_id}/continue - Continue execution
# POST /debug/{session_id}/step - Step execution
# GET /debug/{session_id}/state - Get current state
# GET /debug/{session_id}/events - Poll for events (SSE)
```

### 5.3 Debug Protocol Handler

```python
class DebugProtocolHandler:
    """Handles debug protocol messages (facade pattern).

    Routes messages to appropriate components and handles
    request/response correlation.

    Attributes:
        session: DebugSession instance
        event_bus: EventBus for emitting events

    Example:
        handler = DebugProtocolHandler(session, event_bus)

        # Handle client message
        await handler.handle_message(message)

        # Emit event to client
        handler.emit_breakpoint_hit(node_id, state)
    """

    def __init__(self, session: DebugSession, event_bus: Any):
        """Initialize protocol handler.

        Args:
            session: DebugSession instance
            event_bus: EventBus for events
        """
        self.session = session
        self._event_bus = event_bus

    async def handle_message(self, message: DebugMessage) -> Optional[DebugMessage]:
        """Handle incoming debug protocol message.

        Args:
            message: DebugMessage from client

        Returns:
            Optional response message

        Raises:
            ValueError: If message type is unknown
        """
        handlers = {
            DebugMessageType.SET_BREAKPOINTS: self._handle_set_breakpoints,
            DebugMessageType.CLEAR_BREAKPOINTS: self._handle_clear_breakpoints,
            DebugMessageType.CONTINUE: self._handle_continue,
            DebugMessageType.STEP_OVER: self._handle_step_over,
            DebugMessageType.STEP_INTO: self._handle_step_into,
            DebugMessageType.STEP_OUT: self._handle_step_out,
            DebugMessageType.PAUSE: self._handle_pause,
            DebugMessageType.INSPECT_STATE: self._handle_inspect_state,
            DebugMessageType.GET_DIFF: self._handle_get_diff,
        }

        handler = handlers.get(message.type)
        if not handler:
            raise ValueError(f"Unknown message type: {message.type}")

        return await handler(message)

    async def _handle_set_breakpoints(self, message: DebugMessage) -> DebugMessage:
        """Handle SET_BREAKPOINTS message."""
        data = message.data
        bp = self.session.set_breakpoint(
            node_id=data.get("node_id"),
            position=BreakpointPosition(data.get("position", "before")),
        )

        return DebugMessage(
            type=DebugMessageType.BREAKPOINT_HIT,
            session_id=message.session_id,
            data={"breakpoint_id": bp.id},
            timestamp=asyncio.get_event_loop().time(),
            request_id=message.request_id,
        )

    async def _handle_continue(self, message: DebugMessage) -> DebugMessage:
        """Handle CONTINUE message."""
        await self.session.continue_execution()

        return DebugMessage(
            type=DebugMessageType.RESUMED,
            session_id=message.session_id,
            data={},
            timestamp=asyncio.get_event_loop().time(),
            request_id=message.request_id,
        )

    async def _handle_step_over(self, message: DebugMessage) -> DebugMessage:
        """Handle STEP_OVER message."""
        await self.session.step_over()

        return DebugMessage(
            type=DebugMessageType.RESUMED,
            session_id=message.session_id,
            data={"step_mode": "step_over"},
            timestamp=asyncio.get_event_loop().time(),
            request_id=message.request_id,
        )

    async def _handle_inspect_state(self, message: DebugMessage) -> DebugMessage:
        """Handle INSPECT_STATE message."""
        state = self.session.get_current_state()

        return DebugMessage(
            type=DebugMessageType.STATE_UPDATE,
            session_id=message.session_id,
            data={"state": state},
            timestamp=asyncio.get_event_loop().time(),
            request_id=message.request_id,
        )

    def emit_breakpoint_hit(self, node_id: str, state: Dict[str, Any]) -> None:
        """Emit BREAKPOINT_HIT event to client."""
        try:
            self._event_bus.emit_lifecycle_event(
                "debug_breakpoint_hit",
                {"session_id": self.session.config.session_id, "node_id": node_id}
            )
        except Exception:
            pass
```

---

## 6. Integration with StateGraph

### 6.1 DebugHook Integration Point

```python
class DebugHook:
    """Hook for debugging workflow execution (injected into CompiledGraph).

    This class is the integration point between StateGraph execution
    and the debugging system. It checks breakpoints and controls
    execution pause/resume.

    Design:
        - Non-invasive: Only active when debugging is enabled
        - Minimal overhead: Fast path when no breakpoints
        - Async-safe: Properly handles async pause/resume

    Attributes:
        session_id: Debug session ID
        breakpoint_mgr: BreakpointManager instance
        execution_ctrl: ExecutionController instance
        inspector: StateInspector instance
        enabled: Whether debugging is active

    Example:
        hook = DebugHook(session_id="debug-123", ...)

        # In CompiledGraph.invoke()
        await hook.before_node(node_id, state)

        # Execute node
        result = await node.execute(state)

        # After node
        await hook.after_node(node_id, state, error=None)
    """

    def __init__(
        self,
        session_id: str,
        breakpoint_mgr: BreakpointManager,
        execution_ctrl: ExecutionController,
        inspector: StateInspector,
        event_bus: Any,
    ):
        """Initialize debug hook.

        Args:
            session_id: Debug session ID
            breakpoint_mgr: BreakpointManager instance
            execution_ctrl: ExecutionController instance
            inspector: StateInspector instance
            event_bus: EventBus for events
        """
        self.session_id = session_id
        self._breakpoint_mgr = breakpoint_mgr
        self._execution_ctrl = execution_ctrl
        self._inspector = inspector
        self._event_bus = event_bus
        self._enabled = True

    def disable(self) -> None:
        """Disable debug hook (no breakpoint checks)."""
        self._enabled = False

    async def before_node(
        self, node_id: str, state: Dict[str, Any]
    ) -> None:
        """Called before node execution.

        Checks for BEFORE position breakpoints and pauses if hit.

        Args:
            node_id: Node about to execute
            state: Current workflow state

        Raises:
            asyncio.CancelledError: If execution is cancelled during pause
        """
        if not self._enabled:
            return

        # Evaluate breakpoints
        breakpoints = self._breakpoint_mgr.evaluate_breakpoints(
            state=state,
            node_id=node_id,
            position=BreakpointPosition.BEFORE,
        )

        # Check if should pause
        if self._execution_ctrl.should_pause(node_id, state, breakpoints):
            # Capture snapshot before pause
            self._inspector.capture_snapshot(state, node_id)

            # Pause execution
            context = await self._execution_ctrl.pause(
                node_id=node_id,
                state=state,
                position=BreakpointPosition.BEFORE,
                breakpoints=breakpoints,
            )

            # Emit paused event
            self._emit_paused(context)

    async def after_node(
        self, node_id: str, state: Dict[str, Any], error: Optional[Exception] = None
    ) -> None:
        """Called after node execution.

        Checks for AFTER/ON_ERROR position breakpoints and pauses if hit.

        Args:
            node_id: Node that executed
            state: Current workflow state
            error: Exception if one occurred
        """
        if not self._enabled:
            return

        # Determine position
        position = BreakpointPosition.ON_ERROR if error else BreakpointPosition.AFTER

        # Evaluate breakpoints
        breakpoints = self._breakpoint_mgr.evaluate_breakpoints(
            state=state,
            node_id=node_id,
            position=position,
            error=error,
        )

        # Check if should pause
        if self._execution_ctrl.should_pause(node_id, state, breakpoints):
            # Capture snapshot after pause
            self._inspector.capture_snapshot(state, node_id)

            # Pause execution
            context = await self._execution_ctrl.pause(
                node_id=node_id,
                state=state,
                position=position,
                breakpoints=breakpoints,
                error=error,
            )

            # Emit paused event
            self._emit_paused(context)

    def _emit_paused(self, context: PauseContext) -> None:
        """Emit paused event."""
        try:
            self._event_bus.emit_lifecycle_event(
                "debug_paused",
                {
                    "session_id": self.session_id,
                    "node_id": context.node_id,
                    "position": context.position.value,
                    "breakpoint_ids": context.breakpoint_ids,
                }
            )
        except Exception:
            pass
```

### 6.2 CompiledGraph Modifications

```python
# In victor/framework/graph.py, modify CompiledGraph.invoke()

class CompiledGraph(Generic[StateType]):
    """Compiled graph ready for execution."""

    def __init__(self, ...):
        """..."""
        # ... existing initialization ...
        self._debug_hook: Optional[DebugHook] = None  # Add debug hook

    def set_debug_hook(self, hook: Optional[DebugHook]) -> None:
        """Set debug hook for execution.

        Args:
            hook: DebugHook instance or None to disable debugging
        """
        self._debug_hook = hook

    async def invoke(
        self,
        input_state: StateType,
        *,
        config: Optional[GraphConfig] = None,
        thread_id: Optional[str] = None,
        debug_hook: Optional[DebugHook] = None,  # NEW: Optional debug hook
    ) -> ExecutionResult[StateType]:
        """Execute the graph (SRP: Orchestrates focused helpers).

        Args:
            input_state: Initial state
            config: Override execution config
            thread_id: Thread ID for checkpointing
            debug_hook: Optional DebugHook for debugging (NEW)

        Returns:
            ExecutionResult with final state
        """
        # Use parameter debug hook or instance debug hook
        hook = debug_hook or self._debug_hook

        # ... existing initialization ...

        try:
            while current_node != END:
                # Check iteration limits (existing code)

                # ========== NEW: Debug hook - before node ==========
                if hook:
                    await hook.before_node(current_node, state)

                # Emit node start event (existing code)
                node_start_time = time.time()
                event_emitter.emit_node_start(
                    node_id=current_node,
                    iteration=iteration_controller.iterations,
                )

                # Execute node (existing code)
                success, error, state = await node_executor.execute(
                    node_id=current_node,
                    state=state,
                    timeout_manager=timeout_manager,
                )

                # ========== NEW: Debug hook - after node ==========
                if hook:
                    await hook.after_node(current_node, state, error if not success else None)

                if not success:
                    # ... existing error handling ...

                # Track execution (existing code)
                logger.debug(f"Executed node: {current_node}")
                node_history.append(current_node)

                # ... rest of existing execution logic ...

            # Emit graph completed event (existing code)
            event_emitter.emit_graph_completed(
                success=True,
                iterations=iteration_controller.iterations,
                duration=timeout_manager.get_elapsed(),
                node_count=len(node_history),
            )

            return ExecutionResult(
                state=state,
                success=True,
                iterations=iteration_controller.iterations,
                duration=timeout_manager.get_elapsed(),
                node_history=node_history,
            )

        # ... existing exception handling ...
```

### 6.3 Backward Compatibility

**No impact on non-debugging workflows:**

```python
# Existing code continues to work without changes
app = graph.compile()
result = await app.invoke(initial_state)  # No debug_hook parameter

# Debugging is opt-in via debug_hook parameter
session = DebugSession(config=...)
result = await app.invoke(initial_state, debug_hook=session.create_hook())
```

**Zero overhead when debugging disabled:**

```python
# DebugHook checks `if not self._enabled: return` immediately
# No breakpoint evaluation, no pause checks when disabled
```

---

## 7. Implementation Plan

### 7.1 Module Implementation Effort

| Module | LOC Estimate | Time Estimate | Dependencies |
|--------|--------------|---------------|--------------|
| `breakpoints.py` | 400 LOC | 2-3 days | dataclasses, typing, asyncio |
| `execution.py` | 350 LOC | 2-3 days | asyncio, enum, dataclasses |
| `inspector.py` | 250 LOC | 1-2 days | typing, dataclasses |
| `protocol.py` | 200 LOC | 1 day | enum, dataclasses |
| `hooks.py` | 150 LOC | 1 day | asyncio, typing |
| `session.py` | 200 LOC | 1 day | dataclasses, asyncio |
| `graph.py` modifications | 50 LOC | 0.5 day | existing code |
| **Total** | **1600 LOC** | **9-11 days** | - |

### 7.2 Implementation Sequence

#### Phase 1: Core Data Structures (Days 1-3)
1. **Day 1: Breakpoint types and storage**
   - Implement `BreakpointType`, `BreakpointPosition` enums
   - Implement `WorkflowBreakpoint` dataclass
   - Implement `BreakpointStorage` class
   - Unit tests for breakpoint data structures

2. **Day 2: BreakpointManager**
   - Implement `BreakpointManager` with CRUD operations
   - Implement breakpoint evaluation logic
   - Add event emission for breakpoints
   - Unit tests for BreakpointManager

3. **Day 3: StateSnapshot and StateDiff**
   - Implement `StateSnapshot` dataclass
   - Implement `StateDiff` comparison
   - Implement `StateInspector` class
   - Unit tests for state inspection

#### Phase 2: Execution Control (Days 4-6)
4. **Day 4: ExecutionController**
   - Implement `ExecutionState` and `StepMode` enums
   - Implement `ExecutionController` with pause/resume
   - Add step-over/step-into/step-out logic
   - Unit tests for execution control

5. **Day 5: DebugHook**
   - Implement `DebugHook` class
   - Add before_node/after_node hooks
   - Integrate with BreakpointManager and ExecutionController
   - Unit tests for DebugHook

6. **Day 6: DebugSession**
   - Implement `DebugSession` facade
   - Add session lifecycle management
   - Integrate all components (BreakpointManager, ExecutionController, StateInspector)
   - Unit tests for DebugSession

#### Phase 3: Protocol and Integration (Days 7-8)
7. **Day 7: Debug Protocol**
   - Implement `DebugMessageType` enum
   - Implement `DebugMessage` dataclass
   - Implement `DebugProtocolHandler`
   - Unit tests for protocol handler

8. **Day 8: StateGraph Integration**
   - Modify `CompiledGraph.invoke()` to accept `debug_hook` parameter
   - Add `set_debug_hook()` method to CompiledGraph
   - Add before_node/after_node hook calls in execution loop
   - Integration tests with sample workflow

#### Phase 4: Testing and Documentation (Days 9-11)
9. **Day 9: Unit Tests**
   - Complete unit test coverage (>90%)
   - Add edge case tests (timeouts, cancellation, errors)
   - Add performance benchmarks (overhead when disabled)

10. **Day 10: Integration Tests**
    - End-to-end tests with real workflows
    - Test WebSocket transport (if implemented)
    - Test REST transport (fallback)
    - Test with checkpointing integration

11. **Day 11: Documentation**
    - Update this design doc with any changes
    - Add user guide for workflow debugging
    - Add API reference for debugging modules
    - Add examples to CLAUDE.md

### 7.3 Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **Async complexity** pause/resume coordination | Use asyncio.Event, extensive async testing |
| **Performance impact** on non-debugging workflows | Fast-path checks, microbenchmarks, profile |
| **State serialization** large state objects | Selective serialization, size limits |
| **Thread safety** if workflows use threads | Document non-thread-safe, use thread-local controllers |
| **Integration complexity** with existing code | Minimal invasive changes, extensive testing |

---

## 8. MVP Features

### 8.1 MVP Feature Set (Minimum Viable Product)

**Core Capabilities:**
1. **Node Breakpoints**
   - Set breakpoint before/after specific node
   - Enable/disable breakpoints
   - Clear breakpoints

2. **Conditional Breakpoints**
   - Simple condition function: `lambda state: state["error"] > 0`
   - Evaluate condition at breakpoint position

3. **Exception Breakpoints**
   - Pause on any workflow error
   - Show exception in pause context

4. **Execution Control**
   - Continue from breakpoint
   - Step-over (execute next node)

5. **State Inspection**
   - Get current workflow state at pause
   - Get state summary (keys and types)
   - Get state diff (before/after node)

**Excluded from MVP:**
- Step-into/step-out (complex sub-workflow tracking)
- State breakpoints (pause when state key matches value)
- Hit count breakpoints (ignore first N hits)
- Log points (log message instead of pausing)
- Breakpoint persistence (save to disk)
- WebSocket transport (use REST polling instead)
- Debug client UI (CLI only)

**MVP Deliverables:**
- Core modules (breakpoints, execution, inspector, hooks, session)
- Modified CompiledGraph.invoke() with debug_hook parameter
- CLI commands for debugging: `victor workflow debug`
- Unit tests (>80% coverage)
- Integration tests with sample workflows
- Documentation (user guide, API reference)

### 8.2 Post-MVP Features

**Phase 2 (Future Enhancements):**
1. Advanced stepping (step-into/step-out)
2. State breakpoints
3. Hit count breakpoints
4. Log points
5. Breakpoint persistence
6. WebSocket transport
7. Debug UI (TUI or web)
8. Debug multiple concurrent workflows
9. Breakpoint groups (enable/disable by group)
10. Conditional breakpoint editor (parse expressions)

---

## 9. Unit Test Strategy

### 9.1 Test Structure

```
tests/unit/framework/debugging/
├── conftest.py                  # Shared fixtures
├── test_breakpoints.py          # BreakpointManager tests
├── test_execution.py            # ExecutionController tests
├── test_inspector.py            # StateInspector tests
├── test_hooks.py                # DebugHook tests
├── test_session.py              # DebugSession tests
└── test_protocol.py             # DebugProtocolHandler tests

tests/integration/framework/debugging/
├── test_workflow_debugging.py   # End-to-end workflow debugging
└── test_stategraph_integration.py # StateGraph integration tests
```

### 9.2 Fixtures (conftest.py)

```python
import pytest
from victor.framework.debugging.breakpoints import (
    BreakpointManager,
    BreakpointStorage,
    WorkflowBreakpoint,
    BreakpointType,
    BreakpointPosition,
)
from victor.framework.debugging.execution import (
    ExecutionController,
    ExecutionState,
    StepMode,
)
from victor.framework.debugging.inspector import StateInspector, StateSnapshot, StateDiff
from victor.framework.debugging.session import DebugSession, DebugSessionConfig
from victor.framework.debugging.hooks import DebugHook
from victor.core.events import ObservabilityBus as EventBus


@pytest.fixture
def event_bus():
    """EventBus instance for testing."""
    return EventBus()


@pytest.fixture
def breakpoint_storage():
    """BreakpointStorage instance."""
    return BreakpointStorage()


@pytest.fixture
def breakpoint_manager(event_bus):
    """BreakpointManager instance."""
    return BreakpointManager(event_bus)


@pytest.fixture
def execution_controller():
    """ExecutionController instance."""
    return ExecutionController(session_id="test-session")


@pytest.fixture
def state_inspector():
    """StateInspector instance."""
    return StateInspector()


@pytest.fixture
def sample_state():
    """Sample workflow state for testing."""
    return {
        "task": "Analyze code",
        "file_path": "/tmp/test.py",
        "errors": 0,
        "results": [],
        "metadata": {"iteration": 1},
    }


@pytest.fixture
def debug_session(event_bus):
    """DebugSession instance."""
    config = DebugSessionConfig(
        session_id="test-session",
        workflow_id="test-workflow",
    )
    return DebugSession(config=config, event_bus=event_bus)


@pytest.fixture
def debug_hook(debug_session):
    """DebugHook instance."""
    return debug_session.create_hook()


@pytest.fixture
def sample_workflow():
    """Create sample StateGraph for testing."""
    from victor.framework.graph import StateGraph

    graph = StateGraph()

    async def node_analyze(state):
        state["analysis"] = "complete"
        return state

    async def node_process(state):
        state["errors"] = state.get("errors", 0) + 1
        return state

    graph.add_node("analyze", node_analyze)
    graph.add_node("process", node_process)
    graph.add_edge("analyze", "process")
    graph.set_entry_point("analyze")

    return graph.compile()
```

### 9.3 Test Examples

#### Test 1: BreakpointManager

```python
import pytest
from victor.framework.debugging.breakpoints import BreakpointManager, BreakpointPosition


@pytest.mark.unit
class TestBreakpointManager:
    """Test BreakpointManager."""

    async def test_set_node_breakpoint(self, breakpoint_manager):
        """Test setting a node breakpoint."""
        bp = breakpoint_manager.set_breakpoint(
            node_id="analyze",
            position=BreakpointPosition.BEFORE,
        )

        assert bp.node_id == "analyze"
        assert bp.position == BreakpointPosition.BEFORE
        assert bp.enabled is True
        assert bp.id is not None

    async def test_set_conditional_breakpoint(self, breakpoint_manager):
        """Test setting a conditional breakpoint."""
        condition = lambda state: state.get("errors", 0) > 5

        bp = breakpoint_manager.set_breakpoint(
            node_id="process",
            position=BreakpointPosition.AFTER,
            condition=condition,
        )

        assert bp.type == BreakpointType.CONDITIONAL
        assert bp.condition == condition

    async def test_clear_breakpoint(self, breakpoint_manager):
        """Test clearing a breakpoint."""
        bp = breakpoint_manager.set_breakpoint(node_id="analyze")

        cleared = breakpoint_manager.clear_breakpoint(bp.id)

        assert cleared is True
        assert breakpoint_manager.storage.get(bp.id) is None

    async def test_list_breakpoints(self, breakpoint_manager):
        """Test listing breakpoints."""
        bp1 = breakpoint_manager.set_breakpoint(node_id="analyze")
        bp2 = breakpoint_manager.set_breakpoint(node_id="process")

        all_bps = breakpoint_manager.list_breakpoints()

        assert len(all_bps) == 2
        assert bp1 in all_bps
        assert bp2 in all_bps

    async def test_enable_disable_breakpoint(self, breakpoint_manager):
        """Test enabling/disabling breakpoint."""
        bp = breakpoint_manager.set_breakpoint(node_id="analyze")

        breakpoint_manager.disable_breakpoint(bp.id)
        assert breakpoint_manager.storage.get(bp.id).enabled is False

        breakpoint_manager.enable_breakpoint(bp.id)
        assert breakpoint_manager.storage.get(bp.id).enabled is True

    async def test_evaluate_breakpoints_hit(self, breakpoint_manager, sample_state):
        """Test breakpoint evaluation when hit."""
        bp = breakpoint_manager.set_breakpoint(
            node_id="analyze",
            position=BreakpointPosition.BEFORE,
        )

        hit_bps = breakpoint_manager.evaluate_breakpoints(
            state=sample_state,
            node_id="analyze",
            position=BreakpointPosition.BEFORE,
        )

        assert len(hit_bps) == 1
        assert hit_bps[0].id == bp.id
        assert hit_bps[0].hit_count == 1

    async def test_evaluate_breakpoints_miss(self, breakpoint_manager, sample_state):
        """Test breakpoint evaluation when missed."""
        bp = breakpoint_manager.set_breakpoint(
            node_id="analyze",
            position=BreakpointPosition.BEFORE,
        )

        hit_bps = breakpoint_manager.evaluate_breakpoints(
            state=sample_state,
            node_id="process",  # Wrong node
            position=BreakpointPosition.BEFORE,
        )

        assert len(hit_bps) == 0
        assert bp.hit_count == 0
```

#### Test 2: ExecutionController

```python
import pytest
import asyncio
from victor.framework.debugging.execution import (
    ExecutionController,
    ExecutionState,
    StepMode,
)


@pytest.mark.unit
class TestExecutionController:
    """Test ExecutionController."""

    async def test_pause_and_continue(self, execution_controller, sample_state):
        """Test pause and continue flow."""
        assert execution_controller.state == ExecutionState.RUNNING

        # Pause in background task
        async def pause_task():
            await execution_controller.pause(
                node_id="analyze",
                state=sample_state,
                position=BreakpointPosition.BEFORE,
                breakpoints=[],
            )

        pause_task = asyncio.create_task(pause_task())

        # Wait a bit for pause to occur
        await asyncio.sleep(0.1)

        assert execution_controller.state == ExecutionState.PAUSED

        # Resume
        await execution_controller.continue_execution()

        # Wait for pause to complete
        await pause_task

    async def test_step_over(self, execution_controller, sample_state):
        """Test step over command."""
        # Pause first
        async def pause_task():
            await execution_controller.pause(
                node_id="analyze",
                state=sample_state,
                position=BreakpointPosition.BEFORE,
                breakpoints=[],
            )

        pause_task = asyncio.create_task(pause_task())
        await asyncio.sleep(0.1)

        # Step over
        await execution_controller.step_over()

        assert execution_controller.state == ExecutionState.STEPPING
        assert execution_controller._step_mode == StepMode.STEP_OVER

        # Let pause complete
        await pause_task

    async def test_should_pause_with_breakpoints(self, execution_controller, sample_state):
        """Test should_pause with breakpoints."""
        bp = WorkflowBreakpoint(
            id="bp-1",
            type=BreakpointType.NODE,
            position=BreakpointPosition.BEFORE,
            node_id="analyze",
        )

        should = execution_controller.should_pause(
            node_id="analyze",
            state=sample_state,
            breakpoints=[bp],
        )

        assert should is True
```

#### Test 3: StateInspector

```python
import pytest
from victor.framework.debugging.inspector import StateInspector, StateDiff


@pytest.mark.unit
class TestStateInspector:
    """Test StateInspector."""

    def test_capture_snapshot(self, state_inspector, sample_state):
        """Test capturing state snapshot."""
        snapshot = state_inspector.capture_snapshot(sample_state, "analyze")

        assert snapshot.node_id == "analyze"
        assert snapshot.state == sample_state
        assert snapshot.state_summary == {
            "task": "str",
            "file_path": "str",
            "errors": "int",
            "results": "list",
            "metadata": "dict",
        }

    def test_compare_states_no_changes(self, state_inspector, sample_state):
        """Test comparing identical states."""
        diff = state_inspector.compare_states(sample_state, sample_state)

        assert diff.has_changes() is False
        assert len(diff.changed_keys) == 0
        assert len(diff.added_keys) == 0
        assert len(diff.removed_keys) == 0

    def test_compare_states_with_changes(self, state_inspector, sample_state):
        """Test comparing different states."""
        modified = sample_state.copy()
        modified["errors"] = 5
        modified["new_key"] = "new_value"
        del modified["task"]

        diff = state_inspector.compare_states(sample_state, modified)

        assert diff.has_changes() is True
        assert diff.changed_keys["errors"] == (0, 5)
        assert "new_key" in diff.added_keys
        assert "task" in diff.removed_keys

    def test_get_value_nested(self, state_inspector, sample_state):
        """Test getting nested value."""
        value = state_inspector.get_value(sample_state, "metadata.iteration")

        assert value == 1

    def test_get_value_missing(self, state_inspector, sample_state):
        """Test getting missing value with default."""
        value = state_inspector.get_value(
            sample_state, "missing_key", default="default"
        )

        assert value == "default"
```

#### Test 4: Integration Test

```python
import pytest
from victor.framework.graph import StateGraph


@pytest.mark.integration
@pytest.mark.slow
class TestWorkflowDebugging:
    """Integration tests for workflow debugging."""

    async def test_debug_workflow_with_breakpoint(self, sample_workflow, debug_session):
        """Test debugging a workflow with breakpoint."""
        # Set breakpoint
        bp = debug_session.set_breakpoint(
            node_id="analyze",
            position=BreakpointPosition.BEFORE,
        )

        # Create hook
        hook = debug_session.create_hook()

        # Execute workflow in background
        async def execute_workflow():
            return await sample_workflow.invoke(
                {"task": "test"},
                debug_hook=hook,
            )

        execution_task = asyncio.create_task(execute_workflow())

        # Wait for breakpoint to be hit
        await asyncio.sleep(0.5)

        # Check paused
        assert debug_session.is_paused is True

        context = debug_session.get_pause_context()
        assert context.node_id == "analyze"
        assert bp.id in context.breakpoint_ids

        # Continue execution
        await debug_session.continue_execution()

        # Wait for completion
        result = await execution_task

        assert result.success is True

    async def test_step_through_workflow(self, sample_workflow, debug_session):
        """Test stepping through workflow execution."""
        # Set breakpoint
        debug_session.set_breakpoint(node_id="analyze", position=BreakpointPosition.AFTER)

        hook = debug_session.create_hook()

        async def execute_workflow():
            return await sample_workflow.invoke(
                {"task": "test"},
                debug_hook=hook,
            )

        execution_task = asyncio.create_task(execute_workflow())
        await asyncio.sleep(0.5)

        # First breakpoint hit
        assert debug_session.is_paused is True

        # Step over to next node
        await debug_session.step_over()
        await asyncio.sleep(0.5)

        # Should be paused again
        assert debug_session.is_paused is True
        context = debug_session.get_pause_context()
        assert context.node_id == "process"

        # Continue to completion
        await debug_session.continue_execution()
        result = await execution_task

        assert result.success is True
```

### 9.4 Test Coverage Goals

- **Unit tests**: >90% code coverage
- **Integration tests**: Cover all major debugging flows
- **Edge cases**: Timeouts, cancellation, errors, empty state
- **Performance**: Overhead <5% when debugging disabled

---

## 10. Summary and Next Steps

### 10.1 Design Summary

This document designs a comprehensive breakpoint and debugging system for StateGraph workflows with the following characteristics:

**Key Features:**
1. **Non-invasive integration** with StateGraph via DebugHook
2. **DAP-aligned** breakpoint types (node, conditional, exception, state)
3. **Async-first** execution control with pause/resume/step
4. **Rich state inspection** with snapshots and diffing
5. **Clean separation of concerns** via focused modules (SRP compliance)
6. **Backward compatible** - zero impact on non-debugging workflows

**Architecture:**
- **BreakpointManager**: Manages breakpoint CRUD and evaluation
- **ExecutionController**: Coordinates pause/resume with asyncio
- **StateInspector**: Captures state snapshots and diffs
- **DebugHook**: Integration point with StateGraph execution
- **DebugSession**: Facade combining all components
- **DebugProtocolHandler**: Protocol message handling

**Estimated Effort:**
- **1600 LOC** across 7 modules
- **9-11 days** implementation time
- **MVP**: Core breakpoint, pause/resume, state inspection

### 10.2 Next Steps

1. **Review and approval** of this design document
2. **Create GitHub issue** tracking implementation tasks
3. **Begin implementation** following phased approach (Section 7.2)
4. **Set up CI/CD** for debugging module tests
5. **User documentation** and examples
6. **Beta testing** with real workflows

### 10.3 Open Questions

1. **WebSocket vs REST**: Should we implement WebSocket transport in MVP or defer to post-MVP?
   - **Recommendation**: Defer to post-MVP, use REST polling in MVP

2. **Breakpoint persistence**: Should breakpoints persist across sessions in MVP?
   - **Recommendation**: No, defer to post-MVP (complexity not justified for MVP)

3. **Sub-workflow debugging**: How should step-into work with nested workflows?
   - **Recommendation**: Defer step-into/step-out to post-MVP (requires workflow call stack tracking)

4. **Performance**: What is acceptable overhead for breakpoint checks?
   - **Recommendation**: Target <5% overhead when debugging disabled, <10% when enabled

5. **State size limits**: How should we handle very large state objects (>1MB)?
   - **Recommendation**: Add size limits and selective serialization in post-MVP

---

## Appendix A: Glossary

- **Breakpoint**: A point in workflow execution where debugging pauses
- **Conditional breakpoint**: Breakpoint that triggers when a condition is true
- **Exception breakpoint**: Breakpoint that triggers when an error occurs
- **Pause context**: State snapshot when execution is paused
- **Debug hook**: Integration point between StateGraph and debugging system
- **Step-over**: Execute next node without entering sub-workflows
- **Step-into**: Execute next node, entering sub-workflows
- **Step-out**: Complete current sub-workflow and return to caller
- **State diff**: Difference between two state snapshots
- **Debug session**: Active debugging instance with breakpoints and state

---

## Appendix B: References

1. **Debug Adapter Protocol (DAP)**: https://microsoft.github.io/debug-adapter-protocol/
2. **LangGraph Debugging**: https://langchain-ai.github.io/langgraph/concepts/low_level/#debugging
3. **Python pdb module**: https://docs.python.org/3/library/pdb.html
4. **Existing Victor Infrastructure**:
   - `victor/observability/debugger.py` - AgentDebugger facade
   - `victor/observability/debug/protocol.py` - DAP types
   - `victor/framework/graph.py` - StateGraph implementation
   - `victor/workflows/observability.py` - ObservabilityEmitter

---

**Document Version**: 1.0
**Last Updated**: 2025-01-09
**Status**: Ready for Review
