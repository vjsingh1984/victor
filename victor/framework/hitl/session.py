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

"""Session management for HITL workflows.

This module provides session management for multi-step HITL workflows,
allowing tracking of state across multiple gate executions.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional
from threading import Lock


class SessionState(str, Enum):
    """State of an HITL session.

    Attributes:
        ACTIVE: Session is active and processing
        PAUSED: Session is paused waiting for human input
        COMPLETED: Session completed successfully
        FAILED: Session failed
        TIMEOUT: Session timed out
    """

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class SessionEvent:
    """Event in a HITL session.

    Attributes:
        event_type: Type of event
        timestamp: Unix timestamp
        gate_id: Associated gate ID
        data: Event data
    """

    event_type: str
    timestamp: float = field(default_factory=time.time)
    gate_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HITLSessionConfig:
    """Configuration for HITL sessions.

    Attributes:
        default_timeout: Default timeout for gates
        default_fallback_behavior: Default fallback behavior
        auto_resume: Auto-resume on handler completion
        max_concurrent_gates: Maximum concurrent gates
        persist_history: Whether to persist event history
        max_history_size: Maximum history size
    """

    default_timeout: float = 300.0
    default_fallback_behavior: str = "abort"
    auto_resume: bool = True
    max_concurrent_gates: int = 10
    persist_history: bool = True
    max_history_size: int = 1000


@dataclass
class GateExecutionResult:
    """Result of gate execution.

    Attributes:
        gate_id: ID of the gate
        gate_type: Type of gate
        approved: Whether approved/proceed
        value: Response value
        reason: Optional reason
        executed_at: Timestamp
    """

    gate_id: str
    gate_type: str
    approved: bool
    value: Optional[Any] = None
    reason: Optional[str] = None
    executed_at: float = field(default_factory=time.time)


class HITLSession:
    """A session for managing multi-step HITL workflows.

    Sessions track state across multiple gate executions, maintain
    event history, and provide hooks for lifecycle management.

    Example:
        session = HITLSession(workflow_id="deployment")

        # Execute gates in sequence
        result1 = await session.execute_gate(approval_gate)
        if not result1.approved:
            return

        result2 = await session.execute_gate(input_gate)

        # Get session summary
        summary = session.get_summary()
    """

    def __init__(
        self,
        workflow_id: str,
        config: Optional[HITLSessionConfig] = None,
        session_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a new HITL session.

        Args:
            workflow_id: ID of the workflow
            config: Session configuration
            session_id: Optional custom session ID
            initial_context: Initial context for the session
        """
        self.session_id = session_id or f"session_{uuid.uuid4().hex}"
        self.workflow_id = workflow_id
        self.config = config or HITLSessionConfig()

        self._state = SessionState.ACTIVE
        self._context: Dict[str, Any] = initial_context or {}
        self._history: List[SessionEvent] = []
        self._results: List[GateExecutionResult] = []
        self._created_at = time.time()
        self._updated_at = time.time()

        # Concurrent gate management
        self._pending_gates: Dict[str, asyncio.Event] = {}
        self._lock = Lock()

        # Callbacks
        self._on_state_change: List[Callable[[SessionState, SessionState], None]] = []
        self._on_gate_complete: List[Callable[[GateExecutionResult], None]] = []

    @property
    def state(self) -> SessionState:
        """Current session state."""
        return self._state

    @property
    def context(self) -> Dict[str, Any]:
        """Session context."""
        return dict(self._context)

    @property
    def created_at(self) -> float:
        """Session creation timestamp."""
        return self._created_at

    @property
    def updated_at(self) -> float:
        """Last update timestamp."""
        return self._updated_at

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value.

        Args:
            key: Context key
            default: Default value if not found

        Returns:
            The context value or default
        """
        return self._context.get(key, default)

    def set_context(self, key: str, value: Any) -> None:
        """Set a context value.

        Args:
            key: Context key
            value: Value to set
        """
        self._context[key] = value
        self._updated_at = time.time()

    def update_context(self, **kwargs: Any) -> None:
        """Update multiple context values.

        Args:
            **kwargs: Key-value pairs to update
        """
        self._context.update(kwargs)
        self._updated_at = time.time()

    async def execute_gate(
        self,
        gate: Any,
        context: Optional[Dict[str, Any]] = None,
        handler: Optional[Callable[..., Awaitable]] = None,
    ) -> GateExecutionResult:
        """Execute a gate within this session.

        Args:
            gate: The gate to execute
            context: Additional context for this execution
            handler: Custom handler for the interaction

        Returns:
            GateExecutionResult with the response
        """
        if self._state not in (SessionState.ACTIVE, SessionState.PAUSED):
            raise RuntimeError(f"Cannot execute gate in state: {self._state}")

        # Merge context
        merged_context = {**self._context}
        if context:
            merged_context.update(context)

        # Record start event
        self._add_event("gate_start", gate_id=gate.gate_id, data={"gate_type": gate.gate_type})

        # Change state to paused
        self._set_state(SessionState.PAUSED)

        try:
            # Execute the gate
            response = await gate.execute(context=merged_context, handler=handler)

            # Create result
            result = GateExecutionResult(
                gate_id=gate.gate_id,
                gate_type=gate.gate_type,
                approved=response.approved if hasattr(response, "approved") else True,
                value=getattr(response, "value", None),
                reason=getattr(response, "reason", None),
            )

            self._results.append(result)

            # Update context with result
            self._context[f"_{gate.gate_type}_result"] = result

            # Record complete event
            self._add_event(
                "gate_complete",
                gate_id=gate.gate_id,
                data={"approved": result.approved},
            )

            # Notify callbacks
            for callback in self._on_gate_complete:
                callback(result)

            # Resume if approved
            if result.approved:
                self._set_state(SessionState.ACTIVE)

            return result

        except Exception as e:
            # Record error event
            self._add_event(
                "gate_error",
                gate_id=gate.gate_id,
                data={"error": str(e)},
            )
            self._set_state(SessionState.FAILED)
            raise

    def pause(self) -> None:
        """Pause the session."""
        if self._state == SessionState.ACTIVE:
            self._set_state(SessionState.PAUSED)

    def resume(self) -> None:
        """Resume the session."""
        if self._state == SessionState.PAUSED:
            self._set_state(SessionState.ACTIVE)

    def complete(self) -> None:
        """Mark the session as completed."""
        self._set_state(SessionState.COMPLETED)

    def fail(self, reason: Optional[str] = None) -> None:
        """Mark the session as failed.

        Args:
            reason: Optional reason for failure
        """
        if reason:
            self._add_event("session_failed", data={"reason": reason})
        self._set_state(SessionState.FAILED)

    def get_history(self) -> List[SessionEvent]:
        """Get session event history.

        Returns:
            List of session events
        """
        return list(self._history)

    def get_results(self) -> List[GateExecutionResult]:
        """Get all gate execution results.

        Returns:
            List of execution results
        """
        return list(self._results)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the session.

        Returns:
            Summary dictionary
        """
        gate_counts = {}
        for result in self._results:
            gate_counts[result.gate_type] = gate_counts.get(result.gate_type, 0) + 1

        return {
            "session_id": self.session_id,
            "workflow_id": self.workflow_id,
            "state": self._state.value,
            "created_at": self._created_at,
            "updated_at": self._updated_at,
            "duration": time.time() - self._created_at,
            "gates_executed": len(self._results),
            "gate_counts": gate_counts,
            "events": len(self._history),
            "context_keys": list(self._context.keys()),
        }

    def on_state_change(self, callback: Callable[[SessionState, SessionState], None]) -> None:
        """Register a callback for state changes.

        Args:
            callback: Function to call on state change
        """
        self._on_state_change.append(callback)

    def on_gate_complete(self, callback: Callable[[GateExecutionResult], None]) -> None:
        """Register a callback for gate completion.

        Args:
            callback: Function to call on gate completion
        """
        self._on_gate_complete.append(callback)

    def _set_state(self, new_state: SessionState) -> None:
        """Set the session state and notify callbacks.

        Args:
            new_state: New state to set
        """
        old_state = self._state
        if old_state != new_state:
            self._state = new_state
            self._updated_at = time.time()

            self._add_event(
                "state_change",
                data={"old": old_state.value, "new": new_state.value},
            )

            for callback in self._on_state_change:
                try:
                    callback(old_state, new_state)
                except Exception:
                    pass  # Don't let callback errors break the session

    def _add_event(
        self, event_type: str, gate_id: Optional[str] = None, data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an event to the session history.

        Args:
            event_type: Type of event
            gate_id: Associated gate ID
            data: Event data
        """
        if not self.config.persist_history:
            return

        if len(self._history) >= self.config.max_history_size:
            self._history.pop(0)

        event = SessionEvent(
            event_type=event_type,
            gate_id=gate_id,
            data=data or {},
        )
        self._history.append(event)


class HITLSessionManager:
    """Manager for multiple HITL sessions.

    Provides creation, retrieval, and cleanup of sessions.
    """

    def __init__(self, default_config: Optional[HITLSessionConfig] = None):
        """Initialize the session manager.

        Args:
            default_config: Default configuration for new sessions
        """
        self._sessions: Dict[str, HITLSession] = {}
        self._lock = Lock()
        self._default_config = default_config or HITLSessionConfig()

    def create_session(
        self,
        workflow_id: str,
        config: Optional[HITLSessionConfig] = None,
        session_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> HITLSession:
        """Create a new HITL session.

        Args:
            workflow_id: ID of the workflow
            config: Session configuration
            session_id: Optional custom session ID
            initial_context: Initial context

        Returns:
            The created session
        """
        session = HITLSession(
            workflow_id=workflow_id,
            config=config or self._default_config,
            session_id=session_id,
            initial_context=initial_context,
        )

        with self._lock:
            self._sessions[session.session_id] = session

        return session

    def get_session(self, session_id: str) -> Optional[HITLSession]:
        """Get a session by ID.

        Args:
            session_id: Session ID

        Returns:
            The session or None if not found
        """
        return self._sessions.get(session_id)

    def list_sessions(
        self,
        workflow_id: Optional[str] = None,
        state: Optional[SessionState] = None,
    ) -> List[HITLSession]:
        """List sessions with optional filters.

        Args:
            workflow_id: Filter by workflow ID
            state: Filter by session state

        Returns:
            List of matching sessions
        """
        sessions = list(self._sessions.values())

        if workflow_id:
            sessions = [s for s in sessions if s.workflow_id == workflow_id]

        if state:
            sessions = [s for s in sessions if s.state == state]

        return sessions

    def cleanup_old_sessions(self, max_age_seconds: float = 3600) -> int:
        """Remove old completed/failed sessions.

        Args:
            max_age_seconds: Maximum age to keep sessions

        Returns:
            Number of sessions removed
        """
        now = time.time()
        to_remove = []

        with self._lock:
            for session_id, session in self._sessions.items():
                if session.state in (SessionState.COMPLETED, SessionState.FAILED):
                    if now - session.updated_at > max_age_seconds:
                        to_remove.append(session_id)

            for session_id in to_remove:
                del self._sessions[session_id]

        return len(to_remove)

    def get_all_summaries(self) -> List[Dict[str, Any]]:
        """Get summaries for all sessions.

        Returns:
            List of session summaries
        """
        return [s.get_summary() for s in self._sessions.values()]


# Global session manager
_global_manager: Optional[HITLSessionManager] = None


def get_global_session_manager() -> HITLSessionManager:
    """Get the global session manager.

    Returns:
        The global HITLSessionManager instance
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = HITLSessionManager()
    return _global_manager


__all__ = [
    "SessionState",
    "SessionEvent",
    "HITLSessionConfig",
    "GateExecutionResult",
    "HITLSession",
    "HITLSessionManager",
    "get_global_session_manager",
]
