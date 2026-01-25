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

"""Unified execution context for all workflow runtimes.

MIGRATION NOTICE: This module is deprecated for state management.

For new code, use the canonical state management system:
    - victor.state.WorkflowStateManager - Single workflow scope state
    - victor.state.get_global_manager() - Unified access to all scopes

This module is kept for backward compatibility and type definitions only.

---

Legacy Documentation:

This module provides a single, unified state type for workflow execution,
consolidating the previously fragmented state types:

- WorkflowContext (dataclass in executor.py) - used by WorkflowExecutor
- WorkflowState (TypedDict in yaml_to_graph_compiler.py) - used by StateGraphExecutor
- WorkflowState (TypedDict in adapters.py) - used by workflow adapters

The ExecutionContext TypedDict is designed to be compatible with all runtimes
while providing a consistent interface for state management.

Example:
    # Create execution context
    ctx: ExecutionContext = {
        "data": {"input": "test"},
        "_workflow_id": "wf-123",
        "_current_node": "start",
    }

    # Access data
    input_val = ctx.get("data", {}).get("input")

    # Update state
    ctx["_current_node"] = "process"
    ctx["_node_results"] = {"start": {"success": True}}

Migration Example:
    # OLD (deprecated):
    from victor.workflows.context import ExecutionContext, create_execution_context
    ctx = create_execution_context({"input": "test"})

    # NEW (recommended):
    from victor.state import WorkflowStateManager, StateScope
    mgr = WorkflowStateManager()
    await mgr.set("input", "test")  # Workflow scope
    value = await mgr.get("input")

    # OR for unified access across all scopes:
    from victor.state import get_global_manager
    state = get_global_manager()
    await state.set("input", "test", scope=StateScope.WORKFLOW)
"""

from __future__ import annotations

import logging
import uuid
import warnings
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    TypedDict,
    Union,
)

if TYPE_CHECKING:
    from victor.workflows.executor import NodeResult, TemporalContext, WorkflowContext

logger = logging.getLogger(__name__)


# =============================================================================
# Unified Execution Context (TypedDict)
# =============================================================================


class ExecutionContext(TypedDict, total=False):
    """Unified execution context for all workflow runtimes.

    This TypedDict consolidates the state models from:
    - WorkflowContext (executor.py) - dataclass for WorkflowExecutor
    - WorkflowState (yaml_to_graph_compiler.py) - TypedDict for StateGraphExecutor
    - WorkflowState (adapters.py) - TypedDict for workflow adapters

    All execution metadata uses underscore prefix to avoid conflicts with
    user data keys.

    Attributes:
        # Core data (user-defined workflow data)
        data: Shared context data dictionary
        messages: Conversation messages (for agent workflows)

        # Execution metadata (system-managed)
        _workflow_id: Unique workflow execution ID
        _workflow_name: Name of the workflow being executed
        _current_node: Currently executing node ID
        _node_results: Results from each executed node
        _error: Error message if execution failed

        # Iteration tracking (for loop detection)
        _iteration: Current iteration count
        _visited_nodes: List of visited node IDs

        # Parallel execution
        _parallel_results: Results from parallel node execution

        # Human-in-the-loop
        _hitl_pending: Whether waiting for human input
        _hitl_response: Human response data

        # Temporal context (for backtesting)
        _as_of_date: Point-in-time date for temporal queries
        _lookback_periods: Number of periods to look back
        _include_end_date: Whether to include end date in ranges

        # Completion tracking
        _is_complete: Whether workflow has completed
        _success: Whether workflow completed successfully
    """

    # Core data
    data: Dict[str, Any]
    messages: List[Dict[str, Any]]

    # Execution metadata
    _workflow_id: str
    _workflow_name: str
    _current_node: str
    _node_results: Dict[str, Any]
    _error: Optional[str]

    # Iteration tracking
    _iteration: int
    _visited_nodes: List[str]

    # Parallel execution
    _parallel_results: Dict[str, Any]

    # Human-in-the-loop
    _hitl_pending: bool
    _hitl_response: Optional[Dict[str, Any]]

    # Temporal context
    _as_of_date: Optional[str]
    _lookback_periods: Optional[int]
    _include_end_date: bool

    # Completion tracking
    _is_complete: bool
    _success: bool


# =============================================================================
# Context Factory Functions
# =============================================================================


def create_execution_context(
    initial_data: Optional[Dict[str, Any]] = None,
    workflow_id: Optional[str] = None,
    workflow_name: Optional[str] = None,
) -> ExecutionContext:
    """Create a new ExecutionContext with sensible defaults.

    Args:
        initial_data: Initial data to populate the context
        workflow_id: Optional workflow ID (generated if not provided)
        workflow_name: Optional workflow name

    Returns:
        New ExecutionContext with defaults set

    Example:
        ctx = create_execution_context(
            initial_data={"input": "test"},
            workflow_name="my_workflow",
        )
    """
    ctx: ExecutionContext = {
        "data": initial_data or {},
        "messages": [],
        "_workflow_id": workflow_id or str(uuid.uuid4()),
        "_workflow_name": workflow_name or "",
        "_current_node": "",
        "_node_results": {},
        "_error": None,
        "_iteration": 0,
        "_visited_nodes": [],
        "_parallel_results": {},
        "_hitl_pending": False,
        "_hitl_response": None,
        "_as_of_date": None,
        "_lookback_periods": None,
        "_include_end_date": True,
        "_is_complete": False,
        "_success": False,
    }
    return ctx


# =============================================================================
# Context Wrapper Class (for method-based access)
# =============================================================================


@dataclass
class ExecutionContextWrapper:
    """Wrapper providing method-based access to ExecutionContext.

    This class now delegates to WorkflowStateManager for actual state management,
    replacing the previous direct dict-based approach.

    MIGRATION NOTICE: This wrapper is deprecated. Use WorkflowStateManager directly
    for new code. For unified state access across all scopes, use get_global_manager().

    Example:
        # OLD (deprecated):
        ctx = create_execution_context({"input": "test"})
        wrapper = ExecutionContextWrapper(ctx)
        wrapper.set("output", "result")

        # NEW (recommended):
        from victor.state import WorkflowStateManager
        mgr = WorkflowStateManager()
        await mgr.set("output", "result")
        value = await mgr.get("output")

        # OR for unified access:
        from victor.state import get_global_manager, StateScope
        state = get_global_manager()
        await state.set("output", "result", scope=StateScope.WORKFLOW)
    """

    state: ExecutionContext = field(default_factory=create_execution_context)
    _manager: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize WorkflowStateManager on first use."""
        from victor.state.managers import WorkflowStateManager

        if self._manager is None:
            self._manager = WorkflowStateManager()
            # Initialize with existing data if present
            if "data" in self.state and self.state["data"]:
                # Note: We can't use async here in __post_init__,
                # so we'll lazy-load on first access
                pass

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value from data.

        DEPRECATED: Use WorkflowStateManager.get() instead.
        """
        # Check manager first
        try:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in async context, but this is a sync method
                    # Fall back to direct state access
                    pass
                else:
                    # We can run async code
                    try:
                        return asyncio.run(self._manager.get(key, default))
                    except RuntimeError:
                        pass
            except RuntimeError:
                pass
        except Exception:
            pass

        # Fall back to direct state access
        return self.state.get("data", {}).get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a context value in data.

        DEPRECATED: Use WorkflowStateManager.set() instead.
        """
        # Update both manager and state for compatibility
        try:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in async context
                    pass
                else:
                    # We can run async code
                    try:
                        asyncio.run(self._manager.set(key, value))
                    except RuntimeError:
                        pass
            except RuntimeError:
                pass
        except Exception:
            pass

        # Update state dict
        if "data" not in self.state:
            self.state["data"] = {}
        self.state["data"][key] = value

    def update(self, values: Dict[str, Any]) -> None:
        """Update multiple context values in data.

        DEPRECATED: Use WorkflowStateManager.update() instead.
        """
        # Update both manager and state for compatibility
        try:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    try:
                        asyncio.run(self._manager.update(values))
                    except RuntimeError:
                        pass
            except RuntimeError:
                pass
        except Exception:
            pass

        # Update state dict
        if "data" not in self.state:
            self.state["data"] = {}
        self.state["data"].update(values)

    def get_result(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get result for a specific node."""
        return self.state.get("_node_results", {}).get(node_id)

    def add_result(self, node_id: str, result: Dict[str, Any]) -> None:
        """Add a node result."""
        if "_node_results" not in self.state:
            self.state["_node_results"] = {}
        self.state["_node_results"][node_id] = result

    def has_failures(self) -> bool:
        """Check if any nodes failed."""
        results = self.state.get("_node_results", {})
        return any(
            r.get("status") == "failed" or not r.get("success", True)
            for r in results.values()
            if isinstance(r, dict)
        )

    def get_outputs(self) -> Dict[str, Any]:
        """Get all successful node outputs."""
        results = self.state.get("_node_results", {})
        return {
            node_id: result.get("output")
            for node_id, result in results.items()
            if isinstance(result, dict)
            and result.get("success", True)
            and result.get("output") is not None
        }

    @property
    def workflow_id(self) -> str:
        """Get the workflow ID."""
        return self.state.get("_workflow_id", "")

    @property
    def current_node(self) -> str:
        """Get the current node."""
        return self.state.get("_current_node", "")

    @current_node.setter
    def current_node(self, value: str) -> None:
        """Set the current node."""
        self.state["_current_node"] = value

    @property
    def is_complete(self) -> bool:
        """Check if workflow is complete."""
        return self.state.get("_is_complete", False)

    @is_complete.setter
    def is_complete(self, value: bool) -> None:
        """Set completion status."""
        self.state["_is_complete"] = value

    @property
    def error(self) -> Optional[str]:
        """Get error message."""
        return self.state.get("_error")

    @error.setter
    def error(self, value: Optional[str]) -> None:
        """Set error message."""
        self.state["_error"] = value


# =============================================================================
# Adapter Functions (for backward compatibility)
# =============================================================================


def from_workflow_context(ctx: "WorkflowContext") -> ExecutionContext:
    """Convert legacy WorkflowContext to ExecutionContext.

    Args:
        ctx: Legacy WorkflowContext dataclass

    Returns:
        ExecutionContext with data migrated from WorkflowContext
    """
    from victor.workflows.executor import WorkflowContext

    # Convert node results to dict format
    node_results: Dict[str, Any] = {}
    for node_id, result in ctx.node_results.items():
        node_results[node_id] = {
            "node_id": result.node_id,
            "status": (
                result.status.value if hasattr(result.status, "value") else str(result.status)
            ),
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "duration_seconds": result.duration_seconds,
            "tool_calls_used": result.tool_calls_used,
        }

    # Build temporal context if present
    as_of_date = None
    lookback_periods = None
    include_end_date = True
    if ctx.temporal:
        as_of_date = ctx.temporal.as_of_date
        lookback_periods = ctx.temporal.lookback_periods
        include_end_date = ctx.temporal.include_end_date

    return {
        "data": ctx.data.copy(),
        "messages": [],
        "_workflow_id": ctx.metadata.get("workflow_id", str(uuid.uuid4())),
        "_workflow_name": ctx.metadata.get("workflow_name", ""),
        "_current_node": ctx.metadata.get("current_node", ""),
        "_node_results": node_results,
        "_error": None,
        "_iteration": ctx.metadata.get("iteration", 0),
        "_visited_nodes": list(ctx.node_results.keys()),
        "_parallel_results": {},
        "_hitl_pending": False,
        "_hitl_response": None,
        "_as_of_date": as_of_date,
        "_lookback_periods": lookback_periods,
        "_include_end_date": include_end_date,
        "_is_complete": False,
        "_success": not ctx.has_failures(),
    }


def to_workflow_context(ctx: ExecutionContext) -> "WorkflowContext":
    """Convert ExecutionContext to legacy WorkflowContext.

    Args:
        ctx: ExecutionContext TypedDict

    Returns:
        WorkflowContext dataclass for use with legacy code
    """
    from victor.workflows.executor import (
        WorkflowContext,
        NodeResult,
        ExecutorNodeStatus,
        TemporalContext,
    )

    # Convert node results from dict to NodeResult
    node_results: Dict[str, NodeResult] = {}
    for node_id, result in ctx.get("_node_results", {}).items():
        if isinstance(result, dict):
            # Map status string to enum
            status_str = result.get("status", "completed")
            try:
                status = ExecutorNodeStatus(status_str)
            except ValueError:
                status = (
                    ExecutorNodeStatus.COMPLETED
                    if result.get("success", True)
                    else ExecutorNodeStatus.FAILED
                )

            node_results[node_id] = NodeResult(
                node_id=node_id,
                status=status,
                output=result.get("output"),
                error=result.get("error"),
                duration_seconds=result.get("duration_seconds", 0.0),
                tool_calls_used=result.get("tool_calls_used", 0),
            )

    # Build temporal context if present
    temporal = None
    if ctx.get("_as_of_date"):
        lookback = ctx.get("_lookback_periods")
        temporal = TemporalContext(
            as_of_date=ctx.get("_as_of_date"),
            lookback_periods=lookback if lookback is not None else 1,
            include_end_date=ctx.get("_include_end_date", True),
        )

    # Build metadata
    metadata = {
        "workflow_id": ctx.get("_workflow_id", ""),
        "workflow_name": ctx.get("_workflow_name", ""),
        "current_node": ctx.get("_current_node", ""),
        "iteration": ctx.get("_iteration", 0),
    }

    return WorkflowContext(
        data=ctx.get("data", {}).copy(),
        node_results=node_results,
        metadata=metadata,
        temporal=temporal,
    )


def from_compiler_workflow_state(state: Dict[str, Any]) -> ExecutionContext:
    """Convert yaml_to_graph_compiler WorkflowState to ExecutionContext.

    Args:
        state: WorkflowState dict from yaml_to_graph_compiler

    Returns:
        ExecutionContext with data migrated
    """
    # Extract system fields and user data
    system_keys = {
        "_workflow_id",
        "_current_node",
        "_node_results",
        "_error",
        "_iteration",
        "_parallel_results",
        "_hitl_pending",
        "_hitl_response",
    }

    # User data is everything not in system keys
    user_data = {k: v for k, v in state.items() if k not in system_keys}

    return {
        "data": user_data,
        "messages": [],
        "_workflow_id": state.get("_workflow_id", str(uuid.uuid4())),
        "_workflow_name": "",
        "_current_node": state.get("_current_node", ""),
        "_node_results": state.get("_node_results", {}),
        "_error": state.get("_error"),
        "_iteration": state.get("_iteration", 0),
        "_visited_nodes": list(state.get("_node_results", {}).keys()),
        "_parallel_results": state.get("_parallel_results", {}),
        "_hitl_pending": state.get("_hitl_pending", False),
        "_hitl_response": state.get("_hitl_response"),
        "_as_of_date": None,
        "_lookback_periods": None,
        "_include_end_date": True,
        "_is_complete": False,
        "_success": state.get("_error") is None,
    }


def to_compiler_workflow_state(ctx: ExecutionContext) -> Dict[str, Any]:
    """Convert ExecutionContext to yaml_to_graph_compiler WorkflowState format.

    Args:
        ctx: ExecutionContext

    Returns:
        Dict compatible with yaml_to_graph_compiler WorkflowState
    """
    # Start with user data
    state: Dict[str, Any] = ctx.get("data", {}).copy()

    # Add system fields
    state["_workflow_id"] = ctx.get("_workflow_id", "")
    state["_current_node"] = ctx.get("_current_node", "")
    state["_node_results"] = ctx.get("_node_results", {})
    state["_error"] = ctx.get("_error")
    state["_iteration"] = ctx.get("_iteration", 0)
    state["_parallel_results"] = ctx.get("_parallel_results", {})
    state["_hitl_pending"] = ctx.get("_hitl_pending", False)
    state["_hitl_response"] = ctx.get("_hitl_response")

    return state


def from_adapter_workflow_state(state: Dict[str, Any]) -> ExecutionContext:
    """Convert adapters.py WorkflowState to ExecutionContext.

    Args:
        state: WorkflowState dict from adapters.py

    Returns:
        ExecutionContext with data migrated
    """
    return {
        "data": state.get("context", {}),
        "messages": state.get("messages", []),
        "_workflow_id": str(uuid.uuid4()),
        "_workflow_name": "",
        "_current_node": state.get("current_node", ""),
        "_node_results": state.get("results", {}),
        "_error": state.get("error"),
        "_iteration": 0,
        "_visited_nodes": state.get("visited_nodes", []),
        "_parallel_results": {},
        "_hitl_pending": False,
        "_hitl_response": None,
        "_as_of_date": None,
        "_lookback_periods": None,
        "_include_end_date": True,
        "_is_complete": state.get("is_complete", False),
        "_success": state.get("error") is None,
    }


def to_adapter_workflow_state(ctx: ExecutionContext) -> Dict[str, Any]:
    """Convert ExecutionContext to adapters.py WorkflowState format.

    Args:
        ctx: ExecutionContext

    Returns:
        Dict compatible with adapters.py WorkflowState
    """
    return {
        "context": ctx.get("data", {}),
        "messages": ctx.get("messages", []),
        "current_node": ctx.get("_current_node", ""),
        "visited_nodes": ctx.get("_visited_nodes", []),
        "results": ctx.get("_node_results", {}),
        "error": ctx.get("_error"),
        "is_complete": ctx.get("_is_complete", False),
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core type
    "ExecutionContext",
    # Factory
    "create_execution_context",
    # Wrapper class
    "ExecutionContextWrapper",
    # Adapters
    "from_workflow_context",
    "to_workflow_context",
    "from_compiler_workflow_state",
    "to_compiler_workflow_state",
    "from_adapter_workflow_state",
    "to_adapter_workflow_state",
]
