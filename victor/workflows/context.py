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

This module now uses Pydantic models for type-safe state management,
replacing the legacy TypedDict-based approach.

Migration Example:
    # OLD (TypedDict - deprecated):
    from victor.workflows.context import ExecutionContext, create_execution_context
    ctx = create_execution_context({"input": "test"})

    # NEW (Pydantic - recommended):
    from victor.workflows.models import WorkflowExecutionContextModel
    from victor.workflows.models.adapters import WorkflowExecutionContextAdapter
    ctx = WorkflowExecutionContextAdapter.create_initial(
        workflow_name="my_workflow",
        initial_data={"input": "test"}
    )

    # Access is the same
    input_val = ctx.data["input"]

    # Update state
    ctx.current_node = "process"
    ctx.add_node_result("start", {"success": True})
"""

from __future__ import annotations

import asyncio
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
    Union,
)

from victor.core.async_utils import run_sync
from victor.workflows.models import WorkflowExecutionContextModel
from victor.workflows.models.adapters import WorkflowExecutionContextAdapter

if TYPE_CHECKING:
    from victor.workflows.executor import NodeResult, TemporalContext, WorkflowContext

logger = logging.getLogger(__name__)


# =============================================================================
# Type Aliases for Backward Compatibility
# =============================================================================

# Type alias for backward compatibility
ExecutionContext = WorkflowExecutionContextModel
# =============================================================================


def create_execution_context(
    initial_data: Optional[Dict[str, Any]] = None,
    workflow_id: Optional[str] = None,
    workflow_name: Optional[str] = None,
) -> WorkflowExecutionContextModel:
    """Create a new ExecutionContext with sensible defaults.

    Args:
        initial_data: Initial data to populate the context
        workflow_id: Optional workflow ID (generated if not provided)
        workflow_name: Optional workflow name

    Returns:
        New WorkflowExecutionContextModel with defaults set

    Example:
        ctx = create_execution_context(
            initial_data={"input": "test"},
            workflow_name="my_workflow",
        )
    """
    return WorkflowExecutionContextAdapter.create_initial(
        workflow_id=workflow_id,
        workflow_name=workflow_name or "",
        initial_data=initial_data,
    )


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

    def __post_init__(self):
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
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return run_sync(self._manager.get(key, default))
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
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                run_sync(self._manager.set(key, value))
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
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                run_sync(self._manager.update(values))
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

    return _build_execution_context(
        initial_data=ctx.data,
        workflow_id=ctx.metadata.get("workflow_id", str(uuid.uuid4())),
        workflow_name=ctx.metadata.get("workflow_name", ""),
        current_node=ctx.metadata.get("current_node", ""),
        node_results=node_results,
        iteration=ctx.metadata.get("iteration", 0),
        visited_nodes=list(ctx.node_results.keys()),
        as_of_date=as_of_date,
        lookback_periods=lookback_periods,
        include_end_date=include_end_date,
        success=not ctx.has_failures(),
    )


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
        temporal = TemporalContext(
            as_of_date=ctx.get("_as_of_date"),
            lookback_periods=ctx.get("_lookback_periods"),
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


def from_compiler_workflow_state(state: Dict[str, Any]) -> WorkflowExecutionContextModel:
    """Convert compiled runtime WorkflowState to ExecutionContext.

    Args:
        state: WorkflowState dict from compiled workflow runtime

    Returns:
        WorkflowExecutionContextModel with data migrated
    """
    # Extract system fields and user data
    system_keys = {
        "_workflow_id",
        "_workflow_name",
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

    model = WorkflowExecutionContextAdapter.create_initial(
        workflow_id=state.get("_workflow_id"),
        workflow_name=state.get("_workflow_name", ""),
        initial_data=user_data,
    )

    # Set additional fields
    if state.get("_current_node"):
        model.current_node = state.get("_current_node", "")
    if state.get("_node_results"):
        model.node_results = state.get("_node_results", {})
    if state.get("_error"):
        model.error = state.get("_error")
    if state.get("_iteration"):
        model.iteration = state.get("_iteration", 0)
    if state.get("_parallel_results"):
        model.parallel_results = state.get("_parallel_results", {})
    if state.get("_hitl_pending"):
        model.hitl_pending = state.get("_hitl_pending", False)
    if state.get("_hitl_response"):
        model.hitl_response = state.get("_hitl_response")

    # Populate visited_nodes from node_results keys
    if state.get("_node_results"):
        for node_id in state.get("_node_results", {}).keys():
            model.visit_node(node_id)

    return model


def to_compiler_workflow_state(ctx: WorkflowExecutionContextModel) -> Dict[str, Any]:
    """Convert ExecutionContext to compiled runtime WorkflowState format.

    Args:
        ctx: WorkflowExecutionContextModel

    Returns:
        Dict compatible with compiled runtime WorkflowState
    """
    # Get the base dict from the model
    state = ctx.to_dict()

    # Merge user data to top level (for backward compatibility)
    if "data" in state and state["data"]:
        state.update(state["data"])
        del state["data"]

    return state


def from_adapter_workflow_state(state: Dict[str, Any]) -> WorkflowExecutionContextModel:
    """Convert adapters.py WorkflowState to ExecutionContext.

    Args:
        state: WorkflowState dict from adapters.py

    Returns:
        WorkflowExecutionContextModel with data migrated
    """
    return WorkflowExecutionContextAdapter.create_initial(
        workflow_id=str(uuid.uuid4()),
        current_node=state.get("current_node", ""),
        initial_data=state.get("context", {}),
    )


def to_adapter_workflow_state(ctx: WorkflowExecutionContextModel) -> Dict[str, Any]:
    """Convert ExecutionContext to adapters.py WorkflowState format.

    Args:
        ctx: WorkflowExecutionContextModel

    Returns:
        Dict compatible with adapters.py WorkflowState
    """
    return {
        "context": ctx.data,
        "messages": ctx.messages,
        "current_node": ctx.current_node,
        "visited_nodes": ctx.visited_nodes,
        "results": ctx.node_results,
        "error": ctx.error,
        "is_complete": ctx.is_complete,
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
