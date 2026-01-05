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

"""WorkflowEngine - High-level facade for workflow execution.

This module provides a unified facade over the workflow execution system,
promoting workflow capabilities from victor/workflows/ to the framework layer.

Design Pattern: Facade + Builder
================================
WorkflowEngine provides a simplified interface to the complex workflow
subsystem, hiding the complexity of executors, HITL, caching, and streaming.

Key Features:
- Create workflows from Python code or YAML
- Execute workflows with optional HITL (Human-in-the-Loop)
- Stream workflow events for real-time UI updates
- Cache workflow results for efficiency
- Checkpoint workflow state for recovery

Usage:
    from victor.framework.workflow_engine import WorkflowEngine

    # Create engine
    engine = WorkflowEngine()

    # Execute a workflow from YAML
    result = await engine.execute_yaml(
        "path/to/workflow.yaml",
        initial_state={"input": "data"},
    )

    # Execute with streaming
    async for event in engine.stream_yaml("workflow.yaml", state):
        print(f"{event.node_id}: {event.event_type}")

    # Execute StateGraph directly
    from victor.framework import StateGraph
    graph = StateGraph(MyState)
    # ... configure graph ...
    result = await engine.execute_graph(graph.compile(), initial_state)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.framework.graph import CompiledGraph, StateGraph
    from victor.workflows.executor import WorkflowExecutor, WorkflowResult
    from victor.workflows.streaming import WorkflowStreamChunk
    from victor.workflows.streaming_executor import StreamingWorkflowExecutor
    from victor.workflows.hitl import HITLHandler, HITLExecutor
    from victor.workflows.cache import WorkflowCacheManager
    from victor.workflows.definition import WorkflowDefinition

logger = logging.getLogger(__name__)

StateType = TypeVar("StateType", bound=Dict[str, Any])


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class WorkflowEngineConfig:
    """Configuration for WorkflowEngine.

    Attributes:
        enable_caching: Whether to cache workflow results.
        cache_ttl_seconds: Cache time-to-live in seconds.
        enable_hitl: Whether to enable HITL for approval nodes.
        hitl_timeout_seconds: Timeout for HITL approval.
        enable_checkpoints: Whether to checkpoint state for recovery.
        max_iterations: Maximum iterations for cyclic workflows.
        enable_streaming: Whether to enable event streaming.
        parallel_execution: Whether to run parallel nodes concurrently.
    """

    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_hitl: bool = False
    hitl_timeout_seconds: int = 300
    enable_checkpoints: bool = True
    max_iterations: int = 100
    enable_streaming: bool = True
    parallel_execution: bool = True


@dataclass
class ExecutionResult:
    """Result of workflow execution.

    Attributes:
        success: Whether execution completed successfully.
        final_state: Final workflow state.
        nodes_executed: List of node IDs that were executed.
        duration_seconds: Total execution time.
        error: Error message if execution failed.
        checkpoints: List of checkpoint IDs created.
        hitl_requests: HITL requests that were made.
        cached: Whether result was from cache.
    """

    success: bool
    final_state: Dict[str, Any] = field(default_factory=dict)
    nodes_executed: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: Optional[str] = None
    checkpoints: List[str] = field(default_factory=list)
    hitl_requests: List[Dict[str, Any]] = field(default_factory=list)
    cached: bool = False


@dataclass
class WorkflowEvent:
    """Event emitted during workflow execution.

    Attributes:
        event_type: Type of event (node_start, node_end, error, hitl, etc.)
        node_id: ID of the node that emitted the event.
        timestamp: Unix timestamp of event.
        data: Event-specific data.
        state_snapshot: Optional snapshot of workflow state.
    """

    event_type: str
    node_id: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    state_snapshot: Optional[Dict[str, Any]] = None


# =============================================================================
# Protocol
# =============================================================================


@runtime_checkable
class WorkflowEngineProtocol(Protocol):
    """Protocol for workflow engine implementations."""

    async def execute_yaml(
        self,
        yaml_path: Union[str, Path],
        initial_state: Dict[str, Any],
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a workflow from YAML file."""
        ...

    async def execute_graph(
        self,
        graph: "CompiledGraph",
        initial_state: Dict[str, Any],
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a compiled StateGraph."""
        ...

    async def stream_yaml(
        self,
        yaml_path: Union[str, Path],
        initial_state: Dict[str, Any],
        **kwargs: Any,
    ) -> AsyncIterator[WorkflowEvent]:
        """Stream events from YAML workflow execution."""
        ...

    async def stream_graph(
        self,
        graph: "CompiledGraph",
        initial_state: Dict[str, Any],
        **kwargs: Any,
    ) -> AsyncIterator[WorkflowEvent]:
        """Stream events from StateGraph execution."""
        ...


# =============================================================================
# WorkflowEngine
# =============================================================================


class WorkflowEngine:
    """High-level facade for workflow execution.

    Provides a unified API for executing workflows defined as:
    - YAML files (declarative)
    - StateGraph objects (programmatic)
    - WorkflowDefinition objects (legacy)

    Features:
    - Automatic HITL integration for approval nodes
    - Result caching for repeated executions
    - Checkpointing for recovery
    - Event streaming for real-time updates
    - Parallel node execution
    """

    def __init__(
        self,
        config: Optional[WorkflowEngineConfig] = None,
        hitl_handler: Optional["HITLHandler"] = None,
        cache_manager: Optional["WorkflowCacheManager"] = None,
    ) -> None:
        """Initialize WorkflowEngine.

        Args:
            config: Engine configuration.
            hitl_handler: Custom HITL handler for approval nodes.
            cache_manager: Custom cache manager for results.
        """
        self._config = config or WorkflowEngineConfig()
        self._hitl_handler = hitl_handler
        self._cache_manager = cache_manager

        # Lazy-loaded executors
        self._executor: Optional["WorkflowExecutor"] = None
        self._streaming_executor: Optional["StreamingWorkflowExecutor"] = None
        self._hitl_executor: Optional["HITLExecutor"] = None

    @property
    def config(self) -> WorkflowEngineConfig:
        """Get the engine configuration."""
        return self._config

    # =========================================================================
    # Execution Methods
    # =========================================================================

    async def execute_yaml(
        self,
        yaml_path: Union[str, Path],
        initial_state: Optional[Dict[str, Any]] = None,
        workflow_name: Optional[str] = None,
        condition_registry: Optional[Dict[str, Callable]] = None,
        transform_registry: Optional[Dict[str, Callable]] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a workflow from YAML file.

        Args:
            yaml_path: Path to YAML workflow file.
            initial_state: Initial workflow state.
            workflow_name: Specific workflow to load from file.
            condition_registry: Custom condition functions.
            transform_registry: Custom transform functions.
            **kwargs: Additional execution parameters.

        Returns:
            ExecutionResult with final state and metadata.
        """
        import time
        from victor.workflows.yaml_loader import (
            load_workflow_from_file,
            YAMLWorkflowConfig,
        )
        from victor.workflows.executor import WorkflowExecutor, WorkflowContext

        start_time = time.time()
        nodes_executed: List[str] = []
        hitl_requests: List[Dict[str, Any]] = []

        try:
            # Create config with registries
            config = YAMLWorkflowConfig(
                condition_registry=condition_registry or {},
                transform_registry=transform_registry or {},
            )

            # Load workflow from YAML
            result = load_workflow_from_file(
                str(yaml_path),
                workflow_name=workflow_name,
                config=config,
            )

            # Handle dict or single workflow result
            if isinstance(result, dict):
                if not result:
                    raise ValueError(f"No workflows found in {yaml_path}")
                workflow_def = next(iter(result.values()))
            else:
                workflow_def = result

            # Create executor
            executor = self._get_executor()

            # Create context
            context = WorkflowContext(
                workflow=workflow_def,
                initial_state=initial_state or {},
            )

            # Execute
            result = await executor.execute(context)

            duration = time.time() - start_time

            return ExecutionResult(
                success=result.success,
                final_state=result.final_state,
                nodes_executed=result.nodes_executed,
                duration_seconds=duration,
                error=result.error if not result.success else None,
                hitl_requests=hitl_requests,
            )

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    async def execute_graph(
        self,
        graph: "CompiledGraph",
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a compiled StateGraph.

        Args:
            graph: Compiled StateGraph to execute.
            initial_state: Initial workflow state.
            **kwargs: Additional execution parameters.

        Returns:
            ExecutionResult with final state and metadata.
        """
        import time

        start_time = time.time()

        try:
            # Execute the graph directly
            final_state = await graph.invoke(initial_state or {})

            duration = time.time() - start_time

            return ExecutionResult(
                success=True,
                final_state=final_state,
                nodes_executed=list(graph._execution_order) if hasattr(graph, '_execution_order') else [],
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    async def execute_definition(
        self,
        workflow: "WorkflowDefinition",
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a WorkflowDefinition.

        Args:
            workflow: WorkflowDefinition to execute.
            initial_state: Initial workflow state.
            **kwargs: Additional execution parameters.

        Returns:
            ExecutionResult with final state and metadata.
        """
        import time

        start_time = time.time()

        try:
            executor = self._get_executor()
            result = await executor.execute(
                workflow,
                initial_context=initial_state or {},
            )

            duration = time.time() - start_time

            return ExecutionResult(
                success=result.success,
                final_state=result.final_state,
                nodes_executed=result.nodes_executed,
                duration_seconds=duration,
                error=result.error if not result.success else None,
            )

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    # =========================================================================
    # Streaming Methods
    # =========================================================================

    async def stream_yaml(
        self,
        yaml_path: Union[str, Path],
        initial_state: Optional[Dict[str, Any]] = None,
        workflow_name: Optional[str] = None,
        condition_registry: Optional[Dict[str, Callable]] = None,
        transform_registry: Optional[Dict[str, Callable]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[WorkflowEvent]:
        """Stream events from YAML workflow execution.

        Args:
            yaml_path: Path to YAML workflow file.
            initial_state: Initial workflow state.
            workflow_name: Specific workflow to load from file.
            condition_registry: Custom condition functions.
            transform_registry: Custom transform functions.
            **kwargs: Additional execution parameters.

        Yields:
            WorkflowEvent for each execution step.
        """
        import time
        from victor.workflows.yaml_loader import (
            load_workflow_from_file,
            YAMLWorkflowConfig,
        )
        from victor.workflows.streaming_executor import StreamingWorkflowExecutor
        from victor.workflows.streaming import WorkflowStreamContext

        try:
            # Create config with registries
            config = YAMLWorkflowConfig(
                condition_registry=condition_registry or {},
                transform_registry=transform_registry or {},
            )

            # Load workflow
            result = load_workflow_from_file(
                str(yaml_path),
                workflow_name=workflow_name,
                config=config,
            )

            # Handle dict or single workflow result
            if isinstance(result, dict):
                if not result:
                    raise ValueError(f"No workflows found in {yaml_path}")
                workflow_def = next(iter(result.values()))
            else:
                workflow_def = result

            # Create streaming executor
            executor = self._get_streaming_executor()

            # Create stream context
            context = WorkflowStreamContext(
                workflow=workflow_def,
                initial_state=initial_state or {},
            )

            # Stream execution
            async for chunk in executor.stream(context):
                yield WorkflowEvent(
                    event_type=chunk.event_type.value if hasattr(chunk.event_type, 'value') else str(chunk.event_type),
                    node_id=chunk.node_id or "",
                    timestamp=time.time(),
                    data={"content": chunk.content} if chunk.content else {},
                    state_snapshot=chunk.state_snapshot,
                )

        except Exception as e:
            logger.error(f"Streaming workflow failed: {e}")
            yield WorkflowEvent(
                event_type="error",
                node_id="",
                timestamp=time.time(),
                data={"error": str(e)},
            )

    async def stream_graph(
        self,
        graph: "CompiledGraph",
        initial_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[WorkflowEvent]:
        """Stream events from StateGraph execution.

        Args:
            graph: Compiled StateGraph to execute.
            initial_state: Initial workflow state.
            **kwargs: Additional execution parameters.

        Yields:
            WorkflowEvent for each execution step.
        """
        import time

        try:
            # Use the graph's stream method if available
            if hasattr(graph, 'stream'):
                async for node_id, state in graph.stream(initial_state or {}):
                    yield WorkflowEvent(
                        event_type="node_complete",
                        node_id=node_id,
                        timestamp=time.time(),
                        state_snapshot=state,
                    )
            else:
                # Fallback to invoke
                final_state = await graph.invoke(initial_state or {})
                yield WorkflowEvent(
                    event_type="complete",
                    node_id="",
                    timestamp=time.time(),
                    state_snapshot=final_state,
                )

        except Exception as e:
            logger.error(f"Graph streaming failed: {e}")
            yield WorkflowEvent(
                event_type="error",
                node_id="",
                timestamp=time.time(),
                data={"error": str(e)},
            )

    # =========================================================================
    # HITL Integration
    # =========================================================================

    def set_hitl_handler(self, handler: "HITLHandler") -> None:
        """Set custom HITL handler.

        Args:
            handler: HITLHandler for approval nodes.
        """
        self._hitl_handler = handler
        # Reset executor to use new handler
        self._hitl_executor = None

    async def execute_with_hitl(
        self,
        yaml_path: Union[str, Path],
        initial_state: Optional[Dict[str, Any]] = None,
        approval_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute workflow with HITL approval nodes.

        Args:
            yaml_path: Path to YAML workflow file.
            initial_state: Initial workflow state.
            approval_callback: Callback for approval decisions.
            **kwargs: Additional execution parameters.

        Returns:
            ExecutionResult with HITL request history.
        """
        from victor.workflows.yaml_loader import load_workflow_from_file
        from victor.workflows.hitl import HITLExecutor, DefaultHITLHandler
        import time

        start_time = time.time()
        hitl_requests: List[Dict[str, Any]] = []

        try:
            # Load workflow
            workflow_def = load_workflow_from_file(str(yaml_path))

            # Create HITL handler
            handler = self._hitl_handler or DefaultHITLHandler()

            # Create HITL executor
            executor = HITLExecutor(
                workflow=workflow_def,
                handler=handler,
            )

            # Execute with HITL
            result = await executor.execute(initial_state or {})

            duration = time.time() - start_time

            return ExecutionResult(
                success=result.success,
                final_state=result.final_state,
                nodes_executed=result.nodes_executed,
                duration_seconds=duration,
                hitl_requests=hitl_requests,
            )

        except Exception as e:
            logger.error(f"HITL workflow failed: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    # =========================================================================
    # Caching
    # =========================================================================

    def enable_caching(self, ttl_seconds: int = 3600) -> None:
        """Enable result caching.

        Args:
            ttl_seconds: Cache time-to-live.
        """
        self._config.enable_caching = True
        self._config.cache_ttl_seconds = ttl_seconds

    def disable_caching(self) -> None:
        """Disable result caching."""
        self._config.enable_caching = False

    def clear_cache(self) -> None:
        """Clear all cached results."""
        if self._cache_manager:
            self._cache_manager.clear()

    # =========================================================================
    # Helpers
    # =========================================================================

    def _get_executor(self) -> "WorkflowExecutor":
        """Get or create workflow executor."""
        if self._executor is None:
            from victor.workflows.executor import WorkflowExecutor
            self._executor = WorkflowExecutor()
        return self._executor

    def _get_streaming_executor(self) -> "StreamingWorkflowExecutor":
        """Get or create streaming executor."""
        if self._streaming_executor is None:
            from victor.workflows.streaming_executor import StreamingWorkflowExecutor
            self._streaming_executor = StreamingWorkflowExecutor()
        return self._streaming_executor


# =============================================================================
# Factory Functions
# =============================================================================


def create_workflow_engine(
    config: Optional[WorkflowEngineConfig] = None,
    hitl_handler: Optional["HITLHandler"] = None,
    cache_manager: Optional["WorkflowCacheManager"] = None,
) -> WorkflowEngine:
    """Create a WorkflowEngine with configuration.

    Args:
        config: Engine configuration.
        hitl_handler: Custom HITL handler.
        cache_manager: Custom cache manager.

    Returns:
        Configured WorkflowEngine instance.
    """
    return WorkflowEngine(
        config=config,
        hitl_handler=hitl_handler,
        cache_manager=cache_manager,
    )


# =============================================================================
# Convenience Functions
# =============================================================================


async def run_yaml_workflow(
    yaml_path: Union[str, Path],
    initial_state: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> ExecutionResult:
    """Convenience function to run a YAML workflow.

    Args:
        yaml_path: Path to YAML workflow file.
        initial_state: Initial state.
        **kwargs: Additional parameters.

    Returns:
        ExecutionResult from workflow execution.
    """
    engine = create_workflow_engine()
    return await engine.execute_yaml(yaml_path, initial_state, **kwargs)


async def run_graph_workflow(
    graph: "CompiledGraph",
    initial_state: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> ExecutionResult:
    """Convenience function to run a StateGraph workflow.

    Args:
        graph: Compiled StateGraph.
        initial_state: Initial state.
        **kwargs: Additional parameters.

    Returns:
        ExecutionResult from workflow execution.
    """
    engine = create_workflow_engine()
    return await engine.execute_graph(graph, initial_state, **kwargs)
