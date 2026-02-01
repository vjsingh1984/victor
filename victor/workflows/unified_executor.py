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

"""Victor StateGraph Workflow Executor.

Victor's native workflow execution engine based on StateGraph, providing:
- LangGraph-compatible API for seamless migration
- TypedDict state schemas for type-safe workflow state
- Checkpointing for resumable long-running workflows
- YAML DSL that maps directly to StateGraph patterns
- Streaming execution with real-time state updates

Example:
    from victor.workflows import StateGraphExecutor

    # Execute workflow
    executor = StateGraphExecutor(orchestrator)
    result = await executor.execute(workflow, {"input": "data"})

    # With checkpointing for resumable workflows
    result = await executor.execute(
        workflow,
        {"input": "data"},
        thread_id="session-123",
    )

    # Streaming execution
    async for node_id, state in executor.stream(workflow, {"input": "data"}):
        print(f"Node {node_id} completed: {state}")
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
)
from collections.abc import AsyncIterator

if TYPE_CHECKING:
    from victor.protocols import WorkflowAgentProtocol
    from victor.tools.registry import ToolRegistry
    from victor.workflows.definition import WorkflowDefinition
    from victor.workflows.yaml_to_graph_compiler import (
        YAMLToStateGraphCompiler,
    )
    from victor.framework.graph import CheckpointerProtocol

from victor.workflows.recursion import RecursionContext

logger = logging.getLogger(__name__)


@dataclass
class ExecutorConfig:
    """Configuration for Victor's StateGraph workflow execution.

    Attributes:
        enable_checkpointing: Enable checkpointing for resumable workflows
        max_iterations: Max loop iterations for cyclic workflows
        timeout: Execution timeout in seconds
        interrupt_nodes: Nodes to interrupt before (for human-in-the-loop)
        max_recursion_depth: Maximum recursion depth for nested execution (default: 3)
    """

    enable_checkpointing: bool = True
    max_iterations: int = 25
    timeout: Optional[float] = None
    interrupt_nodes: list[str] = field(default_factory=list)
    default_profile: Optional[str] = None  # Default profile for nodes without explicit profile
    max_recursion_depth: int = 3


@dataclass
class ExecutorResult:
    """Result from Victor StateGraph workflow execution.

    Attributes:
        success: Whether execution succeeded
        state: Final state/context data
        error: Error message if failed
        duration_seconds: Total execution time
        nodes_executed: List of node IDs that were executed
        iterations: Number of iterations (for cyclic workflows)
        checkpoints_saved: Number of checkpoints saved
        interrupted: Whether execution was interrupted (for HITL)
        interrupt_node: Node where execution was interrupted
    """

    success: bool
    state: dict[str, Any]
    error: Optional[str] = None
    duration_seconds: float = 0.0
    nodes_executed: list[str] = field(default_factory=list)
    iterations: int = 0
    checkpoints_saved: int = 0
    interrupted: bool = False
    interrupt_node: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from final state."""
        return self.state.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "success": self.success,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 3),
            "nodes_executed": self.nodes_executed,
            "iterations": self.iterations,
            "checkpoints_saved": self.checkpoints_saved,
            "interrupted": self.interrupted,
            "interrupt_node": self.interrupt_node,
            "state_keys": list(self.state.keys()),
        }


class StateGraphExecutor:
    """Victor's native StateGraph workflow executor.

    A production-grade workflow executor with LangGraph-compatible API:
    - TypedDict state schemas for type-safe workflow state
    - Checkpointing for resumable long-running workflows
    - YAML DSL that maps directly to StateGraph patterns
    - Streaming execution with real-time state updates
    - Human-in-the-loop interrupt support

    Example:
        # Create executor
        executor = StateGraphExecutor(orchestrator)

        # Execute workflow with checkpointing
        result = await executor.execute(
            workflow,
            initial_context={"data_path": "/data/input.csv"},
            thread_id="session-123",
        )

        if result.success:
            print(f"Output: {result.get('output')}")
            print(f"Nodes executed: {result.nodes_executed}")
        else:
            print(f"Failed: {result.error}")

        # Streaming execution
        async for node_id, state in executor.stream(workflow, {"input": "data"}):
            print(f"Completed: {node_id}")
    """

    def __init__(
        self,
        orchestrator: Optional["WorkflowAgentProtocol"] = None,
        orchestrators: Optional[dict[str, "WorkflowAgentProtocol"]] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        config: Optional[ExecutorConfig] = None,
    ):
        """Initialize the workflow executor.

        Args:
            orchestrator: Single agent orchestrator for all agent nodes (legacy)
            orchestrators: Dict mapping profile names to orchestrators (new)
            tool_registry: Tool registry for compute nodes
            config: Execution configuration
        """
        # Support both single orchestrator (legacy) and multiple orchestrators (new)
        if orchestrator is not None and orchestrators is not None:
            raise ValueError("Cannot specify both orchestrator and orchestrators")

        if orchestrators:
            self.orchestrators = orchestrators
            self.orchestrator = orchestrators.get(
                config.default_profile
                if config and config.default_profile
                else next(iter(orchestrators))
            )
        elif orchestrator:
            self.orchestrators = {"default": orchestrator}
            self.orchestrator = orchestrator
        else:
            self.orchestrators = {}
            self.orchestrator = None

        self.tool_registry = tool_registry or self._get_default_tool_registry()
        self.config = config or ExecutorConfig()
        self._compiler: Optional["YAMLToStateGraphCompiler"] = None

    def _get_default_tool_registry(self) -> Optional["ToolRegistry"]:
        """Get the default tool registry if available."""
        try:
            from victor.tools.registry import ToolRegistry
            from victor.core.container import ServiceContainer

            container = ServiceContainer()
            return container.get(ToolRegistry)
        except Exception:
            return None

    def _get_compiler(self) -> "YAMLToStateGraphCompiler":
        """Get or create the StateGraph compiler."""
        if self._compiler is None:
            from victor.workflows.yaml_to_graph_compiler import (
                CompilerConfig,
                YAMLToStateGraphCompiler,
            )

            compiler_config = CompilerConfig(
                max_iterations=self.config.max_iterations,
                timeout=self.config.timeout,
                enable_checkpointing=self.config.enable_checkpointing,
                interrupt_on_hitl=bool(self.config.interrupt_nodes),
            )
            self._compiler = YAMLToStateGraphCompiler(
                orchestrator=self.orchestrator,
                orchestrators=self.orchestrators,
                tool_registry=self.tool_registry,
                config=compiler_config,
            )
        return self._compiler

    async def execute(
        self,
        workflow: "WorkflowDefinition",
        initial_context: Optional[dict[str, Any]] = None,
        *,
        thread_id: Optional[str] = None,
        checkpointer: Optional["CheckpointerProtocol"] = None,
        recursion_context: Optional[RecursionContext] = None,
    ) -> ExecutorResult:
        """Execute a workflow using Victor's StateGraph engine.

        Args:
            workflow: The workflow definition to execute
            initial_context: Initial context/state data
            thread_id: Thread ID for checkpointing
            checkpointer: Custom checkpointer
            recursion_context: RecursionContext for tracking nesting depth

        Returns:
            ExecutorResult with execution outcome

        Recursion Context Flow:
            If recursion_context is not provided, a new one is created with
            max_depth from ExecutorConfig. The context is added to the workflow
            state as "_recursion_context" and is automatically tracked when
            team nodes spawn nested workflows. The context is properly exited
            in a finally block to ensure cleanup.
        """
        start_time = time.time()

        logger.info(f"Executing workflow '{workflow.name}'")

        try:
            from victor.workflows.yaml_to_graph_compiler import WorkflowState

            compiler = self._get_compiler()

            # Override checkpointer if provided
            if checkpointer:
                from victor.workflows.yaml_to_graph_compiler import CompilerConfig

                compiler.config = CompilerConfig(
                    max_iterations=self.config.max_iterations,
                    timeout=self.config.timeout,
                    enable_checkpointing=True,
                    checkpointer=checkpointer,
                )

            # Compile workflow to StateGraph
            compiled = compiler.compile(workflow)

            # Create recursion context if not provided
            if recursion_context is None:
                recursion_context = RecursionContext(max_depth=self.config.max_recursion_depth)

            # Track workflow entry
            recursion_context.enter("workflow", workflow.name)

            try:
                # Prepare initial state
                state: WorkflowState = {
                    "_workflow_id": thread_id or uuid.uuid4().hex,
                    "_current_node": workflow.start_node or "",
                    "_node_results": {},
                    "_error": None,
                    "_iteration": 0,
                    "_parallel_results": {},
                    "_hitl_pending": False,
                    "_hitl_response": None,
                    "_recursion_context": recursion_context,  # Add recursion context to state
                }

                # Merge user context
                if initial_context:
                    state.update(initial_context)  # type: ignore[typeddict-item]

                # Execute
                result = await compiled.invoke(state, thread_id=thread_id)

                # Extract user state (exclude internal fields)
                user_state = {k: v for k, v in result.state.items() if not k.startswith("_")}

                return ExecutorResult(
                    success=result.success,
                    state=user_state,
                    error=result.error,
                    duration_seconds=time.time() - start_time,
                    nodes_executed=result.node_history,
                    iterations=result.iterations,
                )
            finally:
                # Always exit recursion context
                recursion_context.exit()

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            return ExecutorResult(
                success=False,
                state=initial_context or {},
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    async def stream(
        self,
        workflow: "WorkflowDefinition",
        initial_context: Optional[dict[str, Any]] = None,
        *,
        thread_id: Optional[str] = None,
        recursion_context: Optional[RecursionContext] = None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Stream workflow execution, yielding after each node.

        Args:
            workflow: The workflow to execute
            initial_context: Initial context data
            thread_id: Thread ID for checkpointing
            recursion_context: RecursionContext for tracking nesting depth

        Yields:
            Tuple of (node_id, current_state) after each node execution

        Recursion Context Flow:
            If recursion_context is not provided, a new one is created with
            max_depth from ExecutorConfig. The context is added to the workflow
            state as "_recursion_context" and is automatically tracked when
            team nodes spawn nested workflows. The context is properly exited
            in a finally block to ensure cleanup.
        """
        from victor.workflows.yaml_to_graph_compiler import WorkflowState

        compiler = self._get_compiler()
        compiled = compiler.compile(workflow)

        # Create recursion context if not provided
        if recursion_context is None:
            recursion_context = RecursionContext(max_depth=self.config.max_recursion_depth)

        # Track workflow entry
        recursion_context.enter("workflow", workflow.name)

        try:
            # Prepare initial state
            state: WorkflowState = {
                "_workflow_id": thread_id or uuid.uuid4().hex,
                "_current_node": workflow.start_node or "",
                "_node_results": {},
                "_error": None,
                "_iteration": 0,
                "_parallel_results": {},
                "_hitl_pending": False,
                "_hitl_response": None,
                "_recursion_context": recursion_context,  # Add recursion context to state
            }

            if initial_context:
                state.update(initial_context)  # type: ignore[typeddict-item]

            # Stream execution
            async for node_id, current_state in compiled.stream(state, thread_id=thread_id):
                # Filter internal fields for user
                user_state = {k: v for k, v in current_state.items() if not k.startswith("_")}
                yield (node_id, user_state)
        finally:
            # Always exit recursion context
            recursion_context.exit()


# Global default executor instance
_default_executor: Optional[StateGraphExecutor] = None


def get_executor(
    orchestrator: Optional["WorkflowAgentProtocol"] = None,
    config: Optional[ExecutorConfig] = None,
) -> StateGraphExecutor:
    """Get or create the default StateGraph executor.

    Args:
        orchestrator: Agent orchestrator (uses existing if not provided)
        config: Execution config (uses existing if not provided)

    Returns:
        StateGraphExecutor instance
    """
    global _default_executor

    if _default_executor is None or orchestrator is not None:
        _default_executor = StateGraphExecutor(
            orchestrator=orchestrator,
            config=config,
        )
    return _default_executor


async def execute_workflow(
    workflow: "WorkflowDefinition",
    initial_context: Optional[dict[str, Any]] = None,
    *,
    orchestrator: Optional["WorkflowAgentProtocol"] = None,
    thread_id: Optional[str] = None,
) -> ExecutorResult:
    """Execute a workflow using the StateGraph executor.

    Convenience function for one-off workflow execution.

    Args:
        workflow: The workflow to execute
        initial_context: Initial context data
        orchestrator: Agent orchestrator
        thread_id: Thread ID for checkpointing

    Returns:
        ExecutorResult
    """
    executor = get_executor(orchestrator)
    return await executor.execute(
        workflow,
        initial_context,
        thread_id=thread_id,
    )


__all__ = [
    # Config
    "ExecutorConfig",
    # Result
    "ExecutorResult",
    # Executor
    "StateGraphExecutor",
    # Convenience
    "get_executor",
    "execute_workflow",
]
