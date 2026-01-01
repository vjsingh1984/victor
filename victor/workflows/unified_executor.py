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
    AsyncIterator,
    Dict,
    List,
    Optional,
)

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.tools.registry import ToolRegistry
    from victor.workflows.definition import WorkflowDefinition
    from victor.workflows.yaml_to_graph_compiler import (
        CompilerConfig,
        YAMLToStateGraphCompiler,
    )
    from victor.framework.graph import CheckpointerProtocol

logger = logging.getLogger(__name__)


@dataclass
class ExecutorConfig:
    """Configuration for Victor's StateGraph workflow execution.

    Attributes:
        enable_checkpointing: Enable checkpointing for resumable workflows
        max_iterations: Max loop iterations for cyclic workflows
        timeout: Execution timeout in seconds
        interrupt_nodes: Nodes to interrupt before (for human-in-the-loop)
    """

    enable_checkpointing: bool = True
    max_iterations: int = 25
    timeout: Optional[float] = None
    interrupt_nodes: List[str] = field(default_factory=list)


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
    state: Dict[str, Any]
    error: Optional[str] = None
    duration_seconds: float = 0.0
    nodes_executed: List[str] = field(default_factory=list)
    iterations: int = 0
    checkpoints_saved: int = 0
    interrupted: bool = False
    interrupt_node: Optional[str] = None

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from final state."""
        return self.state.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
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
        orchestrator: Optional["AgentOrchestrator"] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        config: Optional[ExecutorConfig] = None,
    ):
        """Initialize the workflow executor.

        Args:
            orchestrator: Agent orchestrator for agent nodes
            tool_registry: Tool registry for compute nodes
            config: Execution configuration
        """
        self.orchestrator = orchestrator
        self.tool_registry = tool_registry or self._get_default_tool_registry()
        self.config = config or ExecutorConfig()
        self._compiler: Optional["YAMLToStateGraphCompiler"] = None

    def _get_default_tool_registry(self) -> Optional["ToolRegistry"]:
        """Get the default tool registry if available."""
        try:
            from victor.tools.registry import get_tool_registry

            return get_tool_registry()
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
                tool_registry=self.tool_registry,
                config=compiler_config,
            )
        return self._compiler

    async def execute(
        self,
        workflow: "WorkflowDefinition",
        initial_context: Optional[Dict[str, Any]] = None,
        *,
        thread_id: Optional[str] = None,
        checkpointer: Optional["CheckpointerProtocol"] = None,
    ) -> ExecutorResult:
        """Execute a workflow using Victor's StateGraph engine.

        Args:
            workflow: The workflow definition to execute
            initial_context: Initial context/state data
            thread_id: Thread ID for checkpointing
            checkpointer: Custom checkpointer

        Returns:
            ExecutorResult with execution outcome
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
            }

            # Merge user context
            for key, value in (initial_context or {}).items():
                state[key] = value

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
        initial_context: Optional[Dict[str, Any]] = None,
        *,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[tuple]:
        """Stream workflow execution, yielding after each node.

        Args:
            workflow: The workflow to execute
            initial_context: Initial context data
            thread_id: Thread ID for checkpointing

        Yields:
            Tuple of (node_id, current_state) after each node execution
        """
        from victor.workflows.yaml_to_graph_compiler import WorkflowState

        compiler = self._get_compiler()
        compiled = compiler.compile(workflow)

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
        }

        for key, value in (initial_context or {}).items():
            state[key] = value

        # Stream execution
        async for node_id, current_state in compiled.stream(state, thread_id=thread_id):
            # Filter internal fields for user
            user_state = {k: v for k, v in current_state.items() if not k.startswith("_")}
            yield (node_id, user_state)


# Global default executor instance
_default_executor: Optional[StateGraphExecutor] = None


def get_executor(
    orchestrator: Optional["AgentOrchestrator"] = None,
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
    initial_context: Optional[Dict[str, Any]] = None,
    *,
    orchestrator: Optional["AgentOrchestrator"] = None,
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
