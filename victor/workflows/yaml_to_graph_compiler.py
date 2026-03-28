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

"""YAML to StateGraph Compiler.

Converts YAML workflow definitions (WorkflowDefinition) to StateGraph for
unified execution through the StateGraph engine. This bridges the declarative
YAML workflow API with the typed, checkpointable StateGraph execution model.

Architecture:
    YAML File → YAMLLoader → WorkflowDefinition → YAMLToStateGraphCompiler → StateGraph → CompiledGraph

Key Features:
    - Preserves YAML node semantics (role, goal, tool_budget, llm_config)
    - Converts condition nodes to StateGraph conditional edges
    - Supports parallel execution via parallel node groups
    - Enables checkpointing and interrupts for HITL nodes
    - Maintains full compatibility with existing YAML workflows

Example:
    from victor.workflows.yaml_loader import load_workflow_from_file
    from victor.workflows.yaml_to_graph_compiler import YAMLToStateGraphCompiler

    # Load YAML workflow
    workflow = load_workflow_from_file("workflows/eda_pipeline.yaml", "eda_pipeline")

    # Compile to StateGraph
    compiler = YAMLToStateGraphCompiler(orchestrator)
    compiled = compiler.compile(workflow)

    # Execute with checkpointing
    result = await compiled.invoke(
        {"data_file": "data.csv"},
        thread_id="session-123",
    )
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Optional,
)

from victor.framework.graph import (
    CheckpointerProtocol,
    CompiledGraph,
    GraphExecutionResult,
    MemoryCheckpointer,
)
from victor.workflows.compiler.boundary import (
    NativeWorkflowGraphCompiler,
    ParsedWorkflowDefinition,
    WorkflowCompilationRequest,
)
from victor.workflows.definition import (
    ConditionNode,
    WorkflowDefinition,
)
from victor.workflows.executors.factory import NodeExecutorFactory as UnifiedNodeExecutorFactory
from victor.workflows.runtime_types import GraphNodeResult, WorkflowState

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# Node Executor Factory
# =============================================================================


class NodeExecutorFactory(UnifiedNodeExecutorFactory):
    """Compatibility wrapper around the canonical workflow node executor factory."""

    def __init__(
        self,
        orchestrator: Optional["AgentOrchestrator"] = None,
        orchestrators: Optional[Dict[str, "AgentOrchestrator"]] = None,
        tool_registry: Optional["ToolRegistry"] = None,
    ):
        self.orchestrator = orchestrator
        self.orchestrators = orchestrators or {}
        self.tool_registry = tool_registry
        self._compat_context = _YAMLCompilerExecutionContext(
            orchestrator=orchestrator,
            orchestrators=self.orchestrators,
            tool_registry=tool_registry,
        )
        super().__init__()

    def _resolve_execution_context(self) -> "_YAMLCompilerExecutionContext":
        return self._compat_context


class _YAMLCompilerOrchestratorPool:
    """Profile-aware orchestrator lookup used by YAML compiler compatibility."""

    def __init__(
        self,
        orchestrator: Optional["AgentOrchestrator"] = None,
        orchestrators: Optional[Dict[str, "AgentOrchestrator"]] = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._orchestrators = orchestrators or {}

    def get_orchestrator(self, profile: Optional[str] = None) -> Optional["AgentOrchestrator"]:
        if profile and profile in self._orchestrators:
            return self._orchestrators[profile]
        return self._orchestrator

    def get_default_orchestrator(self) -> Optional["AgentOrchestrator"]:
        return self._orchestrator


class _YAMLCompilerExecutionContext:
    """Execution context adapter for YAML compiler compatibility."""

    def __init__(
        self,
        orchestrator: Optional["AgentOrchestrator"] = None,
        orchestrators: Optional[Dict[str, "AgentOrchestrator"]] = None,
        tool_registry: Optional["ToolRegistry"] = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.orchestrator_pool = _YAMLCompilerOrchestratorPool(
            orchestrator=orchestrator,
            orchestrators=orchestrators,
        )
        self.tool_registry = tool_registry
        self.services = None


# =============================================================================
# Condition Evaluator
# =============================================================================


class ConditionEvaluator:
    """Evaluates condition nodes to determine branch targets.

    Converts YAML condition expressions to StateGraph conditional edge
    routing functions.
    """

    @staticmethod
    def create_router(
        node: ConditionNode,
    ) -> Callable[[WorkflowState], str]:
        """Create a router function for a condition node.

        The router evaluates the condition and returns the BRANCH NAME
        (not the target node ID). The StateGraph Edge.get_target() method
        uses this branch name to look up the actual target from the branches dict.

        Args:
            node: The condition node

        Returns:
            Function that takes state and returns branch name
        """

        def route(state: WorkflowState) -> str:
            try:
                # Evaluate condition using the node's condition function
                # This should return a branch name like "good", "bad", "default"
                branch = node.condition(dict(state))

                # Verify the branch exists in branches dict
                if branch in node.branches:
                    return branch

                # Check for default branch
                if "default" in node.branches:
                    return "default"

                logger.warning(
                    f"Condition node '{node.id}' returned '{branch}' "
                    f"but no matching branch found"
                )
                # Return a non-existent branch, which will cause Edge.get_target()
                # to return None, and the graph will end
                return "__END__"

            except Exception as e:
                logger.error(
                    f"Condition evaluation failed for node '{node.id}': {e}",
                    exc_info=True,
                )
                # Try default branch on error
                if "default" in node.branches:
                    return "default"
                return "__END__"

        return route


# =============================================================================
# YAML to StateGraph Compiler
# =============================================================================


@dataclass
class CompilerConfig:
    """Configuration for the YAML to StateGraph compiler.

    Attributes:
        max_iterations: Maximum loop iterations (default: 25)
        timeout: Overall execution timeout in seconds
        enable_checkpointing: Whether to enable checkpointing
        checkpointer: Custom checkpointer (uses MemoryCheckpointer if None)
        interrupt_on_hitl: Whether to interrupt on HITL nodes
    """

    max_iterations: int = 25
    timeout: Optional[float] = None
    enable_checkpointing: bool = True
    checkpointer: Optional[CheckpointerProtocol] = None
    interrupt_on_hitl: bool = True


class YAMLToStateGraphCompiler:
    """Compiles YAML WorkflowDefinition to StateGraph.

    This compiler bridges the declarative YAML workflow format with the
    typed StateGraph execution engine, enabling:

    - Unified execution model for all workflow sources
    - Type-safe state management via TypedDict
    - Checkpointing and resume for long-running workflows
    - Human-in-the-loop interrupts
    - Cycle detection and iteration limits

    Example:
        from victor.workflows.yaml_loader import load_workflow_from_file
        from victor.workflows.yaml_to_graph_compiler import YAMLToStateGraphCompiler

        # Load and compile workflow
        workflow = load_workflow_from_file("pipeline.yaml", "eda")
        compiler = YAMLToStateGraphCompiler(orchestrator)
        graph = compiler.compile(workflow)

        # Execute with checkpointing
        result = await graph.invoke(
            {"data_path": "/data/input.csv"},
            thread_id="session-123",
        )

        # Resume from checkpoint
        result = await graph.invoke(
            {},  # State loaded from checkpoint
            thread_id="session-123",
        )
    """

    def __init__(
        self,
        orchestrator: Optional["AgentOrchestrator"] = None,
        orchestrators: Optional[Dict[str, "AgentOrchestrator"]] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        config: Optional[CompilerConfig] = None,
    ):
        """Initialize the compiler.

        Args:
            orchestrator: Default agent orchestrator for executing agent nodes
            orchestrators: Dict mapping profile names to orchestrators
            tool_registry: Tool registry for executing compute nodes
            config: Compiler configuration
        """
        self.orchestrator = orchestrator
        self.orchestrators = orchestrators or {}
        self.tool_registry = tool_registry or self._get_default_tool_registry()
        self.config = config or CompilerConfig()
        self._executor_factory = NodeExecutorFactory(
            orchestrator, orchestrators, self.tool_registry
        )

    def _get_default_tool_registry(self) -> Optional["ToolRegistry"]:
        """Get the default tool registry if available."""
        try:
            from victor.tools.registry import get_tool_registry

            return get_tool_registry()
        except Exception:
            return None

    def _apply_compile_config(
        self,
        workflow: WorkflowDefinition,
        config: CompilerConfig,
    ) -> WorkflowDefinition:
        """Overlay compiler config onto the workflow definition for compilation."""
        return replace(
            workflow,
            max_iterations=config.max_iterations,
            max_execution_timeout_seconds=config.timeout,
        )

    def _build_checkpointer_factory(
        self,
        config: CompilerConfig,
    ) -> Callable[[], Optional[CheckpointerProtocol]]:
        """Create a checkpointer factory matching the legacy compiler config."""

        def create_checkpointer() -> Optional[CheckpointerProtocol]:
            return config.checkpointer or MemoryCheckpointer()

        return create_checkpointer

    def _create_native_graph_compiler(
        self,
        config: CompilerConfig,
    ) -> NativeWorkflowGraphCompiler:
        """Create the shared native workflow compiler backend."""
        return NativeWorkflowGraphCompiler(
            node_executor_factory=self._executor_factory,
            checkpointer_factory=self._build_checkpointer_factory(config),
            enable_checkpointing=config.enable_checkpointing,
            interrupt_on_hitl=config.interrupt_on_hitl,
        )

    def compile(
        self,
        workflow: WorkflowDefinition,
        config_override: Optional[CompilerConfig] = None,
    ) -> CompiledGraph[WorkflowState]:
        """Compile a WorkflowDefinition to a StateGraph.

        Args:
            workflow: The YAML workflow definition to compile
            config_override: Override compiler configuration

        Returns:
            CompiledGraph ready for execution

        Raises:
            ValueError: If workflow is invalid or cannot be compiled
        """
        config = config_override or self.config

        # Validate workflow
        errors = workflow.validate()
        if errors:
            raise ValueError(f"Invalid workflow: {'; '.join(errors)}")

        logger.info(f"Compiling workflow '{workflow.name}' to StateGraph")
        compiled_workflow = self._apply_compile_config(workflow, config)
        parsed = ParsedWorkflowDefinition(
            request=WorkflowCompilationRequest(
                source=f"yaml://{workflow.name}",
                workflow_name=workflow.name,
                validate=True,
            ),
            workflow=compiled_workflow,
        )
        compiled = self._create_native_graph_compiler(config).compile(parsed)

        logger.info(f"Compiled workflow '{workflow.name}' with {len(workflow.nodes)} nodes")

        return compiled

    async def compile_and_execute(
        self,
        workflow: WorkflowDefinition,
        initial_state: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> GraphExecutionResult[WorkflowState]:
        """Convenience method to compile and execute in one step.

        Args:
            workflow: The workflow to execute
            initial_state: Initial state data
            thread_id: Thread ID for checkpointing

        Returns:
            GraphExecutionResult with final state
        """
        compiled = self.compile(workflow)

        # Prepare initial state
        state: WorkflowState = {
            "_workflow_id": uuid.uuid4().hex,
            "_current_node": workflow.start_node or "",
            "_node_results": {},
            "_error": None,
            "_iteration": 0,
            "_parallel_results": {},
            "_hitl_pending": False,
            "_hitl_response": None,
        }

        # Merge user-provided initial state
        if initial_state:
            for key, value in initial_state.items():
                state[key] = value

        return await compiled.invoke(state, thread_id=thread_id)


# =============================================================================
# Convenience Functions
# =============================================================================


def compile_yaml_workflow(
    workflow: WorkflowDefinition,
    orchestrator: Optional["AgentOrchestrator"] = None,
    tool_registry: Optional["ToolRegistry"] = None,
    config: Optional[CompilerConfig] = None,
) -> CompiledGraph[WorkflowState]:
    """Compile a YAML workflow to a StateGraph.

    Convenience function for one-off compilation.

    Args:
        workflow: The workflow definition to compile
        orchestrator: Agent orchestrator for agent nodes
        tool_registry: Tool registry for compute nodes
        config: Compiler configuration

    Returns:
        CompiledGraph ready for execution
    """
    compiler = YAMLToStateGraphCompiler(
        orchestrator=orchestrator,
        tool_registry=tool_registry,
        config=config,
    )
    return compiler.compile(workflow)


async def execute_yaml_workflow(
    workflow: WorkflowDefinition,
    initial_state: Optional[Dict[str, Any]] = None,
    orchestrator: Optional["AgentOrchestrator"] = None,
    tool_registry: Optional["ToolRegistry"] = None,
    thread_id: Optional[str] = None,
    config: Optional[CompilerConfig] = None,
) -> GraphExecutionResult[WorkflowState]:
    """Compile and execute a YAML workflow.

    Convenience function for one-off execution.

    Args:
        workflow: The workflow definition to execute
        initial_state: Initial state data
        orchestrator: Agent orchestrator for agent nodes
        tool_registry: Tool registry for compute nodes
        thread_id: Thread ID for checkpointing
        config: Compiler configuration

    Returns:
        ExecutionResult with final state
    """
    compiler = YAMLToStateGraphCompiler(
        orchestrator=orchestrator,
        tool_registry=tool_registry,
        config=config,
    )
    return await compiler.compile_and_execute(workflow, initial_state, thread_id)


__all__ = [
    # State types
    "WorkflowState",
    "GraphNodeResult",
    # Compiler
    "CompilerConfig",
    "YAMLToStateGraphCompiler",
    "NodeExecutorFactory",
    "ConditionEvaluator",
    # Convenience functions
    "compile_yaml_workflow",
    "execute_yaml_workflow",
]
