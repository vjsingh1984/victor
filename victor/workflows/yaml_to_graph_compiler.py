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

import asyncio
import copy
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    TypedDict,
    Union,
)

from victor.framework.graph import (
    END,
    START,
    Checkpoint,
    CheckpointerProtocol,
    CompiledGraph,
    Edge,
    EdgeType,
    ExecutionResult,
    GraphConfig,
    MemoryCheckpointer,
    Node,
    StateGraph,
)
from victor.workflows.definition import (
    AgentNode,
    ComputeNode,
    ConditionNode,
    NodeType,
    ParallelNode,
    TaskConstraints,
    TransformNode,
    WorkflowDefinition,
    WorkflowNode,
)

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# Workflow State Definition
# =============================================================================


class WorkflowState(TypedDict, total=False):
    """Generic state for compiled YAML workflows.

    This TypedDict serves as the state schema for StateGraph execution.
    It combines workflow context data with execution metadata.

    Attributes:
        _workflow_id: Unique workflow execution ID
        _current_node: Currently executing node ID
        _node_results: Results from each executed node
        _error: Error message if execution failed
        _iteration: Current iteration count (for loop detection)
        _parallel_results: Results from parallel node execution
        _hitl_pending: Whether waiting for human input
        _hitl_response: Human response data

    All other keys are dynamic workflow context data.
    """

    # Execution metadata (prefixed with _ to avoid conflicts)
    _workflow_id: str
    _current_node: str
    _node_results: Dict[str, Any]
    _error: Optional[str]
    _iteration: int
    _parallel_results: Dict[str, Any]
    _hitl_pending: bool
    _hitl_response: Optional[Dict[str, Any]]


# =============================================================================
# Node Execution Result
# =============================================================================


@dataclass
class GraphNodeResult:
    """Result from executing a workflow node in the graph.

    Attributes:
        node_id: ID of the executed node
        success: Whether execution succeeded
        output: Output data from the node
        error: Error message if failed
        duration_seconds: Execution time
        tool_calls_used: Number of tool calls made
    """

    node_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    tool_calls_used: int = 0


# =============================================================================
# Node Executor Factory
# =============================================================================


class NodeExecutorFactory:
    """Factory for creating StateGraph node functions from YAML node definitions.

    This factory converts each YAML node type into an async function that can
    be used as a StateGraph node. The generated functions preserve the original
    node semantics while conforming to the StateGraph execution model.
    """

    def __init__(
        self,
        orchestrator: Optional["AgentOrchestrator"] = None,
        tool_registry: Optional["ToolRegistry"] = None,
    ):
        """Initialize the factory.

        Args:
            orchestrator: Agent orchestrator for executing agent nodes
            tool_registry: Tool registry for executing compute nodes
        """
        self.orchestrator = orchestrator
        self.tool_registry = tool_registry

    def create_executor(
        self,
        node: WorkflowNode,
    ) -> Callable[[WorkflowState], WorkflowState]:
        """Create a StateGraph node function from a workflow node.

        Args:
            node: The workflow node to convert

        Returns:
            Async function suitable for StateGraph node execution
        """
        if isinstance(node, AgentNode):
            return self._create_agent_executor(node)
        elif isinstance(node, ComputeNode):
            return self._create_compute_executor(node)
        elif isinstance(node, TransformNode):
            return self._create_transform_executor(node)
        elif isinstance(node, ParallelNode):
            return self._create_parallel_executor(node)
        elif isinstance(node, ConditionNode):
            # Condition nodes don't execute - they're handled as edges
            return self._create_passthrough_executor(node)
        else:
            logger.warning(f"Unknown node type: {type(node)}, using passthrough")
            return self._create_passthrough_executor(node)

    def _create_agent_executor(
        self,
        node: AgentNode,
    ) -> Callable[[WorkflowState], WorkflowState]:
        """Create executor for an AgentNode.

        Spawns a sub-agent with the specified role and goal to process
        the task using LLM inference and tool execution.
        """
        orchestrator = self.orchestrator

        async def execute_agent(state: WorkflowState) -> WorkflowState:
            start_time = time.time()
            state = dict(state)  # Make mutable copy

            try:
                # Build input context from input_mapping
                input_context = {}
                for param_name, context_key in node.input_mapping.items():
                    if context_key in state:
                        input_context[param_name] = state[context_key]

                # Build the goal with context substitution
                goal = node.goal
                for key, value in state.items():
                    if not key.startswith("_"):
                        goal = goal.replace(f"${{{key}}}", str(value))
                        goal = goal.replace(f"$ctx.{key}", str(value))

                if orchestrator is None:
                    # Fallback: store placeholder result
                    logger.warning(
                        f"No orchestrator available for agent node '{node.id}', "
                        "using placeholder execution"
                    )
                    output = {
                        "node_id": node.id,
                        "role": node.role,
                        "goal": goal,
                        "status": "placeholder",
                        "input_context": input_context,
                    }
                else:
                    # Execute via SubAgentOrchestrator
                    from victor.agent.subagents import (
                        SubAgentOrchestrator,
                        SubAgentRole,
                    )

                    # Map role string to SubAgentRole enum
                    role_map = {
                        "researcher": SubAgentRole.RESEARCHER,
                        "planner": SubAgentRole.PLANNER,
                        "executor": SubAgentRole.EXECUTOR,
                        "reviewer": SubAgentRole.REVIEWER,
                        "writer": SubAgentRole.WRITER,
                        "analyst": SubAgentRole.RESEARCHER,  # Alias
                    }
                    role = role_map.get(node.role.lower(), SubAgentRole.EXECUTOR)

                    # Create and execute sub-agent
                    sub_orchestrator = SubAgentOrchestrator(orchestrator)
                    result = await sub_orchestrator.execute_task(
                        role=role,
                        task=goal,
                        context=input_context,
                        tool_budget=node.tool_budget,
                        allowed_tools=node.allowed_tools,
                    )

                    output = {
                        "response": result.response if result else None,
                        "success": result.success if result else False,
                        "tool_calls": result.tool_calls_used if result else 0,
                    }

                # Store output in state
                output_key = node.output_key or node.id
                state[output_key] = output

                # Update node results
                if "_node_results" not in state:
                    state["_node_results"] = {}
                state["_node_results"][node.id] = GraphNodeResult(
                    node_id=node.id,
                    success=True,
                    output=output,
                    duration_seconds=time.time() - start_time,
                )

            except Exception as e:
                logger.error(f"Agent node '{node.id}' failed: {e}", exc_info=True)
                state["_error"] = f"Agent node '{node.id}' failed: {e}"
                if "_node_results" not in state:
                    state["_node_results"] = {}
                state["_node_results"][node.id] = GraphNodeResult(
                    node_id=node.id,
                    success=False,
                    error=str(e),
                    duration_seconds=time.time() - start_time,
                )

            return state

        return execute_agent

    def _create_compute_executor(
        self,
        node: ComputeNode,
    ) -> Callable[[WorkflowState], WorkflowState]:
        """Create executor for a ComputeNode.

        Executes tools directly without LLM inference, using registered
        handlers for domain-specific logic.
        """
        tool_registry = self.tool_registry

        async def execute_compute(state: WorkflowState) -> WorkflowState:
            start_time = time.time()
            state = dict(state)  # Make mutable copy
            tool_calls_used = 0

            try:
                # Build params from input_mapping
                params = {}
                for param_name, context_key in node.input_mapping.items():
                    # Handle $ctx.key syntax
                    if isinstance(context_key, str) and context_key.startswith("$ctx."):
                        context_key = context_key[5:]
                    if context_key in state:
                        params[param_name] = state[context_key]
                    else:
                        params[param_name] = context_key

                # Check for custom handler
                if node.handler:
                    from victor.workflows.executor import get_compute_handler

                    handler = get_compute_handler(node.handler)
                    if handler:
                        # Create minimal WorkflowContext wrapper
                        from victor.workflows.executor import WorkflowContext

                        context = WorkflowContext(dict(state))
                        result = await handler(node, context, tool_registry)

                        # Transfer context changes back to state
                        for key, value in context.data.items():
                            if not key.startswith("_"):
                                state[key] = value

                        output = result.output if result else None
                        tool_calls_used = result.tool_calls_used if result else 0
                    else:
                        logger.warning(f"Handler '{node.handler}' not found for node '{node.id}'")
                        output = {"error": f"Handler '{node.handler}' not found"}
                else:
                    # Execute tools directly
                    outputs = {}
                    if tool_registry and node.tools:
                        for tool_name in node.tools:
                            # Check constraints
                            if not node.constraints.allows_tool(tool_name):
                                logger.debug(f"Tool '{tool_name}' blocked by constraints")
                                continue

                            try:
                                result = await asyncio.wait_for(
                                    tool_registry.execute(
                                        tool_name,
                                        _exec_ctx={
                                            "workflow_context": state,
                                            "constraints": node.constraints.to_dict(),
                                        },
                                        **params,
                                    ),
                                    timeout=node.constraints.timeout,
                                )
                                tool_calls_used += 1

                                if result.success:
                                    outputs[tool_name] = result.output
                                else:
                                    outputs[tool_name] = {"error": result.error}

                                if node.fail_fast and not result.success:
                                    break

                            except asyncio.TimeoutError:
                                outputs[tool_name] = {"error": "Timeout"}
                                if node.fail_fast:
                                    break
                            except Exception as e:
                                outputs[tool_name] = {"error": str(e)}
                                if node.fail_fast:
                                    break
                    else:
                        outputs = {"status": "no_tools_executed", "params": params}

                    output = outputs

                # Store output in state
                output_key = node.output_key or node.id
                state[output_key] = output

                # Update node results
                if "_node_results" not in state:
                    state["_node_results"] = {}
                state["_node_results"][node.id] = GraphNodeResult(
                    node_id=node.id,
                    success=True,
                    output=output,
                    duration_seconds=time.time() - start_time,
                    tool_calls_used=tool_calls_used,
                )

            except Exception as e:
                logger.error(f"Compute node '{node.id}' failed: {e}", exc_info=True)
                state["_error"] = f"Compute node '{node.id}' failed: {e}"
                if "_node_results" not in state:
                    state["_node_results"] = {}
                state["_node_results"][node.id] = GraphNodeResult(
                    node_id=node.id,
                    success=False,
                    error=str(e),
                    duration_seconds=time.time() - start_time,
                    tool_calls_used=tool_calls_used,
                )

            return state

        return execute_compute

    def _create_transform_executor(
        self,
        node: TransformNode,
    ) -> Callable[[WorkflowState], WorkflowState]:
        """Create executor for a TransformNode.

        Applies a transformation function to the workflow state.
        """

        async def execute_transform(state: WorkflowState) -> WorkflowState:
            start_time = time.time()
            state = dict(state)  # Make mutable copy

            try:
                # Execute transform function
                transformed = node.transform(state)

                # Merge transformed data back into state
                for key, value in transformed.items():
                    state[key] = value

                # Update node results
                if "_node_results" not in state:
                    state["_node_results"] = {}
                state["_node_results"][node.id] = GraphNodeResult(
                    node_id=node.id,
                    success=True,
                    output={"transformed_keys": list(transformed.keys())},
                    duration_seconds=time.time() - start_time,
                )

            except Exception as e:
                logger.error(f"Transform node '{node.id}' failed: {e}", exc_info=True)
                state["_error"] = f"Transform node '{node.id}' failed: {e}"
                if "_node_results" not in state:
                    state["_node_results"] = {}
                state["_node_results"][node.id] = GraphNodeResult(
                    node_id=node.id,
                    success=False,
                    error=str(e),
                    duration_seconds=time.time() - start_time,
                )

            return state

        return execute_transform

    def _create_parallel_executor(
        self,
        node: ParallelNode,
    ) -> Callable[[WorkflowState], WorkflowState]:
        """Create executor for a ParallelNode.

        Executes child nodes in parallel and aggregates results.
        Note: The actual parallel execution is handled by the compiler
        which creates separate graph branches. This executor serves as
        a synchronization point.
        """

        async def execute_parallel(state: WorkflowState) -> WorkflowState:
            start_time = time.time()
            state = dict(state)  # Make mutable copy

            try:
                # Parallel results should already be populated by child nodes
                # This node serves as a join point
                parallel_results = state.get("_parallel_results", {})

                # Apply join strategy
                if node.join_strategy == "all":
                    # All must succeed
                    all_success = all(
                        r.get("success", False) for r in parallel_results.values()
                    )
                    if not all_success:
                        state["_error"] = "Not all parallel nodes succeeded"
                elif node.join_strategy == "any":
                    # At least one must succeed
                    any_success = any(
                        r.get("success", False) for r in parallel_results.values()
                    )
                    if not any_success:
                        state["_error"] = "No parallel nodes succeeded"
                # "merge" strategy just combines all results

                # Update node results
                if "_node_results" not in state:
                    state["_node_results"] = {}
                state["_node_results"][node.id] = GraphNodeResult(
                    node_id=node.id,
                    success="_error" not in state,
                    output={"parallel_nodes": node.parallel_nodes},
                    duration_seconds=time.time() - start_time,
                )

            except Exception as e:
                logger.error(f"Parallel node '{node.id}' failed: {e}", exc_info=True)
                state["_error"] = f"Parallel node '{node.id}' failed: {e}"

            return state

        return execute_parallel

    def _create_passthrough_executor(
        self,
        node: WorkflowNode,
    ) -> Callable[[WorkflowState], WorkflowState]:
        """Create a passthrough executor that just forwards state.

        Used for condition nodes (which are handled as edges) and
        unknown node types.
        """

        async def passthrough(state: WorkflowState) -> WorkflowState:
            state = dict(state)
            if "_node_results" not in state:
                state["_node_results"] = {}
            state["_node_results"][node.id] = GraphNodeResult(
                node_id=node.id,
                success=True,
                output={"passthrough": True},
            )
            return state

        return passthrough


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
        tool_registry: Optional["ToolRegistry"] = None,
        config: Optional[CompilerConfig] = None,
    ):
        """Initialize the compiler.

        Args:
            orchestrator: Agent orchestrator for executing agent nodes
            tool_registry: Tool registry for executing compute nodes
            config: Compiler configuration
        """
        self.orchestrator = orchestrator
        self.tool_registry = tool_registry or self._get_default_tool_registry()
        self.config = config or CompilerConfig()
        self._executor_factory = NodeExecutorFactory(orchestrator, self.tool_registry)

    def _get_default_tool_registry(self) -> Optional["ToolRegistry"]:
        """Get the default tool registry if available."""
        try:
            from victor.tools.registry import get_tool_registry

            return get_tool_registry()
        except Exception:
            return None

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

        # Create StateGraph with WorkflowState schema
        graph = StateGraph(WorkflowState)

        # Track nodes and edges
        nodes_added: Set[str] = set()
        condition_nodes: Dict[str, ConditionNode] = {}
        parallel_nodes: Dict[str, ParallelNode] = {}

        # First pass: categorize nodes and find parallel children
        parallel_children: Set[str] = set()
        for node_id, node in workflow.nodes.items():
            if isinstance(node, ConditionNode):
                condition_nodes[node_id] = node
            elif isinstance(node, ParallelNode):
                parallel_nodes[node_id] = node
                # Mark child nodes - they'll be executed inside the parallel executor
                parallel_children.update(node.parallel_nodes)

        # Second pass: add execution nodes (skip parallel children)
        for node_id, node in workflow.nodes.items():
            # Skip nodes that are children of a parallel node
            if node_id in parallel_children:
                continue

            if isinstance(node, ConditionNode):
                # Condition nodes are handled as passthrough + conditional edge
                executor = self._executor_factory.create_executor(node)
                graph.add_node(node_id, executor)
                nodes_added.add(node_id)
            elif isinstance(node, ParallelNode):
                # Parallel nodes need special handling
                self._add_parallel_node_group(graph, node, workflow, nodes_added)
            else:
                # Regular execution node
                executor = self._executor_factory.create_executor(node)
                graph.add_node(node_id, executor)
                nodes_added.add(node_id)

        # Third pass: add edges (skip parallel children)
        for node_id, node in workflow.nodes.items():
            # Skip nodes that are children of a parallel node
            if node_id in parallel_children:
                continue

            if isinstance(node, ConditionNode):
                # Add conditional edge
                router = ConditionEvaluator.create_router(node)
                graph.add_conditional_edge(
                    node_id,
                    router,
                    node.branches,
                )
            elif isinstance(node, ParallelNode):
                # Parallel node edges are handled in _add_parallel_node_group
                pass
            else:
                # Add normal edges to next nodes
                for next_node_id in node.next_nodes:
                    # Skip edges to parallel children (they're internal to the parallel node)
                    if next_node_id in parallel_children:
                        continue
                    if next_node_id in workflow.nodes or next_node_id == END:
                        graph.add_edge(node_id, next_node_id)

        # Set entry point
        if workflow.start_node:
            graph.set_entry_point(workflow.start_node)
        elif workflow.nodes:
            # Default to first node
            first_node = next(iter(workflow.nodes.keys()))
            graph.set_entry_point(first_node)

        # Configure HITL interrupts
        interrupt_before = []
        if config.interrupt_on_hitl:
            for node_id, node in workflow.nodes.items():
                if node.node_type == NodeType.HITL:
                    interrupt_before.append(node_id)

        # Build checkpointer
        checkpointer = None
        if config.enable_checkpointing:
            checkpointer = config.checkpointer or MemoryCheckpointer()

        # Compile with individual config parameters
        # StateGraph.compile() expects checkpointer and **config_kwargs
        compiled = graph.compile(
            checkpointer=checkpointer,
            max_iterations=config.max_iterations,
            timeout=config.timeout,
            interrupt_before=interrupt_before,
        )

        logger.info(
            f"Compiled workflow '{workflow.name}' with {len(nodes_added)} nodes"
        )

        return compiled

    def _add_parallel_node_group(
        self,
        graph: StateGraph,
        parallel_node: ParallelNode,
        workflow: WorkflowDefinition,
        nodes_added: Set[str],
    ) -> None:
        """Add a parallel node group to the graph.

        Parallel execution is modeled by:
        1. A fork node that splits into parallel branches
        2. Individual parallel branch nodes
        3. A join node that synchronizes results

        For simplicity, we execute parallel nodes sequentially in the
        same node function. True parallel execution would require
        extending the StateGraph model.

        Args:
            graph: The StateGraph being built
            parallel_node: The parallel node definition
            workflow: The full workflow definition
            nodes_added: Set of already-added node IDs
        """
        # Create a combined executor that runs all parallel nodes
        parallel_node_defs = []
        for child_id in parallel_node.parallel_nodes:
            if child_id in workflow.nodes:
                parallel_node_defs.append(workflow.nodes[child_id])

        executors = [
            self._executor_factory.create_executor(node) for node in parallel_node_defs
        ]

        async def execute_parallel_group(state: WorkflowState) -> WorkflowState:
            """Execute all parallel nodes and aggregate results."""
            state = dict(state)
            start_time = time.time()

            if "_parallel_results" not in state:
                state["_parallel_results"] = {}

            # Execute all nodes (can be parallelized with asyncio.gather)
            tasks = [executor(copy.deepcopy(state)) for executor in executors]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Merge results
            for i, (node_def, result) in enumerate(zip(parallel_node_defs, results)):
                if isinstance(result, Exception):
                    state["_parallel_results"][node_def.id] = {
                        "success": False,
                        "error": str(result),
                    }
                else:
                    # Merge state changes from parallel execution
                    for key, value in result.items():
                        if not key.startswith("_"):
                            state[key] = value
                    state["_parallel_results"][node_def.id] = {
                        "success": True,
                        "output": result.get(node_def.output_key or node_def.id),
                    }

            # Record parallel node result
            if "_node_results" not in state:
                state["_node_results"] = {}
            state["_node_results"][parallel_node.id] = GraphNodeResult(
                node_id=parallel_node.id,
                success=all(
                    r.get("success", False)
                    for r in state["_parallel_results"].values()
                ),
                output=state["_parallel_results"],
                duration_seconds=time.time() - start_time,
            )

            return state

        # Add the combined parallel executor
        graph.add_node(parallel_node.id, execute_parallel_group)
        nodes_added.add(parallel_node.id)

        # Add edges to next nodes
        for next_node_id in parallel_node.next_nodes:
            if next_node_id in workflow.nodes or next_node_id == END:
                graph.add_edge(parallel_node.id, next_node_id)

    async def compile_and_execute(
        self,
        workflow: WorkflowDefinition,
        initial_state: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> ExecutionResult[WorkflowState]:
        """Convenience method to compile and execute in one step.

        Args:
            workflow: The workflow to execute
            initial_state: Initial state data
            thread_id: Thread ID for checkpointing

        Returns:
            ExecutionResult with final state
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
    compiler = YAMLToStateGraphCompiler(orchestrator, tool_registry, config)
    return compiler.compile(workflow)


async def execute_yaml_workflow(
    workflow: WorkflowDefinition,
    initial_state: Optional[Dict[str, Any]] = None,
    orchestrator: Optional["AgentOrchestrator"] = None,
    tool_registry: Optional["ToolRegistry"] = None,
    thread_id: Optional[str] = None,
    config: Optional[CompilerConfig] = None,
) -> ExecutionResult[WorkflowState]:
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
    compiler = YAMLToStateGraphCompiler(orchestrator, tool_registry, config)
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
