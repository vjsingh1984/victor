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

"""Unified Workflow Compiler.

Consolidates all workflow compilation paths into a single, consistent pipeline
with integrated caching. This module unifies:

1. WorkflowGraphCompiler (graph_dsl -> CompiledGraph)
2. YAMLToStateGraphCompiler (YAML -> StateGraph -> CompiledGraph)
3. WorkflowDefinitionCompiler (WorkflowDefinition -> CompiledGraph)

Key Features:
- Single entry point for all workflow types
- Integrated caching (definition + execution)
- DRY node execution via NodeExecutorFactory
- True parallel execution via asyncio.gather
- Observability event emission

Example:
    from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
    from pathlib import Path

    # Create compiler with caching
    compiler = UnifiedWorkflowCompiler(enable_caching=True, cache_ttl=3600)

    # Compile from YAML file
    cached_graph = compiler.compile_yaml(Path("workflow.yaml"), "my_workflow")

    # Execute with automatic cache integration
    result = await cached_graph.invoke({"input": "data"})

    # Check cache stats
    print(compiler.get_cache_stats())
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

if TYPE_CHECKING:
    from victor.protocols import WorkflowAgentProtocol
    from victor.framework.config import GraphConfig
    from victor.framework.graph import CompiledGraph, GraphExecutionResult
    from victor.tools.registry import ToolRegistry
    from victor.workflows.cache import (
        WorkflowDefinitionCache,
        WorkflowCacheManager,
    )
    from victor.workflows.definition import (
        AgentNode,
        ComputeNode,
        ConditionNode,
        ParallelNode,
        TransformNode,
        TeamNodeWorkflow,
        WorkflowDefinition,
        WorkflowNode,
    )
    from victor.workflows.graph_dsl import WorkflowGraph
    from victor.workflows.node_runners import NodeRunnerRegistry
    from victor.workflows.graph_compiler import (
        CompilerConfig,
        WorkflowGraphCompiler,
        WorkflowDefinitionCompiler,
    )
    from victor.workflows.yaml_loader import YAMLWorkflowConfig

from victor.core.errors import (
    WorkflowExecutionError,
    ConfigurationValidationError,
    ToolExecutionError,
    ToolTimeoutError,
    ValidationError,
    ConfigurationError,
)

logger = logging.getLogger(__name__)

StateType = TypeVar("StateType", bound=Dict[str, Any])


# =============================================================================
# Compiler Configuration
# =============================================================================


@dataclass
class UnifiedCompilerConfig:
    """Configuration for the UnifiedWorkflowCompiler.

    Attributes:
        enable_caching: Whether to enable caching (default: True)
        cache_ttl: Cache time-to-live in seconds (default: 3600)
        max_cache_entries: Maximum cache entries (default: 500)
        validate_before_compile: Whether to validate workflows before compilation
        enable_observability: Whether to emit observability events
        use_node_runners: Whether to use NodeRunner protocol for execution
        preserve_state_type: Whether to preserve typed state or use dict
        max_iterations: Maximum workflow iterations (default: 25)
        execution_timeout: Overall execution timeout in seconds
        enable_checkpointing: Whether to enable state checkpointing
        max_recursion_depth: Maximum recursion depth for nested execution (default: 3)
    """

    enable_caching: bool = True
    cache_ttl: int = 3600
    max_cache_entries: int = 500
    validate_before_compile: bool = True
    enable_observability: bool = False
    use_node_runners: bool = False
    preserve_state_type: bool = False
    max_iterations: int = 25
    execution_timeout: Optional[float] = None
    enable_checkpointing: bool = True
    max_recursion_depth: int = 3


# =============================================================================
# Node Execution Result
# =============================================================================


@dataclass
class NodeExecutionResult:
    """Result from executing a workflow node.

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
# Node Executor Factory (DRY Consolidation)
# =============================================================================


class NodeExecutorFactory:
    """Factory for creating node executors (DRY consolidation).

    Extracts common node execution logic from all compilers into shared
    factory methods. This eliminates code duplication across:
    - WorkflowGraphCompiler
    - YAMLToStateGraphCompiler
    - WorkflowDefinitionCompiler

    Example:
        factory = NodeExecutorFactory(orchestrator, tool_registry)
        executor = factory.create_executor(node)
        new_state = await executor(current_state)
    """

    def __init__(
        self,
        orchestrator: Optional["WorkflowAgentProtocol"] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        runner_registry: Optional["NodeRunnerRegistry"] = None,
        emitter: Optional[Any] = None,  # ObservabilityEmitter
    ):
        """Initialize the factory.

        Args:
            orchestrator: Agent orchestrator for executing agent nodes
            tool_registry: Tool registry for executing compute nodes
            runner_registry: Optional NodeRunner registry for unified execution
            emitter: Optional ObservabilityEmitter for streaming events
        """
        self.orchestrator = orchestrator
        self.tool_registry = tool_registry
        self._runner_registry = runner_registry
        self._emitter = emitter

        # Keys that need deep copy for isolation in parallel execution
        self._mutable_state_keys = frozenset(
            {"_parallel_results", "_node_results", "_errors", "_checkpoints"}
        )

    def _copy_state_for_parallel(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create an isolated copy of state for parallel execution.

        Uses selective copying strategy:
        - Shallow copy for most keys (user data assumed immutable)
        - Deep copy only for internal mutable tracking structures
        - Shared references for recursion context (all branches track same depth)

        This is significantly faster than full deepcopy while maintaining
        isolation for parallel node execution.

        Note on RecursionContext:
            The _recursion_context is intentionally shared (not deep copied) across
            all parallel branches. This ensures that all parallel executions contribute
            to the same recursion depth tracking, preventing excessive nesting when
            multiple branches spawn sub-workflows concurrently.

        Args:
            state: Current workflow state

        Returns:
            Isolated state copy for child node execution
        """
        # Start with shallow copy (preserves _recursion_context reference)
        child_state = dict(state)

        # Deep copy only mutable internal structures that need isolation
        for key in self._mutable_state_keys:
            if key in child_state:
                child_state[key] = copy.deepcopy(child_state[key])

        return child_state

    def create_executor(
        self,
        node: "WorkflowNode",
    ) -> Callable[[Dict[str, Any]], Any]:
        """Create an executor function for a workflow node.

        Args:
            node: The workflow node to create an executor for

        Returns:
            Async callable that takes state and returns updated state
        """
        from victor.workflows.definition import (
            AgentNode,
            ComputeNode,
            ConditionNode,
            ParallelNode,
            TransformNode,
            TeamNodeWorkflow,
        )

        if isinstance(node, AgentNode):
            return self.create_agent_executor(node)
        elif isinstance(node, ComputeNode):
            return self.create_compute_executor(node)
        elif isinstance(node, ConditionNode):
            return self.create_condition_router(node)
        elif isinstance(node, ParallelNode):
            return self.create_parallel_executor(node)
        elif isinstance(node, TransformNode):
            return self.create_transform_executor(node)
        elif isinstance(node, TeamNodeWorkflow):
            return self.create_team_executor(node)
        else:
            return self._create_passthrough_executor(node)

    def create_agent_executor(
        self,
        node: "AgentNode",
    ) -> Callable[[Dict[str, Any]], Any]:
        """Create executor for an AgentNode.

        Spawns a sub-agent with the specified role and goal to process
        the task using LLM inference and tool execution.

        If the node has a retry_policy configured, the execution will
        automatically retry on failure with exponential backoff.

        If the node has a timeout_seconds configured, the execution will
        be cancelled if it exceeds the timeout.

        Recursion Context:
            This executor preserves the _recursion_context key in the state
            but does not actively use it. Agent nodes do not spawn nested
            workflows, so they don't need to track recursion depth.
        """
        orchestrator = self.orchestrator
        emitter = self._emitter
        retry_policy = node.retry_policy
        timeout_seconds = node.timeout_seconds

        async def execute_agent(state: Dict[str, Any]) -> Dict[str, Any]:
            start_time = time.time()
            state = dict(state)  # Make mutable copy
            node_name = node.name

            # Emit NODE_START if emitter is configured
            if emitter and hasattr(emitter, "emit_node_start"):
                emitter.emit_node_start(node.id, node_name)

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
                    from victor.agent.orchestrator import AgentOrchestrator
                    from typing import cast

                    # Map role string to SubAgentRole enum
                    role_map = {
                        "researcher": SubAgentRole.RESEARCHER,
                        "planner": SubAgentRole.PLANNER,
                        "executor": SubAgentRole.EXECUTOR,
                        "reviewer": SubAgentRole.REVIEWER,
                        "writer": SubAgentRole.EXECUTOR,
                        "analyst": SubAgentRole.RESEARCHER,  # Alias
                    }
                    role = role_map.get(node.role.lower(), SubAgentRole.EXECUTOR)

                    # Create sub-agent orchestrator
                    # Cast to AgentOrchestrator (SubAgentOrchestrator expects it)
                    sub_orchestrator = SubAgentOrchestrator(cast(AgentOrchestrator, orchestrator))

                    # Create the coroutine for execution
                    async def _run_sub_agent() -> Dict[str, Any]:
                        result = await sub_orchestrator.spawn(
                            role=role,
                            task=goal,
                            tool_budget=node.tool_budget,
                            allowed_tools=node.allowed_tools,
                        )
                        return (
                            result.to_dict()
                            if hasattr(result, "to_dict")
                            else {"output": str(result)}
                        )

                    # Execute with timeout if configured
                    if timeout_seconds:
                        try:
                            await asyncio.wait_for(
                                _run_sub_agent(),
                                timeout=timeout_seconds,
                            )
                        except asyncio.TimeoutError:
                            duration = time.time() - start_time
                            error_msg = (
                                f"Agent node '{node.id}' timed out after " f"{timeout_seconds}s"
                            )
                            logger.warning(error_msg)
                            state["_error"] = error_msg
                            state["_timeout"] = True
                            if "_node_results" not in state:
                                state["_node_results"] = {}
                            state["_node_results"][node.id] = NodeExecutionResult(
                                node_id=node.id,
                                success=False,
                                error=error_msg,
                                duration_seconds=duration,
                            )
                            if emitter and hasattr(emitter, "emit_node_error"):
                                emitter.emit_node_error(
                                    node.id,
                                    error=error_msg,
                                    node_name=node_name,
                                    duration=duration,
                                )
                            return state
                    else:
                        result_dict = await _run_sub_agent()
                        output = result_dict

                # Store output in state
                output_key = node.output_key or node.id
                state[output_key] = output

                # Update node results
                duration = time.time() - start_time
                if "_node_results" not in state:
                    state["_node_results"] = {}
                state["_node_results"][node.id] = NodeExecutionResult(
                    node_id=node.id,
                    success=True,
                    output=output,
                    duration_seconds=duration,
                )

                # Emit NODE_COMPLETE if emitter is configured
                if emitter and hasattr(emitter, "emit_node_complete"):
                    emitter.emit_node_complete(
                        node.id,
                        node_name,
                        duration=duration,
                        output=output,
                    )

            except (ToolExecutionError, ToolTimeoutError, ValidationError, ConfigurationError) as e:
                # Known workflow errors - handle with specific logging
                duration = time.time() - start_time
                correlation_id = getattr(e, "correlation_id", str(uuid.uuid4())[:8])
                workflow_id = state.get("_workflow_name", "unknown")
                checkpoint_id = state.get("_checkpoint_id") or state.get("thread_id")

                # Log with correlation ID
                logger.error(
                    f"[{correlation_id}] Agent node '{node.id}' failed in workflow '{workflow_id}': {e}",
                    exc_info=True,
                )

                # Store error in state
                error_msg = f"Agent node '{node.id}' failed: {e}"
                state["_error"] = error_msg
                if "_node_results" not in state:
                    state["_node_results"] = {}
                state["_node_results"][node.id] = NodeExecutionResult(
                    node_id=node.id,
                    success=False,
                    error=str(e),
                    duration_seconds=duration,
                )

                # Emit NODE_ERROR if emitter is configured
                if emitter and hasattr(emitter, "emit_node_error"):
                    emitter.emit_node_error(
                        node.id,
                        error=str(e),
                        node_name=node_name,
                        duration=duration,
                    )

                # Raise WorkflowExecutionError if not continuing on error
                if not getattr(node, "continue_on_error", False):
                    raise WorkflowExecutionError(
                        message=error_msg,
                        workflow_id=workflow_id,
                        node_id=node.id,
                        node_type="agent",
                        checkpoint_id=checkpoint_id,
                        execution_context={"duration": duration, "original_error": str(e)},
                    ) from e

            except Exception as e:
                # Catch-all for truly unexpected errors
                duration = time.time() - start_time
                correlation_id = str(uuid.uuid4())[:8]
                workflow_id = state.get("_workflow_name", "unknown")
                checkpoint_id = state.get("_checkpoint_id") or state.get("thread_id")

                # Log with correlation ID
                logger.error(
                    f"[{correlation_id}] Agent node '{node.id}' failed in workflow '{workflow_id}': {e}",
                    exc_info=True,
                )

                # Store error in state
                error_msg = f"Agent node '{node.id}' failed: {e}"
                state["_error"] = error_msg
                if "_node_results" not in state:
                    state["_node_results"] = {}
                state["_node_results"][node.id] = NodeExecutionResult(
                    node_id=node.id,
                    success=False,
                    error=str(e),
                    duration_seconds=duration,
                )

                # Emit NODE_ERROR if emitter is configured
                if emitter and hasattr(emitter, "emit_node_error"):
                    emitter.emit_node_error(
                        node.id,
                        error=str(e),
                        node_name=node_name,
                        duration=duration,
                    )

                # Raise WorkflowExecutionError if not continuing on error
                if not getattr(node, "continue_on_error", False):
                    raise WorkflowExecutionError(
                        error_msg,
                        workflow_id=workflow_id,
                        node_id=node.id,
                        node_type="agent",
                        checkpoint_id=checkpoint_id,
                        execution_context={
                            "duration": duration,
                            "role": node.role if hasattr(node, "role") else "unknown",
                            "goal": node.goal if hasattr(node, "goal") else "unknown",
                        },
                        correlation_id=correlation_id,
                    ) from e

            return state

        # Wrap with retry if policy is configured
        if retry_policy:
            from victor.workflows.resilience import (
                retry_policy_to_strategy,
                RetryExecutor,
            )

            strategy = retry_policy_to_strategy(retry_policy)
            retry_executor = RetryExecutor(strategy)

            async def execute_agent_with_retry(state: Dict[str, Any]) -> Dict[str, Any]:
                """Wrapper that applies retry policy to agent execution."""
                from typing import cast

                result = await retry_executor.execute_async(lambda: execute_agent(state))
                if result.success:
                    return cast(Dict[str, Any], result.result)
                else:
                    # Return last state with error info
                    state = dict(state)
                    state["_retry_exhausted"] = True
                    state["_retry_attempts"] = result.attempts
                    state["_error"] = (
                        f"Agent node '{node.id}' failed after {result.attempts} attempts"
                    )
                    if emitter and hasattr(emitter, "emit_node_error"):
                        emitter.emit_node_error(
                            node.id,
                            error=f"Retry exhausted after {result.attempts} attempts",
                            node_name=node.name,
                            duration=0,
                        )
                    return state

            return execute_agent_with_retry

        return execute_agent

    def create_compute_executor(
        self,
        node: "ComputeNode",
    ) -> Callable[[Dict[str, Any]], Any]:
        """Create executor for a ComputeNode.

        Executes tools directly without LLM inference, using registered
        handlers for domain-specific logic.

        If the node has a retry_policy configured, the execution will
        automatically retry on failure with exponential backoff.

        Recursion Context:
            This executor preserves the _recursion_context key in the state
            but does not actively use it. Compute nodes execute tools directly
            and do not spawn nested workflows.
        """
        tool_registry = self.tool_registry
        emitter = self._emitter
        retry_policy = node.retry_policy

        async def execute_compute(state: Dict[str, Any]) -> Dict[str, Any]:
            start_time = time.time()
            state = dict(state)  # Make mutable copy
            tool_calls_used = 0
            node_name = node.name

            # Emit NODE_START if emitter is configured
            if emitter and hasattr(emitter, "emit_node_start"):
                emitter.emit_node_start(node.id, node_name)

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
                        if tool_registry is None:
                            raise ValueError(
                                f"Tool registry is required for compute node '{node.id}'"
                            )
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
                            except (ToolExecutionError, ToolTimeoutError, ValidationError) as e:
                                # Known tool errors
                                outputs[tool_name] = {"error": str(e)}
                                if node.fail_fast:
                                    break
                            except Exception as e:
                                # Catch-all for unexpected errors
                                logger.warning(
                                    f"Unexpected error executing tool '{tool_name}': {e}"
                                )
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
                duration = time.time() - start_time
                if "_node_results" not in state:
                    state["_node_results"] = {}
                state["_node_results"][node.id] = NodeExecutionResult(
                    node_id=node.id,
                    success=True,
                    output=output,
                    duration_seconds=duration,
                    tool_calls_used=tool_calls_used,
                )

                # Emit NODE_COMPLETE if emitter is configured
                if emitter and hasattr(emitter, "emit_node_complete"):
                    emitter.emit_node_complete(
                        node.id,
                        node_name,
                        duration=duration,
                        output=output,
                    )

            except Exception as e:
                duration = time.time() - start_time
                correlation_id = str(uuid.uuid4())[:8]
                workflow_id = state.get("_workflow_name", "unknown")
                checkpoint_id = state.get("_checkpoint_id") or state.get("thread_id")

                # Log with correlation ID
                logger.error(
                    f"[{correlation_id}] Compute node '{node.id}' failed in workflow '{workflow_id}': {e}",
                    exc_info=True,
                )

                # Store error in state
                error_msg = f"Compute node '{node.id}' failed: {e}"
                state["_error"] = error_msg
                if "_node_results" not in state:
                    state["_node_results"] = {}
                state["_node_results"][node.id] = NodeExecutionResult(
                    node_id=node.id,
                    success=False,
                    error=str(e),
                    duration_seconds=duration,
                    tool_calls_used=tool_calls_used,
                )

                # Emit NODE_ERROR if emitter is configured
                if emitter and hasattr(emitter, "emit_node_error"):
                    emitter.emit_node_error(
                        node.id,
                        error=str(e),
                        node_name=node_name,
                        duration=duration,
                    )

                # Raise WorkflowExecutionError if not continuing on error
                if not getattr(node, "continue_on_error", False):
                    raise WorkflowExecutionError(
                        error_msg,
                        workflow_id=workflow_id,
                        node_id=node.id,
                        node_type="compute",
                        checkpoint_id=checkpoint_id,
                        execution_context={
                            "duration": duration,
                            "handler": node.handler if hasattr(node, "handler") else "unknown",
                            "tool_calls_used": tool_calls_used,
                        },
                        correlation_id=correlation_id,
                    ) from e

            return state

        # Wrap with retry if policy is configured
        if retry_policy:
            from victor.workflows.resilience import (
                retry_policy_to_strategy,
                RetryExecutor,
            )

            strategy = retry_policy_to_strategy(retry_policy)
            retry_executor = RetryExecutor(strategy)

            async def execute_compute_with_retry(state: Dict[str, Any]) -> Dict[str, Any]:
                """Wrapper that applies retry policy to compute execution."""
                from typing import cast

                result = await retry_executor.execute_async(lambda: execute_compute(state))
                if result.success:
                    return cast(Dict[str, Any], result.result)
                else:
                    # Return last state with error info
                    state = dict(state)
                    state["_retry_exhausted"] = True
                    state["_retry_attempts"] = result.attempts
                    state["_error"] = (
                        f"Compute node '{node.id}' failed after {result.attempts} attempts"
                    )
                    if emitter and hasattr(emitter, "emit_node_error"):
                        emitter.emit_node_error(
                            node.id,
                            error=f"Retry exhausted after {result.attempts} attempts",
                            node_name=node.name,
                            duration=0,
                        )
                    return state

            return execute_compute_with_retry

        return execute_compute

    def create_condition_router(
        self,
        node: "ConditionNode",
    ) -> Callable[[Dict[str, Any]], Any]:
        """Create a router function for a condition node.

        The router evaluates the condition and returns the state unchanged.
        Routing is handled by conditional edges in the graph.

        Recursion Context:
            This executor preserves the _recursion_context key in the state
            but does not actively use it. Condition nodes only route flow
            control and do not spawn nested workflows.
        """

        async def condition_exec(state: Dict[str, Any]) -> Dict[str, Any]:
            state = dict(state)
            if "_node_results" not in state:
                state["_node_results"] = {}
            state["_node_results"][node.id] = NodeExecutionResult(
                node_id=node.id,
                success=True,
                output={"passthrough": True},
            )
            return state

        return condition_exec

    def create_parallel_executor(
        self,
        node: "ParallelNode",
        child_nodes: Optional[List["WorkflowNode"]] = None,
    ) -> Callable[[Dict[str, Any]], Any]:
        """Create executor for a ParallelNode.

        Executes child nodes in true parallel using asyncio.gather().
        Each parallel branch gets an independent state copy.

        Recursion Context:
            The _recursion_context is shared across all parallel branches
            (via shallow copy in _copy_state_for_parallel). This ensures
            that all branches contribute to the same recursion depth tracking,
            preventing excessive nesting when multiple branches spawn workflows.

        Args:
            node: The ParallelNode definition
            child_nodes: Optional list of child node definitions
        """
        factory = self
        emitter = self._emitter

        # Pre-create executors for child nodes if provided
        child_executors = []
        if child_nodes:
            for child in child_nodes:
                child_executors.append((child, factory.create_executor(child)))

        async def execute_parallel(state: Dict[str, Any]) -> Dict[str, Any]:
            start_time = time.time()
            state = dict(state)
            node_name = node.name

            # Emit NODE_START if emitter is configured
            if emitter and hasattr(emitter, "emit_node_start"):
                emitter.emit_node_start(node.id, node_name)

            if "_parallel_results" not in state:
                state["_parallel_results"] = {}

            try:
                if child_executors:
                    # Execute all child nodes in true parallel with asyncio.gather
                    async def run_child(
                        child_node: "WorkflowNode", executor: Callable[..., Any]
                    ) -> tuple[str, bool, Any]:
                        # Use selective copy instead of full deepcopy for performance
                        # Shallow copies most keys, deep copies only mutable internal structures
                        child_state = factory._copy_state_for_parallel(state)
                        try:
                            result_state = await executor(child_state)
                            return (child_node.id, True, result_state)
                        except Exception as e:
                            return (child_node.id, False, str(e))

                    # Create tasks for all children
                    tasks = [run_child(child, executor) for child, executor in child_executors]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Process results
                    for result in results:
                        if isinstance(result, Exception):
                            logger.error(f"Parallel execution failed: {result}")
                            continue

                        if not isinstance(result, tuple) or len(result) != 3:
                            logger.error(f"Invalid parallel result format: {result}")
                            continue

                        child_id, success, result_data = result
                        if success and isinstance(result_data, dict):
                            # Merge state changes from parallel execution
                            for key, value in result_data.items():
                                if not key.startswith("_"):
                                    state[key] = value
                            state["_parallel_results"][child_id] = {
                                "success": True,
                                "output": result_data.get(child_id),
                            }
                        else:
                            state["_parallel_results"][child_id] = {
                                "success": False,
                                "error": str(result_data),
                            }

                # Apply join strategy
                if node.join_strategy == "all":
                    all_success = all(
                        r.get("success", False) for r in state["_parallel_results"].values()
                    )
                    if not all_success:
                        state["_error"] = "Not all parallel nodes succeeded"
                elif node.join_strategy == "any":
                    any_success = any(
                        r.get("success", False) for r in state["_parallel_results"].values()
                    )
                    if not any_success:
                        state["_error"] = "No parallel nodes succeeded"

                # Record parallel node result
                duration = time.time() - start_time
                if "_node_results" not in state:
                    state["_node_results"] = {}
                state["_node_results"][node.id] = NodeExecutionResult(
                    node_id=node.id,
                    success="_error" not in state,
                    output=state["_parallel_results"],
                    duration_seconds=duration,
                )

                # Emit NODE_COMPLETE if emitter is configured
                if emitter and hasattr(emitter, "emit_node_complete"):
                    emitter.emit_node_complete(
                        node.id,
                        node_name,
                        duration=duration,
                        output=state["_parallel_results"],
                    )

            except Exception as e:
                duration = time.time() - start_time
                correlation_id = str(uuid.uuid4())[:8]
                workflow_id = state.get("_workflow_name", "unknown")
                checkpoint_id = state.get("_checkpoint_id") or state.get("thread_id")

                # Log with correlation ID
                logger.error(
                    f"[{correlation_id}] Parallel node '{node.id}' failed in workflow '{workflow_id}': {e}",
                    exc_info=True,
                )

                # Store error in state
                error_msg = f"Parallel node '{node.id}' failed: {e}"
                state["_error"] = error_msg

                if emitter and hasattr(emitter, "emit_node_error"):
                    emitter.emit_node_error(
                        node.id,
                        error=str(e),
                        node_name=node_name,
                        duration=duration,
                    )

                # Raise WorkflowExecutionError if not continuing on error
                if not getattr(node, "continue_on_error", False):
                    raise WorkflowExecutionError(
                        error_msg,
                        workflow_id=workflow_id,
                        node_id=node.id,
                        node_type="parallel",
                        checkpoint_id=checkpoint_id,
                        execution_context={
                            "duration": duration,
                            "branches": len(node.branches) if hasattr(node, "branches") else 0,
                        },
                        correlation_id=correlation_id,
                    ) from e

            return state

        return execute_parallel

    def create_transform_executor(
        self,
        node: "TransformNode",
    ) -> Callable[[Dict[str, Any]], Any]:
        """Create executor for a TransformNode.

        Applies a transformation function to the workflow state.

        Recursion Context:
            This executor preserves the _recursion_context key in the state
            but does not actively use it. Transform nodes apply simple
            transformations and do not spawn nested workflows.
        """
        emitter = self._emitter

        async def execute_transform(state: Dict[str, Any]) -> Dict[str, Any]:
            start_time = time.time()
            state = dict(state)
            node_name = node.name

            # Emit NODE_START if emitter is configured
            if emitter and hasattr(emitter, "emit_node_start"):
                emitter.emit_node_start(node.id, node_name)

            try:
                # Execute transform function
                transformed = node.transform(state)

                # Merge transformed data back into state
                for key, value in transformed.items():
                    state[key] = value

                # Update node results
                duration = time.time() - start_time
                if "_node_results" not in state:
                    state["_node_results"] = {}
                state["_node_results"][node.id] = NodeExecutionResult(
                    node_id=node.id,
                    success=True,
                    output={"transformed_keys": list(transformed.keys())},
                    duration_seconds=duration,
                )

                # Emit NODE_COMPLETE if emitter is configured
                if emitter and hasattr(emitter, "emit_node_complete"):
                    emitter.emit_node_complete(
                        node.id,
                        node_name,
                        duration=duration,
                        output={"transformed_keys": list(transformed.keys())},
                    )

            except Exception as e:
                logger.error(f"Transform node '{node.id}' failed: {e}", exc_info=True)
                duration = time.time() - start_time
                state["_error"] = f"Transform node '{node.id}' failed: {e}"
                if "_node_results" not in state:
                    state["_node_results"] = {}
                state["_node_results"][node.id] = NodeExecutionResult(
                    node_id=node.id,
                    success=False,
                    error=str(e),
                    duration_seconds=duration,
                )

                if emitter and hasattr(emitter, "emit_node_error"):
                    emitter.emit_node_error(
                        node.id,
                        error=str(e),
                        node_name=node_name,
                        duration=duration,
                    )

            return state

        return execute_transform

    def create_team_executor(
        self,
        node: "TeamNodeWorkflow",
    ) -> Callable[[Dict[str, Any]], Any]:
        """Create executor for a TeamNode.

        Spawns an ad-hoc multi-agent team using victor/teams/ infrastructure
        and merges the result back into the workflow state.
        """
        from victor.framework.workflows.nodes import TeamNode
        from victor.framework.state_merging import MergeMode
        from victor.agent.subagents import SubAgentRole
        from victor.teams.types import TeamMember, TeamFormation
        from victor.workflows.team_node_runner import TeamNodeRunner
        from victor.workflows.recursion import RecursionContext

        orchestrator = self.orchestrator
        emitter = self._emitter
        tool_registry = self.tool_registry

        async def execute_team(state: Dict[str, Any]) -> Dict[str, Any]:
            start_time = time.time()
            state = dict(state)
            node_name = node.name

            # Emit NODE_START if emitter is configured
            if emitter and hasattr(emitter, "emit_node_start"):
                emitter.emit_node_start(node.id, node_name)

            try:
                # Extract recursion context from state
                recursion_ctx: Optional[RecursionContext] = state.get("_recursion_context")
                if recursion_ctx is None:
                    # Create new recursion context if not present
                    recursion_ctx = RecursionContext(max_depth=3)
                    logger.warning(
                        f"Team node '{node.id}' executing without parent recursion context, "
                        "created new context"
                    )

                # Use TeamNodeRunner for proper recursion tracking
                if orchestrator is None:
                    raise ValueError(f"Orchestrator is required for team node '{node.id}'")

                from victor.agent.orchestrator import AgentOrchestrator

                runner = TeamNodeRunner(
                    orchestrator=cast(AgentOrchestrator, orchestrator),
                    tool_registry=tool_registry,
                    enable_observability=bool(emitter),
                )

                # Execute team node with recursion context
                result = await runner.execute(
                    node=node,
                    context=state,
                    recursion_ctx=recursion_ctx,
                )

                # Update state with result
                duration = time.time() - start_time
                if "_node_results" not in state:
                    state["_node_results"] = {}

                state["_node_results"][node.id] = NodeExecutionResult(
                    node_id=node.id,
                    success=result.success,
                    output=result.output,
                    error=result.error,
                    duration_seconds=duration,
                    tool_calls_used=getattr(result, "tool_calls_used", 0),
                )

                # Merge output into state using output_key
                if result.success and result.output is not None:
                    if isinstance(result.output, dict):
                        state.update(result.output)
                    else:
                        state[node.output_key] = result.output

                # Emit NODE_COMPLETE if emitter is configured
                if emitter and hasattr(emitter, "emit_node_complete"):
                    emitter.emit_node_complete(
                        node.id,
                        node_name,
                        duration=duration,
                        output=result.output,
                    )

                return state

            except Exception as e:
                duration = time.time() - start_time
                correlation_id = str(uuid.uuid4())[:8]
                workflow_id = state.get("_workflow_name", "unknown")
                checkpoint_id = state.get("_checkpoint_id") or state.get("thread_id")

                # Log with correlation ID
                logger.error(
                    f"[{correlation_id}] Team node '{node.id}' failed in workflow '{workflow_id}': {e}",
                    exc_info=True,
                )

                # Store error in state
                error_msg = f"Team node '{node.id}' failed: {e}"
                state["_error"] = error_msg
                if "_node_results" not in state:
                    state["_node_results"] = {}
                state["_node_results"][node.id] = NodeExecutionResult(
                    node_id=node.id,
                    success=False,
                    error=str(e),
                    duration_seconds=duration,
                )

                # Emit NODE_ERROR if emitter is configured
                if emitter and hasattr(emitter, "emit_node_error"):
                    emitter.emit_node_error(
                        node.id,
                        error=str(e),
                        node_name=node_name,
                        duration=duration,
                    )

                # Raise WorkflowExecutionError if not continuing on error
                if not node.continue_on_error:
                    raise WorkflowExecutionError(
                        error_msg,
                        workflow_id=workflow_id,
                        node_id=node.id,
                        node_type="team",
                        checkpoint_id=checkpoint_id,
                        execution_context={
                            "duration": duration,
                            "team_formation": (
                                node.formation if hasattr(node, "formation") else "unknown"
                            ),
                        },
                        correlation_id=correlation_id,
                    ) from e

                # Return state with error if continuing on error
                return state

        return execute_team

    def _create_passthrough_executor(
        self,
        node: "WorkflowNode",
    ) -> Callable[[Dict[str, Any]], Any]:
        """Create a passthrough executor for unknown node types."""

        async def passthrough(state: Dict[str, Any]) -> Dict[str, Any]:
            state = dict(state)
            if "_node_results" not in state:
                state["_node_results"] = {}
            state["_node_results"][node.id] = NodeExecutionResult(
                node_id=node.id,
                success=True,
                output={"passthrough": True},
            )
            return state

        return passthrough


# =============================================================================
# Cached Compiled Graph Wrapper
# =============================================================================


@dataclass
class CachedCompiledGraph:
    """CompiledGraph wrapper with cache integration.

    Provides automatic cache lookup and storage for execution results,
    enabling efficient repeated workflow execution.

    Attributes:
        compiled_graph: The underlying CompiledGraph
        workflow_name: Name of the workflow
        source_path: Path to the source YAML file (if applicable)
        compiled_at: Timestamp when compilation occurred
        source_mtime: Modification time of source file (for invalidation)
        cache_key: Unique key for cache lookups
        max_execution_timeout_seconds: Workflow-level timeout
        default_node_timeout_seconds: Default timeout for nodes
        max_iterations: Maximum workflow iterations
        max_retries: Maximum retries for workflow
        max_recursion_depth: Maximum recursion depth for nested execution (default: 3)
    """

    compiled_graph: "CompiledGraph[Dict[str, Any]]"
    workflow_name: str
    source_path: Optional[Path] = None
    compiled_at: float = field(default_factory=time.time)
    source_mtime: Optional[float] = None
    cache_key: str = ""
    max_execution_timeout_seconds: Optional[float] = None
    default_node_timeout_seconds: Optional[float] = None
    max_iterations: int = 25
    max_retries: int = 0
    max_recursion_depth: int = 3

    async def invoke(
        self,
        input_state: Dict[str, Any],
        *,
        config: Optional["GraphConfig"] = None,
        thread_id: Optional[str] = None,
        use_cache: bool = True,
        max_recursion_depth: Optional[int] = None,
    ) -> "GraphExecutionResult[Dict[str, Any]]":
        """Execute the compiled workflow.

        Args:
            input_state: Initial state for execution
            config: Optional execution configuration override
            thread_id: Thread ID for checkpointing
            use_cache: Whether to use execution cache (for future use)
            max_recursion_depth: Override max recursion depth (optional)

        Returns:
            GraphExecutionResult with final state
        """
        from victor.framework.graph import GraphExecutionResult
        from victor.workflows.recursion import RecursionContext

        # Prepare state with metadata
        exec_state = self._prepare_state(input_state)
        exec_state["_max_iterations"] = self.max_iterations

        # Create recursion context with override or default depth
        depth = max_recursion_depth if max_recursion_depth is not None else self.max_recursion_depth
        recursion_ctx = RecursionContext(max_depth=depth)

        # Track workflow entry
        recursion_ctx.enter("workflow", self.workflow_name)

        try:
            # Execute with optional workflow-level timeout
            if self.max_execution_timeout_seconds:
                try:
                    result = await asyncio.wait_for(
                        self._execute_with_recursion(exec_state, config, thread_id, recursion_ctx),
                        timeout=self.max_execution_timeout_seconds,
                    )
                    return result
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Workflow '{self.workflow_name}' timed out after "
                        f"{self.max_execution_timeout_seconds}s"
                    )
                    return GraphExecutionResult(
                        state=exec_state,
                        success=False,
                        error=f"Workflow execution timed out after {self.max_execution_timeout_seconds}s",
                        iterations=0,
                        duration=self.max_execution_timeout_seconds,
                        node_history=[],
                    )
            else:
                return await self._execute_with_recursion(
                    exec_state, config, thread_id, recursion_ctx
                )
        finally:
            recursion_ctx.exit()

    async def _execute_with_recursion(
        self,
        exec_state: Dict[str, Any],
        config: Optional["GraphConfig"],
        thread_id: Optional[str],
        recursion_ctx: Any,
    ) -> "GraphExecutionResult[Dict[str, Any]]":
        """Execute compiled graph with recursion context.

        Args:
            exec_state: Prepared execution state
            config: Optional execution configuration override
            thread_id: Thread ID for checkpointing
            recursion_ctx: RecursionContext for tracking nesting depth

        Returns:
            GraphExecutionResult with final state
        """
        # Add recursion context to state for downstream access
        exec_state["_recursion_context"] = recursion_ctx

        # Execute the compiled graph
        return await self.compiled_graph.invoke(
            exec_state,
            config=config,
            thread_id=thread_id,
        )

    async def stream(
        self,
        input_state: Dict[str, Any],
        *,
        config: Optional["GraphConfig"] = None,
        thread_id: Optional[str] = None,
    ) -> AsyncIterator[Tuple[str, Dict[str, Any]]]:
        """Stream workflow execution yielding state after each node.

        Args:
            input_state: Initial state for execution
            config: Optional execution configuration override
            thread_id: Thread ID for checkpointing

        Yields:
            Tuple of (node_id, state) after each node execution
        """
        exec_state = self._prepare_state(input_state)

        async for node_id, state in self.compiled_graph.stream(
            exec_state,
            config=config,
            thread_id=thread_id,
        ):
            yield node_id, state

    def _prepare_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare initial state with workflow metadata."""
        exec_state = dict(state)
        if "_workflow_id" not in exec_state:
            exec_state["_workflow_id"] = uuid.uuid4().hex
        if "_workflow_name" not in exec_state:
            exec_state["_workflow_name"] = self.workflow_name
        if "_node_results" not in exec_state:
            exec_state["_node_results"] = {}
        if "_parallel_results" not in exec_state:
            exec_state["_parallel_results"] = {}
        return exec_state

    def get_graph_schema(self) -> Dict[str, Any]:
        """Get graph structure as dictionary.

        Returns:
            Dictionary describing nodes and edges
        """
        return self.compiled_graph.get_graph_schema()

    @property
    def age_seconds(self) -> float:
        """Get age of this cached compilation in seconds."""
        return time.time() - self.compiled_at


# =============================================================================
# Unified Workflow Compiler
# =============================================================================


class UnifiedWorkflowCompiler:
    """Unified compiler for all workflow types with integrated caching.

    Consolidates compilation paths for:
    - YAML workflow files
    - YAML workflow content strings
    - WorkflowDefinition objects
    - WorkflowGraph objects

    All paths produce CachedCompiledGraph instances that can be executed
    through the unified CompiledGraph.invoke() engine.

    Example:
        # Create compiler with caching
        compiler = UnifiedWorkflowCompiler(enable_caching=True)

        # Compile from YAML
        graph = compiler.compile_yaml(Path("workflow.yaml"), "my_workflow")

        # Compile from definition
        graph = compiler.compile_definition(my_definition)

        # Execute
        result = await graph.invoke({"input": "data"})

        # Check cache
        print(compiler.get_cache_stats())
    """

    def __init__(
        self,
        definition_cache: Optional["WorkflowDefinitionCache"] = None,
        execution_cache: Optional["WorkflowCacheManager"] = None,
        orchestrator: Optional["WorkflowAgentProtocol"] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        runner_registry: Optional["NodeRunnerRegistry"] = None,
        emitter: Optional[Any] = None,  # ObservabilityEmitter
        enable_caching: bool = True,
        cache_ttl: int = 3600,
        config: Optional[UnifiedCompilerConfig] = None,
    ) -> None:
        """Initialize the unified compiler.

        .. deprecated::
            Consider using the plugin architecture instead:
                from victor.workflows.create import create_compiler
                compiler = create_compiler("yaml://", enable_caching=True)

            The plugin architecture provides:
            - Third-party friendly extensibility
            - URI-based compiler selection (like SQLAlchemy)
            - Consistent protocol-based API

            UnifiedWorkflowCompiler continues to work and will be supported
            through v0.7.0. Migration guide: see MIGRATION_GUIDE.md

        Args:
            definition_cache: Cache for parsed YAML definitions
            execution_cache: Cache for execution results
            orchestrator: Agent orchestrator for agent nodes
            tool_registry: Tool registry for compute nodes
            runner_registry: NodeRunner registry for unified execution
            emitter: ObservabilityEmitter for streaming events
            enable_caching: Whether to enable caching (default: True)
            cache_ttl: Cache TTL in seconds (default: 3600)
            config: Full compiler configuration (overrides other params)
        """
        import warnings

        warnings.warn(
            "UnifiedWorkflowCompiler is deprecated but remains supported. "
            "Consider migrating to the plugin architecture for better extensibility: "
            "from victor.workflows.create import create_compiler; "
            "compiler = create_compiler('yaml://', enable_caching=True). "
            "See MIGRATION_GUIDE.md for details. "
            "This deprecation is informational only - UnifiedWorkflowCompiler will "
            "continue to work through v0.7.0.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Use config if provided, otherwise build from params
        if config:
            self._config = config
        else:
            self._config = UnifiedCompilerConfig(
                enable_caching=enable_caching,
                cache_ttl=cache_ttl,
            )

        # Store caches
        self._definition_cache = definition_cache
        self._execution_cache = execution_cache

        # Store dependencies
        self._orchestrator = orchestrator
        self._tool_registry = tool_registry or self._get_default_tool_registry()
        self._runner_registry = runner_registry
        self._emitter = emitter

        # Create node executor factory
        self._executor_factory = NodeExecutorFactory(
            orchestrator=self._orchestrator,
            tool_registry=self._tool_registry,
            runner_registry=self._runner_registry,
            emitter=self._emitter,
        )

        # Lazy-loaded compilers
        self._graph_compiler: Optional["WorkflowGraphCompiler[Any]"] = None
        self._definition_compiler: Optional["WorkflowDefinitionCompiler"] = None

        # Compilation stats
        self._compile_stats = {
            "yaml_compiles": 0,
            "yaml_content_compiles": 0,
            "definition_compiles": 0,
            "graph_compiles": 0,
            "cache_hits": 0,
        }

    def _get_default_tool_registry(self) -> Optional["ToolRegistry"]:
        """Get the default tool registry if available."""
        try:
            from victor.tools.registry import ToolRegistry
            from victor.core.container import ServiceContainer

            container = ServiceContainer()
            # Check if ToolRegistry is registered by attempting to get it
            try:
                return container.get(ToolRegistry)
            except Exception:
                return None
        except Exception:
            return None

    def _get_definition_cache(self) -> "WorkflowDefinitionCache":
        """Get or create definition cache."""
        if self._definition_cache is None:
            from victor.workflows.cache import get_workflow_definition_cache

            self._definition_cache = get_workflow_definition_cache()
        return self._definition_cache

    def _get_execution_cache(self) -> "WorkflowCacheManager":
        """Get or create execution cache."""
        if self._execution_cache is None:
            from victor.workflows.cache import get_workflow_cache_manager

            self._execution_cache = get_workflow_cache_manager()
        return self._execution_cache

    def _get_graph_compiler(self) -> "WorkflowGraphCompiler[Any]":
        """Get or create WorkflowGraph compiler."""
        if self._graph_compiler is None:
            from victor.workflows.graph_compiler import (
                WorkflowGraphCompiler,
                CompilerConfig,
            )

            config = CompilerConfig(
                use_node_runners=self._runner_registry is not None,
                runner_registry=self._runner_registry,
                validate_before_compile=self._config.validate_before_compile,
                preserve_state_type=self._config.preserve_state_type,
                emitter=self._emitter if self._config.enable_observability else None,
                enable_observability=self._config.enable_observability,
            )
            self._graph_compiler = WorkflowGraphCompiler(config)
        return self._graph_compiler

    def _get_definition_compiler(self) -> "WorkflowDefinitionCompiler":
        """Get or create WorkflowDefinition compiler."""
        if self._definition_compiler is None:
            from victor.workflows.graph_compiler import WorkflowDefinitionCompiler

            self._definition_compiler = WorkflowDefinitionCompiler(
                runner_registry=self._runner_registry,
            )
        return self._definition_compiler

    def _compute_config_hash(
        self,
        condition_registry: Optional[Dict[str, Callable[..., Any]]],
        transform_registry: Optional[Dict[str, Callable[..., Any]]],
    ) -> int:
        """Compute hash for cache key based on registries."""
        condition_names = tuple(sorted(condition_registry.keys())) if condition_registry else ()
        transform_names = tuple(sorted(transform_registry.keys())) if transform_registry else ()
        return hash((condition_names, transform_names))

    # =========================================================================
    # YAML Compilation
    # =========================================================================

    def compile_yaml(
        self,
        yaml_path: Union[str, Path],
        workflow_name: Optional[str] = None,
        condition_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        transform_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        **kwargs: Any,
    ) -> CachedCompiledGraph:
        """Compile a workflow from a YAML file.

        Uses the definition cache to avoid redundant parsing, then compiles
        the WorkflowDefinition to a CompiledGraph for execution.

        Args:
            yaml_path: Path to the YAML file
            workflow_name: Specific workflow to compile (if file has multiple)
            condition_registry: Custom condition functions (escape hatches)
            transform_registry: Custom transform functions (escape hatches)
            **kwargs: Additional compilation options

        Returns:
            CachedCompiledGraph ready for execution

        Raises:
            FileNotFoundError: If YAML file not found
            ValueError: If workflow validation fails
        """
        from victor.workflows.yaml_loader import (
            YAMLWorkflowConfig,
            load_workflow_from_file,
        )

        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"YAML file not found: {path}")

        name = workflow_name or "default"
        self._compute_config_hash(condition_registry, transform_registry)

        # Generate cache key
        cache_key = self._generate_yaml_cache_key(path, workflow_name)

        # Get source file mtime for cache validation
        try:
            source_mtime = path.stat().st_mtime
        except (OSError, FileNotFoundError):
            source_mtime = None

        # Check definition cache
        if self._config.enable_caching:
            cache = self._get_definition_cache()
            cached_def = cache.get(cache_key)
            if cached_def is not None:
                self._compile_stats["cache_hits"] += 1
                logger.debug(f"Definition cache hit for {path}:{workflow_name}")
                compiled = self._get_definition_compiler().compile(cached_def)
                return CachedCompiledGraph(
                    compiled_graph=compiled,
                    workflow_name=name,
                    source_path=path,
                    source_mtime=source_mtime,
                    cache_key=cache_key,
                    max_execution_timeout_seconds=cached_def.max_execution_timeout_seconds,
                    default_node_timeout_seconds=cached_def.default_node_timeout_seconds,
                    max_iterations=cached_def.max_iterations,
                    max_retries=cached_def.max_retries,
                    max_recursion_depth=cached_def.metadata.get("max_recursion_depth", 3),
                )

        # Load and parse YAML
        config = YAMLWorkflowConfig(
            condition_registry=condition_registry or {},
            transform_registry=transform_registry or {},
            base_dir=path.parent,
        )

        result = load_workflow_from_file(
            str(yaml_path),
            workflow_name=workflow_name,
            config=config,
        )

        # Handle dict or single workflow result
        if isinstance(result, dict):
            if not result:
                raise ConfigurationValidationError(
                    message="No workflows found in YAML file",
                    config_key=str(yaml_path),
                    recovery_hint="Check that the YAML file contains at least one workflow definition. Ensure the 'workflows:' key exists.",
                )
            workflow_def = next(iter(result.values()))
        else:
            workflow_def = result

        # Cache the definition if enabled
        if self._config.enable_caching:
            cache = self._get_definition_cache()
            cache.set(cache_key, workflow_def)
            logger.debug(f"Cached workflow definition: {name} from {path}")

        self._compile_stats["yaml_compiles"] += 1

        # Compile to CompiledGraph
        compiled = self._get_definition_compiler().compile(workflow_def)
        return CachedCompiledGraph(
            compiled_graph=compiled,
            workflow_name=name,
            source_path=path,
            source_mtime=source_mtime,
            cache_key=cache_key,
            max_execution_timeout_seconds=workflow_def.max_execution_timeout_seconds,
            default_node_timeout_seconds=workflow_def.default_node_timeout_seconds,
            max_iterations=workflow_def.max_iterations,
            max_retries=workflow_def.max_retries,
            max_recursion_depth=workflow_def.metadata.get("max_recursion_depth", 3),
        )

    def compile_yaml_content(
        self,
        yaml_content: str,
        workflow_name: str,
        condition_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        transform_registry: Optional[Dict[str, Callable[..., Any]]] = None,
        **kwargs: Any,
    ) -> CachedCompiledGraph:
        """Compile a workflow from YAML content string.

        Args:
            yaml_content: YAML content as string
            workflow_name: Name of workflow to compile
            condition_registry: Custom condition functions (escape hatches)
            transform_registry: Custom transform functions (escape hatches)
            **kwargs: Additional compilation options

        Returns:
            CachedCompiledGraph ready for execution
        """
        from victor.workflows.yaml_loader import (
            YAMLWorkflowConfig,
            load_workflow_from_yaml,
        )

        # Generate cache key from content hash
        cache_key = self._generate_content_cache_key(yaml_content, workflow_name)

        # Load and parse YAML
        config = YAMLWorkflowConfig(
            condition_registry=condition_registry or {},
            transform_registry=transform_registry or {},
        )
        result = load_workflow_from_yaml(yaml_content, workflow_name, config)

        if isinstance(result, dict):
            workflow_def = result.get(workflow_name) or next(iter(result.values()))
        else:
            workflow_def = result

        self._compile_stats["yaml_content_compiles"] += 1

        # Compile to CompiledGraph
        compiled = self._get_definition_compiler().compile(workflow_def)
        return CachedCompiledGraph(
            compiled_graph=compiled,
            workflow_name=workflow_name,
            cache_key=cache_key,
            max_execution_timeout_seconds=workflow_def.max_execution_timeout_seconds,
            default_node_timeout_seconds=workflow_def.default_node_timeout_seconds,
            max_iterations=workflow_def.max_iterations,
            max_retries=workflow_def.max_retries,
            max_recursion_depth=workflow_def.metadata.get("max_recursion_depth", 3),
        )

    # =========================================================================
    # Definition Compilation
    # =========================================================================

    def compile_definition(
        self,
        definition: "WorkflowDefinition",
        cache_key: Optional[str] = None,
        **kwargs: Any,
    ) -> CachedCompiledGraph:
        """Compile a WorkflowDefinition to a CachedCompiledGraph.

        Args:
            definition: The workflow definition to compile
            cache_key: Optional cache key (generated if not provided)
            **kwargs: Additional compilation options

        Returns:
            CachedCompiledGraph ready for execution

        Raises:
            ConfigurationValidationError: If workflow validation fails
        """
        # Validate if configured
        if self._config.validate_before_compile:
            errors = definition.validate()
            if errors:
                # Extract field names from error messages for better error reporting
                invalid_fields = []
                field_errors = {}
                for error in errors:
                    # Try to extract field name from error message
                    # Format: "Node 'node_id' has error" or "Field 'field_name' error"
                    import re

                    match = re.search(r"'([^']+)'", error)
                    if match:
                        field_name = match.group(1)
                        invalid_fields.append(field_name)
                        field_errors[field_name] = error

                raise ConfigurationValidationError(
                    message=f"Workflow '{definition.name}' validation failed with {len(errors)} error(s)",
                    config_key=definition.name,
                    invalid_fields=invalid_fields,
                    field_errors=field_errors,
                    validation_errors=errors,
                    recovery_hint=f"Fix validation errors in workflow '{definition.name}'. Use 'victor workflow validate <path>' to check.",
                )

        # Generate cache key if not provided
        if not cache_key:
            cache_key = self._generate_definition_cache_key(definition)

        self._compile_stats["definition_compiles"] += 1

        # Compile to CompiledGraph
        compiled = self._get_definition_compiler().compile(definition)
        return CachedCompiledGraph(
            compiled_graph=compiled,
            workflow_name=definition.name,
            cache_key=cache_key,
            max_execution_timeout_seconds=definition.max_execution_timeout_seconds,
            default_node_timeout_seconds=definition.default_node_timeout_seconds,
            max_iterations=definition.max_iterations,
            max_retries=definition.max_retries,
            max_recursion_depth=definition.metadata.get("max_recursion_depth", 3),
        )

    # =========================================================================
    # WorkflowGraph Compilation
    # =========================================================================

    def compile_graph(
        self,
        graph: "WorkflowGraph[Any]",
        name: Optional[str] = None,
        cache_key: Optional[str] = None,
        **kwargs: Any,
    ) -> CachedCompiledGraph:
        """Compile a WorkflowGraph (graph_dsl) to CachedCompiledGraph.

        Args:
            graph: The WorkflowGraph to compile
            name: Optional name override
            cache_key: Optional cache key
            **kwargs: Additional compilation options

        Returns:
            CachedCompiledGraph ready for execution
        """
        # Generate cache key
        if not cache_key:
            cache_key = self._generate_graph_cache_key(graph, name)

        self._compile_stats["graph_compiles"] += 1
        workflow_name = name or getattr(graph, "name", "workflow_graph") or "workflow_graph"

        # Compile to CompiledGraph
        compiled = self._get_graph_compiler().compile(graph, name)
        return CachedCompiledGraph(
            compiled_graph=compiled,
            workflow_name=workflow_name,
            cache_key=cache_key or "",
        )

    # =========================================================================
    # Cache Management
    # =========================================================================

    def clear_cache(self) -> int:
        """Clear all caches.

        Returns:
            Total number of entries cleared
        """
        total = 0

        # Clear definition cache
        if self._definition_cache is not None:
            total += self._definition_cache.clear()
        else:
            from victor.workflows.cache import get_workflow_definition_cache

            total += get_workflow_definition_cache().clear()

        # Clear execution cache
        if self._execution_cache is not None:
            total += self._execution_cache.clear_all()
        else:
            from victor.workflows.cache import get_workflow_cache_manager

            total += get_workflow_cache_manager().clear_all()

        logger.info(f"Cleared {total} cache entries")
        return total

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dictionary with cache and compilation statistics
        """
        stats: Dict[str, Any] = {
            "compilation": dict(self._compile_stats),
            "caching_enabled": self._config.enable_caching,
        }

        # Get definition cache stats
        if self._config.enable_caching:
            if self._definition_cache is not None:
                stats["definition_cache"] = self._definition_cache.get_stats()
            else:
                from victor.workflows.cache import get_workflow_definition_cache

                stats["definition_cache"] = get_workflow_definition_cache().get_stats()

            # Get execution cache stats
            if self._execution_cache is not None:
                stats["execution_cache"] = self._execution_cache.get_all_stats()
            else:
                from victor.workflows.cache import get_workflow_cache_manager

                stats["execution_cache"] = get_workflow_cache_manager().get_all_stats()
        else:
            stats["definition_cache"] = {"enabled": False}
            stats["execution_cache"] = {}

        return stats

    def invalidate_yaml(self, yaml_path: Union[str, Path]) -> int:
        """Invalidate cached definitions for a specific YAML file.

        Args:
            yaml_path: Path to YAML file to invalidate

        Returns:
            Number of cache entries invalidated
        """
        if not self._config.enable_caching:
            return 0

        path = Path(yaml_path)
        cache = self._get_definition_cache()
        count = cache.invalidate(path)
        if count > 0:
            logger.info(f"Invalidated {count} cache entries for: {path}")
        return count

    def set_runner_registry(self, registry: "NodeRunnerRegistry") -> None:
        """Set the NodeRunner registry for execution.

        Args:
            registry: NodeRunnerRegistry with configured runners
        """
        self._runner_registry = registry
        # Reset compilers to use new registry
        self._graph_compiler = None
        self._definition_compiler = None
        # Update executor factory
        self._executor_factory._runner_registry = registry

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _generate_yaml_cache_key(
        self,
        yaml_path: Path,
        workflow_name: Optional[str],
    ) -> str:
        """Generate cache key for YAML file compilation."""
        try:
            mtime = yaml_path.stat().st_mtime
        except OSError:
            mtime = 0

        key_data = f"{yaml_path.resolve()}:{workflow_name or ''}:{mtime}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _generate_content_cache_key(
        self,
        content: str,
        workflow_name: str,
    ) -> str:
        """Generate cache key for YAML content compilation."""
        key_data = f"content:{workflow_name}:{content}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _generate_definition_cache_key(
        self,
        definition: "WorkflowDefinition",
    ) -> str:
        """Generate cache key for definition compilation."""
        # Use definition's serialized form for key
        key_data = json.dumps(definition.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _generate_graph_cache_key(
        self,
        graph: "WorkflowGraph[Any]",
        name: Optional[str],
    ) -> str:
        """Generate cache key for WorkflowGraph compilation."""
        # Use node and edge structure for key
        key_parts = [
            name or "",
            str(sorted(graph._nodes.keys())),
            str(sorted(graph._edges.keys())),
        ]
        key_data = ":".join(key_parts)
        return hashlib.sha256(key_data.encode()).hexdigest()


# =============================================================================
# Convenience Functions
# =============================================================================


def compile_workflow(
    source: Union[Path, str, "WorkflowDefinition"],
    workflow_name: Optional[str] = None,
    enable_caching: bool = True,
    **kwargs: Any,
) -> CachedCompiledGraph:
    """Compile a workflow from various sources.

    Convenience function for one-off compilation.

    Args:
        source: YAML path, YAML content string, or WorkflowDefinition
        workflow_name: Name of workflow (required for YAML content)
        enable_caching: Whether to enable caching
        **kwargs: Additional compilation options

    Returns:
        CachedCompiledGraph ready for execution
    """
    compiler = UnifiedWorkflowCompiler(enable_caching=enable_caching)

    if isinstance(source, Path):
        return compiler.compile_yaml(source, workflow_name, **kwargs)
    elif isinstance(source, str):
        # Check if it's a file path or YAML content
        if Path(source).exists():
            return compiler.compile_yaml(Path(source), workflow_name, **kwargs)
        else:
            if not workflow_name:
                raise ConfigurationValidationError(
                    message="workflow_name required for YAML content",
                    config_key="<yaml_content>",
                    recovery_hint="Provide workflow_name parameter when compiling YAML content string.",
                )
            return compiler.compile_yaml_content(source, workflow_name, **kwargs)
    else:
        # Assume WorkflowDefinition
        return compiler.compile_definition(source, **kwargs)


async def compile_and_execute(
    source: Union[Path, str, "WorkflowDefinition"],
    initial_state: Optional[Dict[str, Any]] = None,
    workflow_name: Optional[str] = None,
    **kwargs: Any,
) -> "GraphExecutionResult[Dict[str, Any]]":
    """Compile and execute a workflow in one step.

    Convenience function for one-off execution.

    Args:
        source: YAML path, YAML content string, or WorkflowDefinition
        initial_state: Initial workflow state
        workflow_name: Name of workflow (required for YAML content)
        **kwargs: Additional compilation/execution options

    Returns:
        GraphExecutionResult with final state
    """
    graph = compile_workflow(source, workflow_name, **kwargs)
    return await graph.invoke(initial_state or {})


def create_unified_compiler(
    enable_caching: bool = True,
    runner_registry: Optional["NodeRunnerRegistry"] = None,
    **kwargs: Any,
) -> UnifiedWorkflowCompiler:
    """Create a UnifiedWorkflowCompiler with default caches.

    Args:
        enable_caching: Whether to enable caching
        runner_registry: Optional NodeRunner registry
        **kwargs: Additional configuration options

    Returns:
        Configured UnifiedWorkflowCompiler instance
    """
    from victor.workflows.cache import (
        get_workflow_definition_cache,
        get_workflow_cache_manager,
    )

    return UnifiedWorkflowCompiler(
        definition_cache=get_workflow_definition_cache(),
        execution_cache=get_workflow_cache_manager(),
        runner_registry=runner_registry,
        enable_caching=enable_caching,
        **kwargs,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "UnifiedCompilerConfig",
    # Execution result
    "NodeExecutionResult",
    # Factory
    "NodeExecutorFactory",
    # Cached graph
    "CachedCompiledGraph",
    # Main compiler
    "UnifiedWorkflowCompiler",
    # Convenience functions
    "compile_workflow",
    "compile_and_execute",
    "create_unified_compiler",
]
