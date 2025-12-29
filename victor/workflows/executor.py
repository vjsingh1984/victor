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

"""Workflow executor for running workflow definitions.

Provides the execution engine that traverses workflow DAGs and
executes agent nodes using the SubAgent infrastructure.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from victor.workflows.definition import (
    AgentNode,
    ConditionNode,
    ParallelNode,
    TransformNode,
    WorkflowDefinition,
    WorkflowNode,
)

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.subagents import SubAgentOrchestrator
    from victor.agent.rl.checkpoint_store import CheckpointStore
    from victor.workflows.cache import WorkflowCache, WorkflowCacheConfig

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Execution status of a workflow node."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeResult:
    """Result from executing a workflow node.

    Attributes:
        node_id: ID of the executed node
        status: Execution status
        output: Output data (for agent nodes: agent result)
        error: Error message if failed
        duration_seconds: Execution time
        tool_calls_used: Tool calls made (for agent nodes)
    """

    node_id: str
    status: NodeStatus
    output: Optional[Any] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    tool_calls_used: int = 0

    @property
    def success(self) -> bool:
        """Check if node completed successfully."""
        return self.status == NodeStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "tool_calls_used": self.tool_calls_used,
        }


@dataclass
class WorkflowContext:
    """Execution context for a workflow.

    Maintains shared state across workflow nodes and provides
    utilities for accessing and updating context data.

    Attributes:
        data: Shared context data
        node_results: Results from executed nodes
        metadata: Execution metadata
    """

    data: Dict[str, Any] = field(default_factory=dict)
    node_results: Dict[str, NodeResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a context value."""
        self.data[key] = value

    def update(self, values: Dict[str, Any]) -> None:
        """Update multiple context values."""
        self.data.update(values)

    def get_result(self, node_id: str) -> Optional[NodeResult]:
        """Get result for a specific node."""
        return self.node_results.get(node_id)

    def add_result(self, result: NodeResult) -> None:
        """Add a node result."""
        self.node_results[result.node_id] = result

    def has_failures(self) -> bool:
        """Check if any nodes failed."""
        return any(r.status == NodeStatus.FAILED for r in self.node_results.values())

    def get_outputs(self) -> Dict[str, Any]:
        """Get all successful node outputs."""
        return {
            node_id: result.output
            for node_id, result in self.node_results.items()
            if result.success and result.output is not None
        }


@dataclass
class WorkflowResult:
    """Result from executing a complete workflow.

    Attributes:
        workflow_name: Name of the executed workflow
        success: Whether workflow completed successfully
        context: Final execution context
        total_duration: Total execution time
        total_tool_calls: Total tool calls across all agents
        error: Error message if failed
    """

    workflow_name: str
    success: bool
    context: WorkflowContext
    total_duration: float = 0.0
    total_tool_calls: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "workflow_name": self.workflow_name,
            "success": self.success,
            "total_duration": self.total_duration,
            "total_tool_calls": self.total_tool_calls,
            "error": self.error,
            "outputs": self.context.get_outputs(),
            "node_results": {
                nid: r.to_dict() for nid, r in self.context.node_results.items()
            },
        }

    def get_output(self, node_id: str) -> Optional[Any]:
        """Get output from a specific node."""
        result = self.context.get_result(node_id)
        return result.output if result else None


class WorkflowExecutor:
    """Executes workflow definitions with optional checkpointing and caching.

    Traverses the workflow DAG and executes nodes using the
    SubAgent infrastructure. Supports checkpointing for workflow
    resumption via the existing RL CheckpointStore.

    Supports optional node-level caching for deterministic nodes
    (TransformNode, ConditionNode) to improve performance on
    repeated workflow executions.

    Attributes:
        orchestrator: Agent orchestrator for spawning agents
        max_parallel: Maximum parallel node executions
        default_timeout: Default timeout per node (seconds)
        checkpointer: Optional CheckpointStore for persistence
        cache: Optional WorkflowCache for node result caching

    Example:
        executor = WorkflowExecutor(orchestrator)
        result = await executor.execute(workflow, {"files": ["main.py"]})

        if result.success:
            print(result.get_output("analyze"))

        # With checkpointing for resumption:
        from victor.agent.rl.checkpoint_store import get_checkpoint_store
        executor = WorkflowExecutor(orchestrator, checkpointer=get_checkpoint_store())
        result = await executor.execute(
            workflow,
            initial_context={"files": ["main.py"]},
            thread_id="my-workflow-123",
        )

        # With caching enabled for deterministic nodes:
        from victor.workflows.cache import WorkflowCache, WorkflowCacheConfig
        cache_config = WorkflowCacheConfig(enabled=True, ttl_seconds=3600)
        executor = WorkflowExecutor(
            orchestrator,
            cache=WorkflowCache(cache_config),
        )
    """

    def __init__(
        self,
        orchestrator: "AgentOrchestrator",
        *,
        max_parallel: int = 4,
        default_timeout: float = 300.0,
        checkpointer: Optional["CheckpointStore"] = None,
        cache: Optional["WorkflowCache"] = None,
        cache_config: Optional["WorkflowCacheConfig"] = None,
    ):
        """Initialize executor.

        Args:
            orchestrator: Agent orchestrator instance
            max_parallel: Maximum parallel executions
            default_timeout: Default timeout per node
            checkpointer: Optional CheckpointStore for persistence
            cache: Optional WorkflowCache for node result caching
            cache_config: Optional config to create a cache (alternative to cache param)
        """
        self.orchestrator = orchestrator
        self.max_parallel = max_parallel
        self.default_timeout = default_timeout
        self._checkpointer = checkpointer
        self._sub_agents: Optional["SubAgentOrchestrator"] = None
        self._active_executions: Dict[str, asyncio.Task] = {}

        # Initialize cache if config provided
        if cache is not None:
            self._cache = cache
        elif cache_config is not None:
            from victor.workflows.cache import WorkflowCache
            self._cache: Optional["WorkflowCache"] = WorkflowCache(cache_config)
        else:
            self._cache = None

    @property
    def sub_agents(self) -> "SubAgentOrchestrator":
        """Get or create SubAgentOrchestrator."""
        if self._sub_agents is None:
            from victor.agent.subagents import SubAgentOrchestrator

            self._sub_agents = SubAgentOrchestrator(self.orchestrator)
        return self._sub_agents

    @property
    def cache(self) -> Optional["WorkflowCache"]:
        """Get the workflow cache (if enabled)."""
        return self._cache

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats, or empty dict if cache disabled
        """
        if self._cache is not None:
            return self._cache.get_stats()
        return {"enabled": False}

    async def execute(
        self,
        workflow: WorkflowDefinition,
        initial_context: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
        thread_id: Optional[str] = None,
    ) -> WorkflowResult:
        """Execute a workflow with optional checkpointing.

        Args:
            workflow: Workflow definition to execute
            initial_context: Initial context data
            timeout: Overall timeout (None = no limit)
            thread_id: Thread ID for checkpointing (enables resume)

        Returns:
            WorkflowResult with execution outcome
        """
        execution_id = uuid.uuid4().hex[:8]
        thread_id = thread_id or execution_id

        logger.info(
            f"Starting workflow '{workflow.name}' (execution_id={execution_id}, thread_id={thread_id})"
        )

        # Check for checkpoint to resume from
        resume_from_node: Optional[str] = None
        if self._checkpointer:
            checkpoint = self._checkpointer.get_latest_checkpoint(f"workflow_{thread_id}")
            if checkpoint:
                logger.info(f"Resuming from checkpoint at node: {checkpoint.state.get('last_node')}")
                initial_context = checkpoint.state.get("context", {})
                resume_from_node = checkpoint.state.get("next_node")

        context = WorkflowContext(
            data=initial_context.copy() if initial_context else {},
            metadata={
                "execution_id": execution_id,
                "workflow_name": workflow.name,
                "thread_id": thread_id,
            },
        )

        start_time = time.time()

        try:
            if timeout:
                await asyncio.wait_for(
                    self._execute_workflow(
                        workflow, context, thread_id, resume_from_node
                    ),
                    timeout=timeout,
                )
            else:
                await self._execute_workflow(
                    workflow, context, thread_id, resume_from_node
                )

            total_duration = time.time() - start_time
            total_tool_calls = sum(
                r.tool_calls_used for r in context.node_results.values()
            )

            success = not context.has_failures()

            logger.info(
                f"Workflow '{workflow.name}' {'completed' if success else 'failed'} "
                f"in {total_duration:.2f}s ({total_tool_calls} tool calls)"
            )

            # Emit RL event for workflow completion
            self._emit_workflow_completed_event(
                workflow_name=workflow.name,
                success=success,
                duration=total_duration,
                tool_calls=total_tool_calls,
            )

            return WorkflowResult(
                workflow_name=workflow.name,
                success=success,
                context=context,
                total_duration=total_duration,
                total_tool_calls=total_tool_calls,
            )

        except asyncio.TimeoutError:
            logger.error(f"Workflow '{workflow.name}' timed out after {timeout}s")
            return WorkflowResult(
                workflow_name=workflow.name,
                success=False,
                context=context,
                total_duration=time.time() - start_time,
                error=f"Workflow timed out after {timeout}s",
            )

        except Exception as e:
            logger.error(f"Workflow '{workflow.name}' failed: {e}", exc_info=True)
            return WorkflowResult(
                workflow_name=workflow.name,
                success=False,
                context=context,
                total_duration=time.time() - start_time,
                error=str(e),
            )

    async def _execute_workflow(
        self,
        workflow: WorkflowDefinition,
        context: WorkflowContext,
        thread_id: str = "",
        resume_from_node: Optional[str] = None,
    ) -> None:
        """Execute the workflow DAG with optional checkpoint resume.

        Args:
            workflow: Workflow to execute
            context: Execution context
            thread_id: Thread ID for checkpointing
            resume_from_node: Node to resume from (if resuming)
        """
        if not workflow.start_node:
            raise ValueError("Workflow has no start node")

        # Track executed nodes to prevent loops
        executed: Set[str] = set()

        # Resume from checkpoint node or start
        start_node = resume_from_node or workflow.start_node
        to_execute: List[str] = [start_node]

        while to_execute:
            node_id = to_execute.pop(0)

            if node_id in executed:
                continue

            node = workflow.get_node(node_id)
            if not node:
                logger.warning(f"Node '{node_id}' not found in workflow")
                continue

            # Execute the node
            result = await self._execute_node(node, context)
            context.add_result(result)
            executed.add(node_id)

            # Determine next nodes
            next_nodes = self._get_next_nodes(node, context)

            # Save checkpoint after each node (for resumption)
            if self._checkpointer and thread_id:
                self._save_workflow_checkpoint(
                    thread_id=thread_id,
                    workflow_name=workflow.name,
                    last_node=node_id,
                    next_node=next_nodes[0] if next_nodes else None,
                    context_data=dict(context.data),
                )

            # Emit RL event for workflow step
            self._emit_workflow_step_event(
                workflow_name=workflow.name,
                node_id=node_id,
                node_type=node.node_type.value if hasattr(node, 'node_type') else 'unknown',
                success=result.status == NodeStatus.COMPLETED,
                duration=result.duration_seconds,
            )

            # If failed, stop unless configured to continue
            if result.status == NodeStatus.FAILED:
                if not workflow.metadata.get("continue_on_failure", False):
                    logger.warning(f"Stopping workflow due to node failure: {node_id}")
                    break

            to_execute.extend(next_nodes)

    def _get_next_nodes(
        self,
        node: WorkflowNode,
        context: WorkflowContext,
    ) -> List[str]:
        """Get next nodes to execute based on node type.

        Args:
            node: Current node
            context: Execution context

        Returns:
            List of next node IDs
        """
        if isinstance(node, ConditionNode):
            # Evaluate condition to determine branch
            next_id = node.evaluate(context.data)
            return [next_id] if next_id else []

        elif isinstance(node, ParallelNode):
            # For parallel nodes, the parallel_nodes are executed
            # during _execute_node; we return next_nodes for continuation
            return node.next_nodes

        else:
            # Standard linear flow
            return node.next_nodes

    async def _execute_node(
        self,
        node: WorkflowNode,
        context: WorkflowContext,
    ) -> NodeResult:
        """Execute a single workflow node.

        Checks cache for cacheable nodes before execution.
        Caches successful results for deterministic nodes.

        Args:
            node: Node to execute
            context: Execution context

        Returns:
            NodeResult with execution outcome
        """
        logger.debug(f"Executing node: {node.id} ({node.node_type.value})")
        start_time = time.time()

        # Check cache for cacheable nodes (TransformNode, ConditionNode)
        if self._cache is not None:
            cached_result = self._cache.get(node, context.data)
            if cached_result is not None:
                logger.debug(f"Cache hit for node: {node.id}")
                # Return cached result with updated timing
                return NodeResult(
                    node_id=cached_result.node_id,
                    status=cached_result.status,
                    output=cached_result.output,
                    error=cached_result.error,
                    duration_seconds=time.time() - start_time,  # Cache lookup time
                    tool_calls_used=cached_result.tool_calls_used,
                )

        try:
            if isinstance(node, AgentNode):
                result = await self._execute_agent_node(node, context, start_time)

            elif isinstance(node, ConditionNode):
                result = await self._execute_condition_node(node, context, start_time)

            elif isinstance(node, ParallelNode):
                result = await self._execute_parallel_node(node, context, start_time)

            elif isinstance(node, TransformNode):
                result = await self._execute_transform_node(node, context, start_time)

            else:
                # Unknown node type - skip
                result = NodeResult(
                    node_id=node.id,
                    status=NodeStatus.SKIPPED,
                    duration_seconds=time.time() - start_time,
                )

            # Cache successful results for cacheable nodes
            if self._cache is not None and result.success:
                self._cache.set(node, context.data, result)

            return result

        except Exception as e:
            logger.error(f"Node '{node.id}' failed: {e}", exc_info=True)
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    async def _execute_agent_node(
        self,
        node: AgentNode,
        context: WorkflowContext,
        start_time: float,
    ) -> NodeResult:
        """Execute an agent node.

        Args:
            node: Agent node to execute
            context: Execution context
            start_time: Node start time

        Returns:
            NodeResult with agent output
        """
        from victor.agent.subagents import SubAgentRole

        # Build agent task from goal and context
        task = self._build_agent_task(node, context)

        # Map role string to SubAgentRole
        role_map = {
            "researcher": SubAgentRole.RESEARCHER,
            "planner": SubAgentRole.PLANNER,
            "executor": SubAgentRole.EXECUTOR,
            "reviewer": SubAgentRole.REVIEWER,
            "tester": SubAgentRole.TESTER,
        }
        role = role_map.get(node.role.lower(), SubAgentRole.EXECUTOR)

        # Spawn and execute agent
        result = await self.sub_agents.spawn(
            role=role,
            task=task,
            tool_budget=node.tool_budget,
            allowed_tools=node.allowed_tools,
            timeout_seconds=int(self.default_timeout),
        )

        # Store output in context
        if result.success and result.summary:
            context.set(node.output_key or node.id, result.summary)

        return NodeResult(
            node_id=node.id,
            status=NodeStatus.COMPLETED if result.success else NodeStatus.FAILED,
            output=result.summary,
            error=result.error,
            duration_seconds=time.time() - start_time,
            tool_calls_used=result.tool_calls_used,
        )

    def _build_agent_task(self, node: AgentNode, context: WorkflowContext) -> str:
        """Build task string for an agent node.

        Args:
            node: Agent node
            context: Execution context

        Returns:
            Task description for agent
        """
        lines = [node.goal]

        # Add mapped inputs from context
        if node.input_mapping:
            lines.append("\n## Context")
            for param, key in node.input_mapping.items():
                value = context.get(key)
                if value is not None:
                    lines.append(f"- **{param}**: {value}")

        # Add outputs from previous nodes
        outputs = context.get_outputs()
        if outputs:
            lines.append("\n## Previous Results")
            for output_key, output_value in outputs.items():
                if isinstance(output_value, str) and len(output_value) > 200:
                    output_value = output_value[:200] + "..."
                lines.append(f"- **{output_key}**: {output_value}")

        return "\n".join(lines)

    async def _execute_condition_node(
        self,
        node: ConditionNode,
        context: WorkflowContext,
        start_time: float,
    ) -> NodeResult:
        """Execute a condition node.

        Args:
            node: Condition node
            context: Execution context
            start_time: Node start time

        Returns:
            NodeResult with selected branch
        """
        try:
            branch = node.condition(context.data)
            next_node = node.branches.get(branch)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output={"branch": branch, "next_node": next_node},
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=f"Condition evaluation failed: {e}",
                duration_seconds=time.time() - start_time,
            )

    async def _execute_parallel_node(
        self,
        node: ParallelNode,
        context: WorkflowContext,
        start_time: float,
    ) -> NodeResult:
        """Execute a parallel node.

        Args:
            node: Parallel node
            context: Execution context
            start_time: Node start time

        Returns:
            NodeResult with combined outputs
        """
        from victor.workflows.definition import WorkflowNode

        # Get the actual nodes to execute
        parallel_nodes: List[WorkflowNode] = []
        for node_id in node.parallel_nodes:
            # This requires access to the workflow definition
            # For now, store node references in metadata
            if "workflow" in context.metadata:
                workflow: WorkflowDefinition = context.metadata["workflow"]
                pnode = workflow.get_node(node_id)
                if pnode:
                    parallel_nodes.append(pnode)

        if not parallel_nodes:
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.SKIPPED,
                duration_seconds=time.time() - start_time,
            )

        # Execute in parallel with semaphore
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def execute_with_semaphore(pnode: WorkflowNode) -> NodeResult:
            async with semaphore:
                return await self._execute_node(pnode, context)

        tasks = [execute_with_semaphore(pnode) for pnode in parallel_nodes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results based on join strategy
        node_results = []
        for r in results:
            if isinstance(r, Exception):
                node_results.append(
                    NodeResult(
                        node_id="unknown",
                        status=NodeStatus.FAILED,
                        error=str(r),
                    )
                )
            else:
                node_results.append(r)
                context.add_result(r)

        # Determine overall status
        if node.join_strategy == "all":
            success = all(r.success for r in node_results)
        elif node.join_strategy == "any":
            success = any(r.success for r in node_results)
        else:  # merge
            success = True

        total_tools = sum(r.tool_calls_used for r in node_results)

        return NodeResult(
            node_id=node.id,
            status=NodeStatus.COMPLETED if success else NodeStatus.FAILED,
            output={"results": [r.output for r in node_results if r.output]},
            duration_seconds=time.time() - start_time,
            tool_calls_used=total_tools,
        )

    async def _execute_transform_node(
        self,
        node: TransformNode,
        context: WorkflowContext,
        start_time: float,
    ) -> NodeResult:
        """Execute a transform node.

        Args:
            node: Transform node
            context: Execution context
            start_time: Node start time

        Returns:
            NodeResult
        """
        try:
            new_data = node.transform(context.data)
            context.update(new_data)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=new_data,
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=f"Transform failed: {e}",
                duration_seconds=time.time() - start_time,
            )

    async def execute_by_name(
        self,
        workflow_name: str,
        initial_context: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> WorkflowResult:
        """Execute a workflow by name from the global registry.

        Args:
            workflow_name: Name of registered workflow
            initial_context: Initial context data
            timeout: Overall timeout

        Returns:
            WorkflowResult
        """
        from victor.workflows.registry import get_global_registry

        registry = get_global_registry()
        workflow = registry.get(workflow_name)

        if not workflow:
            return WorkflowResult(
                workflow_name=workflow_name,
                success=False,
                context=WorkflowContext(),
                error=f"Workflow '{workflow_name}' not found",
            )

        return await self.execute(workflow, initial_context, timeout=timeout)

    def _emit_workflow_step_event(
        self,
        workflow_name: str,
        node_id: str,
        node_type: str,
        success: bool,
        duration: float,
    ) -> None:
        """Emit RL event for workflow step completion.

        Args:
            workflow_name: Name of the workflow
            node_id: ID of the completed node
            node_type: Type of node (agent, condition, parallel, transform)
            success: Whether the step succeeded
            duration: Duration in seconds
        """
        try:
            from victor.agent.rl.hooks import get_rl_hooks, RLEvent, RLEventType

            hooks = get_rl_hooks()
            if hooks is None:
                return

            # Quality based on success and speed
            quality = 0.9 if success else 0.3
            if duration > 60:  # Penalize slow steps
                quality -= 0.1

            event = RLEvent(
                type=RLEventType.WORKFLOW_STEP,
                workflow_name=workflow_name,
                workflow_step=node_id,
                success=success,
                quality_score=quality,
                metadata={
                    "node_type": node_type,
                    "duration_seconds": duration,
                },
            )

            hooks.emit(event)

        except Exception as e:
            logger.debug(f"Workflow step event emission failed: {e}")

    def _emit_workflow_completed_event(
        self,
        workflow_name: str,
        success: bool,
        duration: float,
        tool_calls: int,
    ) -> None:
        """Emit RL event for workflow completion.

        Args:
            workflow_name: Name of the workflow
            success: Whether the workflow succeeded
            duration: Total duration in seconds
            tool_calls: Total tool calls used
        """
        try:
            from victor.agent.rl.hooks import get_rl_hooks, RLEvent, RLEventType

            hooks = get_rl_hooks()
            if hooks is None:
                return

            # Quality based on success and efficiency
            quality = 0.8 if success else 0.2
            if success and duration < 60:  # Fast completion bonus
                quality += 0.1
            if success and tool_calls < 20:  # Efficient bonus
                quality += 0.1

            event = RLEvent(
                type=RLEventType.WORKFLOW_COMPLETED,
                workflow_name=workflow_name,
                success=success,
                quality_score=min(1.0, quality),
                metadata={
                    "duration_seconds": duration,
                    "tool_calls": tool_calls,
                },
            )

            hooks.emit(event)

        except Exception as e:
            logger.debug(f"Workflow completed event emission failed: {e}")

    def _save_workflow_checkpoint(
        self,
        thread_id: str,
        workflow_name: str,
        last_node: str,
        next_node: Optional[str],
        context_data: Dict[str, Any],
    ) -> None:
        """Save workflow checkpoint using existing RL CheckpointStore.

        Args:
            thread_id: Thread identifier for this workflow execution
            workflow_name: Name of the workflow
            last_node: ID of the last completed node
            next_node: ID of the next node to execute (if any)
            context_data: Current context state
        """
        if not self._checkpointer:
            return

        try:
            import time

            version = f"{last_node}_{int(time.time())}"
            self._checkpointer.create_checkpoint(
                learner_name=f"workflow_{thread_id}",
                version=version,
                state={
                    "workflow_name": workflow_name,
                    "last_node": last_node,
                    "next_node": next_node,
                    "context": context_data,
                },
                metadata={
                    "checkpoint_type": "workflow",
                },
            )
            logger.debug(f"Saved workflow checkpoint: {thread_id}/{last_node}")

        except Exception as e:
            logger.warning(f"Failed to save workflow checkpoint: {e}")


__all__ = [
    "NodeStatus",
    "NodeResult",
    "WorkflowContext",
    "WorkflowResult",
    "WorkflowExecutor",
]
