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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, Set

from victor.framework.chain_registry import get_chain_registry
from victor.workflows.definition import (
    AgentNode,
    ComputeNode,
    ConditionNode,
    ParallelNode,
    TransformNode,
    WorkflowDefinition,
    WorkflowNode,
)
from victor.workflows.isolation import IsolationMapper
from victor.workflows.resilience import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
    RetryExecutor,
    retry_policy_to_strategy,
    get_node_circuit_breaker,
)

# Chain handler prefix for referencing registered chains
CHAIN_HANDLER_PREFIX = "chain:"

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.agent.subagents import SubAgentOrchestrator
    from victor.agent.rl.checkpoint_store import CheckpointStore
    from victor.tools.registry import ToolRegistry
    from victor.workflows.cache import WorkflowCache, WorkflowCacheConfig
    from victor.workflows.services import ServiceRegistry, ServiceConfig

logger = logging.getLogger(__name__)


class ComputeHandler(Protocol):
    """Protocol for custom compute node handlers.

    Handlers enable domain-specific execution logic for ComputeNodes.
    Register handlers with register_compute_handler() to extend
    workflow execution capabilities.

    Example:
        async def rl_decision_handler(
            node: ComputeNode,
            context: WorkflowContext,
            tool_registry: ToolRegistry,
        ) -> NodeResult:
            # Load RL policy and make decision
            policy = load_policy(context.get("policy_path"))
            features = {k: context.get(k) for k in node.input_mapping.values()}
            decision = policy.predict(features)
            context.set(node.output_key, decision)
            return NodeResult(node.id, ExecutorNodeStatus.COMPLETED, output=decision)

        register_compute_handler("rl_decision", rl_decision_handler)
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        """Execute custom handler logic.

        Args:
            node: The ComputeNode being executed
            context: Workflow execution context
            tool_registry: Tool registry for tool execution

        Returns:
            NodeResult with execution outcome
        """
        ...


# Global registry for compute handlers
_compute_handlers: Dict[str, ComputeHandler] = {}


def register_compute_handler(name: str, handler: ComputeHandler) -> None:
    """Register a custom compute handler.

    Handlers enable domain-specific execution logic for ComputeNodes.
    When a ComputeNode has a `handler` field matching the registered name,
    the handler will be invoked instead of the default tool execution.

    Args:
        name: Handler name (referenced in YAML as handler: name)
        handler: Async callable implementing ComputeHandler protocol

    Example:
        register_compute_handler("rl_decision", my_rl_handler)

        # In YAML:
        - id: weights
          type: compute
          handler: rl_decision
          inputs:
            features: $ctx.valuation_features
    """
    _compute_handlers[name] = handler
    logger.debug(f"Registered compute handler: {name}")


def get_compute_handler(name: str) -> Optional[ComputeHandler]:
    """Get a registered compute handler by name."""
    return _compute_handlers.get(name)


def list_compute_handlers() -> List[str]:
    """List all registered compute handler names."""
    return list(_compute_handlers.keys())


class ExecutorNodeStatus(Enum):
    """Execution status of a workflow executor node.

    Renamed from NodeStatus to be semantically distinct:
    - ExecutorNodeStatus (here): Executor node status
    - ProtocolNodeStatus (victor.workflows.protocols): Workflow protocol node status
    - FrameworkNodeStatus (victor.framework.graph): Framework graph node status
    """

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
    status: ExecutorNodeStatus
    output: Optional[Any] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    tool_calls_used: int = 0

    @property
    def success(self) -> bool:
        """Check if node completed successfully."""
        return self.status == ExecutorNodeStatus.COMPLETED

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
class TemporalContext:
    """Point-in-time context for backtesting and historical analysis.

    Provides temporal awareness to workflow execution, allowing tools
    to access data as of a specific date for backtesting scenarios.

    Attributes:
        as_of_date: The reference date for point-in-time analysis (YYYY-MM-DD)
        lookback_periods: Number of periods to look back from as_of_date
        period_type: Type of period (days, weeks, months, quarters, years)
        include_end_date: Whether to include the as_of_date in range

    Example YAML:
        - id: fetch_historical
          type: compute
          tools: [sec_filing]
          temporal_context:
            as_of_date: $ctx.analysis_date
            lookback_periods: 8
            period_type: quarters

    Example usage in tools:
        def my_tool(symbol: str, _exec_ctx: dict) -> ToolResult:
            temporal = _exec_ctx.get("temporal_context")
            if temporal:
                data = fetch_data(symbol, as_of=temporal.as_of_date)
            else:
                data = fetch_data(symbol)  # Current data
    """

    as_of_date: Optional[str] = None  # ISO format: YYYY-MM-DD
    lookback_periods: int = 0
    period_type: str = "quarters"  # days, weeks, months, quarters, years
    include_end_date: bool = True

    def get_date_range(self) -> tuple:
        """Calculate start and end dates based on lookback.

        Returns:
            Tuple of (start_date, end_date) as strings
        """
        from datetime import datetime, timedelta

        if not self.as_of_date:
            end_date = datetime.now()
        else:
            end_date = datetime.fromisoformat(self.as_of_date)

        # Calculate start date based on lookback
        if self.period_type == "days":
            delta = timedelta(days=self.lookback_periods)
        elif self.period_type == "weeks":
            delta = timedelta(weeks=self.lookback_periods)
        elif self.period_type == "months":
            delta = timedelta(days=self.lookback_periods * 30)
        elif self.period_type == "quarters":
            delta = timedelta(days=self.lookback_periods * 91)
        elif self.period_type == "years":
            delta = timedelta(days=self.lookback_periods * 365)
        else:
            delta = timedelta(days=0)

        start_date = end_date - delta

        return (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    def is_valid_for_date(self, data_date: str) -> bool:
        """Check if a data date is valid for this temporal context.

        Args:
            data_date: Date to check (YYYY-MM-DD)

        Returns:
            True if data_date is on or before as_of_date
        """
        if not self.as_of_date:
            return True  # No constraint

        from datetime import datetime

        data_dt = datetime.fromisoformat(data_date)
        as_of_dt = datetime.fromisoformat(self.as_of_date)

        if self.include_end_date:
            return data_dt <= as_of_dt
        return data_dt < as_of_dt

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "as_of_date": self.as_of_date,
            "lookback_periods": self.lookback_periods,
            "period_type": self.period_type,
            "include_end_date": self.include_end_date,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalContext":
        """Create from dictionary."""
        return cls(
            as_of_date=data.get("as_of_date"),
            lookback_periods=data.get("lookback_periods", 0),
            period_type=data.get("period_type", "quarters"),
            include_end_date=data.get("include_end_date", True),
        )


@dataclass
class WorkflowContext:
    """Execution context for a workflow.

    Maintains shared state across workflow nodes and provides
    utilities for accessing and updating context data.

    Attributes:
        data: Shared context data
        node_results: Results from executed nodes
        metadata: Execution metadata
        temporal: Optional temporal context for point-in-time analysis
    """

    data: Dict[str, Any] = field(default_factory=dict)
    node_results: Dict[str, NodeResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    temporal: Optional[TemporalContext] = None

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
        return any(r.status == ExecutorNodeStatus.FAILED for r in self.node_results.values())

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
            "node_results": {nid: r.to_dict() for nid, r in self.context.node_results.items()},
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
        tool_registry: Optional["ToolRegistry"] = None,
        service_registry: Optional["ServiceRegistry"] = None,
    ):
        """Initialize executor.

        Args:
            orchestrator: Agent orchestrator instance
            max_parallel: Maximum parallel executions
            default_timeout: Default timeout per node
            checkpointer: Optional CheckpointStore for persistence
            cache: Optional WorkflowCache for node result caching
            cache_config: Optional config to create a cache (alternative to cache param)
            tool_registry: Optional ToolRegistry for ComputeNode execution
            service_registry: Optional ServiceRegistry for infrastructure services
        """
        self.orchestrator = orchestrator
        self.max_parallel = max_parallel
        self.default_timeout = default_timeout
        self._checkpointer = checkpointer
        self._sub_agents: Optional["SubAgentOrchestrator"] = None
        self._active_executions: Dict[str, asyncio.Task] = {}
        self._tool_registry = tool_registry
        self._service_registry = service_registry
        self._services_started = False

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
    def tool_registry(self) -> "ToolRegistry":
        """Get or create ToolRegistry for ComputeNode execution."""
        if self._tool_registry is None:
            from victor.tools.registry import ToolRegistry

            self._tool_registry = ToolRegistry()
        return self._tool_registry

    @property
    def cache(self) -> Optional["WorkflowCache"]:
        """Get the workflow cache (if enabled)."""
        return self._cache

    @property
    def service_registry(self) -> "ServiceRegistry":
        """Get or create ServiceRegistry for infrastructure services."""
        if self._service_registry is None:
            from victor.workflows.services import create_default_registry

            self._service_registry = create_default_registry()
        return self._service_registry

    async def _start_services(
        self,
        workflow: WorkflowDefinition,
        context: WorkflowContext,
    ) -> None:
        """Start infrastructure services defined in workflow metadata.

        Services are started in dependency order before workflow execution.
        Service exports (like DATABASE_URL) are added to workflow context.

        Args:
            workflow: Workflow definition with services in metadata
            context: Execution context to populate with service exports
        """
        services_config = workflow.metadata.get("services", [])
        if not services_config:
            return

        from victor.workflows.services import ServiceConfig
        from victor.workflows.yaml_loader import ServiceConfigYAML

        logger.info(f"Starting {len(services_config)} infrastructure services...")

        # Convert YAML configs to ServiceConfig objects
        configs = []
        for svc_data in services_config:
            yaml_config = ServiceConfigYAML(
                name=svc_data["name"],
                provider=svc_data.get("provider", "docker"),
                preset=svc_data.get("preset"),
                image=svc_data.get("image"),
                command=svc_data.get("command"),
                ports=svc_data.get("ports", []),
                environment=svc_data.get("environment", {}),
                volumes=svc_data.get("volumes", []),
                health_check=svc_data.get("health_check"),
                depends_on=svc_data.get("depends_on", []),
                lifecycle=svc_data.get("lifecycle"),
                exports=svc_data.get("exports", {}),
            )
            configs.append(yaml_config.to_service_config())

        # Start all services (handles dependency ordering)
        try:
            await self.service_registry.start_all(configs, timeout=300.0)
            self._services_started = True

            # Export service connection info to context
            all_exports = self.service_registry.get_all_exports()
            for svc_name, exports in all_exports.items():
                for key, value in exports.items():
                    context.set(f"service_{svc_name}_{key}", value)
                    # Also set short form for common keys
                    if key in ("DATABASE_URL", "REDIS_URL", "KAFKA_URL"):
                        context.set(key, value)

            logger.info(f"All {len(configs)} services started successfully")

        except Exception as e:
            logger.error(f"Failed to start services: {e}")
            raise

    async def _stop_services(self) -> None:
        """Stop all running infrastructure services.

        Services are stopped in reverse dependency order.
        Called automatically after workflow execution.
        """
        if not self._services_started:
            return

        try:
            logger.info("Stopping infrastructure services...")
            await self.service_registry.stop_all(grace_period=30.0)
            self._services_started = False
            logger.info("All services stopped")
        except Exception as e:
            logger.error(f"Error stopping services: {e}")
            # Force cleanup even if graceful stop failed
            try:
                await self.service_registry.cleanup_all()
            except Exception:
                pass

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
        temporal_context: Optional[TemporalContext] = None,
    ) -> WorkflowResult:
        """Execute a workflow with optional checkpointing and temporal context.

        Args:
            workflow: Workflow definition to execute
            initial_context: Initial context data
            timeout: Overall timeout (None = no limit)
            thread_id: Thread ID for checkpointing (enables resume)
            temporal_context: Optional point-in-time context for backtesting

        Returns:
            WorkflowResult with execution outcome

        Example with temporal context:
            # Backtest as of a specific date
            temporal = TemporalContext(
                as_of_date="2023-06-30",
                lookback_periods=8,
                period_type="quarters",
            )
            result = await executor.execute(
                workflow,
                initial_context={"symbol": "AAPL"},
                temporal_context=temporal,
            )
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
                logger.info(
                    f"Resuming from checkpoint at node: {checkpoint.state.get('last_node')}"
                )
                initial_context = checkpoint.state.get("context", {})
                resume_from_node = checkpoint.state.get("next_node")

        # Build temporal context from workflow metadata if not provided
        if temporal_context is None and "temporal_context" in workflow.metadata:
            temporal_context = TemporalContext.from_dict(workflow.metadata["temporal_context"])

        context = WorkflowContext(
            data=initial_context.copy() if initial_context else {},
            metadata={
                "execution_id": execution_id,
                "workflow_name": workflow.name,
                "thread_id": thread_id,
                "workflow": workflow,  # Required for parallel node execution
            },
            temporal=temporal_context,
        )

        start_time = time.time()

        try:
            # Start infrastructure services before workflow execution
            await self._start_services(workflow, context)

            if timeout:
                await asyncio.wait_for(
                    self._execute_workflow(workflow, context, thread_id, resume_from_node),
                    timeout=timeout,
                )
            else:
                await self._execute_workflow(workflow, context, thread_id, resume_from_node)

            total_duration = time.time() - start_time
            total_tool_calls = sum(r.tool_calls_used for r in context.node_results.values())

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

        finally:
            # Always stop services after workflow execution
            await self._stop_services()

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
                node_type=node.node_type.value if hasattr(node, "node_type") else "unknown",
                success=result.status == ExecutorNodeStatus.COMPLETED,
                duration=result.duration_seconds,
            )

            # If failed, stop unless configured to continue
            if result.status == ExecutorNodeStatus.FAILED:
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
        Applies retry policy and circuit breaker when configured.

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

        # Check circuit breaker if enabled
        circuit_breaker = None
        if getattr(node, "circuit_breaker_enabled", False):
            circuit_breaker = get_node_circuit_breaker(node.id)
            if not circuit_breaker.can_execute():
                logger.warning(
                    f"Circuit breaker OPEN for node '{node.id}', "
                    f"state: {circuit_breaker.state.name}"
                )
                return NodeResult(
                    node_id=node.id,
                    status=ExecutorNodeStatus.FAILED,
                    error=f"Circuit breaker is {circuit_breaker.state.name}",
                    duration_seconds=time.time() - start_time,
                )

        # Execute with or without retry policy
        retry_policy = getattr(node, "retry_policy", None)
        if retry_policy:
            result = await self._execute_node_with_retry(
                node, context, start_time, retry_policy
            )
        else:
            result = await self._execute_node_inner(node, context, start_time)

        # Update circuit breaker state
        if circuit_breaker:
            if result.success:
                circuit_breaker.record_success()
            else:
                circuit_breaker.record_failure()

        # Cache successful results for cacheable nodes
        if self._cache is not None and result.success:
            self._cache.set(node, context.data, result)

        return result

    async def _execute_node_with_retry(
        self,
        node: WorkflowNode,
        context: WorkflowContext,
        start_time: float,
        retry_policy: "RetryPolicy",
    ) -> NodeResult:
        """Execute node with retry policy.

        Args:
            node: Node to execute
            context: Execution context
            start_time: Node start time
            retry_policy: Retry policy to apply

        Returns:
            NodeResult with execution outcome
        """
        from victor.workflows.protocols import RetryPolicy as WorkflowRetryPolicy

        strategy = retry_policy_to_strategy(retry_policy)
        executor = RetryExecutor(strategy)

        async def execute_func() -> NodeResult:
            return await self._execute_node_inner(node, context, start_time)

        retry_result = await executor.execute(execute_func)

        if retry_result.success:
            logger.debug(
                f"Node '{node.id}' succeeded after {retry_result.attempts} attempt(s)"
            )
            return retry_result.result
        else:
            logger.warning(
                f"Node '{node.id}' failed after {retry_result.attempts} attempt(s): "
                f"{retry_result.last_exception}"
            )
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=str(retry_result.last_exception) if retry_result.last_exception else "Retry exhausted",
                duration_seconds=time.time() - start_time,
            )

    async def _execute_node_inner(
        self,
        node: WorkflowNode,
        context: WorkflowContext,
        start_time: float,
    ) -> NodeResult:
        """Execute a single workflow node (inner logic).

        Args:
            node: Node to execute
            context: Execution context
            start_time: Node start time

        Returns:
            NodeResult with execution outcome
        """
        try:
            if isinstance(node, AgentNode):
                result = await self._execute_agent_node(node, context, start_time)

            elif isinstance(node, ComputeNode):
                result = await self._execute_compute_node(node, context, start_time)

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
                    status=ExecutorNodeStatus.SKIPPED,
                    duration_seconds=time.time() - start_time,
                )

            return result

        except Exception as e:
            logger.error(f"Node '{node.id}' failed: {e}", exc_info=True)
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
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
            status=ExecutorNodeStatus.COMPLETED if result.success else ExecutorNodeStatus.FAILED,
            output=result.summary,
            error=result.error,
            duration_seconds=time.time() - start_time,
            tool_calls_used=result.tool_calls_used,
        )

    def _build_agent_task(self, node: AgentNode, context: WorkflowContext) -> str:
        """Build task string for an agent node.

        Performs template substitution on node.goal using input_mapping values,
        allowing YAML workflows to use {placeholder} syntax in goal templates.

        Args:
            node: Agent node
            context: Execution context

        Returns:
            Task description for agent with placeholders substituted
        """
        import json

        # Build substitution dict from input_mapping
        substitutions = {}
        if node.input_mapping:
            for param, key in node.input_mapping.items():
                value = context.get(key)
                if value is not None:
                    # Convert complex objects to string for template substitution
                    if not isinstance(value, str):
                        try:
                            value = json.dumps(value, indent=2, default=str)
                        except (TypeError, ValueError):
                            value = str(value)
                    substitutions[param] = value

        # Substitute placeholders in goal template (e.g., {symbol} -> "AAPL")
        goal = node.goal
        for key, value in substitutions.items():
            goal = goal.replace(f"{{{key}}}", value)

        return goal

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
                status=ExecutorNodeStatus.COMPLETED,
                output={"branch": branch, "next_node": next_node},
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
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
                status=ExecutorNodeStatus.SKIPPED,
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
                        status=ExecutorNodeStatus.FAILED,
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
            status=ExecutorNodeStatus.COMPLETED if success else ExecutorNodeStatus.FAILED,
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
                status=ExecutorNodeStatus.COMPLETED,
                output=new_data,
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=f"Transform failed: {e}",
                duration_seconds=time.time() - start_time,
            )

    async def _execute_compute_node(
        self,
        node: ComputeNode,
        context: WorkflowContext,
        start_time: float,
    ) -> NodeResult:
        """Execute a compute node with constraints enforcement and isolation.

        Supports:
        - Custom handlers for domain-specific logic
        - LLM-free tool execution
        - Constraint enforcement (cost tier, tool limits, etc.)
        - Isolation mapping (none, process, docker)

        Args:
            node: Compute node with tools, handler, and constraints
            context: Execution context
            start_time: Node start time

        Returns:
            NodeResult with execution outputs
        """
        try:
            # Determine isolation configuration from constraints
            vertical = context.metadata.get("vertical")
            isolation = IsolationMapper.from_constraints(
                constraints=node.constraints,
                vertical=vertical,
            )
            logger.debug(
                f"Node {node.id}: isolation={isolation.sandbox_type}, "
                f"network={isolation.network_allowed}, vertical={vertical}"
            )

            # Check for custom handler first
            if node.handler:
                # Check if this is a chain reference (e.g., "chain:coding:analyze")
                if node.handler.startswith(CHAIN_HANDLER_PREFIX):
                    chain_name = node.handler[len(CHAIN_HANDLER_PREFIX) :]
                    logger.debug(f"Executing chain '{chain_name}' for node {node.id}")
                    return await self._execute_chain_handler(node, context, chain_name, start_time)

                handler = get_compute_handler(node.handler)
                if handler:
                    logger.debug(f"Using custom handler '{node.handler}' for node {node.id}")
                    return await handler(node, context, self.tool_registry)
                else:
                    logger.warning(
                        f"Handler '{node.handler}' not found for node {node.id}, "
                        f"falling back to default execution"
                    )

            # Default tool execution with constraint enforcement
            tool_params = self._build_compute_params(node, context)
            outputs = {}
            tool_calls_used = 0
            constraints = node.constraints

            # Filter tools based on constraints
            allowed_tools = []
            for tool_name in node.tools:
                # Check against constraint allowlists/blocklists
                if constraints.allows_tool(tool_name):
                    allowed_tools.append(tool_name)
                else:
                    logger.warning(f"Tool '{tool_name}' blocked by constraints for node {node.id}")

            if not allowed_tools and node.tools:
                return NodeResult(
                    node_id=node.id,
                    status=ExecutorNodeStatus.FAILED,
                    error="All tools blocked by constraints",
                    duration_seconds=time.time() - start_time,
                )

            # Build execution context with isolation info
            exec_ctx = {
                "workflow_context": context.data,
                "constraints": constraints.to_dict(),
                "isolation": isolation.to_dict(),
                "temporal_context": context.metadata.get("temporal_context"),
            }

            if node.parallel and len(allowed_tools) > 1:
                # Execute tools in parallel
                async def execute_tool(tool_name: str, exec_context: dict) -> tuple:
                    try:
                        result = await asyncio.wait_for(
                            self.tool_registry.execute(
                                tool_name,
                                _exec_ctx=exec_context,
                                **tool_params,
                            ),
                            timeout=constraints.timeout,
                        )
                        return tool_name, result
                    except asyncio.TimeoutError:
                        from victor.tools.base import ToolResult

                        return tool_name, ToolResult(
                            success=False,
                            output=None,
                            error=f"Tool '{tool_name}' timed out after {constraints.timeout}s",
                        )
                    except Exception as e:
                        from victor.tools.base import ToolResult

                        return tool_name, ToolResult(
                            success=False,
                            output=None,
                            error=str(e),
                        )

                tasks = [execute_tool(tool_name, exec_ctx) for tool_name in allowed_tools]
                results = await asyncio.gather(*tasks)

                for tool_name, result in results:
                    tool_calls_used += 1
                    if tool_calls_used > constraints.max_tool_calls:
                        return NodeResult(
                            node_id=node.id,
                            status=ExecutorNodeStatus.FAILED,
                            error=f"Exceeded max tool calls ({constraints.max_tool_calls})",
                            output=outputs,
                            duration_seconds=time.time() - start_time,
                            tool_calls_used=tool_calls_used,
                        )

                    if result.success:
                        outputs[tool_name] = result.output
                    elif node.fail_fast:
                        return NodeResult(
                            node_id=node.id,
                            status=ExecutorNodeStatus.FAILED,
                            error=f"Tool '{tool_name}' failed: {result.error}",
                            output=outputs,
                            duration_seconds=time.time() - start_time,
                            tool_calls_used=tool_calls_used,
                        )
            else:
                # Execute tools sequentially
                for tool_name in allowed_tools:
                    if tool_calls_used >= constraints.max_tool_calls:
                        return NodeResult(
                            node_id=node.id,
                            status=ExecutorNodeStatus.FAILED,
                            error=f"Exceeded max tool calls ({constraints.max_tool_calls})",
                            output=outputs,
                            duration_seconds=time.time() - start_time,
                            tool_calls_used=tool_calls_used,
                        )

                    try:
                        result = await asyncio.wait_for(
                            self.tool_registry.execute(
                                tool_name,
                                _exec_ctx=exec_ctx,
                                **tool_params,
                            ),
                            timeout=constraints.timeout,
                        )
                        tool_calls_used += 1

                        if result.success:
                            outputs[tool_name] = result.output
                            # Update params with output for chaining
                            tool_params.update({tool_name: result.output})
                        elif node.fail_fast:
                            return NodeResult(
                                node_id=node.id,
                                status=ExecutorNodeStatus.FAILED,
                                error=f"Tool '{tool_name}' failed: {result.error}",
                                output=outputs,
                                duration_seconds=time.time() - start_time,
                                tool_calls_used=tool_calls_used,
                            )

                    except asyncio.TimeoutError:
                        if node.fail_fast:
                            return NodeResult(
                                node_id=node.id,
                                status=ExecutorNodeStatus.FAILED,
                                error=f"Tool '{tool_name}' timed out after {constraints.timeout}s",
                                output=outputs,
                                duration_seconds=time.time() - start_time,
                                tool_calls_used=tool_calls_used,
                            )

            # Store outputs in context
            output_key = node.output_key or node.id
            context.set(output_key, outputs)

            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.COMPLETED,
                output=outputs,
                duration_seconds=time.time() - start_time,
                tool_calls_used=tool_calls_used,
            )

        except Exception as e:
            logger.error(f"ComputeNode '{node.id}' failed: {e}", exc_info=True)
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=f"Compute failed: {e}",
                duration_seconds=time.time() - start_time,
            )

    def _build_compute_params(
        self,
        node: ComputeNode,
        context: WorkflowContext,
    ) -> Dict[str, Any]:
        """Build tool parameters from context using input mapping.

        Args:
            node: Compute node with input mapping
            context: Execution context

        Returns:
            Dictionary of parameters for tool execution
        """
        params = {}
        for param_name, context_key in node.input_mapping.items():
            if isinstance(context_key, str):
                # Could be a context reference or literal value
                value = context.get(context_key)
                if value is not None:
                    params[param_name] = value
                else:
                    # Treat as literal value if not found in context
                    params[param_name] = context_key
            else:
                params[param_name] = context_key
        return params

    async def _execute_chain_handler(
        self,
        node: ComputeNode,
        context: WorkflowContext,
        chain_name: str,
        start_time: float,
    ) -> NodeResult:
        """Execute a registered chain from the ChainRegistry.

        Chains can be referenced in YAML workflows using the format:
            handler: chain:<vertical>:<name>
        or:
            handler: chain:<name>

        The chain is created from the registry and invoked with the
        node's input parameters from context.

        Args:
            node: The ComputeNode referencing the chain
            context: Execution context
            chain_name: Full chain name (e.g., "coding:analyze" or just "analyze")
            start_time: Node start time

        Returns:
            NodeResult with chain execution output

        Example YAML:
            - id: analyze_code
              type: compute
              handler: chain:coding:analyze
              inputs:
                code: $ctx.source_code
              output: analysis_result
        """
        try:
            registry = get_chain_registry()

            # Parse vertical from chain_name if present (e.g., "coding:analyze")
            vertical = None
            name = chain_name
            if ":" in chain_name:
                vertical, name = chain_name.split(":", 1)

            # Try to get as factory first (most common for chains)
            chain_obj = registry.create(name, vertical=vertical)

            if chain_obj is None:
                # Fall back to direct chain lookup
                chain_obj = registry.get(name, vertical=vertical)

            if chain_obj is None:
                logger.error(f"Chain '{chain_name}' not found in registry")
                return NodeResult(
                    node_id=node.id,
                    status=ExecutorNodeStatus.FAILED,
                    error=f"Chain '{chain_name}' not found in ChainRegistry",
                    duration_seconds=time.time() - start_time,
                )

            # Build input parameters from context
            input_params = self._build_compute_params(node, context)

            # Execute the chain
            # Chains can be callables, Runnables (LCEL), or have invoke() method
            if hasattr(chain_obj, "invoke"):
                # LCEL Runnable-style chain
                if asyncio.iscoroutinefunction(chain_obj.invoke):
                    result = await chain_obj.invoke(input_params)
                else:
                    # Run sync invoke in executor
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, lambda: chain_obj.invoke(input_params)
                    )
            elif callable(chain_obj):
                # Simple callable chain
                if asyncio.iscoroutinefunction(chain_obj):
                    result = await chain_obj(**input_params)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: chain_obj(**input_params))
            else:
                # Chain is not callable - treat as static data
                result = chain_obj

            # Store result in context
            output_key = node.output_key or node.id
            context.set(output_key, result)

            logger.debug(f"Chain '{chain_name}' executed successfully for node {node.id}")

            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.COMPLETED,
                output=result,
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Chain execution failed for node {node.id}: {e}", exc_info=True)
            return NodeResult(
                node_id=node.id,
                status=ExecutorNodeStatus.FAILED,
                error=f"Chain execution failed: {e}",
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
    # Core types
    "ExecutorNodeStatus",
    "NodeResult",
    "WorkflowContext",
    "WorkflowResult",
    "WorkflowExecutor",
    # Temporal context for backtesting
    "TemporalContext",
    # Handler extensibility
    "ComputeHandler",
    "register_compute_handler",
    "get_compute_handler",
    "list_compute_handlers",
]
