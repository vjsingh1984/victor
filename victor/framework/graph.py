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

"""StateGraph - LangGraph-compatible graph workflow engine.

This module provides a LangGraph-inspired StateGraph implementation
for building cyclic, stateful agent workflows with typed state management.

Design Principles (SOLID):
    - Single Responsibility: Each class handles one aspect (state, nodes, edges, execution)
    - Open/Closed: Extensible via protocols without modifying core classes
    - Liskov Substitution: All node types implement NodeProtocol
    - Interface Segregation: Small, focused protocols (StateProtocol, EdgeProtocol)
    - Dependency Inversion: Depend on abstractions (protocols) not concretions

Key Differences from victor.workflows:
    - StateGraph: Typed state management (vs. dict-based context)
    - Cyclic Support: Allows cycles with configurable max iterations
    - Edge Types: Explicit conditional/normal edges (vs. next_nodes list)
    - Checkpointing: Full state persistence for recovery
    - Compile Step: Validates and optimizes graph before execution

Example:
    from victor.framework.graph import StateGraph, Node, Edge, END

    # Define typed state
    class AgentState(TypedDict):
        messages: list[str]
        task: str
        result: Optional[str]

    # Create graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("analyze", analyze_task)
    graph.add_node("execute", execute_task)
    graph.add_node("review", review_result)

    # Add edges (including cycles)
    graph.add_edge("analyze", "execute")
    graph.add_conditional_edge(
        "execute",
        should_retry,
        {"retry": "analyze", "done": "review"}
    )
    graph.add_edge("review", END)

    # Set entry point
    graph.set_entry_point("analyze")

    # Compile and run
    app = graph.compile()
    result = await app.invoke({"messages": [], "task": "Fix bug"})
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

# Type variables for generic state
StateType = TypeVar("StateType", bound=Dict[str, Any])
T = TypeVar("T")

# Sentinel for end of graph
END = "__end__"
START = "__start__"


class CopyOnWriteState(Generic[StateType]):
    """Copy-on-write wrapper for workflow state.

    Delays deep copy of state until the first mutation, reducing overhead
    for read-heavy workflows where nodes often only read state.

    This optimization is particularly effective for:
    - Workflows with many conditional branches that only read state
    - Nodes that check conditions without modifying state
    - Large state objects where deep copy is expensive

    Example:
        # Wrap original state
        cow_state = CopyOnWriteState(original_state)

        # Reading doesn't copy
        value = cow_state["key"]  # No copy

        # Writing triggers copy
        cow_state["key"] = "new_value"  # Deep copy happens here

        # Get the final state
        final_state = cow_state.get_state()

    Performance characteristics:
        - Read operations: O(1), no copy overhead
        - First write: O(n) deep copy where n is state size
        - Subsequent writes: O(1), no additional copy

    Thread safety:
        This class is NOT thread-safe. Each thread should have its own
        CopyOnWriteState wrapper if concurrent access is needed.
    """

    __slots__ = ("_source", "_copy", "_modified")

    def __init__(self, source: StateType):
        """Initialize with source state.

        Args:
            source: Original state dictionary (not copied until mutation)
        """
        self._source: StateType = source
        self._copy: Optional[StateType] = None
        self._modified: bool = False

    def _ensure_copy(self) -> StateType:
        """Ensure we have a mutable copy of the state.

        Returns:
            The mutable copy of the state
        """
        if not self._modified:
            self._copy = copy.deepcopy(self._source)
            self._modified = True
        return self._copy  # type: ignore

    def __getitem__(self, key: str) -> Any:
        """Get item without copying.

        Args:
            key: Key to look up

        Returns:
            Value associated with key

        Raises:
            KeyError: If key is not found
        """
        if self._modified:
            return self._copy[key]  # type: ignore
        return self._source[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item, triggering copy on first mutation.

        Args:
            key: Key to set
            value: Value to associate with key
        """
        self._ensure_copy()[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete item, triggering copy on first mutation.

        Args:
            key: Key to delete

        Raises:
            KeyError: If key is not found
        """
        del self._ensure_copy()[key]

    def __contains__(self, key: object) -> bool:
        """Check if key exists without copying.

        Args:
            key: Key to check

        Returns:
            True if key exists, False otherwise
        """
        if self._modified:
            return key in self._copy  # type: ignore
        return key in self._source

    def __len__(self) -> int:
        """Get length without copying.

        Returns:
            Number of items in state
        """
        if self._modified:
            return len(self._copy)  # type: ignore
        return len(self._source)

    def __iter__(self):
        """Iterate over keys without copying.

        Yields:
            Keys from the state
        """
        if self._modified:
            return iter(self._copy)  # type: ignore
        return iter(self._source)

    def get(self, key: str, default: Any = None) -> Any:
        """Get with default without copying.

        Args:
            key: Key to look up
            default: Value to return if key not found

        Returns:
            Value associated with key, or default
        """
        if self._modified:
            return self._copy.get(key, default)  # type: ignore
        return self._source.get(key, default)

    def keys(self):
        """Get keys without copying.

        Returns:
            View of keys in the state
        """
        if self._modified:
            return self._copy.keys()  # type: ignore
        return self._source.keys()

    def values(self):
        """Get values without copying.

        Returns:
            View of values in the state
        """
        if self._modified:
            return self._copy.values()  # type: ignore
        return self._source.values()

    def items(self):
        """Get items without copying.

        Returns:
            View of (key, value) pairs in the state
        """
        if self._modified:
            return self._copy.items()  # type: ignore
        return self._source.items()

    def update(self, other: Dict[str, Any]) -> None:
        """Update state, triggering copy.

        Args:
            other: Dictionary of items to update
        """
        self._ensure_copy().update(other)

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Set default value, may trigger copy.

        If key is not present, sets it to default (triggering copy).
        If key is present, returns its value without copying.

        Args:
            key: Key to look up or set
            default: Value to set if key not found

        Returns:
            Value associated with key (existing or default)
        """
        if key not in self:
            self._ensure_copy()[key] = default
            return default
        return self[key]

    def pop(self, key: str, *args: Any) -> Any:
        """Pop item, triggering copy.

        Args:
            key: Key to pop
            *args: Optional default value

        Returns:
            Value associated with key

        Raises:
            KeyError: If key is not found and no default provided
        """
        return self._ensure_copy().pop(key, *args)

    def copy(self) -> Dict[str, Any]:
        """Create a shallow copy of the current state.

        Returns:
            Shallow copy of the state dictionary
        """
        if self._modified:
            return self._copy.copy()  # type: ignore
        return self._source.copy()  # type: ignore

    def get_state(self) -> StateType:
        """Get the final state.

        Returns the modified copy if mutations occurred,
        otherwise returns the original source.

        Returns:
            The current state (modified copy or original source)
        """
        if self._modified:
            return self._copy  # type: ignore
        return self._source

    @property
    def was_modified(self) -> bool:
        """Check if state was modified (copy was made).

        Returns:
            True if any mutation occurred, False otherwise
        """
        return self._modified

    def to_dict(self) -> Dict[str, Any]:
        """Convert to regular dictionary.

        Returns:
            Dictionary copy of the current state
        """
        return dict(self.get_state())

    def __repr__(self) -> str:
        """String representation for debugging.

        Returns:
            Debug string showing modification status
        """
        status = "modified" if self._modified else "unmodified"
        return f"CopyOnWriteState({status}, keys={list(self.keys())})"


class EdgeType(Enum):
    """Types of edges in the graph."""

    NORMAL = "normal"
    CONDITIONAL = "conditional"


class FrameworkNodeStatus(Enum):
    """Execution status of a framework graph node.

    Renamed from NodeStatus to be semantically distinct:
    - FrameworkNodeStatus (here): Framework graph node status
    - ProtocolNodeStatus (victor.workflows.protocols): Workflow protocol node status
    - ExecutorNodeStatus (victor.workflows.executor): Executor node status
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"




@runtime_checkable
class StateProtocol(Protocol):
    """Protocol for state objects.

    States must be dict-like with copy support.
    """

    def __getitem__(self, key: str) -> Any: ...

    def __setitem__(self, key: str, value: Any) -> None: ...

    def get(self, key: str, default: Any = None) -> Any: ...

    def copy(self) -> "StateProtocol": ...


@runtime_checkable
class NodeFunctionProtocol(Protocol[StateType]):
    """Protocol for node functions.

    Node functions receive state and return updated state.
    Can be sync or async.
    """

    def __call__(self, state: StateType) -> Union[StateType, Awaitable[StateType]]: ...


@runtime_checkable
class ConditionFunctionProtocol(Protocol[StateType]):
    """Protocol for condition functions.

    Condition functions receive state and return a branch name.
    """

    def __call__(self, state: StateType) -> str: ...


@dataclass
class Edge:
    """Represents an edge between nodes.

    Attributes:
        source: Source node ID
        target: Target node ID (or dict for conditional)
        edge_type: Normal or conditional
        condition: Condition function for conditional edges
    """

    source: str
    target: Union[str, Dict[str, str]]
    edge_type: EdgeType = EdgeType.NORMAL
    condition: Optional[Callable[[Any], str]] = None

    def get_target(self, state: Any) -> Optional[str]:
        """Get target node based on state.

        Args:
            state: Current state

        Returns:
            Target node ID or None
        """
        if self.edge_type == EdgeType.NORMAL:
            return self.target if isinstance(self.target, str) else None

        if self.condition is None:
            return None

        branch = self.condition(state)
        if isinstance(self.target, dict):
            return self.target.get(branch)
        return None


@dataclass
class Node:
    """Represents a node in the graph.

    Attributes:
        id: Unique node identifier
        func: Node execution function
        metadata: Additional node metadata
    """

    id: str
    func: Callable[[Any], Union[Any, Awaitable[Any]]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    async def execute(self, state: Any) -> Any:
        """Execute node function.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        result = self.func(state)
        if asyncio.iscoroutine(result):
            return await result
        return result


@dataclass
class WorkflowCheckpoint:
    """WorkflowCheckpoint for workflow state persistence (StateGraph DSL).

    Renamed from Checkpoint to be semantically distinct:
    - GitCheckpoint (victor.agent.checkpoints): Git stash-based
    - ExecutionCheckpoint (victor.agent.time_aware_executor): Time/progress tracking
    - WorkflowCheckpoint (here): Workflow state with thread_id/node_id
    - HITLCheckpoint (victor.framework.hitl): Human-in-the-loop pause/resume

    Attributes:
        checkpoint_id: Unique checkpoint identifier
        thread_id: Thread/execution identifier
        node_id: Current node being executed
        state: State at checkpoint
        timestamp: When checkpoint was created
        metadata: Additional metadata
    """

    checkpoint_id: str
    thread_id: str
    node_id: str
    state: Dict[str, Any]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "thread_id": self.thread_id,
            "node_id": self.node_id,
            "state": self.state,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowCheckpoint":
        """Deserialize checkpoint from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            thread_id=data["thread_id"],
            node_id=data["node_id"],
            state=data["state"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
        )




class CheckpointerProtocol(Protocol):
    """Protocol for checkpoint persistence."""

    async def save(self, checkpoint: WorkflowCheckpoint) -> None:
        """Save a checkpoint."""
        ...

    async def load(self, thread_id: str) -> Optional[WorkflowCheckpoint]:
        """Load latest checkpoint for thread."""
        ...

    async def list(self, thread_id: str) -> List[WorkflowCheckpoint]:
        """List all checkpoints for thread."""
        ...


class MemoryCheckpointer:
    """In-memory checkpoint storage.

    Suitable for development and testing.
    """

    def __init__(self):
        self._checkpoints: Dict[str, List[WorkflowCheckpoint]] = {}

    async def save(self, checkpoint: WorkflowCheckpoint) -> None:
        """Save checkpoint to memory."""
        if checkpoint.thread_id not in self._checkpoints:
            self._checkpoints[checkpoint.thread_id] = []
        self._checkpoints[checkpoint.thread_id].append(checkpoint)

    async def load(self, thread_id: str) -> Optional[WorkflowCheckpoint]:
        """Load latest checkpoint."""
        checkpoints = self._checkpoints.get(thread_id, [])
        return checkpoints[-1] if checkpoints else None

    async def list(self, thread_id: str) -> List[WorkflowCheckpoint]:
        """List all checkpoints."""
        return self._checkpoints.get(thread_id, [])


class RLCheckpointerAdapter:
    """Adapter to use existing RL CheckpointStore for graph checkpointing.

    Bridges the graph's CheckpointerProtocol with the existing
    victor.agent.rl.checkpoint_store.CheckpointStore infrastructure.
    """

    def __init__(self, learner_name: str = "state_graph"):
        """Initialize adapter.

        Args:
            learner_name: Name to use in checkpoint store (default: "state_graph")
        """
        self.learner_name = learner_name
        self._store = None

    def _get_store(self):
        """Lazy load checkpoint store."""
        if self._store is None:
            from victor.agent.rl.checkpoint_store import get_checkpoint_store

            self._store = get_checkpoint_store()
        return self._store

    async def save(self, checkpoint: WorkflowCheckpoint) -> None:
        """Save checkpoint using RL checkpoint store."""
        store = self._get_store()
        # Convert to PolicyCheckpoint format
        store.create_checkpoint(
            learner_name=f"{self.learner_name}_{checkpoint.thread_id}",
            version=checkpoint.checkpoint_id,
            state={
                "node_id": checkpoint.node_id,
                "state": checkpoint.state,
                "timestamp": checkpoint.timestamp,
            },
            metadata=checkpoint.metadata,
        )

    async def load(self, thread_id: str) -> Optional[WorkflowCheckpoint]:
        """Load latest checkpoint from RL checkpoint store."""
        store = self._get_store()
        policy_cp = store.get_latest_checkpoint(f"{self.learner_name}_{thread_id}")
        if not policy_cp:
            return None

        return WorkflowCheckpoint(
            checkpoint_id=policy_cp.version,
            thread_id=thread_id,
            node_id=policy_cp.state.get("node_id", ""),
            state=policy_cp.state.get("state", {}),
            timestamp=policy_cp.state.get("timestamp", 0.0),
            metadata=policy_cp.metadata,
        )

    async def list(self, thread_id: str) -> List[WorkflowCheckpoint]:
        """List all checkpoints for thread."""
        store = self._get_store()
        policy_cps = store.list_checkpoints(f"{self.learner_name}_{thread_id}")

        return [
            WorkflowCheckpoint(
                checkpoint_id=cp.version,
                thread_id=thread_id,
                node_id=cp.state.get("node_id", ""),
                state=cp.state.get("state", {}),
                timestamp=cp.state.get("timestamp", 0.0),
                metadata=cp.metadata,
            )
            for cp in policy_cps
        ]


@dataclass
class GraphConfig:
    """Configuration for graph execution.

    Attributes:
        max_iterations: Maximum cycles allowed (default: 25)
        timeout: Overall execution timeout in seconds
        checkpointer: Optional checkpointer for persistence
        recursion_limit: Maximum recursion depth
        interrupt_before: Nodes to interrupt before execution
        interrupt_after: Nodes to interrupt after execution
        use_copy_on_write: Enable copy-on-write state optimization (default: None uses settings)
        emit_events: Enable EventBus integration for observability (default: True)
        graph_id: Optional identifier for this graph execution
    """

    max_iterations: int = 25
    timeout: Optional[float] = None
    checkpointer: Optional[CheckpointerProtocol] = None
    recursion_limit: int = 100
    interrupt_before: List[str] = field(default_factory=list)
    interrupt_after: List[str] = field(default_factory=list)
    use_copy_on_write: Optional[bool] = None  # None = use settings default
    emit_events: bool = True  # Enable EventBus observability integration
    graph_id: Optional[str] = None  # Optional identifier for event correlation


@dataclass
class ExecutionResult(Generic[StateType]):
    """Result from graph execution.

    Attributes:
        state: Final state
        success: Whether execution succeeded
        error: Error message if failed
        iterations: Number of iterations executed
        duration: Total execution time
        node_history: Sequence of executed nodes
    """

    state: StateType
    success: bool
    error: Optional[str] = None
    iterations: int = 0
    duration: float = 0.0
    node_history: List[str] = field(default_factory=list)


class CompiledGraph(Generic[StateType]):
    """Compiled graph ready for execution.

    The compilation step validates the graph structure and
    creates an optimized execution plan.
    """

    def __init__(
        self,
        nodes: Dict[str, Node],
        edges: Dict[str, List[Edge]],
        entry_point: str,
        state_schema: Optional[Type[StateType]] = None,
        config: Optional[GraphConfig] = None,
    ):
        """Initialize compiled graph.

        Args:
            nodes: Node registry
            edges: Edge registry (source -> list of edges)
            entry_point: Starting node ID
            state_schema: Optional type for state validation
            config: Execution configuration
        """
        self._nodes = nodes
        self._edges = edges
        self._entry_point = entry_point
        self._state_schema = state_schema
        self._config = config or GraphConfig()

    def _should_use_cow(self, exec_config: GraphConfig) -> bool:
        """Determine if copy-on-write should be used.

        Args:
            exec_config: Execution configuration

        Returns:
            True if COW should be enabled
        """
        # Explicit config takes precedence
        if exec_config.use_copy_on_write is not None:
            return exec_config.use_copy_on_write

        # Fall back to settings
        try:
            from victor.config.settings import load_settings

            settings = load_settings()
            return settings.stategraph_copy_on_write_enabled
        except Exception:
            # Default to True if settings can't be loaded
            return True

    def _emit_event(
        self,
        event_type: str,
        graph_id: str,
        data: Dict[str, Any],
        emit_events: bool,
    ) -> None:
        """Emit an event to the EventBus for observability.

        Args:
            event_type: Type of event (graph_started, node_start, etc.)
            graph_id: Graph execution identifier
            data: Event payload data
            emit_events: Whether to emit events (from config)
        """
        if not emit_events:
            return

        try:
            from victor.observability.event_bus import get_event_bus

            bus = get_event_bus()
            bus.emit_lifecycle_event(
                event_type,
                {
                    "graph_id": graph_id,
                    "source": "StateGraph",
                    **data,
                },
            )
        except Exception as e:
            # Don't let event emission failures break graph execution
            logger.debug(f"Failed to emit {event_type} event: {e}")

    async def invoke(
        self,
        input_state: StateType,
        *,
        config: Optional[GraphConfig] = None,
        thread_id: Optional[str] = None,
    ) -> ExecutionResult[StateType]:
        """Execute the graph.

        Args:
            input_state: Initial state
            config: Override execution config
            thread_id: Thread ID for checkpointing

        Returns:
            ExecutionResult with final state
        """
        exec_config = config or self._config
        thread_id = thread_id or uuid.uuid4().hex
        use_cow = self._should_use_cow(exec_config)

        # Check for checkpoint to resume from
        if exec_config.checkpointer:
            checkpoint = await exec_config.checkpointer.load(thread_id)
            if checkpoint:
                logger.info(f"Resuming from checkpoint at node: {checkpoint.node_id}")
                state = checkpoint.state.copy()
                current_node = checkpoint.node_id
            else:
                state = copy.deepcopy(input_state)
                current_node = self._entry_point
        else:
            state = copy.deepcopy(input_state)
            current_node = self._entry_point

        start_time = time.time()
        iterations = 0
        node_history: List[str] = []
        visited_count: Dict[str, int] = {}
        graph_id = exec_config.graph_id or thread_id

        # Emit graph started event for observability
        self._emit_event(
            "graph_started",
            graph_id,
            {
                "entry_point": self._entry_point,
                "node_count": len(self._nodes),
                "thread_id": thread_id,
            },
            exec_config.emit_events,
        )

        try:
            while current_node != END:
                # Check iteration limit
                iterations += 1
                if iterations > exec_config.max_iterations:
                    logger.warning(f"Max iterations ({exec_config.max_iterations}) reached")
                    return ExecutionResult(
                        state=state,
                        success=False,
                        error=f"Max iterations ({exec_config.max_iterations}) exceeded",
                        iterations=iterations,
                        duration=time.time() - start_time,
                        node_history=node_history,
                    )

                # Track cycles
                visited_count[current_node] = visited_count.get(current_node, 0) + 1
                if visited_count[current_node] > exec_config.recursion_limit:
                    return ExecutionResult(
                        state=state,
                        success=False,
                        error=f"Recursion limit exceeded at node: {current_node}",
                        iterations=iterations,
                        duration=time.time() - start_time,
                        node_history=node_history,
                    )

                # Check for interrupt before
                if current_node in exec_config.interrupt_before:
                    logger.info(f"Interrupt before node: {current_node}")
                    # Save checkpoint and return for human intervention
                    if exec_config.checkpointer:
                        await self._save_checkpoint(
                            exec_config.checkpointer, thread_id, current_node, state
                        )
                    return ExecutionResult(
                        state=state,
                        success=True,
                        iterations=iterations,
                        duration=time.time() - start_time,
                        node_history=node_history,
                    )

                # Execute node
                node = self._nodes.get(current_node)
                if not node:
                    return ExecutionResult(
                        state=state,
                        success=False,
                        error=f"Node not found: {current_node}",
                        iterations=iterations,
                        duration=time.time() - start_time,
                        node_history=node_history,
                    )

                logger.debug(f"Executing node: {current_node}")
                node_history.append(current_node)

                # Emit node_start event for observability
                node_start_time = time.time()
                self._emit_event(
                    "node_start",
                    graph_id,
                    {
                        "node_id": current_node,
                        "iteration": iterations,
                    },
                    exec_config.emit_events,
                )

                # Execute with timeout, optionally using copy-on-write
                if use_cow:
                    # Wrap state in COW wrapper for efficient read-heavy nodes
                    cow_state: CopyOnWriteState[StateType] = CopyOnWriteState(state)
                    if exec_config.timeout:
                        remaining = exec_config.timeout - (time.time() - start_time)
                        if remaining <= 0:
                            return ExecutionResult(
                                state=state,
                                success=False,
                                error="Execution timeout",
                                iterations=iterations,
                                duration=time.time() - start_time,
                                node_history=node_history,
                            )
                        result = await asyncio.wait_for(
                            node.execute(cow_state), timeout=remaining  # type: ignore
                        )
                    else:
                        result = await node.execute(cow_state)  # type: ignore

                    # Extract final state from COW wrapper or result
                    if isinstance(result, CopyOnWriteState):
                        state = result.get_state()
                    elif isinstance(result, dict):
                        state = result
                    else:
                        # Node returned something else, use COW state
                        state = cow_state.get_state()
                else:
                    # Traditional deep copy approach
                    if exec_config.timeout:
                        remaining = exec_config.timeout - (time.time() - start_time)
                        if remaining <= 0:
                            return ExecutionResult(
                                state=state,
                                success=False,
                                error="Execution timeout",
                                iterations=iterations,
                                duration=time.time() - start_time,
                                node_history=node_history,
                            )
                        state = await asyncio.wait_for(node.execute(state), timeout=remaining)
                    else:
                        state = await node.execute(state)

                # Emit node_end event for observability
                self._emit_event(
                    "node_end",
                    graph_id,
                    {
                        "node_id": current_node,
                        "iteration": iterations,
                        "duration": time.time() - node_start_time,
                        "success": True,
                    },
                    exec_config.emit_events,
                )

                # WorkflowCheckpoint after execution
                if exec_config.checkpointer:
                    await self._save_checkpoint(
                        exec_config.checkpointer, thread_id, current_node, state
                    )

                # Check for interrupt after
                if current_node in exec_config.interrupt_after:
                    logger.info(f"Interrupt after node: {current_node}")
                    return ExecutionResult(
                        state=state,
                        success=True,
                        iterations=iterations,
                        duration=time.time() - start_time,
                        node_history=node_history,
                    )

                # Get next node
                current_node = self._get_next_node(current_node, state)

            # Emit RL event for successful completion
            self._emit_graph_completed_event(
                success=True,
                iterations=iterations,
                duration=time.time() - start_time,
            )

            # Emit graph_completed event for observability
            self._emit_event(
                "graph_completed",
                graph_id,
                {
                    "success": True,
                    "iterations": iterations,
                    "duration": time.time() - start_time,
                    "node_count": len(node_history),
                },
                exec_config.emit_events,
            )

            return ExecutionResult(
                state=state,
                success=True,
                iterations=iterations,
                duration=time.time() - start_time,
                node_history=node_history,
            )

        except asyncio.TimeoutError:
            # Emit graph_error event for observability
            self._emit_event(
                "graph_error",
                graph_id,
                {
                    "error": "Execution timeout",
                    "iterations": iterations,
                    "duration": time.time() - start_time,
                },
                exec_config.emit_events,
            )
            return ExecutionResult(
                state=state,
                success=False,
                error="Execution timeout",
                iterations=iterations,
                duration=time.time() - start_time,
                node_history=node_history,
            )

        except Exception as e:
            logger.error(f"Graph execution failed: {e}", exc_info=True)
            # Emit graph_error event for observability
            self._emit_event(
                "graph_error",
                graph_id,
                {
                    "error": str(e),
                    "iterations": iterations,
                    "duration": time.time() - start_time,
                },
                exec_config.emit_events,
            )
            return ExecutionResult(
                state=state,
                success=False,
                error=str(e),
                iterations=iterations,
                duration=time.time() - start_time,
                node_history=node_history,
            )

    def _get_next_node(self, current_node: str, state: Any) -> str:
        """Determine next node based on edges and state.

        Args:
            current_node: Current node ID
            state: Current state

        Returns:
            Next node ID or END
        """
        edges = self._edges.get(current_node, [])
        if not edges:
            return END

        for edge in edges:
            target = edge.get_target(state)
            if target:
                return target

        # Default to first edge if no conditional matches
        if edges and edges[0].edge_type == EdgeType.NORMAL:
            return edges[0].target if isinstance(edges[0].target, str) else END

        return END

    async def _save_checkpoint(
        self,
        checkpointer: CheckpointerProtocol,
        thread_id: str,
        node_id: str,
        state: Any,
    ) -> None:
        """Save a checkpoint.

        Args:
            checkpointer: Checkpointer instance
            thread_id: Thread identifier
            node_id: Current node ID
            state: Current state
        """
        checkpoint = WorkflowCheckpoint(
            checkpoint_id=uuid.uuid4().hex,
            thread_id=thread_id,
            node_id=node_id,
            state=dict(state) if isinstance(state, dict) else state,
            timestamp=time.time(),
        )
        await checkpointer.save(checkpoint)

    def _emit_graph_completed_event(
        self,
        success: bool,
        iterations: int,
        duration: float,
    ) -> None:
        """Emit RL event for graph completion."""
        try:
            from victor.agent.rl.hooks import get_rl_hooks, RLEvent, RLEventType

            hooks = get_rl_hooks()
            if hooks is None:
                return

            quality = 0.8 if success else 0.2
            if success and iterations < 10:
                quality += 0.1
            if success and duration < 30:
                quality += 0.1

            event = RLEvent(
                type=RLEventType.WORKFLOW_COMPLETED,
                workflow_name="state_graph",
                success=success,
                quality_score=min(1.0, quality),
                metadata={
                    "iterations": iterations,
                    "duration_seconds": duration,
                    "graph_type": "state_graph",
                },
            )
            hooks.emit(event)

        except Exception as e:
            logger.debug(f"Graph event emission failed: {e}")

    async def stream(
        self,
        input_state: StateType,
        *,
        config: Optional[GraphConfig] = None,
        thread_id: Optional[str] = None,
    ):
        """Stream execution yielding state after each node.

        Args:
            input_state: Initial state
            config: Override execution config
            thread_id: Thread ID for checkpointing

        Yields:
            Tuple of (node_id, state) after each execution
        """
        exec_config = config or self._config
        thread_id = thread_id or uuid.uuid4().hex

        state = copy.deepcopy(input_state)
        current_node = self._entry_point
        iterations = 0
        visited_count: Dict[str, int] = {}

        while current_node != END:
            iterations += 1
            if iterations > exec_config.max_iterations:
                break

            visited_count[current_node] = visited_count.get(current_node, 0) + 1
            if visited_count[current_node] > exec_config.recursion_limit:
                break

            node = self._nodes.get(current_node)
            if not node:
                break

            state = await node.execute(state)
            yield current_node, state

            current_node = self._get_next_node(current_node, state)

    def get_graph_schema(self) -> Dict[str, Any]:
        """Get graph structure as dictionary.

        Returns:
            Dictionary describing nodes and edges
        """
        return {
            "nodes": list(self._nodes.keys()),
            "edges": {
                src: [
                    {
                        "target": e.target,
                        "type": e.edge_type.value,
                    }
                    for e in edges
                ]
                for src, edges in self._edges.items()
            },
            "entry_point": self._entry_point,
        }


class StateGraph(Generic[StateType]):
    """StateGraph builder for creating stateful workflows.

    Provides a LangGraph-compatible API for building graph workflows
    with typed state, cyclic support, and checkpointing.

    Example:
        graph = StateGraph(AgentState)
        graph.add_node("analyze", analyze_func)
        graph.add_node("execute", execute_func)
        graph.add_edge("analyze", "execute")
        graph.add_conditional_edge(
            "execute",
            should_retry,
            {"retry": "analyze", "done": END}
        )
        graph.set_entry_point("analyze")

        app = graph.compile()
        result = await app.invoke(initial_state)
    """

    def __init__(
        self,
        state_schema: Optional[Type[StateType]] = None,
        config_schema: Optional[Type] = None,
    ):
        """Initialize StateGraph.

        Args:
            state_schema: Optional type for state validation
            config_schema: Optional type for config validation
        """
        self._state_schema = state_schema
        self._config_schema = config_schema
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, List[Edge]] = {}
        self._entry_point: Optional[str] = None

    def add_node(
        self,
        node_id: str,
        func: Callable[[StateType], Union[StateType, Awaitable[StateType]]],
        **metadata: Any,
    ) -> "StateGraph[StateType]":
        """Add a node to the graph.

        Args:
            node_id: Unique node identifier
            func: Node execution function
            **metadata: Additional metadata

        Returns:
            Self for chaining

        Raises:
            ValueError: If node already exists
        """
        if node_id in self._nodes:
            raise ValueError(f"Node '{node_id}' already exists")

        self._nodes[node_id] = Node(id=node_id, func=func, metadata=metadata)
        logger.debug(f"Added node: {node_id}")
        return self

    def add_edge(
        self,
        source: str,
        target: str,
    ) -> "StateGraph[StateType]":
        """Add a normal edge between nodes.

        Args:
            source: Source node ID
            target: Target node ID (or END)

        Returns:
            Self for chaining
        """
        if source not in self._edges:
            self._edges[source] = []

        edge = Edge(source=source, target=target, edge_type=EdgeType.NORMAL)
        self._edges[source].append(edge)
        logger.debug(f"Added edge: {source} -> {target}")
        return self

    def add_conditional_edge(
        self,
        source: str,
        condition: Callable[[StateType], str],
        branches: Dict[str, str],
    ) -> "StateGraph[StateType]":
        """Add a conditional edge with multiple branches.

        Args:
            source: Source node ID
            condition: Function that returns branch name
            branches: Mapping from branch names to target node IDs

        Returns:
            Self for chaining
        """
        if source not in self._edges:
            self._edges[source] = []

        edge = Edge(
            source=source,
            target=branches,
            edge_type=EdgeType.CONDITIONAL,
            condition=condition,
        )
        self._edges[source].append(edge)
        logger.debug(f"Added conditional edge: {source} -> {list(branches.values())}")
        return self

    def set_entry_point(self, node_id: str) -> "StateGraph[StateType]":
        """Set the entry point node.

        Args:
            node_id: Node to start execution from

        Returns:
            Self for chaining
        """
        if node_id not in self._nodes:
            raise ValueError(f"Node '{node_id}' not found")
        self._entry_point = node_id
        return self

    def set_finish_point(self, node_id: str) -> "StateGraph[StateType]":
        """Set a node as finish point (adds edge to END).

        Args:
            node_id: Node that finishes the graph

        Returns:
            Self for chaining
        """
        return self.add_edge(node_id, END)

    def compile(
        self,
        checkpointer: Optional[CheckpointerProtocol] = None,
        **config_kwargs: Any,
    ) -> CompiledGraph[StateType]:
        """Compile the graph for execution.

        Validates graph structure and creates optimized execution plan.

        Args:
            checkpointer: Optional checkpointer for persistence
            **config_kwargs: Additional config options

        Returns:
            CompiledGraph ready for execution

        Raises:
            ValueError: If graph is invalid
        """
        # Validate
        errors = self._validate()
        if errors:
            raise ValueError(f"Invalid graph: {'; '.join(errors)}")

        # Create config
        config = GraphConfig(
            checkpointer=checkpointer,
            **config_kwargs,
        )

        return CompiledGraph(
            nodes=self._nodes.copy(),
            edges={k: list(v) for k, v in self._edges.items()},
            entry_point=self._entry_point,
            state_schema=self._state_schema,
            config=config,
        )

    def _validate(self) -> List[str]:
        """Validate graph structure.

        Returns:
            List of error messages
        """
        errors = []

        if not self._nodes:
            errors.append("Graph has no nodes")

        if not self._entry_point:
            errors.append("No entry point set")
        elif self._entry_point not in self._nodes:
            errors.append(f"Entry point '{self._entry_point}' not found")

        # Check edge targets exist
        for source, edges in self._edges.items():
            if source not in self._nodes:
                errors.append(f"Edge source '{source}' not found")

            for edge in edges:
                if isinstance(edge.target, str):
                    if edge.target != END and edge.target not in self._nodes:
                        errors.append(f"Edge target '{edge.target}' not found")
                elif isinstance(edge.target, dict):
                    for branch, target in edge.target.items():
                        if target != END and target not in self._nodes:
                            errors.append(
                                f"Conditional target '{target}' not found " f"(branch: {branch})"
                            )

        # Check all nodes are reachable
        reachable = self._find_reachable()
        for node_id in self._nodes:
            if node_id not in reachable and node_id != self._entry_point:
                errors.append(f"Node '{node_id}' is unreachable")

        return errors

    def _find_reachable(self) -> Set[str]:
        """Find all reachable nodes from entry point."""
        if not self._entry_point:
            return set()

        reachable = set()
        to_visit = [self._entry_point]

        while to_visit:
            node_id = to_visit.pop()
            if node_id in reachable or node_id == END:
                continue

            reachable.add(node_id)

            for edge in self._edges.get(node_id, []):
                if isinstance(edge.target, str):
                    to_visit.append(edge.target)
                elif isinstance(edge.target, dict):
                    to_visit.extend(edge.target.values())

        return reachable


# Convenience factory functions
def create_graph(
    state_schema: Optional[Type[StateType]] = None,
) -> StateGraph[StateType]:
    """Create a new StateGraph.

    Args:
        state_schema: Optional type for state validation

    Returns:
        New StateGraph instance
    """
    return StateGraph(state_schema)


__all__ = [
    # Core types
    "StateGraph",
    "CompiledGraph",
    "Node",
    "Edge",
    "EdgeType",
    "FrameworkNodeStatus",
    # State management
    "CopyOnWriteState",  # Copy-on-write state wrapper (P2 scalability)
    # Execution
    "ExecutionResult",
    "GraphConfig",
    # Checkpointing
    "WorkflowCheckpoint",
    "CheckpointerProtocol",
    "MemoryCheckpointer",
    "RLCheckpointerAdapter",  # Uses existing RL CheckpointStore
    # Protocols
    "StateProtocol",
    "NodeFunctionProtocol",
    "ConditionFunctionProtocol",
    # Constants
    "END",
    "START",
    # Factory
    "create_graph",
]
