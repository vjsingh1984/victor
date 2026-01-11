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

# Import focused configs for ISP compliance
from victor.framework.config import (
    GraphConfig as GraphConfig,
    ExecutionConfig,
    CheckpointConfig,
    InterruptConfig,
    PerformanceConfig,
    ObservabilityConfig,
)

# Type variables for generic state
StateType = TypeVar("StateType", bound=Dict[str, Any])
T = TypeVar("T")

# Sentinel for end of graph
END = "__end__"
START = "__start__"


class CopyOnWriteState(Generic[StateType]):
    """Copy-on-write wrapper for workflow state.

    MIGRATION NOTICE: For persistent state storage across workflow executions,
    use the canonical state management system:
        - victor.state.WorkflowStateManager - Workflow scope state
        - victor.state.get_global_manager() - Unified access to all scopes

    CopyOnWriteState is kept as a performance optimization for workflow graphs,
    providing copy-on-write semantics for state within a single execution.

    ⚠️ THREAD SAFETY WARNING ⚠️:
        This class is NOT thread-safe and must NOT be shared across threads.

        Each thread MUST have its own CopyOnWriteState wrapper instance.
        Sharing the same wrapper instance between threads will lead to
        race conditions, data corruption, and undefined behavior.

        Example of CORRECT usage:
            # Thread 1
            cow_state_1 = CopyOnWriteState(shared_state_dict)
            result_1 = await node1.execute(cow_state_1)

            # Thread 2 (different wrapper!)
            cow_state_2 = CopyOnWriteState(shared_state_dict)
            result_2 = await node2.execute(cow_state_2)

        Example of INCORRECT usage (will cause race conditions):
            cow_state = CopyOnWriteState(shared_state_dict)
            # ❌ DO NOT share cow_state between threads
            await thread1.run(cow_state)  # UNSAFE!
            await thread2.run(cow_state)  # UNSAFE!

    ---

    Legacy Documentation:

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

    Migration Example:
        # OLD (using CopyOnWriteState for persistent storage):
        cow_state = CopyOnWriteState({"task_id": "task-123"})
        cow_state["status"] = "running"
        final_state = cow_state.get_state()

        # NEW (using canonical state management):
        from victor.state import WorkflowStateManager, StateScope

        mgr = WorkflowStateManager()
        await mgr.set("task_id", "task-123")
        await mgr.set("status", "running")

        # OR for unified access:
        from victor.state import get_global_manager
        state = get_global_manager()
        await state.set("task_id", "task-123", scope=StateScope.WORKFLOW)
        await state.set("status", "running", scope=StateScope.WORKFLOW)

    Performance characteristics:
        - Read operations: O(1), no copy overhead
        - First write: O(n) deep copy where n is state size
        - Subsequent writes: O(1), no additional copy
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
            from victor.framework.rl.checkpoint_store import get_checkpoint_store

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


# Note: GraphConfig is now imported from victor.framework.graph.config
# This provides ISP compliance through focused config classes:
# - ExecutionConfig: execution limits
# - CheckpointConfig: state persistence
# - InterruptConfig: interrupt behavior
# - PerformanceConfig: performance optimizations
# - ObservabilityConfig: observability and eventing
# GraphConfig remains as a facade composing these focused configs


@dataclass
class GraphExecutionResult(Generic[StateType]):
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


# =============================================================================
# Graph Execution Helpers (SRP Compliance)
# =============================================================================

class IterationController:
    """Controls graph iteration logic (SRP: Single Responsibility).

    Manages iteration limits and recursion tracking to prevent infinite loops.
    """

    def __init__(self, max_iterations: int, recursion_limit: int):
        """Initialize iteration controller.

        Args:
            max_iterations: Maximum total iterations allowed
            recursion_limit: Maximum visits to same node (recursion depth)
        """
        self.max_iterations = max_iterations
        self.recursion_limit = recursion_limit
        self.iterations = 0
        self.visited_count: Dict[str, int] = {}

    def should_continue(self, current_node: str) -> tuple[bool, Optional[str]]:
        """Check if execution should continue.

        Args:
            current_node: Current node being executed

        Returns:
            Tuple of (should_continue, error_message)
            - (True, None) if execution should continue
            - (False, error_message) if limit exceeded
        """
        # Check iteration limit
        self.iterations += 1
        if self.iterations > self.max_iterations:
            return False, f"Max iterations ({self.max_iterations}) exceeded"

        # Track cycles
        self.visited_count[current_node] = self.visited_count.get(current_node, 0) + 1
        if self.visited_count[current_node] > self.recursion_limit:
            return False, f"Recursion limit exceeded at node: {current_node}"

        return True, None

    def reset(self):
        """Reset iteration state."""
        self.iterations = 0
        self.visited_count.clear()


class TimeoutManager:
    """Manages execution timeouts (SRP: Single Responsibility).

    Tracks elapsed time and enforces timeout limits.
    """

    def __init__(self, timeout: Optional[float]):
        """Initialize timeout manager.

        Args:
            timeout: Overall execution timeout in seconds (None = no limit)
        """
        self.timeout = timeout
        self.start_time: Optional[float] = None

    def start(self):
        """Start timeout tracking."""
        self.start_time = time.time()

    def get_remaining(self) -> Optional[float]:
        """Get remaining time before timeout.

        Returns:
            Remaining seconds, or None if no timeout configured
        """
        if self.timeout is None or self.start_time is None:
            return None
        return self.timeout - (time.time() - self.start_time)

    def is_expired(self) -> bool:
        """Check if timeout has expired.

        Returns:
            True if timeout has been exceeded
        """
        remaining = self.get_remaining()
        return remaining is not None and remaining <= 0

    def get_elapsed(self) -> float:
        """Get elapsed time since start.

        Returns:
            Elapsed seconds, or 0.0 if not started
        """
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


class InterruptHandler:
    """Handles graph interrupts for human-in-the-loop workflows (SRP)."""

    def __init__(self, interrupt_before: List[str], interrupt_after: List[str]):
        """Initialize interrupt handler.

        Args:
            interrupt_before: List of node IDs to interrupt before execution
            interrupt_after: List of node IDs to interrupt after execution
        """
        self.interrupt_before = set(interrupt_before)
        self.interrupt_after = set(interrupt_after)

    def should_interrupt_before(self, node_id: str) -> bool:
        """Check if should interrupt before node execution.

        Args:
            node_id: Node to check

        Returns:
            True if execution should interrupt before this node
        """
        return node_id in self.interrupt_before

    def should_interrupt_after(self, node_id: str) -> bool:
        """Check if should interrupt after node execution.

        Args:
            node_id: Node to check

        Returns:
            True if execution should interrupt after this node
        """
        return node_id in self.interrupt_after


class NodeExecutor:
    """Executes individual graph nodes (SRP: Single Responsibility).

    Handles node lookup, execution with timeout, and copy-on-write state management.
    """

    def __init__(self, nodes: Dict[str, Node], use_copy_on_write: bool):
        """Initialize node executor.

        Args:
            nodes: Dictionary of available nodes
            use_copy_on_write: Whether to use copy-on-write optimization
        """
        self.nodes = nodes
        self.use_copy_on_write = use_copy_on_write

    async def execute(
        self,
        node_id: str,
        state: StateType,
        timeout_manager: TimeoutManager,
    ) -> tuple[bool, Optional[str], StateType]:
        """Execute a node.

        Args:
            node_id: ID of node to execute
            state: Current state
            timeout_manager: Timeout manager for execution limits

        Returns:
            Tuple of (success, error_message, new_state)
        """
        node = self.nodes.get(node_id)
        if not node:
            return False, f"Node not found: {node_id}", state

        try:
            # Check timeout before execution
            if timeout_manager.is_expired():
                return False, "Execution timeout", state

            remaining = timeout_manager.get_remaining()

            # Execute with copy-on-write or traditional approach
            if self.use_copy_on_write:
                cow_state: CopyOnWriteState[StateType] = CopyOnWriteState(state)
                if remaining is not None:
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
                if remaining is not None:
                    state = await asyncio.wait_for(
                        node.execute(state), timeout=remaining
                    )
                else:
                    state = await node.execute(state)

            return True, None, state

        except asyncio.TimeoutError:
            return False, "Execution timeout", state
        except Exception as e:
            return False, str(e), state


class GraphCheckpointManager:
    """Manages state checkpointing for graph workflows (SRP: Single Responsibility).

    Handles loading initial state from checkpoints and saving checkpoints.

    Note: Renamed from CheckpointManager to GraphCheckpointManager to be
    semantically distinct from:
    - GitCheckpointManager (victor.agent.checkpoints): Git stash-based checkpoints
    - ConversationCheckpointManager (victor.storage.checkpoints): Conversation state
    """

    def __init__(self, checkpointer: Optional[CheckpointerProtocol]):
        """Initialize checkpoint manager.

        Args:
            checkpointer: Checkpointer for persistence (None = no checkpointing)
        """
        self.checkpointer = checkpointer

    async def load_initial_state(
        self,
        thread_id: str,
        input_state: StateType,
        entry_point: str,
    ) -> tuple[StateType, str]:
        """Load initial state from checkpoint or use input state.

        Args:
            thread_id: Thread ID for checkpoint lookup
            input_state: Input state if no checkpoint exists
            entry_point: Default entry point if no checkpoint exists

        Returns:
            Tuple of (initial_state, starting_node)
        """
        if self.checkpointer:
            checkpoint = await self.checkpointer.load(thread_id)
            if checkpoint:
                logger.info(f"Resuming from checkpoint at node: {checkpoint.node_id}")
                return checkpoint.state.copy(), checkpoint.node_id

        # No checkpoint, use input state
        return copy.deepcopy(input_state), entry_point

    async def save_checkpoint(
        self,
        thread_id: str,
        node_id: str,
        state: StateType,
    ) -> None:
        """Save checkpoint.

        Args:
            thread_id: Thread ID for checkpoint
            node_id: Current node ID
            state: Current state to checkpoint
        """
        if self.checkpointer:
            checkpoint = WorkflowCheckpoint(
                checkpoint_id=f"{thread_id}_{node_id}_{time.time()}",
                thread_id=thread_id,
                node_id=node_id,
                state=state,
                timestamp=time.time(),
            )
            await self.checkpointer.save(checkpoint)


class GraphEventEmitter:
    """Emits graph execution events for observability (SRP: Single Responsibility)."""

    def __init__(self, graph_id: str, emit_events: bool):
        """Initialize event emitter.

        Args:
            graph_id: Graph identifier for event correlation
            emit_events: Whether to emit events (or silently no-op)
        """
        self.graph_id = graph_id
        self.emit_events = emit_events

    def emit_graph_started(self, entry_point: str, node_count: int, thread_id: str):
        """Emit graph started event."""
        if not self.emit_events:
            return

        try:
            from victor.core.events import get_observability_bus as get_event_bus

            bus = get_event_bus()
            bus.emit_lifecycle_event(
                "graph_started",
                {
                    "graph_id": self.graph_id,
                    "source": "StateGraph",
                    "entry_point": entry_point,
                    "node_count": node_count,
                    "thread_id": thread_id,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit graph_started event: {e}")

    def emit_node_start(self, node_id: str, iteration: int):
        """Emit node start event."""
        if not self.emit_events:
            return

        try:
            from victor.core.events import get_observability_bus as get_event_bus

            bus = get_event_bus()
            bus.emit_lifecycle_event(
                "node_start",
                {
                    "graph_id": self.graph_id,
                    "source": "StateGraph",
                    "node_id": node_id,
                    "iteration": iteration,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit node_start event: {e}")

    def emit_node_complete(self, node_id: str, iteration: int, duration: float):
        """Emit node complete event."""
        if not self.emit_events:
            return

        try:
            from victor.core.events import get_observability_bus as get_event_bus

            bus = get_event_bus()
            bus.emit_lifecycle_event(
                "node_end",
                {
                    "graph_id": self.graph_id,
                    "source": "StateGraph",
                    "node_id": node_id,
                    "iteration": iteration,
                    "duration": duration,
                    "success": True,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit node_end event: {e}")

    def emit_graph_completed(self, success: bool, iterations: int, duration: float, node_count: int):
        """Emit graph completed event."""
        if not self.emit_events:
            return

        try:
            from victor.core.events import get_observability_bus as get_event_bus

            bus = get_event_bus()
            bus.emit_lifecycle_event(
                "graph_completed",
                {
                    "graph_id": self.graph_id,
                    "source": "StateGraph",
                    "success": success,
                    "iterations": iterations,
                    "duration": duration,
                    "node_count": node_count,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit graph_completed event: {e}")

    def emit_graph_error(self, error: str, iterations: int, duration: float):
        """Emit graph error event."""
        if not self.emit_events:
            return

        try:
            from victor.core.events import get_observability_bus as get_event_bus

            bus = get_event_bus()
            bus.emit_lifecycle_event(
                "graph_error",
                {
                    "graph_id": self.graph_id,
                    "source": "StateGraph",
                    "error": error,
                    "iterations": iterations,
                    "duration": duration,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit graph_error event: {e}")


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
        self._debug_hook: Optional[Any] = None  # DebugHook for debugging

    def set_debug_hook(self, hook: Optional[Any]) -> None:
        """Set debug hook for execution.

        Args:
            hook: DebugHook instance or None to disable debugging
        """
        self._debug_hook = hook

    @property
    def graph(self) -> "CompiledGraph[StateType]":
        """Return the compiled graph itself.

        This property provides a self-reference for compatibility
        with APIs that expect a .graph attribute.

        Returns:
            Self reference to the compiled graph
        """
        return self

    def _should_use_cow(self, exec_config: GraphConfig) -> bool:
        """Determine if copy-on-write should be used.

        Args:
            exec_config: Execution configuration (with focused configs)

        Returns:
            True if COW should be enabled
        """
        # Explicit config takes precedence
        if exec_config.performance.use_copy_on_write is not None:
            return exec_config.performance.use_copy_on_write

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
            from victor.core.events import get_observability_bus as get_event_bus

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
        debug_hook: Optional[Any] = None,
    ) -> GraphExecutionResult[StateType]:
        """Execute the graph (SRP: Orchestrates focused helpers).

        Delegates to specialized helper classes for:
        - IterationController: iteration/recursion limits
        - TimeoutManager: timeout tracking
        - InterruptHandler: human-in-the-loop interrupts
        - NodeExecutor: node execution with COW optimization
        - GraphCheckpointManager: state persistence
        - GraphEventEmitter: observability events
        - DebugHook: breakpoint and execution control (optional)

        Args:
            input_state: Initial state
            config: Override execution config
            thread_id: Thread ID for checkpointing
            debug_hook: Optional DebugHook for debugging

        Returns:
            GraphExecutionResult with final state
        """
        exec_config = config or self._config
        thread_id = thread_id or uuid.uuid4().hex
        use_cow = self._should_use_cow(exec_config)
        graph_id = exec_config.observability.graph_id or thread_id

        # Use parameter debug hook or instance debug hook
        hook = debug_hook or self._debug_hook

        # Create helper instances (Dependency Injection)
        iteration_controller = IterationController(
            max_iterations=exec_config.execution.max_iterations,
            recursion_limit=exec_config.execution.recursion_limit,
        )
        timeout_manager = TimeoutManager(timeout=exec_config.execution.timeout)
        interrupt_handler = InterruptHandler(
            interrupt_before=exec_config.interrupt.interrupt_before,
            interrupt_after=exec_config.interrupt.interrupt_after,
        )
        node_executor = NodeExecutor(nodes=self._nodes, use_copy_on_write=use_cow)
        checkpoint_manager = GraphCheckpointManager(checkpointer=exec_config.checkpoint.checkpointer)
        event_emitter = GraphEventEmitter(
            graph_id=graph_id,
            emit_events=exec_config.observability.emit_events,
        )

        # Load initial state (from checkpoint or input)
        state, current_node = await checkpoint_manager.load_initial_state(
            thread_id=thread_id,
            input_state=input_state,
            entry_point=self._entry_point,
        )

        # Start timeout tracking and emit graph started event
        timeout_manager.start()
        event_emitter.emit_graph_started(
            entry_point=self._entry_point,
            node_count=len(self._nodes),
            thread_id=thread_id,
        )

        node_history: List[str] = []

        try:
            while current_node != END:
                # Check iteration limits (delegated to IterationController)
                should_continue, error = iteration_controller.should_continue(current_node)
                if not should_continue:
                    logger.warning(f"Iteration limit reached: {error}")
                    return GraphExecutionResult(
                        state=state,
                        success=False,
                        error=error,
                        iterations=iteration_controller.iterations,
                        duration=timeout_manager.get_elapsed(),
                        node_history=node_history,
                    )

                # Check interrupt before (delegated to InterruptHandler)
                if interrupt_handler.should_interrupt_before(current_node):
                    logger.info(f"Interrupt before node: {current_node}")
                    await checkpoint_manager.save_checkpoint(thread_id, current_node, state)
                    return GraphExecutionResult(
                        state=state,
                        success=True,
                        iterations=iteration_controller.iterations,
                        duration=timeout_manager.get_elapsed(),
                        node_history=node_history,
                    )

                # Debug hook - before node
                if hook:
                    await hook.before_node(current_node, state)

                # Emit node start event
                node_start_time = time.time()
                event_emitter.emit_node_start(
                    node_id=current_node,
                    iteration=iteration_controller.iterations,
                )

                # Execute node (delegated to NodeExecutor)
                success, error, state = await node_executor.execute(
                    node_id=current_node,
                    state=state,
                    timeout_manager=timeout_manager,
                )

                # Debug hook - after node
                if hook:
                    await hook.after_node(
                        current_node, state, error if not success else None
                    )

                if not success:
                    return GraphExecutionResult(
                        state=state,
                        success=False,
                        error=error,
                        iterations=iteration_controller.iterations,
                        duration=timeout_manager.get_elapsed(),
                        node_history=node_history,
                    )

                # Track execution
                logger.debug(f"Executed node: {current_node}")
                node_history.append(current_node)

                # Emit node complete event
                event_emitter.emit_node_complete(
                    node_id=current_node,
                    iteration=iteration_controller.iterations,
                    duration=time.time() - node_start_time,
                )

                # Save checkpoint after node execution
                await checkpoint_manager.save_checkpoint(thread_id, current_node, state)

                # Check interrupt after
                if interrupt_handler.should_interrupt_after(current_node):
                    logger.info(f"Interrupt after node: {current_node}")
                    return GraphExecutionResult(
                        state=state,
                        success=True,
                        iterations=iteration_controller.iterations,
                        duration=timeout_manager.get_elapsed(),
                        node_history=node_history,
                    )

                # Get next node
                current_node = self._get_next_node(current_node, state)

            # Emit RL event for successful completion
            self._emit_graph_completed_event(
                success=True,
                iterations=iteration_controller.iterations,
                duration=timeout_manager.get_elapsed(),
            )

            # Emit graph completed event
            event_emitter.emit_graph_completed(
                success=True,
                iterations=iteration_controller.iterations,
                duration=timeout_manager.get_elapsed(),
                node_count=len(node_history),
            )

            return GraphExecutionResult(
                state=state,
                success=True,
                iterations=iteration_controller.iterations,
                duration=timeout_manager.get_elapsed(),
                node_history=node_history,
            )

        except asyncio.TimeoutError:
            event_emitter.emit_graph_error(
                error="Execution timeout",
                iterations=iteration_controller.iterations,
                duration=timeout_manager.get_elapsed(),
            )
            return GraphExecutionResult(
                state=state,
                success=False,
                error="Execution timeout",
                iterations=iteration_controller.iterations,
                duration=timeout_manager.get_elapsed(),
                node_history=node_history,
            )

        except Exception as e:
            logger.error(f"Graph execution failed: {e}", exc_info=True)
            event_emitter.emit_graph_error(
                error=str(e),
                iterations=iteration_controller.iterations,
                duration=timeout_manager.get_elapsed(),
            )
            return GraphExecutionResult(
                state=state,
                success=False,
                error=str(e),
                iterations=iteration_controller.iterations,
                duration=timeout_manager.get_elapsed(),
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
            from victor.framework.rl.hooks import get_rl_hooks, RLEvent, RLEventType

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
            if iterations > exec_config.execution.max_iterations:
                break

            visited_count[current_node] = visited_count.get(current_node, 0) + 1
            if visited_count[current_node] > exec_config.execution.recursion_limit:
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

        # Create config (use from_legacy to support both legacy and focused config formats)
        config = GraphConfig.from_legacy(
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

    @classmethod
    def from_schema(
        cls,
        schema: Union[Dict[str, Any], str],
        state_schema: Optional[Type[StateType]] = None,
        node_registry: Optional[Dict[str, Callable]] = None,
        condition_registry: Optional[Dict[str, Callable]] = None,
    ) -> "StateGraph[StateType]":
        """Create StateGraph from schema dictionary or YAML string.

        This enables dynamic graph generation from serialized schemas,
        supporting Phase 3.0 requirements for workflow persistence and
        external graph definition.

        Args:
            schema: Either a dictionary schema or YAML string containing:
                - nodes: List of node definitions with id and type
                - edges: List of edge definitions with source, target, type
                - entry_point: Starting node ID
                - Optional: state_schema, metadata
            state_schema: Optional TypedDict type for state validation
            node_registry: Registry of node functions (for 'function' type nodes)
                Maps node function names to callable functions
            condition_registry: Registry of condition functions (for conditional edges)
                Maps condition function names to callable functions

        Returns:
            StateGraph instance ready for compilation

        Raises:
            ValueError: If schema is invalid or missing required fields
            TypeError: If node/condition types are unsupported

        Example:
            # Define schema
            schema = {
                "nodes": [
                    {"id": "analyze", "type": "function", "func": "analyze_task"},
                    {"id": "execute", "type": "function", "func": "execute_task"},
                ],
                "edges": [
                    {"source": "analyze", "target": "execute", "type": "normal"},
                    {
                        "source": "execute",
                        "target": {"retry": "analyze", "done": "__end__"},
                        "type": "conditional",
                        "condition": "should_retry"
                    }
                ],
                "entry_point": "analyze"
            }

            # Create registries
            node_registry = {
                "analyze_task": analyze_task_func,
                "execute_task": execute_task_func,
            }
            condition_registry = {
                "should_retry": should_retry_func,
            }

            # Deserialize
            graph = StateGraph.from_schema(
                schema,
                state_schema=AgentState,
                node_registry=node_registry,
                condition_registry=condition_registry
            )

            # Compile and execute
            app = graph.compile()
            result = await app.invoke(initial_state)

        Example with YAML:
            yaml_schema = \"""
            nodes:
              - id: analyze
                type: function
                func: analyze_task
              - id: execute
                type: function
                func: execute_task
            edges:
              - source: analyze
                target: execute
                type: normal
              - source: execute
                target:
                  retry: analyze
                  done: __end__
                type: conditional
                condition: should_retry
            entry_point: analyze
            \"""

            graph = StateGraph.from_schema(
                yaml_schema,
                node_registry=node_registry,
                condition_registry=condition_registry
            )
        """
        import yaml

        # Parse YAML if string input
        if isinstance(schema, str):
            try:
                schema_dict = yaml.safe_load(schema)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML schema: {e}") from e
        else:
            schema_dict = schema

        # Validate required fields
        required_fields = ["nodes", "edges", "entry_point"]
        missing_fields = [f for f in required_fields if f not in schema_dict]
        if missing_fields:
            raise ValueError(f"Schema missing required fields: {missing_fields}")

        # Initialize registries with defaults
        node_registry = node_registry or {}
        condition_registry = condition_registry or {}

        # Create StateGraph instance
        graph = cls(state_schema=state_schema)

        # Add nodes
        for node_def in schema_dict["nodes"]:
            if not isinstance(node_def, dict):
                raise ValueError(f"Invalid node definition: {node_def}")

            node_id = node_def.get("id")
            if not node_id:
                raise ValueError("Node definition must have 'id' field")

            node_type = node_def.get("type", "function")

            if node_type == "function":
                # Function node - look up in registry
                func_name = node_def.get("func")
                if not func_name:
                    raise ValueError(f"Function node '{node_id}' must specify 'func'")

                if func_name not in node_registry:
                    raise ValueError(
                        f"Node function '{func_name}' not found in node_registry. "
                        f"Available: {list(node_registry.keys())}"
                    )

                node_func = node_registry[func_name]
                metadata = {k: v for k, v in node_def.items() if k not in ["id", "type", "func"]}
                graph.add_node(node_id, node_func, **metadata)

            elif node_type == "passthrough":
                # Passthrough node (identity function)
                def passthrough_func(state):
                    return state

                metadata = {k: v for k, v in node_def.items() if k not in ["id", "type"]}
                graph.add_node(node_id, passthrough_func, **metadata)

            elif node_type == "agent":
                # Agent node - placeholder for workflow execution
                # The actual agent execution is handled by the workflow executor
                def create_agent_placeholder(node_config):
                    def agent_placeholder(state):
                        # Store node config in state for executor to use
                        return {
                            **state,
                            "_pending_agent": node_config,
                        }
                    return agent_placeholder

                metadata = {k: v for k, v in node_def.items() if k not in ["id", "type"]}
                graph.add_node(node_id, create_agent_placeholder(node_def), **metadata)

            elif node_type == "compute":
                # Compute node - placeholder for handler execution
                # The actual compute execution is handled by the workflow executor
                def create_compute_placeholder(node_config):
                    def compute_placeholder(state):
                        # Store node config in state for executor to use
                        return {
                            **state,
                            "_pending_compute": node_config,
                        }
                    return compute_placeholder

                metadata = {k: v for k, v in node_def.items() if k not in ["id", "type"]}
                graph.add_node(node_id, create_compute_placeholder(node_def), **metadata)

            else:
                raise TypeError(f"Unsupported node type: {node_type}")

        # Add edges
        for edge_def in schema_dict["edges"]:
            if not isinstance(edge_def, dict):
                raise ValueError(f"Invalid edge definition: {edge_def}")

            source = edge_def.get("source")
            if not source:
                raise ValueError("Edge definition must have 'source' field")

            target = edge_def.get("target")
            if target is None:
                raise ValueError("Edge definition must have 'target' field")

            edge_type = edge_def.get("type", "normal")

            if edge_type == "normal":
                graph.add_edge(source, target)

            elif edge_type == "conditional":
                condition_name = edge_def.get("condition")
                if not condition_name:
                    raise ValueError(
                        f"Conditional edge from '{source}' must specify 'condition'"
                    )

                if condition_name not in condition_registry:
                    raise ValueError(
                        f"Condition function '{condition_name}' not found in "
                        f"condition_registry. Available: {list(condition_registry.keys())}"
                    )

                if not isinstance(target, dict):
                    raise ValueError(
                        f"Conditional edge target must be dict mapping branches to nodes, "
                        f"got: {type(target)}"
                    )

                condition_func = condition_registry[condition_name]
                graph.add_conditional_edge(source, condition_func, target)

            else:
                raise TypeError(f"Unsupported edge type: {edge_type}")

        # Set entry point
        entry_point = schema_dict["entry_point"]
        if entry_point not in graph._nodes:
            raise ValueError(
                f"Entry point '{entry_point}' not found in nodes. "
                f"Available nodes: {list(graph._nodes.keys())}"
            )

        graph.set_entry_point(entry_point)

        return graph


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
    "GraphExecutionResult",
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
