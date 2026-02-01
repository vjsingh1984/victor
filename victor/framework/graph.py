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
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator

logger = logging.getLogger(__name__)

# Import focused configs for ISP compliance
from victor.framework.config import (
    GraphConfig as GraphConfig,
)
import builtins

# Type variables for generic state
StateType_contra = TypeVar("StateType_contra", contravariant=True)
StateType = TypeVar("StateType", bound=dict[str, Any])
T = TypeVar("T")

# Sentinel for end of graph
END = "__end__"
START = "__start__"


class CopyOnWriteState(Generic[StateType]):
    """Copy-on-write wrapper for workflow state with full thread-safety.

    MIGRATION NOTICE: For persistent state storage across workflow executions,
    use the canonical state management system:
        - victor.state.WorkflowStateManager - Workflow scope state
        - victor.state.get_global_manager() - Unified access to all scopes

    CopyOnWriteState is kept as a performance optimization for workflow graphs,
    providing copy-on-write semantics for state within a single execution.

    ✅ THREAD SAFETY:
        This class IS thread-safe and can be safely shared across threads.
        All read and write operations are protected by an RLock, ensuring
        consistent state access from multiple threads.

        The implementation uses a reentrant lock (RLock) to allow nested
        operations while protecting against concurrent modifications.

        Example of CORRECT usage (multi-threaded):
            cow_state = CopyOnWriteState(initial_state)

            # Thread 1 - safe concurrent read
            value1 = cow_state["key"]

            # Thread 2 - safe concurrent write
            cow_state["other_key"] = "value"

            # Thread 3 - safe concurrent read after write
            value2 = cow_state.get("other_key")

        For graph-level parallelism with branch isolation, create separate
        wrappers per branch:
            # Branch A gets its own isolated copy
            branch_a_state = CopyOnWriteState(shared_state)

            # Branch B gets its own isolated copy
            branch_b_state = CopyOnWriteState(shared_state)

    ---

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
        - Read operations (unmodified): O(1), no copy, no lock contention
        - Read operations (modified): O(1), lock-protected for consistency
        - First write: O(n) deep copy where n is state size
        - Subsequent writes: O(1), lock-protected
    """

    __slots__ = ("_source", "_copy", "_modified", "_lock")

    def __init__(self, source: StateType):
        """Initialize with source state.

        Args:
            source: Original state dictionary (not copied until mutation)
        """
        self._source: StateType = source
        self._copy: Optional[StateType] = None
        self._modified: bool = False
        self._lock = threading.RLock()  # RLock for reentrant thread-safety

    def _ensure_copy(self) -> StateType:
        """Ensure we have a mutable copy of the state (thread-safe).

        Uses double-checked locking for thread-safety while maintaining
        performance for read-heavy workloads. Must be called within the lock
        for write operations.

        Returns:
            The mutable copy of the state
        """
        # Fast path: no lock needed if already modified (immutable after set)
        if self._modified:
            assert self._copy is not None
            return self._copy

        # Slow path: lock and recheck (RLock allows nested calls)
        with self._lock:
            if not self._modified:
                self._copy = copy.deepcopy(self._source)
                self._modified = True
            assert self._copy is not None
            return self._copy

    def _get_current_state(self) -> StateType:
        """Get current state reference (thread-safe read).

        Returns:
            The current state (copy if modified, source otherwise)
        """
        # Fast path for unmodified - source is immutable
        if not self._modified:
            return self._source
        # For modified state, lock ensures consistent read
        with self._lock:
            return self._copy  # type: ignore

    def __getitem__(self, key: str) -> Any:
        """Get item (thread-safe).

        Args:
            key: Key to look up

        Returns:
            Value associated with key

        Raises:
            KeyError: If key is not found
        """
        # Fast path for unmodified - no lock needed, source is immutable
        if not self._modified:
            return self._source[key]
        # Lock for consistent read from modified state
        with self._lock:
            return self._copy[key]  # type: ignore

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item, triggering copy on first mutation (thread-safe).

        Args:
            key: Key to set
            value: Value to associate with key
        """
        with self._lock:
            self._ensure_copy()[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete item, triggering copy on first mutation (thread-safe).

        Args:
            key: Key to delete

        Raises:
            KeyError: If key is not found
        """
        with self._lock:
            del self._ensure_copy()[key]

    def __contains__(self, key: object) -> bool:
        """Check if key exists (thread-safe).

        Args:
            key: Key to check

        Returns:
            True if key exists, False otherwise
        """
        # Fast path for unmodified
        if not self._modified:
            return key in self._source
        with self._lock:
            return key in self._copy  # type: ignore

    def __len__(self) -> int:
        """Get length (thread-safe).

        Returns:
            Number of items in state
        """
        # Fast path for unmodified
        if not self._modified:
            return len(self._source)
        with self._lock:
            return len(self._copy)  # type: ignore

    def __iter__(self) -> Iterator[Any]:
        """Iterate over keys (thread-safe snapshot).

        Returns a snapshot of keys to avoid issues with concurrent modification.

        Returns:
            Iterator over keys from the state
        """
        # Fast path for unmodified
        if not self._modified:
            return iter(list(self._source.keys()))
        # Return snapshot to avoid concurrent modification issues
        with self._lock:
            assert self._copy is not None
            return iter(list(self._copy.keys()))

    def get(self, key: str, default: Any = None) -> Any:
        """Get with default (thread-safe).

        Args:
            key: Key to look up
            default: Value to return if key not found

        Returns:
            Value associated with key, or default
        """
        # Fast path for unmodified
        if not self._modified:
            return self._source.get(key, default)
        with self._lock:
            return self._copy.get(key, default)  # type: ignore

    def keys(self) -> list[Any]:
        """Get keys (thread-safe snapshot).

        Returns:
            List of keys in the state (snapshot for thread-safety)
        """
        # Fast path for unmodified
        if not self._modified:
            return list(self._source.keys())
        with self._lock:
            assert self._copy is not None
            return list(self._copy.keys())

    def values(self) -> list[Any]:
        """Get values (thread-safe snapshot).

        Returns:
            List of values in the state (snapshot for thread-safety)
        """
        # Fast path for unmodified
        if not self._modified:
            return list(self._source.values())
        with self._lock:
            assert self._copy is not None
            return list(self._copy.values())

    def items(self) -> list[tuple[Any, Any]]:
        """Get items (thread-safe snapshot).

        Returns:
            List of (key, value) pairs in the state (snapshot for thread-safety)
        """
        # Fast path for unmodified
        if not self._modified:
            return list(self._source.items())
        with self._lock:
            assert self._copy is not None
            return list(self._copy.items())

    def update(self, other: dict[str, Any]) -> None:
        """Update state, triggering copy (thread-safe).

        Args:
            other: Dictionary of items to update
        """
        with self._lock:
            self._ensure_copy().update(other)

    def setdefault(self, key: str, default: Any = None) -> Any:
        """Set default value, may trigger copy (thread-safe).

        If key is not present, sets it to default (triggering copy).
        If key is present, returns its value without copying.

        Args:
            key: Key to look up or set
            default: Value to set if key not found

        Returns:
            Value associated with key (existing or default)
        """
        with self._lock:
            if key not in self:
                self._ensure_copy()[key] = default
                return default
            return self[key]

    def pop(self, key: str, *args: Any) -> Any:
        """Pop item, triggering copy (thread-safe).

        Args:
            key: Key to pop
            *args: Optional default value

        Returns:
            Value associated with key

        Raises:
            KeyError: If key is not found and no default provided
        """
        with self._lock:
            return self._ensure_copy().pop(key, *args)

    def copy(self) -> dict[str, Any]:
        """Create a shallow copy of the current state (thread-safe).

        Returns:
            Shallow copy of the state dictionary
        """
        # Fast path for unmodified
        if not self._modified:
            return self._source.copy()
        with self._lock:
            assert self._copy is not None
            return self._copy.copy()

    def get_state(self) -> StateType:
        """Get the final state (thread-safe).

        Returns the modified copy if mutations occurred,
        otherwise returns the original source.

        Note: Returns the internal reference. For thread-safe access
        to the returned dict, use copy() or to_dict() instead.

        Returns:
            The current state (modified copy or original source)
        """
        if not self._modified:
            return self._source
        with self._lock:
            return self._copy  # type: ignore

    @property
    def was_modified(self) -> bool:
        """Check if state was modified (copy was made).

        Returns:
            True if any mutation occurred, False otherwise
        """
        return self._modified

    def to_dict(self) -> dict[str, Any]:
        """Convert to regular dictionary (thread-safe deep copy).

        Returns:
            Dictionary copy of the current state
        """
        with self._lock:
            return dict(self.get_state())

    def __repr__(self) -> str:
        """String representation for debugging.

        Returns:
            Debug string showing modification status
        """
        status = "modified" if self._modified else "unmodified"
        with self._lock:
            keys = list(self.keys())
        return f"CopyOnWriteState({status}, keys={keys})"


class EdgeType(Enum):
    """Types of edges in the graph."""

    NORMAL = "normal"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"  # Fork into parallel branches


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

    def __call__(self, state: StateType) -> StateType | Awaitable[StateType]: ...


@runtime_checkable
class ConditionFunctionProtocol(Protocol[StateType_contra]):
    """Protocol for condition functions.

    Condition functions receive state and return a branch name.
    """

    def __call__(self, state: StateType_contra) -> str: ...


@dataclass
class Edge:
    """Represents an edge between nodes.

    Attributes:
        source: Source node ID
        target: Target node ID (or dict for conditional, or list for parallel)
        edge_type: Normal, conditional, or parallel
        condition: Condition function for conditional edges
        merge_func: Function to merge parallel branch results
        join_node: Node ID where parallel branches converge
    """

    source: str
    target: str | dict[str, str] | list[str]
    edge_type: EdgeType = EdgeType.NORMAL
    condition: Optional[Callable[[Any], str]] = None
    merge_func: Optional[Callable[[list[Any]], Any]] = None
    join_node: Optional[str] = None

    def get_target(self, state: Any) -> Optional[str]:
        """Get target node based on state (for non-parallel edges).

        Args:
            state: Current state

        Returns:
            Target node ID or None
        """
        if self.edge_type == EdgeType.NORMAL:
            return self.target if isinstance(self.target, str) else None

        if self.edge_type == EdgeType.PARALLEL:
            # Parallel edges return None here - use get_parallel_targets instead
            return None

        if self.condition is None:
            return None

        branch = self.condition(state)
        if isinstance(self.target, dict):
            return self.target.get(branch)
        return None

    def get_parallel_targets(self) -> list[str]:
        """Get all parallel targets for a parallel edge.

        Returns:
            List of target node IDs for parallel execution
        """
        if self.edge_type != EdgeType.PARALLEL:
            return []
        if isinstance(self.target, list):
            return self.target
        return []

    def is_parallel(self) -> bool:
        """Check if this is a parallel edge."""
        return self.edge_type == EdgeType.PARALLEL


@dataclass
class Node:
    """Represents a node in the graph.

    Attributes:
        id: Unique node identifier
        func: Node execution function
        metadata: Additional node metadata
    """

    id: str
    func: Callable[[Any], Any | Awaitable[Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

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
    state: dict[str, Any]
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> "WorkflowCheckpoint":
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

    async def list(self, thread_id: str) -> builtins.list[WorkflowCheckpoint]:
        """List all checkpoints for thread."""
        ...


class MemoryCheckpointer:
    """In-memory checkpoint storage.

    Suitable for development and testing.
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, list[WorkflowCheckpoint]] = {}

    async def save(self, checkpoint: WorkflowCheckpoint) -> None:
        """Save checkpoint to memory."""
        if checkpoint.thread_id not in self._checkpoints:
            self._checkpoints[checkpoint.thread_id] = []
        self._checkpoints[checkpoint.thread_id].append(checkpoint)

    async def load(self, thread_id: str) -> Optional[WorkflowCheckpoint]:
        """Load latest checkpoint."""
        checkpoints = self._checkpoints.get(thread_id, [])
        return checkpoints[-1] if checkpoints else None

    async def list(self, thread_id: str) -> builtins.list[WorkflowCheckpoint]:
        """List all checkpoints."""
        return self._checkpoints.get(thread_id, [])


class RLCheckpointerAdapter:
    """Adapter to use existing RL CheckpointStore for graph checkpointing.

    Bridges the graph's CheckpointerProtocol with the existing
    victor.agent.rl.checkpoint_store.CheckpointStore infrastructure.
    """

    def __init__(self, learner_name: str = "state_graph") -> None:
        """Initialize adapter.

        Args:
            learner_name: Name to use in checkpoint store (default: "state_graph")
        """
        self.learner_name = learner_name
        self._store: Optional[Any] = None

    def _get_store(self) -> Any:
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

    async def list(self, thread_id: str) -> builtins.list[WorkflowCheckpoint]:
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
class BatchingCheckpointerConfig:
    """Configuration for BatchingCheckpointer.

    Attributes:
        batch_size: Number of checkpoints to accumulate before flushing.
                   Default 10 balances latency vs I/O reduction.
        flush_interval: Maximum seconds between flushes. Default 5.0.
                       Set to None to disable time-based flushing.
        flush_on_load: Whether to flush pending writes before load operations.
                      Default True ensures consistency.
        keep_latest_only: If True, only persist the latest checkpoint per thread
                         when flushing (reduces storage). Default False.
    """

    batch_size: int = 10
    flush_interval: Optional[float] = 5.0
    flush_on_load: bool = True
    keep_latest_only: bool = False


class BatchingCheckpointer:
    """Batching wrapper for checkpoint persistence to reduce I/O pressure.

    Wraps another CheckpointerProtocol implementation and batches writes
    to reduce the frequency of I/O operations. This is particularly useful
    when using persistent backends (SQLite, Redis, PostgreSQL) where each
    write incurs latency.

    The StateGraph engine checkpoints after EVERY node execution. For workflows
    with many nodes, this can become a bottleneck. BatchingCheckpointer accumulates
    checkpoints in memory and flushes them in batches.

    Features:
        - Configurable batch size (default: 10 checkpoints)
        - Time-based auto-flush (default: every 5 seconds)
        - Explicit flush() for graph completion
        - Consistent reads (optional flush before load)
        - Latest-only mode to reduce storage

    Example:
        # Wrap a persistent checkpointer
        sqlite_checkpointer = SQLiteCheckpointer("checkpoints.db")
        batching = BatchingCheckpointer(
            backend=sqlite_checkpointer,
            config=BatchingCheckpointerConfig(
                batch_size=20,
                flush_interval=10.0,
            ),
        )

        # Use with StateGraph
        compiled = graph.compile(checkpointer=batching)
        result = await compiled.invoke(state)

        # Ensure all checkpoints are persisted
        await batching.flush()

    Thread Safety:
        This class uses asyncio.Lock for async safety. It is safe to use
        from multiple asyncio tasks but NOT from multiple threads.
    """

    def __init__(
        self,
        backend: CheckpointerProtocol,
        config: Optional[BatchingCheckpointerConfig] = None,
    ):
        """Initialize batching checkpointer.

        Args:
            backend: Underlying checkpointer to write to
            config: Batching configuration (uses defaults if None)
        """
        self._backend = backend
        self._config = config or BatchingCheckpointerConfig()

        # Pending checkpoints per thread: thread_id -> list of checkpoints
        self._pending: dict[str, list[WorkflowCheckpoint]] = {}

        # Track latest checkpoint per thread for fast reads
        self._latest: dict[str, WorkflowCheckpoint] = {}

        # Async lock for thread safety
        self._lock = asyncio.Lock()

        # Track total pending count
        self._pending_count = 0

        # Last flush time for interval-based flushing
        self._last_flush_time = time.time()

        # Background flush task (if interval-based flushing enabled)
        self._flush_task: Optional[asyncio.Task[None]] = None
        self._shutdown = False

    async def save(self, checkpoint: WorkflowCheckpoint) -> None:
        """Save checkpoint to batch (may trigger flush).

        Adds checkpoint to the pending batch. Automatically flushes if:
        - Batch size is reached
        - Flush interval has elapsed

        Args:
            checkpoint: Checkpoint to save
        """
        async with self._lock:
            thread_id = checkpoint.thread_id

            # Add to pending list
            if thread_id not in self._pending:
                self._pending[thread_id] = []
            self._pending[thread_id].append(checkpoint)
            self._pending_count += 1

            # Update latest reference
            self._latest[thread_id] = checkpoint

            # Check if we should flush
            should_flush = False

            # Batch size trigger
            if self._pending_count >= self._config.batch_size:
                should_flush = True

            # Time interval trigger
            if (
                self._config.flush_interval is not None
                and time.time() - self._last_flush_time >= self._config.flush_interval
            ):
                should_flush = True

            if should_flush:
                await self._flush_unlocked()

    async def load(self, thread_id: str) -> Optional[WorkflowCheckpoint]:
        """Load latest checkpoint for thread.

        Returns the most recent checkpoint, checking both pending
        (in-memory) and persisted checkpoints.

        Args:
            thread_id: Thread identifier

        Returns:
            Latest checkpoint or None if no checkpoints exist
        """
        async with self._lock:
            # Optionally flush before read for consistency
            if self._config.flush_on_load and self._pending_count > 0:
                await self._flush_unlocked()

            # Check in-memory latest first (most recent)
            if thread_id in self._latest:
                return self._latest[thread_id]

            # Fall back to backend
            return await self._backend.load(thread_id)

    async def list(self, thread_id: str) -> builtins.list[WorkflowCheckpoint]:
        """List all checkpoints for thread.

        Combines pending (in-memory) and persisted checkpoints,
        sorted by timestamp.

        Args:
            thread_id: Thread identifier

        Returns:
            List of all checkpoints for the thread
        """
        async with self._lock:
            # Optionally flush before read
            if self._config.flush_on_load and self._pending_count > 0:
                await self._flush_unlocked()

            # Get persisted checkpoints
            persisted = await self._backend.list(thread_id)

            # Add any pending checkpoints not yet flushed
            pending = self._pending.get(thread_id, [])

            # Combine and sort by timestamp
            all_checkpoints = persisted + pending
            all_checkpoints.sort(key=lambda cp: cp.timestamp)

            return all_checkpoints

    async def flush(self) -> int:
        """Flush all pending checkpoints to backend.

        Call this at the end of graph execution to ensure all
        checkpoints are persisted.

        Returns:
            Number of checkpoints flushed
        """
        async with self._lock:
            return await self._flush_unlocked()

    async def _flush_unlocked(self) -> int:
        """Internal flush (must be called with lock held).

        Returns:
            Number of checkpoints flushed
        """
        if self._pending_count == 0:
            return 0

        flushed = 0

        for thread_id, checkpoints in self._pending.items():
            if not checkpoints:
                continue

            if self._config.keep_latest_only:
                # Only save the latest checkpoint per thread
                latest = checkpoints[-1]
                await self._backend.save(latest)
                flushed += 1
            else:
                # Save all pending checkpoints
                for checkpoint in checkpoints:
                    await self._backend.save(checkpoint)
                    flushed += 1

        # Clear pending
        self._pending.clear()
        self._pending_count = 0
        self._last_flush_time = time.time()

        logger.debug(f"BatchingCheckpointer flushed {flushed} checkpoints")
        return flushed

    async def start_background_flush(self) -> None:
        """Start background flush task for interval-based flushing.

        This is optional - interval-based flushing also happens
        automatically on save() calls. Use this for long-running
        workflows where save() calls may be infrequent.
        """
        if self._flush_task is not None:
            return  # Already running

        if self._config.flush_interval is None:
            return  # Interval-based flushing disabled

        self._shutdown = False
        self._flush_task = asyncio.create_task(self._background_flush_loop())

    async def stop_background_flush(self) -> None:
        """Stop background flush task and flush remaining checkpoints."""
        self._shutdown = True

        if self._flush_task is not None:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Final flush
        await self.flush()

    async def _background_flush_loop(self) -> None:
        """Background loop for interval-based flushing."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self._config.flush_interval or 5.0)
                if self._pending_count > 0:
                    await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Background flush error: {e}")

    @property
    def pending_count(self) -> int:
        """Get number of pending (unflushed) checkpoints."""
        return self._pending_count

    @property
    def backend(self) -> CheckpointerProtocol:
        """Get the underlying backend checkpointer."""
        return self._backend

    def get_stats(self) -> dict[str, Any]:
        """Get batching statistics.

        Returns:
            Dictionary with stats including pending count, threads, etc.
        """
        return {
            "pending_count": self._pending_count,
            "pending_threads": len(self._pending),
            "batch_size": self._config.batch_size,
            "flush_interval": self._config.flush_interval,
            "keep_latest_only": self._config.keep_latest_only,
            "last_flush_time": self._last_flush_time,
            "background_flush_active": self._flush_task is not None,
        }

    async def __aenter__(self) -> "BatchingCheckpointer":
        """Async context manager entry."""
        await self.start_background_flush()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - ensures flush on completion."""
        await self.stop_background_flush()


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
    node_history: list[str] = field(default_factory=list)


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
        self.visited_count: dict[str, int] = {}

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

    def reset(self) -> None:
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

    def start(self) -> None:
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

    def __init__(self, interrupt_before: list[str], interrupt_after: list[str]):
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

    def __init__(self, nodes: dict[str, Node], use_copy_on_write: bool):
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
                    result = await asyncio.wait_for(node.execute(cow_state), timeout=remaining)
                else:
                    result = await node.execute(cow_state)

                # Extract final state from COW wrapper or result
                if isinstance(result, CopyOnWriteState):
                    state = result.get_state()
                elif isinstance(result, dict):
                    state = result  # type: ignore[assignment]
                else:
                    # Node returned something else, use COW state
                    state = cow_state.get_state()
            else:
                # Traditional deep copy approach
                if remaining is not None:
                    state = await asyncio.wait_for(node.execute(state), timeout=remaining)
                else:
                    state = await node.execute(state)

            return True, None, state

        except asyncio.TimeoutError:
            return False, "Execution timeout", state
        except Exception as e:
            return False, str(e), state


@dataclass
class ParallelBranchResult:
    """Result from executing a single parallel branch.

    Attributes:
        branch_id: ID of the branch (target node ID)
        success: Whether the branch executed successfully
        state: Final state from the branch
        error: Error message if failed
        node_history: List of nodes executed in this branch
    """

    branch_id: str
    success: bool
    state: Any
    error: Optional[str] = None
    node_history: list[str] = field(default_factory=list)


class ParallelBranchExecutor:
    """Executes parallel branches of a graph concurrently (SRP: Single Responsibility).

    Handles forking state into parallel branches, executing them concurrently,
    and merging results at join nodes.

    Thread-safety is guaranteed by:
    - Each branch gets its own CopyOnWriteState wrapper
    - CopyOnWriteState is now thread-safe (uses RLock)
    - Results are merged atomically after all branches complete
    """

    def __init__(
        self,
        nodes: dict[str, Node],
        edges: dict[str, list[Edge]],
        use_copy_on_write: bool,
    ):
        """Initialize parallel branch executor.

        Args:
            nodes: Dictionary of all nodes
            edges: Dictionary of all edges
            use_copy_on_write: Whether to use copy-on-write optimization
        """
        self.nodes = nodes
        self.edges = edges
        self.use_copy_on_write = use_copy_on_write

    async def execute_parallel_branches(
        self,
        branch_targets: list[str],
        state: Any,
        join_node: Optional[str],
        merge_func: Optional[Callable[[list[Any]], Any]],
        timeout_manager: TimeoutManager,
        max_iterations: int = 100,
    ) -> tuple[bool, Optional[str], Any, list[str]]:
        """Execute multiple branches in parallel.

        Each branch runs independently with its own copy of state.
        Results are merged using the merge function at the join node.

        Args:
            branch_targets: List of node IDs to execute in parallel
            state: Current state to fork
            join_node: Node ID where branches converge (or None)
            merge_func: Function to merge branch states
            timeout_manager: Timeout manager for execution limits
            max_iterations: Max iterations per branch

        Returns:
            Tuple of (success, error_message, merged_state, combined_node_history)
        """
        if not branch_targets:
            return True, None, state, []

        # Create tasks for parallel execution
        tasks = [
            self._execute_branch(
                start_node=target,
                state=copy.deepcopy(state),  # Each branch gets its own copy
                join_node=join_node,
                timeout_manager=timeout_manager,
                max_iterations=max_iterations,
            )
            for target in branch_targets
        ]

        # Execute all branches concurrently
        results: list[ParallelBranchResult] = await asyncio.gather(*tasks)

        # Check for failures
        failed_branches = [r for r in results if not r.success]
        if failed_branches:
            errors = "; ".join(f"{r.branch_id}: {r.error}" for r in failed_branches)
            return False, f"Parallel branch failures: {errors}", state, []

        # Merge states from all branches
        branch_states = [r.state for r in results]
        if merge_func:
            try:
                merged_state = merge_func(branch_states)
            except Exception as e:
                return False, f"State merge failed: {e}", state, []
        else:
            # Default merge: deep merge dictionaries
            merged_state = self._default_merge(branch_states)

        # Combine node histories
        combined_history: list[str] = []
        for result in results:
            combined_history.extend(result.node_history)

        return True, None, merged_state, combined_history

    async def _execute_branch(
        self,
        start_node: str,
        state: Any,
        join_node: Optional[str],
        timeout_manager: TimeoutManager,
        max_iterations: int,
    ) -> ParallelBranchResult:
        """Execute a single branch until it reaches the join node or END.

        Args:
            start_node: Starting node for this branch
            state: Branch's copy of state
            join_node: Node where branch should stop (join point)
            timeout_manager: Timeout manager
            max_iterations: Maximum iterations for this branch

        Returns:
            ParallelBranchResult with execution outcome
        """
        current_node = start_node
        node_history: list[str] = []
        iterations = 0

        node_executor = NodeExecutor(
            nodes=self.nodes,
            use_copy_on_write=self.use_copy_on_write,
        )

        try:
            while current_node != END and iterations < max_iterations:
                # Stop at join node (don't execute it - main loop will)
                if join_node and current_node == join_node:
                    break

                # Check timeout
                if timeout_manager.is_expired():
                    return ParallelBranchResult(
                        branch_id=start_node,
                        success=False,
                        state=state,
                        error="Branch timeout",
                        node_history=node_history,
                    )

                # Execute node
                success, error, state = await node_executor.execute(
                    node_id=current_node,
                    state=state,
                    timeout_manager=timeout_manager,
                )

                if not success:
                    return ParallelBranchResult(
                        branch_id=start_node,
                        success=False,
                        state=state,
                        error=error,
                        node_history=node_history,
                    )

                node_history.append(current_node)
                iterations += 1

                # Get next node
                current_node = self._get_next_node(current_node, state)

            return ParallelBranchResult(
                branch_id=start_node,
                success=True,
                state=state,
                node_history=node_history,
            )

        except Exception as e:
            return ParallelBranchResult(
                branch_id=start_node,
                success=False,
                state=state,
                error=str(e),
                node_history=node_history,
            )

    def _get_next_node(self, current_node: str, state: Any) -> str:
        """Determine next node for a branch.

        Args:
            current_node: Current node ID
            state: Current state

        Returns:
            Next node ID or END
        """
        edges = self.edges.get(current_node, [])
        if not edges:
            return END

        for edge in edges:
            # Skip parallel edges in branch execution
            if edge.is_parallel():
                continue
            target = edge.get_target(state)
            if target:
                return target

        return END

    def _default_merge(self, states: list[Any]) -> Any:
        """Default state merge strategy: deep merge dictionaries.

        Later states override earlier ones for conflicting keys.
        Lists are concatenated, dicts are recursively merged.

        Args:
            states: List of states to merge

        Returns:
            Merged state
        """
        if not states:
            return {}

        if len(states) == 1:
            return states[0]

        result = copy.deepcopy(states[0])

        for state in states[1:]:
            if isinstance(state, dict) and isinstance(result, dict):
                result = self._deep_merge_dicts(result, state)
            else:
                # Non-dict states: last one wins
                result = state

        return result

    def _deep_merge_dicts(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge two dictionaries.

        Args:
            base: Base dictionary
            override: Dictionary to merge on top

        Returns:
            Merged dictionary
        """
        result = copy.deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                # Concatenate lists
                result[key] = result[key] + value
            else:
                result[key] = copy.deepcopy(value)

        return result


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
                return checkpoint.state.copy(), checkpoint.node_id  # type: ignore[return-value]

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

    async def emit_graph_started(self, entry_point: str, node_count: int, thread_id: str) -> None:
        """Emit graph started event."""
        if not self.emit_events:
            return

        try:
            from victor.core.events import get_observability_bus as get_event_bus

            bus = get_event_bus()
            await bus.emit(
                topic="lifecycle.graph_started",
                data={
                    "graph_id": self.graph_id,
                    "source": "StateGraph",
                    "entry_point": entry_point,
                    "node_count": node_count,
                    "thread_id": thread_id,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit graph_started event: {e}")

    async def emit_node_start(self, node_id: str, iteration: int) -> None:
        """Emit node start event."""
        if not self.emit_events:
            return

        try:
            from victor.core.events import get_observability_bus as get_event_bus

            bus = get_event_bus()
            await bus.emit(
                topic="lifecycle.node_start",
                data={
                    "graph_id": self.graph_id,
                    "source": "StateGraph",
                    "node_id": node_id,
                    "iteration": iteration,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit node_start event: {e}")

    async def emit_node_complete(self, node_id: str, iteration: int, duration: float) -> None:
        """Emit node complete event."""
        if not self.emit_events:
            return

        try:
            from victor.core.events import get_observability_bus as get_event_bus

            bus = get_event_bus()
            await bus.emit(
                topic="lifecycle.node_end",
                data={
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

    async def emit_graph_completed(
        self, success: bool, iterations: int, duration: float, node_count: int
    ) -> None:
        """Emit graph completed event."""
        if not self.emit_events:
            return

        try:
            from victor.core.events import get_observability_bus as get_event_bus

            bus = get_event_bus()
            await bus.emit(
                topic="lifecycle.graph_completed",
                data={
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

    async def emit_graph_error(self, error: str, iterations: int, duration: float) -> None:
        """Emit graph error event."""
        if not self.emit_events:
            return

        try:
            from victor.core.events import get_observability_bus as get_event_bus

            bus = get_event_bus()
            await bus.emit(
                topic="lifecycle.graph_error",
                data={
                    "graph_id": self.graph_id,
                    "source": "StateGraph",
                    "error": error,
                    "iterations": iterations,
                    "duration": duration,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit graph_error event: {e}")

    async def emit_parallel_start(
        self, source_node: str, branch_count: int, branch_targets: list[str]
    ) -> None:
        """Emit parallel execution start event."""
        if not self.emit_events:
            return

        try:
            from victor.core.events import get_observability_bus as get_event_bus

            bus = get_event_bus()
            await bus.emit(
                topic="lifecycle.parallel_start",
                data={
                    "graph_id": self.graph_id,
                    "source": "StateGraph",
                    "source_node": source_node,
                    "branch_count": branch_count,
                    "branch_targets": branch_targets,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit parallel_start event: {e}")

    async def emit_parallel_complete(
        self, source_node: str, branch_count: int, success: bool, duration: float
    ) -> None:
        """Emit parallel execution complete event."""
        if not self.emit_events:
            return

        try:
            from victor.core.events import get_observability_bus as get_event_bus

            bus = get_event_bus()
            await bus.emit(
                topic="lifecycle.parallel_complete",
                data={
                    "graph_id": self.graph_id,
                    "source": "StateGraph",
                    "source_node": source_node,
                    "branch_count": branch_count,
                    "success": success,
                    "duration": duration,
                },
            )
        except Exception as e:
            logger.debug(f"Failed to emit parallel_complete event: {e}")


class CompiledGraph(Generic[StateType]):
    """Compiled graph ready for execution.

    The compilation step validates the graph structure and
    creates an optimized execution plan.
    """

    def __init__(
        self,
        nodes: dict[str, Node],
        edges: dict[str, list[Edge]],
        entry_point: str,
        state_schema: Optional[type[StateType]] = None,
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

    async def _emit_event(
        self,
        event_type: str,
        graph_id: str,
        data: dict[str, Any],
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
            await bus.emit(
                topic=f"lifecycle.{event_type}",
                data={
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
        checkpoint_manager = GraphCheckpointManager(
            checkpointer=exec_config.checkpoint.checkpointer
        )
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
        await event_emitter.emit_graph_started(
            entry_point=self._entry_point,
            node_count=len(self._nodes),
            thread_id=thread_id,
        )

        node_history: list[str] = []

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
                await event_emitter.emit_node_start(
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
                    await hook.after_node(current_node, state, error if not success else None)

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
                await event_emitter.emit_node_complete(
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

                # Check for parallel edges and execute branches concurrently
                parallel_edge = self._get_parallel_edge(current_node)
                if parallel_edge:
                    parallel_executor = ParallelBranchExecutor(
                        nodes=self._nodes,
                        edges=self._edges,
                        use_copy_on_write=use_cow,
                    )

                    branch_targets = parallel_edge.get_parallel_targets()
                    logger.debug(
                        f"Executing parallel branches: {branch_targets} "
                        f"(join: {parallel_edge.join_node})"
                    )

                    # Emit parallel execution start event
                    await event_emitter.emit_parallel_start(
                        source_node=current_node,
                        branch_count=len(branch_targets),
                        branch_targets=branch_targets,
                    )

                    parallel_start_time = time.time()
                    (
                        parallel_success,
                        parallel_error,
                        state,
                        branch_history,
                    ) = await parallel_executor.execute_parallel_branches(
                        branch_targets=branch_targets,
                        state=state,
                        join_node=parallel_edge.join_node,
                        merge_func=parallel_edge.merge_func,
                        timeout_manager=timeout_manager,
                        max_iterations=exec_config.execution.max_iterations,
                    )

                    # Emit parallel execution complete event
                    await event_emitter.emit_parallel_complete(
                        source_node=current_node,
                        branch_count=len(branch_targets),
                        success=parallel_success,
                        duration=time.time() - parallel_start_time,
                    )

                    if not parallel_success:
                        return GraphExecutionResult(
                            state=state,
                            success=False,
                            error=parallel_error,
                            iterations=iteration_controller.iterations,
                            duration=timeout_manager.get_elapsed(),
                            node_history=node_history,
                        )

                    # Add branch histories to main history
                    node_history.extend(branch_history)

                    # Continue from join node or END
                    current_node = parallel_edge.join_node or END
                else:
                    # Get next node (normal or conditional edge)
                    current_node = self._get_next_node(current_node, state)

            # Emit RL event for successful completion
            self._emit_graph_completed_event(
                success=True,
                iterations=iteration_controller.iterations,
                duration=timeout_manager.get_elapsed(),
            )

            # Emit graph completed event
            await event_emitter.emit_graph_completed(
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
            await event_emitter.emit_graph_error(
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
            await event_emitter.emit_graph_error(
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

    def _get_parallel_edge(self, current_node: str) -> Optional[Edge]:
        """Get parallel edge from current node if one exists.

        Args:
            current_node: Current node ID

        Returns:
            Parallel edge if found, None otherwise
        """
        edges = self._edges.get(current_node, [])
        for edge in edges:
            if edge.is_parallel():
                return edge
        return None

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
                return  # type: ignore[unreachable]

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
    ) -> AsyncIterator[tuple[str, StateType]]:
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
        visited_count: dict[str, int] = {}

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

    def get_graph_schema(self) -> dict[str, Any]:
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
        state_schema: Optional[type[StateType]] = None,
        config_schema: Optional[type[Any]] = None,
    ):
        """Initialize StateGraph.

        Args:
            state_schema: Optional type for state validation
            config_schema: Optional type for config validation
        """
        self._state_schema = state_schema
        self._config_schema = config_schema
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, list[Edge]] = {}
        self._entry_point: Optional[str] = None

    def add_node(
        self,
        node_id: str,
        func: Callable[[StateType], StateType | Awaitable[StateType]],
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
        branches: dict[str, str],
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

    def add_parallel_edges(
        self,
        source: str,
        targets: list[str],
        join_node: Optional[str] = None,
        merge_func: Optional[Callable[[list[StateType]], StateType]] = None,
    ) -> "StateGraph[StateType]":
        """Add parallel edges that fork execution into concurrent branches.

        Creates a fork point where execution splits into multiple parallel branches.
        Each branch executes independently with its own copy of state.
        Branches converge at the optional join node where states are merged.

        Args:
            source: Source node ID (fork point)
            targets: List of target node IDs to execute in parallel
            join_node: Optional node ID where branches converge.
                      If specified, all branches stop at this node and
                      the main execution continues from here after merging.
            merge_func: Optional function to merge branch states.
                       Signature: (List[StateType]) -> StateType
                       Default: deep merge dictionaries, concatenate lists

        Returns:
            Self for chaining

        Example:
            # Fork into three parallel branches, join at "aggregate"
            graph.add_parallel_edges(
                "start",
                ["branch_a", "branch_b", "branch_c"],
                join_node="aggregate",
                merge_func=lambda states: {"results": [s["result"] for s in states]}
            )

            # Or without explicit join (branches run to END independently)
            graph.add_parallel_edges("fork", ["task1", "task2"])
        """
        if source not in self._edges:
            self._edges[source] = []

        if len(targets) < 2:
            raise ValueError("Parallel edges require at least 2 targets")

        edge = Edge(
            source=source,
            target=targets,
            edge_type=EdgeType.PARALLEL,
            merge_func=merge_func,
            join_node=join_node,
        )
        self._edges[source].append(edge)
        logger.debug(f"Added parallel edges: {source} -> {targets} (join: {join_node})")
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
            entry_point=self._entry_point or "",
            state_schema=self._state_schema,
            config=config,
        )

    def _validate(self) -> list[str]:
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

    def _find_reachable(self) -> set[str]:
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
                elif isinstance(edge.target, list):
                    # Parallel edges: all targets are reachable
                    to_visit.extend(edge.target)
                    # Also add join node if specified
                    if edge.join_node:
                        to_visit.append(edge.join_node)

        return reachable

    @classmethod
    def from_schema(
        cls,
        schema: dict[str, Any] | str,
        state_schema: Optional[type[StateType]] = None,
        node_registry: Optional[dict[str, Callable[..., Any]]] = None,
        condition_registry: Optional[dict[str, Callable[..., Any]]] = None,
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
                def passthrough_func(state: StateType) -> StateType:
                    return state

                metadata = {k: v for k, v in node_def.items() if k not in ["id", "type"]}
                graph.add_node(node_id, passthrough_func, **metadata)

            elif node_type == "agent":
                # Agent node - placeholder for workflow execution
                # The actual agent execution is handled by the workflow executor
                def create_agent_placeholder(
                    node_config: dict[str, Any],
                ) -> Callable[[StateType], StateType]:
                    def agent_placeholder(state: StateType) -> StateType:
                        # Store node config in state for executor to use
                        return {  # type: ignore[return-value]
                            **state,
                            "_pending_agent": node_config,
                        }

                    return agent_placeholder

                metadata = {k: v for k, v in node_def.items() if k not in ["id", "type"]}
                graph.add_node(node_id, create_agent_placeholder(node_def), **metadata)

            elif node_type == "compute":
                # Compute node - placeholder for handler execution
                # The actual compute execution is handled by the workflow executor
                def create_compute_placeholder(
                    node_config: dict[str, Any],
                ) -> Callable[[StateType], StateType]:
                    def compute_placeholder(state: StateType) -> StateType:
                        # Store node config in state for executor to use
                        return {  # type: ignore[return-value]
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
                    raise ValueError(f"Conditional edge from '{source}' must specify 'condition'")

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
    state_schema: Optional[type[StateType]] = None,
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
