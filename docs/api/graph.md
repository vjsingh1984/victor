# StateGraph API Reference

Stateful workflow engine for building cyclic, stateful agent workflows with typed state management. Inspired by LangGraph with Victor-specific enhancements.

## Overview

The StateGraph API provides:
- **Typed state management** with TypedDict support
- **Cyclic graphs** with configurable iteration limits
- **Conditional edges** for dynamic routing
- **Checkpointing** for state persistence and recovery
- **Copy-on-write optimization** for performance
- **Human-in-the-loop** interrupts for interactive workflows

## Quick Example

```python
from victor.framework.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    messages: list[str]
    task: str
    result: str | None

# Define node functions
def analyze_task(state: AgentState) -> AgentState:
    return {"messages": state["messages"] + ["Analyzing..."]}

def execute_task(state: AgentState) -> AgentState:
    return {"result": "Task complete"}

def should_retry(state: AgentState) -> str:
    return "retry" if state["result"] is None else "done"

# Build graph
graph = StateGraph(AgentState)
graph.add_node("analyze", analyze_task)
graph.add_node("execute", execute_task)
graph.add_edge("analyze", "execute")
graph.add_conditional_edge("execute", should_retry, {"retry": "analyze", "done": END})
graph.set_entry_point("analyze")

# Compile and run
app = graph.compile()
result = await app.invoke({"messages": [], "task": "Fix bug", "result": None})
print(result.state)
```

## StateGraph Class

### Constructor

```python
StateGraph(
    state_schema: type[StateType] | None = None,
    config_schema: type | None = None,
) -> StateGraph[StateType]
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|--------|-------------|
| `state_schema` | `type[StateType] \| None` | `None` | Optional TypedDict type for state validation |
| `config_schema` | `type \| None` | `None` | Optional type for config validation |

**Returns**: `StateGraph[StateType]` instance

**Examples**:

```python
from typing import TypedDict

class MyState(TypedDict):
    count: int
    total: int

# With typed state
graph = StateGraph(MyState)

# Without typed state
graph = StateGraph()
```

### Methods

#### add_node()

```python
def add_node(
    self,
    node_id: str,
    func: Callable[[StateType], StateType | Awaitable[StateType]],
    **metadata: Any,
) -> StateGraph[StateType]:
    """Add a node to the graph.

    Args:
        node_id: Unique node identifier
        func: Node execution function (sync or async)
        **metadata: Additional metadata for the node

    Returns:
        Self for chaining

    Raises:
        ValueError: If node already exists
    """
```

**Examples**:

```python
# Sync node
def my_node(state: MyState) -> MyState:
    return {"count": state["count"] + 1}

graph.add_node("increment", my_node)

# Async node
async def async_node(state: MyState) -> MyState:
    await asyncio.sleep(1)
    return {"total": state["total"] + state["count"]}

graph.add_node("compute", async_node)

# With metadata
graph.add_node("process", process_func, category="compute", timeout=30)
```

#### add_edge()

```python
def add_edge(
    self,
    source: str,
    target: str,
) -> StateGraph[StateType]:
    """Add a normal edge between nodes.

    Args:
        source: Source node ID
        target: Target node ID (or END sentinel)

    Returns:
        Self for chaining
    """
```

**Examples**:

```python
from victor.framework.graph import END

# Edge between nodes
graph.add_edge("start", "process")

# Edge to end
graph.add_edge("process", END)

# Chaining
graph.add_edge("start", "process").add_edge("process", "end")
```

#### add_conditional_edge()

```python
def add_conditional_edge(
    self,
    source: str,
    condition: Callable[[StateType], str],
    branches: dict[str, str],
) -> StateGraph[StateType]:
    """Add a conditional edge with multiple branches.

    Args:
        source: Source node ID
        condition: Function that receives state and returns branch name
        branches: Mapping from branch names to target node IDs

    Returns:
        Self for chaining
    """
```

**Examples**:

```python
def route_by_status(state: MyState) -> str:
    if state["count"] > 10:
        return "high"
    elif state["count"] > 5:
        return "medium"
    return "low"

graph.add_conditional_edge(
    "check",
    route_by_status,
    {"high": "scale_up", "medium": "continue", "low": "scale_down"}
)

# With END
graph.add_conditional_edge(
    "validate",
    lambda s: "done" if s["valid"] else "retry",
    {"done": END, "retry": "fix"}
)
```

#### set_entry_point()

```python
def set_entry_point(
    self,
    node_id: str,
) -> StateGraph[StateType]:
    """Set the entry point node.

    Args:
        node_id: Node to start execution from

    Returns:
        Self for chaining

    Raises:
        ValueError: If node not found
    """
```

**Examples**:

```python
graph.set_entry_point("start")
```

#### set_finish_point()

```python
def set_finish_point(
    self,
    node_id: str,
) -> StateGraph[StateType]:
    """Set a node as finish point (adds edge to END).

    Args:
        node_id: Node that finishes the graph

    Returns:
        Self for chaining
    """
```

**Examples**:

```python
graph.set_finish_point("cleanup")  # Equivalent to graph.add_edge("cleanup", END)
```

#### compile()

```python
def compile(
    self,
    checkpointer: CheckpointerProtocol | None = None,
    **config_kwargs: Any,
) -> CompiledGraph[StateType]:
    """Compile the graph for execution.

    Validates graph structure and creates optimized execution plan.

    Args:
        checkpointer: Optional checkpointer for state persistence
        **config_kwargs: Additional configuration options

    Returns:
        CompiledGraph ready for execution

    Raises:
        ValueError: If graph is invalid
    """
```

**Configuration Options**:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_iterations` | `int` | `100` | Maximum total iterations |
| `recursion_limit` | `int` | `10` | Maximum visits to same node |
| `timeout` | `float \| None` | `None` | Overall execution timeout (seconds) |
| `interrupt_before` | `list[str]` | `[]` | Nodes to interrupt before execution |
| `interrupt_after` | `list[str]` | `[]` | Nodes to interrupt after execution |
| `use_copy_on_write` | `bool \| None` | `None` | Enable COW optimization |
| `emit_events` | `bool` | `True` | Enable observability events |

**Examples**:

```python
# Basic compilation
app = graph.compile()

# With checkpointer
from victor.framework.graph import MemoryCheckpointer

checkpointer = MemoryCheckpointer()
app = graph.compile(checkpointer=checkpointer)

# With config
app = graph.compile(
    max_iterations=50,
    timeout=300,
    interrupt_before=["human_review"]
)
```

#### from_schema()

```python
@classmethod
def from_schema(
    cls,
    schema: dict[str, Any] | str,
    state_schema: type[StateType] | None = None,
    node_registry: dict[str, Callable] | None = None,
    condition_registry: dict[str, Callable] | None = None,
) -> StateGraph[StateType]:
    """Create StateGraph from schema dictionary or YAML string.

    Args:
        schema: Dictionary schema or YAML string
        state_schema: Optional TypedDict type for state validation
        node_registry: Registry of node functions
        condition_registry: Registry of condition functions

    Returns:
        StateGraph instance ready for compilation

    Raises:
        ValueError: If schema is invalid
        TypeError: If node/condition types are unsupported
    """
```

**Schema Format**:

```yaml
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
```

**Examples**:

```python
# From YAML
yaml_schema = """
nodes:
  - id: process
    type: function
    func: process_task
edges:
  - source: process
    target: __end__
    type: normal
entry_point: process
"""

graph = StateGraph.from_schema(
    yaml_schema,
    state_schema=MyState,
    node_registry={"process_task": process_task_func}
)

# From dictionary
schema_dict = {
    "nodes": [{"id": "start", "type": "function", "func": "start_func"}],
    "edges": [{"source": "start", "target": "__end__", "type": "normal"}],
    "entry_point": "start"
}

graph = StateGraph.from_schema(
    schema_dict,
    node_registry={"start_func": start_func}
)
```

## CompiledGraph Class

The compiled graph is the executable form returned by `StateGraph.compile()`.

### Methods

#### invoke()

```python
async def invoke(
    self,
    input_state: StateType,
    *,
    config: GraphConfig | None = None,
    thread_id: str | None = None,
    debug_hook: Any | None = None,
) -> GraphExecutionResult[StateType]:
    """Execute the graph.

    Args:
        input_state: Initial state
        config: Override execution config
        thread_id: Thread ID for checkpointing
        debug_hook: Optional DebugHook for debugging

    Returns:
        GraphExecutionResult with final state and metadata
    """
```

**Examples**:

```python
app = graph.compile()

# Basic execution
result = await app.invoke({"count": 0, "total": 0})
print(result.state)      # Final state
print(result.success)    # True if completed
print(result.iterations) # Number of iterations
print(result.duration)   # Execution time

# With checkpointing (resume)
result = await app.invoke(
    {"count": 0, "total": 0},
    thread_id="workflow-123"  # Resumes from checkpoint if exists
)

# With config override
result = await app.invoke(
    input_state,
    config=GraphConfig(execution=ExecutionConfig(max_iterations=10))
)

# With debug hook
from victor.framework.debug import DebugHook

hook = DebugHook()
result = await app.invoke(input_state, debug_hook=hook)
```

#### stream()

```python
async def stream(
    self,
    input_state: StateType,
    *,
    config: GraphConfig | None = None,
    thread_id: str | None = None,
):
    """Stream execution yielding state after each node.

    Args:
        input_state: Initial state
        config: Override execution config
        thread_id: Thread ID for checkpointing

    Yields:
        Tuple of (node_id, state) after each node execution
    """
```

**Examples**:

```python
app = graph.compile()

async for node_id, state in app.stream({"count": 0}):
    print(f"After {node_id}: {state}")
```

#### get_graph_schema()

```python
def get_graph_schema(self) -> dict[str, Any]:
    """Get graph structure as dictionary.

    Returns:
        Dictionary describing nodes and edges
    """
```

**Examples**:

```python
app = graph.compile()
schema = app.get_graph_schema()
print(schema)
# {
#     "nodes": ["analyze", "execute", "review"],
#     "edges": {
#         "analyze": [{"target": "execute", "type": "normal"}],
#         "execute": [{"target": {"retry": "analyze", "done": "review"}, "type": "conditional"}]
#     },
#     "entry_point": "analyze"
# }
```

## GraphExecutionResult

Result object returned by `CompiledGraph.invoke()`.

```python
@dataclass
class GraphExecutionResult(Generic[StateType]):
    """Result from graph execution."""

    state: StateType              # Final state
    success: bool                 # Whether execution succeeded
    error: str | None = None      # Error message if failed
    iterations: int = 0           # Number of iterations executed
    duration: float = 0.0         # Total execution time (seconds)
    node_history: list[str] = field(default_factory=list)  # Executed nodes
```

**Examples**:

```python
result = await app.invoke(initial_state)

if result.success:
    print(f"Completed in {result.iterations} iterations")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Path: {' -> '.join(result.node_history)}")
    print(f"Final state: {result.state}")
else:
    print(f"Failed: {result.error}")
    print(f"State at failure: {result.state}")
```

## CopyOnWriteState

Copy-on-write wrapper for workflow state optimization.

```python
class CopyOnWriteState(Generic[StateType]):
    """Copy-on-write wrapper for workflow state.

    Delays deep copy until first mutation for performance.
    """

    def __init__(self, source: StateType):
        """Initialize with source state (not copied)."""

    def __getitem__(self, key: str) -> Any:
        """Get item without copying."""

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item, triggering copy on first mutation."""

    def get_state(self) -> StateType:
        """Get the final state (modified copy or original source)."""

    @property
    def was_modified(self) -> bool:
        """Check if state was modified (copy was made)."""
```

**Examples**:

```python
from victor.framework.graph import CopyOnWriteState

original_state = {"count": 0, "total": 100}
cow_state = CopyOnWriteState(original_state)

# Reading doesn't copy
value = cow_state["count"]  # No copy

# Writing triggers copy
cow_state["count"] = 1  # Deep copy happens here

# Get final state
final_state = cow_state.get_state()
print(final_state)      # {"count": 1, "total": 100}
print(original_state)   # {"count": 0, "total": 100} - unchanged

# Check if modified
print(cow_state.was_modified)  # True
```

**Performance Characteristics**:
- Read operations: O(1), no copy overhead
- First write: O(n) deep copy where n is state size
- Subsequent writes: O(1), no additional copy

## Checkpointing

StateGraph supports state persistence through checkpointing.

### CheckpointerProtocol

```python
class CheckpointerProtocol(Protocol):
    """Protocol for checkpoint persistence."""

    async def save(self, checkpoint: WorkflowCheckpoint) -> None:
        """Save a checkpoint."""

    async def load(self, thread_id: str) -> WorkflowCheckpoint | None:
        """Load latest checkpoint for thread."""

    async def list(self, thread_id: str) -> list[WorkflowCheckpoint]:
        """List all checkpoints for thread."""
```

### MemoryCheckpointer

In-memory checkpoint storage for development and testing.

```python
from victor.framework.graph import MemoryCheckpointer

checkpointer = MemoryCheckpointer()
app = graph.compile(checkpointer=checkpointer)

# First execution (saves checkpoints)
result1 = await app.invoke(state, thread_id="workflow-1")

# Resume from checkpoint
result2 = await app.invoke(state, thread_id="workflow-1")  # Resumes from last checkpoint

# List checkpoints
checkpoints = await checkpointer.list("workflow-1")
```

### RLCheckpointerAdapter

Adapter to use existing RL CheckpointStore for graph checkpointing.

```python
from victor.framework.graph import RLCheckpointerAdapter

checkpointer = RLCheckpointerAdapter(learner_name="my_workflow")
app = graph.compile(checkpointer=checkpointer)

# Uses RL checkpoint store for persistence
result = await app.invoke(state, thread_id="workflow-1")
```

### WorkflowCheckpoint

```python
@dataclass
class WorkflowCheckpoint:
    """WorkflowCheckpoint for workflow state persistence."""

    checkpoint_id: str       # Unique checkpoint identifier
    thread_id: str           # Thread/execution identifier
    node_id: str             # Current node being executed
    state: dict[str, Any]    # State at checkpoint
    timestamp: float         # When checkpoint was created
    metadata: dict[str, Any] # Additional metadata
```

## Human-in-the-Loop

StateGraph supports human-in-the-loop workflows with interrupts.

```python
# Configure interrupts
app = graph.compile(
    interrupt_before=["human_review"],
    interrupt_after=["critical_decision"]
)

# Execution will pause before human_review node
result = await app.invoke(state, thread_id="workflow-1")
# Result will have intermediate state

# Resume execution
result = await app.invoke(None, thread_id="workflow-1")
# Continues from human_review
```

## Configuration

### GraphConfig

```python
@dataclass
class GraphConfig:
    """Configuration for StateGraph execution."""

    execution: ExecutionConfig
    checkpoint: CheckpointConfig
    interrupt: InterruptConfig
    performance: PerformanceConfig
    observability: ObservabilityConfig
```

### ExecutionConfig

```python
@dataclass
class ExecutionConfig:
    """Execution limits configuration."""

    max_iterations: int = 100
    recursion_limit: int = 10
    timeout: float | None = None
```

### CheckpointConfig

```python
@dataclass
class CheckpointConfig:
    """State persistence configuration."""

    checkpointer: CheckpointerProtocol | None = None
```

### InterruptConfig

```python
@dataclass
class InterruptConfig:
    """Human-in-the-loop interrupt configuration."""

    interrupt_before: list[str] = field(default_factory=list)
    interrupt_after: list[str] = field(default_factory=list)
```

### PerformanceConfig

```python
@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""

    use_copy_on_write: bool | None = None  # None = use settings default
```

### ObservabilityConfig

```python
@dataclass
class ObservabilityConfig:
    """Observability and eventing configuration."""

    emit_events: bool = True
    graph_id: str | None = None
```

## Best Practices

### 1. Use Typed State

```python
# Good - Typed state with validation
class AgentState(TypedDict):
    messages: list[str]
    count: int
    result: str | None

graph = StateGraph(AgentState)

# Avoid - Untyped state
graph = StateGraph()  # No validation
```

### 2. Define Clear Entry Points

```python
# Good - Explicit entry point
graph = StateGraph(MyState)
graph.add_node("start", start_func)
graph.add_node("process", process_func)
graph.set_entry_point("start")

# Avoid - Implicit or missing entry point (will fail validation)
graph = StateGraph(MyState)
graph.add_node("process", process_func)
# No entry point set!
```

### 3. Handle Conditional Edge Defaults

```python
# Good - Always handle all cases
def route(state: MyState) -> str:
    if state.get("error"):
        return "error"
    return "success"

graph.add_conditional_edge(
    "process",
    route,
    {"error": "handle_error", "success": END}
)

# Avoid - Missing default case
def route(state: MyState) -> str:
    return "success" if state["ok"] else None  # What if None?
```

### 4. Use Iteration Limits

```python
# Good - Set appropriate limits
app = graph.compile(
    max_iterations=50,     # Prevent infinite loops
    recursion_limit=5      # Prevent recursion
)

# Avoid - No limits (could run forever)
app = graph.compile()  # Uses defaults, but may be too high
```

### 5. Leverage Checkpointing for Long Workflows

```python
# Good - Enable checkpointing for long workflows
checkpointer = RLCheckpointerAdapter(learner_name="long_workflow")
app = graph.compile(checkpointer=checkpointer)

# Can resume if interrupted
result = await app.invoke(state, thread_id="long-job-1")

# Avoid - No checkpointing for long workflows
app = graph.compile()  # Must restart from beginning on failure
```

### 6. Use Streaming for Progress Updates

```python
# Good - Stream for real-time updates
async for node_id, state in app.stream(initial_state):
    print(f"Completed: {node_id}")
    print(f"Current count: {state['count']}")

# Avoid - Only get final result
result = await app.invoke(initial_state)  # No progress updates
```

## Common Patterns

### Retry Pattern

```python
def should_retry(state: MyState) -> str:
    if state.get("error") and state["attempts"] < 3:
        return "retry"
    return "done"

graph = StateGraph(MyState)
graph.add_node("process", process_func)
graph.add_conditional_edge("process", should_retry, {"retry": "process", "done": END})
```

### Sequential Pipeline

```python
graph = StateGraph(MyState)
graph.add_node("step1", step1_func)
graph.add_node("step2", step2_func)
graph.add_node("step3", step3_func)
graph.add_edge("step1", "step2")
graph.add_edge("step2", "step3")
graph.add_edge("step3", END)
graph.set_entry_point("step1")
```

### Branching Workflow

```python
def route_by_type(state: MyState) -> str:
    return state["task_type"]  # "research", "code", "test"

graph = StateGraph(MyState)
graph.add_node("route", route_func)
graph.add_node("research", research_func)
graph.add_node("code", code_func)
graph.add_node("test", test_func)
graph.add_conditional_edge(
    "route",
    route_by_type,
    {"research": "research", "code": "code", "test": "test"}
)
```

## Error Handling

```python
result = await app.invoke(initial_state)

if not result.success:
    print(f"Error: {result.error}")
    print(f"Failed after {result.iterations} iterations")
    print(f"Nodes executed: {result.node_history}")
    # State at time of failure
    print(f"State: {result.state}")
```

## See Also

- [Agent API](agent.md) - High-level agent interface
- [Tools API](tools.md) - Tool registration and usage
- [Configuration API](config.md) - Settings and profiles
- [Core APIs](core.md) - Events, state, workflows
