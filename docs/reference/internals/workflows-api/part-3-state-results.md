# Workflows API Reference - Part 3

**Part 3 of 4:** State Management and Execution Results

---

## Navigation

- [Part 1: StateGraph, Compiler, Provider](part-1-stategraph-compiler-provider.md)
- [Part 2: Node Types](part-2-node-types.md)
- **[Part 3: State & Results](#)** (Current)
- [Part 4: Configuration & Examples](part-4-configuration-examples.md)
- [**Complete Reference**](../workflows-api.md)

---

### TypedDict State Pattern

Define workflow state using TypedDict for type safety:

```python
from typing import TypedDict, Optional, List, Any

class WorkflowState(TypedDict):
    # Input fields
    query: str

    # Intermediate results
    search_results: Optional[List[str]]
    analysis: Optional[str]

    # Output fields
    final_report: Optional[str]

    # Metadata (prefixed with _)
    _workflow_id: str
    _node_results: Dict[str, Any]
    _parallel_results: Dict[str, Any]
```

### Context Variables

Reference state values in YAML using `$ctx.key` syntax:

```yaml
nodes:
  - id: analyze
    type: agent
    goal: "Analyze the query: $ctx.query"
    input_mapping:
      data: $ctx.search_results
```

### State Transitions

State flows through nodes and is updated at each step:

```python
# Initial state
state = {
    "query": "AI trends 2024",
    "_workflow_id": "abc123"
}

# After search node
state = {
    "query": "AI trends 2024",
    "search_results": ["result1", "result2"],
    "_workflow_id": "abc123",
    "_node_results": {"search": {...}}
}

# After analysis node
state = {
    "query": "AI trends 2024",
    "search_results": ["result1", "result2"],
    "analysis": "Key trends include...",
    "_workflow_id": "abc123",
    "_node_results": {"search": {...}, "analyze": {...}}
}
```

### CopyOnWriteState

Performance optimization that delays deep copy until first mutation:

```python
from victor.framework.graph import CopyOnWriteState

# Wrap original state
cow_state = CopyOnWriteState(original_state)

# Reading doesn't copy
value = cow_state["key"]  # No copy overhead

# Writing triggers copy
cow_state["key"] = "new_value"  # Deep copy happens here

# Get final state
final_state = cow_state.get_state()
```

---

## Execution Results

### GraphExecutionResult

Result from graph execution.

```python
@dataclass
class GraphExecutionResult(Generic[StateType]):
    state: StateType           # Final state
    success: bool              # Whether execution succeeded
    error: Optional[str]       # Error message if failed
    iterations: int            # Number of iterations executed
    duration: float            # Total execution time in seconds
    node_history: List[str]    # Sequence of executed node IDs
```

**Example:**
```python
result = await app.invoke(initial_state)

if result.success:
    print(f"Completed in {result.duration:.2f}s")
    print(f"Nodes executed: {result.node_history}")
    print(f"Final output: {result.state.get('output')}")
else:
    print(f"Failed at iteration {result.iterations}: {result.error}")
```

### NodeExecutionResult

Result from executing a single workflow node.

```python
@dataclass
class NodeExecutionResult:
    node_id: str               # ID of the executed node
    success: bool              # Whether execution succeeded
    output: Any = None         # Output data from the node
    error: Optional[str] = None  # Error message if failed
    duration_seconds: float = 0.0  # Execution time
    tool_calls_used: int = 0   # Number of tool calls made
```

### Streaming Events

When streaming execution, events are yielded as tuples:

```python
async for node_id, state in app.stream(initial_state):
    # node_id: str - ID of completed node
    # state: Dict[str, Any] - Current state after node execution

    # Access node results
    node_result = state.get("_node_results", {}).get(node_id)
    if node_result:
        print(f"Node {node_id}: success={node_result['success']}")
```

### Error Handling

Errors are captured in the result and optionally in state:

```python
result = await compiled.invoke(initial_state)

if not result.success:
    # Error in result
    print(f"Error: {result.error}")

    # Error may also be in state
    if "_error" in result.state:
        print(f"State error: {result.state['_error']}")

    # Check specific node failures
    node_results = result.state.get("_node_results", {})
    for node_id, node_result in node_results.items():
        if not node_result.get("success", True):
            print(f"Node {node_id} failed: {node_result.get('error')}")
```

---

## Configuration

### GraphConfig

Facade configuration for graph execution (ISP compliant).

```python
from victor.framework.config import (
    GraphConfig,
    ExecutionConfig,
    CheckpointConfig,
    InterruptConfig,
    PerformanceConfig,
    ObservabilityConfig,
)

config = GraphConfig(
    execution=ExecutionConfig(
        max_iterations=50,
        timeout=300.0,
        recursion_limit=100
    ),
    checkpoint=CheckpointConfig(
        checkpointer=MemoryCheckpointer(),
        checkpoint_at_end=True
    ),
    interrupt=InterruptConfig(
        interrupt_before=["critical_operation"],
        interrupt_after=["user_approval"]
    ),
    performance=PerformanceConfig(
        use_copy_on_write=True
    ),
    observability=ObservabilityConfig(
        emit_events=True,
        graph_id="my-workflow-123"
    )
)
```

### ExecutionConfig

```python
@dataclass
class ExecutionConfig:
    max_iterations: int = 25        # Maximum cycles allowed
    timeout: Optional[float] = None # Overall timeout in seconds
    recursion_limit: int = 100      # Maximum recursion depth
```

### CheckpointConfig

```python
@dataclass
class CheckpointConfig:
    checkpointer: Optional[CheckpointerProtocol] = None
    checkpoint_at_start: bool = False
    checkpoint_at_end: bool = True
```

### InterruptConfig

```python
@dataclass
class InterruptConfig:
    interrupt_before: List[str] = field(default_factory=list)
    interrupt_after: List[str] = field(default_factory=list)
```

### PerformanceConfig

```python
@dataclass
class PerformanceConfig:
    use_copy_on_write: Optional[bool] = None  # None = use settings default
    enable_state_caching: bool = True
```

### ObservabilityConfig

```python
@dataclass
class ObservabilityConfig:
    emit_events: bool = True
    log_node_execution: bool = False
    graph_id: Optional[str] = None
```

### UnifiedCompilerConfig

```python
@dataclass
class UnifiedCompilerConfig:
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
```

### Checkpointing with thread_id

Use `thread_id` for checkpoint persistence and resumption:

```python
from victor.framework.graph import MemoryCheckpointer

checkpointer = MemoryCheckpointer()
app = graph.compile(checkpointer=checkpointer)

# First execution with thread_id
thread_id = "session-123"
result = await app.invoke(
    initial_state,
    thread_id=thread_id
)

# Resume from checkpoint
resumed_result = await app.invoke(
    {},  # State loaded from checkpoint
    thread_id=thread_id  # Same thread_id to resume
)
```


**Reading Time:** 3 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


## Complete Example
