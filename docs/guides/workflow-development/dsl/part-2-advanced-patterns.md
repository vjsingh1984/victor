# Workflow DSL Guide - Part 2

**Part 2 of 2:** Advanced Patterns and API Reference

---

## Navigation

- [Part 1: Fundamentals](part-1-fundamentals.md)
- **[Part 2: Advanced Patterns](#)** (Current)
- [**Complete Guide](../dsl.md)**

---
## Advanced Patterns

### Parallel Execution

```python
import asyncio

async def parallel_analysis(state: WorkflowState) -> WorkflowState:
    """Run multiple analyses in parallel."""
    tasks = [
        analyze_security(state),
        analyze_performance(state),
        analyze_style(state),
    ]
    results = await asyncio.gather(*tasks)
    state["analyses"] = {
        "security": results[0],
        "performance": results[1],
        "style": results[2],
    }
    return state
```text

### Dynamic Node Selection

```python
def select_processor(state: WorkflowState) -> str:
    """Dynamically select processor based on file type."""
    ext = state["file"].split(".")[-1]
    processors = {
        "py": "python_processor",
        "js": "javascript_processor",
        "ts": "typescript_processor",
        "go": "go_processor",
    }
    return processors.get(ext, "generic_processor")

# Add all possible processors
for proc in ["python_processor", "javascript_processor", ...]:
    graph.add_node(proc, processor_functions[proc])

graph.add_conditional_edge("classify", select_processor, {
    "python_processor": "python_processor",
    "javascript_processor": "javascript_processor",
    # ... etc
})
```

### Error Handling

```python
class WorkflowState(TypedDict):
    # ... other fields
    error: Optional[str]
    error_count: int

async def safe_execute(state: WorkflowState) -> WorkflowState:
    """Node with error handling."""
    try:
        result = await risky_operation(state)
        state["result"] = result
        state["error"] = None
    except Exception as e:
        state["error"] = str(e)
        state["error_count"] += 1
    return state

def handle_error(state: WorkflowState) -> str:
    if state["error"] is None:
        return "success"
    elif state["error_count"] < 3:
        return "retry"
    else:
        return "fail"

graph.add_conditional_edge("execute", handle_error, {
    "success": "finalize",
    "retry": "execute",
    "fail": "error_handler",
})
```text

### Timeout Configuration

```python
app = graph.compile(
    max_iterations=25,  # Maximum node executions (prevents infinite loops)
    timeout=60.0,       # Overall workflow timeout in seconds
)

# Or per-invocation
result = await app.invoke(state, timeout=30.0)
```

## API Reference

### StateGraph

```python
class StateGraph(Generic[T]):
    def __init__(self, state_schema: Type[T]) -> None: ...

    def add_node(self, node_id: str, func: Callable[[T], T]) -> "StateGraph[T]": ...

    def add_edge(self, source: str, target: str) -> "StateGraph[T]": ...

    def add_conditional_edge(
        self,
        source: str,
        condition: Callable[[T], str],
        branches: Dict[str, str],
    ) -> "StateGraph[T]": ...

    def set_entry_point(self, node_id: str) -> "StateGraph[T]": ...

    def compile(
        self,
        checkpointer: Optional[BaseCheckpointer] = None,
        max_iterations: int = 25,
        timeout: Optional[float] = None,
    ) -> "CompiledGraph[T]": ...
```text

### CompiledGraph

```python
class CompiledGraph(Generic[T]):
    async def invoke(
        self,
        state: T,
        timeout: Optional[float] = None,
    ) -> ExecutionResult[T]: ...

    async def stream(
        self,
        state: T,
    ) -> AsyncIterator[StreamEvent]: ...
```

### ExecutionResult

```python
@dataclass
class ExecutionResult(Generic[T]):
    success: bool
    state: T
    node_history: List[str]
    iterations: int
    error: Optional[str]
```text

### END Constant

```python
from victor.framework.graph import END

# Special constant indicating workflow termination
graph.add_edge("final_node", END)
```

---

**Related Documentation**:
- [User Guide: Workflows](../../user-guide/index.md#4-workflows)
- [Developer Guide](../../contributing/index.md)
- [Tool Catalog](../../reference/tools/catalog.md)

*Last Updated: 2025-12-29*

---

**Last Updated:** February 01, 2026
**Reading Time:** 2 min
