# Workflows API Reference - Part 1

**Part 1 of 4:** StateGraph, Compiler, and Provider APIs

---

## Navigation

- **[Part 1: StateGraph, Compiler, Provider](#)** (Current)
- [Part 2: Node Types](part-2-node-types.md)
- [Part 3: State & Results](part-3-state-results.md)
- [Part 4: Configuration & Examples](part-4-configuration-examples.md)
- [**Complete Reference**](../workflows-api.md)

---
# Workflows API Reference

This document provides API reference documentation for Victor's workflow system, including the StateGraph DSL,
  UnifiedWorkflowCompiler, YAML workflow providers, and node types.

## Table of Contents

1. [StateGraph API](#stategraph-api)
2. [UnifiedWorkflowCompiler](#unifiedworkflowcompiler)
3. [BaseYAMLWorkflowProvider](#baseyamlworkflowprovider)
4. [Node Types API](#node-types-api)
5. [State Management](#state-management)
6. [Execution Results](#execution-results)
7. [Configuration](#configuration)

---

## StateGraph API

The `StateGraph` class provides a LangGraph-compatible API for building cyclic, stateful agent workflows with typed state management.

**Module:** `victor.framework.graph`

### StateGraph Class

```python
from victor.framework.graph import StateGraph, END
from typing import TypedDict, Optional

class AgentState(TypedDict):
    messages: list[str]
    task: str
    result: Optional[str]

graph = StateGraph(AgentState)
```text

#### Constructor

```python
StateGraph(
    state_schema: Optional[Type[StateType]] = None,
    config_schema: Optional[Type] = None,
)
```

**Parameters:**
- `state_schema`: Optional TypedDict type for state validation
- `config_schema`: Optional type for config validation

#### Methods

##### add_node()

Add a node to the graph.

```python
def add_node(
    self,
    node_id: str,
    func: Callable[[StateType], Union[StateType, Awaitable[StateType]]],
    **metadata: Any,
) -> "StateGraph[StateType]"
```text

**Parameters:**
- `node_id`: Unique node identifier
- `func`: Node execution function (sync or async)
- `**metadata`: Additional metadata

**Returns:** Self for method chaining

**Raises:** `ValueError` if node already exists

**Example:**
```python
async def analyze_task(state: AgentState) -> AgentState:
    state["analysis"] = "Completed analysis"
    return state

graph.add_node("analyze", analyze_task)
```

##### add_edge()

Add a normal edge between nodes.

```python
def add_edge(
    self,
    source: str,
    target: str,
) -> "StateGraph[StateType]"
```text

**Parameters:**
- `source`: Source node ID
- `target`: Target node ID (or `END` sentinel)

**Returns:** Self for method chaining

**Example:**
```python
graph.add_edge("analyze", "execute")
graph.add_edge("execute", END)  # Terminal edge
```

##### add_conditional_edge()

Add a conditional edge with multiple branches.

```python
def add_conditional_edge(
    self,
    source: str,
    condition: Callable[[StateType], str],
    branches: Dict[str, str],
) -> "StateGraph[StateType]"
```text

**Parameters:**
- `source`: Source node ID
- `condition`: Function that evaluates state and returns a branch name
- `branches`: Mapping from branch names to target node IDs

**Returns:** Self for method chaining

**Example:**
```python
def should_retry(state: AgentState) -> str:
    if state.get("error"):
        return "retry"
    return "done"

graph.add_conditional_edge(
    "execute",
    should_retry,
    {"retry": "analyze", "done": "report"}
)
```

##### set_entry_point()

Set the entry point node for the graph.

```python
def set_entry_point(self, node_id: str) -> "StateGraph[StateType]"
```text

**Parameters:**
- `node_id`: Node to start execution from

**Returns:** Self for method chaining

**Raises:** `ValueError` if node not found

##### set_finish_point()

Set a node as a finish point (adds edge to END).

```python
def set_finish_point(self, node_id: str) -> "StateGraph[StateType]"
```

**Parameters:**
- `node_id`: Node that finishes the graph

**Returns:** Self for method chaining

##### compile()

Compile the graph for execution.

```python
def compile(
    self,
    checkpointer: Optional[CheckpointerProtocol] = None,
    **config_kwargs: Any,
) -> CompiledGraph[StateType]
```text

**Parameters:**
- `checkpointer`: Optional checkpointer for state persistence
- `**config_kwargs`: Additional configuration options

**Returns:** `CompiledGraph` ready for execution

**Raises:** `ValueError` if graph validation fails

**Example:**
```python
from victor.framework.graph import MemoryCheckpointer

checkpointer = MemoryCheckpointer()
app = graph.compile(checkpointer=checkpointer)
```

### CompiledGraph Class

The compiled graph ready for execution.

#### invoke()

Execute the graph to completion.

```python
async def invoke(
    self,
    input_state: StateType,
    *,
    config: Optional[GraphConfig] = None,
    thread_id: Optional[str] = None,
    debug_hook: Optional[Any] = None,
) -> GraphExecutionResult[StateType]
```text

**Parameters:**
- `input_state`: Initial state dictionary
- `config`: Optional execution configuration override
- `thread_id`: Thread ID for checkpointing (auto-generated if not provided)
- `debug_hook`: Optional DebugHook for debugging

**Returns:** `GraphExecutionResult` with final state

**Example:**
```python
result = await app.invoke({
    "messages": [],
    "task": "Fix the bug in auth.py"
})

if result.success:
    print(f"Final state: {result.state}")
else:
    print(f"Error: {result.error}")
```

#### stream()

Stream execution yielding state after each node.

```python
async def stream(
    self,
    input_state: StateType,
    *,
    config: Optional[GraphConfig] = None,
    thread_id: Optional[str] = None,
) -> AsyncIterator[Tuple[str, StateType]]
```text

**Parameters:**
- `input_state`: Initial state dictionary
- `config`: Optional execution configuration override
- `thread_id`: Thread ID for checkpointing

**Yields:** Tuple of `(node_id, state)` after each node execution

**Example:**
```python
async for node_id, state in app.stream({"task": "analyze code"}):
    print(f"Completed node: {node_id}")
    print(f"Current state: {state}")
```

#### get_graph_schema()

Get graph structure as a dictionary.

```python
def get_graph_schema(self) -> Dict[str, Any]
```text

**Returns:** Dictionary describing nodes and edges

### Constants

```python
from victor.framework.graph import END, START

END = "__end__"    # Sentinel for graph termination
START = "__start__" # Sentinel for graph start
```

### TypedDict State Pattern

Define workflow state using TypedDict for type safety:

```python
from typing import TypedDict, Optional, List

class ResearchState(TypedDict):
    query: str
    sources: List[str]
    findings: Optional[str]
    report: Optional[str]
    iteration: int
```text

---

## UnifiedWorkflowCompiler

The `UnifiedWorkflowCompiler` consolidates all workflow compilation paths with integrated two-level caching.

**Module:** `victor.workflows.unified_compiler`

### UnifiedWorkflowCompiler Class

```python
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

compiler = UnifiedWorkflowCompiler(enable_caching=True, cache_ttl=3600)
```

#### Constructor

```python
def __init__(
    self,
    definition_cache: Optional[WorkflowDefinitionCache] = None,
    execution_cache: Optional[WorkflowCacheManager] = None,
    orchestrator: Optional[AgentOrchestrator] = None,
    tool_registry: Optional[ToolRegistry] = None,
    runner_registry: Optional[NodeRunnerRegistry] = None,
    emitter: Optional[Any] = None,
    enable_caching: bool = True,
    cache_ttl: int = 3600,
    config: Optional[UnifiedCompilerConfig] = None,
) -> None
```text

**Parameters:**
- `definition_cache`: Cache for parsed YAML definitions
- `execution_cache`: Cache for execution results
- `orchestrator`: Agent orchestrator for agent nodes
- `tool_registry`: Tool registry for compute nodes
- `runner_registry`: NodeRunner registry for unified execution
- `emitter`: ObservabilityEmitter for streaming events
- `enable_caching`: Whether to enable caching (default: True)
- `cache_ttl`: Cache TTL in seconds (default: 3600)
- `config`: Full compiler configuration

#### compile_yaml()

Compile a workflow from a YAML file.

```python
def compile_yaml(
    self,
    yaml_path: Union[str, Path],
    workflow_name: Optional[str] = None,
    condition_registry: Optional[Dict[str, Callable[..., Any]]] = None,
    transform_registry: Optional[Dict[str, Callable[..., Any]]] = None,
    **kwargs: Any,
) -> CachedCompiledGraph
```

**Parameters:**
- `yaml_path`: Path to the YAML file
- `workflow_name`: Specific workflow to compile (if file has multiple)
- `condition_registry`: Custom condition functions (escape hatches)
- `transform_registry`: Custom transform functions (escape hatches)

**Returns:** `CachedCompiledGraph` ready for execution

**Example:**
```python
from pathlib import Path

# Define escape hatches
conditions = {
    "quality_check": lambda ctx: "pass" if ctx.get("score", 0) > 0.8 else "fail"
}

graph = compiler.compile_yaml(
    Path("workflows/research.yaml"),
    "deep_research",
    condition_registry=conditions
)
```text

#### compile_yaml_content()

Compile from YAML content string.

```python
def compile_yaml_content(
    self,
    yaml_content: str,
    workflow_name: str,
    condition_registry: Optional[Dict[str, Callable[..., Any]]] = None,
    transform_registry: Optional[Dict[str, Callable[..., Any]]] = None,
    **kwargs: Any,
) -> CachedCompiledGraph
```

#### compile_definition()

Compile a WorkflowDefinition object.

```python
def compile_definition(
    self,
    definition: WorkflowDefinition,
    cache_key: Optional[str] = None,
    **kwargs: Any,
) -> CachedCompiledGraph
```text

#### Cache Management

```python
# Clear all caches
cleared_count = compiler.clear_cache()

# Get cache statistics
stats = compiler.get_cache_stats()
# Returns: {"compilation": {...}, "definition_cache": {...}, "execution_cache": {...}}

# Invalidate specific YAML file
compiler.invalidate_yaml(Path("workflows/outdated.yaml"))
```

### CachedCompiledGraph Class

Wrapper around CompiledGraph with cache integration.

```python
@dataclass
class CachedCompiledGraph:
    compiled_graph: CompiledGraph
    workflow_name: str
    source_path: Optional[Path] = None
    compiled_at: float
    source_mtime: Optional[float] = None
    cache_key: str = ""
    max_execution_timeout_seconds: Optional[float] = None
    default_node_timeout_seconds: Optional[float] = None
    max_iterations: int = 25
    max_retries: int = 0
```text

#### invoke()

Execute the compiled workflow.

```python
async def invoke(
    self,
    input_state: Dict[str, Any],
    *,
    config: Optional[GraphConfig] = None,
    thread_id: Optional[str] = None,
    use_cache: bool = True,
) -> GraphExecutionResult
```

**Parameters:**
- `input_state`: Initial state for execution
- `config`: Optional execution configuration override
- `thread_id`: Thread ID for checkpointing
- `use_cache`: Whether to use execution cache

**Returns:** `GraphExecutionResult` with final state

#### stream()

Stream workflow execution.

```python
async def stream(
    self,
    input_state: Dict[str, Any],
    *,
    config: Optional[GraphConfig] = None,
    thread_id: Optional[str] = None,
) -> AsyncIterator[Tuple[str, Dict[str, Any]]]
```text

### Two-Level Caching

The UnifiedWorkflowCompiler implements two-level caching:

1. **Definition Cache**: Caches parsed YAML workflow definitions
   - Key: file path + workflow name + file mtime + config hash
   - Invalidated when source file changes

2. **Execution Cache**: Caches workflow execution results
   - Key: input state hash + workflow definition hash
   - TTL-based expiration

```python
# Check cache stats
stats = compiler.get_cache_stats()
print(f"Definition cache hits: {stats['definition_cache']['hits']}")
print(f"Compilation count: {stats['compilation']['yaml_compiles']}")
```

### Convenience Functions

```python
from victor.workflows.unified_compiler import (
    compile_workflow,
    compile_and_execute,
    create_unified_compiler,
)

# One-off compilation
graph = compile_workflow(Path("workflow.yaml"), "my_workflow")

# Compile and execute in one step
result = await compile_and_execute(
    Path("workflow.yaml"),
    initial_state={"query": "AI trends"},
    workflow_name="research"
)

# Create compiler with default caches
compiler = create_unified_compiler(enable_caching=True)
```text


**Reading Time:** 6 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


## BaseYAMLWorkflowProvider
