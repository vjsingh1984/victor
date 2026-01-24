# Workflows API Reference

This document provides API reference documentation for Victor's workflow system, including the StateGraph DSL, UnifiedWorkflowCompiler, YAML workflow providers, and node types.

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
```

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
```

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
```

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
```

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
```

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
```

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
```

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
```

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
```

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
```

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
```

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
```

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
```

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
```

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
```

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
```

---

## BaseYAMLWorkflowProvider

Template Method pattern base class for vertical-specific workflow providers.

**Module:** `victor.framework.workflows.base_yaml_provider`

### BaseYAMLWorkflowProvider Class

```python
from victor.framework.workflows import BaseYAMLWorkflowProvider

class ResearchWorkflowProvider(BaseYAMLWorkflowProvider):
    def _get_escape_hatches_module(self) -> str:
        return "victor.research.escape_hatches"
```

#### Abstract Methods

##### _get_escape_hatches_module()

Return the module path for escape hatches (CONDITIONS/TRANSFORMS).

```python
@abstractmethod
def _get_escape_hatches_module(self) -> str
```

**Returns:** Fully qualified module path string

**Example:**
```python
def _get_escape_hatches_module(self) -> str:
    return "victor.devops.escape_hatches"
```

#### Optional Override Methods

##### _get_workflows_directory()

Return the directory containing YAML workflow files.

```python
def _get_workflows_directory(self) -> Path
```

**Default:** Derives from escape hatches module path + `/workflows/`

##### get_auto_workflows()

Get automatic workflow triggers based on query patterns.

```python
def get_auto_workflows(self) -> List[Tuple[str, str]]
```

**Returns:** List of `(regex_pattern, workflow_name)` tuples

##### get_workflow_for_task_type()

Map task types to workflow names.

```python
def get_workflow_for_task_type(self, task_type: str) -> Optional[str]
```

#### Workflow Access Methods

```python
# Get all workflows
workflows = provider.get_workflows()  # Dict[str, WorkflowDefinition]

# Get specific workflow
workflow = provider.get_workflow("deep_research")  # Optional[WorkflowDefinition]

# Get workflow names
names = provider.get_workflow_names()  # List[str]
```

#### Compilation and Execution

##### compile_workflow()

Compile a workflow using the unified compiler.

```python
def compile_workflow(self, workflow_name: str) -> CachedCompiledGraph
```

**Example:**
```python
provider = ResearchWorkflowProvider()
compiled = provider.compile_workflow("deep_research")
result = await compiled.invoke({"query": "AI trends"})
```

##### run_compiled_workflow()

Execute a workflow with automatic compilation.

```python
async def run_compiled_workflow(
    self,
    workflow_name: str,
    context: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
) -> GraphExecutionResult
```

##### stream_compiled_workflow()

Stream workflow execution.

```python
async def stream_compiled_workflow(
    self,
    workflow_name: str,
    context: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
) -> AsyncIterator[tuple]
```

**Example:**
```python
async for node_id, state in provider.stream_compiled_workflow(
    "deep_research",
    {"query": "AI trends"}
):
    print(f"Completed: {node_id}")
```

---

## Node Types API

Victor workflows support multiple node types for different execution patterns.

**Module:** `victor.workflows.definition`

### AgentNode

LLM-powered execution node that spawns a sub-agent.

```python
@dataclass
class AgentNode(WorkflowNode):
    role: str = "executor"           # Agent role (researcher, planner, executor, reviewer, writer)
    goal: str = ""                   # Task description
    tool_budget: int = 15            # Maximum tool calls
    allowed_tools: Optional[List[str]] = None  # Specific tools to allow
    input_mapping: Dict[str, str]    # Map context keys to agent inputs
    output_key: Optional[str] = None # Key to store output
    llm_config: Optional[Dict[str, Any]] = None  # LLM configuration
    timeout_seconds: Optional[float] = None  # Execution timeout
    profile: Optional[str] = None    # Provider profile
    disable_embeddings: bool = False # Disable semantic search
```

**YAML Example:**
```yaml
nodes:
  - id: analyze
    type: agent
    role: researcher
    goal: "Analyze ${query} and find relevant information"
    tool_budget: 20
    allowed_tools: [web_search, read_file]
    output_key: analysis_result
    next: [synthesize]
```

### ComputeNode

Direct tool execution without LLM inference.

```python
@dataclass
class ComputeNode(WorkflowNode):
    tools: List[str]                 # Tools to execute in sequence
    input_mapping: Dict[str, str]    # Map context keys to tool parameters
    output_key: Optional[str] = None # Key to store outputs
    constraints: TaskConstraints     # Execution constraints
    handler: Optional[str] = None    # Custom handler name
    fail_fast: bool = True           # Stop on first failure
    parallel: bool = False           # Execute tools in parallel
    execution_target: str = "in-process"  # Execution environment
```

**YAML Example:**
```yaml
nodes:
  - id: fetch_data
    type: compute
    tools: [sec_filing, market_data]
    inputs:
      symbol: $ctx.symbol
    output: financial_data
    constraints:
      llm_allowed: false
      max_cost_tier: FREE
      timeout: 60
    next: [analyze]
```

#### TaskConstraints

```python
@dataclass
class TaskConstraints(ConstraintsProtocol):
    llm_allowed: bool = False        # Whether LLM inference is permitted
    network_allowed: bool = True     # Whether network calls are permitted
    write_allowed: bool = False      # Whether disk writes are permitted
    max_cost_tier: str = "FREE"      # FREE, LOW, MEDIUM, HIGH
    _max_tool_calls: int = 100       # Maximum tool invocations
    _timeout: float = 300.0          # Execution timeout in seconds
    allowed_tools: Optional[List[str]] = None
    blocked_tools: Optional[List[str]] = None
```

**Constraint Presets:**
```python
from victor.workflows.definition import (
    AirgappedConstraints,    # No LLM, no network, no writes
    ComputeOnlyConstraints,  # No LLM, network allowed, no writes
    FullAccessConstraints,   # Full access (use carefully)
)
```

### ConditionNode

Branch based on a condition function.

```python
@dataclass
class ConditionNode(WorkflowNode):
    condition: Callable[[Dict[str, Any]], str]  # Returns branch name
    branches: Dict[str, str]                     # Branch name -> node ID
```

**YAML Example:**
```yaml
nodes:
  - id: check_quality
    type: condition
    condition: "quality_threshold"  # Escape hatch function name
    branches:
      high_quality: proceed
      needs_cleanup: cleanup
      error: error_handler
```

**Escape Hatch Definition:**
```python
# In escape_hatches.py
CONDITIONS = {
    "quality_threshold": lambda ctx: (
        "high_quality" if ctx.get("score", 0) > 0.8
        else "needs_cleanup" if ctx.get("score", 0) > 0.5
        else "error"
    )
}
```

### ParallelNode

Execute multiple nodes concurrently.

```python
@dataclass
class ParallelNode(WorkflowNode):
    parallel_nodes: List[str]   # Node IDs to execute in parallel
    join_strategy: str = "all"  # "all", "any", or "merge"
```

**YAML Example:**
```yaml
nodes:
  - id: parallel_research
    type: parallel
    parallel_nodes: [search_web, search_docs, search_code]
    join_strategy: all
    next: [merge_results]
```

**Join Strategies:**
- `all`: Wait for all nodes to complete (fail if any fail)
- `any`: Continue when any node completes successfully
- `merge`: Merge all results regardless of success/failure

### TransformNode

Transform context data without LLM.

```python
@dataclass
class TransformNode(WorkflowNode):
    transform: Callable[[Dict[str, Any]], Dict[str, Any]]
```

**YAML Example:**
```yaml
nodes:
  - id: format_output
    type: transform
    handler: format_report  # Escape hatch transform function
    next: [final_output]
```

**Escape Hatch Definition:**
```python
TRANSFORMS = {
    "format_report": lambda ctx: {
        **ctx,
        "formatted_report": f"# Report\n\n{ctx.get('analysis', '')}"
    }
}
```

### HITLNode (Human-in-the-Loop)

Pause workflow for human interaction.

```python
from victor.workflows.hitl import HITLNode, HITLNodeType, HITLFallback

@dataclass
class HITLNode(WorkflowNode):
    hitl_type: HITLNodeType         # APPROVAL, CHOICE, REVIEW, CONFIRMATION
    prompt: str                      # Message to display
    context_keys: List[str]          # Keys to show user
    choices: Optional[List[str]]     # For CHOICE type
    default_value: Optional[str]     # Default on timeout
    timeout: float = 300.0           # Timeout in seconds
    fallback: HITLFallback           # ABORT, CONTINUE, SKIP
```

**HITL Types:**
- `APPROVAL`: Binary approve/reject gate
- `CHOICE`: Select from multiple options
- `REVIEW`: Review and optionally modify context
- `CONFIRMATION`: Simple acknowledgment

**YAML Example:**
```yaml
nodes:
  - id: approve_changes
    type: hitl
    hitl_type: approval
    prompt: "Approve the following changes?"
    context_keys: [files_modified, changes_summary]
    timeout: 300
    fallback: abort
    next: [apply_changes]
```

### TeamNode

Spawn an ad-hoc multi-agent team.

```python
@dataclass
class TeamNode:
    id: str
    name: str
    goal: str
    team_formation: TeamFormation    # SEQUENTIAL, PARALLEL, HIERARCHICAL, PIPELINE, CONSENSUS
    members: List[TeamMember]
    config: TeamNodeConfig
    shared_context: Dict[str, Any]
    max_iterations: int = 50
    total_tool_budget: int = 100
```

**YAML Example:**
```yaml
nodes:
  - id: research_team
    type: team
    goal: "Conduct comprehensive research on ${topic}"
    team_formation: sequential
    members:
      - id: researcher
        role: researcher
        goal: "Find information"
        tool_budget: 20
      - id: writer
        role: writer
        goal: "Synthesize findings"
        tool_budget: 10
    timeout_seconds: 600
    merge_strategy: dict
    output_key: team_result
    next: [review]
```

---

## State Management

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

---

## Complete Example

```python
from typing import TypedDict, Optional, List
from victor.framework.graph import StateGraph, END
from victor.framework.config import GraphConfig, ExecutionConfig

# Define typed state
class ResearchState(TypedDict):
    query: str
    sources: List[str]
    findings: Optional[str]
    report: Optional[str]

# Define node functions
async def search(state: ResearchState) -> ResearchState:
    state["sources"] = [f"Source for: {state['query']}"]
    return state

async def analyze(state: ResearchState) -> ResearchState:
    state["findings"] = f"Analyzed: {state['sources']}"
    return state

def should_continue(state: ResearchState) -> str:
    if len(state.get("sources", [])) > 0:
        return "continue"
    return "retry"

async def report(state: ResearchState) -> ResearchState:
    state["report"] = f"# Report\n\n{state['findings']}"
    return state

# Build graph
graph = StateGraph(ResearchState)
graph.add_node("search", search)
graph.add_node("analyze", analyze)
graph.add_node("report", report)

graph.add_edge("search", "analyze")
graph.add_conditional_edge(
    "analyze",
    should_continue,
    {"continue": "report", "retry": "search"}
)
graph.set_entry_point("search")
graph.set_finish_point("report")

# Compile with configuration
config = GraphConfig(
    execution=ExecutionConfig(max_iterations=10, timeout=60.0)
)
app = graph.compile()

# Execute
async def main():
    result = await app.invoke(
        {"query": "AI trends 2024", "sources": [], "findings": None, "report": None},
        config=config
    )

    if result.success:
        print(f"Report:\n{result.state['report']}")
    else:
        print(f"Error: {result.error}")

import asyncio
asyncio.run(main())
```
