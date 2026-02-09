# Workflows API Reference - Part 2

**Part 2 of 4:** Node Types API

---

## Navigation

- [Part 1: StateGraph, Compiler, Provider](part-1-stategraph-compiler-provider.md)
- **[Part 2: Node Types](#)** (Current)
- [Part 3: State & Results](part-3-state-results.md)
- [Part 4: Configuration & Examples](part-4-configuration-examples.md)
- [**Complete Reference**](../workflows-api.md)

---

Template Method pattern base class for vertical-specific workflow providers.

**Module:** `victor.framework.workflows.base_yaml_provider`

### BaseYAMLWorkflowProvider Class

```python
from victor.framework.workflows import BaseYAMLWorkflowProvider

class ResearchWorkflowProvider(BaseYAMLWorkflowProvider):
    def _get_escape_hatches_module(self) -> str:
        return "victor.research.escape_hatches"
```text

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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

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
```text

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


**Reading Time:** 5 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


## State Management
