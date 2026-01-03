# Workflow Architecture

This document explains the relationship between Tasks, Agents, and ComputeNodes in the Victor framework.

![Node Architecture](./assets/node-architecture.svg)

## Core Concepts

### Task
A **Task** is a unit of work that needs to be completed. Tasks are not a first-class object in the framework; instead, they are represented by the `goal` field in nodes.

### Agent
An **Agent** is an LLM-powered orchestrator that can:
- Reason about the task
- Plan a sequence of actions
- Select and invoke tools dynamically
- Handle ambiguity and errors

### Workflow Node Types

Victor workflows are composed of nodes. The two primary execution node types are:

| Node Type | LLM Required | Cost | Deterministic | Use Case |
|-----------|--------------|------|---------------|----------|
| **AgentNode** | Yes | High | No | Complex reasoning, ambiguous tasks |
| **ComputeNode** | No | Low | Yes | Data processing, calculations, pipelines |

## AgentNode vs ComputeNode

### AgentNode - LLM-Powered Execution

```yaml
- id: analyze_code
  type: agent
  role: researcher
  goal: "Find security vulnerabilities in the codebase"
  tool_budget: 20
  allowed_tools: [read_file, grep, ast_analyze]
```

**Characteristics:**
- **Brain + Hands**: LLM decides what to do, then does it
- **Dynamic Tool Selection**: Agent chooses which tools to use based on context
- **Handles Ambiguity**: Can reason about unclear requirements
- **Expensive**: Requires LLM inference for each decision
- **Non-deterministic**: Same input may produce different tool sequences

**When to use AgentNode:**
- Tasks requiring reasoning or planning
- Ambiguous or open-ended goals
- Interactive or conversational workflows
- Complex multi-step investigations

### ComputeNode - Direct Tool Execution

```yaml
- id: run_valuation
  type: compute
  tools: [sec_filing, multi_model_valuation, sector_valuation]
  input_mapping:
    symbol: $ctx.symbol
    date: $ctx.analysis_date
  output: valuation_result
  constraints:
    llm_allowed: false
    max_cost_tier: FREE
    timeout: 60
```

**Characteristics:**
- **Just Hands**: Predefined sequence of tool executions
- **Static Tool List**: Tools are specified upfront in YAML
- **Deterministic**: Same input always produces same tool sequence
- **Cheap**: No LLM inference, just direct tool calls
- **Fast**: No reasoning overhead

**When to use ComputeNode:**
- Data processing pipelines
- Mathematical calculations
- API calls with known inputs
- Cost-sensitive batch operations
- Airgapped/offline environments

## Constraint System

![Constraints Flow](./assets/constraints-flow.svg)

ComputeNodes support fine-grained execution constraints via the `ConstraintsProtocol`:

### Standard Constraints

```yaml
constraints:
  llm_allowed: false      # No LLM inference
  network_allowed: true   # Can make network calls
  write_allowed: false    # Cannot write to disk
  max_cost_tier: LOW      # Only FREE and LOW tier tools
  timeout: 60             # 60 second timeout per tool
  max_tool_calls: 10      # Maximum 10 tool invocations
```

### Preset Constraints

| Preset | LLM | Network | Write | Max Cost |
|--------|-----|---------|-------|----------|
| `AirgappedConstraints` | No | No | No | FREE |
| `ComputeOnlyConstraints` | No | Yes | No | LOW |
| `FullAccessConstraints` | Yes | Yes | Yes | HIGH |

### Custom Constraints

Implement `ConstraintsProtocol` for domain-specific constraints:

```python
from victor.workflows.definition import ConstraintsProtocol

@dataclass
class InvestmentConstraints(ConstraintsProtocol):
    """Constraints for investment workflows."""
    max_api_calls_per_minute: int = 10
    require_sec_compliance: bool = True
    allowed_data_sources: List[str] = field(default_factory=list)

    def allows_tool(self, tool_name: str, cost_tier: str = "FREE") -> bool:
        # Custom validation: only allow SEC-compliant tools
        if self.require_sec_compliance:
            return tool_name in SEC_COMPLIANT_TOOLS
        return True

    # ... implement other abstract methods
```

## Custom Handlers

ComputeNodes support custom execution logic via handlers:

```python
from victor.workflows.executor import register_compute_handler, ComputeHandler

async def rl_decision_handler(
    node: ComputeNode,
    context: WorkflowContext,
    tool_registry: ToolRegistry,
) -> NodeResult:
    """Custom handler for RL-based decisions."""
    # Load policy
    policy = load_policy(context.get("policy_path"))

    # Extract features from context
    features = build_features(context.data)

    # Make prediction
    weights = policy.predict(features)

    # Store in context
    context.set(node.output_key, weights)

    return NodeResult(
        node_id=node.id,
        status=NodeStatus.COMPLETED,
        output=weights,
    )

# Register the handler
register_compute_handler("rl_decision", rl_decision_handler)
```

Use in YAML:

```yaml
- id: determine_weights
  type: compute
  handler: rl_decision  # Uses custom handler
  input_mapping:
    policy_path: $ctx.model_path
    features: $ctx.valuation_features
  output: model_weights
```

## Execution Flow

```
WorkflowDefinition (YAML)
         │
         ▼
   WorkflowExecutor
         │
    ┌────┴────┐
    ▼         ▼
AgentNode  ComputeNode
    │         │
    ▼         ▼
   LLM    Constraints
    │      Check
    ▼         │
   Tool       ▼
Selection  Direct Tool
    │      Execution
    │         │
    └────┬────┘
         ▼
    ToolRegistry
         │
         ▼
    ToolResult
```

## Best Practices

### Use ComputeNode for:
- ✅ Data extraction and transformation
- ✅ Calculations and computations
- ✅ Known, repeatable workflows
- ✅ Cost-sensitive operations
- ✅ Batch processing

### Use AgentNode for:
- ✅ Complex reasoning tasks
- ✅ Open-ended investigations
- ✅ Tasks requiring judgment
- ✅ Interactive workflows
- ✅ Error recovery with reasoning

### Hybrid Workflows

Combine both node types for optimal cost/capability balance:

```yaml
nodes:
  # Cheap: Direct data collection
  - id: fetch_data
    type: compute
    tools: [sec_filing, market_data]
    constraints:
      llm_allowed: false

  # Cheap: Direct computation
  - id: run_models
    type: compute
    tools: [dcf_valuation, pe_valuation]
    constraints:
      llm_allowed: false

  # Expensive: LLM synthesis (only when needed)
  - id: synthesize
    type: agent
    role: analyst
    goal: "Analyze results and provide recommendation"
    tool_budget: 10
```

This pattern keeps 90% of execution LLM-free while using LLM intelligence where it matters most.

## YAML Workflow System

Victor supports declarative YAML-based workflow definitions with hybrid Python/YAML loading.

### YAML Workflow Structure

```yaml
workflows:
  my_workflow:
    description: "Workflow description"

    metadata:
      vertical: investment    # dataanalysis, coding, research, devops, rag
      category: valuation
      complexity: high

    batch_config:
      batch_size: 10
      max_concurrent: 5
      retry_strategy: end_of_batch

    temporal_context:
      as_of_date: $ctx.analysis_date
      lookback_periods: 8

    nodes:
      - id: step1
        type: compute
        handler: my_handler
        inputs:
          symbol: $ctx.symbol
        output: result
        next: [step2]
```

### Hybrid Loading Pattern

Each vertical loads workflows from both Python and YAML sources:

```python
class InvestmentWorkflowProvider(WorkflowProviderProtocol):
    def get_workflows(self):
        # YAML overrides Python (external overrides inline)
        return {**self._load_python_workflows(), **self._load_yaml_workflows()}
```

**Override Order:**
1. Python `@workflow` definitions (default baseline)
2. YAML files in `{vertical}/workflows/*.yaml` (override Python)

## Isolation and Sandboxing

The `IsolationMapper` maps workflow constraints to execution environments.

### Sandbox Types

| Type | Execution | Isolation Level |
|------|-----------|-----------------|
| `none` | Inline | No isolation, direct execution |
| `process` | Subprocess | Process-level isolation |
| `docker` | Container | Full container isolation |

### Vertical Defaults

| Vertical | Default Sandbox | Network | Filesystem |
|----------|-----------------|---------|------------|
| **coding** | process | Yes | Read/Write |
| **research** | none | Yes | Read/Write |
| **devops** | docker | Yes | Read/Write |
| **dataanalysis** | process | Yes | Read/Write |
| **rag** | docker | Yes | Read-only |
| **investment** | process | Yes | Read-only |

### Constraint-Based Overrides

Constraints take precedence over vertical defaults:

```yaml
constraints:
  llm_allowed: false
  network_allowed: false    # Forces airgapped mode
  write_allowed: false
```

## Framework Handlers

Victor provides reusable framework-level handlers for common patterns:

| Handler | Purpose |
|---------|---------|
| `parallel_tools` | Execute multiple tools concurrently |
| `sequential_tools` | Chain tools with output piping |
| `retry_with_backoff` | Retry failed operations with exponential backoff |
| `data_transform` | Apply transformations to context data |
| `conditional_branch` | Evaluate conditions and select next node |

### Usage

```yaml
- id: parallel_fetch
  type: compute
  handler: parallel_tools
  tools: [api_call_1, api_call_2, api_call_3]
  output: combined_results
```

## Domain-Specific Handlers

Each vertical defines domain-specific handlers:

### Investment Vertical Handlers

| Handler | Service Used | Purpose |
|---------|--------------|---------|
| `metadata_fetch` | SymbolMetadataService | Sector, industry, beta |
| `price_data_fetch` | PriceService | Current and historical prices |
| `sec_data_extract` | FinancialDataService | SEC quarterly/TTM data |
| `valuation_compute` | ParallelValuationOrchestrator | Multi-model valuation |
| `sector_valuation` | SectorValuationRouter | Sector-specific routing |
| `rl_weight_decision` | RLModelWeightingService | RL-based weight selection |
| `technical_analysis` | TechnicalAnalysisService | Technical indicators |
| `outcome_tracking` | OutcomeTracker | RL training data |

### Handler Registration

Handlers are registered when the vertical module is imported:

```python
from victor.workflows.executor import register_compute_handler

HANDLERS = {
    "valuation_compute": ValuationComputeHandler(),
    "rl_weight_decision": RLWeightDecisionHandler(),
}

def register_handlers():
    for name, handler in HANDLERS.items():
        register_compute_handler(name, handler)
```

## Workflow Pipelines

Victor supports three primary pipeline patterns:

### 1. Single Item Pipeline

For processing individual items (e.g., single stock analysis):

```yaml
nodes:
  - id: fetch_data
    type: compute
    handler: sec_data_extract
    inputs:
      symbol: $ctx.symbol
    output: financial_data
    next: [analyze]

  - id: analyze
    type: compute
    handler: valuation_compute
    inputs:
      financials: $ctx.financial_data
    output: valuation_result
```

### 2. Batch Pipeline

For processing multiple items in parallel:

```yaml
batch_config:
  batch_size: 10
  max_concurrent: 5

nodes:
  - id: process_batch
    type: parallel
    iterator: $ctx.symbols
    iterator_var: symbol
    parallel_nodes: [item_pipeline]
    join_strategy: all_settled
```

### 3. Temporal Pipeline

For historical/backtest processing with point-in-time data:

```yaml
temporal_context:
  as_of_date: $ctx.backtest_date
  lookback_periods: 8

nodes:
  - id: historical_data
    type: compute
    handler: sec_data_extract
    inputs:
      as_of_date: $ctx.backtest_date  # Point-in-time query
    output: pit_data
```

## Context References

Use `$ctx.` prefix to reference workflow context values:

| Pattern | Description |
|---------|-------------|
| `$ctx.symbol` | Direct context key |
| `$ctx.data.field` | Nested access |
| `$ctx.results[0]` | Array access |
| `$ctx.node_output` | Reference previous node output |

## Creating Custom Handlers

```python
from dataclasses import dataclass
from victor.workflows.executor import NodeResult, NodeStatus

@dataclass
class MyCustomHandler:
    """Custom compute handler."""

    async def __call__(
        self,
        node: ComputeNode,
        context: WorkflowContext,
        tool_registry: ToolRegistry,
    ) -> NodeResult:
        # Extract inputs
        symbol = self._get_input(node, context, "symbol")

        # Execute logic
        result = await my_business_logic(symbol)

        # Store in context
        context.set(node.output_key, result)

        return NodeResult(
            node_id=node.id,
            status=NodeStatus.COMPLETED,
            output=result,
        )

    def _get_input(self, node, context, key, default=None):
        value = node.input_mapping.get(key)
        if isinstance(value, str) and value.startswith("$ctx."):
            return context.get(value[5:]) or default
        return value if value is not None else default
```
