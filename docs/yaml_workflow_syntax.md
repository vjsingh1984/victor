# YAML Workflow Syntax Guide

**Version**: 1.0
**Branch**: refactor/framework-driven-cleanup
**Last Updated**: 2026-04-07

---

## Overview

YAML workflows allow you to define agent execution patterns declaratively without writing Python code. Workflows are defined in YAML format and can include:

- **Agent nodes** - Execute agents with specific roles and goals
- **Condition nodes** - Branch execution based on conditions
- **Transform nodes** - Process and transform state data
- **Parallel nodes** - Execute multiple nodes concurrently
- **HITL nodes** - Human-in-the-loop approval/checkpoints

---

## Basic Structure

### Workflow Definition

```yaml
workflows:
  workflow_name:
    description: "Human-readable description"
    metadata:
      version: "1.0"
      author: "team"
      tags: ["development", "review"]
    nodes:
      - id: node1
        type: agent
        # ... node configuration
```

### Complete Example

```yaml
workflows:
  code_review:
    description: "Automated code review workflow"
    metadata:
      version: "1.0"
      mode: build
    nodes:
      - id: analyze
        type: agent
        role: code_analyzer
        goal: "Analyze code structure and patterns"
        tool_budget: 20
        tools: [read, grep, code_search, symbols]
        next: [review]
      
      - id: review
        type: agent
        role: code_reviewer
        goal: "Review code for issues and improvements"
        tool_budget: 25
        next: [approve]
      
      - id: approve
        type: condition
        condition: "quality_score > 0.8"
        branches:
          true: merge
          false: revise
      
      - id: merge
        type: agent
        role: integrator
        goal: "Merge approved changes"
        tool_budget: 10
      
      - id: revise
        type: agent
        role: developer
        goal: "Make required revisions"
        tool_budget: 30
```

---

## Node Types

### 1. Agent Nodes

Execute an agent with a specific role and goal.

```yaml
- id: node_id
  type: agent
  role: researcher          # Required: Agent role/vertical
  goal: "Task description"   # Required: What the agent should do
  tool_budget: 20           # Optional: Max tool calls (default: varies by mode)
  tools: [read, grep]       # Optional: Allowed tools
  allowed_tools:           # Alternative: Specify tool categories
    - filesystem
    - search
  forbidden_tools:         # Optional: Tools to exclude
    - shell
    - docker
  next: [next_node]         # Optional: Next node(s)
  output: result_var        # Optional: Store output in state
```

**Full Example**:

```yaml
- id: investigate
  type: agent
  role: bug_investigator
  goal: "Investigate the authentication bug in auth.py"
  tool_budget: 15
  tools: [read, grep, code_search, symbols]
  next: [report]
```

### 2. Condition Nodes

Branch execution based on state conditions.

```yaml
- id: decide
  type: condition
  condition: "state_key > value"  # Required: Condition expression
  branches:
    true: node_if_true           # Required: Node ID if condition is true
    false: node_if_false         # Required: Node ID if condition is false
  default: default_node          # Optional: Default if neither matches
```

**Supported Operators**:
- Equality: `key == value`, `key != value`
- Comparison: `key > value`, `key >= value`, `key < value`, `key <= value`
- Membership: `key in [a, b, c]`
- Existence: `key` (truthy if key exists and is not null/false)

**Examples**:

```yaml
# Numeric comparison
- id: check_quality
  type: condition
  condition: "quality_score > 0.8"
  branches:
    true: approve
    false: reject

# String equality
- id: check_status
  type: condition
  condition: "status == 'ready'"
  branches:
    true: deploy
    false: wait

# List membership
- id: check_type
  type: condition
  condition: "file_type in ['py', 'js', 'ts']"
  branches:
    true: analyze
    false: skip
```

### 3. Transform Nodes

Process and transform state data.

```yaml
- id: transform
  type: transform
  transform: transform_name    # Required: Transform function name
  input: source_var            # Optional: Input state key
  output: result_var           # Optional: Output state key
  params:                      # Optional: Transform parameters
    param1: value1
    param2: value2
  next: [next_node]
```

**Built-in Transforms**:

```yaml
# Merge results from parallel branches
- id: merge
  type: transform
  transform: merge_results
  input: results
  output: merged_data
  next: [process]

# Extract specific fields
- id: extract
  type: transform
  transform: extract_field
  params:
    field: quality_score
  output: score
  next: [evaluate]

# Aggregate data
- id: aggregate
  type: transform
  transform: aggregate
  params:
    operation: sum
    field: count
  output: total
  next: [report]
```

### 4. Parallel Nodes

Execute multiple nodes concurrently.

```yaml
- id: split
  type: parallel
  nodes:                       # Required: Nodes to execute in parallel
    - id: task1
      type: agent
      role: worker1
      goal: "Process part 1"
    - id: task2
      type: agent
      role: worker2
      goal: "Process part 2"
    - id: task3
      type: agent
      role: worker3
      goal: "Process part 3"
  join: merge                 # Required: Node to join results
  wait_for: all               # Optional: Wait condition (all/any/n)
  timeout: 300                # Optional: Timeout in seconds
  next: [process]
```

**Example**:

```yaml
- id: parallel_analysis
  type: parallel
  nodes:
    - id: static_analysis
      type: agent
      role: static_analyzer
      goal: "Run static analysis"
      tool_budget: 15
    - id: security_scan
      type: agent
      role: security_scanner
      goal: "Scan for vulnerabilities"
      tool_budget: 20
    - id: performance_check
      type: agent
      role: performance_analyzer
      goal: "Check performance issues"
      tool_budget: 10
  join: merge_findings
  wait_for: all
  next: [report]
```

### 5. HITL Nodes

Human-in-the-loop nodes for approval and feedback.

```yaml
- id: approval
  type: hitl
  hitl_type: approval          # Required: approval/input/confirmation
  prompt: "Review the changes?" # Required: Prompt for human
  timeout: 300                # Optional: Timeout in seconds
  fallback: continue           # Required: What to do on timeout
  options:                    # Optional: Predefined options
    - approve: "Approve and continue"
    - reject: "Reject and revise"
    - modify: "Request modifications"
  next: [process_result]       # Optional: Next node after approval
```

**Examples**:

```yaml
# Approval checkpoint
- id: human_review
  type: hitl
  hitl_type: approval
  prompt: "Review the generated code. Approve to continue, or reject to revise."
  timeout: 600
  fallback: continue
  options:
    - approve: "Approve - continue with implementation"
    - reject: "Reject - return to planning"
    - modify: "Request modifications - provide feedback"
  next: [implement]

# Input request
- id: get_requirements
  type: hitl
  hitl_type: input
  prompt: "Please provide the requirements for this feature."
  timeout: 1200
  fallback: use_defaults
  next: [design]

# Confirmation checkpoint
- id: confirm_deployment
  type: hitl
  hitl_type: confirmation
  prompt: "Confirm deployment to production?"
  timeout: 60
  fallback: abort
  next: [deploy]
```

---

## State Management

### Passing Data Between Nodes

```yaml
workflows:
  stateful_workflow:
    nodes:
      - id: step1
        type: agent
        role: collector
        goal: "Gather information"
        output: findings          # Store output in state
        next: [step2]
      
      - id: step2
        type: agent
        role: analyzer
        goal: "Analyze findings"    # Can access state['findings']
        input: findings            # Explicitly use input
        next: [step3]
```

### State Variables

State is a dictionary that persists across workflow execution:

```yaml
# Initial state
{
  "project_root": "/path/to/project",
  "target_file": "auth.py",
  "quality_threshold": 0.8
}

# Nodes can access and modify state
- id: analyze
  type: agent
  goal: "Analyze {target_file}"    # Template substitution
  next: [evaluate]

- id: evaluate
  type: condition
  condition: "score >= quality_threshold"  # Access state variables
  branches:
    true: approve
    false: reject
```

---

## Control Flow

### Sequential Execution

```yaml
nodes:
  - id: step1
    type: agent
    next: [step2]
  
  - id: step2
    type: agent
    next: [step3]
  
  - id: step3
    type: agent
```

### Conditional Branching

```yaml
nodes:
  - id: check
    type: agent
    next: [decide]
  
  - id: decide
    type: condition
    condition: "status == 'ready'"
    branches:
      true: proceed
      false: wait
  
  - id: proceed
    type: agent
  
  - id: wait
    type: agent
```

### Parallel Execution

```yaml
nodes:
  - id: split
    type: parallel
    nodes:
      - id: task1
        type: agent
      - id: task2
        type: agent
    join: merge
    next: [finalize]
```

### Loops (Cyclic Graphs)

```yaml
nodes:
  - id: process
    type: agent
    goal: "Process item"
    next: [check_done]
  
  - id: check_done
    type: condition
    condition: "has_more_items"
    branches:
      true: process      # Loop back
      false: finish
  
  - id: finish
    type: agent
```

---

## Metadata and Configuration

### Workflow Metadata

```yaml
workflows:
  my_workflow:
    description: "Workflow description"
    metadata:
      version: "1.0"
      author: "Team Name"
      tags: ["tag1", "tag2"]
      mode: explore        # explore/plan/build
      exploration_multiplier: 2.0
      sandbox_only: true
      verbose: false
```

### Mode-Based Configuration

```yaml
# EXPLORE Mode - Read-only exploration
metadata:
  mode: explore
  exploration_multiplier: 3.0
  sandbox_only: true
  tool_budget: 15
  allowed_tools:
    - read
    - grep
    - code_search

# PLAN Mode - Planning without implementation
metadata:
  mode: plan
  exploration_multiplier: 2.5
  verbose_planning: true
  tool_budget: 10

# BUILD Mode - Implementation allowed
metadata:
  mode: build
  allows_modifications: true
  tool_budget: 30
  all_tools_available: true
```

---

## Error Handling

### Error Strategies

```yaml
- id: risky_operation
  type: agent
  goal: "Perform operation that might fail"
  error_strategy: continue     # continue/stop/retry
  retry_attempts: 3           # Optional: Number of retries
  retry_delay: 5              # Optional: Delay between retries (seconds)
  on_error: error_handler     # Optional: Node to call on error
  next: [next_step]
```

### Error Handling Node

```yaml
- id: error_handler
  type: agent
  role: error_recoverer
  goal: "Handle error and attempt recovery"
  input: error                # Error information from previous node
  next: [retry_or_abort]
```

---

## Best Practices

### 1. Keep Workflows Modular

```yaml
# Good: Modular workflow
workflows:
  investigate:
    # ... investigation nodes
  
  analyze:
    # ... analysis nodes
  
  report:
    # ... reporting nodes
```

### 2. Use Descriptive Node IDs

```yaml
# Good: Clear, descriptive IDs
- id: gather_requirements
- id: design_architecture
- id: implement_feature

# Bad: Generic IDs
- id: step1
- id: step2
- id: step3
```

### 3. Define Clear Goals

```yaml
# Good: Specific, actionable goal
- id: analyze
  type: agent
  role: performance_analyzer
  goal: "Analyze the database query performance in user_service.py and identify bottlenecks"
  tool_budget: 20

# Bad: Vague goal
- id: analyze
  type: agent
  goal: "Analyze the code"
```

### 4. Set Appropriate Tool Budgets

```yaml
# Exploration (more tool calls)
- id: explore
  type: agent
  tool_budget: 30
  allowed_tools:
    - read
    - grep
    - code_search
    - symbols
    - semantic_search

# Focused task (fewer tool calls)
- id: focused_task
  type: agent
  tool_budget: 10
  allowed_tools:
    - read
    - write
```

### 5. Use Conditions for Decision Points

```yaml
# Good: Clear branching logic
- id: decide
  type: condition
  condition: "complexity == 'high'"
  branches:
    true: senior_reviewer
    false: standard_reviewer
```

---

## Complete Examples

See `docs/yaml_workflow_examples.md` for complete, runnable workflow examples including:
- Feature development workflow
- Code review workflow
- Bug investigation workflow
- Documentation generation workflow
- Multi-agent team workflow

---

## Migration Guide

See `docs/yaml_workflow_migration.md` for:
- Migrating from programmatic workflows
- Converting StateGraph to YAML
- Best practices for migration
- Common pitfalls and solutions

---

## Advanced Features

### Custom Transforms

Define custom transform functions:

```python
# In your vertical or extension
from victor.workflows.transforms import register_transform

@register_transform("custom_aggregate")
def custom_aggregate(state: dict, params: dict) -> dict:
    """Custom aggregation logic."""
    # Your logic here
    return {"result": aggregated_value}
```

### Dynamic Node Configuration

```yaml
- id: dynamic_task
  type: agent
  role: "{role_var}"        # Template variable from state
  goal: "{goal_var}"
  tool_budget: "{budget_var}"  # Must be numeric
```

### Workflow Composition

```yaml
workflows:
  # Base workflow
  base_analysis:
    nodes:
      - id: analyze
        type: agent
        role: analyst
        next: [report]
      - id: report
        type: agent
        role: reporter
  
  # Extended workflow
  extended_analysis:
    description: "Base analysis with additional steps"
    extends: base_analysis    # Extend base workflow
    additional_nodes:
      - id: extra_check
        type: agent
        role: validator
        before: report         # Insert before 'report' node
```

---

## Reference

### Node Type Summary

| Type | Purpose | Required Fields |
|------|---------|-----------------|
| `agent` | Execute agent | `type`, `role`, `goal` |
| `condition` | Branch execution | `type`, `condition`, `branches` |
| `transform` | Process state | `type`, `transform` |
| `parallel` | Concurrent execution | `type`, `nodes`, `join` |
| `hitl` | Human-in-the-loop | `type`, `hitl_type`, `prompt` |

### Template Variables

Available in `goal`, `condition`, and other text fields:

- `{state_key}` - Access state variables
- `{workflow_name}` - Current workflow name
- `{node_id}` - Current node ID
- `{timestamp}` - Current timestamp

### Built-in State Variables

- `{project_root}` - Project root directory
- `{workspace}` - Current workspace directory
- `{mode}` - Current execution mode
- `{session_id}` - Current session ID

---

## Troubleshooting

### Common Issues

**Issue**: "Failed to parse YAML"
- **Solution**: Validate YAML syntax using online YAML validator
- **Check**: Indentation (spaces, not tabs), quote marks, brackets

**Issue**: "Node not found"
- **Solution**: Ensure referenced nodes exist in the workflow
- **Check**: Node ID spelling, scope (within same workflow)

**Issue**: "Condition evaluation failed"
- **Solution**: Verify condition syntax and variable names
- **Check**: State variables are available, operator syntax

**Issue**: "Parallel execution timeout"
- **Solution**: Increase timeout or reduce parallel node count
- **Check**: Individual node execution time

---

## Performance Tips

1. **Use tool budgets** to limit runaway agent execution
2. **Enable parallel execution** for independent tasks
3. **Cache results** in transform nodes for reuse
4. **Use conditions** to skip unnecessary work
5. **Optimize tool lists** to only necessary tools

---

## Next Steps

1. ✅ Read this syntax guide
2. 📖 See `yaml_workflow_examples.md` for complete examples
3. 📝 See `yaml_workflow_migration.md` for migration guide
4. 🚀 Start creating your own workflows
5. 💡 Experiment with different node types and patterns

---

**Need Help?**
- See examples in `examples/workflows/`
- Check troubleshooting section
- Review mode-based workflows for common patterns
