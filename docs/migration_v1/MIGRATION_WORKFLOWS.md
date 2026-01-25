# Workflow Migration Guide: Victor 0.5.x to 0.5.0

This guide explains how to migrate your workflows from Victor 0.5.x to 0.5.0.

## Table of Contents

1. [Overview](#overview)
2. [StateGraph DSL Changes](#stategraph-dsl-changes)
3. [Workflow Compiler Changes](#workflow-compiler-changes)
4. [Node Type Changes](#node-type-changes)
5. [Edge and Routing Changes](#edge-and-routing-changes)
6. [Migration Examples](#migration-examples)
7. [New Workflow Features](#new-workflow-features)

---

## Overview

### Key Changes

- **YAML-first**: Workflows defined in YAML instead of Python
- **UnifiedCompiler**: `UnifiedWorkflowCompiler` replaces `WorkflowExecutor`
- **Two-level Caching**: Definition and execution caching
- **Checkpointing**: Built-in checkpoint persistence
- **Validation**: Automatic workflow validation

---

## StateGraph DSL Changes

### 1. Graph Definition

**Before (0.5.x)**:
```python
from victor.framework.graph import StateGraph

graph = StateGraph()
graph.add_node("research", research_node)
graph.add_node("write", write_node)
graph.add_edge("research", "write")
graph.set_entry_point("research")
graph.set_finish_point("write")

compiled = graph.compile()
```

**After (0.5.0)**:
```yaml
# workflow.yaml
workflows:
  my_workflow:
    nodes:
      - id: research
        type: agent
        role: researcher
        goal: "Research the topic"
        next: [write]

      - id: write
        type: agent
        role: writer
        goal: "Write summary"
        next: [END]
```

```python
# Python
from victor.framework.workflows import BaseYAMLWorkflowProvider

provider = BaseYAMLWorkflowProvider("workflow.yaml")
compiled = provider.compile_workflow("my_workflow")
```

### 2. State Definition

**Before (0.5.x)**:
```python
from typing import TypedDict

class WorkflowState(TypedDict):
    query: str
    research: str
    summary: str
```

**After (0.5.0)**:
```python
from victor.framework import State

class WorkflowState(State):
    query: str
    research: str
    summary: str
```

---

## Workflow Compiler Changes

### 1. Compilation

**Before (0.5.x)**:
```python
from victor.workflows.executor import WorkflowExecutor

executor = WorkflowExecutor(orchestrator)
result = await executor.execute(workflow, context)
```

**After (0.5.0)**:
```python
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

compiler = UnifiedWorkflowCompiler(orchestrator)
compiled = compiler.compile(workflow)
result = await compiled.invoke(context)

# With checkpointing
result = await compiled.invoke(
    context,
    thread_id="conversation-123",
    checkpoint_ns="workflow-1"
)
```

### 2. Caching

**New in 0.5.0**:
```python
# Two-level caching
compiler = UnifiedWorkflowCompiler(
    orchestrator,
    cache_definitions=True,  # Cache compiled graphs
    cache_executions=True,    # Cache execution results
)

# Clear caches
await compiler.clear_definition_cache("my_workflow")
await compiler.clear_execution_cache(thread_id="123")
```

---

## Node Type Changes

### 1. Agent Nodes

**Before (0.5.x)**:
```python
nodes:
  - id: research
    type: agent
    agent: researcher
    prompt: "Research this topic"
```

**After (0.5.0)**:
```yaml
nodes:
  - id: research
    type: agent
    role: researcher
    goal: "Research this topic"
    next: [write]
    tool_budget: 10
    max_iterations: 5
```

### 2. Compute Nodes

**New in 0.5.0**:
```yaml
nodes:
  - id: process_data
    type: compute
    handler: data_processor  # Registered handler
    inputs:
      data: $ctx.research_results
    next: [analyze]

  - id: analyze
    type: compute
    handler: statistics_calculator
    inputs:
      data: $ctx.processed_data
      method: "mean"
    next: [END]
```

### 3. Condition Nodes

**Enhanced in 0.5.0**:
```yaml
nodes:
  - id: check_quality
    type: condition
    condition: "quality_threshold"  # Escape hatch function
    inputs:
      score: $ctx.quality_score
      threshold: 0.8
    branches:
      "high_quality": proceed
      "needs_work": refine
    default: refine  # Default branch if no match
```

### 4. Parallel Nodes

**New in 0.5.0**:
```yaml
nodes:
  - id: parallel_analysis
    type: parallel
    branches:
      - id: security_check
        type: agent
        role: security_analyst
        goal: "Check security issues"

      - id: quality_check
        type: agent
        role: quality_analyst
        goal: "Check code quality"

      - id: performance_check
        type: agent
        role: performance_analyst
        goal: "Check performance"
    join_strategy: aggregate  # or: all, any, first
    next: [report]
```

### 5. HITL Nodes

**New in 0.5.0**:
```yaml
nodes:
  - id: approval
    type: hitl
    interaction: approval_gate
    config:
      prompt: "Approve this plan?"
      timeout: 300
      default_action: reject
    next:
      approved: execute
      rejected: revise
```

---

## Edge and Routing Changes

### 1. Simple Edges

**Before (0.5.x)**:
```python
graph.add_edge("node1", "node2")
```

**After (0.5.0)**:
```yaml
nodes:
  - id: node1
    type: agent
    goal: "First step"
    next: [node2]
```

### 2. Conditional Edges

**Before (0.5.x)**:
```python
graph.add_conditional_edge(
    "node1",
    should_continue,
    {
        "continue": "node2",
        "stop": END
    }
)
```

**After (0.5.0)**:
```yaml
nodes:
  - id: node1
    type: agent
    goal: "First step"

  - id: check_continue
    type: condition
    condition: "should_continue"
    branches:
      "continue": node2
      "stop": END
```

### 3. Parallel Edges

**New in 0.5.0**:
```yaml
nodes:
  - id: parallel_step
    type: parallel
    branches:
      - id: branch1
        type: agent
        goal: "Branch 1"
        next: [aggregate]

      - id: branch2
        type: agent
        goal: "Branch 2"
        next: [aggregate]

  - id: aggregate
    type: transform
    handler: aggregate_results
    next: [END]
```

---

## Migration Examples

### Example 1: Simple Sequential Workflow

**Before (0.5.x)**:
```python
from victor.framework.graph import StateGraph

graph = StateGraph()
graph.add_node("research", research_node)
graph.add_node("write", write_node)
graph.add_edge("research", "write")
graph.set_entry_point("research")

compiled = graph.compile()
result = await compiled.invoke({"query": "AI trends"})
```

**After (0.5.0)**:
```yaml
# simple_workflow.yaml
workflows:
  research_workflow:
    nodes:
      - id: research
        type: agent
        role: researcher
        goal: "Research {{query}}"
        next: [write]

      - id: write
        type: agent
        role: writer
        goal: "Write summary"
        next: [END]
```

```python
from victor.framework.workflows import BaseYAMLWorkflowProvider

provider = BaseYAMLWorkflowProvider("simple_workflow.yaml")
compiled = provider.compile_workflow("research_workflow")
result = await compiled.invoke({"query": "AI trends"})
```

### Example 2: Conditional Workflow

**Before (0.5.x)**:
```python
def should_continue(state: WorkflowState) -> str:
    if state.get("confidence", 0) > 0.8:
        return "proceed"
    return "retry"

graph.add_conditional_edge(
    "evaluate",
    should_continue,
    {
        "proceed": "finalize",
        "retry": "research"
    }
)
```

**After (0.5.0)**:
```yaml
# conditional_workflow.yaml
workflows:
  conditional_workflow:
    nodes:
      - id: evaluate
        type: agent
        role: evaluator
        goal: "Evaluate confidence"
        next: [check_confidence]

      - id: check_confidence
        type: condition
        condition: "should_continue"
        inputs:
          confidence: $ctx.confidence
          threshold: 0.8
        branches:
          "high": finalize
          "low": retry
        default: retry

      - id: finalize
        type: agent
        role: writer
        goal: "Finalize result"
        next: [END]

      - id: retry
        type: agent
        role: researcher
        goal: "Research more"
        next: [evaluate]
```

### Example 3: Parallel Workflow

**Before (0.5.x)**:
```python
# Complex parallel execution required manual coordination
# No built-in support in 0.5.x
```

**After (0.5.0)**:
```yaml
# parallel_workflow.yaml
workflows:
  parallel_review:
    nodes:
      - id: parallel_review
        type: parallel
        branches:
          - id: security
            type: agent
            role: security_expert
            goal: "Security review"
            next: [aggregate]

          - id: quality
            type: agent
            role: quality_expert
            goal: "Quality review"
            next: [aggregate]

          - id: performance
            type: agent
            role: performance_expert
            goal: "Performance review"
            next: [aggregate]

      - id: aggregate
        type: transform
        handler: aggregate_reviews
        next: [report]

      - id: report
        type: agent
        role: reporter
        goal: "Generate report"
        next: [END]
```

---

## New Workflow Features

### 1. Checkpointing

**New in 0.5.0**:
```python
# Invoke with checkpointing
result = await compiled.invoke(
    context,
    thread_id="user-123",
    checkpoint_ns="workflow-1"
)

# Resume from checkpoint
result = await compiled.invoke(
    updated_context,
    thread_id="user-123",
    checkpoint_ns="workflow-1"
)
```

### 2. Streaming

**New in 0.5.0**:
```python
# Stream workflow execution
async for event in compiled.stream(context):
    if event.type == "node_complete":
        print(f"Node {event.node_id} completed")
    elif event.type == "tool_call":
        print(f"Tool {event.tool} called")
```

### 3. Error Recovery

**New in 0.5.0**:
```yaml
nodes:
  - id: risky_operation
    type: agent
    goal: "Perform risky operation"
    retry_on_failure: true
    max_retries: 3
    retry_backoff: exponential
    on_failure: error_handler  # Node to run on failure
    next: [next_step]

  - id: error_handler
    type: agent
    role: error_handler
    goal: "Handle error gracefully"
    next: [END]
```

### 4. Workflow Metadata

**New in 0.5.0**:
```yaml
workflows:
  my_workflow:
    metadata:
      version: "0.5.0"
      author: "Team"
      description: "Does something great"
      tags: [research, writing]
    nodes: [...]
```

---

## Migration Script

Use the automated migration script:

```bash
# Migrate workflows
python scripts/migrate_workflows.py \
    --source ./old_workflows \
    --dest ./victor/workflows \
    --format yaml

# Validate migrated workflows
victor workflow validate victor/workflows/my_workflow.yaml

# Test workflow execution
victor workflow execute victor/workflows/my_workflow.yaml --dry-run
```

---

## Validation

Validate your workflows after migration:

```bash
# Validate all workflows in directory
victor workflow validate victor/workflows/

# Validate specific workflow
victor workflow validate victor/workflows/my_workflow.yaml

# Check for deprecated patterns
victor-check-deprecated-workflows victor/workflows/
```

---

## Troubleshooting

### Issue: Workflow Not Found

**Symptom**: `ValueError: Workflow 'my_workflow' not found`

**Solution**:
```yaml
# Ensure workflows key exists
workflows:  # Required root key
  my_workflow:  # Workflow name
    nodes: [...]
```

### Issue: Node Validation Failed

**Symptom**: `ValidationError: Invalid node type`

**Solution**:
```yaml
# Use valid node types
nodes:
  - id: node1
    type: agent  # or: compute, condition, parallel, transform, hitl
    ...
```

### Issue: Circular Dependency

**Symptom**: `CircularDependencyError: Workflow has circular edges`

**Solution**:
```yaml
# Ensure no circular paths (unless intentional)
nodes:
  - id: node1
    next: [node2]

  - id: node2
    next: [node3]  # Not back to node1

  - id: node3
    next: [END]
```

---

## Additional Resources

- [Main Migration Guide](./MIGRATION_GUIDE.md)
- [API Migration Guide](./MIGRATION_API.md)
- [Workflow Reference](../reference/internals/workflows-api.md)
- [YAML Workflow Guide](../guides/workflow-development/dsl.md)

---

**Last Updated**: 2025-01-21
**Version**: 0.5.0
