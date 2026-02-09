# Workflow DSL Guide - Part 1

**Part 1 of 2:** Overview through Multi-Agent Workflows

---

## Navigation

- **[Part 1: Fundamentals](#)** (Current)
- [Part 2: Advanced Patterns](part-2-advanced-patterns.md)
- [**Complete Guide](../dsl.md)**

---
# StateGraph DSL Guide

> Build stateful, cyclic agent workflows with Victor's LangGraph-compatible StateGraph DSL

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Building Workflows](#building-workflows)
- [Conditional Branching](#conditional-branching)
- [Checkpointing](#checkpointing)
- [Streaming](#streaming)
- [Integration with Tools](#integration-with-tools)
- [Multi-Agent Workflows](#multi-agent-workflows)
- [Advanced Patterns](#advanced-patterns)
- [API Reference](#api-reference)

## Overview

The StateGraph DSL provides a LangGraph-compatible API for building complex,
  stateful agent workflows. Unlike simple linear pipelines, StateGraph supports:

- **Typed State**: TypedDict-based state schemas for type safety
- **Conditional Edges**: Branch execution based on state conditions
- **Cycles**: Support for retry loops and iterative refinement
- **Checkpointing**: Resume workflows from saved state
- **Streaming**: Stream intermediate states during execution

**Source File**: `victor/framework/graph.py`

## Quick Start

```python
from victor.framework import StateGraph, END
from typing import TypedDict

# 1. Define your state
class TaskState(TypedDict):
    task: str
    result: str
    attempts: int

# 2. Define node functions
async def process(state: TaskState) -> TaskState:
    state["result"] = f"Processed: {state['task']}"
    state["attempts"] += 1
    return state

async def validate(state: TaskState) -> TaskState:
    # Validation logic here
    return state

# 3. Build the graph
graph = StateGraph(TaskState)
graph.add_node("process", process)
graph.add_node("validate", validate)
graph.add_edge("process", "validate")
graph.add_edge("validate", END)
graph.set_entry_point("process")

# 4. Compile and run
app = graph.compile()
result = await app.invoke({"task": "Hello", "result": "", "attempts": 0})
print(result.state)  # {"task": "Hello", "result": "Processed: Hello", "attempts": 1}
```

## Core Concepts

### State

State is a TypedDict that flows through the graph. Each node receives the current state and returns an updated state.

```python
from typing import TypedDict, Optional, List

class WorkflowState(TypedDict):
    # Input
    input: str
    files: List[str]

    # Intermediate
    analysis: Optional[dict]

    # Output
    result: Optional[str]

    # Control
    iteration: int
    error: Optional[str]
```

**State Rules**:
- State must be a TypedDict
- All fields should have default values or be Optional
- Nodes should not delete state keys, only update values
- Use immutable updates when possible

### Nodes

Nodes are async or sync functions that process state.

```python
# Async node (recommended for I/O operations)
async def fetch_data(state: WorkflowState) -> WorkflowState:
    data = await http_client.get(state["url"])
    state["data"] = data
    return state

# Sync node (for pure computation)
def analyze_data(state: WorkflowState) -> WorkflowState:
    state["analysis"] = compute_metrics(state["data"])
    return state

# Node with external dependencies
async def call_agent(state: WorkflowState) -> WorkflowState:
    agent = state.get("_agent")
    result = await agent.run(state["prompt"])
    state["response"] = result.content
    return state
```

### Edges

Edges define transitions between nodes.

```python
# Simple edge: always go from A to B
graph.add_edge("analyze", "execute")

# Edge to END: terminate the workflow
graph.add_edge("finalize", END)

# Conditional edge: choose based on state
graph.add_conditional_edge(
    "validate",           # Source node
    check_condition,      # Condition function
    {                     # Branch mapping
        "pass": "commit",
        "fail": "retry",
        "error": END,
    }
)
```

### Entry Point

Every graph must have an entry point.

```python
graph.set_entry_point("first_node")
```

## Building Workflows

### Linear Workflow

```python
graph = StateGraph(TaskState)
graph.add_node("step1", step1_fn)
graph.add_node("step2", step2_fn)
graph.add_node("step3", step3_fn)

graph.add_edge("step1", "step2")
graph.add_edge("step2", "step3")
graph.add_edge("step3", END)

graph.set_entry_point("step1")
```

### Branching Workflow

```python
def route_by_type(state: TaskState) -> str:
    if state["type"] == "bug":
        return "fix_bug"
    elif state["type"] == "feature":
        return "implement_feature"
    else:
        return "clarify"

graph = StateGraph(TaskState)
graph.add_node("classify", classify_task)
graph.add_node("fix_bug", fix_bug)
graph.add_node("implement_feature", implement_feature)
graph.add_node("clarify", clarify_task)
graph.add_node("finalize", finalize)

graph.add_conditional_edge(
    "classify",
    route_by_type,
    {"fix_bug": "fix_bug", "implement_feature": "implement_feature", "clarify": "clarify"}
)
graph.add_edge("fix_bug", "finalize")
graph.add_edge("implement_feature", "finalize")
graph.add_edge("clarify", "classify")  # Loop back for clarification
graph.add_edge("finalize", END)

graph.set_entry_point("classify")
```

### Retry Loop

```python
MAX_RETRIES = 3

def should_retry(state: TaskState) -> str:
    if state["success"]:
        return "done"
    elif state["attempts"] >= MAX_RETRIES:
        return "fail"
    else:
        return "retry"

graph = StateGraph(TaskState)
graph.add_node("attempt", attempt_task)
graph.add_node("success", handle_success)
graph.add_node("failure", handle_failure)

graph.add_conditional_edge(
    "attempt",
    should_retry,
    {"done": "success", "fail": "failure", "retry": "attempt"}  # Cycle!
)
graph.add_edge("success", END)
graph.add_edge("failure", END)

graph.set_entry_point("attempt")

# Compile with cycle limit
app = graph.compile(max_iterations=10)
```

## Conditional Branching

### Condition Functions

Condition functions receive state and return a string matching a branch name.

```python
def route_decision(state: TaskState) -> str:
    """Return one of the branch names."""
    if not state.get("input"):
        return "error"
    if state["confidence"] > 0.9:
        return "high_confidence"
    elif state["confidence"] > 0.5:
        return "medium_confidence"
    else:
        return "low_confidence"

graph.add_conditional_edge(
    "analyze",
    route_decision,
    {
        "high_confidence": "execute",
        "medium_confidence": "review",
        "low_confidence": "retry",
        "error": END,
    }
)
```

### Multi-Path Convergence

```python
# Multiple paths converge to a single node
graph.add_edge("path_a", "converge")
graph.add_edge("path_b", "converge")
graph.add_edge("path_c", "converge")
graph.add_edge("converge", END)
```

## Checkpointing

Checkpointing allows you to save and resume workflow state.

### Memory Checkpointer

```python
from victor.framework.graph import MemoryCheckpointer

# In-memory checkpointing (development/testing)
checkpointer = MemoryCheckpointer()
app = graph.compile(checkpointer=checkpointer)

# Run workflow with a stable thread_id
thread_id = "example-run-1"
result = await app.invoke(initial_state, thread_id=thread_id)

# Inspect latest checkpoint
checkpoint = await checkpointer.load(thread_id)
resumed = await app.invoke(initial_state, thread_id=thread_id)
```

### RL System Integration

```python
from victor.framework.graph import RLCheckpointerAdapter

# Use Victor's RL checkpoint store for persistence
checkpointer = RLCheckpointerAdapter(learner_name="my_workflow")
app = graph.compile(checkpointer=checkpointer)

# Workflow state persisted across sessions
result = await app.invoke(initial_state)

# Events emitted to RL system for learning
# See victor/agent/rl/hooks.py
```

### Custom Checkpointer

```python
from victor.framework.graph import CheckpointerProtocol, WorkflowCheckpoint

class RedisCheckpointer(CheckpointerProtocol):
    def __init__(self, redis_client):
        self.redis = redis_client

    async def save(self, checkpoint: WorkflowCheckpoint) -> None:
        key = f"workflow:{checkpoint.workflow_id}:{checkpoint.step}"
        await self.redis.set(key, checkpoint.to_json())

    async def load(self, thread_id: str) -> WorkflowCheckpoint | None:
        data = await self.redis.get(thread_id)
        return WorkflowCheckpoint.from_json(data) if data else None

    async def list(self, thread_id: str) -> list[WorkflowCheckpoint]:
        keys = await self.redis.keys(f"workflow:{thread_id}:*")
        return [await self.load(k) for k in keys if k]
```

## Streaming

Stream intermediate states during execution.

```python
async for event in app.stream(initial_state):
    match event.type:
        case "node_start":
            print(f"Starting node: {event.node_id}")
        case "node_complete":
            print(f"Completed node: {event.node_id}")
            print(f"State: {event.state}")
        case "edge_taken":
            print(f"Edge: {event.source} -> {event.target}")
        case "checkpoint":
            print(f"Checkpoint saved: {event.checkpoint_id}")
        case "complete":
            print(f"Final state: {event.state}")
        case "error":
            print(f"Error: {event.error}")
```

## Integration with Tools

### Passing Agent to Nodes

```python
from victor.framework import Agent

class ToolState(TypedDict):
    prompt: str
    result: str
    _agent: Any  # Hidden from serialization

async def use_tools(state: ToolState) -> ToolState:
    agent = state["_agent"]
    result = await agent.run(state["prompt"])
    state["result"] = result.content
    return state

# Create graph
graph = StateGraph(ToolState)
graph.add_node("process", use_tools)
graph.add_edge("process", END)
graph.set_entry_point("process")

# Inject agent into state
agent = await Agent.create()
app = graph.compile()
result = await app.invoke({
    "prompt": "Analyze this code",
    "result": "",
    "_agent": agent,
})
```

### Tool-Specific Nodes

```python
async def search_codebase(state: WorkflowState) -> WorkflowState:
    agent = state["_agent"]
    result = await agent.run(f"Search for: {state['query']}")
    state["search_results"] = parse_search(result.content)
    return state

async def review_code(state: WorkflowState) -> WorkflowState:
    agent = state["_agent"]
    for file in state["files"]:
        result = await agent.run(f"Review {file} for security issues")
        state["reviews"][file] = result.content
    return state

async def apply_fixes(state: WorkflowState) -> WorkflowState:
    agent = state["_agent"]
    for file, issues in state["reviews"].items():
        if issues:
            await agent.run(f"Fix issues in {file}: {issues}")
    return state
```

## Multi-Agent Workflows

### Teams with StateGraph

```python
from victor.framework import StateGraph, END
from victor.framework.teams import TeamMemberSpec, TeamFormation, AgentTeam

class TeamWorkflowState(TypedDict):
    task: str
    research: str
    plan: str
    implementation: str
    review: str

async def run_research_team(state: TeamWorkflowState) -> TeamWorkflowState:
    """Use a research team for initial analysis."""
    team = await AgentTeam.create(
        name="Research",
        goal=f"Research: {state['task']}",
        members=[
            TeamMemberSpec(role="researcher", goal="Find patterns"),
            TeamMemberSpec(role="analyst", goal="Analyze findings"),
        ],
        formation=TeamFormation.SEQUENTIAL,
    )
    result = await team.run()
    state["research"] = result.final_output
    return state

async def run_implementation_team(state: TeamWorkflowState) -> TeamWorkflowState:
    """Use an implementation team for coding."""
    team = await AgentTeam.create(
        name="Implementation",
        goal=f"Implement based on: {state['plan']}",
        members=[
            TeamMemberSpec(role="coder", goal="Write code", tool_budget=30),
            TeamMemberSpec(role="tester", goal="Write tests", tool_budget=20),
        ],
        formation=TeamFormation.PARALLEL,
    )
    result = await team.run()
    state["implementation"] = result.final_output
    return state

# Build workflow
graph = StateGraph(TeamWorkflowState)
graph.add_node("research", run_research_team)
graph.add_node("plan", create_plan)
graph.add_node("implement", run_implementation_team)
graph.add_node("review", code_review)

graph.add_edge("research", "plan")
graph.add_edge("plan", "implement")
graph.add_edge("implement", "review")
graph.add_conditional_edge(
    "review",
    lambda s: "done" if s["review"] == "approved" else "revise",
    {"done": END, "revise": "implement"}
)
graph.set_entry_point("research")
```

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 6 min
**Last Updated:** February 08, 2026**
