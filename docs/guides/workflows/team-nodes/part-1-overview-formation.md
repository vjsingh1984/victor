# Team Nodes Guide - Part 1

**Part 1 of 4:** Overview, Quick Start, YAML Syntax, and Team Formation Types

---

## Navigation

- **[Part 1: Overview & Formation](#)** (Current)
- [Part 2: Recursion & Configuration](part-2-recursion-configuration.md)
- [Part 3: Best Practices & Error Handling](part-3-best-practices-errors.md)
- [Part 4: Complete Examples](part-4-complete-examples.md)
- [**Complete Guide**](../team_nodes.md)

---

# Team Nodes in YAML Workflows

> Execute multi-agent teams within workflow graphs using YAML configuration

## Table of Contents

- [Overview](#overview)
- [When to Use Team Nodes](#when-to-use-team-nodes)
- [Quick Start](#quick-start)
- [YAML Syntax and Configuration](#yaml-syntax-and-configuration)
- [Team Formation Types](#team-formation-types)
- [Recursion Depth Tracking](#recursion-depth-tracking)
- [Member Configuration](#member-configuration)
- [Configuration Examples](#configuration-examples)
- [Best Practices](#best-practices)
- [Error Handling](#error-handling)
- [Complete Examples](#complete-examples)

## Overview

Team nodes enable **hybrid orchestration** by spawning ad-hoc multi-agent teams within workflow graphs. This combines
  the declarative power of workflows with collaborative problem-solving of specialized agents.

### What are Team Nodes?

Team nodes are a special node type in YAML workflows that create **temporary,
  goal-oriented multi-agent teams** as part of workflow execution. Unlike predefined team specifications,
  team nodes are configured directly in workflow YAML and have access to the workflow's shared context.

**Key Features**:

- **5 Team Formations**: Sequential, Parallel, Pipeline, Hierarchical, Consensus
- **Recursion Control**: Unified depth tracking prevents infinite nesting (default: 3 levels)
- **Flexible Configuration**: YAML-first with optional Python customization
- **Context Integration**: Teams inherit workflow context and merge results back
- **Error Resilience**: Continue-on-error and timeout handling
- **State Merging**: Configurable strategies for combining team results with workflow state

### Architecture

```text
Workflow Graph (YAML)
    └── Team Node
        ├── Member 1 (Researcher)
        ├── Member 2 (Executor)
        └── Member 3 (Reviewer)
            └── UnifiedTeamCoordinator
                └── RecursionContext (depth tracking)
                    └── Results merged back to workflow state
```

### Team Nodes vs Agent Nodes

| Feature | Team Nodes | Agent Nodes |
|---------|-----------|-------------|
| **Execution** | Multiple agents collaborating | Single agent working |
| **Best For** | Multi-perspective tasks | Single-responsibility tasks |
| **Configuration** | Team formation + members | Role + goal |
| **Coordination** | Automatic (based on formation) | N/A (single agent) |
| **Complexity** | Higher overhead | Lower overhead |
| **Use Cases** | Code review, feature implementation | Analysis, transformations |

## When to Use Team Nodes

Use team nodes when a task requires **multiple perspectives** or **specialized expertise**:

### Ideal Use Cases

- **Code Review**: Security + Quality + Performance reviewers in parallel
- **Feature Implementation**: Researcher → Architect → Developer → Tester pipeline
- **Complex Debugging**: Parallel investigation with synthesis
- **Documentation**: Researcher → Writer → Reviewer pipeline
- **Data Analysis**: Parallel analysis with aggregation
- **Design Decisions**: Multiple architects reaching consensus

### When NOT to Use

- **Simple Tasks**: Single agent is sufficient and faster
- **Linear Processes**: Regular agent nodes are simpler
- **Low Latency Requirements**: Teams add coordination overhead
- **Resource Constraints**: Teams require more tool budget and compute

## Quick Start

### Minimal Team Node

```yaml
workflows:
  simple_review:
    nodes:
      - id: review_team
        type: team
        goal: "Review the pull request"
        team_formation: parallel
        members:
          - id: security_reviewer
            role: reviewer
            goal: "Check for security vulnerabilities"
          - id: quality_reviewer
            role: reviewer
            goal: "Check code quality"
        next: [summarize]
```text

### Execution

```bash
# Run workflow with team node
victor workflow run simple_review

# Programmatic execution
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

compiler = UnifiedWorkflowCompiler()
compiled = compiler.compile_yaml("workflow.yaml", "simple_review")
result = await compiled.invoke({"pr_number": 123})
```

## YAML Syntax and Configuration

### Basic Structure

```yaml
nodes:
  - id: my_team
    type: team
    name: "My Multi-Agent Team"
    goal: "Overall team objective"
    team_formation: sequential
    timeout_seconds: 300
    max_iterations: 50
    total_tool_budget: 100
    output_key: team_result
    continue_on_error: true
    merge_strategy: dict
    merge_mode: team_wins
    members:
      - id: member_1
        role: researcher
        goal: "Member-specific goal"
        tool_budget: 25
        tools: [read, grep]
        backstory: "Experienced researcher"
        expertise: [analysis, research]
        personality: "thorough"
      - id: member_2
        role: executor
        goal: "Implementation goal"
        tool_budget: 50
        tools: [read, write]
    next: [next_node]
```text

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique node identifier within the workflow |
| `type` | string | Must be `"team"` |
| `goal` | string | Overall team objective (supports template variables) |
| `members` | list | List of member configurations (minimum 1) |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `"Team: {id}"` | Human-readable name for logging |
| `team_formation` | string | `"sequential"` | Team organization pattern |
| `timeout_seconds` | number | `None` (no limit) | Maximum execution time before timeout |
| `max_iterations` | number | `50` | Maximum team iterations (prevents infinite loops) |
| `total_tool_budget` | number | `100` | Total tool call budget across all members |
| `output_key` | string | `"team_result"` | Context key to store result |
| `continue_on_error` | boolean | `true` | Continue workflow execution if team fails |
| `merge_strategy` | string | `"dict"` | How to merge team state (`dict`, `list`, `selective`, `custom`) |
| `merge_mode` | string | `"team_wins"` | Conflict resolution (`team_wins`, `graph_wins`, `merge`, `error`) |
| `next` | list | `[]` | Next node IDs to execute after team completes |

### Template Variables in Goals

Team node goals support template variables from workflow context:

```yaml
nodes:
  - id: gather_context
    type: compute
    output: context_info

  - id: analysis_team
    type: team
    goal: |
      Analyze {{feature_name}}
      Context: {{context_info}}
      Requirements: {{requirements}}
    # Variables are substituted from context
```

## Team Formation Types

Victor supports **5 team formation patterns** for different collaboration models.

### 1. Sequential

Members execute **one after another**, with each member receiving the previous member's output.

```yaml
nodes:
  - id: sequential_team
    type: team
    goal: "Research then implement"
    team_formation: sequential
    members:
      - id: researcher
        role: researcher
        goal: "Research existing patterns"
        tool_budget: 15
      - id: implementer
        role: executor
        goal: "Implement based on research"
        tool_budget: 35
```text

**Use when**: Tasks have clear dependent stages where output feeds into input.

**Flow**: Researcher → [output] → Implementer → [output] → Next Node

**Characteristics**:
- Context chaining between members
- Simple to debug (linear flow)
- Slower than parallel (no concurrency)

### 2. Parallel

All members work **simultaneously** on the same task with shared context.

```yaml
nodes:
  - id: parallel_review
    type: team
    goal: "Comprehensive code review"
    team_formation: parallel
    members:
      - id: security_reviewer
        role: reviewer
        goal: "Check for security vulnerabilities"
        tools: [read, grep]
        expertise: [security]
      - id: quality_reviewer
        role: reviewer
        goal: "Check code quality and maintainability"
        tools: [read, grep]
        expertise: [quality]
      - id: performance_reviewer
        role: reviewer
        goal: "Identify performance bottlenecks"
        tools: [read, grep]
        expertise: [performance]
```

**Use when**: Multiple independent perspectives needed simultaneously.

**Flow**: All members execute concurrently → [aggregate results] → Next Node

**Characteristics**:
- Fastest execution (full concurrency)
- Independent work with shared starting context
- Results aggregated at end

### 3. Pipeline

Output of each member **feeds into the next** with explicit handoff messages.

```yaml
nodes:
  - id: documentation_pipeline
    type: team
    goal: "Create comprehensive documentation"
    team_formation: pipeline
    members:
      - id: researcher
        role: researcher
        goal: "Gather information from codebase"
        tools: [read, grep, overview]
      - id: writer
        role: writer
        goal: "Draft documentation from research"
        tools: [read]
      - id: reviewer
        role: reviewer
        goal: "Review and refine documentation"
        tools: [read]
```text

**Use when**: Processing pipeline with clear stages and handoff requirements.

**Flow**: Researcher → [handoff message] → Writer → [handoff message] → Reviewer → Next Node

**Characteristics**:
- Explicit handoff messages between stages
- Each stage can reject/revise previous work
- Good for review processes with feedback loops

### 4. Hierarchical

A **manager** delegates to workers, then synthesizes their results.

```yaml
nodes:
  - id: feature_team
    type: team
    goal: "Implement complex feature"
    team_formation: hierarchical
    members:
      - id: tech_lead
        role: planner
        goal: "Plan and delegate implementation"
        tool_budget: 20
        # Manager identified by role=planner
      - id: backend_dev
        role: executor
        goal: "Implement backend components"
        tools: [read, write]
      - id: frontend_dev
        role: executor
        goal: "Implement frontend components"
        tools: [read, write]
      - id: qa_tester
        role: reviewer
        goal: "Test and validate implementation"
        tools: [read, grep]
```

**Use when**: Complex tasks requiring planning, delegation, and synthesis.

**Flow**: Manager plans → [delegate tasks] → Workers execute → [report results] → Manager synthesizes → Next Node

**Characteristics**:
- Manager (role=planner) coordinates
- Workers execute in parallel
- Manager aggregates and synthesizes
- Best for complex, multi-component tasks

### 5. Consensus

All members must **agree** on the outcome, requiring multiple rounds if needed.

```yaml
nodes:
  - id: design_review
    type: team
    goal: "Review and approve system design"
    team_formation: consensus
    max_iterations: 30
    members:
      - id: architect_1
        role: planner
        goal: "Evaluate design from scalability perspective"
        expertise: [architecture, scalability]
      - id: architect_2
        role: planner
        goal: "Evaluate design from security perspective"
        expertise: [architecture, security]
      - id: architect_3
        role: planner
        goal: "Evaluate design from maintainability perspective"
        expertise: [architecture, maintainability]
```text

**Use when**: Critical decisions requiring unanimous agreement.

**Flow**: Members propose → [discuss] → [revise] → [vote] → Consensus or repeat

**Characteristics**:
- Multiple rounds until consensus
- Can be time-intensive
- Best for critical decisions requiring alignment

### Formation Comparison

| Formation | Speed | Coordination | Best For |
|-----------|-------|--------------|----------|
| **Sequential** | Slow | Low | Simple dependent tasks |
| **Parallel** | Fast | Medium | Independent reviews |
| **Pipeline** | Medium | High | Stage-gate processes |
| **Hierarchical** | Medium | High | Complex planning |
| **Consensus** | Slow | Very High | Critical decisions |

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 7 min
**Last Updated:** February 08, 2026**
