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

Team nodes enable **hybrid orchestration** by spawning ad-hoc multi-agent teams within workflow graphs. This combines the declarative power of workflows with collaborative problem-solving of specialized agents.

### What are Team Nodes?

Team nodes are a special node type in YAML workflows that create **temporary, goal-oriented multi-agent teams** as part of workflow execution. Unlike predefined team specifications, team nodes are configured directly in workflow YAML and have access to the workflow's shared context.

**Key Features**:

- **5 Team Formations**: Sequential, Parallel, Pipeline, Hierarchical, Consensus
- **Recursion Control**: Unified depth tracking prevents infinite nesting (default: 3 levels)
- **Flexible Configuration**: YAML-first with optional Python customization
- **Context Integration**: Teams inherit workflow context and merge results back
- **Error Resilience**: Continue-on-error and timeout handling
- **State Merging**: Configurable strategies for combining team results with workflow state

### Architecture

```
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
```

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
```

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
```

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
```

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
```

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

## Recursion Depth Tracking

Team nodes participate in **unified recursion tracking** to prevent infinite nesting. All nested execution types count toward the same limit:
- Workflow invoking workflow
- Workflow spawning team
- Team spawning team
- Team spawning workflow

### Default Limits

- **Default max depth**: 3 levels
- **Tracking scope**: Per workflow execution
- **Thread-safe**: Uses reentrant locks for concurrent access

### Configuration

#### Method 1: YAML Metadata

```yaml
workflows:
  my_workflow:
    metadata:
      max_recursion_depth: 5

    nodes:
      - id: outer_team
        type: team
        goal: "Team that may spawn nested workflows"
```

#### Method 2: Execution Settings

```yaml
workflows:
  my_workflow:
    execution:
      max_recursion_depth: 5

    nodes:
      - id: team_node
        type: team
        goal: "Team execution"
```

#### Method 3: Runtime Override

```python
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

compiler = UnifiedWorkflowCompiler()

# Override recursion depth at runtime
context = {
    "max_recursion_depth": 5,  # Overrides YAML default
    "task": "Analyze system"
}

result = await compiler.invoke(context)
```

### Recursion Depth Example

```yaml
# Level 0: Main workflow
workflows:
  main:
    metadata:
      max_recursion_depth: 3

    nodes:
      - id: team_1
        type: team
        goal: "First team (depth 1)"
        members:
          - id: member
            role: executor
            goal: "May trigger workflow"
        # If member spawns workflow:
        # → Level 1: Workflow
        #   → Level 2: Team within workflow
        #     → Level 3: MAX REACHED (would fail)
```

### Execution Stack Example

```
Current Depth: 3 / 3
Execution Stack:
  1. workflow:main
  2. team:analysis_team
  3. workflow:nested_analysis

Attempting to enter team:inner_review_team
ERROR: RecursionDepthError: Maximum recursion depth (3) exceeded
```

### RecursionContext API

The underlying `RecursionContext` class provides:

```python
from victor.workflows.recursion import RecursionContext, RecursionGuard

ctx = RecursionContext(max_depth=3)

# Manual tracking
ctx.enter("workflow", "my_workflow")
ctx.enter("team", "research_team")
print(ctx.current_depth)  # 2
print(ctx.execution_stack)  # ['workflow:my_workflow', 'team:research_team']

# Check if nesting is possible
if ctx.can_nest(2):
    # Can go 2 levels deeper
    pass

# Get depth info
info = ctx.get_depth_info()
# {'current_depth': 2, 'max_depth': 3, 'remaining_depth': 1, ...}

# Automatic cleanup with context manager
with RecursionGuard(ctx, "workflow", "nested"):
    # Nested execution here
    pass  # Automatically exits
```

## Member Configuration

Each team member can be configured with rich properties to guide their behavior.

### Required Member Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique member identifier within the team |
| `goal` | string | Member's specific objective |

### Optional Member Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `role` | string | `"assistant"` | Agent role: `researcher`, `executor`, `planner`, `reviewer`, `assistant` |
| `tool_budget` | number | `25` | Tool call budget for this member |
| `tools` | list | `[]` | Allowed tool names (empty = all tools) |
| `backstory` | string | `None` | Member's background and context |
| `expertise` | list | `[]` | List of expertise areas |
| `personality` | string | `None` | Communication style |

### Member Roles

| Role | Best For | Tool Budget Suggestion |
|------|----------|----------------------|
| `researcher` | Information gathering, analysis | 15-30 |
| `executor` | Implementation, file operations | 40-80 |
| `planner` | Architecture, planning, coordination | 20-40 |
| `reviewer` | Quality checks, validation | 20-35 |
| `assistant` | General assistance | 20-30 |

### Expertise Tags

Expertise tags help guide member behavior and can influence team coordination:

```yaml
members:
  - id: security_reviewer
    role: reviewer
    goal: "Security analysis"
    expertise: [security, authentication, authorization, cryptography]
    # Common expertise areas:
    # - security, performance, quality, testing
    # - architecture, design-patterns, scalability
    # - backend, frontend, devops
    # - debugging, optimization, refactoring
```

### Backstory Templates

Backstories provide context that shapes member behavior:

```yaml
members:
  - id: architect
    role: planner
    backstory: |
      Senior software architect with 15 years of experience
      designing large-scale distributed systems. Expert in
      microservices architecture and cloud-native patterns.
      Known for creating scalable, maintainable solutions.

  - id: security_specialist
    role: reviewer
    backstory: |
      Security engineer with OWASP certification. Has conducted
      hundreds of security audits and penetration tests. Passionate
      about secure coding practices and threat modeling.
```

## Configuration Examples

### Example 1: Simple Parallel Review

```yaml
workflows:
  simple_review:
    description: "Simple code review with 2 reviewers"

    nodes:
      - id: review_team
        type: team
        name: "Code Review Team"
        goal: "Review the pull request for bugs and style issues"
        team_formation: parallel
        timeout_seconds: 300
        total_tool_budget: 50
        output_key: review_result
        members:
          - id: reviewer_1
            role: reviewer
            goal: "Check for bugs and logic errors"
            tool_budget: 25
            tools: [read, grep]

          - id: reviewer_2
            role: reviewer
            goal: "Check for style and best practices"
            tool_budget: 25
            tools: [read, grep]
        next: [summarize]

      - id: summarize
        type: agent
        role: writer
        goal: "Summarize review results: {{review_result}}"
        tool_budget: 10
```

### Example 2: Feature Implementation Pipeline

```yaml
workflows:
  feature_impl:
    description: "Feature implementation with pipeline formation"

    nodes:
      - id: feature_team
        type: team
        name: "Feature Implementation Team"
        goal: |
          Implement the feature: {{feature_name}}

          Requirements: {{requirements}}
          Context: {{project_context}}
        team_formation: pipeline
        timeout_seconds: 600
        max_iterations: 75
        total_tool_budget: 150
        output_key: feature_result
        continue_on_error: false
        members:
          - id: architect
            role: planner
            goal: |
              Design the solution:
              1. Analyze requirements
              2. Create architecture plan
              3. Identify components
              4. Plan integration points
            tool_budget: 30
            tools: [read, grep, overview]
            backstory: |
              Senior architect with 15 years experience
              in large-scale system design.
            expertise: [architecture, design-patterns, scalability]

          - id: researcher
            role: researcher
            goal: |
              Research codebase to find:
              1. Similar implementations
              2. Existing patterns
              3. Integration points
              4. Dependencies
            tool_budget: 25
            tools: [read, grep, code_search, overview]

          - id: developer
            role: executor
            goal: |
              Implement the solution:
              1. Follow architecture plan
              2. Write clean code
              3. Add tests
              4. Document changes
            tool_budget: 65
            tools: [read, write, grep]

          - id: tester
            role: reviewer
            goal: |
              Test and validate:
              1. Run test suite
              2. Check edge cases
              3. Validate requirements
              4. Report issues
            tool_budget: 30
            tools: [read, grep]
        next: [final_report]
```

### Example 3: Hierarchical Analysis

```yaml
workflows:
  complex_analysis:
    description: "Complex analysis with hierarchical team"

    metadata:
      max_recursion_depth: 4

    nodes:
      - id: analysis_team
        type: team
        name: "Analysis Team"
        goal: "Perform comprehensive analysis across multiple domains"
        team_formation: hierarchical
        total_tool_budget: 200
        timeout_seconds: 900
        members:
          - id: coordinator
            role: planner
            goal: |
              Coordinate analysis across domains:
              1. Divide analysis into domains
              2. Assign tasks to analysts
              3. Synthesize findings
              4. Create comprehensive report
            tool_budget: 40
            backstory: |
              Technical lead experienced in coordinating
              complex multi-domain analysis projects.

          - id: security_analyst
            role: researcher
            goal: "Security analysis: vulnerabilities, threats, compliance"
            tool_budget: 40
            tools: [read, grep]
            expertise: [security, threat-modeling, compliance]

          - id: performance_analyst
            role: researcher
            goal: "Performance analysis: bottlenecks, optimization opportunities"
            tool_budget: 40
            tools: [read, grep]
            expertise: [performance, profiling, optimization]

          - id: quality_analyst
            role: researcher
            goal: "Quality analysis: code quality, technical debt, maintainability"
            tool_budget: 40
            tools: [read, grep]
            expertise: [code-quality, technical-debt, maintainability]

          - id: architecture_analyst
            role: researcher
            goal: "Architecture analysis: design patterns, scalability, coupling"
            tool_budget: 40
            tools: [read, grep, overview]
            expertise: [architecture, design-patterns, scalability]
        next: [present_results]

      - id: present_results
        type: agent
        role: writer
        goal: "Create presentation of analysis results"
        tool_budget: 20
        tools: [write]
```

## Best Practices

### 1. Choose the Right Formation

Select formation based on task characteristics:

| Task Type | Recommended Formation | Rationale |
|-----------|----------------------|-----------|
| Dependent stages | **Sequential** or **Pipeline** | Output feeds into input |
| Independent perspectives | **Parallel** | Fastest, no dependencies |
| Planning + delegation | **Hierarchical** | Manager coordinates |
| Agreement required | **Consensus** | Unanimous decision needed |
| Unknown/Exploratory | **Sequential** | Simplest to debug |

### 2. Set Realistic Budgets

Match tool budgets to member responsibilities:

```yaml
# Good: Specific budgets per member
members:
  - id: researcher
    tool_budget: 15  # Limited research
  - id: implementer
    tool_budget: 60  # Heavy implementation
  - id: reviewer
    tool_budget: 25  # Moderate review

# Avoid: Equal budgets when work is uneven
members:
  - id: researcher
    tool_budget: 50  # Too much for research
  - id: implementer
    tool_budget: 50  # Not enough for implementation
```

### 3. Define Clear, Actionable Goals

Use structured goals with clear success criteria:

```yaml
# Good: Specific, actionable goal
goal: |
  Implement user authentication with:
  1. Login endpoint at /auth/login
  2. Session management with JWT tokens
  3. Password hashing with bcrypt
  4. Unit tests with >80% coverage
  Success criteria: All tests pass, no security vulnerabilities

# Avoid: Vague goal
goal: "Do authentication stuff"
```

### 4. Use Expertise Fields Effectively

Expertise tags guide member behavior and help with task routing:

```yaml
members:
  - id: security_reviewer
    role: reviewer
    goal: "Security review"
    expertise: [security, authentication, authorization, cryptography]
    # Specific expertise helps with:
    # - Task assignment
    # - Tool selection
    # - Coordination decisions
```

### 5. Handle Timeouts Gracefully

Set appropriate timeouts and decide on failure behavior:

```yaml
nodes:
  - id: team_with_timeout
    type: team
    timeout_seconds: 300  # 5 minutes
    continue_on_error: true  # Don't fail workflow
    # If team times out, workflow continues to next node
    next: [fallback_handler]

  - id: fallback_handler
    type: agent
    goal: "Handle team timeout gracefully"
    # Executes even if team fails or times out
```

### 6. Monitor Recursion Depth

Be conservative with recursion limits to prevent infinite loops:

```yaml
workflows:
  main_workflow:
    metadata:
      max_recursion_depth: 3  # Be conservative

    nodes:
      - id: team_node
        type: team
        # Team members can spawn workflows, but depth is limited
        # Level 0: main_workflow
        # Level 1: team_node
        # Level 2: workflow spawned by team member
        # Level 3: MAX - no further nesting
```

### 7. Leverage Output Keys

Use specific output keys for clear data flow:

```yaml
nodes:
  - id: analysis_team
    type: team
    output_key: security_analysis_results  # Specific, descriptive

  - id: report_generator
    type: agent
    goal: "Generate report from: {{security_analysis_results}}"
    # Clear data flow with descriptive keys

  # Avoid vague keys
  - id: team
    output_key: result  # Too generic
```

### 8. Configure State Merging

Choose merge strategy based on how team results should integrate:

```yaml
nodes:
  - id: analysis_team
    type: team
    merge_strategy: dict  # Merge dictionaries
    merge_mode: team_wins  # Team values override on conflict

    # Or for selective merging
    merge_strategy: selective
    merge_mode: merge  # Combine team and graph values

    # Or for list aggregation
    merge_strategy: list  # Aggregate lists
```

### 9. Optimize Team Size

Balance collaboration benefits with coordination overhead:

```yaml
# Good: Focused team with 3-4 members
members:
  - id: security_reviewer
  - id: quality_reviewer
  - id: performance_reviewer

# Avoid: Too many members (slow, expensive)
members:
  - id: reviewer_1
  - id: reviewer_2
  # ... 10 more reviewers
```

### 10. Use Continue-On-Error Appropriately

Decide whether team failure should stop the workflow:

```yaml
# For critical operations: Stop on failure
- id: critical_team
  type: team
  continue_on_error: false  # Workflow stops if team fails

# For optional operations: Continue on failure
- id: optional_analysis
  type: team
  continue_on_error: true  # Workflow continues even if team fails
  next: [fallback_analysis]
```

## Error Handling

### Common Errors

#### 1. Recursion Depth Exceeded

```
RecursionDepthError: Maximum recursion depth (3) exceeded.
Attempting to enter team:nested_review_team

Execution stack:
  workflow:main → team:review_team → workflow:sub_analysis → team:nested_review_team
```

**Solutions**:
- Increase `max_recursion_depth` in workflow metadata
- Redesign workflow to reduce nesting
- Use parallel formation instead of nested teams
- Flatten workflow structure

#### 2. Team Execution Failed

```
YAMLWorkflowError: Team execution failed: All members failed to complete
```

**Solutions**:
- Check member goals are achievable with given tool budgets
- Increase `timeout_seconds` if team is timing out
- Verify tools are available and accessible
- Check for errors in individual member execution
- Enable logging for detailed diagnostics

#### 3. Invalid Role

```
ValueError: Unknown role 'invalid_role'
Valid roles: researcher, executor, planner, reviewer, assistant
```

**Solutions**:
- Use valid role: `researcher`, `executor`, `planner`, `reviewer`, `assistant`
- Check role spelling in YAML configuration
- Refer to member role documentation

#### 4. Timeout

```
TimeoutError: Team execution exceeded 300 seconds
```

**Solutions**:
- Increase `timeout_seconds` in team configuration
- Reduce scope of member goals
- Reduce team size (fewer members)
- Optimize tool usage (fewer tool calls)
- Use parallel formation for concurrent execution

#### 5. State Merge Conflict

```
StateMergeError: Conflict merging state for key 'result'
Team value: {...}
Graph value: {...}
```

**Solutions**:
- Change `merge_mode` to `team_wins` (team overrides) or `graph_wins` (graph overrides)
- Use `merge` mode to combine values
- Use different output keys to avoid conflicts
- Implement custom merge strategy

### Error Handling Patterns

#### Continue on Error with Fallback

```yaml
nodes:
  - id: risky_team
    type: team
    goal: "High-risk analysis that may fail"
    continue_on_error: true  # Workflow continues even if team fails
    output_key: analysis_result
    next: [fallback_handler]

  - id: fallback_handler
    type: agent
    goal: |
      Handle team failure gracefully:
      {% if analysis_result %}
        Use team result: {{analysis_result}}
      {% else %}
        Provide default analysis
      {% endif %}
    # Executes even if team fails
```

#### Conditional Retry

```yaml
nodes:
  - id: attempt_1
    type: team
    goal: "First attempt with basic team"
    output_key: result_1
    next: [check_success]

  - id: check_success
    type: condition
    condition: "is_successful"
    branches:
      "true": complete
      "false": attempt_2

  - id: attempt_2
    type: team
    goal: "Retry with different formation and members"
    team_formation: parallel  # Different approach
    members:
      # Different team configuration
```

#### Error Recovery Node

```yaml
nodes:
  - id: analysis_team
    type: team
    goal: "Perform analysis"
    continue_on_error: true
    output_key: analysis_result
    next: [validate_result]

  - id: validate_result
    type: condition
    condition: "has_valid_result"
    branches:
      "true": use_result
      "false": error_recovery

  - id: error_recovery
    type: agent
    goal: "Recover from failed team execution"
    tool_budget: 20
    next: [complete]
```

## Complete Examples

### Example: Multi-Stage Code Review Workflow

```yaml
workflows:
  comprehensive_review:
    description: "Multi-stage code review with specialized teams"

    metadata:
      version: "1.0.0"
      vertical: "coding"

    execution:
      max_recursion_depth: 3
      max_timeout_seconds: 1200

    nodes:
      # Stage 1: Automated analysis
      - id: automated_checks
        type: compute
        name: "Automated Checks"
        tools: [shell]
        inputs:
          commands:
            - "ruff check ."
            - "mypy ."
            - "pytest tests/"
        output: auto_results
        next: [initial_review]

      # Stage 2: Initial human review
      - id: initial_review
        type: team
        name: "Initial Review Team"
        goal: |
          Review the code changes:
          {{diff_summary}}

          Automated results: {{auto_results}}

          Provide initial feedback on:
          1. Code quality
          2. Potential bugs
          3. Security issues
          4. Performance concerns
        team_formation: parallel
        timeout_seconds: 300
        total_tool_budget: 75
        output_key: review_feedback
        members:
          - id: security_reviewer
            role: reviewer
            goal: "Check for security vulnerabilities and issues"
            tool_budget: 25
            tools: [read, grep]
            expertise: [security, authentication, authorization]
            backstory: |
              Security engineer with OWASP certification.
              Expert in identifying vulnerabilities and
              secure coding practices.

          - id: quality_reviewer
            role: reviewer
            goal: "Check code quality and maintainability"
            tool_budget: 25
            tools: [read, grep]
            expertise: [code-quality, design-patterns]
            backstory: |
              Senior developer focused on code quality,
              maintainability, and best practices.

          - id: logic_reviewer
            role: reviewer
            goal: "Check logic correctness and potential bugs"
            tool_budget: 25
            tools: [read, grep]
            expertise: [logic, algorithms, debugging]
            backstory: |
              Experienced developer with strong focus on
              logic correctness and edge cases.
        next: [decide_changes]

      # Stage 3: Decide if changes needed
      - id: decide_changes
        type: condition
        name: "Decision Point"
        condition: "needs_changes"
        branches:
          "true": implementation_team
          "false": final_approval

      # Stage 4a: Implementation team (if changes needed)
      - id: implementation_team
        type: team
        name: "Implementation Team"
        goal: |
          Implement requested changes based on feedback:
          {{review_feedback}}
        team_formation: pipeline
        timeout_seconds: 600
        total_tool_budget: 125
        output_key: changes_result
        members:
          - id: planner
            role: planner
            goal: |
              Plan the implementation approach:
              1. Review all feedback
              2. Prioritize changes
              3. Create implementation plan
              4. Identify risks
            tool_budget: 25
            tools: [read, grep, overview]
            backstory: |
              Technical planner experienced in breaking
              down feedback into actionable implementation plans.

          - id: developer
            role: executor
            goal: |
              Implement the planned changes:
              1. Follow implementation plan
              2. Write clean, tested code
              3. Update documentation
              4. Run tests
            tool_budget: 75
            tools: [read, write, grep, shell]
            backstory: |
              Full-stack developer focused on quality
              implementation and testing.

          - id: verifier
            role: reviewer
            goal: |
              Verify the implementation:
              1. Check all feedback addressed
              2. Run test suite
              3. Validate changes
              4. Report any issues
            tool_budget: 25
            tools: [read, grep, shell]
            backstory: |
              QA specialist with attention to detail
              and passion for quality.
        next: [final_review]

      # Stage 4b: Final approval (if no changes)
      - id: final_approval
        type: agent
        role: planner
        goal: "Approve the changes - no issues found"
        tool_budget: 10
        next: [complete]

      # Stage 5: Final review after changes
      - id: final_review
        type: team
        name: "Final Review Team"
        goal: |
          Final review of implemented changes:
          {{changes_result}}

          Ensure all feedback has been addressed
          and no regressions introduced.
        team_formation: consensus
        timeout_seconds: 300
        total_tool_budget: 75
        output_key: final_result
        members:
          - id: reviewer_1
            role: reviewer
            goal: "Verify changes address all feedback"
            tool_budget: 25
            tools: [read, grep]

          - id: reviewer_2
            role: reviewer
            goal: "Check for regressions and new issues"
            tool_budget: 25
            tools: [read, grep]

          - id: reviewer_3
            role: reviewer
            goal: "Final approval check"
            tool_budget: 25
            tools: [read, grep]
        next: [complete]

      # Stage 6: Complete
      - id: complete
        type: transform
        name: "Mark Complete"
        transform: "status = approved"
        next: []
```

### Example: Research and Implementation Workflow

```yaml
workflows:
  research_and_implement:
    description: "Research new technology and implement POC"

    metadata:
      version: "1.0.0"

    nodes:
      # Phase 1: Research team (parallel)
      - id: research_phase
        type: team
        name: "Research Team"
        goal: |
          Research {{technology}} thoroughly:

          Research Areas:
          1. Core concepts and architecture
          2. Best practices and patterns
          3. Integration options
          4. Potential pitfalls and limitations
          5. Community and ecosystem

          Provide comprehensive findings with examples.
        team_formation: parallel
        timeout_seconds: 900  # 15 minutes
        total_tool_budget: 150
        output_key: research_findings
        members:
          - id: concept_researcher
            role: researcher
            goal: "Research core concepts, architecture, and design principles"
            tool_budget: 50
            tools: [web_search, read]
            backstory: |
              Technology researcher with 10 years experience
              analyzing software architectures and patterns.
            expertise: [research, architecture-analysis, design-patterns]

          - id: best_practices_researcher
            role: researcher
            goal: "Find best practices, patterns, and anti-patterns"
            tool_budget: 50
            tools: [web_search, read]
            backstory: |
              Researcher focused on identifying proven practices
              and common pitfalls in technology adoption.
            expertise: [best-practices, patterns, anti-patterns]

          - id: integration_researcher
            role: researcher
            goal: "Research integration approaches and ecosystem"
            tool_budget: 50
            tools: [web_search, read]
            backstory: |
              Integration specialist with experience in
              system integration and API design.
            expertise: [integration, system-design, apis]
        next: [synthesis]

      # Phase 2: Synthesize research
      - id: synthesis
        type: agent
        role: planner
        goal: |
          Synthesize research findings into comprehensive guide:

          Research Findings: {{research_findings}}

          Create guide covering:
          1. Architecture overview with diagrams
          2. Implementation recommendations
          3. Code examples and patterns
          4. Integration plan with options
          5. Risk assessment and mitigation

          Output should be actionable and clear.
        tool_budget: 30
        tools: [write]
        output: guide
        next: [implementation_team]

      # Phase 3: Implementation team (sequential)
      - id: implementation_team
        type: team
        name: "Implementation Team"
        goal: |
          Implement POC based on research guide:
          {{guide}}

          Create working proof-of-concept demonstrating
          key concepts and integration points.
        team_formation: sequential
        timeout_seconds: 1200  # 20 minutes
        total_tool_budget: 200
        output_key: implementation_result
        members:
          - id: architect
            role: planner
            goal: |
              Design POC architecture:
              1. Define component structure
              2. Plan data flow and interfaces
              3. Identify integration points
              4. Create architecture documentation
            tool_budget: 40
            tools: [write]
            backstory: |
              Software architect specializing in POC design
              and rapid prototyping.
            expertise: [architecture, design, prototyping]

          - id: developer
            role: executor
            goal: |
              Implement core functionality:
              1. Set up project structure
              2. Implement key components
              3. Add error handling
              4. Create sample usage
            tool_budget: 120
            tools: [read, write, grep, shell]
            backstory: |
              Full-stack developer experienced in rapid
              prototyping and POC development.
            expertise: [implementation, debugging, prototyping]

          - id: documenter
            role: writer
            goal: |
              Create documentation:
              1. API documentation
              2. Usage examples
              3. Setup instructions
              4. Integration guide
            tool_budget: 40
            tools: [read, write]
            backstory: |
              Technical writer specializing in developer
              documentation and guides.
            expertise: [documentation, technical-writing]
        next: [testing_team]

      # Phase 4: Testing team (parallel)
      - id: testing_team
        type: team
        name: "Testing Team"
        goal: |
          Test POC implementation comprehensively:
          {{implementation_result}}

          Validate functionality, integration, and robustness.
        team_formation: parallel
        timeout_seconds: 600
        total_tool_budget: 100
        output_key: test_results
        members:
          - id: functional_tester
            role: reviewer
            goal: "Test functional requirements and core features"
            tool_budget: 50
            tools: [read, grep, shell]
            backstory: |
              QA engineer focused on functional testing
              and validation.

          - id: integration_tester
            role: reviewer
            goal: "Test integration points and interfaces"
            tool_budget: 50
            tools: [read, grep, shell]
            backstory: |
              Integration testing specialist with experience
              in API and component integration testing.
        next: [finalize]

      # Phase 5: Finalize
      - id: finalize
        type: agent
        role: writer
        goal: |
          Prepare final report:
          Research: {{research_findings}}
          Guide: {{guide}}
          Implementation: {{implementation_result}}
          Testing: {{test_results}}

          Create comprehensive report including:
          1. Executive summary
          2. Technical deep-dive
          3. Implementation details
          4. Test results and findings
          5. Recommendations

          Save report to file.
        tool_budget: 20
        tools: [write]
        next: []
```

## Additional Resources

### Related Documentation

- [Multi-Agent Teams Guide](../guides/MULTI_AGENT_TEAMS.md) - Team coordination patterns and standalone teams
- [Workflow DSL Guide](../guides/workflow-development/dsl.md) - Python workflow API and StateGraph
- [Workflow Examples](../guides/workflow-development/examples.md) - More workflow examples
- [Workflow User Guide](../user-guide/workflows.md) - General workflow documentation

### API Reference

#### TeamNodeWorkflow

```python
@dataclass
class TeamNodeWorkflow(WorkflowNode):
    """Node that spawns an ad-hoc multi-agent team.

    Attributes:
        id: Unique node identifier
        name: Human-readable name
        goal: Overall goal for the team
        team_formation: How to organize the team (sequential, parallel, etc.)
        members: List of team member configurations
        timeout_seconds: Maximum execution time (None = no limit)
        max_iterations: Maximum team iterations (default: 50)
        total_tool_budget: Total tool calls budget (default: 100)
        merge_strategy: How to merge team state (default: "dict")
        merge_mode: Conflict resolution mode (default: "team_wins")
        output_key: Context key for result (default: "team_result")
        continue_on_error: Continue workflow on failure (default: true)
    """
```

#### TeamFormation

```python
class TeamFormation(str, Enum):
    """Team organization patterns."""
    SEQUENTIAL = "sequential"      # Chain execution with context passing
    PARALLEL = "parallel"          # Simultaneous execution with aggregation
    PIPELINE = "pipeline"          # Output handoff between stages
    HIERARCHICAL = "hierarchical"  # Manager-worker coordination
    CONSENSUS = "consensus"        # Agreement-based decision making
```

#### RecursionContext

```python
@dataclass
class RecursionContext:
    """Tracks recursion depth for nested execution.

    Thread-safe for concurrent access.

    Attributes:
        current_depth: Current nesting level (0 = top-level)
        max_depth: Maximum allowed nesting level (default: 3)
        execution_stack: Stack trace of execution entries

    Methods:
        enter(type, id): Enter a nested level (raises if max exceeded)
        exit(): Exit a nested level
        can_nest(levels): Check if nesting is possible
        get_depth_info(): Get current depth information
    """
```

### Implementation Files

- `victor/workflows/recursion.py` - Recursion tracking implementation
- `victor/workflows/team_node_runner.py` - Team node execution
- `victor/framework/workflows/nodes.py` - TeamNode class definition
- `victor/workflows/yaml_loader.py` - YAML parsing (lines 1050-1106)
- `victor/teams/` - Team coordination infrastructure

### See Also

- [Team Configuration System](../architecture/teams.md) - Team specification architecture
- [State Merging Guide](../architecture/state_merging.md) - State merge strategies
- [Error Handling Guide](../guides/RESILIENCE.md) - Error handling patterns

---

*Last Updated: 2026-01-20*
