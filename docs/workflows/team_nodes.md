# Team Nodes in YAML Workflows

> Execute multi-agent teams within workflow graphs using YAML configuration

## Table of Contents

- [Overview](#overview)
- [When to Use Team Nodes](#when-to-use-team-nodes)
- [YAML Syntax and Configuration](#yaml-syntax-and-configuration)
- [Team Formation Types](#team-formation-types)
- [Recursion Depth Tracking](#recursion-depth-tracking)
- [Configuration Examples](#configuration-examples)
- [Best Practices](#best-practices)
- [Error Handling](#error-handling)
- [Complete Examples](#complete-examples)

## Overview

Team nodes enable **hybrid orchestration** by spawning ad-hoc multi-agent teams within workflow graphs. This combines the best of both worlds:

- **Workflow Graph**: Orchestrate complex, multi-stage processes
- **Multi-Agent Teams**: Collaborative problem solving with specialized agents

**Key Features**:

- **5 Team Formations**: Sequential, Parallel, Pipeline, Hierarchical, Consensus
- **Recursion Control**: Unified depth tracking prevents infinite nesting
- **Flexible Configuration**: YAML-first with Python escape hatches
- **Rich Metadata**: Team member personas, expertise, tool budgets
- **Error Resilience**: Continue-on-error and timeout handling

**Architecture**:

```
Workflow Graph (YAML)
    └── Team Node
        ├── Member 1 (Researcher)
        ├── Member 2 (Executor)
        └── Member 3 (Reviewer)
        └── UnifiedTeamCoordinator
        └── RecursionContext (depth tracking)
```

## When to Use Team Nodes

Use team nodes when a task requires **multiple perspectives** or **specialized expertise**:

### Ideal Use Cases

- **Code Review**: Security + Quality + Performance reviewers in parallel
- **Feature Implementation**: Researcher → Architect → Developer → Tester pipeline
- **Complex Debugging**: Parallel investigation with synthesis
- **Documentation**: Researcher → Writer → Reviewer pipeline
- **Data Analysis**: Parallel analysis with aggregation

### When NOT to Use

- **Simple Tasks**: Single agent is sufficient
- **Linear Processes**: Regular agent nodes are simpler
- **Low Latency**: Teams add coordination overhead

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
| `id` | string | Unique node identifier |
| `type` | string | Must be `"team"` |
| `goal` | string | Overall team objective |
| `members` | list | List of member configurations |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `"Team: {id}"` | Human-readable name |
| `team_formation` | string | `"sequential"` | Team organization pattern |
| `timeout_seconds` | number | `None` | Maximum execution time |
| `max_iterations` | number | `50` | Maximum team iterations |
| `total_tool_budget` | number | `100` | Total tool calls budget |
| `output_key` | string | `"team_result"` | Context key for result |
| `continue_on_error` | boolean | `true` | Continue workflow on team failure |
| `merge_strategy` | string | `"dict"` | State merge strategy |
| `merge_mode` | string | `"team_wins"` | Conflict resolution mode |
| `next` | list | `[]` | Next node IDs |

### Member Configuration

Each member supports these fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique member identifier |
| `role` | string | No | Role: `researcher`, `executor`, `planner`, `reviewer`, `assistant` |
| `goal` | string | Yes | Member's specific objective |
| `tool_budget` | number | No (default: 25) | Tool call budget |
| `tools` | list | No | Allowed tool names |
| `backstory` | string | No | Member's background |
| `expertise` | list | No | List of expertise areas |
| `personality` | string | No | Communication style |

## Team Formation Types

Victor supports **5 team formation patterns** for different collaboration models.

### 1. Sequential

Members execute **one after another**, with context chaining.

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

**Use when**: Tasks have clear stages that depend on previous results.

**Flow**: Researcher → [context] → Implementer

### 2. Parallel

All members work **simultaneously** on the same task.

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

**Flow**: All members execute → [aggregate results]

### 3. Pipeline

Output of each member **feeds into the next** with handoff messages.

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

**Use when**: Processing pipeline with clear stages and handoff.

**Flow**: Researcher → [handoff] → Writer → [handoff] → Reviewer

### 4. Hierarchical

A **manager** delegates to workers, then synthesizes results.

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
        # Manager implicitly identified by role=planner in hierarchical
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

**Flow**: Manager plans → [delegate] → Workers execute → [results] → Manager synthesizes

### 5. Consensus

All members must **agree**, requiring multiple rounds if needed.

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

## Recursion Depth Tracking

Team nodes participate in **unified recursion tracking** to prevent infinite nesting. All nested execution types count toward the same limit:

- Workflow invoking workflow
- Workflow spawning team
- Team spawning team
- Team spawning workflow

### Default Limits

- **Default max depth**: 3 levels
- **Tracking scope**: Per workflow execution

### Configuration

#### YAML Metadata

```yaml
workflows:
  my_workflow:
    metadata:
      max_recursion_depth: 5

    nodes:
      - id: outer_team
        type: team
        goal: "Team that spawns nested workflows"
        members:
          - id: member
            role: executor
            goal: "Execute"
```

#### Execution Settings

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

### Recursion Depth Example

```yaml
# Level 0: Main workflow
workflows:
  main:
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

### Error Example

```python
RecursionDepthError: Maximum recursion depth (3) exceeded.
Attempting to enter team:nested_review_team

Execution stack:
  workflow:main → team:review_team → workflow:sub_analysis → team:nested_review_team
```

## Configuration Examples

### Example 1: Simple Team Node

```yaml
workflows:
  simple_review:
    description: "Simple code review with 2 reviewers"

    nodes:
      - id: review_team
        type: team
        name: "Code Review Team"
        goal: "Review the pull request"
        team_formation: parallel
        timeout_seconds: 300
        total_tool_budget: 50
        output_key: review_result
        members:
          - id: reviewer_1
            role: reviewer
            goal: "Check for bugs and issues"
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

### Example 2: Team with Custom Formation

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
            personality: "strategic and thorough"

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
            backstory: |
              Expert code researcher with deep knowledge
              of software architecture.
            expertise: [code-analysis, design-patterns]
            personality: "methodical and systematic"

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
            backstory: |
              Full-stack developer focused on quality
              and maintainability.
            expertise: [implementation, testing, debugging]
            personality: "quality-focused and collaborative"

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
            backstory: |
              QA specialist with attention to detail
              and passion for quality.
            expertise: [testing, validation, bug-tracking]
            personality: "thorough and detail-oriented"
        next: [final_report]
```

### Example 3: Nested Teams (Within Recursion Limit)

```yaml
workflows:
  complex_analysis:
    description: "Complex analysis with nested teams"

    metadata:
      max_recursion_depth: 4  # Allow one level of nesting

    nodes:
      # Level 1: Outer team
      - id: analysis_team
        type: team
        name: "Analysis Team"
        goal: "Perform comprehensive analysis"
        team_formation: hierarchical
        total_tool_budget: 200
        members:
          - id: coordinator
            role: planner
            goal: "Coordinate analysis across domains"
            tool_budget: 40

          - id: security_analyst
            role: researcher
            goal: "Security analysis"
            tool_budget: 40

          - id: performance_analyst
            role: researcher
            goal: "Performance analysis"
            tool_budget: 40

          - id: quality_analyst
            role: researcher
            goal: "Quality analysis"
            tool_budget: 40
        next: [synthesize]

      - id: synthesize
        type: agent
        role: writer
        goal: "Synthesize team findings"
        tool_budget: 20
```

### Example 4: Runtime Recursion Depth Override

```python
from victor.workflows.yaml_loader import load_workflow_from_file
from victor.workflows.executor import WorkflowExecutor

# Load workflow
workflow = load_workflow_from_file("my_workflow.yaml")

# Override recursion depth at runtime
executor = WorkflowExecutor(
    orchestrator=orchestrator,
    tool_registry=tool_registry,
)

# Execute with custom recursion limit
context = {
    "max_recursion_depth": 5,  # Override YAML default
    "user_task": "Analyze system"
}

result = await executor.execute(workflow, initial_context=context)
```

## Best Practices

### 1. Choose the Right Formation

| Task Type | Recommended Formation |
|-----------|----------------------|
| Dependent stages | **Sequential** or **Pipeline** |
| Independent perspectives | **Parallel** |
| Planning + delegation | **Hierarchical** |
| Agreement required | **Consensus** |
| Unknown/Exploratory | **Sequential** (simplest) |

### 2. Set Realistic Budgets

```yaml
# Good: Specific budgets per member
members:
  - id: researcher
    tool_budget: 15  # Limited research
  - id: implementer
    tool_budget: 60  # Heavy implementation

# Avoid: Equal budgets when work is uneven
members:
  - id: researcher
    tool_budget: 50  # Too much for research
  - id: implementer
    tool_budget: 50  # Not enough for implementation
```

### 3. Define Clear Goals

```yaml
# Good: Specific, actionable goal
goal: |
  Implement user authentication:
  1. Login endpoint
  2. Session management
  3. Password hashing
  4. Unit tests

# Avoid: Vague goal
goal: "Do authentication stuff"
```

### 4. Use Expertise Fields

```yaml
members:
  - id: security_reviewer
    role: reviewer
    goal: "Security review"
    expertise: [security, authentication, authorization]
    # Expertise helps in team formation and routing
```

### 5. Handle Timeouts Gracefully

```yaml
nodes:
  - id: team_with_timeout
    type: team
    timeout_seconds: 300  # 5 minutes
    continue_on_error: true  # Don't fail workflow
    # If team times out, workflow continues
```

### 6. Monitor Recursion Depth

```yaml
workflows:
  main_workflow:
    metadata:
      max_recursion_depth: 3  # Be conservative

    nodes:
      - id: team_node
        type: team
        # Team members can spawn workflows, but depth is limited
```

### 7. Leverage Output Keys

```yaml
nodes:
  - id: analysis_team
    type: team
    output_key: analysis_results  # Specific key

  - id: report_generator
    type: agent
    goal: "Generate report from: {{analysis_results}}"
    # Clear data flow
```

## Error Handling

### Common Errors

#### 1. Recursion Depth Exceeded

```
RecursionDepthError: Maximum recursion depth (3) exceeded.
Execution stack: workflow:main → team:A → workflow:B → team:C
```

**Solution**: Increase `max_recursion_depth` or redesign workflow to reduce nesting.

#### 2. Team Execution Failed

```
YAMLWorkflowError: Team execution failed: All members failed to complete
```

**Solutions**:
- Check `continue_on_error: true` to continue workflow
- Increase `timeout_seconds`
- Verify member goals are achievable
- Check tool budgets

#### 3. Member Not Found

```
ValueError: Unknown role 'invalid_role'
```

**Solution**: Use valid roles: `researcher`, `executor`, `planner`, `reviewer`, `assistant`

#### 4. Timeout

```
TimeoutError: Team execution exceeded 300 seconds
```

**Solutions**:
- Increase `timeout_seconds`
- Reduce scope of goals
- Reduce team size
- Optimize tool usage

### Error Handling Patterns

#### Continue on Error

```yaml
nodes:
  - id: risky_team
    type: team
    goal: "High-risk analysis"
    continue_on_error: true  # Workflow continues even if team fails
    next: [fallback_handler]

  - id: fallback_handler
    type: agent
    goal: "Handle team failure gracefully"
    # Executes even if team fails
```

#### Conditional Retry

```yaml
nodes:
  - id: attempt_1
    type: team
    goal: "First attempt"
    output_key: result_1
    next: [check_success]

  - id: check_success
    type: condition
    condition: "success_flag"
    branches:
      "true": complete
      "false": attempt_2

  - id: attempt_2
    type: team
    goal: "Retry with different approach"
    # Different team configuration
```

## Complete Examples

### Example: Code Review Workflow

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
        tools: [lint, type_check, security_scan]
        constraints: [llm]  # No LLM, just tools
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
        team_formation: parallel
        timeout_seconds: 300
        total_tool_budget: 75
        output_key: review_feedback
        members:
          - id: security_reviewer
            role: reviewer
            goal: "Check for security vulnerabilities"
            tool_budget: 25
            tools: [read, grep]
            expertise: [security, authentication, authorization]

          - id: quality_reviewer
            role: reviewer
            goal: "Check code quality and maintainability"
            tool_budget: 25
            tools: [read, grep]
            expertise: [code-quality, design-patterns]

          - id: logic_reviewer
            role: reviewer
            goal: "Check logic and correctness"
            tool_budget: 25
            tools: [read, grep]
            expertise: [logic, algorithms]
        next: [decide_changes]

      # Stage 3: Decide if changes needed
      - id: decide_changes
        type: condition
        condition: "needs_changes"
        branches:
          "true": implementation_team
          "false": final_approval

      # Stage 4a: Implementation team (if changes needed)
      - id: implementation_team
        type: team
        name: "Implementation Team"
        goal: |
          Implement requested changes:
          {{review_feedback}}
        team_formation: pipeline
        timeout_seconds: 600
        total_tool_budget: 125
        output_key: changes_result
        members:
          - id: planner
            role: planner
            goal: "Plan the implementation approach"
            tool_budget: 25
            tools: [read, grep, overview]

          - id: developer
            role: executor
            goal: "Implement the changes"
            tool_budget: 75
            tools: [read, write, grep]

          - id: tester
            role: reviewer
            goal: "Test the changes"
            tool_budget: 25
            tools: [read, grep]
        next: [final_review]

      # Stage 4b: Final approval (if no changes)
      - id: final_approval
        type: agent
        role: planner
        goal: "Approve the changes"
        tool_budget: 10
        next: [complete]

      # Stage 5: Final review after changes
      - id: final_review
        type: team
        name: "Final Review Team"
        goal: |
          Final review of implemented changes:
          {{changes_result}}
        team_formation: consensus
        timeout_seconds: 300
        total_tool_budget: 75
        output_key: final_result
        members:
          - id: reviewer_1
            role: reviewer
            goal: "Verify changes address feedback"
            tool_budget: 25
            tools: [read, grep]

          - id: reviewer_2
            role: reviewer
            goal: "Check for regressions"
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
        transform: "status = approved"
        next: []

    error_handling:
      retry_policy:
        max_retries: 2
        backoff: exponential
      on_failure:
        - log_error
        - notify_team
```

### Example: Research and Implementation

```yaml
workflows:
  research_and_implement:
    description: "Research new technology and implement POC"

    metadata:
      version: "1.0.0"

    nodes:
      # Phase 1: Research team
      - id: research_phase
        type: team
        name: "Research Team"
        goal: |
          Research {{technology}}:
          1. Core concepts and architecture
          2. Best practices
          3. Integration options
          4. Potential pitfalls
        team_formation: parallel
        timeout_seconds: 900  # 15 minutes
        total_tool_budget: 150
        output_key: research_findings
        members:
          - id: concept_researcher
            role: researcher
            goal: "Research core concepts and architecture"
            tool_budget: 50
            tools: [web_search, read]
            backstory: "Technology researcher with 10 years experience"
            expertise: [research, architecture-analysis]

          - id: best_practices_researcher
            role: researcher
            goal: "Find best practices and patterns"
            tool_budget: 50
            tools: [web_search, read]
            expertise: [best-practices, patterns]

          - id: integration_researcher
            role: researcher
            goal: "Research integration approaches"
            tool_budget: 50
            tools: [web_search, read]
            expertise: [integration, system-design]
        next: [synthesis]

      # Phase 2: Synthesize research
      - id: synthesis
        type: agent
        role: planner
        goal: |
          Synthesize research findings:
          {{research_findings}}

          Create a comprehensive guide covering:
          1. Architecture overview
          2. Implementation recommendations
          3. Code examples
          4. Integration plan
        tool_budget: 30
        tools: [write]
        output: guide
        next: [implementation_team]

      # Phase 3: Implementation team
      - id: implementation_team
        type: team
        name: "Implementation Team"
        goal: |
          Implement POC based on research guide:
          {{guide}}
        team_formation: sequential
        timeout_seconds: 1200  # 20 minutes
        total_tool_budget: 200
        output_key: implementation_result
        members:
          - id: architect
            role: planner
            goal: |
              Design POC architecture:
              1. Component structure
              2. Data flow
              3. Integration points
            tool_budget: 40
            tools: [write]
            backstory: "Software architect specializing in POCs"
            expertise: [architecture, design]

          - id: developer
            role: executor
            goal: |
              Implement core functionality:
              1. Set up project structure
              2. Implement key components
              3. Add error handling
            tool_budget: 120
            tools: [read, write, grep]
            expertise: [implementation, debugging]

          - id: documenter
            role: writer
            goal: |
              Create documentation:
              1. API documentation
              2. Usage examples
              3. Setup instructions
            tool_budget: 40
            tools: [read, write]
            expertise: [documentation, technical-writing]
        next: [testing_team]

      # Phase 4: Testing team
      - id: testing_team
        type: team
        name: "Testing Team"
        goal: |
          Test POC implementation:
          {{implementation_result}}
        team_formation: parallel
        timeout_seconds: 600
        total_tool_budget: 100
        output_key: test_results
        members:
          - id: functional_tester
            role: reviewer
            goal: "Test functional requirements"
            tool_budget: 50
            tools: [read, grep]

          - id: integration_tester
            role: reviewer
            goal: "Test integration points"
            tool_budget: 50
            tools: [read, grep]
        next: [finalize]

      # Phase 5: Finalize
      - id: finalize
        type: agent
        role: writer
        goal: |
          Prepare final report:
          Research: {{research_findings}}
          Implementation: {{implementation_result}}
          Testing: {{test_results}}

          Create:
          1. Executive summary
          2. Technical deep-dive
          3. Recommendations
        tool_budget: 20
        tools: [write]
        next: []
```

## Additional Resources

- [Multi-Agent Teams Guide](../guides/MULTI_AGENT_TEAMS.md) - Team coordination patterns
- [Workflow DSL Guide](../guides/workflow-development/dsl.md) - Python workflow API
- [Workflow Examples](../guides/workflow-development/examples.md) - More workflow examples
- [Team Configuration System](../architecture/teams.md) - Team specification architecture

## API Reference

### TeamNodeWorkflow

```python
@dataclass
class TeamNodeWorkflow(WorkflowNode):
    """Node that spawns an ad-hoc multi-agent team.

    Attributes:
        goal: Overall goal for the team
        team_formation: How to organize the team
        members: List of team member configurations
        timeout_seconds: Maximum execution time
        max_iterations: Maximum team iterations
        total_tool_budget: Total tool calls budget
        output_key: Context key for result
        continue_on_error: Continue workflow on failure
    """
```

### TeamFormation

```python
class TeamFormation(str, Enum):
    SEQUENTIAL = "sequential"      # Chain execution
    PARALLEL = "parallel"          # Simultaneous execution
    PIPELINE = "pipeline"          # Output handoff
    HIERARCHICAL = "hierarchical"  # Manager-worker
    CONSENSUS = "consensus"        # Agreement-based
```

### RecursionContext

```python
@dataclass
class RecursionContext:
    """Tracks recursion depth for nested execution.

    Attributes:
        current_depth: Current nesting level
        max_depth: Maximum allowed depth
        execution_stack: Stack trace for debugging
    """
```

**See Also**:
- `victor/workflows/recursion.py` - Recursion tracking implementation
- `victor/workflows/team_node_runner.py` - Team node execution
- `victor/workflows/yaml_loader.py` - YAML parsing (lines 1050-1101)
- `victor/teams/` - Team coordination infrastructure
