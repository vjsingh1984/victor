# Team Nodes Guide - Part 2

**Part 2 of 4:** Recursion Depth Tracking and Configuration Examples

---

## Navigation

- [Part 1: Overview & Formation](part-1-overview-formation.md)
- **[Part 2: Recursion & Configuration](#)** (Current)
- [Part 3: Best Practices & Error Handling](part-3-best-practices-errors.md)
- [Part 4: Complete Examples](part-4-complete-examples.md)
- [**Complete Guide**](../team_nodes.md)

---

## Recursion Depth Tracking

Team nodes participate in **unified recursion tracking** to prevent infinite nesting. All nested execution types count
  toward the same limit:
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

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 5 min
**Last Updated:** February 08, 2026**
