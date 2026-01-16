# Team Node Migration Guide

This guide explains how to migrate from manual team spawning to using team nodes in YAML workflows. Team nodes provide a declarative way to spawn multi-agent teams within workflow graphs.

## Overview

**Before (Manual Team Spawning):**
- Required Python code to spawn teams
- Manual coordination and state merging
- Complex error handling
- Difficult to maintain and version control

**After (Team Nodes):**
- Declarative YAML configuration
- Automatic team coordination
- Built-in state merging strategies
- Easy validation and testing
- Single source of truth

## Key Concepts

### Team Formations

Team nodes support 5 formation patterns:

| Formation | Description | Use Case |
|-----------|-------------|----------|
| `sequential` | Execute members one after another, context chaining | Multi-stage processing where each member builds on previous work |
| `parallel` | Execute all members simultaneously, independent work | Independent analysis tasks that can run concurrently |
| `hierarchical` | Manager delegates to workers, synthesizes results | Complex tasks requiring coordination and synthesis |
| `pipeline` | Output of one member feeds into the next | Assembly-line style processing |
| `consensus` | All members must agree (multiple rounds if needed) | Decision-making requiring agreement |

### State Merging Strategies

Team results are merged back into workflow graph state:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `dict` | Merge dictionaries (default) | Most team workflows |
| `list` | Append to list | Sequential execution |
| `custom` | Custom merge function | Complex merging logic |

### Merge Modes

How conflicts are resolved during merging:

| Mode | Description |
|------|-------------|
| `team_wins` | Team state overwrites graph state (default) |
| `graph_wins` | Graph state preserved |
| `merge` | Attempt to merge both |
| `error` | Raise error on conflict |

## Migration Examples

### Example 1: Simple Sequential Team

**Before (Python Code):**
```python
from victor.teams import create_coordinator, TeamConfig, TeamMember
from victor.agent.subagents import SubAgentRole

# Manually create team members
members = [
    TeamMember(
        id="researcher",
        role=SubAgentRole.RESEARCHER,
        name="Researcher",
        goal="Find authentication code",
    ),
    TeamMember(
        id="implementer",
        role=SubAgentRole.EXECUTOR,
        name="Implementer",
        goal="Implement the fix",
    ),
]

# Create team config
config = TeamConfig(
    name="Auth Fix Team",
    goal="Fix authentication vulnerability",
    members=members,
    formation=TeamFormation.SEQUENTIAL,
)

# Manually spawn and coordinate
coordinator = create_coordinator(config, orchestrator)
result = await coordinator.run()
```

**After (YAML Team Node):**
```yaml
workflows:
  auth_fix:
    description: "Fix authentication vulnerability using team"
    nodes:
      - id: fix_auth
        type: team
        name: "Auth Fix Team"
        goal: "Fix authentication vulnerability"
        team_formation: sequential
        timeout_seconds: 300
        total_tool_budget: 50
        output_key: fix_result
        members:
          - id: researcher
            role: researcher
            name: "Researcher"
            goal: "Find authentication code"
            tool_budget: 20
            tools: [read, grep, code_search]

          - id: implementer
            role: executor
            name: "Implementer"
            goal: "Implement the fix"
            tool_budget: 30
            tools: [read, write, edit]
        next: [verify]

      - id: verify
        type: agent
        role: tester
        goal: "Verify the fix: {{fix_result}}"
        tool_budget: 10
```

### Example 2: Parallel Review Team

**Before (Python Code):**
```python
# Manually create parallel team
members = [
    TeamMember(id="security", role=SubAgentRole.RESEARCHER, goal="Security review"),
    TeamMember(id="performance", role=SubAgentRole.RESEARCHER, goal="Performance review"),
    TeamMember(id="quality", role=SubAgentRole.RESEARCHER, goal="Code quality review"),
]

config = TeamConfig(
    name="Review Team",
    goal="Comprehensive code review",
    members=members,
    formation=TeamFormation.PARALLEL,
)

coordinator = create_coordinator(config, orchestrator)
result = await coordinator.run()

# Manually merge results
security_result = result.member_results["security"].output
performance_result = result.member_results["performance"].output
quality_result = result.member_results["quality"].output

# Continue with merged results...
```

**After (YAML Team Node):**
```yaml
workflows:
  parallel_review:
    description: "Parallel code review with team"
    nodes:
      - id: review_team
        type: team
        name: "Review Team"
        goal: "Comprehensive code review of changes: {{changes}}"
        team_formation: parallel
        timeout_seconds: 600
        total_tool_budget: 75
        output_key: review_results
        merge_strategy: dict
        merge_mode: team_wins
        members:
          - id: security_reviewer
            role: reviewer
            name: "Security Reviewer"
            goal: "Review for security vulnerabilities"
            tool_budget: 25
            tools: [read, grep]
            backstory: "Security expert with 10 years experience"
            expertise: ["security", "vulnerabilities", "owasp"]

          - id: performance_reviewer
            role: reviewer
            name: "Performance Reviewer"
            goal: "Review for performance issues"
            tool_budget: 25
            tools: [read, grep]
            backstory: "Performance optimization specialist"
            expertise: ["performance", "optimization", "profiling"]

          - id: quality_reviewer
            role: reviewer
            name: "Code Quality Reviewer"
            goal: "Review for code quality and maintainability"
            tool_budget: 25
            tools: [read, grep]
            backstory: "Senior developer focused on code quality"
            expertise: ["code-quality", "maintainability", "design-patterns"]
        next: [synthesize]

      - id: synthesize
        type: agent
        role: writer
        goal: |
          Synthesize review findings into comprehensive report:
          {{review_results}}

          Include:
          1. Summary of findings
          2. Critical issues
          3. Recommendations
          4. Priority ranking
        tool_budget: 15
```

### Example 3: Hierarchical Team with Delegation

**Before (Python Code):**
```python
# Complex hierarchical setup
manager = TeamMember(
    id="tech_lead",
    role=SubAgentRole.PLANNER,
    goal="Coordinate refactoring",
    is_manager=True,
    can_delegate=True,
    delegation_targets=["backend", "frontend", "tester"],
)

workers = [
    TeamMember(id="backend", role=SubAgentRole.EXECUTOR, goal="Backend refactoring"),
    TeamMember(id="frontend", role=SubAgentRole.EXECUTOR, goal="Frontend refactoring"),
    TeamMember(id="tester", role=SubAgentRole.TESTER, goal="Write tests"),
]

config = TeamConfig(
    name="Refactoring Team",
    goal="Refactor authentication system",
    members=[manager] + workers,
    formation=TeamFormation.HIERARCHICAL,
)

# Complex coordination logic...
```

**After (YAML Team Node):**
```yaml
workflows:
  hierarchical_refactor:
    description: "Hierarchical refactoring team"
    nodes:
      - id: refactor_team
        type: team
        name: "Refactoring Team"
        goal: "Refactor authentication system for better security"
        team_formation: hierarchical
        timeout_seconds: 900
        total_tool_budget: 150
        output_key: refactor_result
        members:
          - id: tech_lead
            role: planner
            name: "Technical Lead"
            goal: |
              Coordinate the refactoring effort:
              1. Plan the refactoring approach
              2. Delegate tasks to team members
              3. Review and integrate work
              4. Ensure quality and consistency
            tool_budget: 30
            tools: [read, grep, overview]
            is_manager: true
            can_delegate: true
            delegation_targets: [backend_dev, frontend_dev, qa_tester]
            backstory: "15 years experience leading software teams"
            expertise: ["architecture", "team-leadership", "security"]
            personality: "collaborative and strategic"

          - id: backend_dev
            role: executor
            name: "Backend Developer"
            goal: "Refactor backend authentication code"
            tool_budget: 40
            tools: [read, write, edit]
            reports_to: tech_lead
            backstory: "Senior backend developer specializing in security"
            expertise: ["backend", "security", "authentication"]

          - id: frontend_dev
            role: executor
            name: "Frontend Developer"
            goal: "Update frontend authentication flows"
            tool_budget: 40
            tools: [read, write, edit]
            reports_to: tech_lead
            backstory: "Frontend developer with React expertise"
            expertise: ["frontend", "react", "ui-ux"]

          - id: qa_tester
            role: tester
            name: "QA Engineer"
            goal: "Write and execute comprehensive tests"
            tool_budget: 40
            tools: [read, write, shell]
            reports_to: tech_lead
            backstory: "QA engineer with testing automation expertise"
            expertise: ["testing", "automation", "quality-assurance"]
        next: [verify]

      - id: verify
        type: compute
        name: "Run Test Suite"
        tools: [shell]
        inputs:
          command: pytest tests/test_auth.py -v
        output: test_results
```

## Common Migration Patterns

### Pattern 1: Add Team to Existing Workflow

**Existing workflow:**
```yaml
workflows:
  feature_development:
    nodes:
      - id: plan
        type: agent
        role: planner
        goal: "Plan the feature"
        next: [implement]

      - id: implement
        type: agent
        role: executor
        goal: "Implement feature"
        next: [test]
```

**Add team node:**
```yaml
workflows:
  feature_development:
    nodes:
      - id: plan
        type: agent
        role: planner
        goal: "Plan the feature"
        next: [team_review]

      # NEW: Team node for collaborative implementation
      - id: team_review
        type: team
        name: "Implementation Team"
        goal: "Implement feature: {{plan_result}}"
        team_formation: pipeline
        members:
          - id: architect
            role: planner
            name: "Architect"
            goal: "Design the implementation"
            tool_budget: 15

          - id: developer
            role: executor
            name: "Developer"
            goal: "Write the code"
            tool_budget: 30

          - id: reviewer
            role: reviewer
            name: "Reviewer"
            goal: "Review the implementation"
            tool_budget: 15
        next: [test]
```

### Pattern 2: Replace Multiple Agents with Team

**Before:**
```yaml
nodes:
  - id: research
    type: agent
    role: researcher
    goal: "Research the issue"
    next: [analyze]

  - id: analyze
    type: agent
    role: researcher
    goal: "Analyze findings: {{research_result}}"
    next: [propose]

  - id: propose
    type: agent
    role: planner
    goal: "Propose solution: {{analysis_result}}"
    next: [implement]
```

**After:**
```yaml
nodes:
  - id: investigation_team
    type: team
    name: "Investigation Team"
    goal: "Investigate and propose solution for: {{issue}}"
    team_formation: sequential
    members:
      - id: researcher
        role: researcher
        goal: "Research the issue"
        tool_budget: 20

      - id: analyst
        role: researcher
        goal: "Analyze research findings"
        tool_budget: 20

      - id: proposer
        role: planner
        goal: "Propose solution based on analysis"
        tool_budget: 15
    output_key: proposal
    next: [implement]
```

### Pattern 3: Conditional Team Selection

```yaml
nodes:
  - id: assess_complexity
    type: agent
    role: planner
    goal: "Assess task complexity (simple/complex)"
    tool_budget: 5
    output: complexity
    next: [decide_team]

  - id: decide_team
    type: condition
    condition: "complexity.level"
    branches:
      "simple": simple_executor
      "complex": complex_team

  - id: simple_executor
    type: agent
    role: executor
    goal: "Execute simple task: {{task}}"
    tool_budget: 20
    next: [complete]

  - id: complex_team
    type: team
    name: "Complex Task Team"
    goal: "Handle complex task: {{task}}"
    team_formation: hierarchical
    members:
      - id: manager
        role: planner
        goal: "Coordinate complex task execution"
        is_manager: true
        can_delegate: true

      - id: worker1
        role: executor
        goal: "Execute delegated tasks"
        reports_to: manager

      - id: worker2
        role: executor
        goal: "Execute delegated tasks"
        reports_to: manager
    next: [complete]
```

## Configuration Reference

### Team Node Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | string | Yes | - | Unique node identifier |
| `type` | string | Yes | - | Must be `"team"` |
| `name` | string | No | - | Human-readable name |
| `goal` | string | Yes | - | Overall team goal |
| `team_formation` | string | No | `"sequential"` | Formation pattern |
| `timeout_seconds` | integer | No | `600` | Maximum execution time |
| `total_tool_budget` | integer | No | `100` | Total tool calls across members |
| `max_iterations` | integer | No | `50` | Maximum total iterations |
| `output_key` | string | No | `"team_result"` | Key for result in graph state |
| `merge_strategy` | string | No | `"dict"` | How to merge results |
| `merge_mode` | string | No | `"team_wins"` | Conflict resolution |
| `continue_on_error` | boolean | No | `true` | Continue workflow on team failure |
| `members` | list | Yes | - | Team member definitions |
| `next` | list | No | `[]` | Next node IDs |

### Team Member Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | string | Yes | - | Unique member ID within team |
| `role` | string | Yes | - | SubAgentRole (researcher, planner, executor, etc.) |
| `name` | string | Yes | - | Display name |
| `goal` | string | Yes | - | Member's specific goal |
| `tool_budget` | integer | No | `15` | Max tool calls for this member |
| `tools` | list | No | - | Allowed tools (default: role-based) |
| `backstory` | string | No | `""` | Rich persona description |
| `expertise` | list | No | `[]` | Domain expertise areas |
| `personality` | string | No | `""` | Communication style |
| `can_delegate` | boolean | No | `false` | Can delegate to other members |
| `delegation_targets` | list | No | `[]` | Member IDs to delegate to |
| `reports_to` | string | No | - | Manager member ID (hierarchical) |
| `is_manager` | boolean | No | `false` | Is this the team manager |
| `priority` | integer | No | `0` | Execution priority (lower = earlier) |
| `memory` | boolean | No | `false` | Enable memory persistence |
| `cache` | boolean | No | `true` | Cache tool results |
| `verbose` | boolean | No | `false` | Show detailed logs |
| `max_iterations` | integer | No | - | Per-member iteration limit |

## Things to Watch Out For

### 1. Recursion Limits

Team nodes count toward the workflow's recursion depth. Default is 3 levels.

```yaml
# Workflow-level configuration
workflows:
  my_workflow:
    execution:
      max_recursion_depth: 5  # Increase for nested teams
    nodes:
      - id: outer_team
        type: team
        members:
          - id: inner_team_member
            role: planner
            # This member spawns another team
            # With max_recursion_depth=3, this is the limit
```

**Symptoms of hitting recursion limit:**
- Error: "Maximum recursion depth exceeded"
- Team stops executing midway

**Solutions:**
- Increase `max_recursion_depth` in workflow execution settings
- Flatten team structure (use parallel instead of nested sequential)
- Use larger team instead of nested teams

### 2. Timeout Configuration

Team timeouts apply to the entire team, not per member.

```yaml
# BAD: Each member gets 300 seconds
- id: team
  type: team
  timeout_seconds: 300
  members:
    - id: member1  # Gets ~100 seconds
    - id: member2  # Gets ~100 seconds
    - id: member3  # Gets ~100 seconds

# GOOD: Budget time appropriately
- id: team
  type: team
  timeout_seconds: 900  # 15 minutes total
  members:
    - id: member1
      goal: "Quick task"  # Expected to finish in 5 min
    - id: member2
      goal: "Long task"  # Expected to finish in 10 min
```

### 3. Tool Budget Allocation

`total_tool_budget` is shared across all members.

```yaml
# BAD: Members might exceed budget
- id: team
  type: team
  total_tool_budget: 30
  members:
    - id: member1
      tool_budget: 20  # Uses 20
    - id: member2
      tool_budget: 20  # Tries to use 20, total would be 40!

# GOOD: Budget matches total
- id: team
  type: team
  total_tool_budget: 40
  members:
    - id: member1
      tool_budget: 20
    - id: member2
      tool_budget: 20
```

### 4. State Merging Conflicts

Understand merge modes to avoid unexpected state overwrites.

```yaml
# Example: Graph state has {"status": "pending"}
# Team returns {"status": "completed", "result": "success"}

- id: team
  type: team
  merge_mode: team_wins  # Graph state becomes {"status": "completed", "result": "success"}
  # merge_mode: graph_wins  # Graph state becomes {"status": "pending", "result": "success"}
  # merge_mode: merge  # Attempts to merge both
  # merge_mode: error  # Raises error on key conflict
```

### 5. Formation-Specific Requirements

Some formations have specific member requirements:

**HIERARCHICAL:** Must have exactly one manager
```yaml
# BAD: No manager
- id: team
  type: team
  team_formation: hierarchical
  members:
    - id: worker1
      role: executor
    - id: worker2
      role: executor
  # Error: Hierarchical teams must have exactly one manager

# GOOD: Has manager
- id: team
  type: team
  team_formation: hierarchical
  members:
    - id: manager
      role: planner
      is_manager: true
      can_delegate: true
    - id: worker1
      role: executor
      reports_to: manager
```

**CONSENSUS:** All members must have can_delegate or no members should have it
```yaml
# CONSENSUS: All members can propose and discuss
- id: team
  type: team
  team_formation: consensus
  members:
    - id: member1
      role: planner
      can_delegate: true  # Can propose alternatives
    - id: member2
      role: planner
      can_delegate: true  # Can propose alternatives
```

### 6. Memory and Caching

Memory and caching have performance implications:

```yaml
# Enable memory for learning across tasks
- id: team
  type: team
  members:
    - id: researcher
      role: researcher
      memory: true  # Remembers discoveries
      memory_config:
        enabled: true
        persist_across_sessions: true
        relevance_threshold: 0.7

# Disable caching for real-time data
- id: team
  type: team
  members:
    - id: data_fetcher
      role: researcher
      cache: false  # Always fetch fresh data
```

## Validation and Testing

### Validate Workflow Syntax

```bash
# Validate YAML syntax
victor workflow validate path/to/workflow.yaml

# Check with schema
python -m pytest tests/integration/workflows/test_workflow_yaml_validation.py
```

### Test Team Execution

```python
# Test team node in isolation
from victor.workflows.yaml_loader import load_workflow_from_file
from victor.workflows.executor import WorkflowExecutor

workflow = load_workflow_from_file("my_workflow.yaml", workflow_name="my_team_workflow")
executor = WorkflowExecutor(orchestrator)

result = await executor.execute(workflow, initial_context={"task": "test task"})
assert result["team_result"]["success"] == True
```

### Common Validation Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Team node missing required 'id' field` | No `id` field | Add unique `id` |
| `Hierarchical teams must have exactly one manager` | No manager or multiple managers | Set `is_manager: true` on exactly one member |
| `Team member IDs must be unique` | Duplicate member IDs | Use unique `id` for each member |
| `Invalid team formation: 'invalid'` | Typo in formation | Use valid formation (sequential, parallel, etc.) |
| `Maximum recursion depth exceeded` | Nested teams too deep | Increase `max_recursion_depth` |

## Performance Considerations

### Parallel vs Sequential

```yaml
# Parallel: Faster for independent tasks
- id: parallel_team
  type: team
  team_formation: parallel
  members:
    - id: analyzer1
      role: researcher
      goal: "Analyze component A"
    - id: analyzer2
      role: researcher
      goal: "Analyze component B"  # Runs concurrently
  # Total time: ~max(member_time)

# Sequential: Better for dependent tasks
- id: sequential_team
  type: team
  team_formation: sequential
  members:
    - id: analyzer1
      role: researcher
      goal: "Analyze and pass to next"
    - id: analyzer2
      role: researcher
      goal: "Build on previous analysis"  # Waits for analyzer1
  # Total time: ~sum(member_times)
```

### Tool Budget Optimization

```yaml
# Allocate budget based on expected workload
- id: optimized_team
  type: team
  total_tool_budget: 100
  members:
    - id: quick_task
      role: executor
      tool_budget: 10  # Quick task
    - id: medium_task
      role: executor
      tool_budget: 30  # Medium task
    - id: heavy_task
      role: executor
      tool_budget: 60  # Heavy task
```

## Best Practices

1. **Use descriptive member names and goals** - Helps with debugging and logs
2. **Set appropriate timeouts** - Base on formation and member count
3. **Budget tool calls wisely** - Heavier tasks get more budget
4. **Test formations independently** - Verify team behavior before integrating
5. **Use personas for complex tasks** - `backstory`, `expertise`, `personality` improve results
6. **Enable memory for research tasks** - Persists discoveries across team execution
7. **Monitor recursion depth** - Plan team structure to avoid hitting limits
8. **Handle errors gracefully** - Use `continue_on_error` appropriately
9. **Document merge strategies** - Comment on expected state changes
10. **Validate workflows** - Use `victor workflow validate` before deployment

## Advanced Examples

See example workflows for production-ready team node usage:
- `victor/coding/workflows/examples/team_review.yaml` - Code review team
- `victor/research/workflows/examples/team_research.yaml` - Research team
- `victor/dataanalysis/workflows/examples/team_analysis.yaml` - Data analysis team

## Further Reading

- [YAML Workflow Reference](../api-reference/workflows.md)
- [Team Formation Types](../architecture/TEAMS.md)
- [Workflow Best Practices](../development/WORKFLOW_GUIDELINES.md)
- [State Merging Strategies](../architecture/STATE_MERGING.md)
