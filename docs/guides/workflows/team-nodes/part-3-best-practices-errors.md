# Team Nodes Guide - Part 3

**Part 3 of 4:** Best Practices and Error Handling

---

## Navigation

- [Part 1: Overview & Formation](part-1-overview-formation.md)
- [Part 2: Recursion & Configuration](part-2-recursion-configuration.md)
- **[Part 3: Best Practices & Errors](#)** (Current)
- [Part 4: Complete Examples](part-4-complete-examples.md)
- [**Complete Guide**](../team_nodes.md)

---

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

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 5 min
**Last Updated:** February 08, 2026**
