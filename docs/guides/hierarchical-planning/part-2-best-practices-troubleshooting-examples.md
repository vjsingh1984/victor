# Hierarchical Planning Guide - Part 2

**Part 2 of 2:** Best Practices, Troubleshooting, Examples, and Additional Resources

---

## Navigation

- [Part 1: Concepts & Usage](part-1-concepts-usage-replanning.md)
- **[Part 2: Best Practices, Troubleshooting, Examples](#)** (Current)
- [**Complete Guide](../HIERARCHICAL_PLANNING.md)**

---

## Best Practices

### 1. Define Clear Goals

Start with well-defined objectives:

```python
# Good: Clear, specific goal
goal = "Implement JWT authentication for REST API"

# Avoid: Vague goal
goal = "Fix authentication"
```

### 2. Estimate Complexity

Provide complexity estimates for prioritization:

```python
step1 = PlanStep(
    name="setup_auth",
    complexity=0.3,  # Low complexity
    estimated_duration="5min"
)

step2 = PlanStep(
    name="implement_jwt",
    complexity=0.7,  # High complexity
    estimated_duration="30min"
)
```

### 3. Handle Dependencies

Declare step dependencies explicitly:

```python
step1 = PlanStep(name="setup_database", ...)
step2 = PlanStep(name="create_users_table", ...)
step3 = PlanStep(name="implement_auth", ...)

# step2 depends on step1
step2.add_dependency(step1)

# step3 depends on step2
step3.add_dependency(step2)
```

### 4. Monitor Execution

Track plan execution progress:

```python
async def execute_plan_with_monitoring(plan):
    """Execute plan with progress tracking."""
    for step in plan.steps:
        logger.info(f"Executing: {step.name}")

        result = await step.execute()

        if result.success:
            logger.info(f"Completed: {step.name}")
        else:
            logger.error(f"Failed: {step.name}")
            # Trigger re-planning
            return await replan_around_failure(plan, step)
```

---

## Troubleshooting

### Plan Generation Fails

**Problem**: Unable to generate execution plan.

**Solutions**:
1. **Simplify goal**: Break down into smaller objectives
2. **Provide context**: Give more information about the task
3. **Adjust parameters**: Modify complexity estimates

```python
# Simplify goal
simple_goal = "Add JWT authentication to existing endpoint"

# Provide context
context = {
    "current_setup": "Uses basic API key auth",
    "requirements": ["JWT tokens", "Refresh tokens", "Revocation"]
}
```

### Execution Deadlock

**Problem**: Steps waiting for each other indefinitely.

**Solutions**:
1. **Review dependencies**: Check for circular dependencies
2. **Reorder steps**: Change step execution order
3. **Break cycles**: Identify and resolve dependency cycles

```python
# Detect circular dependencies
if plan.has_circular_dependencies():
    # Break cycle
    plan.remove_dependency(step1, step2)
    plan.add_dependency(step1, step3)
```

### Replanning Loop

**Problem**: Continuous re-planning without progress.

**Solutions**:
1. **Increase failure threshold**: Allow more failures before re-planning
2. **Adjust strategy**: Use different replanning approach
3. **Manual intervention**: Request user guidance

```python
# Increase threshold
plan = await planner.create_plan(
    goal=goal,
    max_replans=5,  # Allow more replans
    failure_tolerance=2  # More failures before replan
)
```

---

## Examples

### Example 1: Feature Implementation

```python
from victor.agent.planning import AutonomousPlanner

async def implement_feature():
    """Implement user authentication feature."""
    planner = AutonomousPlanner(orchestrator)

    # Create plan
    plan = await planner.plan_for_goal(
        goal="Implement user authentication with JWT tokens",
        context={
            "framework": "FastAPI",
            "database": "PostgreSQL",
            "requirements": ["JWT", "Refresh tokens", "Revocation"]
        }
    )

    # Execute plan
    result = await planner.execute_plan(plan)

    if result.success:
        print("Feature implemented successfully!")
    else:
        print(f"Implementation failed: {result.error}")
```

### Example 2: Refactoring Project

```python
async def refactor_codebase():
    """Refactor from REST to GraphQL."""
    planner = AutonomousPlanner(orchestrator)

    plan = await planner.plan_for_goal(
        goal="Migrate user management API from REST to GraphQL",
        max_steps=15,
        context={
            "current_architecture": "REST",
            "target_architecture": "GraphQL",
            "endpoints": ["users", "auth", "permissions"]
        }
    )

    result = await planner.execute_plan(plan)
    return result
```

### Example 3: CI/CD Setup

```python
async def setup_cicd():
    """Set up CI/CD pipeline."""
    planner = AutonomousPlanner(orchestrator)

    plan = await planner.plan_for_goal(
        goal="Set up GitHub Actions workflow for testing and deployment",
        context={
            "platform": "GitHub",
            "requirements": ["Unit tests", "Integration tests", "Docker build"]
        }
    )

    result = await planner.execute_plan(plan)
    return result
```

---

## Additional Resources

- [Planning API Reference](../../reference/internals/planning.md)
- [Workflow Examples](../../examples/workflows/)
- [Autonomous Planner Guide](../AUTONOMOUS_PLANNER.md)

---

## See Also

- [Documentation Home](../../README.md)


**Reading Time:** 2 min
**Last Updated:** February 01, 2026
