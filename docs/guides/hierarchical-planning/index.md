# Hierarchical Planning Guide

Guide to hierarchical planning for breaking down complex tasks into manageable subtasks.

---

## Quick Summary

Hierarchical planning enables Victor AI to break down complex tasks into manageable, executable subtasks with:
- **Automatic dependency tracking** between tasks
- **Parallel execution** of independent tasks
- **Dynamic re-planning** based on execution feedback
- **Complexity estimation** for task prioritization

**Ideal for:**
- Large feature implementation
- Complex refactoring projects
- Multi-phase projects
- Tasks with clear dependencies

---

## Guide Parts

### [Part 1: Concepts & Usage](part-1-concepts-usage-replanning.md)
- What is Hierarchical Planning?
- Key Concepts (ExecutionPlan, PlanStep, Dependencies)
- Getting Started
- Advanced Usage
- Replanning Strategies

### [Part 2: Best Practices, Troubleshooting, Examples](part-2-best-practices-troubleshooting-examples.md)
- Best Practices
- Troubleshooting
- Examples (Feature Implementation, Refactoring, CI/CD)
- Additional Resources

---

## Quick Start

```python
from victor.agent.planning import AutonomousPlanner

async def implement_feature():
    """Implement user authentication feature."""
    planner = AutonomousPlanner(orchestrator)

    # Create plan
    plan = await planner.plan_for_goal(
        goal="Implement user authentication with JWT tokens",
        max_steps=10
    )

    # Execute plan
    result = await planner.execute_plan(plan)

    if result.success:
        print("Feature implemented successfully!")
```

---

## Related Documentation

- [Planning API Reference](../../reference/internals/planning.md)
- [Workflow Examples](../../examples/workflows/)
- [Autonomous Planner Guide](../AUTONOMOUS_PLANNER.md)

---

**Last Updated:** February 01, 2026
**Reading Time:** 12 min (all parts)
