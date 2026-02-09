# New Capabilities API Reference - Part 1

**Part 1 of 2:** Hierarchical Planning, Memory, and Skill APIs

---

## Navigation

- **[Part 1: Planning, Memory, Skills](#)** (Current)
- [Part 2: Multimodal, Persona, Performance](part-2-multimodal-persona-performance.md)
- [**Complete Reference**](../NEW_CAPABILITIES_API.md)

---

> **Note**: This legacy API documentation is retained for reference. For current docs, see `docs/reference/api/`.

Complete API reference for Victor AI's new agentic capabilities.

## Table of Contents

- [Hierarchical Planning APIs](#hierarchical-planning-apis)
- [Memory APIs](#memory-apis)
- [Skill APIs](#skill-apis)
- [Multimodal APIs](#multimodal-apis) *(in Part 2)*
- [Persona APIs](#persona-apis) *(in Part 2)*
- [Performance APIs](#performance-apis) *(in Part 2)*
- [Configuration APIs](#configuration-apis) *(in Part 2)*

---

## Hierarchical Planning APIs

### AutonomousPlanner

```python
class AutonomousPlanner:
    """Autonomous planning for goal-oriented execution."""

    def __init__(self, orchestrator: AgentOrchestrator):
        """Initialize planner.

        Args:
            orchestrator: Agent orchestrator instance
        """
        ...

    async def plan_for_goal(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        max_steps: int = 10,
        max_iterations: int = 3
    ) -> ExecutionPlan:
        """Generate execution plan for goal.

        Args:
            goal: High-level goal to achieve
            context: Additional context for planning
            max_steps: Maximum number of steps in plan
            max_iterations: Maximum planning iterations

        Returns:
            ExecutionPlan: Generated execution plan

        Raises:
            PlanningError: If planning fails
        """
        ...

    async def execute_plan(
        self,
        plan: ExecutionPlan,
        ...
    ) -> ExecutionResult:
        """Execute generated plan.

        Args:
            plan: Execution plan to run
            ...

        Returns:
            ExecutionResult: Execution result with outcomes
        """
        ...
```

[Content continues through Memory and Skill APIs...]


**Reading Time:** 1 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


**Continue to [Part 2: Multimodal, Persona, Performance APIs](part-2-multimodal-persona-performance.md)**
