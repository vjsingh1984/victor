# Hierarchical Planning Guide

## Overview

Hierarchical planning enables Victor AI to break down complex tasks into manageable, executable subtasks with automatic dependency tracking and dynamic re-planning. This guide explains how to use hierarchical planning effectively.

## Table of Contents

- [What is Hierarchical Planning?](#what-is-hierarchical-planning)
- [Key Concepts](#key-concepts)
- [Getting Started](#getting-started)
- [Advanced Usage](#advanced-usage)
- [Replanning Strategies](#replanning-strategies)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## What is Hierarchical Planning?

Hierarchical planning is a task decomposition approach that:

1. **Breaks down complex goals** into smaller, executable steps
2. **Tracks dependencies** between tasks
3. **Enables parallel execution** of independent tasks
4. **Supports dynamic re-planning** based on execution feedback
5. **Estimates complexity** for task prioritization

### When to Use Hierarchical Planning

**Ideal for:**
- Large feature implementation (e.g., "Implement user authentication")
- Complex refactoring projects (e.g., "Migrate from REST to GraphQL")
- Multi-phase projects (e.g., "Set up CI/CD pipeline")
- Tasks with clear dependencies (e.g., "Build microservice architecture")

**Not ideal for:**
- Simple, single-step tasks
- Quick fixes or tweaks
- Exploratory tasks without clear goals
- Tasks requiring real-time adaptation

## Key Concepts

### ExecutionPlan

An `ExecutionPlan` contains:
- **Goal**: High-level objective
- **Steps**: Ordered list of `PlanStep` objects
- **Dependencies**: Links between steps
- **Status**: Overall plan status

### PlanStep

A `PlanStep` represents a single task:
- **id**: Unique identifier
- **description**: What needs to be done
- **step_type**: Type of work (RESEARCH, PLANNING, IMPLEMENTATION, TESTING, REVIEW, DEPLOYMENT)
- **depends_on**: List of step IDs that must complete first
- **estimated_tool_calls**: Estimated complexity
- **requires_approval**: Whether user approval is needed
- **sub_agent_role**: Optional sub-agent delegation
- **status**: Current status (PENDING, IN_PROGRESS, COMPLETED, FAILED, SKIPPED, BLOCKED)

### StepType

Different types of work:
- **RESEARCH**: Information gathering (read-only)
- **PLANNING**: Sub-planning or task breakdown
- **IMPLEMENTATION**: Code changes
- **TESTING**: Running tests
- **REVIEW**: Code review or validation
- **DEPLOYMENT**: Deployment actions

## Getting Started

### 1. Enable Hierarchical Planning

```python
from victor.config.settings import Settings
from victor.agent import AgentOrchestrator

# Enable hierarchical planning
settings = Settings()
settings.enable_hierarchical_planning = True

# Create orchestrator
orchestrator = AgentOrchestrator(
    settings=settings,
    provider="anthropic",
    model="claude-sonnet-4-5",
)
```

### 2. Create a Plan

```python
# Generate a plan for a complex goal
goal = "Implement user authentication with JWT tokens"

plan = await orchestrator.planning.plan_for_goal(goal)

# Review the plan
print(plan.to_markdown())
```

Example output:

```markdown
# Execution Plan: Implement user authentication with JWT tokens

## Steps (8 total, estimated 85 tool calls)

### 1. Research existing authentication patterns
- **Type**: RESEARCH
- **Est. Tool Calls**: 10
- **Dependencies**: None
- **Status**: PENDING

### 2. Design authentication architecture
- **Type**: PLANNING
- **Est. Tool Calls**: 8
- **Dependencies**: 1
- **Status**: PENDING

### 3. Install required dependencies
- **Type**: IMPLEMENTATION
- **Est. Tool Calls**: 5
- **Dependencies**: 2
- **Status**: PENDING

### 4. Implement JWT middleware
- **Type**: IMPLEMENTATION
- **Est. Tool Calls**: 20
- **Dependencies**: 3
- **Status**: PENDING

### 5. Create authentication endpoints
- **Type**: IMPLEMENTATION
- **Est. Tool Calls**: 15
- **Dependencies**: 4
- **Status**: PENDING

### 6. Write unit tests
- **Type**: TESTING
- **Est. Tool Calls**: 12
- **Dependencies**: 5
- **Status**: PENDING

### 7. Integration testing
- **Type**: TESTING
- **Est. Tool Calls**: 10
- **Dependencies**: 6
- **Status**: PENDING

### 8. Code review and documentation
- **Type**: REVIEW
- **Est. Tool Calls**: 5
- **Dependencies**: 7
- **Status**: PENDING
```

### 3. Execute a Plan

```python
# Execute plan with auto-approval
result = await orchestrator.planning.execute_plan(
    plan,
    auto_approve=True
)

# Check result
print(f"Status: {result.status}")
print(f"Completed: {result.completed_steps}/{result.total_steps}")
print(f"Duration: {result.duration}s")

if result.failed:
    print(f"Failed steps: {[s.id for s in result.failed_steps]}")
```

### 4. Interactive Execution

```python
# Execute with user approval for each step
result = await orchestrator.planning.execute_plan(
    plan,
    auto_approve=False,  # Require approval for each step
    approval_callback=lambda step: input(f"Execute {step.description}? (y/n): ") == "y"
)
```

## Advanced Usage

### Custom Plan Creation

```python
from victor.agent.planning import ExecutionPlan, PlanStep, StepType

# Create a custom plan
plan = ExecutionPlan(
    goal="Optimize database queries",
    steps=[
        PlanStep(
            id="1",
            description="Analyze slow queries",
            step_type=StepType.RESEARCH,
            estimated_tool_calls=10
        ),
        PlanStep(
            id="2",
            description="Add database indexes",
            step_type=StepType.IMPLEMENTATION,
            depends_on=["1"],
            estimated_tool_calls=15
        ),
        PlanStep(
            id="3",
            description="Benchmark performance",
            step_type=StepType.TESTING,
            depends_on=["2"],
            estimated_tool_calls=8
        ),
        PlanStep(
            id="4",
            description="Document changes",
            step_type=StepType.REVIEW,
            depends_on=["3"],
            estimated_tool_calls=5
        ),
    ]
)
```

### Sub-Agent Delegation

```python
# Delegate steps to sub-agents
plan = ExecutionPlan(
    goal="Build microservice architecture",
    steps=[
        PlanStep(
            id="1",
            description="Research microservice patterns",
            step_type=StepType.RESEARCH,
            sub_agent_role="researcher"  # Delegate to researcher sub-agent
        ),
        PlanStep(
            id="2",
            description="Implement service layer",
            step_type=StepType.IMPLEMENTATION,
            depends_on=["1"],
            sub_agent_role="executor"  # Delegate to executor sub-agent
        ),
        PlanStep(
            id="3",
            description="Write integration tests",
            step_type=StepType.TESTING,
            depends_on=["2"],
            sub_agent_role="tester"  # Delegate to tester sub-agent
        ),
    ]
)
```

### Conditional Execution

```python
# Execute plan with condition checking
async def execute_with_conditions(plan):
    for step in plan.steps:
        if not step.is_ready(plan.completed_steps):
            continue

        # Check custom condition
        if step.step_type == StepType.DEPLOYMENT:
            # Require extra approval for deployment
            approval = input(f"Approve deployment? (yes/no): ")
            if approval.lower() != "yes":
                step.status = StepStatus.SKIPPED
                continue

        # Execute step
        result = await orchestrator.planning.execute_step(step)
        step.result = result
        step.status = StepStatus.COMPLETED if result.success else StepStatus.FAILED

        if not result.success:
            # Stop on failure
            break

    return plan
```

### Parallel Execution

```python
import asyncio

# Execute independent steps in parallel
async def execute_parallel(plan):
    # Group steps by readiness
    ready_steps = [s for s in plan.steps if s.is_ready(plan.completed_steps)]

    # Execute in parallel
    tasks = [orchestrator.planning.execute_step(s) for s in ready_steps]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Update step statuses
    for step, result in zip(ready_steps, results):
        if isinstance(result, Exception):
            step.status = StepStatus.FAILED
            step.result = StepResult(success=False, error=str(result))
        else:
            step.status = StepStatus.COMPLETED if result.success else StepStatus.FAILED
            step.result = result

    return plan
```

## Replanning Strategies

### Automatic Replanning on Failure

```python
async def execute_with_replanning(plan, max_retries=2):
    retry_count = 0

    for step in plan.steps:
        if not step.is_ready(plan.completed_steps):
            continue

        result = await orchestrator.planning.execute_step(step)

        if not result.success and retry_count < max_retries:
            print(f"Step {step.id} failed, replanning...")

            # Replan this step
            new_plan = await orchestrator.planning.plan_for_goal(
                f"Fix and complete: {step.description}"
            )

            # Replace failed step with new plan
            plan.steps.extend(new_plan.steps)
            retry_count += 1
            continue

        step.result = result
        step.status = StepStatus.COMPLETED if result.success else StepStatus.FAILED

    return plan
```

### Dynamic Plan Adjustment

```python
# Adjust plan based on intermediate results
async def execute_with_adjustment(plan):
    for step in plan.steps:
        if not step.is_ready(plan.completed_steps):
            continue

        result = await orchestrator.planning.execute_step(step)

        # Adjust subsequent steps based on result
        if result.success and result.metadata.get("complexity", "medium") == "high":
            # Add more testing steps for complex implementations
            additional_test = PlanStep(
                id=f"{step.id}_extra_test",
                description=f"Additional testing for {step.description}",
                step_type=StepType.TESTING,
                depends_on=[step.id],
                estimated_tool_calls=10
            )
            plan.steps.append(additional_test)

        step.result = result
        step.status = StepStatus.COMPLETED if result.success else StepStatus.FAILED

    return plan
```

### Human-in-the-Loop Replanning

```python
# Allow manual plan adjustment during execution
async def execute_with_manual_adjustment(plan):
    i = 0
    while i < len(plan.steps):
        step = plan.steps[i]

        if not step.is_ready(plan.completed_steps):
            i += 1
            continue

        # Show step and ask for confirmation
        print(f"\nNext step: {step.description}")
        choice = input("Execute (e), Modify (m), Skip (s), Abort (a)? ").lower()

        if choice == "e":
            result = await orchestrator.planning.execute_step(step)
            step.result = result
            step.status = StepStatus.COMPLETED if result.success else StepStatus.FAILED
            i += 1
        elif choice == "m":
            # Allow modification
            new_description = input(f"New description [{step.description}]: ") or step.description
            step.description = new_description
        elif choice == "s":
            step.status = StepStatus.SKIPPED
            i += 1
        elif choice == "a":
            print("Execution aborted")
            break

    return plan
```

## Best Practices

### 1. Start with Clear Goals

```python
# Good: Specific, actionable goal
goal = "Implement user authentication with JWT tokens and refresh token rotation"

# Bad: Vague, unclear goal
goal = "Fix auth"
```

### 2. Use Appropriate Step Types

```python
# Match step types to actual work
PlanStep(
    id="1",
    description="Review existing code",
    step_type=StepType.REVIEW,  # Correct: read-only review
)

PlanStep(
    id="2",
    description="Add new feature",
    step_type=StepType.IMPLEMENTATION,  # Correct: code changes
)

PlanStep(
    id="3",
    description="Run tests",
    step_type=StepType.TESTING,  # Correct: testing
)
```

### 3. Estimate Realistically

```python
# Consider complexity when estimating tool calls
PlanStep(
    id="1",
    description="Add simple logging",
    estimated_tool_calls=5  # Low complexity
)

PlanStep(
    id="2",
    description="Refactor database layer",
    estimated_tool_calls=30  # High complexity
)
```

### 4. Define Dependencies Correctly

```python
# Good: Clear dependency chain
steps = [
    PlanStep(id="1", description="Design API", depends_on=[]),
    PlanStep(id="2", description="Implement API", depends_on=["1"]),
    PlanStep(id="3", description="Test API", depends_on=["2"]),
]

# Bad: Circular dependency (will cause issues)
steps = [
    PlanStep(id="1", description="Task A", depends_on=["2"]),
    PlanStep(id="2", description="Task B", depends_on=["1"]),
]
```

### 5. Use Approval Gates for Critical Steps

```python
# Require approval for destructive operations
PlanStep(
    id="1",
    description="Delete production database",
    step_type=StepType.DEPLOYMENT,
    requires_approval=True  # Safety gate
)
```

### 6. Leverage Sub-Agent Delegation

```python
# Delegate specialized work to appropriate sub-agents
PlanStep(
    id="1",
    description="Research best practices",
    sub_agent_role="researcher"  # Research specialist
)

PlanStep(
    id="2",
    description="Implement feature",
    sub_agent_role="executor"  # Implementation specialist
)

PlanStep(
    id="3",
    description="Security review",
    sub_agent_role="security_auditor"  # Security specialist
)
```

### 7. Monitor Progress

```python
# Track plan execution
async def execute_with_monitoring(plan):
    start_time = time.time()

    for step in plan.steps:
        step_start = time.time()

        result = await orchestrator.planning.execute_step(step)
        step.result = result
        step.status = StepStatus.COMPLETED if result.success else StepStatus.FAILED

        step_duration = time.time() - step_start
        print(f"Step {step.id}: {step.duration:.2f}s - {result.success}")

    total_duration = time.time() - start_time
    print(f"Total: {total_duration:.2f}s")

    return plan
```

## Troubleshooting

### Plan Generation Fails

**Problem**: Planner fails to generate a valid plan.

**Solutions**:
1. **Simplify the goal**: Break into smaller, clearer objectives
2. **Check LLM provider**: Ensure provider supports tool calling
3. **Adjust decomposition settings**: Reduce max depth or subtasks
4. **Use custom plan**: Create plan manually

```python
# Adjust decomposition settings
settings.hierarchical_planning_max_depth = 3  # Reduce depth
settings.hierarchical_planning_max_subtasks = 7  # Reduce subtasks

# Try simpler goal
simple_goal = "Implement basic authentication"
plan = await orchestrator.planning.plan_for_goal(simple_goal)
```

### Plan Execution Stalls

**Problem**: Plan execution stops or hangs.

**Solutions**:
1. **Check dependencies**: Ensure no circular dependencies
2. **Verify step readiness**: Check `is_ready()` returns True
3. **Review step results**: Check for errors in step results
4. **Use timeout**: Add timeout to step execution

```python
# Add timeout to step execution
import asyncio

try:
    result = await asyncio.wait_for(
        orchestrator.planning.execute_step(step),
        timeout=300  # 5 minutes
    )
except asyncio.TimeoutError:
    print(f"Step {step.id} timed out")
    step.status = StepStatus.FAILED
```

### Steps Fail Repeatedly

**Problem**: Same step fails on retry.

**Solutions**:
1. **Replan the step**: Generate new approach
2. **Break into smaller steps**: Decompose further
3. **Adjust context**: Provide more context or guidance
4. **Use sub-agent**: Delegate to specialized sub-agent

```python
# Replan failed step
if step.status == StepStatus.FAILED:
    new_plan = await orchestrator.planning.plan_for_goal(
        f"Alternative approach for: {step.description}\n"
        f"Previous attempt failed: {step.result.error}"
    )
```

### Dependencies Not Respected

**Problem**: Steps execute in wrong order.

**Solutions**:
1. **Validate dependencies**: Check dependency graph
2. **Check execution logic**: Ensure `is_ready()` is called
3. **Review dependency list**: Verify correct step IDs

```python
# Validate dependencies
def validate_plan(plan):
    step_ids = {s.id for s in plan.steps}
    for step in plan.steps:
        for dep_id in step.depends_on:
            if dep_id not in step_ids:
                raise ValueError(f"Invalid dependency: {dep_id} not found")

validate_plan(plan)
```

## Examples

### Example 1: Feature Implementation

```python
from victor.agent.planning import AutonomousPlanner, StepType

async def implement_feature():
    # Create planner
    planner = AutonomousPlanner(orchestrator)

    # Generate plan
    plan = await planner.plan_for_goal(
        "Implement file upload functionality with S3 integration"
    )

    # Review and approve
    print(plan.to_markdown())
    approval = input("Approve plan? (y/n): ")

    if approval.lower() == "y":
        # Execute plan
        result = await planner.execute_plan(plan, auto_approve=True)

        if result.status == "completed":
            print("Feature implemented successfully!")
        else:
            print(f"Execution failed: {result.error}")
```

### Example 2: Refactoring Project

```python
async def refactor_codebase():
    planner = AutonomousPlanner(orchestrator)

    # Generate refactoring plan
    plan = await planner.plan_for_goal(
        "Refactor monolithic codebase into microservices architecture"
    )

    # Execute with human approval for each step
    result = await planner.execute_plan(
        plan,
        auto_approve=False,
        approval_callback=lambda step: input(
            f"Execute: {step.description}? (y/n): "
        ).lower() == "y"
    )

    return result
```

### Example 3: CI/CD Setup

```python
async def setup_cicd():
    planner = AutonomousPlanner(orchestrator)

    plan = await planner.plan_for_goal(
        "Set up complete CI/CD pipeline with testing, linting, and deployment"
    )

    # Execute with extra approval for deployment steps
    async def smart_approval(step):
        if step.step_type == StepType.DEPLOYMENT:
            return input(f"DEPLOYMENT: {step.description}. Approve? (yes/no): ").lower() == "yes"
        return True

    result = await planner.execute_plan(
        plan,
        auto_approve=False,
        approval_callback=smart_approval
    )

    return result
```

### Example 4: Multi-Phase Project

```python
async def multi_phase_project():
    planner = AutonomousPlanner(orchestrator)

    # Phase 1: Planning
    print("=== Phase 1: Planning ===")
    design_plan = await planner.plan_for_goal(
        "Design system architecture for real-time chat application"
    )
    await planner.execute_plan(design_plan, auto_approve=True)

    # Phase 2: Implementation
    print("\n=== Phase 2: Implementation ===")
    impl_plan = await planner.plan_for_goal(
        "Implement core chat functionality"
    )
    await planner.execute_plan(impl_plan, auto_approve=True)

    # Phase 3: Testing
    print("\n=== Phase 3: Testing ===")
    test_plan = await planner.plan_for_goal(
        "Comprehensive testing including load testing and security audit"
    )
    await planner.execute_plan(test_plan, auto_approve=True)

    # Phase 4: Deployment
    print("\n=== Phase 4: Deployment ===")
    deploy_plan = await planner.plan_for_goal(
        "Deploy to production with monitoring"
    )
    result = await planner.execute_plan(deploy_plan, auto_approve=True)

    return result
```

## Additional Resources

- [API Reference](../api/NEW_CAPABILITIES_API.md)
- [User Guide](../user-guide/index.md)
- [Enhanced Memory Guide](ENHANCED_MEMORY.md)
- [Dynamic Skills Guide](DYNAMIC_SKILLS.md)
- [Source Code](https://github.com/your-repo/tree/main/victor/agent/planning)

---

**Last Updated:** February 01, 2026
**Reading Time:** 3 minutes
