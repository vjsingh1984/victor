# Real-World Credit Assignment Examples

This document provides practical examples of using the credit assignment system in real Victor agent workflows.

## Example 1: Multi-Agent Code Review Workflow

```python
"""
Multi-agent code review workflow with automatic credit assignment.

Three agents collaborate to review a pull request:
- Security bot: Checks for vulnerabilities
- Performance bot: Analyzes performance impact
- Style bot: Enforces code style standards

Credit is assigned fairly using Shapley values.
"""

from victor.framework import Agent, StateGraph
from victor.teams import UnifiedTeamCoordinator, TeamFormation
from victor.teams.mixins.credit_assignment import (
    CreditAssignmentMixin,
    extract_reward_from_member_result,
)
from victor.framework.rl import CreditMethodology


class CreditAwareReviewCoordinator(CreditAssignmentMixin, UnifiedTeamCoordinator):
    """Team coordinator with automatic credit tracking."""
    pass


async def review_pull_request(pr_number: int, diff: str):
    """Review a PR with multiple agents and track credit."""

    # Create orchestrator for each agent
    security_orchestrator = Agent.from_provider("anthropic", "claude-3-5-sonnet-20241022")
    performance_orchestrator = Agent.from_provider("anthropic", "claude-3-5-sonnet-20241022")
    style_orchestrator = Agent.from_provider("anthropic", "claude-3-haiku-20240307")

    # Create credit-aware coordinator
    coordinator = CreditAwareReviewCoordinator(None)
    coordinator.enable_credit_tracking(methodology=CreditMethodology.SHAPLEY)

    # Add team members
    coordinator.add_member("security", security_orchestrator)
    coordinator.add_member("performance", performance_orchestrator)
    coordinator.add_member("style", style_orchestrator)

    # Set formation (parallel review)
    coordinator.set_formation(TeamFormation.PARALLEL)

    # Execute review
    context = {
        "pr_number": pr_number,
        "diff": diff,
        "task": "Review PR for security, performance, and style",
    }

    result = await coordinator.execute_task("review", context)

    # Get credit attribution
    attribution = coordinator.get_team_credit_attribution()

    print(f"Review completed: {attribution['success']}")
    print(f"Total quality score: {attribution['total_reward']:.2f}")
    print("\nAgent contributions:")

    for agent, credits in attribution['member_attribution'].items():
        total = sum(credits.values())
        print(f"  {agent}: {total:+.2f}")

        # Show top contributors
        top_contributors = sorted(
            credits.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        for contributor, amount in top_contributors:
            if contributor != agent:
                print(f"    ← {contributor}: {amount:+.2f}")

    # Export detailed report
    coordinator.export_credit_report(
        output_path=f"pr_{pr_number}_credit_report.html",
        format="html"
    )

    return result
```

## Example 2: StateGraph Workflow with Credit Tracking

```python
"""
Feature development workflow with credit assignment.

Tracks which stages (analysis, implementation, testing, review)
contribute most to the success of feature development.
"""

from victor.framework.graph import StateGraph, END
from victor.framework.rl import create_credit_aware_workflow


def create_feature_workflow():
    """Create a credit-aware feature development workflow."""

    # Define state type
    from typing import TypedDict

    class FeatureState(TypedDict):
        task: str
        status: str
        analysis: str
        implementation: str
        tests: str
        review_comments: str
        reward: float
        agent_id: str

    # Create workflow stages
    graph = StateGraph(FeatureState)

    def analyze(state: FeatureState) -> FeatureState:
        # Analyze requirements
        state["status"] = "analyzing"
        state["analysis"] = "Requirements analyzed"
        state["agent_id"] = "analyst"
        state["reward"] = 0.3  # Moderate reward for analysis
        return state

    def implement(state: FeatureState) -> FeatureState:
        # Implement feature
        state["status"] = "implementing"
        state["implementation"] = "Code written"
        state["agent_id"] = "developer"
        state["reward"] = 0.5  # High reward for implementation
        return state

    def test(state: FeatureState) -> FeatureState:
        # Run tests
        state["status"] = "testing"
        state["tests"] = "Tests passed"
        state["agent_id"] = "tester"
        state["reward"] = 0.2  # Lower reward for testing
        return state

    def review(state: FeatureState) -> FeatureState:
        # Code review
        state["status"] = "reviewing"
        state["review_comments"] = "LGTM!"
        state["agent_id"] = "reviewer"
        state["reward"] = 0.1  # Small reward for review
        return state

    def should_iterate(state: FeatureState) -> str:
        """Decide whether to iterate or finish."""
        if state["reward"] < 0.5:
            return "iterate"  # Needs more work
        return "finish"

    # Build graph
    graph.add_node("analyze", analyze)
    graph.add_node("implement", implement)
    graph.add_node("test", test)
    graph.add_node("review", review)
    graph.add_node("iterate", implement)  # Re-run implementation

    graph.add_edge("analyze", "implement")
    graph.add_edge("implement", "test")
    graph.add_edge("test", "review")

    # Conditional edge: iterate or finish
    graph.add_conditional_edge(
        "review",
        should_iterate,
        {"iterate": "iterate", "finish": "__end__"}
    )

    graph.set_entry_point("analyze")

    # Make credit-aware
    credit_graph = create_credit_aware_workflow(
        graph,
        reward_key="reward",
        agent_key="agent_id",
    )

    return credit_graph


async def develop_feature(task_description: str):
    """Execute feature development with credit tracking."""

    credit_graph = create_feature_workflow()
    app = credit_graph.compile(
        enable_credit=True,
        credit_methodology="gae",  # Generalized Advantage Estimation
    )

    # Initial state
    initial_state: FeatureState = {
        "task": task_description,
        "status": "pending",
        "analysis": "",
        "implementation": "",
        "tests": "",
        "review_comments": "",
        "reward": 0.0,
        "agent_id": "",
    }

    # Execute workflow
    result = await app.invoke(initial_state)

    # Get credit attribution
    attribution = app.get_credit_attribution()

    print(f"\nFeature: {task_description}")
    print(f"Status: {result['status']}")
    print(f"Total reward: {attribution['total_reward']:.2f}")
    print(f"Transitions: {attribution['transition_count']}")
    print(f"Duration: {attribution['duration']:.2f}s")

    # Show agent contributions
    print("\nAgent contributions:")
    for agent, credit in attribution['agent_attribution'].items():
        total = sum(credit.values())
        print(f"  {agent}: {total:+.3f}")

    return result


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        result = await develop_feature("Add user authentication")
        print(f"Final state: {result}")

    asyncio.run(main())
```

## Example 3: Failed Task Analysis with Hindsight

```python
"""
Analyze failed task execution using hindsight credit assignment.

Useful for understanding what went wrong and learning from failures.
"""

from victor.framework.rl import (
    CreditAssignmentIntegration,
    CreditMethodology,
    ActionMetadata,
    compute_credit_metrics,
)


def analyze_failed_task(attempts: list[dict]):
    """Analyze a failed task using hindsight credit.

    Args:
        attempts: List of execution attempts with rewards
    """

    # Build trajectory from attempts
    trajectory = []
    rewards = []

    for i, attempt in enumerate(attempts):
        metadata = ActionMetadata(
            agent_id=attempt.get("agent", "agent_1"),
            action_id=f"attempt_{i}",
            turn_index=i,
            step_index=attempt.get("step", 0),
            tool_name=attempt.get("tool"),
            method_name=attempt.get("method"),
        )
        trajectory.append(metadata)
        rewards.append(attempt.get("reward", -0.1))  # Negative = failure

    # Use hindsight to reframe failure
    integration = CreditAssignmentIntegration()
    signals = integration.assign_credit(
        trajectory,
        rewards,
        methodology=CreditMethodology.HINDSIGHT,
    )

    # Compute metrics
    metrics = compute_credit_metrics(signals)

    print("=== Failed Task Analysis (Hindsight) ===")
    print(f"\nAttempts: {len(attempts)}")
    print(f"Total reward: {sum(rewards):.2f}")
    print(f"Hindsight credit: {metrics['total_credit']:.2f}")
    print(f"Positive ratio: {metrics['positive_ratio']:.1%}")

    # Show which attempts were most valuable
    print("\nMost valuable attempts:")
    for signal in sorted(signals, key=lambda s: s.credit, reverse=True)[:5]:
        print(f"  {signal.action_id}: {signal.credit:+.3f}")

    # Show critical actions (bifurcation points)
    critical = integration.identify_critical_actions(trajectory, rewards)
    if critical:
        print(f"\nCritical actions (bifurcation points): {critical}")

    return signals


# Example: Analyzing failed code generation
if __name__ == "__main__":
    failed_attempts = [
        {"agent": "coder", "tool": "generate", "reward": -0.2, "step": 0},
        {"agent": "coder", "tool": "generate", "reward": -0.3, "step": 1},
        {"agent": "reviewer", "tool": "validate", "reward": -0.5, "step": 2},
        {"agent": "coder", "tool": "fix", "reward": -0.1, "step": 3},
    ]

    signals = analyze_failed_task(failed_attempts)

    print("\nLearning opportunities:")
    for signal in signals:
        if signal.credit > 0:
            print(f"  {signal.action_id} → positive despite failure")
```

## Example 4: Comparing Credit Methodologies

```python
"""
Compare different credit assignment methodologies on the same workflow.

Helps choose the best methodology for your use case.
"""

from victor.framework.rl import (
    CreditAssignmentIntegration,
    CreditMethodology,
    ActionMetadata,
)


def compare_methodologies(execution_log: list[dict]):
    """Compare credit assignment methodologies.

    Args:
        execution_log: List of (agent, action, reward) tuples
    """

    # Build trajectory
    trajectory = []
    rewards = []

    for i, entry in enumerate(execution_log):
        metadata = ActionMetadata(
            agent_id=entry["agent"],
            action_id=f"{entry['agent']}_{i}",
            turn_index=i // 3,
            step_index=i,
        )
        trajectory.append(metadata)
        rewards.append(entry["reward"])

    # Compare methodologies
    methodologies = [
        CreditMethodology.GAE,
        CreditMethodology.SHAPLEY,
        CreditMethodology.MONTE_CARLO,
        CreditMethodology.N_STEP_RETURNS,
    ]

    results = {}

    for method in methodologies:
        integration = CreditAssignmentIntegration()
        signals = integration.assign_credit(trajectory, rewards, methodology=method)

        # Get attribution per agent
        agents = set(s.metadata.agent_id for s in signals if s.metadata)
        agent_credits = {}
        for agent in agents:
            agent_signals = [s for s in signals if s.metadata and s.metadata.agent_id == agent]
            agent_credits[agent] = sum(s.credit for s in agent_signals)

        results[method.value] = {
            "total_credit": sum(s.credit for s in signals),
            "agent_credits": agent_credits,
            "signal_count": len(signals),
        }

    # Display comparison
    print("=== Credit Methodology Comparison ===\n")

    for method_name, data in results.items():
        print(f"{method_name.upper()}:")
        print(f"  Total credit: {data['total_credit']:.3f}")
        print(f"  Agent attribution:")
        for agent, credit in data['agent_credits'].items():
            print(f"    {agent}: {credit:+.3f}")
        print()

    # Recommend best methodology
    print("Recommendation:")
    if len(agents) > 2:
        print("  → Use SHAPLEY for fair multi-agent attribution")
    elif len(trajectory) > 50:
        print("  → Use GAE for long-horizon workflows")
    else:
        print("  → Use Monte Carlo for simple workflows")


# Example usage
if __name__ == "__main__":
    log = [
        {"agent": "planner", "reward": 0.3},
        {"agent": "coder", "reward": 0.5},
        {"agent": "tester", "reward": 0.2},
        {"agent": "reviewer", "reward": 0.1},
        {"agent": "planner", "reward": -0.2},
        {"agent": "coder", "reward": 0.6},
    ]

    compare_methodologies(log)
```

## Example 5: Real-Time Credit Tracking During Execution

```python
"""
Track credit in real-time as agents execute.

Shows how to monitor credit accumulation during workflow execution.
"""

import asyncio
from victor.framework.rl import CreditTracer, CreditMethodology


async def execute_with_realtime_credit_tracking(agent, task: str):
    """Execute task with real-time credit tracking."""

    # Create tracer
    tracer = CreditTracer()

    # Start trace
    tracer.start_trace({"task": task, "agent": agent.agent_id})

    print(f"=== Executing: {task} ===\n")

    # Simulate agent execution steps
    steps = [
        ("analyze", 0.3, 0.5),
        ("implement", 0.5, 1.2),
        ("test", 0.2, 0.8),
        ("deploy", 0.1, 0.4),
    ]

    for i, (action, reward, duration) in enumerate(steps):
        print(f"[{i+1}] {action}...")

        # Execute action
        await asyncio.sleep(duration)  # Simulate work

        # Record transition
        tracer.record_transition(
            from_node=f"step_{i}" if i > 0 else "start",
            to_node=f"step_{i+1}",
            state_before={"step": i},
            state_after={"step": i+1, "action": action},
            node_output={"reward": reward, "action": action},
        )

        # Get intermediate credit
        trace = tracer.get_active_trace()
        if trace:
            print(f"    Running total: {trace.total_reward:.3f}")

    # End trace
    tracer.end_trace({"step": len(steps), "status": "done"}, success=True)

    # Assign credit
    trace = tracer.get_trace_history()[-1]
    signals = tracer.assign_credit_to_trace(
        trace,
        methodology=CreditMethodology.GAE
    )

    print(f"\n=== Results ===")
    print(f"Total reward: {trace.total_reward:.3f}")
    print(f"Duration: {trace.duration:.2f}s")

    # Show credit per action
    for signal in signals:
        print(f"  {signal.metadata.method_name if signal.metadata else signal.action_id}: "
              f"{signal.credit:+.3f}")


# Example usage
if __name__ == "__main__":
    from victor.framework import Agent

    agent = Agent.from_default()
    asyncio.run(execute_with_realtime_credit_tracking(
        agent,
        "Implement user login feature"
    ))
```

## Key Takeaways

1. **Use SHAPLEY for multi-agent teams**: Ensures fair attribution across collaborators
2. **Use GAE for general workflows**: Best balance of bias and variance
3. **Use HINDSIGHT for failures**: Extracts learning from unsuccessful executions
4. **Track credit in StateGraph**: Automatic attribution for complex workflows
5. **Compare methodologies**: Choose the best one for your use case

These examples demonstrate the flexibility and power of the credit assignment system in real-world scenarios!
